"""Validate stored forecast identities, schemas, and independently reloadable scores."""

import numpy as np
import pandas as pd

from tsfm_lab.contracts import EvaluationSpec, fixed_interval

PREDICTION_IDENTITIES = ("request_id", "series_id", "adapter_id", "point_statistic")
_COLUMNS = (
    "request_id",
    "series_id",
    "adapter_id",
    "origin",
    "target_timestamp",
    "lead",
    "actual",
    "prediction",
    "point_statistic",
)
_COUNTS = {
    "attempted_windows",
    "successful_windows",
    "failed_windows",
    "forecast_cases",
    "unique_target_timestamps",
}


class StoreError(ValueError):
    """A run is unavailable, malformed, or inconsistent with its artifacts."""


def _object(value, name):
    if not isinstance(value, dict):
        raise StoreError(f"{name} must be an object")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise StoreError(f"{name} must be a nonempty string")


def _number(value, name):
    if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
        raise StoreError(f"{name} must be a finite nonnegative number")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise StoreError(f"{name} must be an integer >= {minimum}")


def _metrics(metrics, spec):
    metrics = _object(metrics, "metrics")
    scale, reason, count = (
        metrics["scale"],
        metrics["scale_reason"],
        metrics["scale_count"],
    )
    _integer(count, "scale_count")
    if "mase" not in spec.metrics:
        if scale is not None or reason != "not_requested" or count != 0:
            raise StoreError(
                "Unrequested MASE requires absent scale and not_requested status"
            )
    elif reason is None:
        _number(scale, "scale")
        if scale == 0 or count == 0:
            raise StoreError("Defined MASE needs positive scale and scale_count")
    elif reason == "zero_scale":
        if type(scale) not in (int, float) or scale != 0 or count == 0:
            raise StoreError("zero_scale needs zero scale and positive scale_count")
    elif reason != "insufficient_scale_history" or scale is not None or count != 0:
        raise StoreError("Invalid MASE scaling status")
    groups = [_object(metrics["overall"], "overall metrics")]
    by_lead = _object(metrics["by_lead"], "by_lead metrics")
    if len(by_lead) != spec.horizon or any(
        not isinstance(lead, str)
        or not lead.isdecimal()
        or str(int(lead)) != lead
        or not 1 <= int(lead) <= spec.horizon
        for lead in by_lead
    ):
        raise StoreError("Per-lead metrics must cover the configured horizon")
    groups.extend(_object(group, "lead metrics") for group in by_lead.values())
    for group in groups:
        if set(group) != set(spec.metrics):
            raise StoreError("Metric names must equal the configured selection")
        for name, metric in group.items():
            metric = _object(metric, "metric value")
            if set(metric) != {"value", "reason"}:
                raise StoreError("Metric values require value and reason")
            if name == "mase" and reason is not None:
                if metric["value"] is not None or metric["reason"] != reason:
                    raise StoreError("Undefined MASE must match its scale reason")
            else:
                _number(metric["value"], name)
                if metric["reason"] is not None:
                    raise StoreError("Defined metrics cannot have an undefined reason")


def _checkpoint_identity(model, runtime, spec, environment):
    adapter = model["adapter_id"]
    from tsfm_lab.models.checkpoint_profiles import CHECKPOINT_REPOSITORIES
    from tsfm_lab.models.registry import adapter_profile

    if adapter not in CHECKPOINT_REPOSITORIES:
        if model["source"] in {"huggingface", "local"}:
            raise StoreError("Checkpoint sources require a supported model adapter")
        return
    from tsfm_lab.models.checkpoint_profiles import checkpoint_adapter
    from tsfm_lab.models.checkpoints import validate_checkpoint_record

    record = validate_checkpoint_record(model["options"]["checkpoint"])
    profile = adapter_profile(adapter)
    if (
        model["source"] != record["source"]
        or model["family"] != profile.family
        or checkpoint_adapter(record["architecture"]) != adapter
    ):
        raise StoreError("Checkpoint source/family disagrees with the adapter")
    caps = model["capabilities"]
    max_context = record["native_context_length"] - profile.context_reserved
    if (
        caps["max_context"] != max_context
        or caps["max_horizon"] != record["native_prediction_length"]
        or spec.context_length > caps["max_context"]
        or spec.horizon > caps["max_horizon"]
        or runtime["device"] != "cpu"
        or runtime["dtype"] != "float32"
        or runtime["batch_size"] != 1
    ):
        raise StoreError("Checkpoint capabilities or runtime disagree with metadata")
    distributions = tuple(name for name, _ in profile.runtime_requirements)
    if adapter == "chronos_2":
        from tsfm_lab.models.chronos2 import FORECAST_OPTIONS

        options = runtime.get("forecast_options")
        if (
            not isinstance(options, dict)
            or set(options) != set(FORECAST_OPTIONS)
            or any(options[key] is not value for key, value in FORECAST_OPTIONS.items())
        ):
            raise StoreError("Chronos-2 runtime requires its recorded forecast options")
    if adapter == "timesfm_2p5":
        from tsfm_lab.models.timesfm import FORECAST_OPTIONS

        options = runtime.get("forecast_options")
        if (
            runtime.get("torch_compile") is not False
            or not isinstance(options, dict)
            or set(options) != set(FORECAST_OPTIONS)
            or any(options[key] is not value for key, value in FORECAST_OPTIONS.items())
        ):
            raise StoreError("TimesFM runtime requires its recorded forecast options")
    for key in distributions:
        _text(environment[key], f"environment {key}")


def _schema(result):
    result = _object(result, "result")
    if (
        type(result["schema_version"]) is not int
        or result["schema_version"] != 1
        or result["status"] != "completed"
    ):
        raise StoreError("Invalid completed result schema or status")
    config = _object(result["configuration"], "configuration")
    dataset = _object(config["dataset"], "dataset configuration")
    model = _object(config["model"], "model configuration")
    runtime = _object(config["runtime"], "runtime configuration")
    for key in ("series_id", "target_name", "units", "frequency"):
        _text(dataset[key], f"dataset {key}")
    interval = fixed_interval(dataset["frequency"])
    for key in ("adapter_id", "adapter_version", "family", "source"):
        _text(model[key], f"model {key}")
    _object(model["options"], "model options")
    _object(model["capabilities"], "model capabilities")
    for key in ("device", "dtype"):
        _text(runtime[key], f"runtime {key}")
    _integer(runtime["batch_size"], "batch_size", 1)
    if "seed" in runtime:
        _integer(runtime["seed"], "seed")
    evaluation = _object(config["evaluation"], "evaluation configuration")
    if set(evaluation) != set(EvaluationSpec.__dataclass_fields__):
        raise StoreError("Resolved evaluation requires every declared field")
    spec = EvaluationSpec(**evaluation)
    provenance = _object(result["provenance"], "provenance")
    source = _object(provenance["source"], "source provenance")
    _text(source["package_version"], "package version")
    if source["git_revision"] is not None:
        _text(source["git_revision"], "git revision")
    if source["git_dirty"] is not None and type(source["git_dirty"]) is not bool:
        raise StoreError("git_dirty must be a boolean or null")
    environment = _object(provenance["environment"], "environment provenance")
    for key in ("python", "numpy", "pandas", "os", "machine"):
        _text(environment[key], f"environment {key}")
    _checkpoint_identity(model, runtime, spec, environment)
    if "project_verification" in provenance:
        from tsfm_lab.models.verification import validate_verification_record

        validate_verification_record(
            provenance["project_verification"],
            model["options"]["checkpoint"],
            model["adapter_id"],
        )
    fingerprint = _object(provenance["dataset"], "dataset provenance")[
        "canonical_sha256"
    ]
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(c not in "0123456789abcdef" for c in fingerprint)
    ):
        raise StoreError("Invalid canonical dataset fingerprint")
    _text(provenance["metric_version"], "metric version")
    history = _object(provenance["scale_history"], "scale history")
    _integer(history["observations"], "scale history observations")
    _integer(history["seasonal_period"], "scale history seasonal period", 1)
    for key in ("start", "end_exclusive"):
        timestamp = pd.Timestamp(history[key])
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise StoreError("Scale history timestamps must be timezone-aware")
    if (
        history["seasonal_period"] != spec.seasonal_period
        or pd.Timestamp(history["end_exclusive"]) != spec.evaluation_start
        or history["observations"] < spec.context_length
        or spec.evaluation_start.value - pd.Timestamp(history["start"]).value
        != history["observations"] * interval.value
    ):
        raise StoreError("Scale history must agree with evaluation metadata")
    counts = _object(result["counts"], "counts")
    if set(counts) != _COUNTS:
        raise StoreError("Invalid prediction count fields")
    for key, value in counts.items():
        _integer(value, key)
    timing = _object(result["timing_seconds"], "timing_seconds")
    if not {"load", "inference", "evaluation"} <= set(timing):
        raise StoreError("Stage timing fields are missing")
    for key, value in timing.items():
        _number(value, f"{key} timing")
    _metrics(result["metrics"], spec)
    expected_scale_count = (
        max(history["observations"] - spec.seasonal_period, 0)
        if "mase" in spec.metrics
        else 0
    )
    if result["metrics"]["scale_count"] != expected_scale_count:
        raise StoreError("Scale count disagrees with recorded history metadata")
    if "name" in result:
        _text(result["name"], "experiment name")
    return config, spec, interval


def _frame(frame):
    if not isinstance(frame, pd.DataFrame) or set(frame.columns) != set(_COLUMNS):
        raise StoreError("Predictions must have the exact numerical-core columns")
    if len(frame.columns) != len(_COLUMNS) or frame.empty:
        raise StoreError("Predictions must be nonempty with unique columns")
    frame = frame.loc[:, _COLUMNS].copy(deep=True)
    for column in PREDICTION_IDENTITIES:
        if (
            not frame[column]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise StoreError(f"Prediction {column} must contain nonempty strings")
    for column in ("origin", "target_timestamp"):
        timestamps = [pd.Timestamp(value) for value in frame[column]]
        if any(pd.isna(value) or value.tzinfo is None for value in timestamps):
            raise StoreError("Prediction timestamps must be timezone-aware")
        frame[column] = pd.to_datetime(timestamps, utc=True)
    for column in ("actual", "prediction", "lead"):
        values = pd.to_numeric(frame[column], errors="raise")
        if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise StoreError(f"Prediction {column} must contain finite numbers")
        frame[column] = values
    leads = frame.lead
    if (leads <= 0).any() or (leads % 1 != 0).any() or (leads >= 2**63).any():
        raise StoreError("Prediction leads must be positive int64 integers")
    frame["lead"] = leads.astype("int64")
    frame[["actual", "prediction"]] = frame[["actual", "prediction"]].astype("float64")
    if not np.isfinite(frame[["actual", "prediction"]]).all().all():
        raise StoreError("Prediction floats exceed float64 limits")
    return frame


def _grid(frame, result, config, spec, interval):
    if set(frame.series_id) != {config["dataset"]["series_id"]} or set(
        frame.adapter_id
    ) != {config["model"]["adapter_id"]}:
        raise StoreError(
            "Prediction series or adapter identity disagrees with metadata"
        )
    if not set(frame.point_statistic) <= {"deterministic", spec.point_statistic}:
        raise StoreError("Prediction statistic disagrees with the configured task")
    if frame.duplicated(["series_id", "origin", "target_timestamp", "lead"]).any():
        raise StoreError("Predictions contain duplicate forecast keys")
    duration = spec.evaluation_end.value - spec.evaluation_start.value
    if duration % interval.value or duration // interval.value < spec.horizon:
        raise StoreError("Evaluation interval does not define complete grid windows")
    window_count = 1 + (duration // interval.value - spec.horizon) // spec.stride
    windows = frame.groupby("origin", sort=False)
    if windows.ngroups != window_count or frame.request_id.nunique() != window_count:
        raise StoreError("Predictions must cover every configured rolling origin")
    for origin, window in windows:
        offset = origin.value + interval.value - spec.evaluation_start.value
        if (
            offset < 0
            or offset % interval.value
            or offset // interval.value % spec.stride
            or offset // interval.value + spec.horizon > duration // interval.value
        ):
            raise StoreError("Prediction origin falls outside the configured grid")
        if (
            window.request_id.nunique() != 1
            or len(window) != spec.horizon
            or sorted(window.lead) != list(range(1, spec.horizon + 1))
        ):
            raise StoreError(
                "Each prediction window needs one request and complete leads"
            )
        for row in window.itertuples():
            if (
                row.target_timestamp.value - origin.value
                != int(row.lead) * interval.value
            ):
                raise StoreError("Prediction targets disagree with lead and frequency")
    expected = {
        "attempted_windows": window_count,
        "successful_windows": window_count,
        "failed_windows": 0,
        "forecast_cases": len(frame),
        "unique_target_timestamps": frame.target_timestamp.nunique(),
    }
    if result["counts"] != expected:
        raise StoreError("Prediction counts do not match result metadata")


def _scores(frame, result, spec):
    from tsfm_lab.evaluation.metrics import score_with_scale

    ordered = frame.sort_values(["origin", "lead"])
    saved = result["metrics"]
    scores = score_with_scale(
        ordered.actual.to_numpy().reshape(-1, spec.horizon),
        ordered.prediction.to_numpy().reshape(-1, spec.horizon),
        scale=saved["scale"],
        scale_reason=saved["scale_reason"],
        scale_count=saved["scale_count"],
        metrics=spec.metrics,
    )
    groups = [(saved["overall"], scores.overall)]
    groups.extend(
        (saved["by_lead"][str(lead)], group) for lead, group in scores.by_lead.items()
    )
    for stored, calculated in groups:
        for name, metric in calculated.items():
            value = stored[name]["value"]
            if stored[name]["reason"] != metric.reason or (
                (value is None) != (metric.value is None)
            ):
                raise StoreError("Stored metric status disagrees with predictions")
            if value is not None and not np.isclose(
                value, metric.value, rtol=1e-12, atol=0
            ):
                raise StoreError("Stored metric value disagrees with predictions")


def validate_predictions(frame: pd.DataFrame, result: dict) -> pd.DataFrame:
    """Validate complete metadata, forecast grids, and scores without source access."""
    try:
        config, spec, interval = _schema(result)
        frame = _frame(frame)
        _grid(frame, result, config, spec, interval)
        _scores(frame, result, spec)
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError) as exc:
        raise StoreError(f"Invalid prediction artifacts: {exc}") from exc
    return frame
