"""Strict future metadata and cross-file validation without source data or weights."""

import copy
import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd

from tsfm_lab.artifact_validation import StoreError, _checkpoint_identity
from tsfm_lab.contracts import (
    Dataset,
    ForecastRequest,
    ModelCapabilities,
    fixed_interval,
    utc_timestamp,
)
from tsfm_lab.data import canonical_fingerprint
from tsfm_lab.point_tasks import output_statistics

PREDICTION_COLUMNS = (
    "request_id",
    "series_id",
    "adapter_id",
    "origin",
    "target_timestamp",
    "lead",
    "prediction",
    "point_statistic",
)
OBSERVATION_COLUMNS = ("series_id", "timestamp", "observed_value")
UNAVAILABLE = {
    "schema_version": 1,
    "mode": "future_forecast",
    "status": "unavailable",
    "reason": "future_actuals_absent",
    "metrics": None,
}
_HEX = re.compile(r"[0-9a-f]{64}")


def object_fields(value, fields, label, *, optional=()):
    if type(value) is not dict or not set(fields) <= set(value) <= set(fields) | set(
        optional
    ):
        raise StoreError(f"Invalid {label} fields")
    return value


def text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise StoreError(f"{label} must be nonempty text")
    return value


def integer(value, label, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise StoreError(f"{label} must be an integer >= {minimum}")
    return value


def identity(value, label):
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise StoreError(f"Invalid {label}")
    return value


def timestamp(value, label):
    if not isinstance(value, (str, pd.Timestamp)):
        raise StoreError(f"{label} must be an aware timestamp")
    try:
        return utc_timestamp(value, label).as_unit("ns")
    except (ValueError, TypeError, OverflowError) as exc:
        raise StoreError(f"Invalid {label}: {exc}") from exc


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise StoreError("Duplicate future JSON field")
        result[key] = value
    return result


def decode_json(raw):
    try:
        result = json.loads(raw, object_pairs_hook=_unique)
        json.dumps(result, allow_nan=False)
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object")
        return result
    except (ValueError, UnicodeError, TypeError, RecursionError) as exc:
        raise StoreError(f"Invalid future JSON record: {exc}") from exc


def resolved_configuration(config, origin):
    origin = timestamp(origin, "origin")
    if (
        config.forecast.origin != "latest"
        and timestamp(config.forecast.origin, "explicit origin") != origin
    ):
        raise StoreError("Explicit future origin disagrees with the resolved origin")
    result = config.resolved_dict()
    result["forecast"]["origin"] = origin.isoformat()
    return result


def validate_configuration(configuration):
    from tsfm_lab.future_config import future_config_from_mapping
    from tsfm_lab.local_paths import recorded_path_resolver

    try:
        config = future_config_from_mapping(
            configuration,
            base_directory=Path.cwd(),
            _path_resolver=recorded_path_resolver,
        )
        if config.forecast.origin == "latest":
            raise StoreError("Saved future configuration requires a concrete origin")
        if config.resolved_dict() != configuration:
            raise StoreError("Saved future configuration must be fully resolved")
        return config
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        raise StoreError(f"Invalid saved future configuration: {exc}") from exc


def _preprocessing(model, runtime):
    if model["source"] == "builtin":
        return {"policy": "identity_original_units_v1"}
    return {
        "policy": "native_adapter_and_inverse_v1",
        "adapter_id": model["adapter_id"],
        "adapter_version": model["adapter_version"],
        "point_forecast": model["options"]["checkpoint"].get("point_forecast"),
        "forecast_options": copy.deepcopy(runtime.get("forecast_options", {})),
    }


def build_result(
    config,
    dataset,
    forecast_result,
    run_id,
    *,
    started_at,
    completed_at,
    elapsed_seconds,
):
    """Construct the shared strict persisted shape from actual detached snapshots."""
    from tsfm_lab import __version__
    from tsfm_lab.evaluation.runner import _source_revision

    request, output = forecast_result.request, forecast_result.output
    configuration = resolved_configuration(config, request.origin)
    model, runtime = forecast_result.model, forecast_result.runtime
    requested_origin = config.entered_origin
    result = {
        "schema_version": 2,
        "mode": "future_forecast",
        "run_id": run_id,
        "status": "completed",
        "configuration": configuration,
        "request": {
            "request_id": request.request_id,
            "series_id": request.series_id,
            "origin": request.origin.isoformat(),
            "context_start": request.context_timestamps[0].isoformat(),
            "context_end": request.context_timestamps[-1].isoformat(),
            "context_length": len(request.context_values),
            "horizon": len(request.target_timestamps),
            "frequency": dataset.frequency,
            "target_timestamps": [
                value.isoformat() for value in request.target_timestamps
            ],
            "context_sha256": forecast_result.context_sha256,
        },
        "output": {
            "adapter_id": output.adapter_id,
            "requested_task": config.forecast.point_statistic,
            "point_statistic": output.point_statistic,
            "target_name": dataset.target_name,
            "units": dataset.units,
        },
        "counts": {
            "source_observations": forecast_result.source_observations,
            "context_observations": len(request.context_values),
            "forecast_requests": 1,
            "forecast_rows": len(output.point),
        },
        "timing": {
            "started_at": timestamp(started_at, "started_at").isoformat(),
            "completed_at": timestamp(completed_at, "completed_at").isoformat(),
            "elapsed_seconds": elapsed_seconds,
        },
        "provenance": {
            "input_sha256": hashlib.sha256(config.source_yaml.encode()).hexdigest(),
            "source": {"package_version": __version__, **_source_revision()},
            "environment": forecast_result.environment,
            "dataset": {
                **dataset.provenance,
                "canonical_sha256": canonical_fingerprint(dataset),
                "observations": len(dataset.values),
            },
            "context_sha256": forecast_result.context_sha256,
            "requested_origin": (
                requested_origin
                if isinstance(requested_origin, str)
                else requested_origin.isoformat()
            ),
            "model": model,
            "model_identity": forecast_result.model_identity,
            "runtime": runtime,
            "preprocessing": _preprocessing(model, runtime),
        },
        "evaluation": copy.deepcopy(UNAVAILABLE),
    }
    result = decode_json(json.dumps(result, allow_nan=False))
    predictions = pd.DataFrame(
        {
            "request_id": request.request_id,
            "series_id": request.series_id,
            "adapter_id": output.adapter_id,
            "origin": request.origin,
            "target_timestamp": request.target_timestamps,
            "lead": range(1, len(output.point) + 1),
            "prediction": output.point,
            "point_statistic": output.point_statistic,
        },
        columns=PREDICTION_COLUMNS,
    )
    observations = pd.DataFrame(
        {
            "series_id": request.series_id,
            "timestamp": request.context_timestamps,
            "observed_value": request.context_values,
        },
        columns=OBSERVATION_COLUMNS,
    )
    predictions, observations = validate_future(
        result, predictions, observations, configuration=configuration, run_id=run_id
    )
    return result, predictions, observations


def _frames(predictions, observations):
    frames = []
    for frame, columns, times, numbers in (
        (
            predictions,
            PREDICTION_COLUMNS,
            ("origin", "target_timestamp"),
            ("prediction", "lead"),
        ),
        (observations, OBSERVATION_COLUMNS, ("timestamp",), ("observed_value",)),
    ):
        if (
            not isinstance(frame, pd.DataFrame)
            or tuple(frame.columns) != columns
            or frame.empty
        ):
            raise StoreError(
                "Future CSV must have exact ordered columns and nonempty rows"
            )
        frame = frame.copy(deep=True)
        for column in set(columns) - set(times) - set(numbers):
            if (
                not frame[column]
                .map(lambda value: isinstance(value, str) and bool(value.strip()))
                .all()
            ):
                raise StoreError(f"Future {column} values must be text")
        for column in times:
            frame[column] = pd.to_datetime(
                [timestamp(value, column) for value in frame[column]], utc=True
            )
        for column in numbers:
            values = pd.to_numeric(frame[column], errors="raise")
            if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
                raise StoreError(f"Future {column} values must be finite real numbers")
            frame[column] = values.astype("float64")
        frames.append(frame)
    predictions, observations = frames
    leads = predictions.lead
    if (leads < 1).any() or (leads % 1 != 0).any() or (leads > 10_000).any():
        raise StoreError("Future leads must be positive supported integers")
    predictions["lead"] = leads.astype("int64")
    return predictions, observations


def _provenance(result, config):
    from tsfm_lab.future import model_identity
    from tsfm_lab.models.registry import adapter_profile

    provenance = object_fields(
        result["provenance"],
        {
            "source",
            "input_sha256",
            "environment",
            "dataset",
            "context_sha256",
            "requested_origin",
            "model",
            "model_identity",
            "runtime",
            "preprocessing",
        },
        "future provenance",
    )
    source = object_fields(
        provenance["source"],
        {"package_version", "git_revision", "git_dirty"},
        "implementation source",
    )
    text(source["package_version"], "package version")
    identity(provenance["input_sha256"], "input YAML fingerprint")
    if source["git_revision"] is not None:
        text(source["git_revision"], "source revision")
    if source["git_dirty"] is not None and type(source["git_dirty"]) is not bool:
        raise StoreError("Source dirty status must be boolean or null")
    dataset = object_fields(
        provenance["dataset"],
        {"canonical_sha256", "source_sha256", "observations"},
        "source provenance",
        optional={
            "source_path",
            "parser_policy",
            "timestamp_column",
            "target_column",
            "source_timezone",
        },
    )
    identity(dataset["canonical_sha256"], "source canonical fingerprint")
    identity(dataset["source_sha256"], "raw source fingerprint")
    integer(dataset["observations"], "source observations")
    for key in set(dataset) - {"observations", "canonical_sha256", "source_sha256"}:
        text(dataset[key], f"source {key}")
    for key in ("source_path", "timestamp_column", "target_column"):
        configured = "path" if key == "source_path" else key
        if key in dataset and dataset[key] != config.dataset[configured]:
            raise StoreError("Source provenance disagrees with the saved configuration")
    model = object_fields(
        provenance["model"],
        {
            "adapter_id",
            "adapter_version",
            "family",
            "source",
            "options",
            "capabilities",
        },
        "model descriptor",
    )
    for field in ("adapter_id", "adapter_version", "family", "source"):
        text(model[field], f"model {field}")
    if (
        model["adapter_id"] != config.model["adapter_id"]
        or model["source"] != config.model["source"]
    ):
        raise StoreError("Future model descriptor disagrees with the configuration")
    object_fields(
        model["capabilities"],
        set(ModelCapabilities.__dataclass_fields__),
        "model capabilities",
    )
    caps = ModelCapabilities(**model["capabilities"])
    if (
        caps.target_count != 1
        or caps.covariates is not False
        or caps.output_types != ["point"]
        or caps.devices != ["cpu"]
    ):
        raise StoreError("Future records require univariate CPU point capabilities")
    if (
        config.forecast.context_length < caps.min_context
        or (
            caps.max_context is not None
            and config.forecast.context_length > caps.max_context
        )
        or (caps.max_horizon is not None and config.forecast.horizon > caps.max_horizon)
    ):
        raise StoreError("Future request exceeds recorded model capabilities")
    runtime = object_fields(
        provenance["runtime"],
        {"device", "dtype", "batch_size", "seed", "seed_used", "deterministic"},
        "effective runtime",
        optional={"torch_compile", "forecast_options"},
    )
    if any(
        runtime[key] != value or type(runtime[key]) is not type(value)
        for key, value in config.runtime.items()
    ):
        raise StoreError("Effective runtime disagrees with configuration")
    if runtime["seed_used"] is not False or runtime["deterministic"] is not True:
        raise StoreError(
            "Curated future runtime requires unused seed and deterministic execution"
        )
    if "torch_compile" in runtime and runtime["torch_compile"] is not False:
        raise StoreError("Curated future runtime requires eager execution")
    if model["source"] == "builtin" and {"torch_compile", "forecast_options"} & set(
        runtime
    ):
        raise StoreError("Baseline runtime cannot claim checkpoint forecast options")
    environment = provenance["environment"]
    packages = (
        set()
        if model["source"] == "builtin"
        else {
            name
            for name, _ in adapter_profile(model["adapter_id"]).runtime_requirements
        }
    )
    from tsfm_lab.platform_runtime import optional_provenance_packages

    packages |= optional_provenance_packages(model["adapter_id"])
    object_fields(
        environment,
        {"python", "numpy", "pandas", "os", "machine"},
        "runtime environment",
        optional=packages,
    )
    for value in environment.values():
        text(value, "environment version")
    expected = model_identity(
        model["adapter_id"], model["adapter_version"], model["options"]
    )
    if provenance["model_identity"] != expected or provenance[
        "preprocessing"
    ] != _preprocessing(model, runtime):
        raise StoreError("Future numerical model or preprocessing identity disagrees")
    if model["source"] == "builtin":
        period = config.model["options"].get("seasonal_period", 1)
        if (
            model["family"] != "baseline"
            or model["options"] != config.model["options"]
            or model["capabilities"]
            != json.loads(json.dumps(asdict(ModelCapabilities(min_context=period))))
        ):
            raise StoreError("Baseline descriptor disagrees with its numerical options")
    else:
        _checkpoint_identity(
            model, runtime, config.forecast, environment, config.dataset["frequency"]
        )
        checkpoint = model["options"]["checkpoint"]
        if model["source"] == "local":
            if checkpoint["path"] != config.model["path"]:
                raise StoreError("Local checkpoint path disagrees with configuration")
        elif (
            checkpoint["repository"] != config.model["repository"]
            or checkpoint["resolved_revision"] != config.model["revision"]
        ):
            raise StoreError("Checkpoint revision disagrees with saved configuration")
    return provenance


def _metadata(result, configuration, run_id):
    object_fields(
        result,
        {
            "schema_version",
            "mode",
            "run_id",
            "status",
            "configuration",
            "request",
            "output",
            "counts",
            "timing",
            "provenance",
            "evaluation",
        },
        "future result",
    )
    if (
        type(result["schema_version"]) is not int
        or result["schema_version"] != 2
        or result["mode"] != "future_forecast"
        or result["status"] != "completed"
    ):
        raise StoreError("Invalid future result schema, mode, or status")
    if (
        not isinstance(result["run_id"], str)
        or str(UUID(result["run_id"])) != result["run_id"]
        or (run_id is not None and result["run_id"] != run_id)
    ):
        raise StoreError("Future run identity disagrees")
    if configuration is not None and result["configuration"] != configuration:
        raise StoreError(
            "Future result configuration disagrees with its initial snapshot"
        )
    config = validate_configuration(result["configuration"])
    if (
        result["evaluation"] != UNAVAILABLE
        or type(result["evaluation"].get("schema_version")) is not int
    ):
        raise StoreError("Future accuracy must be unavailable: future_actuals_absent")
    timing = object_fields(
        result["timing"],
        {"started_at", "completed_at", "elapsed_seconds"},
        "future timing",
    )
    if timestamp(timing["completed_at"], "completion") < timestamp(
        timing["started_at"], "start"
    ):
        raise StoreError("Future completion precedes its start")
    elapsed = timing["elapsed_seconds"]
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise StoreError("Future elapsed time must be finite and nonnegative")
    counts = object_fields(
        result["counts"],
        {
            "source_observations",
            "context_observations",
            "forecast_requests",
            "forecast_rows",
        },
        "future counts",
    )
    for key, value in counts.items():
        integer(value, key)
    if (
        counts["source_observations"] < config.forecast.context_length
        or counts["context_observations"] != config.forecast.context_length
        or counts["forecast_requests"] != 1
        or counts["forecast_rows"] != config.forecast.horizon
    ):
        raise StoreError("Future counts disagree with the configured request")
    return config, _provenance(result, config)


def validate_future(
    result, predictions, observations, *, configuration=None, run_id=None
):
    """Validate strict metadata, exact grids, and reconstructed numerical identities."""
    from tsfm_lab.future import context_identity, request_identity

    try:
        config, provenance = _metadata(result, configuration, run_id)
        predictions, observations = _frames(predictions, observations)
        spec, dataset = config.forecast, config.dataset
        if len(predictions) != spec.horizon or len(observations) != spec.context_length:
            raise StoreError("Future CSV lengths disagree with L/H")
        request = object_fields(
            result["request"],
            {
                "request_id",
                "series_id",
                "origin",
                "context_start",
                "context_end",
                "context_length",
                "horizon",
                "frequency",
                "target_timestamps",
                "context_sha256",
            },
            "future request",
        )
        output = object_fields(
            result["output"],
            {"adapter_id", "requested_task", "point_statistic", "target_name", "units"},
            "future output",
        )
        origin = timestamp(request["origin"], "origin")
        delta = int(fixed_interval(dataset["frequency"]).value)
        if (
            origin != spec.origin
            or request["series_id"] != dataset["series_id"]
            or request["frequency"] != dataset["frequency"]
            or type(request["context_length"]) is not int
            or request["context_length"] != spec.context_length
            or type(request["horizon"]) is not int
            or request["horizon"] != spec.horizon
        ):
            raise StoreError("Future request disagrees with configuration")
        expected = [
            int(origin.value) + step * delta for step in range(1, spec.horizon + 1)
        ]
        if expected[-1] > int(pd.Timestamp.max.value):
            raise StoreError("Future grid exceeds nanosecond timestamp bounds")
        targets = pd.to_datetime(expected, unit="ns", utc=True)
        if (
            type(request["target_timestamps"]) is not list
            or len(request["target_timestamps"]) != spec.horizon
            or not pd.DatetimeIndex(
                [timestamp(value, "target") for value in request["target_timestamps"]]
            ).equals(targets)
        ):
            raise StoreError("Future request target grid disagrees")
        start = int(origin.value) - (spec.context_length - 1) * delta
        if start < int(pd.Timestamp.min.value):
            raise StoreError("Consumed context exceeds nanosecond timestamp bounds")
        context = pd.to_datetime(
            [start + step * delta for step in range(spec.context_length)],
            unit="ns",
            utc=True,
        )
        if not observations.timestamp.equals(
            pd.Series(context, name="timestamp", index=observations.index)
        ) or set(observations.series_id) != {dataset["series_id"]}:
            raise StoreError("Observed context grid or series identity disagrees")
        if (
            timestamp(request["context_start"], "context start") != context[0]
            or timestamp(request["context_end"], "context end") != origin
        ):
            raise StoreError("Observed context boundaries disagree")
        for name, wanted in (
            ("request_id", request["request_id"]),
            ("series_id", dataset["series_id"]),
            ("adapter_id", config.model["adapter_id"]),
            ("point_statistic", output["point_statistic"]),
        ):
            if set(predictions[name]) != {wanted}:
                raise StoreError(f"Future prediction {name} disagrees")
        if (
            set(predictions.origin) != {origin}
            or list(predictions.lead) != list(range(1, spec.horizon + 1))
            or not predictions.target_timestamp.equals(
                pd.Series(targets, name="target_timestamp", index=predictions.index)
            )
        ):
            raise StoreError("Future prediction origin, leads, or grid disagree")
        if output != {
            "adapter_id": config.model["adapter_id"],
            "requested_task": spec.point_statistic,
            "point_statistic": output["point_statistic"],
            "target_name": dataset["target_column"],
            "units": dataset["units"],
        } or output["point_statistic"] not in output_statistics(spec.point_statistic):
            raise StoreError("Future output point semantics or measurement disagrees")
        from tsfm_lab.models.settings import model_settings

        expected_statistic = (
            "deterministic"
            if config.model["source"] == "builtin"
            else model_settings(config.model["adapter_id"]).supported_tasks[0]
        )
        if output["point_statistic"] != expected_statistic:
            raise StoreError(
                "Future output actual statistic disagrees with its adapter"
            )
        snapshot = ForecastRequest(
            request["request_id"],
            dataset["series_id"],
            origin,
            context,
            observations.observed_value.to_numpy(),
            targets,
        )
        measurement = {
            "target_name": dataset["target_column"],
            "units": dataset["units"],
            "frequency": dataset["frequency"],
        }
        consumed = context_identity(snapshot, **measurement)
        if (
            consumed != request["context_sha256"]
            or consumed != provenance["context_sha256"]
        ):
            raise StoreError(
                "Consumed-context identity disagrees with saved observations"
            )
        calculated = request_identity(
            snapshot,
            **measurement,
            point_statistic=spec.point_statistic,
            model_identity=provenance["model_identity"],
            runtime=provenance["runtime"],
        )
        if identity(request["request_id"], "request identity") != calculated:
            raise StoreError("Future numerical request identity disagrees")
        if (
            provenance["dataset"]["observations"]
            != result["counts"]["source_observations"]
        ):
            raise StoreError("Source counts disagree across metadata")
        entered = provenance["requested_origin"]
        if entered != "latest" and timestamp(entered, "entered origin") != origin:
            raise StoreError("Entered origin policy disagrees with resolved origin")
        observed = Dataset(
            context,
            observations.observed_value.to_numpy(),
            dataset["frequency"],
            target_name=dataset["target_column"],
            units=dataset["units"],
            series_id=dataset["series_id"],
            provenance={
                key: value
                for key, value in provenance["dataset"].items()
                if key != "observations"
            },
        )
        from tsfm_lab.workflows import validate_workflow_dataset

        validate_workflow_dataset(config.workflow, observed)
        if (
            result["counts"]["source_observations"] == spec.context_length
            and canonical_fingerprint(observed)
            != provenance["dataset"]["canonical_sha256"]
        ):
            raise StoreError("Complete saved source context fingerprint disagrees")
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
        raise StoreError(f"Invalid future artifacts: {exc}") from exc
    return predictions, observations
