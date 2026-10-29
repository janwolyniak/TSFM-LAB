"""Terminal-independent draft validation for the guided experiment workflow."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tsfm_lab.application import load_dataset
from tsfm_lab.config import ConfigError, ExperimentConfig, config_from_mapping
from tsfm_lab.config import _dataset as validated_dataset_options
from tsfm_lab.contracts import Dataset, fixed_interval
from tsfm_lab.data import canonical_fingerprint
from tsfm_lab.evaluation.windows import evaluation_bounds
from tsfm_lab.models.catalog import (
    CHECKPOINT_REPOSITORIES,
    catalog_entries,
    catalog_info,
)
from tsfm_lab.models.factory import build_model
from tsfm_lab.models.settings import model_settings
from tsfm_lab.point_tasks import validate_model_frequency
from tsfm_lab.workflows import (
    normalize_workflow,
    validate_workflow_dataset,
    workflow_summary,
)

_DEFAULTS = {
    "mode": "historical_evaluation",
    "forecast_origin": "latest",
    "name": "New experiment",
    "adapter": "last_value",
    "point_statistic": "median",
    "seasonal_period": "4",
    "source": "huggingface",
    "repository": CHECKPOINT_REPOSITORIES["chronos_bolt"][0],
    "revision": "",
    "cache_dir": "",
    "checkpoint_path": "",
    "dataset_path": "",
    "timestamp_column": "timestamp",
    "target_column": "demand",
    "frequency": "1h",
    "timezone": "UTC",
    "units": "unspecified",
    "context_length": "8",
    "horizon": "4",
    "stride": "4",
    "mase_period": "4",
    "evaluation_start": "",
    "evaluation_end": "",
    "usage_profile": "",
    "finance_target_kind": "",
    "finance_measurement": "",
    "energy_measurement": "",
    "energy_quantity": "",
    "weather_variable": "",
    "weather_station": "",
    "weather_measurement_semantics": "",
    "weather_description": "",
}
_LABELS = {
    "seasonal_period": "Baseline seasonal period",
    "context_length": "Context length",
    "horizon": "Forecast horizon",
    "stride": "Evaluation stride",
    "mase_period": "MASE seasonal lag",
}


def default_fields(initial_adapter: str = "last_value") -> dict[str, str]:
    if initial_adapter not in {entry["adapter_id"] for entry in catalog_entries()}:
        raise ConfigError("Choose a supported model adapter")
    fields = {**_DEFAULTS, "adapter": initial_adapter}
    settings = model_settings(initial_adapter)
    fields["context_length"] = str(settings.default_context)
    fields["horizon"] = str(settings.default_horizon)
    info = catalog_info(initial_adapter)
    if info["point_statistic"] in info["supported_tasks"]:
        fields["point_statistic"] = info["point_statistic"]
    if initial_adapter in CHECKPOINT_REPOSITORIES:
        fields["repository"] = CHECKPOINT_REPOSITORIES[initial_adapter][0]
        variant = info["variants"][0]
        if variant["min_context"] == variant["max_context"]:
            fields["context_length"] = str(variant["min_context"])
    return fields


def demo_fields(path: Path) -> dict[str, str]:
    return {
        **default_fields(),
        "name": "Seasonal demo",
        "dataset_path": str(path),
        "units": "synthetic_units",
    }


def profile_example_fields(profile: str, path: Path) -> dict[str, str]:
    """Describe original fictional examples without converting their observations."""
    choices = {
        "finance": (
            "close",
            "toy_quote_per_asset",
            {"finance_target_kind": "close_price"},
        ),
        "energy": ("power", "kW", {"energy_measurement": "power"}),
        "temperature": ("temperature", "degC", {}),
        "weather": (
            "wind_speed",
            "m/s",
            {
                "weather_variable": "wind_speed",
                "weather_station": "synthetic-station",
                "weather_measurement_semantics": "instantaneous",
            },
        ),
    }
    if profile not in choices:
        return demo_fields(path)
    target, units, options = choices[profile]
    return {
        **demo_fields(path),
        "usage_profile": profile,
        "target_column": target,
        "units": units,
        **options,
    }


def workflow_from_fields(fields: dict[str, str]) -> dict | None:
    """Serialize only the selected profile's reviewed options."""
    profile = fields.get("usage_profile", "")
    if not profile:
        return None
    options = {}
    if profile == "finance":
        options["target_kind"] = fields["finance_target_kind"]
        if options["target_kind"] == "other":
            options["measurement"] = fields["finance_measurement"]
    elif profile == "energy":
        options["measurement"] = fields["energy_measurement"]
        if options["measurement"] == "other":
            options["quantity"] = fields["energy_quantity"]
    elif profile == "weather":
        options = {
            "variable": fields["weather_variable"],
            "station": fields["weather_station"],
            "measurement_semantics": fields["weather_measurement_semantics"],
        }
        if options["measurement_semantics"] == "other":
            options["description"] = fields["weather_description"]
    try:
        return normalize_workflow(
            {"schema_version": 1, "profile": profile, "options": options}
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def profile_guidance(profile: str, frequency: str = "") -> str:
    guidance = {
        "finance": "Declare the measured financial target and its units. OHLC reports require separate execution data and explicit timing and cost inputs.",
        "energy": "Power and energy are different measurements. Keep the input units and interval; no conversion or aggregation is applied.",
        "temperature": "Keep the declared temperature units, including legitimate negative readings. Values are not clipped or converted.",
        "weather": "Use one measured variable from one station and declare whether each value is instantaneous or summarizes an interval.",
    }.get(
        profile,
        "Forecast a regular time series in its declared units; choose historical evaluation or future forecasting after preview.",
    )
    if profile in {"energy", "temperature", "weather"} and frequency:
        if fixed_interval(frequency) == fixed_interval("1h"):
            guidance += " Seasonal examples for hourly data: 24 observations per day or 168 per week. Choose lags explicitly; current settings are retained."
    return guidance


def _draft(fields: dict[str, str]) -> dict[str, str]:
    if not isinstance(fields, dict) or set(fields) - set(_DEFAULTS):
        raise ConfigError("The experiment form contains unknown fields")
    if any(not isinstance(value, str) for value in fields.values()):
        raise ConfigError("Experiment fields must contain text")
    return {**_DEFAULTS, **fields}


def _integer(fields: dict[str, str], name: str) -> int:
    value = fields[name].strip()
    if re.fullmatch(r"[0-9]+", value) is None:
        raise ConfigError(f"{_LABELS[name]} must be a positive whole number")
    try:
        result = int(value)
    except ValueError as exc:
        raise ConfigError(f"{_LABELS[name]} is too large") from exc
    if result <= 0:
        raise ConfigError(f"{_LABELS[name]} must be greater than zero")
    return result


def _model(fields: dict[str, str]) -> dict:
    adapter = fields["adapter"]
    if adapter in {"last_value", "seasonal_naive"}:
        options = (
            {"seasonal_period": _integer(fields, "seasonal_period")}
            if adapter == "seasonal_naive"
            else {}
        )
        return {"source": "builtin", "adapter_id": adapter, "options": options}
    if adapter not in CHECKPOINT_REPOSITORIES:
        raise ConfigError("Choose a supported model adapter")
    result = {"source": fields["source"], "adapter_id": adapter, "options": {}}
    if fields["source"] == "local":
        result["path"] = fields["checkpoint_path"]
    elif fields["source"] == "huggingface":
        result.update(
            repository=fields["repository"], revision=fields["revision"], offline=True
        )
        if fields["cache_dir"].strip():
            result["cache_dir"] = fields["cache_dir"]
    else:
        raise ConfigError("Choose a cached Hugging Face checkpoint or local directory")
    return result


def _dataset(fields: dict[str, str]) -> dict:
    result = {
        "path": fields["dataset_path"],
        "timestamp_column": fields["timestamp_column"],
        "target_column": fields["target_column"],
        "frequency": fields["frequency"],
        "units": fields["units"],
    }
    if fields["timezone"].strip():
        result["timezone"] = fields["timezone"]
    return result


def _mapping(fields: dict[str, str], workspace: Path) -> dict:
    mapping = {
        "schema_version": 1,
        "name": fields["name"],
        "model": _model(fields),
        "dataset": _dataset(fields),
    }
    mode = fields["mode"]
    if mode == "future_forecast":
        mapping.update(
            schema_version=2,
            mode=mode,
            forecast={
                "context_length": _integer(fields, "context_length"),
                "horizon": _integer(fields, "horizon"),
                "point_statistic": fields["point_statistic"],
                "origin": fields["forecast_origin"] or "latest",
            },
        )
    elif mode == "historical_evaluation":
        mapping["evaluation"] = {
            "context_length": _integer(fields, "context_length"),
            "horizon": _integer(fields, "horizon"),
            "stride": _integer(fields, "stride"),
            "seasonal_period": _integer(fields, "mase_period"),
            "evaluation_start": fields["evaluation_start"],
            "evaluation_end": fields["evaluation_end"],
            "metrics": ["mae", "rmse", "mase"],
            "point_statistic": fields["point_statistic"],
        }
    else:
        raise ConfigError("Choose historical evaluation or future forecasting")
    mapping["output"] = {"directory": str(workspace)}
    workflow = workflow_from_fields(fields)
    if workflow is not None:
        mapping["workflow"] = workflow
    return mapping


def _minimal_mapping(workspace: Path) -> dict:
    fields = {
        **default_fields(),
        "dataset_path": "validation.csv",
        "context_length": "1",
        "horizon": "1",
        "evaluation_start": "2000-01-01T01:00:00Z",
        "evaluation_end": "2000-01-01T02:00:00Z",
    }
    return _mapping(fields, workspace)


def validate_model_fields(
    fields: dict[str, str], *, workspace: Path, base_directory: Path
) -> None:
    fields = _draft(fields)
    selection = _model(fields)
    settings = (
        model_settings(
            fields["adapter"],
            repository=(
                fields["repository"] if fields["source"] == "huggingface" else None
            ),
            revision=(
                fields["revision"] or None
                if fields["source"] == "huggingface"
                else None
            ),
        )
        if fields["adapter"] in CHECKPOINT_REPOSITORIES
        else model_settings(fields["adapter"])
    )
    context = settings.default_context
    if fields["adapter"] == "seasonal_naive":
        context = _integer(fields, "seasonal_period")
    if fields["mode"] == "future_forecast":
        from tsfm_lab.future_config import future_config_from_mapping

        prototype = {
            **fields,
            "context_length": str(context),
            "horizon": "1",
            "dataset_path": "validation.csv",
            "frequency": "1h",
            "forecast_origin": "latest",
        }
        future_config_from_mapping(
            _mapping(prototype, workspace), base_directory=base_directory
        )
    else:
        mapping = _minimal_mapping(workspace)
        mapping["name"], mapping["model"] = fields["name"], selection
        mapping["evaluation"].update(
            point_statistic=fields["point_statistic"], context_length=context
        )
        config_from_mapping(mapping, base_directory=base_directory)


def inspect_data(
    fields: dict[str, str], *, workspace: Path, base_directory: Path
) -> Dataset:
    """Validate file mapping and read data; callers schedule this off the UI thread."""
    fields = _draft(fields)
    options = validated_dataset_options(_dataset(fields), Path(base_directory))
    try:
        dataset = load_dataset(options)
        validate_workflow_dataset(workflow_from_fields(fields), dataset)
        return dataset
    except (ValueError, OSError, ImportError) as exc:
        raise ConfigError(str(exc)) from exc


def suggested_bounds(dataset: Dataset, fields: dict[str, str]) -> dict[str, str]:
    """Propose initial bounds; the caller never applies them over existing input."""
    fields = _draft(fields)
    context, horizon = _integer(fields, "context_length"), _integer(fields, "horizon")
    if len(dataset.values) < context + horizon:
        raise ConfigError("Adjust context and horizon to fit the inspected dataset")
    start = max(context, len(dataset.values) // 2)
    if start + horizon > len(dataset.values):
        start = context
    return {
        "evaluation_start": dataset.timestamps[start].isoformat(),
        "evaluation_end": (
            dataset.timestamps[-1] + fixed_interval(dataset.frequency)
        ).isoformat(),
    }


def config_from_fields(
    fields: dict[str, str], *, workspace: Path, base_directory: Path
) -> ExperimentConfig:
    fields = _draft(fields)
    if fields["mode"] == "future_forecast":
        from tsfm_lab.future_config import future_config_from_mapping

        return future_config_from_mapping(
            _mapping(fields, workspace), base_directory=base_directory
        )
    return config_from_mapping(
        _mapping(fields, workspace), base_directory=base_directory
    )


@dataclass(frozen=True)
class PreparedExperiment:
    config: ExperimentConfig
    windows: int
    forecast_cases: int
    observations: int

    def summary(self) -> str:
        config, spec = self.config, self.config.evaluation
        model = config.model
        details = model.get("path") or model.get("repository", "Built-in baseline")
        display_name = catalog_info(model["adapter_id"])["display_name"]
        revision = (
            f"\nCheckpoint SHA: {model['revision']} (offline)"
            if "revision" in model
            else ""
        )
        season = model["options"].get("seasonal_period")
        baseline = f"\nBaseline seasonal period: {season}" if season else ""
        return (
            f"{config.name}\n\nUsage: {workflow_summary(config.workflow)}\nModel: {display_name}\nTarget: {config.dataset['target_column']} "
            f"({config.dataset['units']}) · Frequency: {config.dataset['frequency']}\n"
            f"Context: {spec.context_length} observations · Horizon: {spec.horizon} observations\n"
            f"Forecast task: {spec.point_statistic}\n"
            f"Stride: {spec.stride} · MASE seasonal lag: {spec.seasonal_period}{baseline}\n"
            f"Windows: {self.windows} · Forecast cases: {self.forecast_cases}\n"
            f"Observations: {self.observations}\n\n"
            f"Start: {spec.evaluation_start.isoformat()}\n"
            f"End (exclusive): {spec.evaluation_end.isoformat()}\n"
            f"Workspace: {config.output_directory}\n"
            f"\nDataset: {config.dataset['path']}\nSource: {details}{revision}"
        )


@dataclass(frozen=True)
class PreparedFutureForecast:
    config: object
    observations: int
    request: object
    dataset_sha256: str
    source_sha256: str

    def review_plan(self):
        """Retain the displayed source identity through screen confirmation."""
        from tsfm_lab.guided import review_guided

        plan = review_guided(self.config, offline=True)
        if (
            plan.dataset_sha256 != self.dataset_sha256
            or plan.source_sha256 != self.source_sha256
        ):
            raise ConfigError(
                "The dataset changed during review; review the forecast again"
            )
        return plan

    def summary(self):
        config, spec, request = self.config, self.config.forecast, self.request
        return (
            f"{config.name}\n\nMode: Future forecasting\nUsage: {workflow_summary(config.workflow)}\n"
            f"Model: {catalog_info(config.model['adapter_id'])['display_name']}\n"
            f"Target: {config.dataset['target_column']} ({config.dataset['units']}) · Frequency: {config.dataset['frequency']}\n"
            f"Context: {spec.context_length} observations · Horizon: {spec.horizon} observations\n"
            f"Forecast task: {spec.point_statistic}\nOrigin: {request.origin.isoformat()}\n"
            f"Future targets: {request.target_timestamps[0].isoformat()} to {request.target_timestamps[-1].isoformat()}\n"
            "Accuracy unavailable: future_actuals_absent\n"
            f"Workspace: {config.output_directory}"
        )


def prepare_experiment(
    fields: dict[str, str],
    *,
    workspace: Path,
    base_directory: Path,
    resolve_model: bool = True,
) -> PreparedExperiment:
    """Validate the complete task without model initialization or downloads."""
    config = config_from_fields(
        fields, workspace=workspace, base_directory=base_directory
    )
    if fields.get("mode") == "future_forecast":
        from tsfm_lab.future import prepare_future_request

        dataset = load_dataset(config.dataset)
        validate_workflow_dataset(config.workflow, dataset)
        request = prepare_future_request(dataset, config.forecast, model=config.model)
        if resolve_model:
            model = build_model(config.model, config.runtime)
            validate_model_frequency(model, dataset.frequency)
            model.validate(config.forecast)
        return PreparedFutureForecast(
            config,
            len(dataset.values),
            request,
            canonical_fingerprint(dataset),
            dataset.provenance["source_sha256"],
        )
    try:
        dataset = load_dataset(config.dataset)
        validate_workflow_dataset(config.workflow, dataset)
        start, end = evaluation_bounds(dataset, config.evaluation)
        if resolve_model:
            model = build_model(config.model, config.runtime)
            validate_model_frequency(model, dataset.frequency)
            model.validate(config.evaluation)
    except (ValueError, OSError, ImportError) as exc:
        raise ConfigError(str(exc)) from exc
    spec = config.evaluation
    windows = 1 + (end - start - spec.horizon) // spec.stride
    return PreparedExperiment(
        config, windows, windows * spec.horizon, len(dataset.values)
    )
