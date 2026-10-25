"""Terminal-independent draft validation for the guided experiment workflow."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tsfm_lab.application import load_dataset
from tsfm_lab.config import ConfigError, ExperimentConfig, config_from_mapping
from tsfm_lab.contracts import Dataset, fixed_interval
from tsfm_lab.evaluation.windows import evaluation_bounds
from tsfm_lab.models.factory import build_model

_DEFAULTS = {
    "name": "New experiment",
    "adapter": "last_value",
    "seasonal_period": "4",
    "source": "huggingface",
    "repository": "amazon/chronos-bolt-tiny",
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
}
_LABELS = {
    "seasonal_period": "Baseline seasonal period",
    "context_length": "Context length",
    "horizon": "Forecast horizon",
    "stride": "Evaluation stride",
    "mase_period": "MASE seasonal lag",
}


def default_fields(initial_adapter: str = "last_value") -> dict[str, str]:
    if initial_adapter not in {"last_value", "seasonal_naive", "chronos_bolt"}:
        raise ConfigError("Choose a supported model adapter")
    return {**_DEFAULTS, "adapter": initial_adapter}


def demo_fields(path: Path) -> dict[str, str]:
    return {
        **default_fields(),
        "name": "Seasonal demo",
        "dataset_path": str(path),
        "units": "synthetic_units",
    }


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
    if adapter != "chronos_bolt":
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
    return {
        "schema_version": 1,
        "name": fields["name"],
        "model": _model(fields),
        "dataset": _dataset(fields),
        "evaluation": {
            "context_length": _integer(fields, "context_length"),
            "horizon": _integer(fields, "horizon"),
            "stride": _integer(fields, "stride"),
            "seasonal_period": _integer(fields, "mase_period"),
            "evaluation_start": fields["evaluation_start"],
            "evaluation_end": fields["evaluation_end"],
            "metrics": ["mae", "rmse", "mase"],
            "point_statistic": "median",
        },
        "output": {"directory": str(workspace)},
    }


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
    mapping = _minimal_mapping(workspace)
    mapping["name"], mapping["model"] = fields["name"], _model(fields)
    if fields["adapter"] == "seasonal_naive":
        mapping["evaluation"]["context_length"] = _integer(fields, "seasonal_period")
    config_from_mapping(mapping, base_directory=base_directory)


def inspect_data(
    fields: dict[str, str], *, workspace: Path, base_directory: Path
) -> Dataset:
    """Validate file mapping and read data; callers schedule this off the UI thread."""
    mapping = _minimal_mapping(workspace)
    mapping["dataset"] = _dataset(_draft(fields))
    config = config_from_mapping(mapping, base_directory=base_directory)
    try:
        return load_dataset(config.dataset)
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
    return config_from_mapping(
        _mapping(_draft(fields), workspace), base_directory=base_directory
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
        display_name = {
            "last_value": "Last value",
            "seasonal_naive": "Seasonal naive",
            "chronos_bolt": "Chronos-Bolt",
        }[model["adapter_id"]]
        revision = (
            f"\nCheckpoint SHA: {model['revision']} (offline)"
            if "revision" in model
            else ""
        )
        season = model["options"].get("seasonal_period")
        baseline = f"\nBaseline seasonal period: {season}" if season else ""
        return (
            f"{config.name}\n\nModel: {display_name}\nTarget: {config.dataset['target_column']} "
            f"({config.dataset['units']}) · Frequency: {config.dataset['frequency']}\n"
            f"Context: {spec.context_length} observations · Horizon: {spec.horizon} observations\n"
            f"Stride: {spec.stride} · MASE seasonal lag: {spec.seasonal_period}{baseline}\n"
            f"Windows: {self.windows} · Forecast cases: {self.forecast_cases}\n"
            f"Observations: {self.observations}\n\n"
            f"Start: {spec.evaluation_start.isoformat()}\n"
            f"End (exclusive): {spec.evaluation_end.isoformat()}\n"
            f"Workspace: {config.output_directory}\n"
            f"\nDataset: {config.dataset['path']}\nSource: {details}{revision}"
        )


def prepare_experiment(
    fields: dict[str, str], *, workspace: Path, base_directory: Path
) -> PreparedExperiment:
    """Validate the complete task without model initialization or downloads."""
    config = config_from_fields(
        fields, workspace=workspace, base_directory=base_directory
    )
    try:
        dataset = load_dataset(config.dataset)
        start, end = evaluation_bounds(dataset, config.evaluation)
        model = build_model(config.model, config.runtime)
        model.validate(config.evaluation)
    except (ValueError, OSError, ImportError) as exc:
        raise ConfigError(str(exc)) from exc
    spec = config.evaluation
    windows = 1 + (end - start - spec.horizon) // spec.stride
    return PreparedExperiment(
        config, windows, windows * spec.horizon, len(dataset.values)
    )
