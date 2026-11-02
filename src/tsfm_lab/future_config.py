"""Strict future configuration without historical evaluation placeholders."""

import copy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path, PurePath
from typing import Any

import pandas as pd
import yaml

from tsfm_lab.config import (
    MAX_CONFIG_BYTES,
    ConfigError,
    _ConfigLoader,
    _dataset,
    _integer,
    _local_path,
    _mapping,
    _model,
    _runtime,
    _text,
)
from tsfm_lab.contracts import positive_int, utc_timestamp
from tsfm_lab.local_paths import expand_local_user
from tsfm_lab.models.settings import model_settings, validate_settings
from tsfm_lab.point_tasks import output_statistics
from tsfm_lab.workflows import normalize_workflow, validate_workflow_units

MAX_FUTURE_CONTEXT = 100_000
MAX_FUTURE_HORIZON = 10_000


@dataclass(frozen=True)
class FutureForecastSpec:
    context_length: int
    horizon: int
    point_statistic: str = "median"
    origin: str | pd.Timestamp = "latest"

    def __post_init__(self):
        for field, maximum in (
            ("context_length", MAX_FUTURE_CONTEXT),
            ("horizon", MAX_FUTURE_HORIZON),
        ):
            value = positive_int(getattr(self, field), field)
            if value > maximum:
                raise ValueError(
                    f"Future {field} must not exceed {maximum} observations"
                )
            object.__setattr__(self, field, value)
        output_statistics(self.point_statistic)
        if not isinstance(self.origin, str) or self.origin != "latest":
            try:
                origin = utc_timestamp(self.origin, "forecast.origin").as_unit("ns")
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(
                    "forecast.origin requires latest or an aware UTC-nanosecond timestamp"
                ) from exc
            object.__setattr__(self, "origin", origin)


@dataclass(frozen=True)
class FutureForecastConfig:
    name: str
    dataset: dict[str, Any]
    model: dict[str, Any]
    forecast: FutureForecastSpec
    runtime: dict[str, Any]
    output_directory: PurePath
    source_yaml: str
    source_path: Path
    workflow: dict[str, Any] | None = None
    _record_only: bool = dataclass_field(default=False, repr=False, compare=False)

    @property
    def entered_origin(self):
        source = yaml.load(self.source_yaml, Loader=_ConfigLoader)
        if not isinstance(source, dict) or not isinstance(source.get("forecast"), dict):
            raise ConfigError("Future configuration lacks its original forecast input")
        return source["forecast"].get("origin", "latest")

    def resolved_dict(self) -> dict:
        origin = self.forecast.origin
        return {
            "schema_version": 2,
            "mode": "future_forecast",
            "name": self.name,
            "model": copy.deepcopy(self.model),
            "dataset": copy.deepcopy(self.dataset),
            "forecast": {
                "context_length": self.forecast.context_length,
                "horizon": self.forecast.horizon,
                "point_statistic": self.forecast.point_statistic,
                "origin": origin if isinstance(origin, str) else origin.isoformat(),
            },
            "runtime": copy.deepcopy(self.runtime),
            "output": {"directory": str(self.output_directory)},
            **(
                {"workflow": normalize_workflow(self.workflow)}
                if self.workflow is not None
                else {}
            ),
        }


def _validate_future(
    raw, source_path: Path, source_yaml: str, *, _path_resolver=_local_path
) -> FutureForecastConfig:
    required = {"schema_version", "mode", "model", "dataset", "forecast"}
    fields = _mapping(
        raw,
        "future configuration",
        required | {"name", "runtime", "output", "workflow"},
        required,
    )
    if (
        _integer(fields["schema_version"], "schema_version") != 2
        or fields["mode"] != "future_forecast"
    ):
        raise ConfigError(
            "Future configuration requires schema_version 2 and mode future_forecast"
        )
    model = _model(fields["model"], source_path.parent, _path_resolver=_path_resolver)
    values = _mapping(
        fields["forecast"],
        "forecast",
        {"context_length", "horizon", "point_statistic", "origin"},
        {"context_length", "horizon"},
    )
    for key in ("context_length", "horizon"):
        _integer(values[key], f"forecast.{key}")
    for key in ("point_statistic", "origin"):
        if key in values:
            _text(values[key], f"forecast.{key}")
    settings = model_settings(
        model["adapter_id"],
        repository=model.get("repository"),
        revision=model.get("revision"),
    )
    forecast = FutureForecastSpec(
        **{"point_statistic": settings.supported_tasks[0], **values}
    )
    dataset = _dataset(
        fields["dataset"], source_path.parent, _path_resolver=_path_resolver
    )
    validate_settings(model, forecast, dataset["frequency"])
    workflow = normalize_workflow(fields.get("workflow"))
    validate_workflow_units(workflow, dataset["units"])
    output = _mapping(fields.get("output", {}), "output", {"directory"})
    return FutureForecastConfig(
        name=_text(fields.get("name", source_path.stem), "name"),
        dataset=dataset,
        model=model,
        forecast=forecast,
        runtime=_runtime(fields.get("runtime", {}), model),
        output_directory=_path_resolver(
            output.get("directory", "./results"), "output.directory", source_path.parent
        ),
        source_yaml=source_yaml,
        source_path=source_path,
        workflow=workflow,
        _record_only=_path_resolver is not _local_path,
    )


def load_future_config(path: str | Path) -> FutureForecastConfig:
    """Read strict schema 2 without reading datasets, models, or preferences."""
    try:
        source_path = expand_local_user(path).resolve()
        with source_path.open("rb") as source:
            raw = source.read(MAX_CONFIG_BYTES + 1)
        if len(raw) > MAX_CONFIG_BYTES:
            raise ConfigError("Configuration exceeds the 1 MiB limit")
        text = raw.decode("utf-8")
        return _validate_future(
            yaml.load(text, Loader=_ConfigLoader), source_path, text
        )
    except ConfigError:
        raise
    except (OSError, ValueError, TypeError, RuntimeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Invalid future configuration: {exc}") from exc


def future_config_from_mapping(
    mapping: dict, *, base_directory: Path, _path_resolver=_local_path
) -> FutureForecastConfig:
    """Validate future Python/manual choices through the same strict YAML boundary."""
    try:
        directory = expand_local_user(base_directory).resolve()
        text = yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True)
        if len(text.encode("utf-8")) > MAX_CONFIG_BYTES:
            raise ConfigError("Configuration exceeds the 1 MiB limit")
        return _validate_future(
            yaml.load(text, Loader=_ConfigLoader),
            directory / "interactive.yaml",
            text,
            _path_resolver=_path_resolver,
        )
    except ConfigError:
        raise
    except (OSError, ValueError, TypeError, RuntimeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Invalid future interactive configuration: {exc}") from exc
