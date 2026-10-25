"""Strict, local-only YAML configuration for synchronous baseline experiments."""

import copy
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from tsfm_lab.contracts import EvaluationSpec, fixed_interval

MAX_CONFIG_BYTES = 1024 * 1024


class ConfigError(ValueError):
    """The configuration cannot be parsed or violates the supported schema."""


class _ConfigLoader(yaml.SafeLoader):
    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise ConfigError("YAML aliases are not supported")
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):
        if not isinstance(node, yaml.MappingNode):
            raise ConfigError("Expected a YAML mapping")
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigError("Configuration mapping keys must be strings")
            if key in result:
                raise ConfigError(f"Duplicate configuration key: {key}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


# Keep unquoted ISO timestamps as strings without changing global SafeLoader.
_ConfigLoader.yaml_implicit_resolvers = {
    key: [
        (tag, pattern)
        for tag, pattern in resolvers
        if tag != "tag:yaml.org,2002:timestamp"
    ]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    dataset: dict[str, Any]
    model: dict[str, Any]
    evaluation: EvaluationSpec
    runtime: dict[str, Any]
    output_directory: Path
    source_yaml: str
    source_path: Path

    def resolved_dict(self) -> dict[str, Any]:
        evaluation = asdict(self.evaluation)
        for name in ("evaluation_start", "evaluation_end"):
            evaluation[name] = evaluation[name].isoformat()
        evaluation["metrics"] = list(evaluation["metrics"])
        return {
            "schema_version": 1,
            "name": self.name,
            "dataset": copy.deepcopy(self.dataset),
            "model": copy.deepcopy(self.model),
            "evaluation": evaluation,
            "runtime": copy.deepcopy(self.runtime),
            "output": {"directory": str(self.output_directory)},
        }


def _mapping(value: Any, label: str, allowed: set[str], required=()) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a mapping")
    unknown = value.keys() - allowed
    missing = set(required) - value.keys()
    if unknown:
        raise ConfigError(f"Unknown {label} fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ConfigError(f"Missing {label} fields: {', '.join(sorted(missing))}")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a nonempty string")
    return value


def _integer(value: Any, label: str, *, minimum=1) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{label} must be an integer >= {minimum}")
    return value


def _local_path(value: Any, label: str, directory: Path) -> Path:
    text = _text(value, label)
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", text) or text.startswith("//"):
        raise ConfigError(f"{label} must be a local path, not a URL")
    try:
        path = Path(text).expanduser()
        return (
            (directory / path).resolve() if not path.is_absolute() else path.resolve()
        )
    except (OSError, RuntimeError, OverflowError, ValueError) as exc:
        raise ConfigError(f"Invalid {label}: {exc}") from exc


def _dataset(value: Any, directory: Path) -> dict[str, Any]:
    required = {"path", "timestamp_column", "target_column", "frequency"}
    allowed = required | {"timezone", "units", "series_id", "format"}
    fields = _mapping(value, "dataset", allowed, required)
    result = {key: _text(item, f"dataset.{key}") for key, item in fields.items()}
    path = _local_path(result["path"], "dataset.path", directory)
    formats = {".csv": "csv", ".parquet": "parquet", ".pq": "parquet"}
    inferred = formats.get(path.suffix.lower())
    if inferred is None:
        raise ConfigError("dataset.path must end with .csv, .parquet, or .pq")
    if result.get("format", inferred) != inferred:
        raise ConfigError("dataset.format must match the dataset file extension")
    if result["timestamp_column"] == result["target_column"]:
        raise ConfigError("dataset timestamp and target columns must be different")
    fixed_interval(result["frequency"])
    result.update(path=str(path), format=inferred)
    result.setdefault("units", "unspecified")
    result.setdefault("series_id", "series_0")
    return result


def _model(value: Any) -> dict[str, Any]:
    fields = _mapping(
        value, "model", {"source", "adapter_id", "options"}, {"source", "adapter_id"}
    )
    if fields["source"] != "builtin":
        raise ConfigError("model.source currently supports only builtin")
    adapter = _text(fields["adapter_id"], "model.adapter_id")
    if adapter not in {"last_value", "seasonal_naive"}:
        raise ConfigError("model.adapter_id must be last_value or seasonal_naive")
    allowed = {"seasonal_period"} if adapter == "seasonal_naive" else set()
    options = _mapping(fields.get("options", {}), "model.options", allowed, allowed)
    if allowed:
        options = {
            "seasonal_period": _integer(
                options["seasonal_period"], "model.options.seasonal_period"
            )
        }
    return {"source": "builtin", "adapter_id": adapter, "options": dict(options)}


def _evaluation(value: Any) -> EvaluationSpec:
    required = {"context_length", "horizon", "evaluation_start", "evaluation_end"}
    allowed = required | {
        "stride",
        "seasonal_period",
        "metrics",
        "point_statistic",
        "aggregation",
    }
    fields = dict(_mapping(value, "evaluation", allowed, required))
    for key in ("context_length", "horizon", "stride", "seasonal_period"):
        if key in fields:
            _integer(fields[key], f"evaluation.{key}")
    for key in ("evaluation_start", "evaluation_end", "point_statistic", "aggregation"):
        if key in fields:
            _text(fields[key], f"evaluation.{key}")
    if "metrics" in fields:
        if not isinstance(fields["metrics"], list):
            raise ConfigError("evaluation.metrics must be a list of names")
        for name in fields["metrics"]:
            _text(name, "evaluation.metrics entry")
    return EvaluationSpec(**fields)


def _runtime(value: Any) -> dict[str, Any]:
    fields = _mapping(value, "runtime", {"device", "dtype", "batch_size", "seed"})
    result = {"device": "cpu", "dtype": "float64", "batch_size": 1, "seed": 0}
    result.update(fields)
    _text(result["device"], "runtime.device")
    _text(result["dtype"], "runtime.dtype")
    _integer(result["batch_size"], "runtime.batch_size")
    _integer(result["seed"], "runtime.seed", minimum=0)
    if result["device"] != "cpu" or result["dtype"] != "float64":
        raise ConfigError("Baseline runtime requires device cpu and dtype float64")
    if result["batch_size"] != 1:
        raise ConfigError("The synchronous runner currently supports batch_size 1")
    return result


def _validate(raw: Any, source_path: Path, source_yaml: str) -> ExperimentConfig:
    required = {"schema_version", "model", "dataset", "evaluation"}
    allowed = required | {"name", "runtime", "output"}
    fields = _mapping(raw, "configuration", allowed, required)
    if _integer(fields["schema_version"], "schema_version") != 1:
        raise ConfigError("Only schema_version 1 is supported")
    model = _model(fields["model"])
    evaluation = _evaluation(fields["evaluation"])
    if model["adapter_id"] == "seasonal_naive":
        if evaluation.context_length < model["options"]["seasonal_period"]:
            raise ConfigError("context_length must contain a complete baseline season")
    output = _mapping(fields.get("output", {}), "output", {"directory"})
    return ExperimentConfig(
        name=_text(fields.get("name", source_path.stem), "name"),
        dataset=_dataset(fields["dataset"], source_path.parent),
        model=model,
        evaluation=evaluation,
        runtime=_runtime(fields.get("runtime", {})),
        output_directory=_local_path(
            output.get("directory", "./results"), "output.directory", source_path.parent
        ),
        source_yaml=source_yaml,
        source_path=source_path,
    )


def load_config(path: str | Path) -> ExperimentConfig:
    """Read and validate YAML without reading datasets or creating output files."""
    source_name = str(path)
    try:
        source_path = Path(path).expanduser().resolve()
        source_name = source_path.name
        with source_path.open("rb") as stream:
            contents = stream.read(MAX_CONFIG_BYTES + 1)
        if len(contents) > MAX_CONFIG_BYTES:
            raise ConfigError("Configuration exceeds the 1 MiB limit")
        source_yaml = contents.decode("utf-8")
        raw = yaml.load(source_yaml, Loader=_ConfigLoader)
        return _validate(raw, source_path, source_yaml)
    except ConfigError:
        raise
    except (
        OSError,
        UnicodeError,
        yaml.YAMLError,
        ValueError,
        TypeError,
        RuntimeError,
        OverflowError,
    ) as exc:
        raise ConfigError(f"Invalid configuration {source_name}: {exc}") from exc
