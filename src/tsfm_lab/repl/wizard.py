"""Inline experiment questions over the shared configuration and evaluation rules."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from tsfm_lab.config import ConfigError, ExperimentConfig
from tsfm_lab.contracts import fixed_interval, utc_timestamp
from tsfm_lab.evaluation.windows import evaluation_bounds
from tsfm_lab.models.catalog import (
    CHECKPOINT_REPOSITORIES,
    catalog_entries,
    catalog_info,
    checkpoint_info,
)
from tsfm_lab.models.factory import build_model
from tsfm_lab.models.verification import checkpoint_verification, verification_label
from tsfm_lab.point_tasks import validate_model_frequency
from tsfm_lab.tui.forms import (
    config_from_fields,
    default_fields,
    demo_fields,
    inspect_data,
    prepare_experiment,
    suggested_bounds,
    validate_model_fields,
)

_ADAPTERS = tuple(entry["adapter_id"] for entry in catalog_entries())


class _Cancel(Exception):
    pass


class _Back(Exception):
    pass


def _required(value: str) -> None:
    if not value.strip():
        raise ConfigError("Enter a value, or /cancel to stop setup")


def _positive(value: str) -> int:
    if re.fullmatch(r"[0-9]+", value) is None:
        raise ConfigError("Enter a positive whole number")
    try:
        result = int(value)
    except ValueError as exc:
        raise ConfigError("That number is too large") from exc
    if result <= 0:
        raise ConfigError("Enter a number greater than zero")
    return result


def _question(ask, emit, label, default="", choices=(), validate=None) -> str:
    while True:
        answer = ask(label, default=default, choices=tuple(choices))
        if not isinstance(answer, str):
            emit("Enter text, or /cancel to stop setup.")
            continue
        answer = answer.strip()
        if answer == "/cancel":
            raise _Cancel
        if answer == "/back":
            raise _Back
        answer = answer or default
        aliases = {"y": "yes", "n": "no"}
        if choices:
            canonical_choices = {choice.casefold(): choice for choice in choices}
            canonical = aliases.get(answer.casefold(), answer.casefold())
            if canonical not in canonical_choices:
                emit(f"Choose {', '.join(choices)}, or /cancel.")
                continue
            answer = canonical_choices[canonical]
        try:
            if validate:
                validate(answer)
        except (ConfigError, ValueError, OSError, RuntimeError, OverflowError) as exc:
            emit(str(exc))
            continue
        return answer


class _Guide:
    def __init__(self, ask, emit, workspace, demo_path, adapter, ttm_demo_path=None):
        self.ask, self.emit = ask, emit
        self.workspace, self.demo_path = Path(workspace), Path(demo_path).resolve()
        self.arguments = {"workspace": self.workspace, "base_directory": Path.cwd()}
        self.fields = default_fields(adapter or "last_value")
        self.fields["dataset_path"] = str(self.demo_path)
        self._ordinary_demo_path = self.demo_path
        self._ttm_demo_path = Path(ttm_demo_path).resolve() if ttm_demo_path else None
        self._checkpoint_profile = None
        self.dataset = self.capabilities = None
        self._model = None
        self._evaluation_started = False

    def set(self, key, label, *, choices=(), validate=None, default=None):
        self.fields[key] = _question(
            self.ask,
            self.emit,
            label,
            self.fields[key] if default is None else default,
            choices,
            validate,
        )

    def model(self):
        previous_adapter = self.fields["adapter"]
        self.set("adapter", "Model adapter", choices=_ADAPTERS)
        information = catalog_info(self.fields["adapter"])
        if self.fields["adapter"] != previous_adapter:
            statistic = information["point_statistic"]
            self.fields["point_statistic"] = (
                statistic if statistic in information["supported_tasks"] else "median"
            )
        self.set(
            "point_statistic", "Forecast task", choices=information["supported_tasks"]
        )
        for notice in information["runtime_notices"]:
            self.emit(notice)
        if self.fields["adapter"] == "seasonal_naive":
            self.set("seasonal_period", "Baseline seasonal period", validate=_positive)
        elif self.fields["adapter"] in CHECKPOINT_REPOSITORIES:
            repositories = CHECKPOINT_REPOSITORIES[self.fields["adapter"]]
            if self.fields["repository"] not in repositories:
                self.fields.update(repository=repositories[0], revision="")
            self.emit(
                "Use an existing cached or local checkpoint; setup does not download models."
            )
            source = _question(
                self.ask,
                self.emit,
                "Checkpoint source",
                "cached" if self.fields["source"] == "huggingface" else "local",
                ("cached", "local"),
            )
            self.fields["source"] = "huggingface" if source == "cached" else "local"
            if source == "cached":
                self.emit("Curated checkpoint variants:")
                for variant in information["variants"]:
                    self.emit(f"  {variant['variant']} · {variant['repository']}")
                    context = (
                        f"exactly {variant['min_context']}"
                        if variant["min_context"] == variant["max_context"]
                        else f"up to {variant['max_context']}"
                    )
                    self.emit(
                        f"    Context {context} · Horizon up to {variant['max_horizon']} observations"
                    )
                    self.emit(f"    {variant['verification_label']}")
                previous_repository = self.fields["repository"]
                self.set("repository", "Checkpoint repository", choices=repositories)
                variants = [
                    item
                    for item in information["variants"]
                    if item["repository"] == self.fields["repository"]
                ]
                selected = variants[0]
                if len(variants) > 1:
                    previous_profile = self._checkpoint_profile
                    default_profile = next(
                        (
                            item["variant"]
                            for item in variants
                            if item["reference_revision"] == self.fields["revision"]
                        ),
                        None,
                    )
                    if default_profile is None and previous_profile in {
                        item["variant"] for item in variants
                    }:
                        default_profile = previous_profile
                    self._checkpoint_profile = _question(
                        self.ask,
                        self.emit,
                        "Checkpoint profile",
                        default_profile or variants[0]["variant"],
                        tuple(item["variant"] for item in variants),
                    )
                    selected = checkpoint_info(
                        self.fields["repository"], self._checkpoint_profile
                    )
                    if self._checkpoint_profile != previous_profile:
                        self.fields["revision"] = selected["reference_revision"]
                if (
                    self.fields["repository"] != previous_repository
                    or not self.fields["revision"]
                ):
                    self.fields["revision"] = selected["reference_revision"]
                self.set("revision", "Exact checkpoint SHA", validate=self._sha)
                self.set("cache_dir", "Cache directory (optional; - uses the default)")
                if self.fields["cache_dir"] == "-":
                    self.fields["cache_dir"] = ""
            else:
                self.set(
                    "checkpoint_path", "Local checkpoint directory", validate=_required
                )
        validate_model_fields(self.fields, **self.arguments)
        prototype = {
            **self.fields,
            "dataset_path": str(self.demo_path),
            "context_length": (
                self.fields["seasonal_period"]
                if self.fields["adapter"] == "seasonal_naive"
                else "1"
            ),
            "horizon": "1",
            "evaluation_start": "2000-01-01T01:00:00Z",
            "evaluation_end": "2000-01-01T02:00:00Z",
        }
        config = config_from_fields(prototype, **self.arguments)
        model = build_model(config.model, config.runtime)
        self._model = model
        self.capabilities = model.metadata().capabilities
        next_demo = (
            self._ttm_demo_path
            if self.fields["adapter"] == "ttm_r2"
            else self._ordinary_demo_path
        )
        if next_demo is not None:
            if (
                Path(self.fields["dataset_path"]).expanduser().resolve()
                == self.demo_path
            ):
                self.fields["dataset_path"] = str(next_demo)
            self.demo_path = next_demo
        if self.capabilities.min_context == self.capabilities.max_context:
            self.fields["context_length"] = str(self.capabilities.min_context)
        record = getattr(model, "configuration", lambda: {})().get("checkpoint")
        if record:
            snapshot = checkpoint_verification(record)
            self.emit(f"Selected checkpoint: {verification_label(snapshot)}")
            for check in snapshot["adapter_representative_checks"]:
                self.emit(
                    f"  Adapter reference: {check['repository']} @ {check['revision'][:12]} · {check['device']}/{check['dtype']}"
                )

    @staticmethod
    def _sha(value):
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ConfigError("Enter an exact 40-character lowercase checkpoint SHA")

    def _data_path(self, value):
        _required(value)
        if "://" in value or value.startswith("//"):
            raise ConfigError("Choose a local CSV or Parquet file")
        path = Path(value).expanduser()
        if path.suffix.lower() not in {".csv", ".parquet", ".pq"} or not path.is_file():
            raise ConfigError("Choose an existing local CSV or Parquet file")

    def data(self):
        self.set("dataset_path", "Dataset path", validate=self._data_path)
        if Path(self.fields["dataset_path"]).expanduser().resolve() == self.demo_path:
            selected = demo_fields(self.demo_path)
            for key in (
                "timestamp_column",
                "target_column",
                "frequency",
                "timezone",
                "units",
            ):
                self.fields[key] = selected[key]
        else:
            for key, label, validator in (
                ("timestamp_column", "Timestamp column", _required),
                ("target_column", "Target column", _required),
                ("frequency", "Observation frequency", fixed_interval),
                (
                    "timezone",
                    "Source timezone (optional; - leaves it unspecified)",
                    None,
                ),
                ("units", "Target units", _required),
            ):
                self.set(key, label, validate=validator)
            if self.fields["timezone"] == "-":
                self.fields["timezone"] = ""
        self.dataset = inspect_data(self.fields, **self.arguments)
        validate_model_frequency(self._model, self.dataset.frequency)
        self.emit(
            f"Validated {len(self.dataset.values)} observations of {self.dataset.target_name} ({self.dataset.units}), frequency {self.dataset.frequency}."
        )

    def _limits(self):
        size, caps = len(self.dataset.values), self.capabilities
        maximum = min(size - 1, caps.max_context or size - 1)
        if maximum < caps.min_context:
            raise ConfigError(
                "The dataset is too short for this model's minimum context"
            )
        return caps.min_context, maximum

    def evaluation(self):
        low, high = self._limits()
        if not self._evaluation_started:
            context = max(low, min(8, len(self.dataset.values) // 2, high))
            horizon = min(
                4,
                len(self.dataset.values) - context,
                self.capabilities.max_horizon or 4,
            )
            self.fields.update(
                context_length=str(context),
                horizon=str(horizon),
                stride=str(horizon),
                mase_period=str(min(4, context)),
            )
            self._evaluation_started = True
        self.set("name", "Experiment name", validate=_required)

        def context_limit(value):
            if not low <= _positive(value) <= high:
                raise ConfigError(
                    f"Context length must be between {low} and {high} observations"
                )

        self.set("context_length", "Context length", validate=context_limit)
        maximum_horizon = min(
            len(self.dataset.values) - int(self.fields["context_length"]),
            self.capabilities.max_horizon or len(self.dataset.values),
        )

        def horizon_limit(value):
            if _positive(value) > maximum_horizon:
                raise ConfigError(
                    f"Forecast horizon must be between 1 and {maximum_horizon} observations"
                )

        self.set("horizon", "Forecast horizon", validate=horizon_limit)
        self.set("stride", "Evaluation stride", validate=_positive)
        self.set("mase_period", "MASE seasonal lag", validate=_positive)
        proposals = suggested_bounds(self.dataset, self.fields)
        for key, value in proposals.items():
            if not self.fields[key]:
                self.fields[key] = value
        self.set(
            "evaluation_start",
            "First target timestamp (inclusive)",
            validate=self._start,
        )
        self.set(
            "evaluation_end", "Target end timestamp (exclusive)", validate=self._end
        )

    def _start(self, value):
        timestamp = utc_timestamp(value, "First target timestamp")
        position = int(self.dataset.timestamps.get_indexer([timestamp])[0])
        if position < int(self.fields["context_length"]):
            raise ConfigError(
                "The first target must be on the dataset grid with sufficient preceding context"
            )
        if position + int(self.fields["horizon"]) > len(self.dataset.values):
            raise ConfigError("The first target leaves no complete forecast horizon")

    def _end(self, value):
        candidate = {**self.fields, "evaluation_end": value}
        config = config_from_fields(candidate, **self.arguments)
        evaluation_bounds(self.dataset, config.evaluation)

    def run(self):
        phase = 0
        self.emit("New experiment. Enter /back to revisit a step or /cancel to stop.")
        while True:
            try:
                if phase == 0:
                    self.model()
                elif phase == 1:
                    self.data()
                elif phase == 2:
                    self.evaluation()
                else:
                    prepared = prepare_experiment(self.fields, **self.arguments)
                    self.emit(prepared.summary())
                    answer = _question(
                        self.ask,
                        self.emit,
                        "Run experiment?",
                        "no",
                        ("yes", "no", "back"),
                    )
                    if answer == "yes":
                        return prepared.config
                    if answer == "no":
                        return None
                    phase = 2
                    continue
                phase += 1
            except _Back:
                phase = max(0, phase - 1)
            except (ConfigError, ValueError, OSError, ImportError) as exc:
                self.emit(str(exc))
                try:
                    answer = _question(
                        self.ask,
                        self.emit,
                        "Setup needs attention: retry, back, or cancel?",
                        "cancel",
                        ("retry", "back", "cancel"),
                    )
                except _Back:
                    answer = "back"
                if answer == "cancel":
                    return None
                phase = max(0, phase - 1) if answer == "back" else min(phase, 2)


def build_experiment(
    ask: Callable,
    emit: Callable[[str], None],
    *,
    workspace: Path,
    demo_path: Path,
    adapter: str | None = None,
    demo: bool = False,
    ttm_demo_path: Path | None = None,
) -> ExperimentConfig | None:
    """Ask inline questions and return a reviewed config without execution or writes."""
    try:
        if not demo:
            return _Guide(ask, emit, workspace, demo_path, adapter, ttm_demo_path).run()
        if adapter in CHECKPOINT_REPOSITORIES:
            raise ConfigError("Use guided setup to select a pretrained checkpoint")
        fields = demo_fields(demo_path)
        fields["adapter"] = adapter or "last_value"
        arguments = {"workspace": workspace, "base_directory": Path.cwd()}
        dataset = inspect_data(fields, **arguments)
        fields.update(suggested_bounds(dataset, fields))
        prepared = prepare_experiment(fields, **arguments)
        emit(prepared.summary())
        answer = _question(ask, emit, "Run experiment?", "no", ("yes", "no"))
        return prepared.config if answer == "yes" else None
    except (_Cancel, _Back):
        emit("Experiment setup cancelled.")
        return None
