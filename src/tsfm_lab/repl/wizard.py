"""Inline experiment questions over the shared configuration and evaluation rules."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from tsfm_lab.config import ConfigError, ExperimentConfig
from tsfm_lab.contracts import fixed_interval, utc_timestamp
from tsfm_lab.evaluation.windows import evaluation_bounds
from tsfm_lab.models.catalog import (
    CHECKPOINT_REPOSITORIES,
    catalog_info,
)
from tsfm_lab.models.catalog import (
    checkpoint_info as checkpoint_info,
)
from tsfm_lab.models.factory import build_model
from tsfm_lab.models.settings import model_settings
from tsfm_lab.models.verification import checkpoint_verification, verification_label
from tsfm_lab.point_tasks import validate_model_frequency
from tsfm_lab.tui.forms import (
    config_from_fields,
    default_fields,
    demo_fields,
    inspect_data,
    prepare_experiment,
    profile_example_fields,
    profile_guidance,
    suggested_bounds,
    validate_model_fields,
)
from tsfm_lab.workflows import PROFILE_NAMES

if TYPE_CHECKING:
    from tsfm_lab.guided import GuidedRunPlan


class _Cancel(Exception):
    pass


class _Back(Exception):
    pass


def _required(value: str) -> None:
    if not value.strip():
        raise ConfigError("Enter a value, or /cancel to stop setup")


def _declared_units(value: str) -> None:
    _required(value)
    if value.strip().casefold() == "unspecified":
        raise ConfigError("Declare the target's units for this usage profile")


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
    def __init__(
        self,
        ask,
        emit,
        workspace,
        demo_path,
        adapter,
        ttm_demo_path=None,
        profile_demo_paths=None,
        seed_fields=None,
        explicit_fields=(),
        checkpoint_variant=None,
        preferences=None,
        guided_preparation=False,
    ):
        self.ask, self.emit = ask, emit
        self.guided_preparation = guided_preparation
        self._preparation_offline = False
        self.workspace, self.demo_path = Path(workspace), Path(demo_path).resolve()
        self.arguments = {"workspace": self.workspace, "base_directory": Path.cwd()}
        self.fields = default_fields(adapter or "last_value")
        if preferences is not None:
            self.arguments["base_directory"] = preferences.workspace
            self.fields.update(
                usage_profile=preferences.category,
                cache_dir=str(preferences.cache_directory),
            )
        self.fields["dataset_path"] = str(self.demo_path)
        supplied = dict(seed_fields or {})
        variant = supplied.pop("checkpoint_variant", None)
        if set(supplied) - set(self.fields) or any(
            not isinstance(value, str) for value in supplied.values()
        ):
            raise ConfigError("The assisted setup contains invalid form fields")
        self._provided = set(supplied) | set(explicit_fields)
        self._assisted = seed_fields is not None
        self.fields.update(supplied)
        if self._assisted and "units" not in supplied:
            self.fields["units"] = ""
            self._provided.add("units")
        self._ordinary_demo_path = self.demo_path
        self._ttm_demo_path = Path(ttm_demo_path).resolve() if ttm_demo_path else None
        self._profile_demo_paths = {
            profile: Path(path).resolve()
            for profile, path in (profile_demo_paths or {}).items()
        }
        self._checkpoint_profile = checkpoint_variant or variant
        self.dataset = self.capabilities = None
        self._model = None
        self._options_started = False
        self._example_declarations = {}
        self._example_path = None

    def _default(self, key, value):
        if key not in self._provided:
            self.fields[key] = value

    def profile(self):
        previous = self.fields["usage_profile"]
        self.set(
            "usage_profile",
            "Usage profile",
            choices=PROFILE_NAMES,
            default=self.fields["usage_profile"] or "generic",
        )
        if self.fields["usage_profile"] != previous:
            self._invalidate_data()
        self.emit(profile_guidance(self.fields["usage_profile"]))

    def _invalidate_data(self):
        self.dataset = None
        for key in ("evaluation_start", "evaluation_end"):
            if key not in self._provided:
                self.fields[key] = ""

    def set(self, key, label, *, choices=(), validate=None, default=None):
        previous = self.fields[key]
        self.fields[key] = _question(
            self.ask,
            self.emit,
            label,
            self.fields[key] if default is None else default,
            choices,
            validate,
        )
        if self.fields[key] != previous:
            self._provided.add(key)

    def model(self):
        from tsfm_lab.repl.model_selection import select_model

        previous = self._model_identity()
        previous_adapter = self.fields["adapter"]
        selected = select_model(
            self.ask,
            self.emit,
            self.fields,
            checkpoint_variant=self._checkpoint_profile,
            allow_preparation=self.guided_preparation,
            offline=self._preparation_offline,
        )
        self._preparation_offline = (
            selected.pop("preparation_offline", "false") == "true"
        )
        self._checkpoint_profile = selected.pop("checkpoint_variant", None)
        self.fields.update(selected)
        adapter = self.fields["adapter"]
        information = catalog_info(adapter)
        if adapter != previous_adapter and (
            "point_statistic" not in self._provided
            or self.fields["point_statistic"] not in information["supported_tasks"]
        ):
            statistic = information["point_statistic"]
            self.fields["point_statistic"] = (
                statistic if statistic in information["supported_tasks"] else "median"
            )
            self._provided.discard("point_statistic")
        changed = previous != self._model_identity()
        self.settings = model_settings(
            adapter,
            repository=(
                self.fields["repository"]
                if self.fields["source"] == "huggingface"
                else None
            ),
            revision=(
                self.fields["revision"]
                if self.fields["source"] == "huggingface"
                else None
            ),
        )
        if changed:
            self._model = None
            self._invalidate_data()
        self.capabilities = (
            self._model.metadata().capabilities
            if self._model
            else self.settings.capabilities
        )
        if changed or not self._options_started:
            self._default("context_length", str(self.settings.default_context))
            self._default("horizon", str(self.settings.default_horizon))
        self.emit(f"Preprocessing (fixed): {self.settings.preprocessing}")
        self.emit("Runtime: CPU · fixed numerical precision · batch size 1.")
        for notice in information["runtime_notices"]:
            self.emit(notice)
        next_demo = self._profile_demo_paths.get(self.fields["usage_profile"])
        if next_demo is None:
            next_demo = (
                self._ttm_demo_path if adapter == "ttm_r2" else self._ordinary_demo_path
            )
        if next_demo is not None:
            if self._resolved_data_path(self.fields["dataset_path"]) == self.demo_path:
                self._default("dataset_path", str(next_demo))
            self.demo_path = next_demo

    def _model_identity(self):
        keys = ("adapter", "source", "repository", "revision", "checkpoint_path")
        return tuple(self.fields[key] for key in keys)

    def model_options(self, *, apply_dataset_bounds=False):
        self.set(
            "point_statistic", "Forecast task", choices=self.settings.supported_tasks
        )
        if self.fields["adapter"] == "seasonal_naive":
            self.set("seasonal_period", "Baseline seasonal period", validate=_positive)
        caps = self.capabilities
        low, high = caps.min_context, caps.max_context
        if self.fields["adapter"] == "seasonal_naive":
            low = max(low, int(self.fields["seasonal_period"]))
        future = self.fields["mode"] == "future_forecast"
        if future:
            high = min(high or 100000, 100000)
        if apply_dataset_bounds and self.dataset is not None:
            available = len(self.dataset.values) - (0 if future else 1)
            high = min(high or available, available)
            if high < low:
                raise ConfigError(
                    "The dataset is too short for this model's minimum context"
                )
        context = (
            f"exactly {low}"
            if low == high
            else f"up to {high} (minimum {low})" if high else f"at least {low}"
        )
        horizon = (
            f"up to {caps.max_horizon}" if caps.max_horizon else "any positive length"
        )
        self.emit(f"Model options · Context {context} · Horizon {horizon} observations")

        def context_limit(value):
            number = _positive(value)
            if number < low or (high is not None and number > high):
                raise ConfigError(
                    f"Context length must be between {low} and {high} observations"
                )

        def horizon_limit(value):
            maximum = caps.max_horizon
            if future:
                maximum = min(maximum or 10000, 10000)
            elif apply_dataset_bounds and self.dataset is not None:
                remaining = len(self.dataset.values) - int(
                    self.fields["context_length"]
                )
                maximum = min(maximum or remaining, remaining)
            if maximum is not None and _positive(value) > maximum:
                raise ConfigError(
                    f"Forecast horizon must be between 1 and {maximum} observations"
                )
            _positive(value)

        self.set("context_length", "Context length", validate=context_limit)
        self.set("horizon", "Forecast horizon", validate=horizon_limit)
        validate_model_fields(
            {**self.fields, "name": "Model validation"}, **self.arguments
        )
        self._default("stride", self.fields["horizon"])
        self._options_started = True

    def _checkpoint_model(self):
        prototype = {
            **self.fields,
            "name": "Model validation",
            "stride": "1",
            "mase_period": "1",
            "evaluation_start": "2000-01-01T01:00:00Z",
            "evaluation_end": "2000-01-01T02:00:00Z",
        }
        config = config_from_fields(prototype, **self.arguments)
        future = self.fields["mode"] == "future_forecast"
        spec = config.forecast if future else config.evaluation
        if len(self.dataset.values) < spec.context_length + (
            0 if future else spec.horizon
        ):
            self.emit(
                "Adjust context to fit the observed history."
                if future
                else "Adjust context and horizon to fit the inspected dataset."
            )
            remaining = len(self.dataset.values) - spec.context_length
            if remaining > 0 and not future:
                self.emit(
                    f"Forecast horizon must be between 1 and {remaining} observations for this dataset."
                )
            self.model_options(apply_dataset_bounds=True)
            prototype.update(
                context_length=self.fields["context_length"],
                horizon=self.fields["horizon"],
            )
            config = config_from_fields(prototype, **self.arguments)
        if self.guided_preparation and config.model["source"] == "huggingface":
            return
        model = build_model(config.model, config.runtime)
        self._model = model
        self.capabilities = model.metadata().capabilities
        validate_model_frequency(model, self.dataset.frequency)
        caps = self.capabilities
        needs_correction = False
        for key, low, high in (
            ("context_length", caps.min_context, caps.max_context),
            ("horizon", 1, caps.max_horizon),
        ):
            value = int(self.fields[key])
            if value < low or (high is not None and value > high):
                if key in self._provided:
                    self.emit(
                        f"{key} conflicts with this checkpoint's limits; correct the model options."
                    )
                    needs_correction = True
                    continue
                default = (
                    max(low, min(value, high)) if high is not None else max(low, value)
                )
                self.fields[key] = str(default)
                self.emit(
                    f"Checkpoint metadata refines {key} to {default}; review before execution."
                )
        if needs_correction:
            self.model_options(apply_dataset_bounds=True)
        needed = int(self.fields["context_length"]) + (
            0 if future else int(self.fields["horizon"])
        )
        if len(self.dataset.values) < needed:
            raise ConfigError(
                "The dataset is too short for this checkpoint's context and horizon"
            )
        record = getattr(model, "configuration", lambda: {})().get("checkpoint")
        if record:
            snapshot = checkpoint_verification(record)
            self.emit(f"Selected checkpoint: {verification_label(snapshot)}")
            for check in snapshot["adapter_representative_checks"]:
                self.emit(
                    f"  Adapter reference: {check['repository']} @ {check['revision'][:12]} · {check['device']}/{check['dtype']}"
                )

    def _data_path(self, value):
        _required(value)
        if "://" in value or value.startswith("//"):
            raise ConfigError("Choose a local CSV or Parquet file")
        path = self._resolved_data_path(value)
        if path.suffix.lower() not in {".csv", ".parquet", ".pq"} or not path.is_file():
            raise ConfigError("Choose an existing local CSV or Parquet file")

    def _resolved_data_path(self, value):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.arguments["base_directory"] / path
        return path.resolve()

    def data(self):
        self.set("dataset_path", "Dataset path", validate=self._data_path)
        if self._resolved_data_path(self.fields["dataset_path"]) == self.demo_path:
            mapped = self.fields["usage_profile"] in self._profile_demo_paths
            selected = (
                profile_example_fields(self.fields["usage_profile"], self.demo_path)
                if mapped
                else demo_fields(self.demo_path)
            )
            preserve_units = (
                self._example_path == self.demo_path
                and self._example_declarations.get("units")
                and self.fields["units"] != self._example_declarations["units"]
            )
            for key in (
                "timestamp_column",
                "target_column",
                "frequency",
                "timezone",
                "units",
            ):
                if key != "units" or not preserve_units:
                    self._default(key, selected[key])
            for key in (
                "finance_target_kind",
                "energy_measurement",
                "weather_variable",
                "weather_station",
                "weather_measurement_semantics",
            ):
                if not self.fields[key] or (
                    mapped and self._example_path != self.demo_path
                ):
                    self._default(key, selected[key])
            if mapped:
                self._example_path = self.demo_path
                self._example_declarations = {
                    key: selected[key]
                    for key in (
                        "units",
                        "finance_target_kind",
                        "energy_measurement",
                        "weather_variable",
                        "weather_station",
                        "weather_measurement_semantics",
                    )
                    if selected[key]
                }
        else:
            for key, value in self._example_declarations.items():
                if key not in self._provided and self.fields[key] == value:
                    self.fields[key] = "unspecified" if key == "units" else ""
            self._example_declarations = {}
            self._example_path = None
        if self._resolved_data_path(
            self.fields["dataset_path"]
        ) != self.demo_path or self._provided & {
            "timestamp_column",
            "target_column",
            "frequency",
            "timezone",
        }:
            for key, label, validator in (
                ("timestamp_column", "Timestamp column", _required),
                ("target_column", "Target column", _required),
                ("frequency", "Observation frequency", fixed_interval),
                (
                    "timezone",
                    "Source timezone (optional; - leaves it unspecified)",
                    None,
                ),
            ):
                self.set(key, label, validate=validator)
            if self.fields["timezone"] == "-":
                self.fields["timezone"] = ""
        self.measurement()
        if (
            self.fields["usage_profile"] != "generic"
            or self._resolved_data_path(self.fields["dataset_path"]) != self.demo_path
            or "units" in self._provided
            or self._assisted
        ):
            label = {
                "finance": "Financial target units",
                "energy": "Power or energy units",
                "temperature": "Temperature units",
                "weather": "Weather measurement units",
            }.get(self.fields["usage_profile"], "Target units")
            self.set(
                "units",
                label,
                validate=(
                    _required
                    if self.fields["usage_profile"] == "generic" and not self._assisted
                    else _declared_units
                ),
            )
        previous = self.dataset
        self.dataset = inspect_data(self.fields, **self.arguments)
        if previous is not None and previous.provenance != self.dataset.provenance:
            for key in ("evaluation_start", "evaluation_end"):
                if key not in self._provided:
                    self.fields[key] = ""
        self.emit(
            profile_guidance(self.fields["usage_profile"], self.dataset.frequency)
        )
        self.emit(
            f"Validated {len(self.dataset.values)} observations of {self.dataset.target_name} ({self.dataset.units}), frequency {self.dataset.frequency}."
        )

        self.emit("Dataset preview · original units · UTC timestamps")
        positions = list(range(min(3, len(self.dataset.values))))
        positions += list(
            range(max(3, len(self.dataset.values) - 3), len(self.dataset.values))
        )
        for position in positions:
            self.emit(
                f"  {self.dataset.timestamps[position].isoformat()}  {self.dataset.values[position]:g} {self.dataset.units}"
            )

    def measurement(self):
        profile = self.fields["usage_profile"]
        if profile == "finance":
            self.set(
                "finance_target_kind",
                "Finance target kind",
                choices=("close_price", "return", "other"),
            )
            if self.fields["finance_target_kind"] == "other":
                self.set(
                    "finance_measurement", "Finance measurement", validate=_required
                )
        elif profile == "energy":
            self.set(
                "energy_measurement",
                "Energy measurement",
                choices=("power", "energy", "other"),
            )
            if self.fields["energy_measurement"] == "other":
                self.set("energy_quantity", "Energy quantity", validate=_required)
        elif profile == "weather":
            self.set("weather_variable", "Weather variable", validate=_required)
            self.set("weather_station", "Weather station", validate=_required)
            self.set(
                "weather_measurement_semantics",
                "Weather measurement semantics",
                choices=(
                    "instantaneous",
                    "interval_mean",
                    "interval_total",
                    "interval_max",
                    "other",
                ),
            )
            if self.fields["weather_measurement_semantics"] == "other":
                self.set(
                    "weather_description",
                    "Weather semantics description",
                    validate=_required,
                )

    def _limits(self):
        size, caps = len(self.dataset.values), self.capabilities
        available = size - (0 if self.fields["mode"] == "future_forecast" else 1)
        maximum = min(available, caps.max_context or available)
        if maximum < caps.min_context:
            raise ConfigError(
                "The dataset is too short for this model's minimum context"
            )
        return caps.min_context, maximum

    def evaluation(self):
        labels = {
            "historical_evaluation": "Historical evaluation",
            "future_forecast": "Future forecasting",
        }
        previous = self.fields["mode"]
        selected = _question(
            self.ask,
            self.emit,
            "Forecast mode",
            labels.get(previous, previous),
            tuple(labels.values()),
        )
        self.fields["mode"] = next(
            mode for mode, label in labels.items() if label == selected
        )
        if previous != self.fields["mode"]:
            for key in (
                "evaluation_start",
                "evaluation_end",
                "stride",
                "mase_period",
                "forecast_origin",
            ):
                self.fields[key] = ""
                self._provided.discard(key)
            self.fields["forecast_origin"] = (
                "latest" if self.fields["mode"] == "future_forecast" else ""
            )
        self._checkpoint_model()
        self._limits()
        self.emit(f"Mode: {selected}.")
        self.set("name", "Experiment name", validate=_required)
        if self.fields["mode"] == "future_forecast":
            self.set(
                "forecast_origin",
                "Forecast origin (latest or final observed timestamp)",
                default=self.fields["forecast_origin"] or "latest",
                validate=self._future_origin,
            )
            self.emit(
                "Accuracy unavailable: future_actuals_absent. No historical scores or returns are substituted."
            )
            return
        self.set(
            "stride",
            "Evaluation stride",
            default=self.fields["stride"] or self.fields["horizon"],
            validate=_positive,
        )
        self.set(
            "mase_period",
            "MASE seasonal lag",
            default=self.fields["mase_period"] or "4",
            validate=_positive,
        )
        proposals = suggested_bounds(self.dataset, self.fields)
        for key, value in proposals.items():
            if not self.fields[key]:
                self._default(key, value)
        self.set(
            "evaluation_start",
            "First target timestamp (inclusive)",
            validate=self._start,
        )
        self.set(
            "evaluation_end", "Target end timestamp (exclusive)", validate=self._end
        )

    def _future_origin(self, value):
        if (
            value != "latest"
            and utc_timestamp(value, "Forecast origin") != self.dataset.timestamps[-1]
        ):
            raise ConfigError("Forecast origin must equal the final observed timestamp")

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
                    self.profile()
                elif phase == 1:
                    self.model()
                elif phase == 2:
                    self.model_options()
                elif phase == 3:
                    self.data()
                elif phase == 4:
                    self.evaluation()
                else:
                    options = (
                        {"resolve_model": False} if self.guided_preparation else {}
                    )
                    prepared = prepare_experiment(
                        self.fields, **self.arguments, **options
                    )
                    self.emit(prepared.summary())
                    plan = None
                    if self.guided_preparation:
                        from tsfm_lab.guided import review_guided

                        plan = review_guided(
                            prepared.config, offline=self._preparation_offline
                        )
                        self._preparation_summary(plan)
                    needs_download = (
                        plan is not None
                        and plan.checkpoint is not None
                        and not plan.checkpoint.cached
                    )
                    answer = _question(
                        self.ask,
                        self.emit,
                        (
                            "Download missing checkpoint files and run experiment?"
                            if needs_download
                            else "Run experiment?"
                        ),
                        "no",
                        ("yes", "no", "back"),
                    )
                    if answer == "yes":
                        return plan if plan is not None else prepared.config
                    if answer == "no":
                        return None
                    phase = 4
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
                phase = (
                    max(0, phase - 1)
                    if answer == "back"
                    else (2 if phase >= 4 else phase)
                )

    def _preparation_summary(self, plan):
        checkpoint = plan.checkpoint
        if checkpoint is None:
            self.emit("Ready for a local run. No checkpoint download is required.")
            return
        self.emit(
            f"Checkpoint: {checkpoint.repository} @ {checkpoint.resolved_revision}"
        )
        self.emit(
            f"Cache: {'checked cached files; reuse without a download' if checkpoint.cached else 'missing files; confirmation required'}"
        )
        size = checkpoint.download_bytes
        self.emit(
            f"Required transfer: {size:,} bytes"
            if size is not None
            else "Required transfer: size unknown"
        )


def build_experiment(
    ask: Callable,
    emit: Callable[[str], None],
    *,
    workspace: Path,
    demo_path: Path,
    adapter: str | None = None,
    demo: bool = False,
    ttm_demo_path: Path | None = None,
    profile_demo_paths: dict[str, Path] | None = None,
    seed_fields: dict[str, str] | None = None,
    explicit_fields=(),
    checkpoint_variant: str | None = None,
    preferences=None,
    guided_preparation: bool = False,
) -> ExperimentConfig | GuidedRunPlan | None:
    """Return a reviewed config, or an explicitly confirmed preparation plan.

    Default callers retain local-only review. Guided preparation can cache public
    configuration metadata; it never transfers weights or runs inference here.
    """
    try:
        if not demo:
            return _Guide(
                ask,
                emit,
                workspace,
                demo_path,
                adapter,
                ttm_demo_path,
                profile_demo_paths,
                seed_fields,
                explicit_fields,
                checkpoint_variant,
                preferences,
                guided_preparation,
            ).run()
        if seed_fields is not None or explicit_fields or checkpoint_variant is not None:
            raise ConfigError("Assisted suggestions require guided setup")
        if adapter in CHECKPOINT_REPOSITORIES:
            raise ConfigError("Use guided setup to select a pretrained checkpoint")
        fields = demo_fields(demo_path)
        fields["adapter"] = adapter or "last_value"
        arguments = {"workspace": workspace, "base_directory": Path.cwd()}
        dataset = inspect_data(fields, **arguments)
        fields.update(suggested_bounds(dataset, fields))
        prepared = prepare_experiment(fields, **arguments)
        emit(prepared.summary())
        plan = None
        if guided_preparation:
            from tsfm_lab.guided import review_guided

            plan = review_guided(prepared.config, offline=True)
        answer = _question(ask, emit, "Run experiment?", "no", ("yes", "no"))
        return (
            (plan if plan is not None else prepared.config) if answer == "yes" else None
        )
    except (_Cancel, _Back):
        emit("Experiment setup cancelled.")
        return None
