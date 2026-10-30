"""Keyboard-first experiment setup with validation away from terminal rendering."""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    ContentSwitcher,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
)

from tsfm_lab.config import ConfigError, ExperimentConfig
from tsfm_lab.models.catalog import (
    CHECKPOINT_REPOSITORIES,
    catalog_entries,
    catalog_info,
)
from tsfm_lab.tui.forms import (
    PreparedExperiment,
    PreparedFutureForecast,
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

_PAGES = ("model", "data", "evaluation", "review")


def _field(label: str, key: str, value: str = "", placeholder: str = "") -> Vertical:
    control = Input(value=value, placeholder=placeholder, id=f"wizard-{key}")
    control.tooltip = label
    return Vertical(Label(label), control, classes="wizard-field")


def _selection(label: str, key: str, choices, value: str) -> Vertical:
    control = Select(choices, value=value, allow_blank=False, id=f"wizard-{key}")
    control.tooltip = label
    return Vertical(Label(label), control, classes="wizard-field")


def _same_path(value: str, path: Path) -> bool:
    try:
        return Path(value).expanduser().resolve() == path.expanduser().resolve()
    except (OSError, ValueError, RuntimeError):
        return False


class WizardScreen(Screen[ExperimentConfig | None]):
    """Return a validated in-memory config; run execution belongs to the caller."""

    BINDINGS = [("escape", "cancel", "Cancel")]
    DEFAULT_CSS = """
    WizardScreen { layout: vertical; }
    #wizard-title { height: 1; padding: 0 1; text-style: bold; }
    #wizard-step { height: 2; padding: 0 1; }
    #wizard-content { height: 1fr; }
    .wizard-page { height: 1fr; padding: 0 1; }
    .wizard-field { height: auto; margin-bottom: 1; }
    .wizard-field Label { height: auto; }
    .wizard-field Input, .wizard-field Select { width: 1fr; }
    .wizard-conditional { height: auto; }
    #wizard-error { height: auto; max-height: 4; padding: 0 1; color: $error; }
    #wizard-data-summary, #wizard-summary { height: auto; margin-bottom: 1; }
    #wizard-controls { height: 3; padding: 0 1; }
    #wizard-controls Button { width: 1fr; min-width: 8; margin-right: 1; }
    #wizard-use-demo, #wizard-inspect { margin-bottom: 1; }
    """

    def __init__(
        self,
        workspace: Path,
        demo_path: Path,
        initial_adapter: str = "last_value",
        profile_demo_paths: dict[str, Path] | None = None,
    ) -> None:
        super().__init__()
        self.workspace, self.demo_path = Path(workspace), Path(demo_path)
        self._ordinary_demo_path = self.demo_path
        self._profile_demo_paths = {
            profile: Path(path) for profile, path in (profile_demo_paths or {}).items()
        }
        self.base_directory = Path.cwd().resolve()
        self._defaults = default_fields(initial_adapter)
        self._defaults["usage_profile"] = "generic"
        self._step_index = 0
        self._busy = False
        self._cancelled = False
        self._prepared: PreparedExperiment | None = None
        self._future_plan = None
        self._review_snapshot: dict[str, str] | None = None
        self._adapter_reference = initial_adapter
        self._profile_reference: tuple[str, str] | None = None
        self._usage_profile_reference = "generic"
        self._mode_reference = "historical_evaluation"
        self._example_declarations: dict[str, str] = {}

    @property
    def step(self) -> str:
        return _PAGES[self._step_index]

    @property
    def busy(self) -> bool:
        return self._busy

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("New experiment", id="wizard-title")
        yield Static("", id="wizard-step")
        with ContentSwitcher(initial="wizard-model", id="wizard-content"):
            with VerticalScroll(id="wizard-model", classes="wizard-page"):
                yield _field("Experiment name", "name", self._defaults["name"])
                yield _selection(
                    "Forecast model",
                    "adapter",
                    [
                        (entry["display_name"], entry["adapter_id"])
                        for entry in catalog_entries()
                    ],
                    self._defaults["adapter"],
                )
                yield _selection(
                    "Forecast task",
                    "point-statistic",
                    [
                        (value, value)
                        for value in catalog_info(self._defaults["adapter"])[
                            "supported_tasks"
                        ]
                    ],
                    self._defaults["point_statistic"],
                )
                yield _selection(
                    "Usage profile",
                    "usage-profile",
                    [(profile.title(), profile) for profile in PROFILE_NAMES],
                    self._defaults["usage_profile"],
                )
                with Vertical(
                    id="wizard-seasonal-fields", classes="wizard-conditional"
                ):
                    yield _field(
                        "Baseline seasonal period (observations)",
                        "seasonal-period",
                        "4",
                    )
                with Vertical(
                    id="wizard-checkpoint-fields", classes="wizard-conditional"
                ):
                    yield Static(
                        "Use existing checkpoints. This wizard never downloads model files.",
                        markup=False,
                    )
                    yield _selection(
                        "Checkpoint source",
                        "source",
                        [
                            ("Cached Hugging Face checkpoint", "huggingface"),
                            ("Local checkpoint directory", "local"),
                        ],
                        "huggingface",
                    )
                    with Vertical(id="wizard-hub-fields", classes="wizard-conditional"):
                        yield _selection(
                            "Repository",
                            "repository",
                            self._repository_choices(),
                            self._defaults["repository"],
                        )
                        with Vertical(
                            id="wizard-profile-fields", classes="wizard-conditional"
                        ):
                            yield _selection(
                                "Checkpoint profile",
                                "checkpoint-profile",
                                [("Default", "default")],
                                "default",
                            )
                        yield _field(
                            "Exact checkpoint SHA (40 lowercase characters)", "revision"
                        )
                        yield _field("Cache directory (optional)", "cache-dir")
                    with Vertical(
                        id="wizard-local-fields", classes="wizard-conditional"
                    ):
                        yield _field("Local checkpoint directory", "checkpoint-path")
            with VerticalScroll(id="wizard-data", classes="wizard-page"):
                yield Button("Use demo dataset", id="wizard-use-demo")
                yield Static("", id="wizard-usage-guidance", markup=False)
                with Vertical(id="wizard-finance-usage", classes="wizard-conditional"):
                    yield _selection(
                        "Finance target kind",
                        "finance-target-kind",
                        [
                            ("Choose target", ""),
                            ("Close price", "close_price"),
                            ("Return", "return"),
                            ("Other", "other"),
                        ],
                        "",
                    )
                    with Vertical(
                        id="wizard-finance-other", classes="wizard-conditional"
                    ):
                        yield _field("Finance measurement", "finance-measurement")
                with Vertical(id="wizard-energy-usage", classes="wizard-conditional"):
                    yield _selection(
                        "Energy measurement",
                        "energy-measurement",
                        [
                            ("Choose measurement", ""),
                            ("Power", "power"),
                            ("Energy", "energy"),
                            ("Other", "other"),
                        ],
                        "",
                    )
                    with Vertical(
                        id="wizard-energy-other", classes="wizard-conditional"
                    ):
                        yield _field("Energy quantity", "energy-quantity")
                with Vertical(id="wizard-weather-usage", classes="wizard-conditional"):
                    yield _field("Weather variable", "weather-variable")
                    yield _field("Weather station", "weather-station")
                    yield _selection(
                        "Weather measurement semantics",
                        "weather-measurement-semantics",
                        [
                            ("Choose semantics", ""),
                            ("Instantaneous", "instantaneous"),
                            ("Interval mean", "interval_mean"),
                            ("Interval total", "interval_total"),
                            ("Interval maximum", "interval_max"),
                            ("Other", "other"),
                        ],
                        "",
                    )
                    with Vertical(
                        id="wizard-weather-other", classes="wizard-conditional"
                    ):
                        yield _field(
                            "Weather semantics description", "weather-description"
                        )
                for label, key in (
                    ("CSV or Parquet path", "dataset_path"),
                    ("Timestamp column", "timestamp_column"),
                    ("Target column", "target_column"),
                    ("Observation frequency", "frequency"),
                    ("Source timezone (required for naive timestamps)", "timezone"),
                    ("Target units", "units"),
                ):
                    yield _field(label, key.replace("_", "-"), self._defaults[key])
                yield Button("Inspect dataset", id="wizard-inspect")
                yield Static("", id="wizard-data-summary", markup=False)
            with VerticalScroll(id="wizard-evaluation", classes="wizard-page"):
                yield _selection(
                    "Forecast mode",
                    "mode",
                    [
                        ("Historical evaluation", "historical_evaluation"),
                        ("Future forecasting", "future_forecast"),
                    ],
                    self._defaults["mode"],
                )
                yield _field(
                    "Forecast origin (latest or final observed timestamp)",
                    "forecast-origin",
                    self._defaults["forecast_origin"],
                )
                for label, key in (
                    ("Context length (observations)", "context_length"),
                    ("Forecast horizon (observations)", "horizon"),
                    ("Evaluation stride (observations)", "stride"),
                    ("MASE seasonal lag (observations)", "mase_period"),
                    (
                        "First target timestamp (inclusive, with timezone)",
                        "evaluation_start",
                    ),
                    (
                        "Target end timestamp (exclusive, with timezone)",
                        "evaluation_end",
                    ),
                ):
                    yield _field(label, key.replace("_", "-"), self._defaults[key])
            with VerticalScroll(id="wizard-review", classes="wizard-page"):
                yield Static("", id="wizard-summary", markup=False)
        yield Static("", id="wizard-error", markup=False)
        with Horizontal(id="wizard-controls"):
            yield Button("Cancel", id="wizard-cancel")
            yield Button("Back", id="wizard-back")
            yield Button("Next", id="wizard-next", variant="primary")
            yield Button("Run", id="wizard-run", variant="success", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self._sync_checkpoint_profiles()
        self._update_model_visibility()
        self._update_usage_visibility()
        self._update_mode_visibility()
        self._show_step()

    def _fields(self) -> dict[str, str]:
        fields = {}
        for key in self._defaults:
            control = self.query_one(f"#wizard-{key.replace('_', '-')}")
            fields[key] = str(control.value)
        return fields

    def _arguments(self) -> dict:
        return {"workspace": self.workspace, "base_directory": self.base_directory}

    def _repository_choices(self, adapter=None) -> list[tuple[str, str]]:
        repositories = CHECKPOINT_REPOSITORIES.get(
            adapter or self._defaults["adapter"],
            CHECKPOINT_REPOSITORIES["chronos_bolt"],
        )
        return [
            (f"{catalog_info(repository)['display_name']} · {repository}", repository)
            for repository in repositories
        ]

    def _update_model_visibility(self) -> None:
        adapter = self.query_one("#wizard-adapter", Select).value
        source = self.query_one("#wizard-source", Select).value
        self.query_one("#wizard-seasonal-fields").display = adapter == "seasonal_naive"
        self.query_one("#wizard-checkpoint-fields").display = (
            adapter in CHECKPOINT_REPOSITORIES
        )
        self.query_one("#wizard-hub-fields").display = source == "huggingface"
        self.query_one("#wizard-local-fields").display = source == "local"

    def _update_mode_visibility(self):
        future = self.query_one("#wizard-mode", Select).value == "future_forecast"
        for key in ("stride", "mase-period", "evaluation-start", "evaluation-end"):
            self.query_one(f"#wizard-{key}", Input).parent.display = not future
        self.query_one("#wizard-forecast-origin", Input).parent.display = future

    def _update_usage_visibility(self) -> None:
        profile = str(self.query_one("#wizard-usage-profile", Select).value)
        for name in ("finance", "energy", "weather"):
            self.query_one(f"#wizard-{name}-usage").display = profile == name
        for name, key in (
            ("finance", "finance-target-kind"),
            ("energy", "energy-measurement"),
            ("weather", "weather-measurement-semantics"),
        ):
            self.query_one(f"#wizard-{name}-other").display = (
                self.query_one(f"#wizard-{key}", Select).value == "other"
            )
        frequency = self.query_one("#wizard-frequency", Input).value
        try:
            guidance = profile_guidance(profile, frequency)
        except ValueError:
            guidance = profile_guidance(profile)
        self.query_one("#wizard-usage-guidance", Static).update(guidance)
        self.query_one("#wizard-use-demo", Button).label = (
            "Use profile example" if profile != "generic" else "Use demo dataset"
        )

    def _sync_usage_profile(self) -> None:
        profile = str(self.query_one("#wizard-usage-profile", Select).value)
        previous_demo = self.demo_path
        self.demo_path = self._profile_demo_paths.get(profile, self._ordinary_demo_path)
        path = self.query_one("#wizard-dataset-path", Input).value
        if (
            profile != self._usage_profile_reference
            and path.strip()
            and _same_path(path, previous_demo)
        ):
            self._apply_demo_fields()
        self._usage_profile_reference = profile

    def _apply_demo_fields(self) -> None:
        profile = str(self.query_one("#wizard-usage-profile", Select).value)
        defaults = (
            profile_example_fields(profile, self.demo_path)
            if profile in self._profile_demo_paths
            else demo_fields(self.demo_path)
        )
        for key in (
            "dataset_path",
            "timestamp_column",
            "target_column",
            "frequency",
            "timezone",
            "units",
        ):
            self.query_one(f"#wizard-{key.replace('_', '-')}", Input).value = defaults[
                key
            ]
        for key in (
            "finance_target_kind",
            "energy_measurement",
            "weather_variable",
            "weather_station",
            "weather_measurement_semantics",
        ):
            control = self.query_one(f"#wizard-{key.replace('_', '-')}")
            if defaults[key]:
                control.value = defaults[key]
        self._example_declarations = {
            key: defaults[key]
            for key in (
                "units",
                "finance_target_kind",
                "energy_measurement",
                "weather_variable",
                "weather_station",
                "weather_measurement_semantics",
            )
            if defaults[key] and profile in self._profile_demo_paths
        }
        self._update_usage_visibility()

    def _clear_example_declarations(self) -> None:
        path = self.query_one("#wizard-dataset-path", Input).value
        if not path.strip() or _same_path(path, self.demo_path):
            return
        declarations, self._example_declarations = self._example_declarations, {}
        for key, value in declarations.items():
            control = self.query_one(f"#wizard-{key.replace('_', '-')}")
            if str(control.value) == value:
                control.value = "unspecified" if key == "units" else ""

    def _sync_checkpoint_profiles(self) -> None:
        adapter = str(self.query_one("#wizard-adapter", Select).value)
        repository = str(self.query_one("#wizard-repository", Select).value)
        variants = [
            item
            for item in catalog_info(adapter).get("variants", [])
            if item["repository"] == repository
        ]
        self.query_one("#wizard-profile-fields").display = len(variants) > 1
        if len(variants) <= 1:
            self._profile_reference = None
            return
        control = self.query_one("#wizard-checkpoint-profile", Select)
        revision = self.query_one("#wizard-revision", Input)
        names = {item["variant"] for item in variants}
        previous = control.value
        selected = next(
            (
                item["variant"]
                for item in variants
                if item["reference_revision"] == revision.value
            ),
            None,
        )
        if selected is None:
            selected = previous if previous in names else variants[0]["variant"]
        control.set_options(
            [(item["display_name"], item["variant"]) for item in variants]
        )
        control.value = selected
        self._select_checkpoint_profile(repository, str(selected))

    def _select_checkpoint_profile(self, repository: str, name: str) -> None:
        reference = (repository, name)
        if reference == self._profile_reference:
            return
        variant = catalog_info(repository, name)
        self._profile_reference = reference
        self.query_one("#wizard-revision", Input).value = variant["reference_revision"]
        caps = variant["capabilities"]
        if caps["min_context"] == caps["max_context"]:
            self.query_one("#wizard-context-length", Input).value = str(
                caps["min_context"]
            )

    def _show_step(self) -> None:
        self.query_one("#wizard-content", ContentSwitcher).current = (
            f"wizard-{self.step}"
        )
        self.query_one("#wizard-step", Static).update(
            f"{self._step_index + 1} of 4 · {self.step.title()}"
        )
        self.query_one("#wizard-back", Button).disabled = (
            self._busy or self._step_index == 0
        )
        self.query_one("#wizard-next", Button).display = self.step != "review"
        self.query_one("#wizard-run", Button).display = self.step == "review"
        self.query_one("#wizard-run", Button).disabled = (
            self._prepared is None or self._busy
        )
        self.query_one("#wizard-error", Static).update("")
        self.call_after_refresh(self._focus_step)

    def _focus_step(self) -> None:
        first = {
            "model": "name",
            "data": "dataset-path",
            "evaluation": "mode",
            "review": "run",
        }[self.step]
        self.query_one(f"#wizard-{first}").focus(scroll_visible=False)
        self.query_one(f"#wizard-{self.step}", VerticalScroll).scroll_home(
            animate=False, immediate=True
        )

    def _invalidate(self) -> None:
        self._prepared, self._review_snapshot = None, None
        self._future_plan = None
        if self.is_mounted:
            self.query_one("#wizard-run", Button).disabled = True

    @on(Input.Changed)
    def input_changed(self, event: Input.Changed) -> None:
        self._invalidate()
        if self.is_mounted:
            if event.input.id == "wizard-dataset-path":
                self._clear_example_declarations()
            self._update_usage_visibility()

    @on(Select.Changed)
    def selection_changed(self, event: Select.Changed) -> None:
        self._invalidate()
        if self.is_mounted:
            if event.select.id == "wizard-adapter":
                adapter = str(event.value)
                if adapter != self._adapter_reference:
                    information = catalog_info(adapter)
                    task = self.query_one("#wizard-point-statistic", Select)
                    task.set_options(
                        [(value, value) for value in information["supported_tasks"]]
                    )
                    task.value = default_fields(adapter)["point_statistic"]
                    self._adapter_reference = adapter
                repository = self.query_one("#wizard-repository", Select)
                choices = self._repository_choices(event.value)
                previous = repository.value
                repository.set_options(choices)
                repository.value = (
                    previous
                    if previous in {value for _, value in choices}
                    else choices[0][1]
                )
                if previous != repository.value:
                    self.query_one("#wizard-revision", Input).value = ""
                self._sync_checkpoint_profiles()
            elif event.select.id == "wizard-repository":
                self._sync_checkpoint_profiles()
            elif event.select.id == "wizard-checkpoint-profile":
                repository = str(self.query_one("#wizard-repository", Select).value)
                adapter = str(self.query_one("#wizard-adapter", Select).value)
                names = {
                    item["variant"]
                    for item in catalog_info(adapter).get("variants", [])
                    if item["repository"] == repository
                }
                if event.value in names and len(names) > 1:
                    self._select_checkpoint_profile(repository, str(event.value))
            elif event.select.id == "wizard-usage-profile":
                self._sync_usage_profile()
            elif event.select.id == "wizard-mode":
                mode = str(event.value)
                if mode != self._mode_reference:
                    for key in (
                        "stride",
                        "mase-period",
                        "evaluation-start",
                        "evaluation-end",
                    ):
                        self.query_one(f"#wizard-{key}", Input).value = ""
                    self.query_one("#wizard-forecast-origin", Input).value = (
                        "latest" if mode == "future_forecast" else ""
                    )
                    if mode == "historical_evaluation":
                        self.query_one("#wizard-stride", Input).value = self.query_one(
                            "#wizard-horizon", Input
                        ).value
                        self.query_one("#wizard-mase-period", Input).value = "4"
                    self._mode_reference = mode
                self._update_mode_visibility()
            self._update_model_visibility()
            self._update_usage_visibility()

    @on(Input.Submitted)
    def input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.action_next()

    @on(Button.Pressed)
    def button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        action = {
            "wizard-cancel": self.action_cancel,
            "wizard-back": self.action_back,
            "wizard-next": self.action_next,
            "wizard-run": self.action_run,
            "wizard-use-demo": self._use_demo,
            "wizard-inspect": lambda: self._inspect(False),
        }.get(event.button.id)
        if action:
            action()

    def action_cancel(self) -> None:
        self._cancelled = True
        self.dismiss(None)

    def action_back(self) -> None:
        if not self._busy and self._step_index:
            self._invalidate()
            self._step_index -= 1
            self._show_step()

    def action_next(self) -> None:
        if self._busy:
            return
        if self.step == "model":
            try:
                validate_model_fields(self._fields(), **self._arguments())
            except ConfigError as exc:
                self.query_one("#wizard-error", Static).update(str(exc))
                return
            self._step_index = 1
            self._show_step()
        elif self.step == "data":
            self._inspect(True)
        elif self.step == "evaluation":
            self._review()

    def action_run(self) -> None:
        if (
            self._prepared is not None
            and self._review_snapshot == self._fields()
            and not self._busy
        ):
            self.dismiss(self._future_plan or self._prepared.config)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        if self._cancelled or not self.is_mounted:
            return
        for control in self.query("Input, Select"):
            control.disabled = busy
        for name in ("next", "inspect", "use-demo"):
            self.query_one(f"#wizard-{name}", Button).disabled = busy
        self.query_one("#wizard-back", Button).disabled = busy or self._step_index == 0
        self.query_one("#wizard-run", Button).disabled = busy or self._prepared is None

    def _use_demo(self) -> None:
        if self._busy:
            return
        self._apply_demo_fields()
        self._inspect(False)

    @work(exclusive=True, group="wizard-validation", exit_on_error=False)
    async def _inspect(self, advance: bool) -> None:
        if self._busy:
            return
        self._set_busy(True)
        fields = self._fields()
        self.query_one("#wizard-error", Static).update("Inspecting dataset…")
        try:
            dataset = await asyncio.to_thread(inspect_data, fields, **self._arguments())
            if self._cancelled:
                return
            proposal = ""
            try:
                bounds = suggested_bounds(dataset, fields)
                for key, value in bounds.items():
                    control = self.query_one(f"#wizard-{key.replace('_', '-')}", Input)
                    if not control.value.strip():
                        control.value = value
                        proposal = "Proposed bounds are visible on the Evaluation step."
            except ConfigError:
                proposal = "Set a context and horizon that fit this dataset."
            self.query_one("#wizard-data-summary", Static).update(
                f"Validated {len(dataset.values)} observations · {dataset.target_name} · {dataset.frequency}\n{dataset.timestamps[0].isoformat()} to {dataset.timestamps[-1].isoformat()}\n{proposal}"
            )
            self.query_one("#wizard-error", Static).update("")
            if advance:
                self._step_index = 2
                self._show_step()
        except (ConfigError, ValueError, OSError, ImportError) as exc:
            if not self._cancelled:
                self.query_one("#wizard-error", Static).update(str(exc))
        finally:
            self._set_busy(False)

    @work(exclusive=True, group="wizard-validation", exit_on_error=False)
    async def _review(self) -> None:
        if self._busy:
            return
        self._set_busy(True)
        fields = self._fields()
        self.query_one("#wizard-error", Static).update("Validating experiment…")
        try:
            prepared = await asyncio.to_thread(
                prepare_experiment, fields, **self._arguments()
            )
            future_plan = (
                await asyncio.to_thread(prepared.review_plan)
                if isinstance(prepared, PreparedFutureForecast)
                else None
            )
            if self._cancelled:
                return
            self._prepared, self._review_snapshot = prepared, fields
            self._future_plan = future_plan
            self.query_one("#wizard-summary", Static).update(prepared.summary())
            self._step_index = 3
            self._show_step()
        except (ConfigError, ValueError, OSError, ImportError) as exc:
            if not self._cancelled:
                self.query_one("#wizard-error", Static).update(str(exc))
        finally:
            self._set_busy(False)
