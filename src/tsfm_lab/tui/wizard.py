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
from tsfm_lab.tui.forms import (
    PreparedExperiment,
    default_fields,
    demo_fields,
    inspect_data,
    prepare_experiment,
    suggested_bounds,
    validate_model_fields,
)

_PAGES = ("model", "data", "evaluation", "review")


def _field(label: str, key: str, value: str = "", placeholder: str = "") -> Vertical:
    control = Input(value=value, placeholder=placeholder, id=f"wizard-{key}")
    control.tooltip = label
    return Vertical(Label(label), control, classes="wizard-field")


def _selection(label: str, key: str, choices, value: str) -> Vertical:
    control = Select(choices, value=value, allow_blank=False, id=f"wizard-{key}")
    control.tooltip = label
    return Vertical(Label(label), control, classes="wizard-field")


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
        self, workspace: Path, demo_path: Path, initial_adapter: str = "last_value"
    ) -> None:
        super().__init__()
        self.workspace, self.demo_path = Path(workspace), Path(demo_path)
        self.base_directory = Path.cwd().resolve()
        self._defaults = default_fields(initial_adapter)
        self._step_index = 0
        self._busy = False
        self._cancelled = False
        self._prepared: PreparedExperiment | None = None
        self._review_snapshot: dict[str, str] | None = None

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
                        ("Last value", "last_value"),
                        ("Seasonal naive", "seasonal_naive"),
                        ("Chronos-Bolt", "chronos_bolt"),
                    ],
                    self._defaults["adapter"],
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
                            [
                                ("Chronos-Bolt Tiny", "amazon/chronos-bolt-tiny"),
                                ("Chronos-Bolt Small", "amazon/chronos-bolt-small"),
                            ],
                            self._defaults["repository"],
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
        self._update_model_visibility()
        self._show_step()

    def _fields(self) -> dict[str, str]:
        fields = {}
        for key in self._defaults:
            control = self.query_one(f"#wizard-{key.replace('_', '-')}")
            fields[key] = str(control.value)
        return fields

    def _arguments(self) -> dict:
        return {"workspace": self.workspace, "base_directory": self.base_directory}

    def _update_model_visibility(self) -> None:
        adapter = self.query_one("#wizard-adapter", Select).value
        source = self.query_one("#wizard-source", Select).value
        self.query_one("#wizard-seasonal-fields").display = adapter == "seasonal_naive"
        self.query_one("#wizard-checkpoint-fields").display = adapter == "chronos_bolt"
        self.query_one("#wizard-hub-fields").display = source == "huggingface"
        self.query_one("#wizard-local-fields").display = source == "local"

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
            "evaluation": "context-length",
            "review": "run",
        }[self.step]
        self.query_one(f"#wizard-{first}").focus(scroll_visible=False)
        self.query_one(f"#wizard-{self.step}", VerticalScroll).scroll_home(
            animate=False, immediate=True
        )

    def _invalidate(self) -> None:
        self._prepared, self._review_snapshot = None, None
        if self.is_mounted:
            self.query_one("#wizard-run", Button).disabled = True

    @on(Input.Changed)
    def input_changed(self) -> None:
        self._invalidate()

    @on(Select.Changed)
    def selection_changed(self) -> None:
        self._invalidate()
        if self.is_mounted:
            self._update_model_visibility()

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
            self.dismiss(self._prepared.config)

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
        defaults = demo_fields(self.demo_path)
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
            if self._cancelled:
                return
            self._prepared, self._review_snapshot = prepared, fields
            self.query_one("#wizard-summary", Static).update(prepared.summary())
            self._step_index = 3
            self._show_step()
        except (ConfigError, ValueError, OSError, ImportError) as exc:
            if not self._cancelled:
                self.query_one("#wizard-error", Static).update(str(exc))
        finally:
            self._set_busy(False)
