"""Interactive shell over shared configuration, managed execution, and records."""

import asyncio
from contextlib import ExitStack
from importlib.resources import as_file, files
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, ProgressBar, RichLog

from tsfm_lab.config import ExperimentConfig
from tsfm_lab.execution import ManagedRun
from tsfm_lab.local_paths import expand_local_user
from tsfm_lab.repl.text import literal
from tsfm_lab.run_modes import forecast_spec, is_future
from tsfm_lab.tui.views import HistoryScreen, ModelScreen, ResultsScreen
from tsfm_lab.tui.wizard import WizardScreen
from tsfm_lab.workflows import PROFILE_NAMES, workflow_summary

_STAGES = {
    "pending": "Waiting to start",
    "preparing": "Preparing data and model",
    "loading": "Loading model",
    "running": "Generating forecasts",
    "evaluating": "Calculating metrics",
    "saving": "Verifying and saving results",
}


class HomeScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="home-body"):
            yield Label("TSFM LAB", id="home-brand")
            yield Label("Local time-series forecasting experiments", id="home-subtitle")
            yield Label("Experiment workspace", id="workspace-label")
            yield Input(str(self.app.workspace), id="workspace-path")
            with Horizontal(id="home-actions"):
                yield Button("New experiment", id="new-experiment", variant="primary")
                yield Button("Browse models", id="browse-models")
                yield Button("Experiment history", id="open-history")
                yield Button("Compare experiments", id="open-comparison")
            yield Label(
                "Start with the bundled seasonal demo or select your own local data.",
                id="home-description",
            )
            yield Label(
                "No automatic model downloads. Results stay in your workspace.",
                id="home-status",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#new-experiment", Button).focus()

    def on_screen_resume(self) -> None:
        self.query_one("#workspace-path", Input).value = str(self.app.workspace)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "new-experiment": self.app.action_new_experiment,
            "browse-models": self.app.action_models,
            "open-history": self.app.action_history,
            "open-comparison": self.app.action_history,
        }
        if event.button.id in actions:
            actions[event.button.id]()


class RunningScreen(Screen):
    BINDINGS = [Binding("escape", "cancel", "Cancel run")]

    def __init__(self, config: ExperimentConfig):
        super().__init__()
        self.config = config
        self.cancellation_requested = False

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="running-body"):
            yield Label(self.config.name, id="running-title")
            dataset, spec = self.config.dataset, forecast_spec(self.config)
            mode_detail = (
                f" · Origin {spec.origin}"
                if is_future(self.config)
                else f" · Stride {spec.stride}"
            )
            yield Label(
                Text(
                    literal(
                        f"Usage {workflow_summary(self.config.workflow)}\n"
                        f"{dataset['target_column']} · {dataset['units']} · {dataset['frequency']}\n"
                        f"Context {spec.context_length} · Horizon {spec.horizon} observations{mode_detail}"
                    )
                ),
                id="execution-target",
            )
            yield Label("Preparing experiment", id="execution-stage")
            yield ProgressBar(total=None, show_eta=False, id="execution-progress")
            yield Label("Waiting for worker", id="execution-count")
            yield RichLog(markup=False, highlight=False, wrap=True, id="execution-log")
        with Horizontal(id="running-actions"):
            yield Button("Cancel run", id="cancel-run", variant="warning")
            yield Button("Back to home", id="run-back", disabled=True)
        yield Footer()

    def message(self, value: str) -> None:
        self.query_one("#execution-stage", Label).update(Text(value))
        self.query_one("#execution-log", RichLog).write(Text(value))

    def stopped(self, value: str) -> None:
        self.message(value)
        self.query_one("#execution-progress", ProgressBar).update(total=1, progress=1)
        self.query_one("#cancel-run", Button).disabled = True
        self.query_one("#run-back", Button).disabled = False
        self.query_one("#run-back", Button).focus()

    def request_cancellation(self) -> None:
        if self.cancellation_requested:
            return
        self.cancellation_requested = True
        self.message("Cancellation requested. Waiting for the worker to stop…")
        self.query_one("#cancel-run", Button).disabled = True

    def action_cancel(self) -> None:
        self.app.action_cancel_run()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-run":
            self.action_cancel()
        elif event.button.id == "run-back":
            self.app.pop_screen()


class LabApp(App):
    TITLE = "TSFM Lab"
    SUB_TITLE = "Forecast evaluation"
    CSS_PATH = "app.tcss"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("n", "new_experiment", "New experiment"),
        Binding("m", "models", "Models"),
        Binding("h", "history", "History"),
        Binding("q", "request_quit", "Quit"),
        Binding("ctrl+q", "request_quit", show=False, priority=True),
    ]

    def __init__(
        self, workspace=Path("results"), *, demo_path=None, worker_factory=ManagedRun
    ):
        super().__init__()
        self.workspace = expand_local_user(workspace).resolve()
        self.demo_path = Path(demo_path) if demo_path else None
        self.profile_demo_paths = {}
        self.worker_factory = worker_factory
        self.active_run = None
        self._quitting = False
        self._resources = ExitStack()

    def on_mount(self) -> None:
        if self.demo_path is None:
            self.demo_path = self._resources.enter_context(
                as_file(files("tsfm_lab.resources").joinpath("seasonal.csv"))
            )
        self.profile_demo_paths = {
            profile: self._resources.enter_context(
                as_file(files("tsfm_lab.resources").joinpath(f"{profile}-demo.csv"))
            )
            for profile in PROFILE_NAMES
            if profile != "generic"
        }
        self.push_screen(HomeScreen())
        self.set_interval(0.1, self._poll_execution)

    async def on_unmount(self) -> None:
        if self.active_run is not None:
            await asyncio.to_thread(self.active_run.close)
            self.active_run = None
        self._resources.close()

    def check_action(self, action, parameters):
        if action in {"new_experiment", "models", "history"}:
            return isinstance(self.screen, HomeScreen) and self.active_run is None
        return True

    def _read_workspace(self) -> bool:
        try:
            value = self.screen.query_one("#workspace-path", Input).value.strip()
            if not value or "://" in value:
                raise ValueError("Enter a local workspace directory")
            self.workspace = expand_local_user(value).resolve()
            return True
        except (OSError, RuntimeError, ValueError) as exc:
            self.screen.query_one("#home-status", Label).update(Text(str(exc)))
            return False

    def action_new_experiment(self) -> None:
        if isinstance(self.screen, HomeScreen) and self._read_workspace():
            self.push_screen(
                WizardScreen(
                    self.workspace,
                    self.demo_path,
                    profile_demo_paths=self.profile_demo_paths,
                ),
                self._configured,
            )

    def action_models(self) -> None:
        if isinstance(self.screen, HomeScreen) and self._read_workspace():
            self.push_screen(ModelScreen(), self._selected_model)

    def _selected_model(self, adapter) -> None:
        if adapter:
            self.push_screen(
                WizardScreen(
                    self.workspace,
                    self.demo_path,
                    initial_adapter=adapter,
                    profile_demo_paths=self.profile_demo_paths,
                ),
                self._configured,
            )

    def action_history(self) -> None:
        if isinstance(self.screen, HomeScreen) and self._read_workspace():
            self.push_screen(HistoryScreen(self.workspace))

    def _configured(self, config) -> None:
        if config is None:
            return
        from tsfm_lab.guided import GuidedRunPlan

        selected = config
        if isinstance(config, GuidedRunPlan):
            config = config.config
        self.workspace = config.output_directory
        self.push_screen(RunningScreen(config))
        self.call_after_refresh(self._start_run, selected)

    def _start_run(self, config) -> None:
        from tsfm_lab.guided import GuidedRunPlan
        from tsfm_lab.guided_execution import ManagedGuidedRun

        try:
            self.active_run = (
                ManagedGuidedRun(config)
                if isinstance(config, GuidedRunPlan)
                else self.worker_factory(config)
            )
            self.active_run.start()
        except (OSError, ValueError, RuntimeError) as exc:
            if self.active_run is not None:
                self.active_run.close()
            self.active_run = None
            self.screen.stopped(f"Could not start experiment: {exc}")

    def action_cancel_run(self) -> None:
        if self.active_run is not None and self.active_run.is_running:
            self.active_run.request_cancel()
            self.screen.request_cancellation()

    def _poll_execution(self) -> None:
        if self.active_run is None:
            return
        for event in self.active_run.poll():
            kind = event["type"]
            if kind == "stage":
                if not self.screen.cancellation_requested:
                    self.screen.message(_STAGES.get(event["stage"], event["stage"]))
            elif kind == "progress":
                completed, total = event["completed"], event["total"]
                self.screen.query_one("#execution-progress", ProgressBar).update(
                    total=total, progress=completed
                )
                self.screen.query_one("#execution-count", Label).update(
                    f"Windows {completed} / {total}"
                )
            elif kind in {"completed", "failed", "cancelled"}:
                self._finished(event)
                return

    def _finished(self, event) -> None:
        run_id = (
            event.get("run_id")
            or event.get("result", {}).get("run_id")
            or self.active_run.run_id
        )
        self.active_run.close()
        self.active_run = None
        if self._quitting:
            self.exit()
        elif event["type"] == "completed":
            self.pop_screen()
            self.push_screen(ResultsScreen(self.workspace, run_id))
        else:
            message = (
                "Experiment cancelled"
                if event["type"] == "cancelled"
                else str(event.get("message", event.get("error", "Experiment failed")))
            )
            self.screen.stopped(message)
            if event.get("log_tail"):
                self.screen.query_one("#execution-log", RichLog).write(
                    Text(event["log_tail"])
                )
            if run_id:
                self.screen.query_one("#execution-log", RichLog).write(
                    Text(f"Saved record: {run_id}")
                )

    def action_request_quit(self) -> None:
        if self.active_run is not None and self.active_run.is_running:
            self._quitting = True
            self.action_cancel_run()
        else:
            self.exit()
