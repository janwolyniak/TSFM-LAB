"""Whole keyboard workflow and UI stop semantics, using disposable local runs."""

import asyncio
from dataclasses import replace

import pandas as pd
import pytest

pytest.importorskip("textual")

from textual.app import App  # noqa: E402
from textual.widgets import Button, DataTable, Footer, Label, ProgressBar  # noqa: E402

from tsfm_lab.application import run_from_config  # noqa: E402
from tsfm_lab.config import load_config  # noqa: E402
from tsfm_lab.experiments import ExperimentStore  # noqa: E402
from tsfm_lab.tui.app import HomeScreen, LabApp, RunningScreen  # noqa: E402
from tsfm_lab.tui.views import ResultsScreen  # noqa: E402
from tsfm_lab.tui.wizard import WizardScreen  # noqa: E402


async def wait_for(pilot, condition):
    for _ in range(300):
        await pilot.pause(0.01)
        if condition():
            return
    raise AssertionError("Interactive operation did not finish")


async def activate(pilot, app, widget_id):
    """Reach every control through keyboard focus traversal."""
    for _ in range(40):
        if app.focused is not None and app.focused.id == widget_id:
            await pilot.press("enter")
            await pilot.pause()
            return
        await pilot.press("tab")
    raise AssertionError(f"Keyboard could not reach {widget_id}")


def test_running_profile_metadata_is_literal_and_strips_terminal_controls(
    config_factory,
):
    config = replace(
        load_config(config_factory()),
        workflow={
            "schema_version": 1,
            "profile": "weather",
            "options": {
                "variable": "wind [/red]",
                "station": "[red]station[/red]\x1b[31m",
                "measurement_semantics": "instantaneous",
            },
        },
    )

    class ProfileApp(App):
        def on_mount(self):
            self.push_screen(RunningScreen(config))

    async def exercise():
        app = ProfileApp()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            rendered = str(app.screen.query_one("#execution-target", Label).render())
            assert "wind [/red]" in rendered
            assert "[red]station[/red]" in rendered
            assert "\x1b" not in rendered

    asyncio.run(exercise())


def test_keyboard_demo_runs_in_spawn_worker_and_matches_shell(config_factory, tmp_path):
    async def exercise():
        workspace = tmp_path / "interactive"
        app = LabApp(workspace=workspace)
        async with app.run_test(size=(80, 24)) as pilot:
            assert isinstance(app.screen, HomeScreen)
            await activate(pilot, app, "new-experiment")
            assert isinstance(app.screen, WizardScreen)
            await activate(pilot, app, "wizard-next")
            await activate(pilot, app, "wizard-use-demo")
            await wait_for(pilot, lambda: not app.screen.busy)
            await activate(pilot, app, "wizard-next")
            await wait_for(pilot, lambda: app.screen.step == "evaluation")
            await activate(pilot, app, "wizard-next")
            await wait_for(pilot, lambda: app.screen.step == "review")
            await activate(pilot, app, "wizard-run")
            assert not isinstance(app.screen, WizardScreen), str(
                app.screen.query_one("#wizard-error").render()
            )
            await wait_for(
                pilot,
                lambda: isinstance(app.screen, ResultsScreen)
                or (
                    isinstance(app.screen, RunningScreen)
                    and not app.screen.query_one("#run-back", Button).disabled
                ),
            )
            assert isinstance(app.screen, ResultsScreen), str(
                app.screen.query_one("#execution-stage", Label).render()
            )
            await wait_for(pilot, lambda: app.screen.completed)
            assert app.active_run is None
            assert app.screen.query_one("#overall-metrics", DataTable).row_count == 3
            run_id = app.screen.run_id
            await pilot.press("escape")
            assert isinstance(app.screen, HomeScreen)
        store = ExperimentStore(workspace)
        result, predictions = store.load_result(run_id)
        assert result["counts"]["successful_windows"] == 4
        assert len(predictions) == 16
        assert result["metrics"]["overall"]["mae"]["value"] == 2.5
        shell = run_from_config(config_factory())
        shell_result, shell_predictions = ExperimentStore(
            tmp_path / "workspace"
        ).load_result(shell["run_id"])
        assert result["metrics"] == shell_result["metrics"]
        pd.testing.assert_frame_equal(predictions, shell_predictions)

    asyncio.run(exercise())


class ControlledWorker:
    """Only event delivery is mocked; the app owns its usual screen lifecycle."""

    def __init__(self, config):
        self.events = []
        self.is_running = True
        self.run_id = None
        self.cancelled = False
        self.closed = False

    def start(self):
        pass

    def poll(self):
        events, self.events = self.events, []
        return events

    def request_cancel(self):
        self.cancelled = True

    def close(self):
        self.closed = True
        self.is_running = False


async def running_app(app, pilot, config):
    app._configured(config)
    await wait_for(pilot, lambda: app.active_run is not None)
    return app.active_run


def test_cancel_waits_for_terminal_event_and_keeps_request_visible(config_factory):
    async def exercise():
        config = load_config(config_factory())
        app = LabApp(config.output_directory, worker_factory=ControlledWorker)
        async with app.run_test(size=(80, 24)) as pilot:
            worker = await running_app(app, pilot, config)
            screen = app.screen
            cancel = screen.query_one("#cancel-run", Button)
            assert cancel.region.bottom <= screen.query_one(Footer).region.y
            await pilot.press("ctrl+p")
            assert app.screen is screen
            await pilot.press("escape")
            assert worker.cancelled
            assert not worker.closed
            assert screen.query_one("#run-back", Button).disabled
            worker.events = [
                {"type": "stage", "stage": "saving"},
                {"type": "progress", "completed": 2, "total": 4},
            ]
            await pilot.pause(0.15)
            assert "Cancellation requested" in str(
                screen.query_one("#execution-stage", Label).render()
            )
            assert screen.query_one("#execution-progress", ProgressBar).progress == 2
            worker.is_running = False
            worker.events = [{"type": "cancelled"}]
            await wait_for(pilot, lambda: worker.closed)
            assert app.active_run is None
            assert not screen.query_one("#run-back", Button).disabled
            assert "Experiment cancelled" in str(
                screen.query_one("#execution-stage", Label).render()
            )
            await activate(pilot, app, "run-back")
            assert isinstance(app.screen, HomeScreen)

    asyncio.run(exercise())


def test_home_model_and_history_navigation_use_current_workspace(tmp_path):
    async def exercise():
        app = LabApp(tmp_path / "runs")
        async with app.run_test(size=(80, 24)) as pilot:
            await activate(pilot, app, "browse-models")
            await pilot.press("down", "enter")
            assert isinstance(app.screen, WizardScreen)
            assert app.screen.query_one("#wizard-adapter").value == "seasonal_naive"
            await pilot.press("escape")
            assert isinstance(app.screen, HomeScreen)
            await activate(pilot, app, "open-comparison")
            await wait_for(
                pilot,
                lambda: "0 saved experiments"
                in str(app.screen.query_one("#history-status").render()),
            )
            await pilot.press("escape")
            assert isinstance(app.screen, HomeScreen)
            assert not (tmp_path / "runs").exists()

    asyncio.run(exercise())


def test_ctrl_q_waits_for_worker_before_exiting(config_factory):
    async def exercise():
        config = load_config(config_factory())
        app = LabApp(config.output_directory, worker_factory=ControlledWorker)
        async with app.run_test() as pilot:
            worker = await running_app(app, pilot, config)
            await pilot.press("ctrl+q")
            assert app._quitting
            assert worker.cancelled
            assert not worker.closed
            assert app.is_running
            worker.is_running = False
            worker.events = [{"type": "cancelled"}]
            await wait_for(pilot, lambda: worker.closed)
        assert not app.is_running

    asyncio.run(exercise())


def test_failed_worker_shows_error_and_literal_log_tail(config_factory):
    async def exercise():
        config = load_config(config_factory())
        app = LabApp(config.output_directory, worker_factory=ControlledWorker)
        async with app.run_test() as pilot:
            worker = await running_app(app, pilot, config)
            worker.is_running = False
            worker.events = [
                {
                    "type": "failed",
                    "message": "Missing local checkpoint",
                    "log_tail": "[red]native warning[/red]",
                }
            ]
            await wait_for(pilot, lambda: worker.closed)
            assert isinstance(app.screen, RunningScreen)
            assert "Missing local checkpoint" in str(
                app.screen.query_one("#execution-stage", Label).render()
            )
            assert not app.screen.query_one("#run-back", Button).disabled
            assert not list(config.output_directory.glob("experiments/*"))
            assert "[red]native warning[/red]" in "\n".join(
                line.text for line in app.screen.query_one("#execution-log").lines
            )

    asyncio.run(exercise())


def test_start_failure_returns_control(config_factory):
    class FailedStart(ControlledWorker):
        def start(self):
            raise OSError("Process unavailable")

    async def exercise():
        config = load_config(config_factory())
        app = LabApp(config.output_directory, worker_factory=FailedStart)
        async with app.run_test() as pilot:
            app._configured(config)
            await wait_for(
                pilot, lambda: not app.screen.query_one("#run-back", Button).disabled
            )
            assert app.active_run is None
            assert "Process unavailable" in str(
                app.screen.query_one("#execution-stage", Label).render()
            )

    asyncio.run(exercise())
