"""Headless keyboard, validation, and responsive inspection of the experiment wizard."""

import asyncio
import threading

import pandas as pd
import pytest

pytest.importorskip("textual")

from textual.app import App
from textual.widgets import Button, Footer, Input, Select, Static

from tsfm_lab.config import ExperimentConfig
from tsfm_lab.tui import wizard as wizard_module
from tsfm_lab.tui.wizard import WizardScreen

_PENDING = object()


@pytest.fixture
def demo(tmp_path):
    path = tmp_path / "seasonal.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=32, freq="h", tz="UTC"),
            "demand": [10 + i // 4 + i % 4 for i in range(32)],
        }
    ).to_csv(path, index=False)
    return path


class WizardApp(App):
    def __init__(self, workspace, demo, initial_adapter="last_value"):
        super().__init__()
        self.wizard = WizardScreen(workspace, demo, initial_adapter)
        self.result = _PENDING

    def on_mount(self):
        self.push_screen(self.wizard, self.receive_result)

    def receive_result(self, result):
        self.result = result


async def settle(pilot, screen):
    await pilot.pause()
    for _ in range(100):
        if not screen.busy:
            await pilot.pause()
            return
        await asyncio.sleep(0.01)
        await pilot.pause()
    raise AssertionError("Wizard validation did not finish")


async def go_to_evaluation(app, pilot):
    screen = app.wizard
    await pilot.click("#wizard-next")
    assert screen.step == "data"
    await pilot.click("#wizard-use-demo")
    await settle(pilot, screen)
    await pilot.click("#wizard-next")
    await settle(pilot, screen)
    assert screen.step == "evaluation"


def test_initial_model_controls_and_keyboard_focus_fit_80_by_24(tmp_path, demo):
    async def exercise():
        app = WizardApp(tmp_path / "runs", demo)
        async with app.run_test(size=(80, 24)) as pilot:
            screen = app.wizard
            assert screen.step == "model"
            assert app.focused.id == "wizard-name"
            await pilot.press("tab")
            assert app.focused.id == "wizard-adapter"
            screen.query_one("#wizard-next", Button).focus()
            await pilot.press("enter")
            assert screen.step == "data"
            button = screen.query_one("#wizard-next", Button)
            assert 0 <= button.region.y < 24
            assert button.region.bottom <= screen.query_one(Footer).region.y

    asyncio.run(exercise())


def test_demo_returns_validated_config_without_yaml_or_result_writes(tmp_path, demo):
    async def exercise():
        app = WizardApp(tmp_path / "runs", demo)
        async with app.run_test(size=(80, 24)) as pilot:
            await go_to_evaluation(app, pilot)
            await pilot.click("#wizard-next")
            await settle(pilot, app.wizard)
            assert app.wizard.step == "review"
            assert not app.wizard.query_one("#wizard-run", Button).disabled
            assert "Forecast cases: 16" in str(
                app.wizard.query_one("#wizard-summary", Static).render()
            )
            app.wizard.query_one("#wizard-run", Button).focus()
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.result, ExperimentConfig)
            assert app.result.evaluation.context_length == 8
            assert app.result.evaluation.horizon == 4
            assert app.result.evaluation.stride == 4
            assert app.result.evaluation.evaluation_start == pd.Timestamp(
                "2026-01-01T16:00Z"
            )
            assert app.result.evaluation.evaluation_end == pd.Timestamp(
                "2026-01-02T08:00Z"
            )
            assert not app.result.source_path.exists()
            assert not (tmp_path / "runs").exists()

    asyncio.run(exercise())


def test_demo_dataset_preserves_name_model_and_explicit_evaluation_choices(
    tmp_path, demo
):
    async def exercise():
        app = WizardApp(tmp_path / "runs", demo, "seasonal_naive")
        async with app.run_test(size=(80, 24)) as pilot:
            screen = app.wizard
            choices = {
                "name": "Keep my experiment name",
                "seasonal-period": "24",
                "context-length": "24",
                "horizon": "4",
                "stride": "1",
                "mase-period": "24",
                "evaluation-start": "2026-01-02T00:00:00Z",
                "evaluation-end": "2026-01-02T08:00:00Z",
            }
            for name, value in choices.items():
                screen.query_one(f"#wizard-{name}", Input).value = value
            await pilot.pause()
            await pilot.click("#wizard-next")
            assert screen.step == "data"
            await pilot.click("#wizard-use-demo")
            await settle(pilot, screen)
            for name, value in choices.items():
                assert screen.query_one(f"#wizard-{name}", Input).value == value
            assert screen.query_one("#wizard-adapter", Select).value == "seasonal_naive"
            await pilot.click("#wizard-next")
            await settle(pilot, screen)
            await pilot.click("#wizard-next")
            await settle(pilot, screen)
            assert screen.step == "review"
            assert not screen.query_one("#wizard-run", Button).disabled
            screen.action_run()
            await pilot.pause()
            assert app.result.name == choices["name"]
            assert app.result.model["options"]["seasonal_period"] == 24
            assert app.result.evaluation.context_length == 24
            assert app.result.evaluation.stride == 1
            assert app.result.evaluation.seasonal_period == 24
            assert not (tmp_path / "runs").exists()

    asyncio.run(exercise())


def test_escape_cancels_without_submission(tmp_path, demo):
    async def exercise():
        app = WizardApp(tmp_path / "runs", demo)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("escape")
            await pilot.pause()
            assert app.result is None
            assert not (tmp_path / "runs").exists()

    asyncio.run(exercise())


def test_invalid_seasonal_period_blocks_model_step(tmp_path, demo):
    async def exercise():
        app = WizardApp(tmp_path / "runs", demo, "seasonal_naive")
        async with app.run_test(size=(80, 24)) as pilot:
            app.wizard.query_one("#wizard-seasonal-period", Input).value = "0"
            await pilot.click("#wizard-next")
            assert app.wizard.step == "model"
            assert "greater than zero" in str(
                app.wizard.query_one("#wizard-error", Static).render()
            )
            assert app.result is _PENDING

    asyncio.run(exercise())


def test_chronos_source_visibility_and_offline_sha_validation(tmp_path, demo):
    async def exercise():
        app = WizardApp(tmp_path / "runs", demo, "chronos_bolt")
        async with app.run_test(size=(80, 24)) as pilot:
            screen = app.wizard
            assert screen.query_one("#wizard-checkpoint-fields").display
            assert screen.query_one("#wizard-hub-fields").display
            screen.query_one("#wizard-revision", Input).value = "main"
            await pilot.click("#wizard-next")
            assert screen.step == "model"
            assert "40-character SHA" in str(
                screen.query_one("#wizard-error", Static).render()
            )
            screen.query_one("#wizard-source", Select).value = "local"
            await pilot.pause()
            assert screen.query_one("#wizard-local-fields").display
            assert not screen.query_one("#wizard-hub-fields").display
            screen.query_one("#wizard-checkpoint-path", Input).value = (
                "models/existing-checkpoint"
            )
            await pilot.pause(0.3)
            await pilot.click("#wizard-next")
            assert screen.step == "data"
            assert not (tmp_path / "runs").exists()

    asyncio.run(exercise())


def test_bad_dataset_stays_on_data_step_with_actionable_error(tmp_path, demo):
    async def exercise():
        app = WizardApp(tmp_path / "runs", demo)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.click("#wizard-next")
            app.wizard.query_one("#wizard-dataset-path", Input).value = str(
                tmp_path / "missing.csv"
            )
            await pilot.pause(0.3)
            await pilot.click("#wizard-next")
            await settle(pilot, app.wizard)
            assert app.wizard.step == "data"
            assert "missing.csv" in str(
                app.wizard.query_one("#wizard-error", Static).render()
            )
            assert app.result is _PENDING

    asyncio.run(exercise())


def test_invalid_evaluation_does_not_enable_run_or_repair_timestamps(tmp_path, demo):
    async def exercise():
        app = WizardApp(tmp_path / "runs", demo)
        async with app.run_test(size=(80, 24)) as pilot:
            await go_to_evaluation(app, pilot)
            start = app.wizard.query_one("#wizard-evaluation-start", Input)
            start.value = "2026-01-01T16:30:00Z"
            await pilot.click("#wizard-next")
            await settle(pilot, app.wizard)
            assert app.wizard.step == "evaluation"
            assert start.value == "2026-01-01T16:30:00Z"
            assert app.wizard.query_one("#wizard-run", Button).disabled
            assert "dataset grid" in str(
                app.wizard.query_one("#wizard-error", Static).render()
            )

    asyncio.run(exercise())


def test_back_preserves_draft_and_a_later_edit_invalidates_review(tmp_path, demo):
    async def exercise():
        app = WizardApp(tmp_path / "runs", demo)
        async with app.run_test(size=(80, 24)) as pilot:
            await go_to_evaluation(app, pilot)
            app.wizard.query_one("#wizard-stride", Input).value = "2"
            await pilot.click("#wizard-back")
            assert app.wizard.step == "data"
            await pilot.click("#wizard-next")
            await settle(pilot, app.wizard)
            assert app.wizard.query_one("#wizard-stride", Input).value == "2"
            await pilot.click("#wizard-next")
            await settle(pilot, app.wizard)
            assert app.wizard.step == "review"
            app.wizard.query_one("#wizard-name", Input).value = "Changed after review"
            await pilot.pause()
            assert app.wizard.query_one("#wizard-run", Button).disabled
            app.wizard.action_run()
            assert app.result is _PENDING

    asyncio.run(exercise())


def test_slow_inspection_keeps_cancellation_responsive(tmp_path, demo, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = wizard_module.inspect_data
    main_thread = threading.get_ident()

    def slow(*args, **kwargs):
        assert threading.get_ident() != main_thread
        entered.set()
        assert release.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(wizard_module, "inspect_data", slow)

    async def exercise():
        app = WizardApp(tmp_path / "runs", demo)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.click("#wizard-next")
            await pilot.click("#wizard-use-demo")
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(0.01)
            assert entered.is_set()
            assert app.wizard.busy
            await pilot.press("escape")
            await pilot.pause()
            assert app.result is None
            release.set()

    try:
        asyncio.run(exercise())
    finally:
        release.set()
