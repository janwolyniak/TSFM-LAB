"""Headless navigation over real disposable experiment records; no inference in views."""

import asyncio
import threading

import pandas as pd
import pytest

pytest.importorskip("textual")

from textual.app import App  # noqa: E402
from textual.widgets import Button, DataTable, Footer, Input, Static  # noqa: E402

from tsfm_lab.application import run_from_config  # noqa: E402
from tsfm_lab.experiments import ExperimentStore  # noqa: E402
from tsfm_lab.tui import views  # noqa: E402
from tsfm_lab.tui.view_data import preview_rows  # noqa: E402
from tsfm_lab.tui.views import (  # noqa: E402
    ComparisonScreen,
    HistoryScreen,
    ModelScreen,
    ResultsScreen,
)


class ViewApp(App):
    def __init__(self, screen):
        super().__init__()
        self.view = screen
        self.returned = "pending"

    def on_mount(self):
        self.push_screen(self.view, self._returned)

    def _returned(self, result):
        self.returned = result


async def settle(pilot, condition):
    for _ in range(100):
        await pilot.pause(0.02)
        if condition():
            return
    raise AssertionError("Screen did not finish its background operation")


def text(screen, selector):
    return str(screen.query_one(selector, Static).render())


def test_model_browser_keyboard_selection_and_literal_details():
    async def exercise():
        screen = ModelScreen()
        app = ViewApp(screen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            table = screen.query_one("#model-table", DataTable)
            assert table.row_count == 5
            await pilot.press("down")
            assert "complete season" in text(screen, "#model-details")
            await pilot.press("enter")
            await pilot.pause()
            assert app.returned == "seasonal_naive"

    asyncio.run(exercise())


def test_chronos_browser_explains_local_requirements_without_loading(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Browsing must not load or download a checkpoint")

    monkeypatch.setattr("tsfm_lab.models.factory.build_model", forbidden)

    async def exercise():
        screen = ModelScreen()
        app = ViewApp(screen)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("down", "down")
            assert "cached or local checkpoint" in text(screen, "#model-details")
            assert "apache-2.0" in text(screen, "#model-details")
            back = screen.query_one("#view-back", Button)
            assert back.region.y >= 0
            assert back.region.bottom <= screen.query_one(Footer).region.y
            await pilot.press("escape")
            assert app.returned is None

    asyncio.run(exercise())


def test_model_catalog_error_is_visible_and_selection_is_disabled(monkeypatch):
    def broken():
        raise OSError("synthetic catalog error")

    monkeypatch.setattr(views, "catalog_entries", broken)

    async def exercise():
        screen = ModelScreen()
        async with ViewApp(screen).run_test() as pilot:
            await pilot.pause()
            assert "synthetic catalog error" in text(screen, "#model-details")
            assert screen.query_one("#use-model", Button).disabled

    asyncio.run(exercise())


def test_history_keyboard_selection_comparison_and_back_preserve_cursor(
    config_factory, tmp_path
):
    run_from_config(config_factory())
    run_from_config(config_factory("seasonal_naive"))

    async def exercise():
        screen = HistoryScreen(tmp_path / "workspace")
        app = ViewApp(screen)
        async with app.run_test(size=(80, 24)) as pilot:
            await settle(pilot, lambda: len(screen.records) == 2)
            await pilot.press("space", "down", "space")
            assert len(screen.selected) == 2
            await pilot.press("c")
            await settle(
                pilot,
                lambda: isinstance(app.screen, ComparisonScreen)
                and app.screen.query_one("#comparison-table", DataTable).row_count == 2,
            )
            assert "Matching forecast tasks" in text(app.screen, "#comparison-status")
            assert "Units: synthetic_units" in text(app.screen, "#comparison-summary")
            await pilot.press("escape")
            assert app.screen is screen
            assert screen.query_one("#history-table", DataTable).cursor_row == 1
            await pilot.press("r")
            await settle(pilot, lambda: not screen.selected)
            assert screen.query_one("#history-compare", Button).disabled

    asyncio.run(exercise())


def test_history_enter_opens_result_and_escape_returns(config_factory, tmp_path):
    value = run_from_config(config_factory())

    async def exercise():
        screen = HistoryScreen(tmp_path / "workspace")
        app = ViewApp(screen)
        async with app.run_test(size=(80, 24)) as pilot:
            await settle(pilot, lambda: bool(screen.records))
            await pilot.press("enter")
            await settle(
                pilot,
                lambda: isinstance(app.screen, ResultsScreen) and app.screen.completed,
            )
            assert app.screen.run_id == value["run_id"]
            await pilot.press("escape")
            assert app.screen is screen

    asyncio.run(exercise())


def test_completed_result_shows_task_before_metrics_and_exports_without_inference(
    config_factory, tmp_path, monkeypatch
):
    value = run_from_config(config_factory())

    def forbidden(*args, **kwargs):
        raise AssertionError("Viewing and export must not execute inference")

    monkeypatch.setattr("tsfm_lab.application.run_experiment", forbidden)
    destination = tmp_path / "forecasts.csv"

    async def exercise():
        screen = ResultsScreen(tmp_path / "workspace", value["run_id"])
        async with ViewApp(screen).run_test(size=(80, 24)) as pilot:
            await settle(pilot, lambda: screen.completed)
            summary = text(screen, "#result-summary")
            assert "Target: demand" in summary
            assert "Units: synthetic_units" in summary
            assert "Horizon: 4 observations" in summary
            assert screen.query_one("#overall-metrics", DataTable).row_count == 3
            assert screen.query_one("#lead-metrics", DataTable).row_count == 4
            assert screen.query_one("#forecast-preview", DataTable).row_count == 12
            screen.query_one("#export-path", Input).value = str(destination)
            await pilot.press("ctrl+e")
            await settle(pilot, lambda: destination.exists())
            await settle(
                pilot, lambda: not screen.query_one("#export-button", Button).disabled
            )
            original = destination.read_bytes()
            await pilot.press("ctrl+e")
            await settle(
                pilot, lambda: "already exists" in text(screen, "#export-status")
            )
            assert destination.read_bytes() == original
            back = screen.query_one("#view-back", Button)
            assert back.region.bottom <= screen.query_one(Footer).region.y

    asyncio.run(exercise())


def test_failed_and_corrupt_results_never_display_completed_metrics(
    config_factory, tmp_path
):
    completed = run_from_config(config_factory())
    workspace = tmp_path / "workspace"
    store = ExperimentStore(workspace)
    failed = store.create("failed run", {}, "")
    store.set_status(
        failed, "failed", {"message": "synthetic model failure", "stage": "loading"}
    )
    predictions = workspace / "experiments" / completed["run_id"] / "predictions.csv"
    predictions.write_text("corrupted artifact")

    async def exercise():
        for run_id, message in [
            (failed, "synthetic model failure"),
            (completed["run_id"], "Cannot show verified results"),
        ]:
            screen = ResultsScreen(workspace, run_id)
            async with ViewApp(screen).run_test(size=(80, 24)) as pilot:
                await settle(
                    pilot,
                    lambda expected=message, view=screen: expected
                    in text(view, "#result-status"),
                )
                assert not screen.completed
                assert not screen.query_one("#completed-result").display
                assert screen.query_one("#overall-metrics", DataTable).row_count == 0
                assert screen.query_one("#export-button", Button).disabled

    asyncio.run(exercise())


def test_failed_history_entry_cannot_be_selected_for_comparison(tmp_path):
    store = ExperimentStore(tmp_path / "workspace")
    run_id = store.create("failed", {}, "")
    store.set_status(run_id, "failed")

    async def exercise():
        screen = HistoryScreen(tmp_path / "workspace")
        async with ViewApp(screen).run_test() as pilot:
            await settle(pilot, lambda: bool(screen.records))
            await pilot.press("space", "c")
            assert not screen.selected
            assert "completed experiments" in text(screen, "#history-status")

    asyncio.run(exercise())


def test_incompatible_comparison_is_an_error_view(config_factory, tmp_path):
    a = run_from_config(config_factory())
    b = run_from_config(config_factory(evaluation={"context_length": 9}))

    async def exercise():
        screen = ComparisonScreen(tmp_path / "workspace", [a["run_id"], b["run_id"]])
        async with ViewApp(screen).run_test(size=(80, 24)) as pilot:
            await settle(
                pilot, lambda: "different" in text(screen, "#comparison-status")
            )
            assert screen.query_one("#comparison-table", DataTable).row_count == 0
            assert "Comparison unavailable" in text(screen, "#comparison-summary")

    asyncio.run(exercise())


def test_empty_history_is_read_only_and_errors_are_visible(tmp_path, monkeypatch):
    workspace = tmp_path / "absent"

    async def empty():
        screen = HistoryScreen(workspace)
        async with ViewApp(screen).run_test() as pilot:
            await settle(pilot, lambda: "0 saved" in text(screen, "#history-status"))
            assert not workspace.exists()
            assert screen.query_one("#history-view", Button).disabled

    asyncio.run(empty())

    class BrokenStore:
        def __init__(self, *args):
            pass

        def list_runs(self):
            raise OSError("synthetic history failure")

    monkeypatch.setattr(views, "ExperimentStore", BrokenStore)

    async def broken():
        screen = HistoryScreen(workspace)
        async with ViewApp(screen).run_test() as pilot:
            await settle(
                pilot,
                lambda: "synthetic history failure" in text(screen, "#history-status"),
            )

    asyncio.run(broken())


def test_slow_history_read_runs_off_ui_thread_and_back_stays_responsive(
    tmp_path, monkeypatch
):
    started, release = threading.Event(), threading.Event()
    thread_ids = []

    class SlowStore:
        def __init__(self, *args):
            pass

        def list_runs(self):
            thread_ids.append(threading.get_ident())
            started.set()
            release.wait(timeout=3)
            return []

    monkeypatch.setattr(views, "ExperimentStore", SlowStore)

    async def exercise():
        app = ViewApp(HistoryScreen(tmp_path))
        async with app.run_test(size=(80, 24)) as pilot:
            try:
                await settle(pilot, started.is_set)
                assert thread_ids[0] != threading.get_ident()
                await pilot.press("escape")
                assert app.returned is None
            finally:
                release.set()

    asyncio.run(exercise())


def test_undefined_mase_reason_is_visible_in_the_metric_table(config_factory, tmp_path):
    path = config_factory()
    source = tmp_path / "seasonal.csv"
    frame = pd.read_csv(source)
    frame["demand"] = 5
    frame.to_csv(source, index=False)
    value = run_from_config(path)

    async def exercise():
        screen = ResultsScreen(tmp_path / "workspace", value["run_id"])
        async with ViewApp(screen).run_test(size=(80, 24)) as pilot:
            await settle(pilot, lambda: screen.completed)
            table = screen.query_one("#overall-metrics", DataTable)
            cells = [
                str(cell)
                for row in range(table.row_count)
                for cell in table.get_row_at(row)
            ]
            assert "undefined (zero scale)" in cells

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "origin,target",
    [
        ("2026-01-01T00:00:01+00:00", "2026-01-01T00:00:02+00:00"),
        (
            "2026-01-01T00:00:00.000000001+00:00",
            "2026-01-01T00:00:00.000000002+00:00",
        ),
        ("2025-12-31T23:59:59+00:00", "2026-01-01T00:00:00+00:00"),
    ],
)
def test_preview_preserves_distinct_timestamps_year_and_precision(origin, target):
    frame = pd.DataFrame(
        [
            {
                "origin": pd.Timestamp(origin),
                "target_timestamp": pd.Timestamp(target),
                "lead": 1,
                "actual": 1.0,
                "prediction": 2.0,
            }
        ]
    )
    row = preview_rows(frame)[0]
    assert row[:2] == (origin, target)
    assert row[0] != row[1]
