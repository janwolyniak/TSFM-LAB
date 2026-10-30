"""Offline future workflows, strict historical refusal, and managed lifecycle."""

import asyncio
import io
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from tsfm_lab.cli import app
from tsfm_lab.comparison import compare_runs
from tsfm_lab.config import ConfigError, load_config
from tsfm_lab.execution import ManagedRun
from tsfm_lab.experiments import ExperimentStore, StoreError
from tsfm_lab.exports import export_predictions
from tsfm_lab.future_application import run_future_config
from tsfm_lab.future_config import future_config_from_mapping, load_future_config
from tsfm_lab.guided import execute_guided, review_guided
from tsfm_lab.guided_execution import ManagedGuidedRun, PreparationRecord
from tsfm_lab.models.baselines import LastValue
from tsfm_lab.repl.records import RecordCommands


def test_historical_export_retains_core_only_optional_import_boundary(tmp_path):
    root = Path(__file__).resolve().parents[1]
    program = """
import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class CoreOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'yaml','typer','rich','torch','httpx'}:
            raise ModuleNotFoundError('Optional import attempted: ' + fullname, name=fullname)
sys.meta_path.insert(0, CoreOnly())
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.exports import export_predictions
run = ExperimentStore(sys.argv[2]).list_runs()[0]['run_id']
assert export_predictions(run, sys.argv[2], sys.argv[3])['rows'] > 0
"""
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            program,
            str(root / "src"),
            str(root / "tests/fixtures/release_legacy_v1/workspace"),
            str(tmp_path / "legacy.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )


@pytest.fixture
def future_path(config_factory):
    path = config_factory()
    mapping = yaml.safe_load(path.read_text())
    mapping.pop("evaluation")
    mapping.update(
        schema_version=2,
        mode="future_forecast",
        forecast={
            "context_length": 8,
            "horizon": 4,
            "origin": "latest",
        },
    )
    path.write_text(yaml.safe_dump(mapping))
    return path


def collect(worker, timeout=10, until=None):
    events, deadline = [], time.monotonic() + timeout
    while time.monotonic() < deadline:
        events.extend(worker.poll())
        if (until and until(events)) or not worker.is_running:
            return events
        time.sleep(0.005)
    pytest.fail(f"Future worker timed out: {events}")


def future_crash(config, events, cancellation):
    from tsfm_lab.application import load_dataset
    from tsfm_lab.future import resolve_future
    from tsfm_lab.future_store import FutureStore

    resolved = resolve_future(config, load_dataset(config.dataset))
    store = FutureStore(config.output_directory)
    run_id = store.create(resolved.resolved_dict(), config.source_yaml)
    events.send({"type": "stage", "stage": "running", "run_id": run_id})
    os._exit(7)


def future_late_completion(config, events, cancellation):
    result = run_future_config(config)
    events.send({"type": "stage", "stage": "settling", "run_id": result["run_id"]})
    while not cancellation.is_set():
        time.sleep(0.005)


def future_wrong_configuration(config, events, cancellation):
    fields = config.resolved_dict()
    fields["name"] += " different request"
    changed = future_config_from_mapping(
        fields, base_directory=config.source_path.parent
    )
    result = run_future_config(changed)
    events.send({"type": "completed", "run_id": result["run_id"], "result": result})


@dataclass
class FutureLostIdentity:
    plan: object
    operation_id: str
    status: str

    def __call__(self, config, events, cancellation):
        from tsfm_lab.future_store import FutureStore

        record = PreparationRecord(config, self.plan, self.operation_id)
        store = FutureStore(config.output_directory)
        run_id = store.create(config.resolved_dict(), config.source_yaml)
        record.link_run(run_id)
        store.set_status(run_id, self.status)
        events.send({"type": self.status, "run_id": None})


def test_offline_future_run_has_eight_checked_outputs_and_no_scores(future_path):
    config = load_future_config(future_path)
    stages, windows = [], []
    summary = run_future_config(
        config,
        progress=stages.append,
        window_progress=lambda done, total: windows.append((done, total)),
    )
    store = ExperimentStore(config.output_directory)
    manifest = store.get_manifest(summary["run_id"])
    result, predictions, observations = store.load_future_result(summary["run_id"])
    assert manifest["schema_version"] == 3 and manifest["kind"] == "future_forecast"
    assert len(manifest["artifacts"]) == 8
    assert stages == ["preparing", "loading", "running", "saving"]
    assert windows == [(0, 1), (1, 1)]
    assert len(observations) == 8 and len(predictions) == 4
    assert (predictions.prediction == observations.observed_value.iloc[-1]).all()
    assert predictions.target_timestamp.min() > observations.timestamp.max()
    assert result["evaluation"]["reason"] == "future_actuals_absent"
    assert "metrics" not in result and "metrics" not in summary
    assert "actual" not in predictions
    with pytest.raises(StoreError, match="[Ff]uture"):
        store.load_result(summary["run_id"])
    other = run_future_config(config)
    with pytest.raises(ValueError, match="[Ff]uture"):
        compare_runs([summary["run_id"], other["run_id"]], config.output_directory)


@pytest.mark.parametrize("format", ["csv", "json", "parquet"])
def test_future_exports_and_record_views_reload_without_source_or_model(
    future_path, tmp_path, monkeypatch, format
):
    if format == "parquet":
        pytest.importorskip("pyarrow")
    config = load_future_config(future_path)
    summary = run_future_config(config)
    Path(config.dataset["path"]).rename(tmp_path / "source-now-unavailable.csv")
    monkeypatch.setattr(
        "tsfm_lab.future_application.build_model",
        lambda *a: pytest.fail("Reload invoked model"),
    )
    target = tmp_path / f"future.{format}"
    exported = export_predictions(summary["run_id"], config.output_directory, target)
    assert exported["rows"] == 4
    output = io.StringIO()
    records = RecordCommands(Console(file=output, width=100), config.output_directory)
    records.history()
    records.show(summary["run_id"])
    assert "Future forecast" in output.getvalue()
    assert "Accuracy unavailable" in output.getvalue()
    assert "Actual" not in output.getvalue()


@pytest.mark.parametrize("json_output", [False, True])
def test_future_script_and_show_commands(future_path, json_output):
    runner = CliRunner()
    run = runner.invoke(
        app, ["run", str(future_path), *(["--json"] if json_output else [])]
    )
    assert run.exit_code == 0, run.output
    config = load_future_config(future_path)
    run_id = ExperimentStore(config.output_directory).list_runs()[0]["run_id"]
    shown = runner.invoke(
        app,
        [
            "experiments",
            "show",
            run_id,
            "--workspace",
            str(config.output_directory),
            *(["--json"] if json_output else []),
        ],
    )
    assert shown.exit_code == 0, shown.output
    if json_output:
        assert json.loads(run.stdout)["mode"] == "future_forecast"
        assert (
            json.loads(shown.stdout)["result"]["evaluation"]["status"] == "unavailable"
        )
    else:
        assert (
            "Accuracy unavailable" in run.stdout and "Future forecast" in shown.stdout
        )


@pytest.mark.parametrize("stage", ["preparing", "loading", "running", "saving"])
def test_future_cancellation_leaves_identifiable_incomplete_record(future_path, stage):
    config = load_future_config(future_path)
    cancelled = False

    def progress(value):
        nonlocal cancelled
        cancelled = value == stage

    with pytest.raises(KeyboardInterrupt):
        run_future_config(config, progress=progress, should_cancel=lambda: cancelled)
    manifest = ExperimentStore(config.output_directory).list_runs()[0]
    assert manifest["kind"] == "future_forecast" and manifest["status"] == "cancelled"
    with pytest.raises(ValueError):
        ExperimentStore(config.output_directory).load_future_result(manifest["run_id"])


def test_future_adapter_failure_records_failed_status(future_path, monkeypatch):
    class Broken(LastValue):
        def predict(self, requests):
            raise RuntimeError("fake future inference failure")

    monkeypatch.setattr("tsfm_lab.future_application.build_model", lambda *a: Broken())
    config = load_future_config(future_path)
    with pytest.raises(RuntimeError, match="fake future inference failure"):
        run_future_config(config)
    assert ExperimentStore(config.output_directory).list_runs()[0]["status"] == "failed"


def test_guided_future_plan_rejects_changed_raw_bytes_before_model(
    future_path, monkeypatch
):
    config = load_future_config(future_path)
    plan = review_guided(config)
    assert plan.config.forecast.origin != "latest"
    source = Path(config.dataset["path"])
    rows = source.read_text().splitlines()
    timestamp, number = rows[1].split(",")
    rows[1] = f"{timestamp},{float(number):.15f}"
    source.write_text("\n".join(rows) + "\n")
    monkeypatch.setattr(
        "tsfm_lab.guided._runtime_imports",
        lambda *a: pytest.fail("Runtime preceded data review"),
    )
    with pytest.raises(ValueError):
        execute_guided(plan)
    prep = next((config.output_directory / "preparations").glob("*.json"))
    data = PreparationRecord.load(prep).data
    assert data["schema_version"] == 2 and data["mode"] == "future_forecast"
    assert data["status"] == "failed"


def test_historical_python_loader_keeps_explicit_future_boundary(future_path):
    with pytest.raises(ConfigError, match="[Ff]uture"):
        load_config(future_path)


def test_future_finance_refuses_before_accounting(tmp_path, monkeypatch):
    from finance_helpers import price_fixture

    from tsfm_lab.finance.backtest import backtest_run

    prices = price_fixture(tmp_path)
    fields = yaml.safe_load(prices.source_config.read_text())
    fields.pop("evaluation")
    fields.update(
        schema_version=2,
        mode="future_forecast",
        forecast={"context_length": 2, "horizon": 1},
        output={"directory": str(prices.workspace)},
    )
    config = future_config_from_mapping(fields, base_directory=tmp_path)
    summary = run_future_config(config)
    monkeypatch.setattr(
        "tsfm_lab.finance.backtest.evaluate_cases",
        lambda *a, **kw: pytest.fail("Future record entered accounting"),
    )
    with pytest.raises(StoreError, match="[Ff]uture"):
        backtest_run(
            summary["run_id"],
            prices.workspace,
            prices.dataset,
            prices.ohlc,
            prices.rules,
            target_is_price=True,
        )


def test_fake_timesfm_future_saves_full_native_provenance(
    future_path, timesfm_checkpoint, timesfm_runtime
):
    config = load_future_config(future_path)
    fields = config.resolved_dict()
    fields["model"] = {
        "source": "local",
        "adapter_id": "timesfm_2p5",
        "path": str(timesfm_checkpoint),
    }
    fields["runtime"]["dtype"] = "float32"
    config = future_config_from_mapping(fields, base_directory=future_path.parent)
    summary = run_future_config(config)
    result, predictions, _ = ExperimentStore(
        config.output_directory
    ).load_future_result(summary["run_id"])
    assert result["output"]["point_statistic"] == "median"
    assert (
        result["provenance"]["model"]["options"]["checkpoint"]["architecture"]
        == "TimesFmModelForPrediction"
    )
    assert result["provenance"]["runtime"]["forecast_options"]
    assert len(predictions) == 4 and timesfm_runtime.predictions


@pytest.mark.parametrize("guided", [False, True])
def test_real_future_managed_worker_publishes_checked_completion(future_path, guided):
    config = load_future_config(future_path)
    worker = ManagedGuidedRun(review_guided(config)) if guided else ManagedRun(config)
    try:
        worker.start()
        events = collect(worker)
        terminals = [
            e for e in events if e["type"] in {"completed", "failed", "cancelled"}
        ]
        assert len(terminals) == 1 and terminals[0]["type"] == "completed", events
        result = terminals[0]["result"]
        assert result["mode"] == "future_forecast" and "metrics" not in result
        ExperimentStore(config.output_directory).load_future_result(result["run_id"])
        if guided:
            prep = PreparationRecord.load(worker._record.path).data
            assert prep["schema_version"] == 2 and prep["status"] == "completed"
            assert prep["plan_identity"] == worker.plan.plan_identity
    finally:
        worker.close()


def test_crashed_future_worker_is_failed_not_historical(future_path):
    config = load_future_config(future_path)
    worker = ManagedRun(config, _target=future_crash)
    try:
        worker.start()
        events = collect(worker)
        assert events[-1]["type"] == "failed"
        manifest = ExperimentStore(config.output_directory).get_manifest(worker.run_id)
        assert manifest["schema_version"] == 3 and manifest["status"] == "failed"
    finally:
        worker.close()


def test_late_future_completion_survives_cancel(future_path):
    config = load_future_config(future_path)
    worker = ManagedRun(config, grace_seconds=0.2, _target=future_late_completion)
    try:
        worker.start()
        collect(
            worker,
            until=lambda events: any(e.get("stage") == "settling" for e in events),
        )
        worker.request_cancel()
        events = collect(worker)
        assert events[-1]["type"] == "completed", events
    finally:
        worker.close()


@pytest.mark.parametrize("guided", [False, True])
def test_completion_for_another_configuration_is_not_accepted(future_path, guided):
    config = load_future_config(future_path)
    worker = (
        ManagedGuidedRun(review_guided(config), _target=future_wrong_configuration)
        if guided
        else ManagedRun(config, _target=future_wrong_configuration)
    )
    try:
        worker.start()
        events = collect(worker)
        assert events[-1]["type"] == "failed", events
        assert (
            "configuration" in events[-1]["message"]
            or "reviewed plan" in events[-1]["message"]
        )
    finally:
        worker.close()


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_future_guided_ipc_loss_recovers_linked_run(future_path, status):
    config = load_future_config(future_path)
    plan = review_guided(config)
    worker = ManagedGuidedRun(plan)
    worker._inner._target = FutureLostIdentity(plan, worker.operation_id, status)
    try:
        worker.start()
        events = collect(worker)
        assert events[-1]["type"] == status, events
        assert events[-1]["run_id"] == worker.run_id
        assert (
            ExperimentStore(config.output_directory).get_manifest(worker.run_id)[
                "status"
            ]
            == status
        )
    finally:
        worker.close()


def test_future_results_screen_has_no_actual_or_metric_tables(future_path):
    pytest.importorskip("textual")
    from test_tui_views import ViewApp, settle, text
    from textual.widgets import DataTable

    from tsfm_lab.tui.views import ResultsScreen

    config = load_future_config(future_path)
    result = run_future_config(config)

    async def exercise():
        screen = ResultsScreen(config.output_directory, result["run_id"])
        async with ViewApp(screen).run_test(size=(80, 24)) as pilot:
            await settle(pilot, lambda: screen.completed)
            assert "Accuracy unavailable" in text(screen, "#result-summary")
            assert not screen.query_one("#overall-metrics").display
            assert screen.query_one("#forecast-preview", DataTable).row_count == 4

    asyncio.run(exercise())
