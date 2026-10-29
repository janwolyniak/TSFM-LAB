"""Real managed baseline workers settle preparation and core records together."""

import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from tsfm_lab.application import run_config
from tsfm_lab.config import load_config
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.guided import review_guided
from tsfm_lab.guided_execution import GuidedWorker, ManagedGuidedRun, PreparationRecord


def collect(worker, *, until=None, timeout=10):
    events, deadline = [], time.monotonic() + timeout
    while time.monotonic() < deadline:
        batch = worker.poll()
        events.extend(batch)
        if any(
            event["type"] in {"completed", "failed", "cancelled"} for event in batch
        ):
            assert not worker._inner._process.is_alive()
        if (until and until(events)) or (until is None and not worker.is_running):
            return events
        time.sleep(0.005)
    pytest.fail(f"Guided managed worker timed out: {events}")


def hanging_before_core(config, events, cancellation):
    if os.name != "nt":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    events.send({"type": "stage", "stage": "downloading", "run_id": None})
    while True:
        time.sleep(0.05)


def crash_during_core(config, events, cancellation):
    store = ExperimentStore(config.output_directory)
    run_id = store.create(config.name, config.resolved_dict(), config.source_yaml)
    store.set_status(run_id, "running")
    events.send({"type": "stage", "stage": "running", "run_id": run_id})
    os._exit(7)


def late_completion(config, events, cancellation):
    result = run_config(config, include_outputs=True)
    events.send({"type": "stage", "stage": "settling", "run_id": result["run_id"]})
    while not cancellation.is_set():
        time.sleep(0.005)


def legacy_completion(config, events, cancellation):
    result = run_config(config)
    events.send({"type": "completed", "run_id": result["run_id"], "result": result})


@dataclass
class LostIdentityWorker:
    plan: object
    operation_id: str
    status: str
    reported: str | None = None

    def __call__(self, config, events, cancellation):
        state = PreparationRecord(config, self.plan, self.operation_id)
        store = ExperimentStore(config.output_directory)
        run_id = store.create(config.name, config.resolved_dict(), config.source_yaml)
        state.link_run(run_id)
        store.set_status(run_id, self.status)
        events.send({"type": self.reported or self.status, "run_id": None})


def terminal(events):
    selected = [
        event
        for event in events
        if event["type"] in {"completed", "failed", "cancelled"}
    ]
    assert len(selected) == 1
    return selected[0]


def record(worker):
    return json.loads(worker._record.path.read_text())


def test_guided_spawn_reports_stages_outputs_and_reloads_without_model(config_factory):
    plan = review_guided(load_config(config_factory()))
    worker = ManagedGuidedRun(plan)
    assert not plan.config.output_directory.exists()
    try:
        worker.start()
        event = terminal(collect(worker))
        assert event["type"] == "completed"
        assert event["operation_id"] == worker.operation_id
        assert event["result"]["run_id"] == worker.run_id
        assert Path(event["preparation_path"]).is_file()
        assert {"forecast.svg", "provenance.json", "predictions.csv"} <= event[
            "result"
        ]["outputs"].keys()
        stages = [item["stage"] for item in record(worker)["history"]]
        assert stages == [
            "cache",
            "preparing",
            "loading",
            "running",
            "evaluating",
            "saving",
            "completed",
        ]
        assert record(worker)["run_id"] == worker.run_id
        assert worker.poll() == []
    finally:
        log_path = worker.log_path
        worker.close()
    assert not log_path.exists()
    result, predictions = ExperimentStore(plan.config.output_directory).load_result(
        event["run_id"]
    )
    assert len(predictions) == 16
    assert result["metrics"]["overall"]["mae"]["value"] == 2.5


def test_guided_cancel_before_core_leaves_no_orphan_or_completed_result(config_factory):
    plan = review_guided(load_config(config_factory()))
    worker = ManagedGuidedRun(plan)
    try:
        worker.start()
        worker.request_cancel()
        event = terminal(collect(worker))
        assert event["type"] == "cancelled"
        assert record(worker)["status"] == "cancelled"
        assert not (plan.config.output_directory / "experiments").exists()
    finally:
        worker.close()
    assert worker._inner._closed


def test_guided_worker_refuses_stale_review_without_creating_core_record(
    config_factory,
):
    config = load_config(config_factory())
    plan = review_guided(config)
    source = Path(config.dataset["path"])
    source.write_text(source.read_text().replace("16.0", "16.5", 1))
    # A direct value replacement independent of CSV float formatting.
    lines = source.read_text().splitlines()
    columns = lines[1].split(",")
    columns[-1] = str(float(columns[-1]) + 1)
    lines[1] = ",".join(columns)
    source.write_text("\n".join(lines) + "\n")
    worker = ManagedGuidedRun(plan)
    try:
        worker.start()
        event = terminal(collect(worker))
        assert event["type"] == "failed"
        assert "changed" in event["message"]
        assert record(worker)["status"] == "failed"
        assert not (config.output_directory / "experiments").exists()
    finally:
        worker.close()


def test_forced_stop_before_core_settles_preparation_as_cancelled(config_factory):
    worker = ManagedGuidedRun(
        review_guided(load_config(config_factory())),
        grace_seconds=0.02,
        _target=hanging_before_core,
    )
    try:
        worker.start()
        collect(
            worker,
            until=lambda events: any(
                item.get("stage") == "downloading" for item in events
            ),
        )
        worker.request_cancel()
        event = terminal(collect(worker))
        assert event["type"] == "cancelled"
        assert record(worker)["status"] == "cancelled"
        assert not worker._inner._process.is_alive()
    finally:
        worker.close()


def test_crashed_core_and_preparation_both_remain_failed(config_factory):
    plan = review_guided(load_config(config_factory()))
    worker = ManagedGuidedRun(plan, _target=crash_during_core)
    try:
        worker.start()
        event = terminal(collect(worker))
        assert event["type"] == "failed"
        assert record(worker)["status"] == "failed"
        assert (
            ExperimentStore(plan.config.output_directory).get_manifest(event["run_id"])[
                "status"
            ]
            == "failed"
        )
    finally:
        worker.close()


def test_completed_outputs_win_late_cancellation_after_worker_stops(config_factory):
    plan = review_guided(load_config(config_factory()))
    worker = ManagedGuidedRun(plan, _target=late_completion)
    try:
        worker.start()
        collect(
            worker,
            until=lambda events: any(
                item.get("stage") == "settling" for item in events
            ),
        )
        event = terminal(worker.close())
        assert event["type"] == "completed"
        assert "forecast.svg" in event["result"]["outputs"]
        assert record(worker)["status"] == "completed"
    finally:
        worker.close()


def test_legacy_completion_cannot_claim_guided_output_completion(config_factory):
    worker = ManagedGuidedRun(
        review_guided(load_config(config_factory())), _target=legacy_completion
    )
    try:
        worker.start()
        event = terminal(collect(worker))
        assert event["type"] == "failed"
        assert "complete output contract" in event["message"]
        assert record(worker)["status"] == "failed"
    finally:
        worker.close()


def test_byte_progress_persistence_is_throttled_and_terminal_keeps_latest(
    config_factory, monkeypatch
):
    plan = review_guided(load_config(config_factory()))
    clock, saves = [0.0], []
    monkeypatch.setattr("tsfm_lab.guided_execution.monotonic", lambda: clock[0])
    monkeypatch.setattr(
        "tsfm_lab.guided_execution._atomic",
        lambda path, raw: saves.append(json.loads(raw)),
    )
    state = PreparationRecord(plan.config, plan)
    state.update({"stage": "verifying"})
    for value in range(1, 101):
        clock[0] += 0.001
        state.update(
            {"type": "transfer", "completed_bytes": value, "total_bytes": None}
        )
    assert len(saves) == 2
    assert state.data["completed_bytes"] == 100
    clock[0] = 0.3
    state.update({"completed_bytes": 101})
    assert len(saves) == 3
    state.update({"completed_bytes": 102})
    state.finish("cancelled")
    assert len(saves) == 4
    assert saves[-1]["completed_bytes"] == 102
    assert saves[-1]["status"] == "cancelled"


def test_worker_throttles_transfer_ipc_and_flushes_latest_before_terminal(monkeypatch):
    clock, emitted = [0.0], []
    monkeypatch.setattr("tsfm_lab.guided_execution.monotonic", lambda: clock[0])

    def execute(plan, *, progress, **kwargs):
        progress({"type": "stage", "stage": "verifying"})
        for value in range(100):
            clock[0] += 0.001
            progress(
                {"type": "transfer", "completed_bytes": value, "total_bytes": None}
            )
        raise KeyboardInterrupt

    monkeypatch.setattr("tsfm_lab.guided.execute_guided", execute)
    worker = GuidedWorker(None, "fake-operation")
    worker(
        None,
        SimpleNamespace(send=emitted.append),
        SimpleNamespace(is_set=lambda: False),
    )
    transfers = [item for item in emitted if item["type"] == "transfer"]
    assert 2 <= len(transfers) <= 4
    assert transfers[-1]["completed_bytes"] == 99
    assert emitted[-1]["type"] == "cancelled"


@pytest.mark.parametrize(
    "status,reported",
    [
        ("failed", "failed"),
        ("cancelled", "cancelled"),
        ("failed", "cancelled"),
        ("cancelled", "failed"),
    ],
)
def test_terminal_recovers_run_link_when_created_ipc_event_was_lost(
    config_factory, status, reported
):
    plan = review_guided(load_config(config_factory()))
    worker = ManagedGuidedRun(plan)
    worker._inner._target = LostIdentityWorker(
        plan, worker.operation_id, status, reported
    )
    try:
        worker.start()
        event = terminal(collect(worker))
        assert event["type"] == status
        assert event["run_id"] == record(worker)["run_id"] == worker.run_id
        assert event["run_id"] is not None
        assert (
            ExperimentStore(plan.config.output_directory).get_manifest(event["run_id"])[
                "status"
            ]
            == status
        )
    finally:
        worker.close()
