"""Real spawn workers use disposable data and never load optional model runtimes."""

import json
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import tsfm_lab.execution as execution_module
from tsfm_lab.application import run_config, run_from_config
from tsfm_lab.config import load_config
from tsfm_lab.execution import ManagedRun
from tsfm_lab.experiments import ExperimentStore


def _record(config, events):
    store = ExperimentStore(config.output_directory)
    run_id = store.create(config.name, config.resolved_dict(), config.source_yaml)
    store.set_status(run_id, "running")
    events.send({"type": "stage", "stage": "running", "run_id": run_id})
    return run_id


def cooperative_worker(config, events, cancellation):
    run_id = None

    def created(value):
        nonlocal run_id
        run_id = value
        events.send({"type": "stage", "stage": "pending", "run_id": run_id})

    def stage(value):
        events.send({"type": "stage", "stage": value, "run_id": run_id})
        if value == "running":
            while not cancellation.is_set():
                time.sleep(0.005)

    try:
        run_config(
            config,
            on_created=created,
            progress=stage,
            should_cancel=cancellation.is_set,
        )
    except KeyboardInterrupt:
        events.send({"type": "cancelled", "run_id": run_id})


def unresponsive_worker(config, events, cancellation):
    if os.name != "nt":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    _record(config, events)
    time.sleep(60)


def crashing_worker(config, events, cancellation):
    _record(config, events)
    os._exit(7)


def crash_without_identity(config, events, cancellation):
    store = ExperimentStore(config.output_directory)
    store.create(config.name, config.resolved_dict(), config.source_yaml)
    os._exit(8)


def fake_completed_worker(config, events, cancellation):
    run_id = _record(config, events)
    events.send(
        {"type": "completed", "run_id": run_id, "result": {"status": "completed"}}
    )


def completion_race_worker(config, events, cancellation):
    run_id = None

    def created(value):
        nonlocal run_id
        run_id = value
        events.send({"type": "stage", "stage": "pending", "run_id": value})

    result = run_config(config, on_created=created)
    events.send({"type": "completed", "run_id": run_id, "result": result})
    events.send({"type": "stage", "stage": "settling", "run_id": run_id})
    while not cancellation.is_set():
        time.sleep(0.005)


def logging_worker(config, events, cancellation):
    _record(config, events)
    print("python child output", flush=True)
    os.write(2, b"native child warning\n")
    os._exit(9)


def collect(worker, until=None, timeout=10):
    events, deadline = [], time.monotonic() + timeout
    while time.monotonic() < deadline:
        batch = worker.poll()
        events.extend(batch)
        for event in batch:
            if event["type"] in {"completed", "failed", "cancelled"}:
                assert not worker._process.is_alive()
        if (until and until(events)) or (until is None and not worker.is_running):
            return events
        time.sleep(0.005)
    pytest.fail(f"Managed worker timed out; events: {events}")


def test_spawned_baseline_matches_shell_application_and_reports_monotonic_progress(
    config_factory,
):
    path = config_factory()
    expected = run_from_config(path)
    worker = ManagedRun(load_config(path))
    assert not worker.is_running and worker.run_id is None and worker.log_path is None
    try:
        worker.start()
        events = collect(worker)
        terminal = [event for event in events if event["type"] == "completed"]
        assert len(terminal) == 1
        assert terminal[0]["result"]["metrics"] == expected["metrics"]
        assert terminal[0]["result"]["counts"] == expected["counts"]
        assert terminal[0]["run_id"] == worker.run_id != expected["run_id"]
        stages = [event["stage"] for event in events if event["type"] == "stage"]
        assert stages == [
            "pending",
            "preparing",
            "loading",
            "running",
            "evaluating",
            "saving",
        ]
        progress = [event for event in events if event["type"] == "progress"]
        assert progress[0]["completed"] == 0
        assert progress[-1]["completed"] == progress[-1]["total"] == 4
        assert [event["completed"] for event in progress] == sorted(
            event["completed"] for event in progress
        )
        assert all(
            set(event) <= {"type", "completed", "total", "run_id"} for event in progress
        )
        assert worker.poll() == []
    finally:
        log = worker.log_path
        worker.close()
    assert not log.exists()


def test_cooperative_cancellation_retains_cancelled_record(config_factory):
    config = load_config(config_factory())
    worker = ManagedRun(config, _target=cooperative_worker)
    try:
        worker.start()
        before = collect(
            worker,
            until=lambda events: any(
                event.get("stage") == "running" for event in events
            ),
        )
        assert worker.run_id is not None
        worker.request_cancel()
        worker.request_cancel()
        after = collect(worker)
        assert sum(event["type"] == "cancellation_requested" for event in after) == 1
        assert [
            event["type"]
            for event in after
            if event["type"] in {"completed", "failed", "cancelled"}
        ] == ["cancelled"]
        assert all(event["type"] != "cancelled" for event in before)
        assert worker._terminated_at is None
        manifest = ExperimentStore(config.output_directory).get_manifest(worker.run_id)
        assert manifest["status"] == "cancelled"
        assert manifest["error"]["stage"] == "running"
    finally:
        worker.close()


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX child signal delivery; Windows console event is tested separately",
)
def test_worker_ignores_sigint_until_controller_requests_event_cancellation(
    config_factory,
):
    config = load_config(config_factory())
    worker = ManagedRun(config, _target=cooperative_worker)
    try:
        worker.start()
        collect(
            worker,
            until=lambda events: any(
                event.get("stage") == "running" for event in events
            ),
        )
        os.kill(worker._process.pid, signal.SIGINT)
        time.sleep(0.1)
        assert worker._process.is_alive()
        assert all(
            event["type"] not in {"completed", "cancelled", "failed"}
            for event in worker.poll()
        )
        assert (
            ExperimentStore(config.output_directory).get_manifest(worker.run_id)[
                "status"
            ]
            == "running"
        )
        worker.request_cancel()
        assert collect(worker)[-1]["type"] == "cancelled"
        assert (
            ExperimentStore(config.output_directory).get_manifest(worker.run_id)[
                "status"
            ]
            == "cancelled"
        )
    finally:
        worker.close()


def test_unresponsive_worker_is_forcibly_stopped_and_reaped_without_false_completion(
    config_factory,
):
    config = load_config(config_factory())
    worker = ManagedRun(config, grace_seconds=0.02, _target=unresponsive_worker)
    try:
        worker.start()
        collect(
            worker,
            until=lambda events: any(
                event.get("stage") == "running" for event in events
            ),
        )
        worker.request_cancel()
        events = collect(worker)
        assert worker._terminated_at is not None
        assert worker._process.exitcode != 0
        assert events[-1]["type"] == "cancelled"
        assert (
            ExperimentStore(config.output_directory).get_manifest(worker.run_id)[
                "status"
            ]
            == "cancelled"
        )
    finally:
        worker.close()


def test_crashed_child_becomes_failed_after_it_is_reaped(config_factory):
    config = load_config(config_factory())
    worker = ManagedRun(config, _target=crashing_worker)
    try:
        worker.start()
        events = collect(worker)
        assert events[-1]["type"] == "failed"
        assert "code 7" in events[-1]["message"]
        assert (
            ExperimentStore(config.output_directory).get_manifest(worker.run_id)[
                "status"
            ]
            == "failed"
        )
    finally:
        worker.close()


def test_unknown_crashed_run_identity_never_guesses_or_marks_completed(config_factory):
    config = load_config(config_factory())
    worker = ManagedRun(config, _target=crash_without_identity)
    try:
        worker.start()
        events = collect(worker)
        assert events[-1]["type"] == "failed" and events[-1]["run_id"] is None
        assert worker.run_id is None
        assert [
            record["status"]
            for record in ExperimentStore(config.output_directory).list_runs()
        ] == ["pending"]
    finally:
        worker.close()


def test_child_completion_message_cannot_complete_an_active_record(config_factory):
    config = load_config(config_factory())
    worker = ManagedRun(config, _target=fake_completed_worker)
    try:
        worker.start()
        events = collect(worker)
        assert events[-1]["type"] == "failed"
        assert (
            ExperimentStore(config.output_directory).get_manifest(worker.run_id)[
                "status"
            ]
            == "failed"
        )
    finally:
        worker.close()


def test_completion_wins_late_cancellation_without_overwriting_terminal_state(
    config_factory,
):
    config = load_config(config_factory())
    worker = ManagedRun(config, _target=completion_race_worker)
    try:
        worker.start()
        events = collect(
            worker,
            until=lambda events: any(
                event.get("stage") == "settling" for event in events
            ),
        )
        assert all(event["type"] != "completed" for event in events)
        worker.request_cancel()
        events = collect(worker)
        assert events[-1]["type"] == "completed"
        assert events[-1]["result"]["metrics"]["mae"]["value"] == 2.5
        assert (
            ExperimentStore(config.output_directory).get_manifest(worker.run_id)[
                "status"
            ]
            == "completed"
        )
    finally:
        worker.close()


def test_close_reaps_worker_and_returns_terminal_events(config_factory):
    worker = ManagedRun(
        load_config(config_factory()), grace_seconds=0.01, _target=unresponsive_worker
    )
    worker.start()
    collect(
        worker,
        until=lambda events: any(event.get("stage") == "running" for event in events),
    )
    pid, log = worker._process.pid, worker.log_path
    before = time.monotonic()
    events = worker.close()
    assert time.monotonic() - before < 2
    assert events[-1]["type"] == "cancelled"
    assert not worker.is_running and not log.exists()
    assert pid not in {child.pid for child in multiprocessing.active_children()}
    assert worker.close() == []


def test_child_python_and_native_output_are_isolated_from_parent_terminal(
    config_factory, capfd
):
    worker = ManagedRun(load_config(config_factory()), _target=logging_worker)
    try:
        worker.start()
        events = collect(worker)
        captured = capfd.readouterr()
        assert "python child output" not in captured.out + captured.err
        assert "native child warning" not in captured.out + captured.err
        text = worker.log_path.read_text()
        assert "python child output" in text and "native child warning" in text
        assert "native child warning" in events[-1]["log_tail"]
    finally:
        worker.close()


@pytest.mark.parametrize("grace", [-1, True, float("nan"), float("inf"), "3"])
def test_invalid_grace_is_rejected_without_starting_worker(config_factory, grace):
    with pytest.raises(ValueError, match="grace_seconds"):
        ManagedRun(load_config(config_factory()), grace)


def test_managed_run_cannot_start_twice(config_factory):
    worker = ManagedRun(load_config(config_factory()), grace_seconds=0.01)
    with pytest.raises(RuntimeError, match="Start"):
        worker.request_cancel()
    try:
        worker.start()
        with pytest.raises(RuntimeError, match="once"):
            worker.start()
    finally:
        worker.close()


def test_bad_preflight_has_no_workspace_and_no_false_completed_event(
    config_factory, tmp_path
):
    config = load_config(config_factory())
    (tmp_path / "seasonal.csv").unlink()
    worker = ManagedRun(config)
    try:
        worker.start()
        events = collect(worker)
        assert events[-1]["type"] == "failed" and events[-1]["run_id"] is None
        assert not config.output_directory.exists()
    finally:
        worker.close()


def test_immediate_cancellation_during_spawn_never_creates_a_record(config_factory):
    config = load_config(config_factory())
    worker = ManagedRun(config)
    try:
        worker.start()
        worker.request_cancel()
        events = collect(worker)
        assert events[-1]["type"] == "cancelled"
        assert events[-1]["run_id"] is None
        assert not config.output_directory.exists()
    finally:
        worker.close()


def test_poll_does_not_wait_for_metadata_reconciliation(config_factory, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = execution_module._reconcile

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=3)
        return original(*args, **kwargs)

    monkeypatch.setattr(execution_module, "_reconcile", blocked)
    worker = ManagedRun(load_config(config_factory()), _target=crashing_worker)
    try:
        worker.start()
        collect(worker, until=lambda events: entered.is_set())
        before = time.monotonic()
        assert worker.poll() == []
        assert time.monotonic() - before < 0.1
        assert worker.is_running
        release.set()
        assert collect(worker)[-1]["type"] == "failed"
    finally:
        release.set()
        worker.close()


def test_close_reports_a_completion_that_won_the_cancellation_race(config_factory):
    config = load_config(config_factory())
    worker = ManagedRun(config, _target=completion_race_worker)
    worker.start()
    collect(
        worker,
        until=lambda events: any(event.get("stage") == "settling" for event in events),
    )
    events = worker.close()
    assert events[-1]["type"] == "completed"
    assert (
        ExperimentStore(config.output_directory).get_manifest(worker.run_id)["status"]
        == "completed"
    )


def test_fresh_spawn_tracker_accepts_terminal_capture_with_invalid_fileno(
    config_factory,
):
    path = config_factory()
    script = """
import io, json, sys, time
from multiprocessing import resource_tracker
from tsfm_lab.config import load_config
from tsfm_lab.execution import ManagedRun
class Capture(io.StringIO):
    def fileno(self): return -1
assert resource_tracker._resource_tracker._fd is None
original_out, original_err = sys.stdout, sys.stderr
sys.stdout, sys.stderr = Capture(), Capture()
worker = ManagedRun(load_config(sys.argv[1]))
events = []
try:
    worker.start()
    deadline = time.monotonic() + 10
    while worker.is_running and time.monotonic() < deadline:
        events.extend(worker.poll())
        time.sleep(0.005)
    assert not worker.is_running
    print(json.dumps(events[-1]), file=original_out, flush=True)
except BaseException as exc:
    print(repr(exc), file=original_err, flush=True)
    raise
finally:
    worker.close()
"""
    source = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        env={**os.environ, "PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["type"] == "completed"


def test_process_creation_failure_cleans_temporary_log(config_factory, monkeypatch):
    import tsfm_lab.worker_entry as launcher

    def fail(*args, **kwargs):
        raise OSError("worker process unavailable")

    monkeypatch.setattr(launcher.subprocess, "Popen", fail)
    worker = ManagedRun(load_config(config_factory()))
    with pytest.raises(OSError, match="process unavailable"):
        worker.start()
    assert not worker.log_path.exists()
    assert not worker.is_running
    worker.close()
