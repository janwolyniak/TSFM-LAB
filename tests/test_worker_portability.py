"""Bounded real stdio probes distinguish framing, import and shutdown failures."""

import hashlib
import json
import os
import time


def transport_probe(config, events, cancellation):
    """This top-level hook needs only stdlib; pickle imports no model/data stack."""
    assert "TSFM_PRIVATE_SENTINEL" not in os.environ
    event = {
        "type": "probe",
        "digest": hashlib.sha256(config["blob"]).hexdigest(),
        "payload": "",
    }
    size = len(json.dumps(event, separators=(",", ":")).encode())
    # The actual binary frame prefix contains both Ctrl+Z and carriage return.
    event["payload"] = "x" * (0x1A0D - size)
    events.send(event)
    print("private Python worker log", flush=True)
    os.write(2, b"private native worker log\r\n")
    if config["wait"]:
        if not cancellation.wait(timeout=8):
            raise AssertionError("Cancellation control byte never arrived")
    events.send({"type": "finished", "cancelled": cancellation.is_set()})


def _enable_child_trace(monkeypatch):
    from tsfm_lab import worker_entry

    # Only the disposable test child receives timed stack diagnostics. No
    # environment values, pickle data or configuration fields are printed.
    prefix = (
        "import faulthandler, sys\n"
        "faulthandler.dump_traceback_later(4, repeat=True, file=sys.stderr)\n"
    )
    suffix = "\nfaulthandler.cancel_dump_traceback_later()\n"
    monkeypatch.setattr(
        worker_entry, "BOOTSTRAP", prefix + worker_entry.BOOTSTRAP + suffix
    )
    monkeypatch.setenv("TSFM_PRIVATE_SENTINEL", "fake-worker-portability-sentinel")


def _diagnostic(process, log):
    return {
        "exitcode": process.exitcode,
        "event_error": process.events.error,
        "event_reader_finished": process.events.finished.is_set(),
        "control_error": process.control.error,
        "startup_bytes_pending": bool(process.control.startup),
        "log_tail": log.read_bytes()[-8192:].decode("utf-8", errors="replace"),
    }


def _transport_roundtrip(tmp_path, monkeypatch, *, wait):
    from tsfm_lab import worker_entry

    _enable_child_trace(monkeypatch)
    log = tmp_path / "private-worker.log"
    blob = bytes(range(256)) * 512  # Exceeds a Windows anonymous pipe's buffer.
    process = None
    received = []
    try:
        with log.open("ab", buffering=0) as output:
            process, events, control = worker_entry.launch_worker(
                {"blob": blob, "wait": wait}, transport_probe, log, output
            )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            while events.poll():
                try:
                    event = events.recv()
                except EOFError:
                    break
                received.append(event)
                if wait and event["type"] == "probe":
                    control.set()
            if not process.is_alive() and events.finished.is_set():
                break
            time.sleep(0.005)
        assert not process.is_alive(), _diagnostic(process, log)
        assert events.finished.is_set(), _diagnostic(process, log)
        assert process.exitcode == 0, _diagnostic(process, log)
        assert [event["type"] for event in received] == ["probe", "finished"], (
            received,
            _diagnostic(process, log),
        )
        assert received[0]["digest"] == hashlib.sha256(blob).hexdigest()
        assert len(json.dumps(received[0], separators=(",", ":")).encode()) == 0x1A0D
        assert received[1]["cancelled"] is wait
        contents = log.read_bytes()
        assert b"private Python worker log" in contents
        assert b"private native worker log" in contents
        assert b"fake-worker-portability-sentinel" not in contents
        assert not events.error and not control.error
    finally:
        if process is not None:
            if process.is_alive():
                process.kill()
            process.join(timeout=2)
            process.close()


def test_binary_stdio_roundtrip_exits_with_event_pipe_eof(tmp_path, monkeypatch):
    _transport_roundtrip(tmp_path, monkeypatch, wait=False)


def test_binary_stdio_roundtrip_receives_cancellation(tmp_path, monkeypatch):
    _transport_roundtrip(tmp_path, monkeypatch, wait=True)


def test_baseline_startup_reaches_application_after_private_handshake(
    config_factory, monkeypatch
):
    from tsfm_lab.config import load_config
    from tsfm_lab.execution import ManagedRun

    _enable_child_trace(monkeypatch)
    worker = ManagedRun(load_config(config_factory()))
    events = []
    try:
        worker.start()
        deadline = time.monotonic() + 10
        while worker.is_running and time.monotonic() < deadline:
            events.extend(worker.poll())
            time.sleep(0.005)
        assert not worker.is_running, _diagnostic(worker._process, worker.log_path)
        assert events[-1]["type"] == "completed", (
            events[-1],
            _diagnostic(worker._process, worker.log_path),
        )
        assert worker._process.exitcode == 0
    finally:
        worker.close()
