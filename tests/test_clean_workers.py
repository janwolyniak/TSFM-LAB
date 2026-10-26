"""Clean worker environments exist before project imports, with bounded private IPC."""

import json
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from tsfm_lab import worker_entry
from tsfm_lab.config import load_config
from tsfm_lab.execution import ManagedRun

CREDENTIAL_NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_ACCESS_TOKEN",
    "OPENAI_REFRESH_TOKEN",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "TSFM_PRIVATE_SENTINEL",
)


def collect(workers, timeout=15):
    groups = [[] for _ in workers]
    deadline = time.monotonic() + timeout
    while any(worker.is_running for worker in workers) and time.monotonic() < deadline:
        for worker, events in zip(workers, groups, strict=True):
            events.extend(worker.poll())
        time.sleep(0.005)
    assert all(not worker.is_running for worker in workers)
    return groups


def oversized_event(config, events, cancellation):
    events.send({"type": "stage", "stage": "X" * (worker_entry.MAX_EVENT + 1)})


def invalid_frame(config, events, cancellation):
    events.stream.write(struct.pack("!I", worker_entry.MAX_EVENT + 1))
    events.stream.flush()
    while not cancellation.is_set():
        time.sleep(0.005)


def flood_events(config, events, cancellation):
    for index in range(1000):
        events.send({"type": "stage", "stage": str(index), "run_id": None})
    while not cancellation.is_set():
        time.sleep(0.005)


def test_allowlist_preserves_nonsecret_runtime_flags_without_mutating_input():
    values = {
        "HOME": "/fictional/home",
        "HF_HOME": "/fictional/cache",
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "2",
        "PYTHONPATH": "/injected",
        "OPENAI_API_KEY": "fake-openai-value",
        "GEMINI_API_KEY": "fake-gemini-value",
        "HF_TOKEN": "fake-hub-value",
        "HTTP_PROXY": "fake-proxy-value",
        "ARBITRARY_PRIVATE_VARIABLE": "fake-private-value",
    }
    original = dict(values)
    selected = worker_entry.worker_environment(values)
    assert values == original
    assert selected == {
        "HOME": "/fictional/home",
        "HF_HOME": "/fictional/cache",
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "2",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    }


def test_credentials_absent_at_startup_before_project_imports(
    config_factory, tmp_path, monkeypatch, capfd
):
    fake = {
        name: f"fake-clean-worker-{index}-credential"
        for index, name in enumerate(CREDENTIAL_NAMES)
    }
    for name, value in fake.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    marker = tmp_path / "startup-environment.json"
    prefix = (
        "import json, os, sys\n"
        f"with open({str(marker)!r}, 'w') as stream:\n"
        "    json.dump({'names': sorted(os.environ), 'isolated': sys.flags.isolated, "
        "'hub_offline': os.environ.get('HF_HUB_OFFLINE'), "
        "'transformers_offline': os.environ.get('TRANSFORMERS_OFFLINE')}, stream)\n"
    )
    monkeypatch.setattr(worker_entry, "BOOTSTRAP", prefix + worker_entry.BOOTSTRAP)
    original = worker_entry.subprocess.Popen
    launches = []

    def checked(*args, **kwargs):
        assert all(os.environ.get(name) == value for name, value in fake.items())
        assert not set(CREDENTIAL_NAMES).intersection(kwargs["env"])
        assert all(value not in json.dumps(args) for value in fake.values())
        launches.append(dict(kwargs["env"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(worker_entry.subprocess, "Popen", checked)
    config = load_config(config_factory())
    worker = ManagedRun(config)
    try:
        worker.start()
        events = collect([worker])[0]
        assert worker._process.exitcode == 0
        assert events[-1]["type"] == "completed"
        startup = json.loads(marker.read_text())
        assert not set(CREDENTIAL_NAMES).intersection(startup["names"])
        assert startup["isolated"] == 1
        assert startup["hub_offline"] == startup["transformers_offline"] == "0"
        assert all(os.environ.get(name) == value for name, value in fake.items())
        public = json.dumps(events) + worker.log_path.read_text()
        captured = capfd.readouterr()
        public += captured.out + captured.err
        for path in config.output_directory.rglob("*"):
            if path.is_file():
                public += path.read_bytes().decode("utf-8", errors="replace")
        assert all(value not in public for value in fake.values())
        assert len(launches) == 1
    finally:
        log = worker.log_path
        worker.close()
    assert not log.exists()


def test_concurrent_launches_do_not_remove_parent_credentials(
    config_factory, tmp_path, monkeypatch
):
    fake = {
        name: f"fake-concurrent-{index}-credential"
        for index, name in enumerate(CREDENTIAL_NAMES)
    }
    for name, value in fake.items():
        monkeypatch.setenv(name, value)
    original = worker_entry.subprocess.Popen
    snapshots = []

    def checked(*args, **kwargs):
        assert all(os.environ.get(name) == value for name, value in fake.items())
        snapshots.append(set(kwargs["env"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(worker_entry.subprocess, "Popen", checked)
    configuration = load_config(config_factory())
    workers = [
        ManagedRun(
            replace(configuration, output_directory=tmp_path / f"parallel-{index}")
        )
        for index in range(2)
    ]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda worker: worker.start(), workers))
        groups = collect(workers)
        assert all(events[-1]["type"] == "completed" for events in groups)
        assert all(worker._process.exitcode == 0 for worker in workers)
        assert len(snapshots) == 2
        assert all(not names.intersection(CREDENTIAL_NAMES) for names in snapshots)
        assert all(os.environ.get(name) == value for name, value in fake.items())
    finally:
        for worker in workers:
            worker.close()


def test_normal_exit_has_no_daemon_stdio_shutdown_failure(config_factory):
    worker = ManagedRun(load_config(config_factory()))
    try:
        worker.start()
        events = collect([worker])[0]
        assert events[-1]["type"] == "completed"
        assert worker._process.exitcode == 0
        assert "Fatal Python" not in worker.log_path.read_text()
        assert worker._process.events.finished.is_set()
    finally:
        worker.close()


@pytest.mark.parametrize("target", [oversized_event, invalid_frame])
def test_invalid_private_events_fail_bounded_and_reap_child(config_factory, target):
    worker = ManagedRun(load_config(config_factory()), _target=target)
    try:
        worker.start()
        events = collect([worker])[0]
        assert events[-1]["type"] == "failed"
        assert events[-1]["run_id"] is None
        assert len(json.dumps(events)) < 2048
        assert not worker._process.is_alive()
        assert not worker.config.output_directory.exists()
    finally:
        worker.close()


def test_event_flood_is_backpressured_and_poll_remains_bounded(config_factory):
    worker = ManagedRun(
        load_config(config_factory()), grace_seconds=0.02, _target=flood_events
    )
    try:
        worker.start()
        deadline = time.monotonic() + 5
        while (
            worker._process.events.items.qsize() < worker_entry.MAX_QUEUED
            and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        assert worker._process.events.items.qsize() == worker_entry.MAX_QUEUED
        before = time.monotonic()
        events = worker.poll()
        assert time.monotonic() - before < 0.1
        assert len(events) <= worker_entry.MAX_QUEUED
        worker.request_cancel()
        assert collect([worker])[0][-1]["type"] == "cancelled"
    finally:
        worker.close()


def test_oversized_trusted_startup_fails_without_child_or_artifacts(
    config_factory, monkeypatch
):
    configuration = replace(
        load_config(config_factory()), source_yaml="X" * worker_entry.MAX_STARTUP
    )
    called = []

    def forbidden(*args, **kwargs):
        called.append(True)
        raise AssertionError("Oversized startup must not launch")

    monkeypatch.setattr(worker_entry.subprocess, "Popen", forbidden)
    worker = ManagedRun(configuration)
    with pytest.raises(ValueError, match="size limit"):
        worker.start()
    assert not called
    assert not worker.log_path.exists()
    assert not configuration.output_directory.exists()
    worker.close()


@pytest.mark.parametrize("wrapper", ["Events", "Control"])
def test_ipc_setup_failure_reaps_created_child_and_closes_pipes(
    config_factory, monkeypatch, wrapper
):
    original = worker_entry.subprocess.Popen
    processes = []

    def capture(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    def fail(*args, **kwargs):
        raise RuntimeError("event reader unavailable")

    monkeypatch.setattr(worker_entry.subprocess, "Popen", capture)
    monkeypatch.setattr(worker_entry, wrapper, fail)
    worker = ManagedRun(load_config(config_factory()))
    with pytest.raises(RuntimeError, match="reader unavailable"):
        worker.start()
    assert len(processes) == 1 and processes[0].poll() is not None
    assert processes[0].stdin.closed and processes[0].stdout.closed
    assert not worker.log_path.exists()
    worker.close()


def test_temporary_control_failure_does_not_publish_fake_completion(
    config_factory, monkeypatch
):
    original = worker_entry.Control._write

    def fail(self):
        self.startup = b""
        self.error = True

    monkeypatch.setattr(worker_entry.Control, "_write", fail)
    worker = ManagedRun(load_config(config_factory()))
    try:
        worker.start()
        events = collect([worker])[0]
        assert events[-1]["type"] == "failed"
        assert not worker.config.output_directory.exists()
    finally:
        worker.close()
    monkeypatch.setattr(worker_entry.Control, "_write", original)


def test_event_writer_handles_partial_pipe_writes():
    class ShortWriter:
        def __init__(self):
            self.output = bytearray()

        def write(self, data):
            count = min(3, len(data))
            self.output.extend(data[:count])
            return count

        def flush(self):
            pass

    stream = ShortWriter()
    event = {"type": "stage", "stage": "X" * 12000}
    worker_entry.EventWriter(stream).send(event)
    size = struct.unpack("!I", stream.output[:4])[0]
    assert size == len(stream.output) - 4
    assert json.loads(stream.output[4:]) == event


@pytest.mark.parametrize("failed_name", ["tsfm-worker-events", "tsfm-worker-control"])
@pytest.mark.parametrize("after_start", [False, True])
def test_ipc_thread_start_failure_reaps_child_and_closes_pipes(
    config_factory, monkeypatch, failed_name, after_start
):
    original_process = worker_entry.subprocess.Popen
    original_start = worker_entry.threading.Thread.start
    processes, threads = [], []

    def capture(*args, **kwargs):
        process = original_process(*args, **kwargs)
        processes.append(process)
        return process

    def start(thread):
        if thread.name.startswith("tsfm-worker-"):
            threads.append(thread)
        if thread.name == failed_name and not after_start:
            raise RuntimeError("worker thread unavailable")
        result = original_start(thread)
        if thread.name == failed_name:
            raise RuntimeError("worker thread unavailable")
        return result

    monkeypatch.setattr(worker_entry.subprocess, "Popen", capture)
    monkeypatch.setattr(worker_entry.threading.Thread, "start", start)
    worker = ManagedRun(load_config(config_factory()))
    with pytest.raises(RuntimeError, match="thread unavailable"):
        worker.start()
    assert len(processes) == 1 and processes[0].poll() is not None
    assert processes[0].stdin.closed and processes[0].stdout.closed
    assert all(not thread.is_alive() for thread in threads)
    assert worker._process is None and not worker.is_running
    assert not worker.log_path.exists()
    assert not worker.config.output_directory.exists()
    worker.close()
