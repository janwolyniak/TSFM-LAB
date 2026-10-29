"""Private native-console fixture; no providers, vault, model weights or GUI."""

import ctypes
import io
import json
import os
import signal
import sys
import time
from pathlib import Path


def waiting_worker(config, events, cancellation):
    from tsfm_lab.application import run_config

    run_id = None

    def created(value):
        nonlocal run_id
        run_id = value
        events.send({"type": "stage", "stage": "pending", "run_id": value})

    def stage(value):
        events.send({"type": "stage", "stage": value, "run_id": run_id})
        if value == "running":
            (config.output_directory.parent / "console-ready").write_text("ready")
            while not cancellation.is_set():
                time.sleep(0.01)

    try:
        run_config(
            config,
            progress=stage,
            on_created=created,
            should_cancel=cancellation.is_set,
        )
    except KeyboardInterrupt:
        events.send({"type": "cancelled", "run_id": run_id})


def drive(source, config_path, report):
    sys.path.insert(0, source)
    from rich.console import Console

    # Import under its stable module name so the isolated worker can unpickle it.
    from windows_console_helpers import waiting_worker as target

    from tsfm_lab.config import load_config
    from tsfm_lab.execution import ManagedRun
    from tsfm_lab.experiments import ExperimentStore
    from tsfm_lab.repl.progress import execute

    config = load_config(config_path)
    worker = ManagedRun(config, _target=target)

    def pause(seconds):
        if (Path(report).parent / "console-stop").exists():
            signal.raise_signal(signal.SIGINT)
        time.sleep(seconds)

    event = execute(
        config,
        Console(file=io.StringIO()),
        worker_factory=lambda _: worker,
        pause=pause,
    )
    result = {
        "event": event["type"],
        "worker_alive": worker._process.is_alive(),
        "exitcode": worker._process.exitcode,
        "status": ExperimentStore(config.output_directory).get_manifest(
            event["run_id"]
        )["status"],
        "log_exists": worker.log_path.exists(),
    }
    Path(report).write_text(json.dumps(result), encoding="utf-8")


def send_control(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.FreeConsole()
    if not kernel.AttachConsole(int(pid)):
        error = ctypes.get_last_error()
        print(json.dumps({"console_unavailable": error}))
        return 77 if error in {5, 6} else 1
    try:
        # Ignore broadcast Ctrl+C only in this disposable sender. The target's
        # private console also contains its ordinary managed child worker.
        if not kernel.SetConsoleCtrlHandler(None, True):
            return 1
        if not kernel.GenerateConsoleCtrlEvent(0, 0):
            return 1
        time.sleep(0.1)
    finally:
        kernel.FreeConsole()
    return 0


if __name__ == "__main__":
    if os.name != "nt":
        raise SystemExit("Native Windows fixture only")
    if sys.argv[1] == "signal":
        raise SystemExit(send_control(sys.argv[2]))
    drive(*sys.argv[2:])
