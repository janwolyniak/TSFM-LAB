"""Monitor the existing inference worker and report progress within scrollback."""

import signal
import threading
import time
from contextlib import contextmanager

from rich.live import Live
from rich.text import Text

from tsfm_lab.execution import ManagedRun
from tsfm_lab.repl.text import literal

STAGES = {
    "cache": "Checking checkpoint cache",
    "downloading": "Downloading reviewed checkpoint files",
    "verifying": "Verifying checkpoint files",
    "pending": "Waiting to start",
    "preparing": "Preparing data",
    "loading": "Loading model",
    "running": "Forecasting",
    "evaluating": "Calculating metrics",
    "saving": "Verifying and saving results",
}


def _status(stage, started, windows=None, transfer=None, *, stopping=False):
    elapsed = max(0.0, time.monotonic() - started)
    if stopping:
        return Text(f"  Stopping worker… · elapsed {elapsed:.1f}s", style="dim")
    label = STAGES.get(stage, stage)
    if windows is not None:
        label = f"Forecasting · {windows[0]}/{windows[1]} windows"
    if transfer is not None:
        done, total = transfer
        measured = f"{done:,}" if total is None else f"{done:,}/{total:,}"
        label += f" · {measured} bytes measured"
    return Text(
        literal(f"  {label} · ctrl+c to cancel · elapsed {elapsed:.1f}s"), style="dim"
    )


@contextmanager
def _interrupt_request():
    stop = [False]
    main = threading.current_thread() is threading.main_thread()
    previous = signal.getsignal(signal.SIGINT) if main else None

    def request(signum, frame):
        stop[0] = True

    if main:
        signal.signal(signal.SIGINT, request)
    try:
        yield stop
    finally:
        if main:
            signal.signal(signal.SIGINT, previous)


def execute(config, console, *, worker_factory=ManagedRun, pause=time.sleep):
    # Defer SIGINT handling to poll boundaries so IPC reads cannot be interrupted.
    with _interrupt_request() as stop:
        return _monitor(config, console, worker_factory, pause, stop)


def execute_guided(plan, console):
    from tsfm_lab.guided_execution import ManagedGuidedRun

    return execute(plan, console, worker_factory=ManagedGuidedRun)


def _monitor(config, console, worker_factory, pause, stop):
    """Return the final event only after the worker stops; Ctrl+C requests a stop."""
    worker = worker_factory(config)
    requested = False
    terminal = None
    started = time.monotonic()
    stage, windows, transfer = "pending", None, None
    try:
        worker.start()
        with Live(console=console, transient=True, refresh_per_second=8) as live:
            live.update(_status(stage, started))
            while terminal is None:
                try:
                    if stop[0] and not requested:
                        worker.request_cancel()
                        requested = True
                        console.print(
                            Text(
                                "  Cancellation requested; waiting for the worker to stop.",
                                style="yellow",
                            )
                        )
                        live.update(Text("  Stopping worker…", style="dim"))
                    for event in worker.poll():
                        kind = event["type"]
                        if kind in {"completed", "failed", "cancelled"}:
                            terminal = event
                            break
                        if requested:
                            continue
                        if kind == "stage":
                            if windows is not None or transfer is not None:
                                console.print(
                                    _status(stage, started, windows, transfer)
                                )
                            stage, windows, transfer = event["stage"], None, None
                            console.print(_status(stage, started))
                        elif kind == "progress":
                            windows = (event["completed"], event["total"])
                        elif kind == "transfer":
                            done, total = event["completed_bytes"], event.get(
                                "total_bytes"
                            )
                            if (
                                type(done) is not int
                                or done < 0
                                or (
                                    total is not None
                                    and (type(total) is not int or total < done)
                                )
                            ):
                                raise ValueError("Invalid measured byte progress")
                            transfer = (done, total)
                    live.update(
                        _status(stage, started, windows, transfer, stopping=requested)
                    )
                    if terminal is None:
                        pause(0.05)
                except KeyboardInterrupt:
                    stop[0] = True
    finally:
        # A second interrupt must not leave an inference process alive.
        while True:
            try:
                final = worker.close()
                break
            except KeyboardInterrupt:
                continue
        if terminal is None:
            terminal = next(
                (
                    event
                    for event in reversed(final)
                    if event["type"] in {"completed", "cancelled", "failed"}
                ),
                None,
            )
    if terminal is None:
        raise RuntimeError("Worker stopped without a final status")
    console.print(
        Text(f"  Elapsed {max(0.0, time.monotonic() - started):.1f}s", style="dim")
    )
    return terminal
