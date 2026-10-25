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
    "pending": "Waiting to start",
    "preparing": "Preparing data",
    "loading": "Loading model",
    "running": "Forecasting",
    "evaluating": "Calculating metrics",
    "saving": "Verifying and saving results",
}


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


def _monitor(config, console, worker_factory, pause, stop):
    """Return the final event only after the worker stops; Ctrl+C requests a stop."""
    worker = worker_factory(config)
    requested = False
    terminal = None
    try:
        worker.start()
        with Live(console=console, transient=True, refresh_per_second=8) as live:
            live.update(Text("  Starting experiment · ctrl+c to cancel", style="dim"))
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
                            live.update(
                                Text(
                                    literal(
                                        f"  {STAGES.get(event['stage'], event['stage'])} · ctrl+c to cancel"
                                    ),
                                    style="dim",
                                )
                            )
                        elif kind == "progress":
                            live.update(
                                Text(
                                    f"  Forecasting · {event['completed']}/{event['total']} windows · ctrl+c to cancel",
                                    style="dim",
                                )
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
    return terminal
