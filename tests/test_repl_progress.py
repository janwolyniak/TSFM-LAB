import io
import os
import signal

import pytest
from rich.console import Console

from tsfm_lab.repl.progress import execute


class Worker:
    def __init__(self, config):
        self.calls = 0
        self.cancelled = self.closed = False

    def start(self):
        pass

    def poll(self):
        self.calls += 1
        if self.calls == 1:
            raise KeyboardInterrupt
        if self.calls == 2:
            assert self.cancelled and not self.closed
            return [{"type": "stage", "stage": "saving"}]
        return [{"type": "completed", "run_id": "late-completion"}]

    def request_cancel(self):
        self.cancelled = True

    def close(self):
        self.closed = True
        return []


def test_request_waits_and_late_completion_remains_authoritative():
    worker = Worker(None)
    output = io.StringIO()
    event = execute(
        None,
        Console(file=output),
        worker_factory=lambda _: worker,
        pause=lambda _: None,
    )
    assert event == {"type": "completed", "run_id": "late-completion"}
    assert worker.closed
    assert "Cancellation requested" in output.getvalue()


def test_unexpected_poll_failure_still_closes_worker():
    worker = Worker(None)

    def fail():
        raise OSError("broken pipe")

    worker.poll = fail
    with pytest.raises(OSError, match="broken pipe"):
        execute(None, Console(file=io.StringIO()), worker_factory=lambda _: worker)
    assert worker.closed


def test_sigint_is_deferred_until_poll_boundary_and_handler_is_restored():
    worker = Worker(None)
    previous = signal.getsignal(signal.SIGINT)

    def poll():
        worker.calls += 1
        if worker.calls == 1:
            os.kill(os.getpid(), signal.SIGINT)
            assert not worker.cancelled
            return []
        assert worker.cancelled
        return [{"type": "cancelled"}]

    worker.poll = poll
    event = execute(
        None,
        Console(file=io.StringIO()),
        worker_factory=lambda _: worker,
        pause=lambda _: None,
    )
    assert event["type"] == "cancelled" and worker.closed
    assert signal.getsignal(signal.SIGINT) is previous
