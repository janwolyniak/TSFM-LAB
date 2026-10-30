"""One spawned inference process with nonblocking events and truthful cancellation."""

import math
import tempfile
import threading
from concurrent.futures import Future
from pathlib import Path
from time import monotonic

from tsfm_lab.config import ExperimentConfig
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.future_config import FutureForecastConfig
from tsfm_lab.worker_entry import launch_worker

_KILL_DELAY = 0.25
_TERMINAL = {"completed", "failed", "cancelled"}


def _worker_main(config, events, cancellation) -> None:
    from tsfm_lab.run_modes import run_validated

    run_id, last_progress = None, 0.0

    def created(identifier):
        nonlocal run_id
        run_id = identifier
        events.send({"type": "stage", "stage": "pending", "run_id": run_id})

    def stage(value):
        events.send({"type": "stage", "stage": value, "run_id": run_id})

    def progress(done, total):
        nonlocal last_progress
        now = monotonic()
        if done in (0, total) or now - last_progress >= 0.05:
            events.send(
                {
                    "type": "progress",
                    "completed": done,
                    "total": total,
                    "run_id": run_id,
                }
            )
            last_progress = now

    try:
        result = run_validated(
            config,
            progress=stage,
            window_progress=progress,
            should_cancel=cancellation.is_set,
            on_created=created,
        )
        events.send({"type": "completed", "result": result, "run_id": result["run_id"]})
    except KeyboardInterrupt:
        events.send({"type": "cancelled", "run_id": run_id})
    except Exception as exc:
        events.send({"type": "failed", "message": str(exc), "run_id": run_id})


def _log_tail(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 8192))
            return stream.read(8192).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _reconcile(
    config, run_id, pending, cancellation_requested, exitcode, stage, log_path
):
    """Read/write only small metadata after the inference process has been reaped."""
    pending = pending or {}
    requested = cancellation_requested or pending.get("type") == "cancelled"
    message = (
        pending.get("message")
        or f"Worker exited with code {exitcode} before completing its record"
    )
    try:
        if run_id is None:
            event = {"type": "cancelled" if requested else "failed", "run_id": None}
            if not requested:
                event["message"] = message
        else:
            store = ExperimentStore(config.output_directory)
            manifest = store.get_manifest(run_id)
            future = isinstance(config, FutureForecastConfig)
            if future != (manifest.get("kind") == "future_forecast"):
                raise ValueError("Worker record mode disagrees with its configuration")
            status = manifest["status"]
            if status == "completed":
                if future:
                    from tsfm_lab.future_validation import resolved_configuration

                    saved, _, _ = store.load_future_result(run_id)
                    if saved["configuration"] != resolved_configuration(
                        config, saved["request"]["origin"]
                    ):
                        raise ValueError(
                            "Completed future record disagrees with its worker configuration"
                        )
                return {
                    "type": "completed",
                    "run_id": run_id,
                    "result": store.result_summary(run_id),
                }
            if status not in _TERMINAL:
                status = "cancelled" if requested else "failed"
                manifest = store.set_status(
                    run_id,
                    status,
                    error={
                        "type": "Cancelled" if requested else "WorkerExit",
                        "message": message,
                        "stage": stage or manifest["status"],
                    },
                )
            event = {"type": status, "run_id": run_id}
            if status == "failed":
                error = manifest.get("error")
                event["message"] = (
                    error.get("message", message)
                    if isinstance(error, dict)
                    else message
                )
    except Exception as exc:
        event = {
            "type": "failed",
            "run_id": run_id,
            "message": f"Cannot reconcile experiment record: {exc}",
        }
    if event["type"] == "failed":
        tail = _log_tail(log_path)
        if tail:
            event["log_tail"] = tail
    return event


class ManagedRun:
    """Poll regularly; final events follow process reaping and metadata settlement."""

    def __init__(self, config: ExperimentConfig, grace_seconds=3.0, *, _target=None):
        if not isinstance(config, (ExperimentConfig, FutureForecastConfig)):
            raise ValueError("ManagedRun requires a validated ExperimentConfig")
        if isinstance(config, FutureForecastConfig) and config._record_only:
            raise ValueError(
                "Saved future configurations are read-only; load a native configuration before starting a worker"
            )
        if (
            isinstance(grace_seconds, bool)
            or not isinstance(grace_seconds, (int, float))
            or not math.isfinite(grace_seconds)
            or grace_seconds < 0
        ):
            raise ValueError("grace_seconds must be a finite nonnegative number")
        self.config, self.grace_seconds = config, float(grace_seconds)
        self._target = _target or _worker_main
        self._process = self._events = self._cancellation = None
        self._started = self._closed = self._terminal_emitted = False
        self._run_id = self._pending_terminal = self._last_stage = None
        self._requested_at = self._terminated_at = None
        self._outbox = []
        self._finalizer = None
        self._log_path = None

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @property
    def log_path(self) -> Path | None:
        """Controller-owned diagnostic log, retained until close()."""
        return self._log_path

    @property
    def is_running(self) -> bool:
        return self._started and not self._closed and not self._terminal_emitted

    def start(self) -> None:
        if self._started or self._closed:
            raise RuntimeError("A managed run can only be started once")
        if isinstance(self.config, FutureForecastConfig) and self.config._record_only:
            raise ValueError(
                "Saved future configurations are read-only; load a native configuration before starting a worker"
            )
        try:
            with tempfile.NamedTemporaryFile(
                mode="ab", prefix="tsfm-run-", suffix=".log", delete=False
            ) as output:
                self._log_path = Path(output.name)
                self._process, self._events, self._cancellation = launch_worker(
                    self.config, self._target, self._log_path, output
                )
            self._started = True
        except BaseException:
            if self._process is not None:
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=1.0)
                if not self._process.is_alive():
                    self._process.close()
            self._events = None
            if self._log_path is not None:
                self._log_path.unlink(missing_ok=True)
            raise

    def request_cancel(self) -> None:
        if not self._started:
            raise RuntimeError("Start the managed run before requesting cancellation")
        if (
            self._closed
            or self._terminal_emitted
            or self._requested_at is not None
            or not self._process.is_alive()
        ):
            return
        self._requested_at = monotonic()
        self._cancellation.set()
        self._outbox.append({"type": "cancellation_requested", "run_id": self._run_id})

    def _drain(self) -> None:
        for _ in range(128):
            if self._events is None or not self._events.poll():
                break
            try:
                event = self._events.recv()
            except (EOFError, OSError):
                self._events.close()
                self._events = None
                return
            if event.get("run_id") is not None:
                self._run_id = event["run_id"]
            if event["type"] in _TERMINAL:
                self._pending_terminal = event
            else:
                if event["type"] == "stage":
                    self._last_stage = event["stage"]
                self._outbox.append(event)

    def _settle(self) -> None:
        self._process.join(timeout=0)
        if not self._process.events.finished.is_set():
            return
        self._drain()
        self._finalizer = Future()
        exitcode = self._process.exitcode

        def finish():
            event = _reconcile(
                self.config,
                self._run_id,
                self._pending_terminal,
                self._requested_at is not None,
                exitcode,
                self._last_stage,
                self._log_path,
            )
            self._finalizer.set_result(event)

        threading.Thread(
            target=finish, name="tsfm-record-finalizer", daemon=True
        ).start()

    def poll(self) -> list[dict]:
        """Drain small events and advance stop/reap state without waiting or file IO."""
        if not self._started or self._closed:
            return []
        if (
            self._process.events.error or self._process.control.error
        ) and self._process.is_alive():
            self._process.kill()
        self._drain()
        if self._finalizer is None:
            alive, now = self._process.is_alive(), monotonic()
            if (
                alive
                and self._requested_at is not None
                and now - self._requested_at >= self.grace_seconds
            ):
                if self._terminated_at is None:
                    self._process.terminate()
                    self._terminated_at = now
                elif now - self._terminated_at >= _KILL_DELAY:
                    self._process.kill()
            if not self._process.is_alive():
                self._settle()
        if (
            self._finalizer is not None
            and self._finalizer.done()
            and not self._terminal_emitted
        ):
            self._outbox.append(self._finalizer.result())
            self._terminal_emitted = True
        events, self._outbox = self._outbox, []
        return events

    def close(self) -> list[dict]:
        """Bound cleanup, retain late completion truth, and remove the temporary log."""
        if self._closed:
            return []
        events = []
        if self._started:
            self.request_cancel()
            deadline = monotonic() + self.grace_seconds + _KILL_DELAY + 1.0
            while self.is_running and monotonic() < deadline:
                events.extend(self.poll())
                if self._process.is_alive():
                    self._process.join(timeout=0.02)
                elif self._finalizer is not None:
                    try:
                        self._finalizer.result(timeout=0.02)
                    except TimeoutError:
                        pass
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1.0)
            events.extend(self.poll())
            if self._process.is_alive():
                raise RuntimeError("Inference worker did not stop")
            self._process.close()
        if self._events is not None:
            self._events.close()
            self._events = None
        if self._log_path is not None:
            self._log_path.unlink(missing_ok=True)
        self._closed = True
        return events
