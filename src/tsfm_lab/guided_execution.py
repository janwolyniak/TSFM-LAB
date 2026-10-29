"""Managed guided work with a durable preparation record and existing worker IPC."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from uuid import UUID, uuid4

from tsfm_lab.execution import ManagedRun
from tsfm_lab.experiments import ExperimentStore, _atomic
from tsfm_lab.run_modes import is_future

_TERMINAL = {"completed", "failed", "cancelled"}
_STAGES = {
    "pending",
    "cache",
    "downloading",
    "verifying",
    "preparing",
    "loading",
    "running",
    "evaluating",
    "saving",
} | _TERMINAL


def _now():
    return datetime.now(UTC).isoformat()


class PreparationRecord:
    def __init__(self, config, plan, operation_id=None):
        if is_future(config) and config._record_only:
            raise ValueError(
                "Saved future configurations are read-only; preparation requires a native configuration"
            )
        self.operation_id = operation_id or str(uuid4())
        if str(UUID(self.operation_id)) != self.operation_id:
            raise ValueError("Preparation ID must be a canonical UUID")
        root = Path(config.output_directory).resolve() / "preparations"
        if root.is_symlink():
            raise ValueError("Preparation directory must not be a symbolic link")
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"{self.operation_id}.json"
        if self.path.exists() or self.path.is_symlink():
            current = self.load(self.path)
            if (
                current.data["status"] != "pending"
                or current.data["configuration_sha256"] != plan.configuration_sha256
                or current.data["dataset_sha256"] != plan.dataset_sha256
                or (
                    is_future(config)
                    and (
                        current.data.get("schema_version") != 2
                        or current.data.get("mode") != "future_forecast"
                        or current.data.get("origin")
                        != config.forecast.origin.isoformat()
                        or current.data.get("plan_identity") != plan.plan_identity
                    )
                )
            ):
                raise ValueError(
                    "Preparation was already started or belongs to another review"
                )
            self.data = current.data
            self._last_saved = current._last_saved
        else:
            now = _now()
            self.data = {
                "schema_version": 2 if is_future(config) else 1,
                "operation_id": self.operation_id,
                "configuration_sha256": plan.configuration_sha256,
                "dataset_sha256": plan.dataset_sha256,
                "run_id": None,
                "status": "pending",
                "stage": "pending",
                "created_at": now,
                "updated_at": now,
                "elapsed_seconds": 0.0,
                "history": [],
                **(
                    {
                        "mode": "future_forecast",
                        "origin": config.forecast.origin.isoformat(),
                        "plan_identity": plan.plan_identity,
                    }
                    if is_future(config)
                    else {}
                ),
            }
            self._save()

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.is_symlink() or path.stat().st_size > 1024 * 1024:
            raise ValueError("Invalid preparation record")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["schema_version"] not in {1, 2} or data["stage"] not in _STAGES:
            raise ValueError("Invalid preparation record")
        if data["schema_version"] == 2:
            required = {
                "schema_version",
                "operation_id",
                "configuration_sha256",
                "dataset_sha256",
                "run_id",
                "status",
                "stage",
                "created_at",
                "updated_at",
                "elapsed_seconds",
                "history",
                "mode",
                "origin",
                "plan_identity",
            }
            if (
                not required
                <= set(data)
                <= required
                | {"error", "completed_bytes", "total_bytes", "checkpoint_fingerprint"}
            ):
                raise ValueError("Invalid future preparation fields")
            origin = datetime.fromisoformat(data["origin"])
            if (
                data["mode"] != "future_forecast"
                or origin.tzinfo is None
                or data["stage"] == "evaluating"
            ):
                raise ValueError("Invalid future preparation mode or origin")
            fingerprint = data.get("checkpoint_fingerprint")
            if fingerprint is not None and (
                not isinstance(fingerprint, str)
                or len(fingerprint) != 64
                or any(c not in "0123456789abcdef" for c in fingerprint)
            ):
                raise ValueError("Invalid prepared checkpoint fingerprint")
        result = object.__new__(cls)
        result.path, result.data = path, data
        result.operation_id = data["operation_id"]
        result._last_saved = monotonic()
        return result

    @property
    def elapsed(self):
        return max(
            0.0,
            (
                datetime.now(UTC) - datetime.fromisoformat(self.data["created_at"])
            ).total_seconds(),
        )

    def _save(self):
        self.data.update(updated_at=_now(), elapsed_seconds=self.elapsed)
        _atomic(
            self.path,
            (json.dumps(self.data, indent=2, allow_nan=False) + "\n").encode(),
        )
        self._last_saved = monotonic()

    def update(self, event):
        stage = event.get("stage", self.data["stage"])
        if stage not in _STAGES - _TERMINAL or (
            self.data.get("mode") == "future_forecast" and stage == "evaluating"
        ):
            raise ValueError("Invalid preparation stage")
        changed = stage != self.data["stage"]
        if changed:
            self.data["history"].append(
                {"stage": stage, "elapsed_seconds": self.elapsed}
            )
        self.data.update(stage=stage, status="running")
        for key in ("completed_bytes", "total_bytes"):
            if key in event:
                self.data[key] = event[key]
        done, total = self.data.get("completed_bytes"), self.data.get("total_bytes")
        final = total is not None and done == total
        if changed or final or monotonic() - self._last_saved >= 0.25:
            self._save()

    def link_run(self, run_id):
        if str(UUID(run_id)) != run_id:
            raise ValueError("Run ID must be a canonical UUID")
        self.data["run_id"] = run_id
        self._save()

    def finish(self, status, error=None):
        if status not in _TERMINAL:
            raise ValueError("Invalid preparation terminal status")
        if self.data["status"] == status:
            return
        if status == "completed":
            self.data.pop("error", None)
        if error is not None:
            self.data["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "stage": self.data["stage"],
            }
        self.data["history"].append({"stage": status, "elapsed_seconds": self.elapsed})
        self.data.update(stage=status, status=status)
        self._save()

    def record_checkpoint(self, fingerprint):
        if self.data["schema_version"] != 2:
            raise ValueError("Prepared future fingerprint requires schema 2")
        self.data["checkpoint_fingerprint"] = fingerprint
        self._save()


@dataclass(frozen=True)
class GuidedWorker:
    plan: object
    operation_id: str

    def __call__(self, config, events, cancellation):
        from tsfm_lab.guided import execute_guided

        run_id, last_progress = None, 0.0
        last_transfer, pending_transfer = float("-inf"), None

        def created(value):
            nonlocal run_id
            run_id = value
            events.send(
                {
                    "type": "stage",
                    "stage": "preparing",
                    "run_id": value,
                    "operation_id": self.operation_id,
                }
            )

        def send(value):
            events.send(
                value
                | {
                    "type": value.get("type", "stage"),
                    "run_id": run_id,
                    "operation_id": self.operation_id,
                }
            )

        def flush_transfer():
            nonlocal pending_transfer, last_transfer
            if pending_transfer is not None:
                send(pending_transfer)
                pending_transfer = None
                last_transfer = monotonic()

        def stage(value):
            nonlocal pending_transfer
            if value.get("type") == "transfer":
                pending_transfer = value
                done, total = value.get("completed_bytes"), value.get("total_bytes")
                final = total is not None and done == total
                if done == 0 or final or monotonic() - last_transfer >= 0.05:
                    flush_transfer()
            else:
                flush_transfer()
                send(value)

        def windows(done, total):
            nonlocal last_progress
            now = monotonic()
            if done in (0, total) or now - last_progress >= 0.05:
                events.send(
                    {
                        "type": "progress",
                        "completed": done,
                        "total": total,
                        "run_id": run_id,
                        "operation_id": self.operation_id,
                    }
                )
                last_progress = now

        try:
            result = execute_guided(
                self.plan,
                operation_id=self.operation_id,
                progress=stage,
                window_progress=windows,
                should_cancel=cancellation.is_set,
                on_created=created,
            )
            flush_transfer()
            events.send({"type": "completed", "run_id": run_id, "result": result})
        except KeyboardInterrupt:
            flush_transfer()
            events.send({"type": "cancelled", "run_id": run_id})
        except Exception as exc:
            flush_transfer()
            events.send({"type": "failed", "run_id": run_id, "message": str(exc)})


class ManagedGuidedRun:
    """Compose the ordinary worker; settle preparation only after it has stopped."""

    def __init__(self, plan, grace_seconds=3.0, *, _target=None):
        self.plan, self.operation_id = plan, str(uuid4())
        self._record = None
        self._recovered_run_id = None
        self._terminal = False
        self._cancelled = False
        self._inner = ManagedRun(
            plan.config,
            grace_seconds,
            _target=_target or GuidedWorker(plan, self.operation_id),
        )

    @property
    def run_id(self):
        return self._inner.run_id or self._recovered_run_id

    @property
    def log_path(self):
        return self._inner.log_path

    @property
    def is_running(self):
        return self._inner.is_running

    def start(self):
        self._record = PreparationRecord(self.plan.config, self.plan, self.operation_id)
        try:
            self._inner.start()
        except BaseException as exc:
            self._record.finish("failed", exc)
            raise

    def request_cancel(self):
        self._cancelled = True
        self._inner.request_cancel()

    def _settle(self, event):
        if self._terminal:
            return event
        self._terminal = True
        record = PreparationRecord.load(self._record.path)
        run_id = event.get("run_id") or record.data.get("run_id")
        self._recovered_run_id = run_id
        event = event | {"run_id": run_id}
        store = ExperimentStore(self.plan.config.output_directory)
        try:
            if is_future(self.plan.config) and (
                record.data.get("schema_version") != 2
                or record.data.get("plan_identity") != self.plan.plan_identity
                or record.data.get("configuration_sha256")
                != self.plan.configuration_sha256
                or record.data.get("dataset_sha256") != self.plan.dataset_sha256
                or record.data.get("origin")
                != self.plan.config.forecast.origin.isoformat()
            ):
                raise ValueError(
                    "Future preparation record disagrees with its reviewed plan"
                )
            if run_id is not None:
                manifest = store.get_manifest(run_id)
                if manifest["status"] == "completed":
                    expected_schema = 3 if is_future(self.plan.config) else 2
                    if manifest["schema_version"] != expected_schema:
                        raise ValueError(
                            "Guided run did not publish its complete output contract"
                        )
                    if is_future(self.plan.config):
                        from tsfm_lab.guided import validate_reviewed_checkpoint

                        saved, _, _ = store.load_future_result(run_id)
                        validate_reviewed_checkpoint(
                            self.plan,
                            saved["provenance"]["model"]["options"].get("checkpoint"),
                            record.data.get("checkpoint_fingerprint"),
                        )
                        if (
                            saved["configuration"] != self.plan.config.resolved_dict()
                            or saved["provenance"]["dataset"]["canonical_sha256"]
                            != self.plan.dataset_sha256
                            or saved["provenance"]["dataset"]["source_sha256"]
                            != self.plan.source_sha256
                        ):
                            raise ValueError(
                                "Future output disagrees with its reviewed plan"
                            )
                    result = store.result_summary(run_id) | {
                        "outputs": store.output_paths(run_id)
                    }
                    event = {"type": "completed", "run_id": run_id, "result": result}
                elif manifest["status"] in _TERMINAL:
                    event = event | {"type": manifest["status"]}
                    if manifest["status"] == "failed" and manifest.get("error"):
                        event["message"] = manifest["error"].get(
                            "message", event.get("message", "Experiment failed")
                        )
                elif manifest["status"] not in _TERMINAL:
                    status = (
                        "cancelled"
                        if self._cancelled or event["type"] == "cancelled"
                        else "failed"
                    )
                    store.set_status(
                        run_id,
                        status,
                        error={
                            "type": "WorkerExit",
                            "message": event.get(
                                "message", "Managed guided worker stopped"
                            ),
                            "stage": (
                                manifest["status"]
                                if is_future(self.plan.config)
                                else record.data["stage"]
                            ),
                        },
                    )
                    event = event | {"type": status, "run_id": run_id}
            if event["type"] == "completed" and run_id is None:
                raise ValueError(
                    "Guided worker reported completion without a saved run"
                )
            record.finish(
                event["type"],
                (
                    RuntimeError(event["message"])
                    if event.get("message") and event["type"] == "failed"
                    else None
                ),
            )
        except Exception as exc:
            event = {"type": "failed", "run_id": run_id, "message": str(exc)}
            record.finish("failed", exc)
        detail = {
            "operation_id": self.operation_id,
            "preparation_path": str(record.path),
        }
        if event["type"] == "completed":
            event["result"].update(detail)
        return event | detail

    def _events(self, events):
        return [
            self._settle(event) if event["type"] in _TERMINAL else event
            for event in events
        ]

    def poll(self):
        return self._events(self._inner.poll())

    def close(self):
        if self.is_running:
            self._cancelled = True
        return self._events(self._inner.close())
