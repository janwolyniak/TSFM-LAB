"""Atomic local experiment records and verified numerical artifacts."""

import hashlib
import io
import json
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd

from tsfm_lab.artifact_validation import (
    PREDICTION_IDENTITIES,
    StoreError,
    validate_predictions,
)
from tsfm_lab.workflows import normalize_workflow, validate_workflow_units

_ACTIVE = ("pending", "preparing", "loading", "running", "evaluating", "saving")
_TERMINAL = {"completed", "failed", "cancelled"}
_INITIAL = {"configuration.json", "input.yaml"}
_ARTIFACTS = _INITIAL | {"metrics.json", "result.json", "predictions.csv"}


def _json(value) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + "\n"
    ).encode()


def _decode(raw: bytes) -> dict:
    try:
        value = json.loads(raw)
        _json(value)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object")
        return value
    except (UnicodeError, ValueError, TypeError) as exc:
        raise StoreError(f"Invalid JSON record: {exc}") from exc


def _atomic(path: Path, content: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read(path: Path) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise StoreError(f"Artifact must be a regular file: {path.name}")
        return path.read_bytes()
    except OSError as exc:
        raise StoreError(f"Cannot read {path.name}: {exc}") from exc


def _identity(raw: bytes) -> dict:
    return {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}


def _workflow(configuration):
    if not isinstance(configuration, dict):
        raise StoreError("Workflow configuration must be an object")
    workflow = normalize_workflow(configuration.get("workflow"))
    if workflow is not None:
        dataset = configuration.get("dataset")
        if not isinstance(dataset, dict):
            raise StoreError("Workflow configuration requires dataset metadata")
        validate_workflow_units(workflow, dataset.get("units"))
    return workflow


class ExperimentStore:
    """One private writer directory per UUID; manifests become completed last."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()
        self.root = self.workspace / "experiments"

    def _directory(self, run_id: str) -> Path:
        try:
            if not isinstance(run_id, str) or str(UUID(run_id)) != run_id:
                raise ValueError("Expected canonical UUID")
        except (ValueError, AttributeError) as exc:
            raise StoreError("run_id must be a canonical UUID") from exc
        directory = self.root / run_id
        if self.root.is_symlink() or directory.is_symlink():
            raise StoreError("Experiment directories must not be symbolic links")
        return directory

    def create(self, name: str, configuration: dict, source_yaml: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Experiment name must be nonempty")
        if not isinstance(configuration, dict) or not isinstance(source_yaml, str):
            raise ValueError("Expected configuration object and original YAML text")
        _workflow(configuration)
        artifacts = {
            "configuration.json": _json(configuration),
            "input.yaml": source_yaml.encode(),
        }
        run_id = str(uuid4())
        directory = self._directory(run_id)
        self.root.mkdir(parents=True, exist_ok=True)
        directory.mkdir(exist_ok=False)
        now = datetime.now(UTC).isoformat()
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "name": name,
            "status": "pending",
            "created_at": now,
            "updated_at": now,
            "artifacts": {key: _identity(raw) for key, raw in artifacts.items()},
        }
        for key, raw in artifacts.items():
            _atomic(directory / key, raw)
        _atomic(directory / "manifest.json", _json(manifest))
        return run_id

    def get_manifest(self, run_id: str) -> dict:
        manifest = _decode(_read(self._directory(run_id) / "manifest.json"))
        try:
            if (
                type(manifest["schema_version"]) is not int
                or manifest["schema_version"] != 1
            ):
                raise ValueError("Unsupported manifest schema")
            if manifest["run_id"] != run_id or manifest["status"] not in {
                *_ACTIVE,
                *_TERMINAL,
            }:
                raise ValueError("Invalid manifest identity or status")
            if not isinstance(manifest["name"], str) or not manifest["name"].strip():
                raise ValueError("Invalid experiment name")
            for key in ("created_at", "updated_at"):
                timestamp = datetime.fromisoformat(manifest[key])
                if timestamp.tzinfo is None:
                    raise ValueError("Manifest timestamps need a timezone")
            names = set(manifest["artifacts"])
            if not _INITIAL <= names <= _ARTIFACTS or (
                manifest["status"] == "completed" and names != _ARTIFACTS
            ):
                raise ValueError("Invalid artifact allowlist")
            for identity in manifest["artifacts"].values():
                if (
                    set(identity) != {"sha256", "size"}
                    or type(identity["size"]) is not int
                    or identity["size"] < 0
                ):
                    raise ValueError("Invalid artifact identity")
                if len(identity["sha256"]) != 64 or any(
                    c not in "0123456789abcdef" for c in identity["sha256"]
                ):
                    raise ValueError("Invalid artifact checksum")
        except (KeyError, ValueError, TypeError, AttributeError) as exc:
            raise StoreError(f"Invalid run manifest: {exc}") from exc
        return manifest

    def set_status(self, run_id: str, status: str, error: dict | None = None) -> dict:
        manifest = self.get_manifest(run_id)
        current = manifest["status"]
        if current in _TERMINAL or status == "completed":
            raise ValueError(
                "Terminal runs cannot change; completion requires complete()"
            )
        if status not in {*_ACTIVE, "failed", "cancelled"}:
            raise ValueError("Unsupported experiment status")
        if status in _ACTIVE and _ACTIVE.index(status) <= _ACTIVE.index(current):
            raise ValueError("Experiment stages must move forward")
        if error is not None and not isinstance(error, dict):
            raise ValueError("Experiment error must be an object")
        manifest.update(status=status, updated_at=datetime.now(UTC).isoformat())
        if error is not None:
            manifest["error"] = error
        _atomic(self._directory(run_id) / "manifest.json", _json(manifest))
        return manifest

    def _verified(self, run_id: str, manifest: dict) -> dict[str, bytes]:
        directory = self._directory(run_id)
        contents = {}
        for name, identity in manifest["artifacts"].items():
            raw = _read(directory / name)
            if _identity(raw) != identity:
                raise StoreError(f"Artifact checksum or size mismatch: {name}")
            contents[name] = raw
        return contents

    def _result(self, run_id: str, manifest: dict) -> tuple[dict, pd.DataFrame]:
        contents = self._verified(run_id, manifest)
        try:
            initial = _decode(contents["configuration.json"])
            contents["input.yaml"].decode("utf-8")
            result = _decode(contents["result.json"])
            if _workflow(initial) != _workflow(result["configuration"]):
                raise StoreError(
                    "Workflow metadata disagrees with the initial configuration"
                )
            if (
                type(result.get("schema_version")) is not int
                or result.get("schema_version") != 1
                or result.get("status") != "completed"
            ):
                raise StoreError("Invalid completed result schema or status")
            if result.get("run_id", run_id) != run_id or result.get(
                "metrics"
            ) != _decode(contents["metrics.json"]):
                raise StoreError("Result identity or metrics disagree with artifacts")
            frame = pd.read_csv(
                io.BytesIO(contents["predictions.csv"]),
                keep_default_na=False,
                dtype={name: str for name in PREDICTION_IDENTITIES},
                float_precision="round_trip",
            )
            return result, validate_predictions(frame, result)
        except (KeyError, UnicodeError, ValueError, TypeError) as exc:
            raise StoreError(f"Cannot reload result: {exc}") from exc

    def complete(
        self,
        run_id: str,
        result: dict,
        predictions: pd.DataFrame,
        *,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict:
        manifest = self.get_manifest(run_id)
        if manifest["status"] in _TERMINAL:
            raise ValueError("Terminal experiments cannot be overwritten")
        if cancel_check and cancel_check():
            raise KeyboardInterrupt("Experiment cancellation requested before saving")
        try:
            result = _decode(_json(result))
        except (TypeError, ValueError) as exc:
            raise StoreError(f"Invalid result snapshot: {exc}") from exc
        result.pop("predictions", None)
        frame = validate_predictions(predictions, result)
        if _workflow(self.load_configuration(run_id)) != _workflow(
            result["configuration"]
        ):
            raise StoreError(
                "Workflow metadata disagrees with the initial configuration"
            )
        if cancel_check and cancel_check():
            raise KeyboardInterrupt("Experiment cancellation requested before saving")
        if manifest["status"] != "saving":
            manifest = self.set_status(run_id, "saving")
        artifacts = {
            "result.json": _json(result),
            "metrics.json": _json(result["metrics"]),
            "predictions.csv": frame.to_csv(index=False, float_format="%.17g").encode(),
        }
        for name, raw in artifacts.items():
            if cancel_check and cancel_check():
                raise KeyboardInterrupt(
                    "Experiment cancellation requested while saving"
                )
            _atomic(self._directory(run_id) / name, raw)
            manifest["artifacts"][name] = _identity(raw)
        self._result(run_id, manifest)
        manifest.update(status="completed", updated_at=datetime.now(UTC).isoformat())
        if cancel_check and cancel_check():
            raise KeyboardInterrupt(
                "Experiment cancellation requested before completion"
            )
        _atomic(self._directory(run_id) / "manifest.json", _json(manifest))
        return manifest

    def result_summary(self, run_id: str) -> dict:
        """Read only checked result metadata for process lifecycle notifications.

        Prediction verification belongs to load_result; this avoids parsing large
        forecast tables on the terminal's process-monitoring path.
        """
        manifest = self.get_manifest(run_id)
        if manifest["status"] != "completed":
            raise StoreError("Only completed experiments have result summaries")
        raw = _read(self._directory(run_id) / "result.json")
        if _identity(raw) != manifest["artifacts"]["result.json"]:
            raise StoreError("Result metadata checksum or size mismatch")
        result = _decode(raw)
        try:
            metrics = result["metrics"]["overall"]
            counts = result["counts"]
            if not isinstance(metrics, dict) or not isinstance(counts, dict):
                raise ValueError("Invalid summary fields")
            return {
                "run_id": run_id,
                "name": manifest["name"],
                "status": "completed",
                "workspace": str(self.workspace),
                "directory": str(self._directory(run_id)),
                "metrics": metrics,
                "counts": counts,
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise StoreError(f"Invalid completed result summary: {exc}") from exc

    def load_result(self, run_id: str) -> tuple[dict, pd.DataFrame]:
        manifest = self.get_manifest(run_id)
        if manifest["status"] != "completed":
            raise StoreError("Only completed experiments have reloadable results")
        return self._result(run_id, manifest)

    def load_configuration(self, run_id: str) -> dict:
        """Read a detached, checked setup snapshot without source files or runtimes."""
        manifest = self.get_manifest(run_id)
        raw = _read(self._directory(run_id) / "configuration.json")
        if _identity(raw) != manifest["artifacts"]["configuration.json"]:
            raise StoreError("Configuration checksum or size mismatch")
        configuration = _decode(raw)
        try:
            _workflow(configuration)
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            raise StoreError(f"Invalid workflow configuration: {exc}") from exc
        return configuration

    def list_runs(self) -> list[dict]:
        if self.root.is_symlink():
            raise StoreError("Experiment directories must not be symbolic links")
        if not self.root.exists():
            return []
        records = []
        for directory in self.root.iterdir():
            if not directory.is_dir() and not directory.is_symlink():
                continue
            manifest = None
            try:
                manifest = self.get_manifest(directory.name)
                if manifest["status"] == "completed":
                    self.load_result(directory.name)
            except (StoreError, OSError) as exc:
                manifest = {
                    **(manifest or {}),
                    "run_id": directory.name,
                    "name": manifest["name"] if manifest else directory.name,
                    "status": "corrupt",
                    "created_at": manifest["created_at"] if manifest else "",
                    "error": {"type": "StoreError", "message": str(exc)},
                }
            records.append(manifest)
        return sorted(records, key=lambda record: record["created_at"], reverse=True)
