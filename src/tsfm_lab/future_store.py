"""Strict future records beside historical experiments, with completed status last."""

from datetime import UTC, datetime
from uuid import uuid4

import pandas as pd

from tsfm_lab.artifact_validation import StoreError
from tsfm_lab.experiments import ExperimentStore, _atomic, _decode, _identity, _json

FUTURE_ARTIFACTS = frozenset(
    {
        "configuration.json",
        "input.yaml",
        "result.json",
        "predictions.csv",
        "observations.csv",
        "evaluation.json",
        "forecast.svg",
        "provenance.json",
    }
)
INITIAL = frozenset({"configuration.json", "input.yaml"})
ACTIVE = ("pending", "preparing", "loading", "running", "saving")
TERMINAL = frozenset({"completed", "failed", "cancelled"})


def validate_manifest(manifest, run_id):
    from tsfm_lab.future_validation import (
        identity,
        integer,
        object_fields,
        text,
        timestamp,
    )

    try:
        fields = {
            "schema_version",
            "kind",
            "run_id",
            "name",
            "status",
            "created_at",
            "updated_at",
            "artifacts",
        }
        object_fields(manifest, fields, "future manifest", optional={"error"})
        if (
            type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != 3
            or manifest["kind"] != "future_forecast"
        ):
            raise StoreError("Unsupported future manifest schema or kind")
        if manifest["run_id"] != run_id or manifest["status"] not in {
            *ACTIVE,
            *TERMINAL,
        }:
            raise StoreError("Invalid future manifest identity or status")
        text(manifest["name"], "future experiment name")
        if timestamp(manifest["updated_at"], "updated_at") < timestamp(
            manifest["created_at"], "created_at"
        ):
            raise StoreError("Future manifest update precedes creation")
        artifacts = manifest["artifacts"]
        if (
            not isinstance(artifacts, dict)
            or not INITIAL <= set(artifacts) <= FUTURE_ARTIFACTS
        ):
            raise StoreError("Invalid future artifact allowlist")
        if manifest["status"] == "completed" and set(artifacts) != FUTURE_ARTIFACTS:
            raise StoreError("Completed future records require all eight artifacts")
        for item in artifacts.values():
            object_fields(item, {"size", "sha256"}, "future artifact identity")
            integer(item["size"], "artifact size", minimum=0)
            identity(item["sha256"], "artifact checksum")
        if "error" in manifest:
            object_fields(
                manifest["error"],
                {"type", "message"},
                "future error",
                optional={"stage"},
            )
            text(manifest["error"]["type"], "error type")
            if not isinstance(manifest["error"]["message"], str):
                raise StoreError("Future error message must be text")
            if (
                "stage" in manifest["error"]
                and manifest["error"]["stage"] not in ACTIVE
            ):
                raise StoreError("Invalid future error stage")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise StoreError(f"Invalid future manifest: {exc}") from exc
    return manifest


class FutureStore(ExperimentStore):
    def create(self, configuration: dict, source_yaml: str) -> str:
        from tsfm_lab.future_validation import validate_configuration

        validate_configuration(configuration)
        if not isinstance(source_yaml, str):
            raise StoreError("Future input YAML must be text")
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
            "schema_version": 3,
            "kind": "future_forecast",
            "run_id": run_id,
            "name": configuration["name"],
            "status": "pending",
            "created_at": now,
            "updated_at": now,
            "artifacts": {name: _identity(raw) for name, raw in artifacts.items()},
        }
        for name, raw in artifacts.items():
            _atomic(directory / name, raw)
        _atomic(directory / "manifest.json", _json(manifest))
        return run_id

    def set_status(self, run_id, status, error=None):
        manifest = self.get_manifest(run_id)
        if manifest["schema_version"] != 3:
            raise StoreError("Future status updates require a future record")
        current = manifest["status"]
        if current in TERMINAL or status == "completed":
            raise StoreError(
                "Terminal runs cannot change; completion requires complete()"
            )
        if status not in {*ACTIVE, "failed", "cancelled"}:
            raise StoreError("Unsupported future status; no evaluation stage exists")
        if status in ACTIVE and ACTIVE.index(status) <= ACTIVE.index(current):
            raise StoreError("Future stages must move forward")
        manifest.update(status=status, updated_at=datetime.now(UTC).isoformat())
        if error is not None:
            manifest["error"] = error
        validate_manifest(manifest, run_id)
        _atomic(self._directory(run_id) / "manifest.json", _json(manifest))
        return manifest

    def _future_result(self, run_id, manifest):
        import io

        from tsfm_lab.future_outputs import validate_outputs
        from tsfm_lab.future_validation import decode_json, validate_future

        contents = self._verified(run_id, manifest)
        try:
            configuration = decode_json(contents["configuration.json"])
            contents["input.yaml"].decode("utf-8")
            result = decode_json(contents["result.json"])
            if (
                _identity(contents["input.yaml"])["sha256"]
                != result["provenance"]["input_sha256"]
            ):
                raise StoreError("Future original input fingerprint disagrees")
            if configuration["name"] != manifest["name"]:
                raise StoreError("Future name disagrees with the initial manifest")
            predictions = pd.read_csv(
                io.BytesIO(contents["predictions.csv"]),
                keep_default_na=False,
                float_precision="round_trip",
                dtype={
                    **{
                        name: str
                        for name in (
                            "request_id",
                            "series_id",
                            "adapter_id",
                            "point_statistic",
                        )
                    },
                    "prediction": "float64",
                },
            )
            observations = pd.read_csv(
                io.BytesIO(contents["observations.csv"]),
                keep_default_na=False,
                float_precision="round_trip",
                dtype={"series_id": str, "observed_value": "float64"},
            )
            predictions, observations = validate_future(
                result,
                predictions,
                observations,
                configuration=configuration,
                run_id=run_id,
            )
            if decode_json(contents["evaluation.json"]) != result["evaluation"]:
                raise StoreError("Future evaluation disagrees across artifacts")
            validate_outputs(
                {name: contents[name] for name in ("forecast.svg", "provenance.json")},
                result,
                predictions,
                observations,
            )
            return result, predictions, observations
        except (KeyError, UnicodeError, TypeError, ValueError, OverflowError) as exc:
            raise StoreError(f"Cannot reload future result: {exc}") from exc

    def load_future_result(self, run_id):
        manifest = self.get_manifest(run_id)
        if manifest["schema_version"] != 3 or manifest["status"] != "completed":
            raise StoreError("Only completed future records have future results")
        return self._future_result(run_id, manifest)

    def complete(self, run_id, result, predictions, observations, *, cancel_check=None):
        from tsfm_lab.future_outputs import build_outputs, validate_outputs
        from tsfm_lab.future_validation import validate_future

        def check():
            if cancel_check and cancel_check():
                raise KeyboardInterrupt("Future cancellation requested while saving")

        manifest = self.get_manifest(run_id)
        if manifest["schema_version"] != 3 or manifest["status"] in TERMINAL:
            raise StoreError("Only active future records can complete")
        check()
        result = _decode(_json(result))
        predictions, observations = validate_future(
            result,
            predictions,
            observations,
            configuration=self.load_configuration(run_id),
            run_id=run_id,
        )
        check()
        if manifest["status"] != "saving":
            manifest = self.set_status(run_id, "saving")
        outputs = build_outputs(result, predictions, observations)
        validate_outputs(outputs, result, predictions, observations)
        artifacts = {
            "result.json": _json(result),
            "predictions.csv": predictions.to_csv(
                index=False, float_format="%.17g", lineterminator="\n"
            ).encode(),
            "observations.csv": observations.to_csv(
                index=False, float_format="%.17g", lineterminator="\n"
            ).encode(),
            "evaluation.json": _json(result["evaluation"]),
            **outputs,
        }
        for name, raw in artifacts.items():
            check()
            _atomic(self._directory(run_id) / name, raw)
            manifest["artifacts"][name] = _identity(raw)
        self._future_result(run_id, manifest)
        check()
        manifest.update(status="completed", updated_at=datetime.now(UTC).isoformat())
        validate_manifest(manifest, run_id)
        _atomic(self._directory(run_id) / "manifest.json", _json(manifest))
        return manifest

    def result_summary(self, run_id):
        result, _, _ = self.load_future_result(run_id)
        manifest = self.get_manifest(run_id)
        return {
            "run_id": run_id,
            "name": manifest["name"],
            "mode": "future_forecast",
            "status": "completed",
            "origin": result["request"]["origin"],
            "horizon": result["counts"]["forecast_rows"],
            "evaluation": result["evaluation"],
            "counts": result["counts"],
            "workspace": str(self.workspace),
            "directory": str(self._directory(run_id)),
            "outputs": {
                name: str(self._directory(run_id) / name)
                for name in sorted(manifest["artifacts"])
            },
        }
