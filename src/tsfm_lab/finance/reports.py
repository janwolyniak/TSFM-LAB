"""Self-contained finance records with independent recalculation and atomic saves."""

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pandas as pd

from tsfm_lab.artifact_validation import _schema
from tsfm_lab.evaluation.metrics import METRIC_VERSION, score_with_scale
from tsfm_lab.experiments import _json as _core_json
from tsfm_lab.finance.analysis import EXECUTION_POLICY, evaluate_cases
from tsfm_lab.finance.contracts import POLICY, BacktestRules, FinanceReportError
from tsfm_lab.local_paths import expand_local_user

MAX_REPORT_BYTES = 64 * 1024 * 1024
_FIELDS = {
    "schema_version",
    "policy",
    "report_id",
    "created_at",
    "implementation",
    "parent",
    "original_dataset",
    "execution_data",
    "rules",
    "resolved_policy",
    "cases",
    "evaluation",
}
_ARTIFACTS = {
    "configuration.json",
    "input.yaml",
    "metrics.json",
    "result.json",
    "predictions.csv",
}


def _native_json(value):
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise FinanceReportError("Report object keys must be strings")
        for item in value.values():
            _native_json(item)
    elif isinstance(value, list):
        for item in value:
            _native_json(item)
    elif type(value) not in {str, int, float, bool, type(None)}:
        raise FinanceReportError("Reports require native JSON values")
    elif type(value) is float and not math.isfinite(value):
        raise FinanceReportError("Reports cannot contain nonfinite numbers")


def _canonical(value):
    try:
        _native_json(value)
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if len(raw) > MAX_REPORT_BYTES:
            raise FinanceReportError("Finance report exceeds the 64 MiB size limit")
        return raw
    except (ValueError, TypeError, OverflowError, RecursionError, UnicodeError) as exc:
        if isinstance(exc, FinanceReportError):
            raise
        raise FinanceReportError(f"Invalid finance JSON: {exc}") from exc


def _object(value, fields, label):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise FinanceReportError(f"Invalid {label} fields")
    return value


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise FinanceReportError(f"{label} must be a nonempty string")


def _sha(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise FinanceReportError(f"Invalid {label} SHA-256 identity")


def _uuid(value, label):
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise FinanceReportError(f"{label} must be a canonical UUID")


def _provenance(value, fingerprint, label):
    if not isinstance(value, dict):
        raise FinanceReportError(f"{label} provenance must be an object")
    for name in ("source_sha256", "canonical_sha256"):
        if name in value:
            _sha(value[name], label)
    if "canonical_sha256" in value and value["canonical_sha256"] != fingerprint:
        raise FinanceReportError(f"{label} provenance identity disagrees")
    for name in (
        "source_path",
        "parser_policy",
        "source_timezone",
        "open_source_timezone",
        "close_source_timezone",
    ):
        if name in value:
            _text(value[name], f"{label} {name}")


def _implementation(value):
    _object(
        value,
        {"package_version", "git_revision", "git_dirty", "environment"},
        "implementation",
    )
    _text(value["package_version"], "Implementation package version")
    if value["git_revision"] is not None:
        _text(value["git_revision"], "Implementation Git revision")
    if value["git_dirty"] is not None and type(value["git_dirty"]) is not bool:
        raise FinanceReportError("Implementation git_dirty must be boolean or null")
    environment = _object(
        value["environment"],
        {"python", "numpy", "pandas", "os", "machine"},
        "implementation environment",
    )
    for name, item in environment.items():
        _text(item, f"Implementation environment {name}")


def _parent(payload):
    parent = _object(payload["parent"], {"run_id", "result", "artifacts"}, "parent")
    _uuid(parent["run_id"], "Parent run_id")
    result = parent["result"]
    config, spec, interval = _schema(result)
    if spec.point_statistic != "median":
        raise FinanceReportError("Finance reports require the saved median task")
    if config["dataset"]["units"] == "unspecified":
        raise FinanceReportError("Declare price quote units in the parent dataset")
    if result.get("run_id", parent["run_id"]) != parent["run_id"]:
        raise FinanceReportError("Parent run identity disagrees")
    if spec.stride != 1:
        raise FinanceReportError("Finance reports require parent stride 1")
    artifacts = _object(parent["artifacts"], _ARTIFACTS, "parent artifacts")
    for name, identity in artifacts.items():
        _object(identity, {"sha256", "size"}, f"{name} identity")
        _sha(identity["sha256"], name)
        if type(identity["size"]) is not int or identity["size"] < 0:
            raise FinanceReportError("Artifact sizes must be nonnegative integers")
    for name, value in (("result.json", result), ("metrics.json", result["metrics"])):
        raw = _core_json(value)
        if artifacts[name] != {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
        }:
            raise FinanceReportError(f"Copied parent {name} identity disagrees")
    length, remainder = divmod(
        (spec.evaluation_end - spec.evaluation_start).value, interval.value
    )
    windows = length - spec.horizon + 1
    if remainder or windows <= 0:
        raise FinanceReportError("Parent evaluation bounds must define a complete grid")
    expected = {
        "attempted_windows": windows,
        "successful_windows": windows,
        "failed_windows": 0,
        "forecast_cases": windows * spec.horizon,
        "unique_target_timestamps": length,
    }
    if result["counts"] != expected:
        raise FinanceReportError("Parent counts disagree with its evaluation grid")
    original = _object(
        payload["original_dataset"],
        {"canonical_sha256", "metadata", "provenance", "observations"},
        "original dataset",
    )
    execution = _object(
        payload["execution_data"],
        {"canonical_sha256", "provenance", "observations"},
        "execution data",
    )
    fingerprint = result["provenance"]["dataset"]["canonical_sha256"]
    if original["canonical_sha256"] != fingerprint or _canonical(
        original["metadata"]
    ) != _canonical(config["dataset"]):
        raise FinanceReportError("Original dataset disagrees with the parent identity")
    for label, value in (("Original dataset", original), ("Execution data", execution)):
        _sha(value["canonical_sha256"], label)
        if type(value["observations"]) is not int or value["observations"] <= 0:
            raise FinanceReportError(
                "Dataset observation counts must be positive integers"
            )
        _provenance(value["provenance"], value["canonical_sha256"], label)
    required = result["provenance"]["scale_history"]["observations"] + length
    if (
        original["observations"] < required
        or execution["observations"] != original["observations"]
    ):
        raise FinanceReportError(
            "Dataset observations do not cover the parent evaluation"
        )
    return config, spec, interval, windows


def _equal(actual, expected, label="evaluation"):
    if isinstance(expected, dict):
        _object(actual, expected, label)
        for name in expected:
            _equal(actual[name], expected[name], f"{label}.{name}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise FinanceReportError(f"{label} has an inconsistent list length")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _equal(left, right, f"{label}[{index}]")
    elif type(expected) is float:
        if type(actual) not in {int, float} or not math.isclose(
            actual, expected, rel_tol=1e-12, abs_tol=0
        ):
            raise FinanceReportError(f"{label} disagrees with recalculated arithmetic")
    elif type(actual) is not type(expected) or actual != expected:
        raise FinanceReportError(f"{label} disagrees with recalculated metadata")


def _lead_metrics(cases, result, spec):
    if result["provenance"]["metric_version"] != METRIC_VERSION:
        raise FinanceReportError("Unsupported parent point metric version")
    metrics = result["metrics"]
    scores = score_with_scale(
        [[case["actual_close"]] for case in cases],
        [[case["forecast_close"]] for case in cases],
        scale=metrics["scale"],
        scale_reason=metrics["scale_reason"],
        scale_count=metrics["scale_count"],
        metrics=spec.metrics,
    )
    _equal(
        metrics["by_lead"]["1"],
        {name: asdict(value) for name, value in scores.overall.items()},
        "parent lead-1 metrics",
    )


def _validate_payload(payload):
    try:
        _object(payload, _FIELDS, "finance report")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != 1
            or payload["policy"] != POLICY
        ):
            raise FinanceReportError("Unsupported finance report schema or policy")
        _uuid(payload["report_id"], "Report ID")
        _text(payload["created_at"], "created_at")
        created = datetime.fromisoformat(payload["created_at"])
        if created.tzinfo is None or created.utcoffset() is None:
            raise FinanceReportError(
                "created_at must be an ISO timezone-aware timestamp"
            )
        _implementation(payload["implementation"])
        config, spec, interval, windows = _parent(payload)
        _object(payload["rules"], BacktestRules.__dataclass_fields__, "rules")
        rules = BacktestRules(**payload["rules"])
        if _canonical(payload["resolved_policy"]) != _canonical(EXECUTION_POLICY):
            raise FinanceReportError(
                "Resolved policy disagrees with the implementation"
            )
        cases = payload["cases"]
        if not isinstance(cases, list) or len(cases) != windows:
            raise FinanceReportError(
                "Finance cases must retain every successful origin"
            )
        expected = evaluate_cases(
            cases,
            rules,
            frequency=config["dataset"]["frequency"],
            series_id=config["dataset"]["series_id"],
        )
        for index, case in enumerate(cases):
            target = spec.evaluation_start + index * interval
            if (
                pd.Timestamp(case["target_timestamp"]) != target
                or pd.Timestamp(case["origin"]) != target - interval
            ):
                raise FinanceReportError(
                    "Finance cases disagree with the parent origin grid"
                )
        _lead_metrics(cases, payload["parent"]["result"], spec)
        _equal(payload["evaluation"], expected)
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OverflowError,
        RecursionError,
    ) as exc:
        if isinstance(exc, FinanceReportError):
            raise
        raise FinanceReportError(f"Invalid finance report: {exc}") from exc


def seal_report(payload: dict) -> dict:
    """Snapshot and seal a valid payload, without repairing calculated results."""
    if not isinstance(payload, dict):
        raise FinanceReportError("A finance report must be an object")
    snapshot = json.loads(
        _canonical(
            {key: value for key, value in payload.items() if key != "fingerprint"}
        )
    )
    _validate_payload(snapshot)
    snapshot["fingerprint"] = hashlib.sha256(_canonical(snapshot)).hexdigest()
    _canonical(snapshot)
    return snapshot


def validate_report(report: dict) -> dict:
    """Validate a detached report using only its embedded data and calculation code."""
    snapshot = json.loads(_canonical(report))
    _object(snapshot, _FIELDS | {"fingerprint"}, "sealed finance report")
    fingerprint = snapshot["fingerprint"]
    _sha(fingerprint, "report")
    payload = {key: value for key, value in snapshot.items() if key != "fingerprint"}
    if hashlib.sha256(_canonical(payload)).hexdigest() != fingerprint:
        raise FinanceReportError("Finance report fingerprint mismatch")
    _validate_payload(payload)
    return snapshot


def _publish_bytes(raw: bytes, destination: str | Path) -> dict:
    """Atomically publish bounded bytes without replacing a destination."""
    if not isinstance(raw, bytes) or len(raw) > MAX_REPORT_BYTES:
        raise FinanceReportError("Finance output must be bytes within the 64 MiB limit")
    try:
        target = expand_local_user(destination).absolute()
    except (OSError, ValueError, RuntimeError) as exc:
        raise FinanceReportError(f"Invalid finance destination: {exc}") from exc
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Finance destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".tsfm-finance-", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(target),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def save_report(report: dict, destination: str | Path) -> dict:
    """Recalculate before saving; an existing file or symlink is never replaced."""
    return _publish_bytes(_canonical(validate_report(report)) + b"\n", destination)


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise FinanceReportError(f"Duplicate finance JSON key: {key}")
        result[key] = value
    return result


def load_report(path: str | Path) -> dict:
    """Reload bounded UTF-8 JSON without accessing any referenced source paths."""
    try:
        source = Path(path).expanduser()
        if source.is_symlink() or not source.is_file():
            raise FinanceReportError("Finance report must be a regular nonsymlink file")
        with source.open("rb") as stream:
            raw = stream.read(MAX_REPORT_BYTES + 1)
        if len(raw) > MAX_REPORT_BYTES:
            raise FinanceReportError("Finance report exceeds the 64 MiB size limit")

        def invalid_constant(value):
            raise FinanceReportError(f"Nonfinite finance JSON value: {value}")

        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=invalid_constant,
        )
        return validate_report(value)
    except (
        OSError,
        ValueError,
        TypeError,
        OverflowError,
        RecursionError,
        UnicodeError,
    ) as exc:
        if isinstance(exc, FinanceReportError):
            raise
        raise FinanceReportError(f"Cannot load finance report: {exc}") from exc
