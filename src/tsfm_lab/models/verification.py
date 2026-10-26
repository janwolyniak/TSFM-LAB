"""Detached project-check snapshots, separate from consumed checkpoint identity."""

import json
import math
import re

from tsfm_lab.models._verification_evidence import PROJECT_CHECKS
from tsfm_lab.models.checkpoint_profiles import checkpoint_adapter

_FILES = {"config.json", "model.safetensors"}
_FIELDS = {
    "schema_version",
    "adapter_id",
    "checkpoint_fingerprint",
    "compatibility",
    "project_inference_status",
    "selected_checkpoint_checks",
    "adapter_representative_checks",
}
_CHECK_FIELDS = {
    "id",
    "adapter_id",
    "repository",
    "revision",
    "files",
    "source",
    "runtime",
    "device",
    "dtype",
    "checks",
    "reference",
}


class VerificationError(ValueError):
    """A project-check snapshot is malformed or contradicts its checkpoint."""


def _native(value):
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise VerificationError("Project verification keys must be strings")
        for item in value.values():
            _native(item)
    elif isinstance(value, list):
        for item in value:
            _native(item)
    elif type(value) not in {str, int, float, bool, type(None)}:
        raise VerificationError("Project verification requires native JSON values")
    elif type(value) is float and not math.isfinite(value):
        raise VerificationError("Project verification cannot contain nonfinite values")


def _snapshot(value):
    try:
        _native(value)
        raw = json.dumps(value, allow_nan=False)
        if len(raw.encode("utf-8")) > 1024 * 1024:
            raise VerificationError("Project verification exceeds the 1 MiB limit")
        return json.loads(raw)
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        if isinstance(exc, VerificationError):
            raise
        raise VerificationError(f"Invalid project verification JSON: {exc}") from exc


def _object(value, fields, name):
    if not isinstance(value, dict) or set(value) != fields:
        raise VerificationError(f"Invalid {name} fields")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise VerificationError(f"{name} must be a nonempty string")


def _hash(value, length, name):
    if (
        not isinstance(value, str)
        or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None
    ):
        raise VerificationError(f"Invalid {name} identity")


def _check(item, adapter_id):
    _object(item, _CHECK_FIELDS, "project inference check")
    for name in ("id", "repository", "device", "dtype", "reference"):
        _text(item[name], name)
    if item["adapter_id"] != adapter_id:
        raise VerificationError("Project inference evidence belongs to another adapter")
    _hash(item["revision"], 40, "tested checkpoint revision")
    _object(item["files"], _FILES, "tested files")
    for identity in item["files"].values():
        _object(identity, {"sha256", "size"}, "tested file identity")
        _hash(identity["sha256"], 64, "tested file")
        if type(identity["size"]) is not int or identity["size"] <= 0:
            raise VerificationError("Tested file size must be a positive integer")
    source = item["source"]
    _object(source, {"package_version", "git_revision", "git_dirty"}, "tested source")
    if source["package_version"] is not None:
        _text(source["package_version"], "tested package version")
    if source["git_revision"] is not None:
        _hash(source["git_revision"], 40, "tested source revision")
    if source["git_dirty"] is not None and type(source["git_dirty"]) is not bool:
        raise VerificationError("Tested git_dirty must be boolean or null")
    if not isinstance(item["runtime"], dict) or not item["runtime"]:
        raise VerificationError("Tested runtime must be a nonempty object")
    for name, value in item["runtime"].items():
        _text(name, "runtime distribution")
        _text(value, "tested runtime version")
    if not isinstance(item["checks"], list) or not item["checks"]:
        raise VerificationError("Executed checks must be a nonempty list")
    for value in item["checks"]:
        _text(value, "executed check")


def _checks(value, adapter_id):
    if not isinstance(value, list):
        raise VerificationError("Project checks must be a list")
    result = {}
    for item in value:
        _check(item, adapter_id)
        if item["id"] in result:
            raise VerificationError("Project check IDs must be unique")
        result[item["id"]] = item
    return result


def _matches(item, checkpoint):
    return all(item["files"][name] == checkpoint["files"][name] for name in _FILES)


def validate_verification_record(value, checkpoint_record, adapter_id):
    """Check an embedded snapshot without consulting today's evidence registry."""
    from tsfm_lab.models.checkpoints import validate_checkpoint_record

    try:
        checkpoint = validate_checkpoint_record(checkpoint_record)
        if checkpoint_adapter(checkpoint["architecture"]) != adapter_id:
            raise VerificationError("Verification adapter disagrees with checkpoint")
        result = _snapshot(value)
        _object(result, _FIELDS, "project verification")
        if type(result["schema_version"]) is not int or result["schema_version"] != 1:
            raise VerificationError("Unsupported project verification schema")
        if (
            result["adapter_id"] != adapter_id
            or result["checkpoint_fingerprint"] != checkpoint["fingerprint"]
        ):
            raise VerificationError(
                "Project verification disagrees with selected checkpoint"
            )
        if result["compatibility"] != {
            "status": "checked",
            "scope": "metadata_and_loader_profile",
        }:
            raise VerificationError("Unsupported checkpoint compatibility scope")
        selected = _checks(result["selected_checkpoint_checks"], adapter_id)
        representatives = _checks(result["adapter_representative_checks"], adapter_id)
        expected = "recorded" if selected else "not_recorded"
        if result["project_inference_status"] != expected:
            raise VerificationError(
                "Project inference status disagrees with recorded checks"
            )
        for identifier, item in selected.items():
            if (
                not _matches(item, checkpoint)
                or representatives.get(identifier) != item
            ):
                raise VerificationError(
                    "Selected-checkpoint inference evidence does not match"
                )
        return result
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        if isinstance(exc, VerificationError):
            raise
        raise VerificationError(f"Invalid project verification: {exc}") from exc


def checkpoint_verification(checkpoint_record):
    """Snapshot maintainer evidence for exact config/weight bytes, including local copies."""
    from tsfm_lab.models.checkpoints import validate_checkpoint_record

    checkpoint = validate_checkpoint_record(checkpoint_record)
    adapter_id = checkpoint_adapter(checkpoint["architecture"])
    representatives = _snapshot(
        [item for item in PROJECT_CHECKS if item["adapter_id"] == adapter_id]
    )
    selected = [item for item in representatives if _matches(item, checkpoint)]
    value = {
        "schema_version": 1,
        "adapter_id": adapter_id,
        "checkpoint_fingerprint": checkpoint["fingerprint"],
        "compatibility": {"status": "checked", "scope": "metadata_and_loader_profile"},
        "project_inference_status": "recorded" if selected else "not_recorded",
        "selected_checkpoint_checks": selected,
        "adapter_representative_checks": representatives,
    }
    return validate_verification_record(value, checkpoint, adapter_id)


def verification_label(value):
    """Use the same project-status wording in shell, guided setup, and history."""
    if value is None:
        return "Project verification was not recorded"
    if value["project_inference_status"] == "recorded":
        return "Project inference check recorded for these config/weight files"
    return "Compatibility checked; project inference check not recorded for these files"
