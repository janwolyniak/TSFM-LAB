"""Bounded, path-free provider suggestions with no execution authority."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass

from tsfm_lab.assistant.resolution import (
    DraftResolution as DraftResolution,
)
from tsfm_lab.assistant.resolution import (
    resolve_draft as resolve_draft,
)

MAX_DRAFT_BYTES = 32768
ALLOWED_FIELDS = frozenset(
    {
        "name",
        "usage_profile",
        "adapter",
        "repository",
        "checkpoint_variant",
        "point_statistic",
        "seasonal_period",
        "timestamp_column",
        "target_column",
        "frequency",
        "timezone",
        "units",
        "context_length",
        "horizon",
        "stride",
        "mase_period",
        "evaluation_start",
        "evaluation_end",
        "finance_target_kind",
        "finance_measurement",
        "energy_measurement",
        "energy_quantity",
        "weather_variable",
        "weather_station",
        "weather_measurement_semantics",
        "weather_description",
    }
)
HISTORICAL_ONLY_FIELDS = frozenset(
    {"stride", "mase_period", "evaluation_start", "evaluation_end"}
)
FUTURE_FIELDS = (ALLOWED_FIELDS - HISTORICAL_ONLY_FIELDS) | {"origin"}
MISSING_FIELDS = ALLOWED_FIELDS | FUTURE_FIELDS | {"dataset_ref", "intent"}
_TOP_FIELDS = {
    "schema_version",
    "state",
    "intent",
    "dataset_ref",
    "fields",
    "explicit_fields",
    "assumptions",
    "missing_fields",
}


class DraftError(ValueError):
    """The provider did not return a supported experiment suggestion."""


@dataclass(frozen=True)
class ExperimentDraft:
    schema_version: int
    state: str
    intent: str
    dataset_ref: str | None
    fields: dict[str, str]
    explicit_fields: tuple[str, ...]
    assumptions: tuple[str, ...]
    missing_fields: tuple[str, ...]


def _text(value, label, *, maximum=500):
    if (
        type(value) is not str
        or len(value) > maximum
        or any(unicodedata.category(character) in {"Cc", "Cs"} for character in value)
    ):
        raise DraftError(f"{label} must be bounded text without control characters")
    return value


def _list(value, label, *, allowed=None):
    if type(value) is not list or len(value) > 40:
        raise DraftError(f"{label} must be a list with at most 40 entries")
    entries = tuple(_text(item, label) for item in value)
    if len(set(entries)) != len(entries):
        raise DraftError(f"{label} must not contain duplicates")
    if allowed is not None and set(entries) - set(allowed):
        raise DraftError(f"{label} contains unsupported field names")
    return entries


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DraftError("Draft JSON contains a duplicate key")
        result[key] = value
    return result


def _constant(_value):
    raise DraftError("Draft JSON must contain ordinary finite JSON values")


def decode_draft(text: str) -> ExperimentDraft:
    """Decode strict JSON; provider status and provenance remain advisory."""
    if type(text) is not str:
        raise DraftError("The experiment draft must be JSON text")
    try:
        if len(text.encode("utf-8")) > MAX_DRAFT_BYTES:
            raise DraftError("The experiment draft exceeds the local size limit")
        raw = json.loads(text, object_pairs_hook=_object, parse_constant=_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise DraftError("The experiment draft must be one valid JSON object") from exc
    if type(raw) is not dict or set(raw) != _TOP_FIELDS:
        raise DraftError("The experiment draft has missing or unknown top-level fields")
    if type(raw["schema_version"]) is not int or raw["schema_version"] not in {1, 2}:
        raise DraftError("Only ExperimentDraft schema_version 1 or 2 is supported")
    if raw["state"] not in ("needs_input", "ready", "unsupported"):
        raise DraftError("Unsupported experiment draft state")
    if raw["intent"] not in ("historical_evaluation", "future_forecast", "unclear"):
        raise DraftError("Unsupported experiment draft intent")
    if raw["schema_version"] == 2 and raw["intent"] not in {
        "future_forecast",
        "unclear",
    }:
        raise DraftError("Future draft schema 2 requires future intent")
    reference = raw["dataset_ref"]
    if reference is not None:
        _text(reference, "dataset_ref", maximum=100)
    fields = raw["fields"]
    allowed = FUTURE_FIELDS if raw["schema_version"] == 2 else ALLOWED_FIELDS
    if type(fields) is not dict or set(fields) - allowed:
        raise DraftError("Experiment draft fields contain unknown or forbidden fields")
    fields = {key: _text(value, key) for key, value in fields.items()}
    return ExperimentDraft(
        raw["schema_version"],
        raw["state"],
        raw["intent"],
        reference,
        fields,
        _list(raw["explicit_fields"], "explicit_fields", allowed=fields),
        _list(raw["assumptions"], "assumptions"),
        _list(
            raw["missing_fields"],
            "missing_fields",
            allowed=allowed | {"dataset_ref", "intent"},
        ),
    )


def draft_contract(mode="historical_evaluation") -> dict:
    """Return a fresh JSON schema for the exact path-free provider response."""
    if mode not in {"historical_evaluation", "future_forecast"}:
        raise DraftError("Choose a supported forecast mode")
    future = mode == "future_forecast"
    names = sorted(FUTURE_FIELDS if future else ALLOWED_FIELDS)
    field_list = {
        "type": "array",
        "items": {"type": "string", "enum": names},
        "maxItems": 40,
        "uniqueItems": True,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_TOP_FIELDS),
        "properties": {
            "schema_version": {"type": "integer", "const": 2 if future else 1},
            "state": {
                "type": "string",
                "enum": ["needs_input", "ready", "unsupported"],
            },
            "intent": {
                "type": "string",
                "enum": (
                    ["future_forecast", "unclear"]
                    if future
                    else ["historical_evaluation", "future_forecast", "unclear"]
                ),
            },
            "dataset_ref": {"type": ["string", "null"], "maxLength": 100},
            "fields": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    name: {"type": "string", "maxLength": 500} for name in names
                },
            },
            "explicit_fields": field_list,
            "assumptions": {
                "type": "array",
                "items": {"type": "string", "maxLength": 500},
                "maxItems": 40,
                "uniqueItems": True,
            },
            "missing_fields": {
                **field_list,
                "items": {
                    "type": "string",
                    "enum": sorted(set(names) | {"dataset_ref", "intent"}),
                },
            },
        },
    }
