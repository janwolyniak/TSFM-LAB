"""Strict CSV ingestion and versioned identities for one regular target series."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from tsfm_lab.contracts import Dataset, fixed_interval

_PARSER_POLICY = "csv_utf8_strict_v1"
_ISO_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?)?"
)


def canonical_fingerprint(dataset: Dataset) -> str:
    """Hash canonical UTC nanoseconds, float64 values, and import semantics.

    Source paths and raw file bytes are deliberately excluded. Integer timestamps
    and floating values use little-endian 64-bit encoding; metadata is sorted,
    compact UTF-8 JSON with a versioned serialization identifier.
    """
    metadata = {
        "serialization": "tsfm_dataset_v1",
        "series_id": dataset.series_id,
        "target_name": dataset.target_name,
        "units": dataset.units,
        "frequency_ns": fixed_interval(dataset.frequency).value,
        "parser_policy": dataset.provenance.get("parser_policy", "in_memory_v1"),
        "timestamp_column": dataset.provenance.get("timestamp_column"),
        "target_column": dataset.provenance.get("target_column"),
        "source_timezone": dataset.provenance.get("source_timezone"),
        "observations": len(dataset.values),
    }
    serialized = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    try:
        timestamps = dataset.timestamps.as_unit("ns").asi8.astype("<i8", copy=False)
    except (ValueError, OverflowError) as exc:
        raise ValueError("timestamps must be representable as UTC nanoseconds") from exc
    values = np.asarray(dataset.values, dtype="<f8")
    digest = hashlib.sha256(serialized + b"\n")
    digest.update(timestamps.tobytes(order="C"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _read_rows(raw: bytes) -> tuple[list[str], list[list[str]]]:
    try:
        text = raw.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(reader)
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise ValueError(
            "CSV must contain a UTF-8 header and well-formed rows"
        ) from exc
    if not header or any(not name for name in header):
        raise ValueError("CSV column names must be nonempty")
    if len(set(header)) != len(header):
        raise ValueError("CSV contains duplicate column names")
    if not rows:
        raise ValueError("CSV must contain at least one observation")
    for row_number, row in enumerate(rows, start=2):
        if len(row) != len(header):
            raise ValueError(
                f"CSV row {row_number} has {len(row)} fields; expected {len(header)}"
            )
    return header, rows


def _parse_timestamps(
    values: list[str], timezone: str | None, column: str
) -> tuple[pd.DatetimeIndex, str]:
    timestamps = []
    for row_number, value in enumerate(values, start=2):
        try:
            if not _ISO_TIMESTAMP.fullmatch(value.strip()):
                raise ValueError("expected an ISO date or timestamp")
            timestamp = pd.Timestamp(value)
            if pd.isna(timestamp):
                raise ValueError("missing timestamp")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Invalid timestamp in column {column!r} at CSV row {row_number}"
            ) from exc
        timestamps.append(timestamp)
    awareness = {timestamp.tzinfo is not None for timestamp in timestamps}
    if len(awareness) != 1:
        raise ValueError(f"Column {column!r} mixes timezone-aware and naive timestamps")
    if awareness == {True}:
        return pd.DatetimeIndex([t.tz_convert("UTC") for t in timestamps]), "embedded"
    if timezone is None:
        raise ValueError(f"Naive timestamps in column {column!r} require a timezone")
    try:
        localized = pd.DatetimeIndex(timestamps).tz_localize(
            timezone, ambiguous="raise", nonexistent="raise"
        )
    except Exception as exc:
        # pandas raises backend-specific timezone exceptions as well as ValueError.
        raise ValueError(
            f"Cannot localize column {column!r} with timezone {timezone!r}: {exc}"
        ) from exc
    return localized.tz_convert("UTC"), timezone


def _parse_values(values: list[str], column: str) -> np.ndarray:
    parsed = []
    for row_number, value in enumerate(values, start=2):
        try:
            number = float(value)
            if not np.isfinite(number):
                raise ValueError("nonfinite target")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Column {column!r} must contain finite numeric values; "
                f"invalid target at CSV row {row_number}"
            ) from exc
        parsed.append(number)
    return np.asarray(parsed, dtype=np.float64)


def _validate_grid(
    timestamps: pd.DatetimeIndex, interval: pd.Timedelta, column: str
) -> None:
    if timestamps.has_duplicates:
        duplicate = int(np.flatnonzero(timestamps.duplicated())[0]) + 2
        raise ValueError(
            f"Duplicate timestamp in column {column!r} at CSV row {duplicate}"
        )
    differences = timestamps[1:] - timestamps[:-1]
    for position, difference in enumerate(differences, start=3):
        if difference <= pd.Timedelta(0):
            raise ValueError(
                f"Column {column!r} must be strictly ascending; CSV row {position}"
            )
        if difference != interval:
            raise ValueError(
                f"Column {column!r} has an irregular interval at CSV row {position}"
            )


def load_csv(
    path: str | Path,
    *,
    timestamp_column: str,
    target_column: str,
    frequency: str,
    timezone: str | None = None,
    units: str = "unspecified",
    series_id: str = "series_0",
) -> Dataset:
    """Load ISO-dated CSV rows without sorting, filling, or inferring frequency."""
    interval = fixed_interval(frequency)
    if not isinstance(timestamp_column, str) or not timestamp_column:
        raise ValueError("timestamp_column must be a nonempty string")
    if not isinstance(target_column, str) or not target_column:
        raise ValueError("target_column must be a nonempty string")
    if timestamp_column == target_column:
        raise ValueError("timestamp_column and target_column must differ")
    source = Path(path)
    raw = source.read_bytes()
    header, rows = _read_rows(raw)
    for column in (timestamp_column, target_column):
        if column not in header:
            raise ValueError(f"CSV is missing selected column {column!r}")
    time_index = header.index(timestamp_column)
    value_index = header.index(target_column)
    timestamps, source_timezone = _parse_timestamps(
        [row[time_index] for row in rows], timezone, timestamp_column
    )
    _validate_grid(timestamps, interval, timestamp_column)
    values = _parse_values([row[value_index] for row in rows], target_column)
    provenance = {
        "source_path": str(source.resolve()),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "parser_policy": _PARSER_POLICY,
        "timestamp_column": timestamp_column,
        "target_column": target_column,
        "source_timezone": source_timezone,
    }
    dataset = Dataset(
        timestamps=timestamps,
        values=values,
        frequency=frequency,
        target_name=target_column,
        units=units,
        series_id=series_id,
        provenance=provenance,
    )
    provenance["canonical_sha256"] = canonical_fingerprint(dataset)
    return Dataset(
        timestamps=dataset.timestamps,
        values=dataset.values,
        frequency=dataset.frequency,
        target_name=dataset.target_name,
        units=dataset.units,
        series_id=dataset.series_id,
        provenance=provenance,
    )
