"""Optional, strict Parquet ingestion for a single regular target series."""

from __future__ import annotations

import hashlib
import importlib
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from tsfm_lab.contracts import Dataset, fixed_interval
from tsfm_lab.data import _parse_timestamps, _validate_grid, canonical_fingerprint


def _arrow_modules():
    try:
        return importlib.import_module("pyarrow"), importlib.import_module(
            "pyarrow.parquet"
        )
    except ImportError as exc:
        raise ImportError(
            "Parquet support requires PyArrow; install the optional "
            "'tsfm-lab[parquet]' extra in your environment."
        ) from exc


def _read_table(raw: bytes, columns: tuple[str, str], arrow, parquet):
    try:
        source = parquet.ParquetFile(
            arrow.BufferReader(raw), arrow_extensions_enabled=False
        )
        names = source.schema_arrow.names
        if len(set(names)) != len(names):
            raise ValueError("Parquet contains duplicate field names")
        if any(not name for name in names):
            raise ValueError("Parquet field names must be nonempty")
        for column in columns:
            if column not in names:
                raise ValueError(f"Parquet is missing selected column {column!r}")
        table = source.read(columns=list(columns), use_threads=False)
    except (arrow.ArrowException, OSError) as exc:
        raise ValueError(
            "Cannot read a valid Parquet file from the supplied bytes"
        ) from exc
    if table.num_rows == 0:
        raise ValueError("Parquet must contain at least one observation")
    return table


def _timestamps(column, *, timezone: str | None, name: str, arrow):
    if column.null_count:
        raise ValueError(f"Parquet timestamp column {name!r} contains missing values")
    kind = column.type
    if arrow.types.is_string(kind) or arrow.types.is_large_string(kind):
        return _parse_timestamps(column.to_pylist(), timezone, name)
    if not (arrow.types.is_timestamp(kind) or arrow.types.is_date(kind)):
        raise ValueError(
            f"Parquet timestamp column {name!r} must contain datetimes or ISO strings"
        )
    try:
        index = pd.DatetimeIndex(column.to_pandas())
        if index.tz is not None:
            return index.tz_convert("UTC"), "embedded"
        if timezone is None:
            raise ValueError(f"Naive timestamps in column {name!r} require a timezone")
        index = index.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
        return index.tz_convert("UTC"), timezone
    except Exception as exc:
        # pandas timezone backends raise several exception types for DST failures.
        raise ValueError(f"Cannot localize Parquet column {name!r}: {exc}") from exc


def _values(column, *, name: str, arrow) -> np.ndarray:
    if not (
        arrow.types.is_integer(column.type) or arrow.types.is_floating(column.type)
    ):
        raise ValueError(
            f"Parquet target column {name!r} must contain integer or floating values"
        )
    if column.null_count:
        raise ValueError(f"Parquet target column {name!r} contains missing values")
    values = np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Parquet target column {name!r} must contain finite values")
    return values


def load_parquet(
    path: str | Path,
    *,
    timestamp_column: str,
    target_column: str,
    frequency: str,
    timezone: str | None = None,
    units: str = "unspecified",
    series_id: str = "series_0",
) -> Dataset:
    """Read selected real targets and timestamps without sorting or filling.

    PyArrow is loaded only for this call. Typed datetimes retain their precision;
    naive datetimes need an explicit timezone, and strings follow the CSV loader's
    ISO timestamp policy. Native integer/float targets are accepted, with boolean,
    string, decimal, and nested target schemas explicitly unsupported.
    """
    interval = fixed_interval(frequency)
    for name, value in (
        ("timestamp_column", timestamp_column),
        ("target_column", target_column),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a nonempty string")
    if timestamp_column == target_column:
        raise ValueError("timestamp_column and target_column must differ")
    arrow, parquet = _arrow_modules()
    source = Path(path)
    raw = source.read_bytes()
    table = _read_table(raw, (timestamp_column, target_column), arrow, parquet)
    try:
        timestamps, source_timezone = _timestamps(
            table[timestamp_column],
            timezone=timezone,
            name=timestamp_column,
            arrow=arrow,
        )
        _validate_grid(timestamps, interval, timestamp_column)
    except ValueError as exc:
        message = re.sub(
            r"CSV row (\d+)",
            lambda match: f"Parquet row {int(match.group(1)) - 1}",
            str(exc),
        )
        raise ValueError(message) from exc
    values = _values(table[target_column], name=target_column, arrow=arrow)
    provenance = {
        "source_path": str(source.resolve()),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "parser_policy": "parquet_strict_v1",
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
    return replace(
        dataset,
        provenance={**provenance, "canonical_sha256": canonical_fingerprint(dataset)},
    )
