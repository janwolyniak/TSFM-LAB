"""Strict positive-price OHLC bars with explicit opening and closing times."""

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tsfm_lab.contracts import Dataset, fixed_interval
from tsfm_lab.data import _parse_timestamps, _parse_values, _read_rows
from tsfm_lab.finance.contracts import FinanceError

_COLUMNS = ("open_timestamp", "close_timestamp", "open", "high", "low", "close")
_PARSER_POLICY = "ohlc_csv_utf8_strict_v1"
_PRICES = ("opens", "highs", "lows", "closes")


def _times(values, name):
    try:
        timestamps = pd.DatetimeIndex(values).copy(deep=True)
        if timestamps.empty or timestamps.hasnans or timestamps.tz is None:
            raise ValueError("timestamps must be nonempty and timezone-aware")
        if not timestamps.is_unique or not timestamps.is_monotonic_increasing:
            raise ValueError("timestamps must be unique and strictly ascending")
        return timestamps.tz_convert("UTC").as_unit("ns")
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinanceError(f"Invalid {name}: {exc}") from exc


def _prices(values, name):
    try:
        if np.ma.isMaskedArray(values) and np.ma.getmaskarray(values).any():
            raise ValueError("masked prices are unsupported")
        array = np.asarray(values)
        if array.ndim != 1 or not array.size or array.dtype.kind not in "iuf":
            raise ValueError("prices must be a nonempty real numeric vector")
        # Mixed boolean/numeric lists lose their boolean dtype under np.asarray.
        if any(isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError("boolean prices are unsupported")
        result = np.array(array, dtype=np.float64, copy=True)
        if not np.isfinite(result).all() or (result <= 0).any():
            raise ValueError("prices must be finite and strictly positive")
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinanceError(f"Invalid {name}: {exc}") from exc
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class OHLCData:
    open_timestamps: pd.DatetimeIndex
    close_timestamps: pd.DatetimeIndex
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    frequency: str
    units: str = "unspecified"
    series_id: str = "series_0"
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        try:
            interval = fixed_interval(self.frequency)
            for name in ("units", "series_id"):
                value = getattr(self, name)
                if not isinstance(value, str) or not value.strip():
                    raise FinanceError(f"{name} must be a nonempty string")
            if not isinstance(self.provenance, dict):
                raise FinanceError("provenance must be a dictionary")
            for name in ("open_timestamps", "close_timestamps"):
                object.__setattr__(self, name, _times(getattr(self, name), name))
            for name in _PRICES:
                object.__setattr__(self, name, _prices(getattr(self, name), name))
            length = len(self.close_timestamps)
            if any(
                len(getattr(self, name)) != length
                for name in ("open_timestamps", *_PRICES)
            ):
                raise FinanceError("OHLC columns must have equal lengths")
            if not (
                (self.close_timestamps[1:] - self.close_timestamps[:-1]) == interval
            ).all():
                raise FinanceError(
                    "close_timestamps must follow the declared regular frequency"
                )
            if not (self.open_timestamps < self.close_timestamps).all():
                raise FinanceError("Each bar must open strictly before it closes")
            if not (self.open_timestamps[1:] >= self.close_timestamps[:-1]).all():
                raise FinanceError("OHLC bars must not overlap")
            if (self.highs < np.maximum(self.opens, self.closes)).any():
                raise FinanceError("highs must be at least each bar's open and close")
            if (self.lows > np.minimum(self.opens, self.closes)).any():
                raise FinanceError("lows must be at most each bar's open and close")
            object.__setattr__(self, "provenance", copy.deepcopy(self.provenance))
        except (TypeError, ValueError, OverflowError) as exc:
            if isinstance(exc, FinanceError):
                raise
            raise FinanceError(str(exc)) from exc


def ohlc_fingerprint(data: OHLCData) -> str:
    """Hash exact UTC nanoseconds, float64 prices, and versioned import semantics."""
    metadata = {
        "serialization": "tsfm_ohlc_v1",
        "series_id": data.series_id,
        "units": data.units,
        "frequency_ns": fixed_interval(data.frequency).value,
        "observations": len(data.closes),
        "parser_policy": data.provenance.get("parser_policy", "in_memory_v1"),
        "open_source_timezone": data.provenance.get("open_source_timezone"),
        "close_source_timezone": data.provenance.get("close_source_timezone"),
        "columns": list(_COLUMNS),
    }
    digest = hashlib.sha256(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    for times in (data.open_timestamps, data.close_timestamps):
        digest.update(
            times.as_unit("ns").asi8.astype("<i8", copy=False).tobytes(order="C")
        )
    for name in _PRICES:
        digest.update(np.asarray(getattr(data, name), dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def load_ohlc_csv(
    path: str | Path,
    *,
    frequency: str,
    timezone: str | None = None,
    units: str = "unspecified",
    series_id: str = "series_0",
) -> OHLCData:
    """Read fixed OHLC columns without sorting, filling, or inferring timing."""
    try:
        fixed_interval(frequency)
        source = Path(path)
        raw = source.read_bytes()
        header, rows = _read_rows(raw)
        if set(header) != set(_COLUMNS):
            raise FinanceError(
                f"OHLC CSV requires exactly these columns: {', '.join(_COLUMNS)}"
            )
        columns = {name: [row[header.index(name)] for row in rows] for name in _COLUMNS}
        opens, open_timezone = _parse_timestamps(
            columns["open_timestamp"], timezone, "open_timestamp"
        )
        closes, close_timezone = _parse_timestamps(
            columns["close_timestamp"], timezone, "close_timestamp"
        )
        provenance = {
            "source_path": str(source.resolve()),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "parser_policy": _PARSER_POLICY,
            "open_source_timezone": open_timezone,
            "close_source_timezone": close_timezone,
        }
        data = OHLCData(
            open_timestamps=opens,
            close_timestamps=closes,
            **{
                name + "s": _parse_values(columns[name], name)
                for name in ("open", "high", "low", "close")
            },
            frequency=frequency,
            units=units,
            series_id=series_id,
            provenance=provenance,
        )
        data.provenance["canonical_sha256"] = ohlc_fingerprint(data)
        return data
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, FinanceError):
            raise
        raise FinanceError(f"Invalid OHLC CSV: {exc}") from exc


def validate_price_alignment(dataset: Dataset, ohlc: OHLCData) -> None:
    """Require the complete source price series to equal the OHLC close series."""
    if not isinstance(dataset, Dataset) or not isinstance(ohlc, OHLCData):
        raise FinanceError("Price alignment requires a Dataset and OHLCData")
    if dataset.series_id != ohlc.series_id or dataset.units != ohlc.units:
        raise FinanceError("Price and OHLC series identity/units must match")
    if fixed_interval(dataset.frequency) != fixed_interval(ohlc.frequency):
        raise FinanceError("Price and OHLC frequency must match")
    try:
        times = dataset.timestamps.tz_convert("UTC").as_unit("ns")
    except (ValueError, OverflowError) as exc:
        raise FinanceError("Price timestamps must fit UTC nanoseconds") from exc
    if not times.equals(ohlc.close_timestamps):
        raise FinanceError("Price timestamps must exactly match every OHLC close")
    if not np.array_equal(dataset.values, ohlc.closes):
        raise FinanceError("Price values must exactly match every OHLC close")
