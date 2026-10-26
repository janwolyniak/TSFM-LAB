"""Validated numerical contracts shared by data, models, and evaluation."""

from dataclasses import dataclass, field
from numbers import Integral
from typing import Literal

import numpy as np
import pandas as pd

from tsfm_lab.point_tasks import output_statistics


def positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def fixed_interval(frequency: str) -> pd.Timedelta:
    if not isinstance(frequency, str):
        raise ValueError("frequency must be a positive fixed duration string")
    try:
        offset = pd.tseries.frequencies.to_offset(frequency)
        nanos = offset.nanos
        if nanos <= 0:
            raise ValueError("frequency must be positive")
        return pd.Timedelta(nanos, unit="ns")
    except (TypeError, ValueError, AttributeError, OverflowError) as exc:
        raise ValueError("frequency must be a positive fixed duration") from exc


def utc_timestamp(value: str | pd.Timestamp, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{name} must be a valid timezone-aware timestamp")
    return timestamp.tz_convert("UTC")


def _times(values: pd.DatetimeIndex, name: str) -> pd.DatetimeIndex:
    try:
        index = pd.DatetimeIndex(values).copy(deep=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain timezone-aware timestamps") from exc
    if index.empty or index.hasnans or index.tz is None:
        raise ValueError(f"{name} must contain nonempty timezone-aware timestamps")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError(f"{name} must be unique and strictly ascending")
    return index.tz_convert("UTC")


def _values(values: np.ndarray, name: str) -> np.ndarray:
    try:
        original = np.asarray(values)
        if original.dtype.kind not in "iuf":
            raise ValueError(f"{name} must contain real numeric values")
        result = np.array(original, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite numeric vector")
    result.setflags(write=False)
    return result


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class Dataset:
    timestamps: pd.DatetimeIndex
    values: np.ndarray
    frequency: str
    target_name: str = "target"
    units: str = "unspecified"
    series_id: str = "series_0"
    provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        times = _times(self.timestamps, "timestamps")
        values = _values(self.values, "values")
        if len(times) != len(values):
            raise ValueError("timestamps and values must have equal lengths")
        interval = fixed_interval(self.frequency)
        if len(times) > 1 and not ((times[1:] - times[:-1]) == interval).all():
            raise ValueError("timestamps must follow the declared regular frequency")
        for name in ("target_name", "units", "series_id"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "timestamps", times)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "provenance", dict(self.provenance))


@dataclass(frozen=True)
class EvaluationSpec:
    context_length: int
    horizon: int
    evaluation_start: str | pd.Timestamp
    evaluation_end: str | pd.Timestamp
    stride: int = 1
    seasonal_period: int = 1
    metrics: tuple[str, ...] = ("mae", "rmse", "mase")
    point_statistic: str = "median"
    aggregation: str = "forecast_case_micro_v1"

    def __post_init__(self) -> None:
        for name in ("context_length", "horizon", "stride", "seasonal_period"):
            object.__setattr__(self, name, positive_int(getattr(self, name), name))
        start = utc_timestamp(self.evaluation_start, "evaluation_start")
        end = utc_timestamp(self.evaluation_end, "evaluation_end")
        if start >= end:
            raise ValueError("evaluation_start must precede evaluation_end")
        metrics = tuple(self.metrics)
        if not metrics or len(set(metrics)) != len(metrics):
            raise ValueError("metrics must be nonempty and unique")
        if not set(metrics) <= {"mae", "rmse", "mase"}:
            raise ValueError("unsupported metric; choose mae, rmse, or mase")
        output_statistics(self.point_statistic)
        if self.aggregation != "forecast_case_micro_v1":
            raise ValueError("unsupported metric aggregation")
        object.__setattr__(self, "evaluation_start", start)
        object.__setattr__(self, "evaluation_end", end)
        object.__setattr__(self, "metrics", metrics)


@dataclass(frozen=True)
class ForecastRequest:
    request_id: str
    series_id: str
    origin: pd.Timestamp
    context_timestamps: pd.DatetimeIndex
    context_values: np.ndarray
    target_timestamps: pd.DatetimeIndex

    def __post_init__(self) -> None:
        for name in ("request_id", "series_id"):
            _text(getattr(self, name), name)
        origin = utc_timestamp(self.origin, "origin")
        context = _times(self.context_timestamps, "context_timestamps")
        targets = _times(self.target_timestamps, "target_timestamps")
        values = _values(self.context_values, "context_values")
        if len(context) != len(values) or context[-1] != origin:
            raise ValueError("context must have matching values and end at origin")
        if targets[0] <= origin:
            raise ValueError("target timestamps must follow the forecast origin")
        grid = context.append(targets)
        intervals = grid[1:] - grid[:-1]
        if len(intervals) > 1 and not (intervals == intervals[0]).all():
            raise ValueError("context and targets must follow one regular grid")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "context_timestamps", context)
        object.__setattr__(self, "context_values", values)
        object.__setattr__(self, "target_timestamps", targets)


@dataclass(frozen=True)
class ForecastOutput:
    request_id: str
    series_id: str
    origin: pd.Timestamp
    target_timestamps: pd.DatetimeIndex
    point: np.ndarray
    adapter_id: str
    point_statistic: Literal["deterministic", "median", "mean", "native_point"] = (
        "deterministic"
    )

    def __post_init__(self) -> None:
        for name in ("request_id", "series_id", "adapter_id"):
            _text(getattr(self, name), name)
        origin = utc_timestamp(self.origin, "origin")
        targets = _times(self.target_timestamps, "target_timestamps")
        point = _values(self.point, "point forecasts")
        if len(targets) != len(point) or targets[0] <= origin:
            raise ValueError("forecast length and target timestamps must align")
        if self.point_statistic not in output_statistics("native_point"):
            raise ValueError("unsupported forecast point statistic")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "target_timestamps", targets)
        object.__setattr__(self, "point", point)


@dataclass(frozen=True)
class EvaluationWindow:
    request: ForecastRequest
    actual: np.ndarray
    first_target_position: int

    def __post_init__(self) -> None:
        actual = _values(self.actual, "actual")
        if len(actual) != len(self.request.target_timestamps):
            raise ValueError("actual values must match target timestamps")
        object.__setattr__(self, "actual", actual)
        positive_int(self.first_target_position, "first_target_position")


@dataclass(frozen=True)
class ModelCapabilities:
    min_context: int = 1
    max_context: int | None = None
    max_horizon: int | None = None
    target_count: int = 1
    covariates: bool = False
    output_types: tuple[str, ...] = ("point",)
    devices: tuple[str, ...] = ("cpu",)

    def __post_init__(self) -> None:
        for name in ("min_context", "target_count"):
            positive_int(getattr(self, name), name)
        for name in ("max_context", "max_horizon"):
            if getattr(self, name) is not None:
                positive_int(getattr(self, name), name)
        if self.max_context is not None and self.max_context < self.min_context:
            raise ValueError("max_context must not be smaller than min_context")


@dataclass(frozen=True)
class ModelMetadata:
    adapter_id: str
    adapter_version: str
    family: str
    capabilities: ModelCapabilities

    def __post_init__(self) -> None:
        for name in ("adapter_id", "adapter_version", "family"):
            _text(getattr(self, name), name)
