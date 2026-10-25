"""Point metrics with explicit pooling and a fixed pre-evaluation MASE scale."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

METRIC_VERSION = "point_metrics_v1"
SUPPORTED_METRICS = ("mae", "rmse", "mase")


@dataclass(frozen=True)
class MetricValue:
    value: float | None
    reason: str | None = None


@dataclass(frozen=True)
class Scores:
    overall: dict[str, MetricValue]
    by_lead: dict[int, dict[str, MetricValue]]
    scale: float | None
    scale_reason: str | None
    scale_count: int


def _numeric_array(values: ArrayLike, *, name: str, ndim: int) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return array


def _difference(left: NDArray[np.float64], right: NDArray[np.float64]):
    with np.errstate(over="raise", invalid="raise"):
        try:
            return left - right
        except FloatingPointError as exc:
            raise ValueError("Nonfinite arithmetic while calculating errors") from exc


def _mean_absolute(errors: NDArray[np.float64]) -> float:
    magnitudes = np.abs(errors)
    maximum = float(np.max(magnitudes))
    if maximum == 0:
        return 0.0
    return maximum * float(np.mean(magnitudes / maximum))


def _rmse(errors: NDArray[np.float64]) -> float:
    maximum = float(np.max(np.abs(errors)))
    if maximum == 0:
        return 0.0
    return maximum * float(np.sqrt(np.mean((errors / maximum) ** 2)))


def _scale(
    history: NDArray[np.float64], period: int
) -> tuple[float | None, str | None]:
    if len(history) <= period:
        return None, "insufficient_scale_history"
    scale = _mean_absolute(_difference(history[period:], history[:-period]))
    return (scale, "zero_scale") if scale == 0 else (scale, None)


def _metrics(
    errors: NDArray[np.float64],
    names: tuple[str, ...],
    scale: float | None,
    scale_reason: str | None,
) -> dict[str, MetricValue]:
    mae = _mean_absolute(errors)
    results = {}
    for name in names:
        if name == "mase" and scale_reason:
            results[name] = MetricValue(None, scale_reason)
            continue
        if name == "mae":
            value = mae
        elif name == "rmse":
            value = _rmse(errors)
        else:
            value = mae / scale
        if not np.isfinite(value):
            raise ValueError(f"Nonfinite arithmetic while calculating {name}")
        results[name] = MetricValue(float(value))
    return results


def score_forecasts(
    actual: ArrayLike,
    predicted: ArrayLike,
    history: ArrayLike,
    *,
    seasonal_period: int = 1,
    metrics: tuple[str, ...] = SUPPORTED_METRICS,
) -> Scores:
    """Score all forecast cases and leads using history strictly before evaluation."""
    if (
        isinstance(seasonal_period, bool)
        or not isinstance(seasonal_period, Integral)
        or seasonal_period <= 0
    ):
        raise ValueError("seasonal_period must be a positive integer")
    if isinstance(metrics, str):
        raise ValueError("metrics must be a sequence of metric names")
    names = tuple(metrics)
    if any(not isinstance(name, str) for name in names):
        raise ValueError("metrics must contain string names")
    if not names or len(set(names)) != len(names):
        raise ValueError("metrics must be nonempty and contain no duplicates")
    if any(name not in SUPPORTED_METRICS for name in names):
        raise ValueError(f"Unsupported metric; choose from {SUPPORTED_METRICS}")
    observed = _numeric_array(actual, name="actual", ndim=2)
    forecasts = _numeric_array(predicted, name="predicted", ndim=2)
    past = _numeric_array(history, name="history", ndim=1)
    if observed.shape != forecasts.shape or 0 in observed.shape:
        raise ValueError("actual and predicted must have the same nonempty [W,H] shape")
    errors = _difference(forecasts, observed)
    scale, reason = (
        _scale(past, int(seasonal_period))
        if "mase" in names
        else (None, "not_requested")
    )
    return Scores(
        overall=_metrics(errors, names, scale, reason),
        by_lead={
            lead + 1: _metrics(errors[:, lead], names, scale, reason)
            for lead in range(errors.shape[1])
        },
        scale=scale,
        scale_reason=reason,
        scale_count=max(len(past) - int(seasonal_period), 0) if "mase" in names else 0,
    )
