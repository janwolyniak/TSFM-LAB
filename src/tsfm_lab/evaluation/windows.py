"""Fixed-context rolling origins with evaluator-only future observations."""

from collections.abc import Iterator

import pandas as pd

from tsfm_lab.contracts import (
    Dataset,
    EvaluationSpec,
    EvaluationWindow,
    ForecastRequest,
)


def _position(dataset: Dataset, timestamp: str | pd.Timestamp, *, end: bool) -> int:
    boundary = pd.Timestamp(timestamp)
    if pd.isna(boundary) or boundary.tzinfo is None:
        raise ValueError("Evaluation boundaries must be timezone-aware timestamps")
    boundary = boundary.tz_convert("UTC")
    index = dataset.timestamps
    if end and boundary == index[-1] + pd.tseries.frequencies.to_offset(
        dataset.frequency
    ):
        return len(index)
    position = int(index.get_indexer([boundary])[0])
    if position < 0:
        label = "evaluation_end" if end else "evaluation_start"
        raise ValueError(f"{label} must exactly match the dataset grid")
    return position


def evaluation_bounds(dataset: Dataset, spec: EvaluationSpec) -> tuple[int, int]:
    """Resolve the half-open target interval and reject impossible windows."""
    start = _position(dataset, spec.evaluation_start, end=False)
    end = _position(dataset, spec.evaluation_end, end=True)
    if start < spec.context_length:
        raise ValueError("Insufficient history before evaluation_start for context")
    if end <= start:
        raise ValueError("evaluation_end must follow evaluation_start")
    if start + spec.horizon > end:
        raise ValueError("Evaluation interval contains no complete forecast horizon")
    return start, end


def iter_windows(dataset: Dataset, spec: EvaluationSpec) -> Iterator[EvaluationWindow]:
    """Yield complete windows without making future values part of a request."""
    start, end = evaluation_bounds(dataset, spec)
    for first in range(start, end - spec.horizon + 1, spec.stride):
        context_start = first - spec.context_length
        request = ForecastRequest(
            request_id=f"{dataset.series_id}:{first}",
            series_id=dataset.series_id,
            origin=dataset.timestamps[first - 1],
            context_timestamps=dataset.timestamps[context_start:first],
            context_values=dataset.values[context_start:first],
            target_timestamps=dataset.timestamps[first : first + spec.horizon],
        )
        yield EvaluationWindow(
            request=request,
            actual=dataset.values[first : first + spec.horizon],
            first_target_position=first,
        )
