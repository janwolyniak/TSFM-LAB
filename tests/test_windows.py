import numpy as np
import pandas as pd
import pytest

from tsfm_lab.contracts import Dataset, EvaluationSpec
from tsfm_lab.evaluation.windows import evaluation_bounds, iter_windows


def dataset():
    return Dataset(
        timestamps=pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC"),
        values=np.arange(1, 9, dtype=float),
        frequency="1h",
    )


def spec(**changes):
    options = dict(
        context_length=3,
        horizon=2,
        stride=1,
        evaluation_start="2026-01-01T04:00:00Z",
        evaluation_end="2026-01-01T08:00:00Z",
    )
    options.update(changes)
    return EvaluationSpec(**options)


def test_documented_rolling_origins_and_alignment():
    series = dataset()
    windows = list(iter_windows(series, spec()))
    assert evaluation_bounds(series, spec()) == (4, 8)
    assert [window.first_target_position for window in windows] == [4, 5, 6]
    for first, window in zip([4, 5, 6], windows, strict=True):
        np.testing.assert_array_equal(
            window.request.context_values, series.values[first - 3 : first]
        )
        np.testing.assert_array_equal(window.actual, series.values[first : first + 2])
        assert window.request.origin == series.timestamps[first - 1]
        assert window.request.target_timestamps.equals(
            series.timestamps[first : first + 2]
        )
    assert [list(w.request.context_values) for w in windows] == [
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
    ]


@pytest.mark.parametrize("stride,positions", [(1, [4, 5, 6]), (2, [4, 6]), (3, [4])])
def test_stride_counts_complete_horizons(stride, positions):
    windows = list(iter_windows(dataset(), spec(stride=stride)))
    assert [window.first_target_position for window in windows] == positions


def test_single_complete_holdout_at_first_allowed_origin():
    windows = list(
        iter_windows(
            dataset(),
            spec(
                evaluation_start="2026-01-01T03:00:00Z",
                evaluation_end="2026-01-01T05:00:00Z",
            ),
        )
    )
    assert len(windows) == 1
    np.testing.assert_array_equal(windows[0].request.context_values, [1, 2, 3])
    np.testing.assert_array_equal(windows[0].actual, [4, 5])


def test_exclusive_end_on_existing_timestamp_discards_partial_final_window():
    windows = list(iter_windows(dataset(), spec(evaluation_end="2026-01-01T07:00:00Z")))
    assert [window.first_target_position for window in windows] == [4, 5]
    assert windows[-1].request.target_timestamps[-1] == pd.Timestamp(
        "2026-01-01T06:00Z"
    )


def test_timezone_equivalent_bounds_match_same_positions():
    assert evaluation_bounds(
        dataset(),
        spec(
            evaluation_start="2026-01-01T05:00:00+01:00",
            evaluation_end="2026-01-01T09:00:00+01:00",
        ),
    ) == (4, 8)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"evaluation_start": "2026-01-01T04:30:00Z"}, "grid"),
        ({"evaluation_start": "2025-12-31T23:00:00Z"}, "grid"),
        ({"evaluation_end": "2026-01-01T08:30:00Z"}, "grid"),
        ({"evaluation_end": "2026-01-01T09:00:00Z"}, "grid"),
        ({"evaluation_start": "2026-01-01T02:00:00Z"}, "history"),
        ({"horizon": 5}, "complete"),
    ],
)
def test_invalid_bounds_fail_without_shifting(changes, match):
    with pytest.raises(ValueError, match=match):
        list(iter_windows(dataset(), spec(**changes)))


def test_window_request_owns_only_historical_values():
    series = dataset()
    window = next(iter_windows(series, spec()))
    assert not np.shares_memory(window.request.context_values, series.values)
    assert not np.shares_memory(window.actual, series.values)
    assert not window.request.context_values.flags.writeable
    assert not window.actual.flags.writeable
    assert not hasattr(window.request, "actual")
    np.testing.assert_array_equal(window.request.context_values, [2, 3, 4])


def test_future_perturbation_preserves_first_request_but_updates_later_history():
    original = dataset()
    changed_values = original.values.copy()
    changed_values[4:] += 100
    changed = Dataset(original.timestamps, changed_values, original.frequency)
    before = list(iter_windows(original, spec()))
    after = list(iter_windows(changed, spec()))
    np.testing.assert_array_equal(
        before[0].request.context_values, after[0].request.context_values
    )
    assert before[0].request.request_id == after[0].request.request_id
    assert before[1].request.context_values[-1] != after[1].request.context_values[-1]
