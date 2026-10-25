import json
import math
from dataclasses import asdict

import numpy as np
import pytest

from tsfm_lab.evaluation.metrics import score_forecasts, score_with_scale


def test_unrequested_mase_does_not_trigger_scale_overflow():
    scores = score_forecasts([[1.0]], [[1.0]], [-1e308, 1e308], metrics=("mae",))
    assert scores.overall["mae"].value == 0
    assert scores.scale is None
    assert scores.scale_reason == "not_requested"
    assert scores.scale_count == 0


def test_hand_calculated_pooled_and_per_lead_metrics():
    actual = [[5, 6], [6, 7], [7, 8]]
    predicted = [[4, 4], [5, 5], [6, 6]]
    scores = score_forecasts(actual, predicted, [1, 2, 3, 4])
    assert scores.overall["mae"].value == 1.5
    assert scores.overall["rmse"].value == pytest.approx(math.sqrt(2.5))
    assert scores.overall["mase"].value == 1.5
    assert scores.scale == 1
    assert scores.scale_count == 3
    assert scores.by_lead[1]["mae"].value == 1
    assert scores.by_lead[2]["rmse"].value == 2


def test_rmse_pools_squared_errors_before_root():
    scores = score_forecasts([[0, 0], [0, 0]], [[0, 0], [3, 4]], [0, 1])
    assert scores.overall["rmse"].value == 2.5
    assert scores.overall["rmse"].value != (0 + math.sqrt(12.5)) / 2


def test_seasonal_scale_uses_exact_pre_evaluation_history():
    # Lag-two differences are 2, 4, 4, 4, with a fixed average of 3.5.
    scores = score_forecasts(
        [[0, 0]], [[7, -7]], [1, 10, 3, 14, 7, 18], seasonal_period=2
    )
    assert scores.scale == 3.5
    assert scores.scale_count == 4
    assert scores.overall["mase"].value == 2


@pytest.mark.parametrize("predicted", [[[5, 5]], [[6, 7]]])
def test_constant_scaling_history_is_undefined_even_for_perfect_forecast(predicted):
    scores = score_forecasts([[5, 5]], predicted, [5, 5, 5])
    assert scores.scale == 0
    assert scores.scale_reason == "zero_scale"
    assert scores.overall["mase"].value is None
    assert scores.overall["mase"].reason == "zero_scale"
    assert scores.by_lead[1]["mase"].reason == "zero_scale"
    encoded = json.dumps(asdict(scores), allow_nan=False)
    assert '"value": null' in encoded


@pytest.mark.parametrize("history", [[], [1], [1, 2]])
def test_insufficient_history_has_explicit_undefined_status(history):
    scores = score_forecasts([[1]], [[2]], history, seasonal_period=2)
    assert scores.scale is None
    assert scores.scale_count == 0
    assert scores.overall["mase"].reason == "insufficient_scale_history"
    assert scores.overall["mae"].value == 1


def test_explicit_metric_subset():
    scores = score_forecasts([[0]], [[1]], [0, 1], metrics=("rmse",))
    assert set(scores.overall) == {"rmse"}
    assert set(scores.by_lead[1]) == {"rmse"}


@pytest.mark.parametrize("seasonal_period", [0, -1, True, 1.5, "2"])
def test_invalid_seasonal_period_rejected(seasonal_period):
    with pytest.raises(ValueError, match="positive integer"):
        score_forecasts([[0]], [[1]], [0, 1], seasonal_period=seasonal_period)


@pytest.mark.parametrize("metrics", [(), ("mae", "mae"), ("unknown",), "mae", ([],)])
def test_invalid_metric_selection_rejected(metrics):
    with pytest.raises(ValueError):
        score_forecasts([[0]], [[1]], [0, 1], metrics=metrics)


@pytest.mark.parametrize(
    "actual,predicted,history",
    [
        ([1, 2], [1, 2], [0, 1]),
        ([[1, 2]], [[1]], [0, 1]),
        ([[]], [[]], [0, 1]),
        (np.empty((0, 2)), np.empty((0, 2)), [0, 1]),
        ([[np.nan]], [[1]], [0, 1]),
        ([[1]], [[np.inf]], [0, 1]),
        ([[1]], [[1]], [0, np.inf]),
        ([[1]], [[1]], [[0, 1]]),
        ([[1j]], [[1]], [0, 1]),
        ([[True]], [[1]], [0, 1]),
    ],
)
def test_invalid_metric_arrays_rejected(actual, predicted, history):
    with pytest.raises(ValueError):
        score_forecasts(actual, predicted, history)


def test_large_finite_errors_do_not_overflow_intermediate_squares_or_sums():
    scores = score_forecasts([[0, 0]], [[1e300, -1e300]], [0, 1e300])
    assert scores.overall["mae"].value == 1e300
    assert scores.overall["rmse"].value == 1e300
    assert scores.overall["mase"].value == 1


@pytest.mark.parametrize(
    "actual,predicted,history",
    [
        ([[-1e308]], [[1e308]], [0, 1]),
        ([[0]], [[1]], [-1e308, 1e308]),
        ([[0]], [[1e308]], [0, 1e-308]),
    ],
)
def test_nonfinite_arithmetic_is_explicit_error(actual, predicted, history):
    with pytest.raises(ValueError, match="Nonfinite arithmetic"):
        score_forecasts(actual, predicted, history)


@pytest.mark.parametrize(
    "history,period,metrics",
    [
        ([1, 2, 3, 4], 1, ("mae", "rmse", "mase")),
        ([1, 4, 2, 7, 4, 10], 2, ("mase",)),
        ([5, 5, 5], 1, ("mae", "mase")),
        ([1], 2, ("mae", "rmse", "mase")),
        ([1, 2, 3, 4], 1, ("rmse",)),
    ],
)
def test_recorded_scale_rescoring_matches_original_algorithm(history, period, metrics):
    actual = [[0.1, -3.1], [2.7, -0.0]]
    predicted = [[np.nextafter(0.1, 1), 2.3], [1.1, -0.0]]
    original = score_forecasts(
        actual, predicted, history, seasonal_period=period, metrics=metrics
    )
    rescored = score_with_scale(
        actual,
        predicted,
        scale=original.scale,
        scale_reason=original.scale_reason,
        scale_count=original.scale_count,
        metrics=metrics,
    )
    assert asdict(rescored) == asdict(original)


@pytest.mark.parametrize(
    "scale,reason,count,metrics",
    [
        (-1, None, 1, ("mase",)),
        (np.inf, None, 1, ("mase",)),
        (np.nan, None, 1, ("mase",)),
        (True, None, 1, ("mase",)),
        ("1", None, 1, ("mase",)),
        (1j, None, 1, ("mase",)),
        (1, "unknown", 1, ("mase",)),
        (1, [], 1, ("mase",)),
        (1, None, True, ("mase",)),
        (1, None, -1, ("mase",)),
        (1, None, 1.0, ("mase",)),
        (1, None, 0, ("mase",)),
        (1, "zero_scale", 1, ("mase",)),
        (1, "insufficient_scale_history", 1, ("mase",)),
        (1, "not_requested", 1, ("mase",)),
        (0, None, 1, ("mase",)),
        (0, "zero_scale", 0, ("mase",)),
        (None, None, 0, ("mase",)),
        (None, "zero_scale", 0, ("mase",)),
        (None, "insufficient_scale_history", 1, ("mase",)),
        (None, "not_requested", 0, ("mase",)),
        (1, None, 1, ("mae",)),
        (None, "insufficient_scale_history", 0, ("mae",)),
        (None, "not_requested", 1, ("mae",)),
    ],
)
def test_invalid_recorded_scale_metadata_is_rejected(scale, reason, count, metrics):
    with pytest.raises(ValueError):
        score_with_scale(
            [[0]],
            [[1]],
            scale=scale,
            scale_reason=reason,
            scale_count=count,
            metrics=metrics,
        )


@pytest.mark.parametrize("metrics", [(), ("unknown",), ("mae", "mae"), "mae", None])
def test_recorded_scale_path_reuses_metric_name_validation(metrics):
    with pytest.raises(ValueError):
        score_with_scale(
            [[0]],
            [[1]],
            scale=None,
            scale_reason="not_requested",
            scale_count=0,
            metrics=metrics,
        )


def test_recorded_scale_path_rejects_bad_forecast_arrays():
    with pytest.raises(ValueError, match="same nonempty"):
        score_with_scale([[0, 1]], [[1]], scale=1.0, scale_reason=None, scale_count=1)
    with pytest.raises(ValueError, match="finite"):
        score_with_scale([[0]], [[np.inf]], scale=1.0, scale_reason=None, scale_count=1)
