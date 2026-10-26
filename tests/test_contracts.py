"""Public numerical boundaries reject ambiguous data and own their buffers."""

import numpy as np
import pandas as pd
import pytest

from tsfm_lab.contracts import (
    Dataset,
    EvaluationSpec,
    ForecastOutput,
    ForecastRequest,
    ModelCapabilities,
    fixed_interval,
)


@pytest.mark.parametrize(
    "values", [[True, False], [1 + 2j, 3], ["1", "2"], [np.nan, 2]]
)
def test_dataset_rejects_nonreal_or_nonfinite_values(values):
    with pytest.raises(ValueError):
        Dataset(
            pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"), values, "1h"
        )


@pytest.mark.parametrize("frequency", ["ME", "0h", "-1h", "not-a-frequency", 1])
def test_dataset_requires_positive_fixed_frequency(frequency):
    with pytest.raises(ValueError, match="frequency"):
        Dataset(
            pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
            [1, 2],
            frequency,
        )


@pytest.mark.parametrize(
    "frequency", ["100000000000000000000000h", "9999999999999999999ns", "9999999999D"]
)
def test_oversized_fixed_duration_has_a_validation_error(frequency):
    with pytest.raises(ValueError, match="frequency must be a positive fixed duration"):
        fixed_interval(frequency)


@pytest.mark.parametrize(
    "name", ["context_length", "horizon", "stride", "seasonal_period"]
)
@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_spec_rejects_invalid_integer_parameters(name, value):
    kwargs = {
        "context_length": 3,
        "horizon": 2,
        "evaluation_start": "2026-01-01T03:00:00Z",
        "evaluation_end": "2026-01-01T08:00:00Z",
        name: value,
    }
    with pytest.raises(ValueError, match=name):
        EvaluationSpec(**kwargs)


@pytest.mark.parametrize(
    "options",
    [
        {"metrics": ()},
        {"metrics": ("mae", "mae")},
        {"metrics": ("unknown",)},
        {"point_statistic": "mean"},
        {"aggregation": "macro"},
        {"evaluation_start": "2026-01-01"},
        {"evaluation_end": "2025-01-01T00:00:00Z"},
    ],
)
def test_spec_rejects_unsupported_or_ambiguous_settings(options):
    kwargs = dict(
        context_length=3,
        horizon=2,
        evaluation_start="2026-01-01T03:00:00Z",
        evaluation_end="2026-01-01T08:00:00Z",
    )
    with pytest.raises(ValueError):
        EvaluationSpec(**(kwargs | options))


def test_native_point_task_is_explicit_and_median_remains_default():
    options = dict(
        context_length=3,
        horizon=2,
        evaluation_start="2026-01-01T03:00:00Z",
        evaluation_end="2026-01-01T08:00:00Z",
    )
    assert EvaluationSpec(**options).point_statistic == "median"
    assert (
        EvaluationSpec(**options, point_statistic="native_point").point_statistic
        == "native_point"
    )


def test_contracts_detach_both_input_and_output_values():
    timestamps = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    original = np.array([1.0, 2.0, 3.0, 4.0])
    dataset = Dataset(timestamps, original, "1h")
    request = ForecastRequest(
        "r", "s", timestamps[1], timestamps[:2], dataset.values[:2], timestamps[2:]
    )
    output = ForecastOutput(
        "r", "s", timestamps[1], timestamps[2:], original[2:], "adapter"
    )
    original[:] = -100
    np.testing.assert_array_equal(dataset.values, [1, 2, 3, 4])
    np.testing.assert_array_equal(request.context_values, [1, 2])
    np.testing.assert_array_equal(output.point, [3, 4])
    assert request.context_values.base is None
    assert not request.context_values.flags.writeable
    assert not output.point.flags.writeable
    request.context_values.setflags(write=True)
    request.context_values[:] = 99
    np.testing.assert_array_equal(dataset.values, [1, 2, 3, 4])


def test_request_rejects_gaps_between_history_and_horizon():
    timestamps = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="regular grid"):
        ForecastRequest("r", "s", timestamps[1], timestamps[:2], [1, 2], timestamps[3:])


@pytest.mark.parametrize(
    "options",
    [{"min_context": 0}, {"max_horizon": True}, {"max_context": 1, "min_context": 2}],
)
def test_capabilities_have_valid_bounds(options):
    with pytest.raises(ValueError):
        ModelCapabilities(**options)
