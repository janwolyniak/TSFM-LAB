import numpy as np
import pandas as pd
import pytest

from tsfm_lab.contracts import EvaluationSpec, ForecastRequest
from tsfm_lab.models.baselines import LastValue, SeasonalNaive


def request():
    times = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")
    return ForecastRequest("a", "sensor", times[3], times[:4], [1, 7, 3, 5], times[4:])


def test_baselines_handle_leads_longer_than_a_season():
    req = request()
    (last,) = LastValue().predict((req,))
    (seasonal,) = SeasonalNaive(3).predict((req,))
    np.testing.assert_array_equal(last.point, [5, 5, 5, 5, 5, 5])
    np.testing.assert_array_equal(seasonal.point, [7, 3, 5, 7, 3, 5])
    assert seasonal.request_id == req.request_id
    assert seasonal.target_timestamps.equals(req.target_timestamps)
    assert seasonal.point_statistic == "deterministic"


def test_seasonal_period_is_independent_of_mase_scale_lag():
    req = request()
    model = SeasonalNaive(3)
    spec = EvaluationSpec(
        4,
        6,
        req.target_timestamps[0],
        req.target_timestamps[-1] + pd.Timedelta("1h"),
        seasonal_period=1,
    )
    model.validate(spec)
    assert model.configuration() == {"seasonal_period": 3}


@pytest.mark.parametrize("period", [True, 0, -1, 1.5])
def test_seasonality_must_be_a_positive_integer(period):
    with pytest.raises(ValueError):
        SeasonalNaive(period)


def test_seasonal_baseline_rejects_short_history():
    with pytest.raises(ValueError, match="season"):
        SeasonalNaive(5).predict((request(),))
