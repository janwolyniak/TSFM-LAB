"""Independent timing, threshold, and signal-versus-fill finance checks."""

import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tsfm_lab.finance.analysis import evaluate_cases
from tsfm_lab.finance.contracts import BacktestRules, FinanceError


def rules(**changes):
    return BacktestRules(
        **(
            {
                "initial_capital": 100,
                "threshold_bps": 0,
                "fee_bps": 0,
                "slippage_bps": 0,
                "short_financing_bps": 0,
                "signal_latency_seconds": 0,
            }
            | changes
        )
    )


def cases(forecasts, *, opening=100):
    times = pd.date_range("2026-01-01", periods=len(forecasts) + 1, freq="h", tz="UTC")
    return [
        {
            "request_id": str(index),
            "series_id": "asset",
            "origin": times[index].isoformat(),
            "target_timestamp": times[index + 1].isoformat(),
            "lead": 1,
            "point_statistic": "median",
            "origin_close": 100.0,
            "forecast_close": float(forecast),
            "actual_close": 100.0,
            "open_timestamp": (times[index] + pd.Timedelta(minutes=15)).isoformat(),
            "close_timestamp": times[index + 1].isoformat(),
            "open": float(opening),
            "high": 120.0,
            "low": 80.0,
            "close": 100.0,
        }
        for index, forecast in enumerate(forecasts)
    ]


def evaluate(rows, specification=None):
    return evaluate_cases(
        rows, specification or rules(), frequency="1h", series_id="asset"
    )


@pytest.mark.parametrize(
    "values,threshold,positions",
    [
        ([101, 99], 100, [0, 0]),
        ([np.nextafter(101.0, np.inf), np.nextafter(99.0, -np.inf)], 100, [1, -1]),
        (
            [100, np.nextafter(100.0, np.inf), np.nextafter(100.0, -np.inf)],
            0,
            [0, 1, -1],
        ),
    ],
)
def test_threshold_ties_are_flat_without_erasing_true_adjacent_excess(
    values, threshold, positions
):
    result = evaluate(cases(values), rules(threshold_bps=threshold))
    assert result["signal_positions"] == positions


def test_future_open_changes_trade_returns_without_changing_forecast_signal():
    first = evaluate(cases([102, 98], opening=100))
    second = evaluate(cases([102, 98], opening=110))
    assert first["signal_positions"] == second["signal_positions"] == [1, -1]
    assert (
        first["diagnostics"]["forecast_implied_returns"]
        == second["diagnostics"]["forecast_implied_returns"]
    )
    assert (
        first["strategy"]["curve"][0]["equity"]
        != second["strategy"]["curve"][0]["equity"]
    )


def test_close_to_close_direction_is_separate_from_realized_intrabar_return():
    rows = cases([110], opening=105)
    rows[0]["actual_close"] = rows[0]["close"] = 108.0
    result = evaluate(rows)
    diagnostic = result["diagnostics"]
    assert diagnostic["forecast_return_mae"] == pytest.approx(0.02)
    assert diagnostic["forecast_return_rmse"] == pytest.approx(0.02)
    assert diagnostic["directional_accuracy_3way"] == 1
    assert diagnostic["actual_close_returns"] == pytest.approx([0.08])
    assert diagnostic["realized_open_to_close_returns"] == pytest.approx(
        [108 / 105 - 1]
    )


def test_explicit_latency_equality_and_contiguous_boundaries_fail():
    rows = cases([102])
    with pytest.raises(FinanceError, match="latency"):
        evaluate(rows, rules(signal_latency_seconds=900))
    with pytest.raises(FinanceError, match="latency"):
        evaluate(rows, rules(signal_latency_seconds=901))
    assert evaluate(rows, rules(signal_latency_seconds=899))["cases"] == 1
    rows[0]["open_timestamp"] = rows[0]["origin"]
    with pytest.raises(FinanceError, match="latency"):
        evaluate(rows)


def test_three_way_zero_direction_and_cash_counts_use_full_denominator():
    result = evaluate(cases([100, 102, 98]))
    assert result["diagnostics"]["directional_accuracy_3way"] == pytest.approx(1 / 3)
    assert result["diagnostics"]["actual_direction_counts"] == {
        "negative": 0,
        "zero": 3,
        "positive": 0,
    }
    assert result["benchmarks"]["cash"]["metrics"]["final_capital"] == 100
    assert result["benchmarks"]["cash"]["metrics"]["total_costs"] == 0


@pytest.mark.parametrize(
    "fault",
    [
        "lead",
        "bool_price",
        "unknown_field",
        "actual",
        "origin_price",
        "series",
        "ids",
        "order",
        "nonpositive_forecast",
        "statistic",
    ],
)
def test_invalid_cases_fail_without_silent_repair(fault):
    rows = cases([102, 98])
    if fault == "lead":
        rows[0]["lead"] = 2
    elif fault == "bool_price":
        rows[0]["open"] = True
    elif fault == "unknown_field":
        rows[0]["extra"] = "ignored?"
    elif fault == "actual":
        rows[0]["actual_close"] = 101
    elif fault == "origin_price":
        rows[1]["origin_close"] = 101
    elif fault == "series":
        rows[0]["series_id"] = "other"
    elif fault == "ids":
        rows[1]["request_id"] = rows[0]["request_id"]
    elif fault == "order":
        rows.reverse()
    elif fault == "nonpositive_forecast":
        rows[0]["forecast_close"] = 0
    else:
        rows[0]["point_statistic"] = "mean"
    with pytest.raises(FinanceError):
        evaluate(rows)


def test_case_evaluation_does_not_mutate_inputs():
    rows = cases([102, 98])
    original = copy.deepcopy(rows)
    evaluate(rows)
    assert rows == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("initial_capital", 0),
        ("threshold_bps", -1),
        ("fee_bps", True),
        ("slippage_bps", float("nan")),
        ("short_financing_bps", -1),
        ("signal_latency_seconds", float("inf")),
        ("signal_latency_seconds", -1),
    ],
)
def test_rule_inputs_are_explicit_finite_and_validated(field, value):
    with pytest.raises(FinanceError):
        replace(rules(), **{field: value})
