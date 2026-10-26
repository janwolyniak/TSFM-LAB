"""Causal signal selection and independent reconstruction of derived OHLC cases."""

from decimal import Context, Decimal, localcontext
from numbers import Real

import numpy as np
import pandas as pd

from tsfm_lab.contracts import Dataset, fixed_interval, utc_timestamp
from tsfm_lab.finance.contracts import BacktestRules, FinanceError
from tsfm_lab.finance.ledger import buy_and_hold, simulate_intrabar
from tsfm_lab.finance.ohlc import OHLCData

EXECUTION_POLICY = {
    "target_kind": "explicitly_declared_positive_close_price",
    "units": "declared_quote_units_per_one_asset_unit",
    "signal_lead": 1,
    "stride": 1,
    "signal_basis": "forecast_target_close_relative_to_origin_close",
    "trade_basis": "next_bar_open_to_that_bar_close",
    "latency": "open_strictly_after_origin_close_plus_declared_latency",
    "threshold_comparison": "decimal_shortest_float64_v1",
    "positions": "unit_long_flat_short_after_entry_cost",
    "liquidation": "every_target_close",
    "costs": "proportional_actual_notional_both_legs",
    "short_financing": "entry_short_notional_per_held_bar",
    "cash_interest": 0.0,
    "volatility": "net_bar_return_sample_std_ddof1_unannualized",
    "drawdown": "period_end_including_initial_capital",
    "turnover": "gross_traded_notional_divided_by_initial_capital",
}
CASE_FIELDS = {
    "request_id",
    "series_id",
    "origin",
    "target_timestamp",
    "lead",
    "point_statistic",
    "origin_close",
    "forecast_close",
    "actual_close",
    "open_timestamp",
    "close_timestamp",
    "open",
    "high",
    "low",
    "close",
}


def _numbers(cases, name):
    values = [row[name] for row in cases]
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
        raise FinanceError(f"{name} must contain real price numbers")
    result = np.array(values, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise FinanceError(f"{name} must contain finite positive prices")
    return result


def _times(cases, name):
    return pd.DatetimeIndex([utc_timestamp(row[name], name) for row in cases])


def _validated_cases(cases, rules, frequency, series_id):
    if not isinstance(rules, BacktestRules):
        raise FinanceError("A validated BacktestRules object is required")
    if not isinstance(cases, list) or not cases:
        raise FinanceError("Finance cases must be a nonempty list")
    seen = set()
    for row in cases:
        if not isinstance(row, dict) or set(row) != CASE_FIELDS:
            raise FinanceError("Finance cases have unexpected or missing fields")
        identifier = row["request_id"]
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise FinanceError("Finance request IDs must be nonempty and unique")
        seen.add(identifier)
        if row["series_id"] != series_id:
            raise FinanceError("Finance cases disagree with the declared series")
        if type(row["lead"]) is not int or row["lead"] != 1:
            raise FinanceError("The first OHLC policy requires lead 1")
        if row["point_statistic"] not in {"median", "deterministic"}:
            raise FinanceError(
                "Finance cases require the saved median/deterministic task"
            )
    origins = _times(cases, "origin")
    origin_prices = _numbers(cases, "origin_close")
    Dataset(origins, origin_prices, frequency, series_id=series_id)
    bars = OHLCData(
        _times(cases, "open_timestamp"),
        _times(cases, "close_timestamp"),
        _numbers(cases, "open"),
        _numbers(cases, "high"),
        _numbers(cases, "low"),
        _numbers(cases, "close"),
        frequency,
        series_id=series_id,
    )
    targets = _times(cases, "target_timestamp")
    interval = fixed_interval(frequency)
    if (
        not targets.equals(bars.close_timestamps)
        or not ((targets - origins) == interval).all()
    ):
        raise FinanceError("Each target must be the next bar's explicit close")
    if not ((bars.open_timestamps - origins) > rules.latency).all():
        raise FinanceError(
            "Entry open must be later than origin close plus signal latency"
        )
    if not np.array_equal(_numbers(cases, "actual_close"), bars.closes):
        raise FinanceError("OHLC closes disagree with saved forecast actuals")
    if not np.array_equal(origin_prices[1:], bars.closes[:-1]):
        raise FinanceError("Consecutive cases must retain the same observed close")
    return bars, origin_prices, _numbers(cases, "forecast_close")


def _positions(forecasts, origins, threshold_bps):
    positions = []
    # Decimal shortest float64 representations define ties without a broad tolerance.
    # Ratio-then-subtract would turn 101/100 at 100 bps into a false long signal.
    with localcontext(Context(prec=64)):
        threshold = Decimal(str(float(threshold_bps)))
        for predicted, observed in zip(forecasts, origins, strict=True):
            origin = Decimal(str(float(observed)))
            difference = (Decimal(str(float(predicted))) - origin) * Decimal(10000)
            boundary = origin * threshold
            positions.append(
                1 if difference > boundary else -1 if difference < -boundary else 0
            )
    return np.asarray(positions, dtype=np.int64)


def _diagnostics(forecasts, origins, actual, opens):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            predicted = (forecasts - origins) / origins
            observed = (actual - origins) / origins
            holding = (actual - opens) / opens
            errors = (forecasts - actual) / origins
            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(np.mean(errors * errors)))
    except FloatingPointError as exc:
        raise FinanceError(
            "Return diagnostics exceed finite float64 arithmetic"
        ) from exc
    forecast_sign, actual_sign = np.sign(predicted), np.sign(observed)

    def counts(values):
        return {
            "negative": int(np.sum(values < 0)),
            "zero": int(np.sum(values == 0)),
            "positive": int(np.sum(values > 0)),
        }

    return {
        "basis": "origin_close_to_target_close_simple_return",
        "cases": len(origins),
        "forecast_return_mae": mae,
        "forecast_return_rmse": rmse,
        "directional_accuracy_3way": float(np.mean(forecast_sign == actual_sign)),
        "zero_move_rule": "exact_negative_zero_positive_sign",
        "forecast_direction_counts": counts(forecast_sign),
        "actual_direction_counts": counts(actual_sign),
        "forecast_implied_returns": predicted.tolist(),
        "actual_close_returns": observed.tolist(),
        "realized_open_to_close_returns": holding.tolist(),
    }


def evaluate_cases(cases, rules, *, frequency, series_id):
    """Reconstruct signals, diagnostics, and all capital paths without inference."""
    try:
        bars, origins, forecasts = _validated_cases(cases, rules, frequency, series_id)
        positions = _positions(forecasts, origins, rules.threshold_bps)
        strategy = simulate_intrabar(
            bars.opens, bars.highs, bars.lows, bars.closes, positions, rules=rules
        )
        cash = simulate_intrabar(
            bars.opens,
            bars.highs,
            bars.lows,
            bars.closes,
            np.zeros(len(cases), dtype=np.int64),
            rules=rules,
        )
        hold = buy_and_hold(bars.opens, bars.closes, rules=rules)
        for path in (strategy, cash, hold):
            for row, case in zip(path["curve"], cases, strict=True):
                row["timestamp"] = case["close_timestamp"]
        return {
            "cases": len(cases),
            "signal_positions": positions.tolist(),
            "diagnostics": _diagnostics(forecasts, origins, bars.closes, bars.opens),
            "strategy": strategy,
            "benchmarks": {"cash": cash, "buy_and_hold": hold},
        }
    except FinanceError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise FinanceError(f"Invalid OHLC finance cases: {exc}") from exc
