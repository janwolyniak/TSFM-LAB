"""Finite cash/share ledgers for flat-between-bars trades and buy-and-hold."""

import math
import statistics
from numbers import Integral, Real

import numpy as np

from tsfm_lab.finance.contracts import BacktestRules, FinanceError


def _finite(value, label):
    if not math.isfinite(value):
        raise FinanceError(f"{label} must remain finite")
    return value


def _sum(values, label):
    try:
        return _finite(math.fsum(values), label)
    except OverflowError as exc:
        raise FinanceError(f"{label} overflowed") from exc


def _array(values, name, *, positions=False):
    try:
        if np.ma.isMaskedArray(values) and np.ma.getmaskarray(values).any():
            raise FinanceError(f"{name} cannot contain masked observations")
        array = np.asarray(values, dtype=object)
        if array.ndim != 1 or not len(array):
            raise FinanceError(f"{name} must be a nonempty one-dimensional array")
        result = []
        for value in array:
            if positions:
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise FinanceError("positions must contain strict integers")
                if value not in {-1, 0, 1}:
                    raise FinanceError("positions must be -1, 0, or 1")
                result.append(int(value))
            else:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise FinanceError(f"{name} must contain real numeric prices")
                price = _finite(float(value), name)
                if price <= 0:
                    raise FinanceError(f"{name} prices must be positive")
                result.append(price)
        return tuple(result)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinanceError(f"Invalid {name}: {exc}") from exc


def _inputs(rules, **arrays):
    if not isinstance(rules, BacktestRules):
        raise FinanceError("rules must be validated BacktestRules")
    result = {
        name: _array(values, name, positions=name == "positions")
        for name, values in arrays.items()
    }
    if len({len(values) for values in result.values()}) != 1:
        raise FinanceError("Ledger inputs must have the same length")
    return result


def _units(equity, price, position, rules):
    if position == 0:
        return 0.0
    factor = 1 + (rules.fee_bps + rules.slippage_bps) / 10000
    denominator = price * factor
    # This order also handles a finite price whose cost-adjusted product overflows.
    amount = (
        equity / denominator if math.isfinite(denominator) else equity / price / factor
    )
    units = _finite(position * amount, "Position units")
    if units == 0:
        raise FinanceError("Nonzero position units underflowed to zero")
    return units


def _transaction(notional, rules):
    return {
        "notional": _finite(notional, "Traded notional"),
        "fee": _finite(notional * (rules.fee_bps / 10000), "Fee"),
        "slippage": _finite(notional * (rules.slippage_bps / 10000), "Slippage"),
    }


def _holding_value(units, price):
    value = _finite(units * price, "Position value")
    if units != 0 and value == 0:
        raise FinanceError("Nonzero position value underflowed to zero")
    return value


def _entry(equity, price, position, rules):
    units = _units(equity, price, position, rules)
    value = _holding_value(units, price)
    entry = _transaction(abs(value), rules)
    cash = _sum(
        [equity, -value, -entry["fee"], -entry["slippage"]],
        "Entry cash",
    )
    return units, cash, entry


def _record(
    previous,
    equity,
    position,
    units,
    *,
    cash,
    ending_cash,
    ending_units,
    entry,
    exit,
    financing,
    gross_pnl,
):
    if equity <= 0:
        raise FinanceError("Close equity is nonpositive after costs and financing")
    net_pnl = _finite(equity - previous, "Net P&L")
    traded = _sum([entry["notional"], exit["notional"]], "Period traded notional")
    row = {
        "equity": _finite(equity, "Close equity"),
        "entry_equity": previous,
        "position": position,
        "units": units,
        "entry_cash": cash,
        "exit_cash": ending_cash,
        "exit_units": ending_units,
        "net_return": _finite(net_pnl / previous, "Period return"),
        "traded_notional": traded,
        "short_financing": financing,
        "gross_pnl": _finite(gross_pnl, "Gross P&L"),
        "net_pnl": net_pnl,
        "turnover_entry_equity": _finite(traded / previous, "Period turnover"),
    }
    for prefix, transaction in (("entry", entry), ("exit", exit)):
        row.update({f"{prefix}_{key}": value for key, value in transaction.items()})
    return row


def _summary(curve, initial):
    """Use initial-capital turnover and close-equity drawdowns including inception.

    Volatility is the unannualized sample standard deviation of period net returns
    (ddof=1), with an explicit undefined status when fewer than two exist.
    """
    peak, drawdown = initial, 0.0
    for row in curve:
        peak = max(peak, row["equity"])
        drawdown = max(drawdown, 1 - row["equity"] / peak)
    returns = [row["net_return"] for row in curve]
    volatility = {"value": None, "reason": "insufficient_returns"}
    if len(returns) >= 2:
        try:
            volatility = {
                "value": _finite(statistics.stdev(returns), "Sample volatility"),
                "reason": None,
            }
        except OverflowError as exc:
            raise FinanceError("Sample volatility overflowed") from exc
    traded = _sum((row["traded_notional"] for row in curve), "Total traded notional")
    fees = _sum(
        (row[key] for row in curve for key in ("entry_fee", "exit_fee")), "Total fees"
    )
    slippage = _sum(
        (row[key] for row in curve for key in ("entry_slippage", "exit_slippage")),
        "Total slippage",
    )
    financing = _sum((row["short_financing"] for row in curve), "Total financing")
    final = curve[-1]["equity"]
    return {
        "final_capital": final,
        "cumulative_return": _finite((final - initial) / initial, "Cumulative return"),
        "max_drawdown": drawdown,
        "volatility_unannualized": volatility,
        "total_traded_notional": traded,
        "turnover_initial_capital": _finite(traded / initial, "Total turnover"),
        "fees": fees,
        "slippage": slippage,
        "short_financing": financing,
        "total_costs": _sum([fees, slippage, financing], "Total costs"),
    }


def simulate_intrabar(
    opens, highs, lows, closes, positions, *, rules: BacktestRules
) -> dict:
    """Enter at each open and fully liquidate at that bar's close, without netting.

    Units target equity exposure after entry costs. Every nonflat bar pays both
    transaction legs, including when consecutive signals have the same sign.
    Row turnover divides both leg notionals by pre-entry equity; total turnover
    divides all traded notional by initial capital.
    """
    data = _inputs(
        rules, opens=opens, highs=highs, lows=lows, closes=closes, positions=positions
    )
    curve, equity = [], rules.initial_capital
    bars = zip(
        *(data[name] for name in ("opens", "highs", "lows", "closes", "positions")),
        strict=True,
    )
    for opening, high, low, close, position in bars:
        if not low <= opening <= high or not low <= close <= high:
            raise FinanceError("OHLC prices must satisfy low <= open/close <= high")
        previous = equity
        units, cash, entry = _entry(previous, opening, position, rules)
        if position == -1:
            at_high = _sum(
                [cash, _holding_value(units, high)], "Short equity at observed high"
            )
            if at_high <= 0:
                raise FinanceError(
                    "Short equity is nonpositive at the observed bar high"
                )
        closing_value = _holding_value(units, close)
        exit = _transaction(abs(closing_value), rules)
        financing = _finite(
            (
                entry["notional"] * (rules.short_financing_bps / 10000)
                if position == -1
                else 0.0
            ),
            "Short financing",
        )
        equity = _sum(
            [cash, closing_value, -exit["fee"], -exit["slippage"], -financing],
            "Close equity",
        )
        curve.append(
            _record(
                previous,
                equity,
                position,
                units,
                cash=cash,
                ending_cash=equity,
                ending_units=0.0,
                entry=entry,
                exit=exit,
                financing=financing,
                gross_pnl=units * (close - opening),
            )
        )
    return {"curve": curve, "metrics": _summary(curve, rules.initial_capital)}


def buy_and_hold(opens, closes, *, rules: BacktestRules) -> dict:
    """Buy once at the first open, hold across gaps, and sell at the last close.

    Curve returns mark the held shares at each close. Sample volatility uses
    those net period returns with ddof=1 and is undefined for a single bar.
    """
    data = _inputs(rules, opens=opens, closes=closes)
    units, cash, purchase = _entry(rules.initial_capital, data["opens"][0], 1, rules)
    curve, previous, previous_price = [], rules.initial_capital, data["opens"][0]
    zero = _transaction(0.0, rules)
    for index, close in enumerate(data["closes"]):
        final = index == len(data["closes"]) - 1
        closing_value = _holding_value(units, close)
        sale = _transaction(closing_value, rules) if final else zero
        equity = _sum(
            [cash, closing_value, -sale["fee"], -sale["slippage"]],
            "Buy-and-hold close equity",
        )
        curve.append(
            _record(
                previous,
                equity,
                1,
                units,
                cash=cash,
                ending_cash=equity if final else cash,
                ending_units=0.0 if final else units,
                entry=purchase if index == 0 else zero,
                exit=sale,
                financing=0.0,
                gross_pnl=units * (close - previous_price),
            )
        )
        previous, previous_price = equity, close
    return {"curve": curve, "metrics": _summary(curve, rules.initial_capital)}
