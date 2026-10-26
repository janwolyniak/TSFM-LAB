"""Cash/share accounting uses independently hand-calculated synthetic prices."""

import copy
import math
from dataclasses import replace

import numpy as np
import pytest

from tsfm_lab.finance.contracts import BacktestRules, FinanceError
from tsfm_lab.finance.ledger import buy_and_hold, simulate_intrabar

RULES = BacktestRules(100.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@pytest.mark.parametrize("field", ["opens", "highs", "lows", "closes", "positions"])
def test_masked_ledger_observations_are_rejected(field):
    inputs = dict(
        opens=[100.0], highs=[101.0], lows=[99.0], closes=[100.0], positions=[1]
    )
    inputs[field] = np.ma.array(inputs[field], mask=[True])
    with pytest.raises(FinanceError, match="masked"):
        simulate_intrabar(**inputs, rules=RULES)


def run(opens, closes, positions, *, rules=RULES, highs=None, lows=None):
    highs = np.maximum(opens, closes) if highs is None else highs
    lows = np.minimum(opens, closes) if lows is None else lows
    return simulate_intrabar(opens, highs, lows, closes, positions, rules=rules)


def test_long_cash_shares_and_capital_conservation():
    result = run([10], [12], [1])
    row = result["curve"][0]
    assert row["units"] == 10
    assert row["entry_cash"] == 0
    assert row["exit_units"] == 0
    assert row["exit_cash"] == row["equity"] == 120
    assert row["gross_pnl"] == row["net_pnl"] == 20
    assert row["net_return"] == pytest.approx(0.2)
    assert row["entry_notional"] == 100
    assert row["exit_notional"] == 120
    assert row["turnover_entry_equity"] == pytest.approx(2.2)
    assert row["entry_cash"] + row["units"] * 10 == row["entry_equity"]
    assert result["metrics"]["cumulative_return"] == pytest.approx(0.2)


def test_entry_sizes_after_costs_and_both_legs_pay_fees_and_slippage():
    rules = replace(RULES, initial_capital=103, fee_bps=100, slippage_bps=200)
    result = run([10], [10], [1], rules=rules)
    row = result["curve"][0]
    assert row["units"] == pytest.approx(10)
    assert row["entry_cash"] == pytest.approx(0, abs=1e-12)
    for leg in ["entry", "exit"]:
        assert row[f"{leg}_notional"] == pytest.approx(100)
        assert row[f"{leg}_fee"] == pytest.approx(1)
        assert row[f"{leg}_slippage"] == pytest.approx(2)
    assert row["equity"] == pytest.approx(97)
    assert row["gross_pnl"] == 0
    assert row["net_pnl"] == pytest.approx(-6)
    assert result["metrics"]["fees"] == pytest.approx(2)
    assert result["metrics"]["slippage"] == pytest.approx(4)
    assert result["metrics"]["total_costs"] == pytest.approx(6)
    reconstructed = math.fsum(
        [row["entry_cash"], row["units"] * 10, row["entry_fee"], row["entry_slippage"]]
    )
    assert reconstructed == pytest.approx(103)


def test_short_uses_entry_notional_financing_and_costs_both_legs():
    rules = replace(
        RULES,
        initial_capital=103,
        fee_bps=100,
        slippage_bps=200,
        short_financing_bps=50,
    )
    result = run([10], [9], [-1], rules=rules, highs=[11], lows=[8])
    row = result["curve"][0]
    assert row["units"] == pytest.approx(-10)
    assert row["entry_cash"] == pytest.approx(200)
    assert row["entry_notional"] == pytest.approx(100)
    assert row["exit_notional"] == pytest.approx(90)
    assert row["short_financing"] == pytest.approx(0.5)
    assert row["exit_fee"] == pytest.approx(0.9)
    assert row["exit_slippage"] == pytest.approx(1.8)
    assert row["gross_pnl"] == pytest.approx(10)
    assert row["equity"] == pytest.approx(106.8)
    assert row["net_pnl"] == pytest.approx(3.8)
    assert result["metrics"]["total_costs"] == pytest.approx(6.2)


@pytest.mark.parametrize("position", [-1, 1])
def test_constant_price_cost_oracle_compounds_every_round_trip(position):
    rules = replace(RULES, fee_bps=100, slippage_bps=200, short_financing_bps=50)
    result = run([10, 10], [10, 10], [position, position], rules=rules)
    factor = (1 - 0.03 - (0.005 if position == -1 else 0)) / (1 + 0.03)
    assert result["curve"][0]["equity"] == pytest.approx(100 * factor)
    assert result["metrics"]["final_capital"] == pytest.approx(100 * factor**2)
    assert all(row["exit_units"] == 0 for row in result["curve"])


@pytest.mark.parametrize("positions", [[1, -1], [1, 1]])
def test_every_bar_liquidates_and_reenters_without_boundary_netting(positions):
    result = run([10, 10], [10, 10], positions)
    first, second = result["curve"]
    assert first["exit_notional"] + second["entry_notional"] == 200
    assert first["traded_notional"] == second["traded_notional"] == 200
    assert result["metrics"]["total_traded_notional"] == 400
    assert result["metrics"]["turnover_initial_capital"] == 4


def test_flat_cash_has_no_trades_costs_or_financing():
    rules = replace(RULES, fee_bps=100, slippage_bps=100, short_financing_bps=1e6)
    result = run([10, 30], [12, 20], [0, 0], rules=rules)
    assert [row["equity"] for row in result["curve"]] == [100, 100]
    assert all(row["units"] == 0 for row in result["curve"])
    for key in [
        "total_traded_notional",
        "fees",
        "slippage",
        "short_financing",
        "total_costs",
    ]:
        assert result["metrics"][key] == 0
    assert result["metrics"]["volatility_unannualized"] == {
        "value": 0.0,
        "reason": None,
    }


def test_period_end_drawdown_and_unannualized_sample_volatility():
    result = run([100, 100], [120, 90], [1, 1])
    assert [row["equity"] for row in result["curve"]] == pytest.approx([120, 108])
    assert result["metrics"]["max_drawdown"] == pytest.approx(0.1)
    assert result["metrics"]["volatility_unannualized"] == {
        "value": pytest.approx(math.sqrt(0.045)),
        "reason": None,
    }
    first_loss = run([10], [8], [1])
    assert first_loss["metrics"]["max_drawdown"] == pytest.approx(0.2)
    assert first_loss["metrics"]["volatility_unannualized"] == {
        "value": None,
        "reason": "insufficient_returns",
    }
    intrabar_only = run([10], [10], [1], highs=[100], lows=[1])
    assert intrabar_only["metrics"]["max_drawdown"] == 0


@pytest.mark.parametrize("high", [20, 21])
def test_short_nonpositive_equity_at_observed_high_rejects_even_if_close_recovers(high):
    with pytest.raises(FinanceError, match="observed bar high"):
        run([10], [9], [-1], highs=[high], lows=[8])


def test_nonpositive_equity_after_liquidation_costs_or_financing_is_not_clipped():
    costly = replace(RULES, initial_capital=110, fee_bps=1000)
    with pytest.raises(FinanceError, match="nonpositive after costs"):
        run([100], [190], [-1], rules=costly, highs=[190], lows=[95])
    financed = replace(RULES, short_financing_bps=11000)
    with pytest.raises(FinanceError, match="nonpositive after costs"):
        run([100], [100], [-1], rules=financed)


@pytest.mark.parametrize("position", [True, False, np.bool_(True), 1.0, "1", 2, -2])
def test_positions_are_strict_signed_integers(position):
    with pytest.raises(FinanceError, match="positions"):
        run([10], [10], [position])


@pytest.mark.parametrize(
    "bad", [0, -1, math.nan, math.inf, -math.inf, True, "10", 1 + 2j, 10**400]
)
def test_prices_are_positive_finite_real_numbers(bad):
    with pytest.raises(FinanceError):
        simulate_intrabar([bad], [20], [1], [10], [0], rules=RULES)


@pytest.mark.parametrize(
    "opens,highs,lows,closes,positions",
    [
        ([], [], [], [], []),
        ([10], [10, 10], [10], [10], [0]),
        ([[10]], [10], [10], [10], [0]),
        ([10], [9], [8], [9], [0]),
        ([10], [12], [11], [12], [0]),
        ([10], [12], [9], [13], [0]),
    ],
)
def test_shapes_lengths_and_ohlc_consistency_are_validated(
    opens, highs, lows, closes, positions
):
    with pytest.raises(FinanceError):
        simulate_intrabar(opens, highs, lows, closes, positions, rules=RULES)


def test_underflowed_units_and_overflowed_traded_notional_raise():
    with pytest.raises(FinanceError, match="underflowed"):
        run([1e300], [1e300], [1], rules=replace(RULES, initial_capital=1e-300))
    with pytest.raises(FinanceError, match="finite|overflowed"):
        run([1], [1], [1], rules=replace(RULES, initial_capital=1e308))


def test_nonzero_units_cannot_silently_produce_zero_traded_notional():
    rules = replace(RULES, initial_capital=5e-324, fee_bps=9000)
    with pytest.raises(FinanceError, match="position value underflowed"):
        run([0.49], [0.49], [1], rules=rules)


def test_inputs_are_snapshotted_without_mutation_or_retained_arrays():
    opens, highs, lows, closes, positions = [np.array([10, 10]) for _ in range(5)]
    positions[:] = [1, -1]
    originals = [array.copy() for array in (opens, highs, lows, closes, positions)]
    result = simulate_intrabar(opens, highs, lows, closes, positions, rules=RULES)
    before = copy.deepcopy(result)
    for actual, expected in zip(
        (opens, highs, lows, closes, positions), originals, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
        actual[:] = 999
    assert result == before


def test_costed_buy_and_hold_buys_once_marks_gaps_and_sells_once():
    rules = replace(RULES, initial_capital=110, fee_bps=1000)
    result = buy_and_hold([100, 180], [120, 90], rules=rules)
    first, final = result["curve"]
    assert first["units"] == final["units"] == pytest.approx(1)
    assert first["entry_notional"] == pytest.approx(100)
    assert first["entry_fee"] == pytest.approx(10)
    assert first["exit_notional"] == 0
    assert first["exit_units"] == pytest.approx(1)
    assert first["equity"] == pytest.approx(120)
    assert final["entry_notional"] == final["entry_fee"] == 0
    assert final["exit_notional"] == pytest.approx(90)
    assert final["exit_fee"] == pytest.approx(9)
    assert final["exit_units"] == 0
    assert final["equity"] == final["exit_cash"] == pytest.approx(81)
    assert final["gross_pnl"] == pytest.approx(-30)
    assert final["net_pnl"] == pytest.approx(-39)
    assert result["metrics"]["total_traded_notional"] == pytest.approx(190)
    assert result["metrics"]["fees"] == pytest.approx(19)
    assert result["metrics"]["max_drawdown"] == pytest.approx(0.325)


def test_single_bar_buy_and_hold_charges_both_legs_and_has_undefined_volatility():
    rules = replace(RULES, initial_capital=103, fee_bps=100, slippage_bps=200)
    result = buy_and_hold([10], [10], rules=rules)
    assert result["metrics"]["final_capital"] == pytest.approx(97)
    assert result["metrics"]["total_costs"] == pytest.approx(6)
    assert (
        result["metrics"]["volatility_unannualized"]["reason"] == "insufficient_returns"
    )


def test_buy_and_hold_validates_inputs_and_does_not_charge_short_financing():
    rules = replace(RULES, short_financing_bps=1e6)
    result = buy_and_hold([10, 20], [10, 20], rules=rules)
    assert result["metrics"]["final_capital"] == 200
    assert result["metrics"]["short_financing"] == 0
    for opens, closes in [([], []), ([10], [10, 10]), ([True], [10]), ([10], [0])]:
        with pytest.raises(FinanceError):
            buy_and_hold(opens, closes, rules=rules)


def test_rules_must_be_validated_contract_objects():
    with pytest.raises(FinanceError, match="BacktestRules"):
        run([10], [10], [0], rules={})
