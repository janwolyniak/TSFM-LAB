"""Finance consumes original checked prices and saved predictions, never models."""

from dataclasses import replace

import numpy as np
import pytest
from finance_helpers import price_fixture

from tsfm_lab.contracts import Dataset
from tsfm_lab.finance.backtest import backtest_run
from tsfm_lab.finance.contracts import FinanceError


def evaluate(value, **changes):
    return backtest_run(
        value.run_id,
        value.workspace,
        value.dataset,
        value.ohlc,
        value.rules,
        target_is_price=True,
        **changes,
    )


def test_checked_saved_prices_produce_separate_derived_report(tmp_path):
    value = price_fixture(tmp_path)
    report = evaluate(value)
    assert report["policy"] == "ohlc_next_open_to_target_close_v1"
    assert report["parent"]["run_id"] == value.run_id
    assert report["evaluation"]["signal_positions"] == [1] * 6
    assert len(report["cases"]) == 6
    assert (
        report["parent"]["result"]["metrics"]["overall"]
        == value.result.to_dict()["metrics"]["overall"]
    )
    assert report["implementation"]["environment"]["numpy"]
    assert (
        report["original_dataset"]["canonical_sha256"]
        == value.dataset.provenance["canonical_sha256"]
    )
    assert not (value.workspace / "finance").exists()


def test_higher_horizon_retains_every_lead_one_case(tmp_path):
    value = price_fixture(tmp_path, horizon=3)
    report = evaluate(value)
    assert len(report["cases"]) == value.result.counts["successful_windows"]
    assert all(row["lead"] == 1 for row in report["cases"])


def test_stride_mismatch_is_rejected_before_finance_analysis(tmp_path):
    value = price_fixture(tmp_path, stride=2)
    with pytest.raises(FinanceError, match="stride 1"):
        evaluate(value)


def test_original_dataset_identity_cannot_be_replaced(tmp_path):
    value = price_fixture(tmp_path)
    changed = value.dataset.values.copy()
    changed[0] += 1
    other = Dataset(**{**vars(value.dataset), "values": changed})
    shifted_bars = replace(value.ohlc, closes=changed)
    with pytest.raises(FinanceError, match="identity"):
        backtest_run(
            value.run_id,
            value.workspace,
            other,
            shifted_bars,
            value.rules,
            target_is_price=True,
        )


def test_price_declaration_and_quote_units_are_required(tmp_path):
    value = price_fixture(tmp_path / "valid")
    with pytest.raises(FinanceError, match="declare"):
        backtest_run(
            value.run_id,
            value.workspace,
            value.dataset,
            value.ohlc,
            value.rules,
            target_is_price=False,
        )
    unspecified = price_fixture(tmp_path / "unspecified", units="unspecified")
    with pytest.raises(FinanceError, match="quote units"):
        evaluate(unspecified)


def test_ohlc_closes_must_match_original_forecast_target(tmp_path):
    value = price_fixture(tmp_path)
    changed = value.ohlc.closes.copy()
    changed[3] += 0.1
    bars = replace(value.ohlc, closes=changed)
    with pytest.raises(FinanceError, match="close"):
        backtest_run(
            value.run_id,
            value.workspace,
            value.dataset,
            bars,
            value.rules,
            target_is_price=True,
        )


def test_future_entry_price_changes_execution_but_not_signal(tmp_path):
    value = price_fixture(tmp_path)
    original = evaluate(value)
    openings = value.ohlc.opens.copy()
    openings[2] *= 1.01
    bars = replace(value.ohlc, opens=openings, provenance={})
    changed = backtest_run(
        value.run_id,
        value.workspace,
        value.dataset,
        bars,
        value.rules,
        target_is_price=True,
    )
    assert (
        original["evaluation"]["signal_positions"]
        == changed["evaluation"]["signal_positions"]
    )
    assert (
        original["evaluation"]["strategy"]["curve"][0]["equity"]
        != changed["evaluation"]["strategy"]["curve"][0]["equity"]
    )
    np.testing.assert_array_equal(value.ohlc.opens, value.ohlc.closes)
