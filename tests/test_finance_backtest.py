"""Finance consumes original checked prices and saved predictions, never models."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from finance_helpers import price_fixture

from tsfm_lab.contracts import Dataset
from tsfm_lab.experiments import ExperimentStore, StoreError
from tsfm_lab.finance.backtest import backtest_run
from tsfm_lab.finance.contracts import FinanceError
from tsfm_lab.finance.reports import load_report, save_report


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


def guided_parent(value):
    store = ExperimentStore(value.workspace)
    run_id = store.create("Guided price fixture", {}, "# Direct fixture\n")
    store.complete(
        run_id, value.result.to_dict(), value.result.predictions, include_outputs=True
    )
    return SimpleNamespace(**(vars(value) | {"run_id": run_id}))


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


def test_guided_parent_keeps_report_schema_and_numerical_results(tmp_path):
    original = price_fixture(tmp_path)
    historical_report = evaluate(original)
    guided = guided_parent(original)
    report = evaluate(guided)
    store = ExperimentStore(guided.workspace)
    manifest = store.get_manifest(guided.run_id)
    core = {
        "configuration.json",
        "input.yaml",
        "result.json",
        "metrics.json",
        "predictions.csv",
    }
    assert manifest["schema_version"] == 2
    assert set(manifest["artifacts"]) == core | {"forecast.svg", "provenance.json"}
    assert report["schema_version"] == 1
    assert report["parent"]["run_id"] == guided.run_id
    assert report["parent"]["artifacts"] == {
        name: manifest["artifacts"][name] for name in core
    }
    for field in ("evaluation", "cases", "rules", "resolved_policy"):
        assert report[field] == historical_report[field]
    assert (
        report["parent"]["result"]["metrics"]
        == historical_report["parent"]["result"]["metrics"]
    )
    destination = tmp_path / "guided-finance-report.json"
    save_report(report, destination)
    assert load_report(destination) == report


@pytest.mark.parametrize("artifact", ["forecast.svg", "provenance.json"])
@pytest.mark.parametrize("rehash", [False, True])
def test_guided_parent_output_tampering_fails_before_finance_accounting(
    tmp_path, monkeypatch, artifact, rehash
):
    guided = guided_parent(price_fixture(tmp_path))
    store = ExperimentStore(guided.workspace)
    directory = store.root / guided.run_id
    manifest = store.get_manifest(guided.run_id)
    path = directory / artifact
    content = path.read_bytes() + b"\nchanged output\n"
    path.write_bytes(content)
    if rehash:
        manifest["artifacts"][artifact] = {
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        (directory / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        "tsfm_lab.finance.backtest.evaluate_cases",
        lambda *args, **kwargs: pytest.fail("Corrupt guided parent reached accounting"),
    )
    with pytest.raises(StoreError, match="checksum|size|disagree"):
        evaluate(guided)


@pytest.mark.parametrize("statistic", ["deterministic", "median", "native_point"])
def test_native_task_cannot_enter_finance_through_its_output_label(
    tmp_path, monkeypatch, statistic
):
    value = price_fixture(tmp_path)
    store = ExperimentStore(value.workspace)
    result, frame = store.load_result(value.run_id)
    result["configuration"]["evaluation"]["point_statistic"] = "native_point"
    frame["point_statistic"] = statistic
    run_id = store.create("Native point fixture", {}, "# Direct fixture")
    store.complete(run_id, result, frame)
    monkeypatch.setattr(
        "tsfm_lab.finance.backtest.evaluate_cases",
        lambda *args, **kwargs: pytest.fail("Native task must fail before accounting"),
    )
    with pytest.raises(FinanceError, match="median task"):
        backtest_run(
            run_id,
            value.workspace,
            value.dataset,
            value.ohlc,
            value.rules,
            target_is_price=True,
        )


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
