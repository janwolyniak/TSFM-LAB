"""Explicit explanation facts retain local truth while excluding private inputs."""

import builtins
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from finance_helpers import price_fixture

from tsfm_lab.assistant.evidence import (
    EvidenceError,
    bounded_json,
    experiment_evidence,
    fact_map,
    finance_evidence,
    model_evidence,
)
from tsfm_lab.contracts import Dataset, EvaluationSpec
from tsfm_lab.evaluation.runner import run_experiment
from tsfm_lab.experiments import ExperimentStore, StoreError
from tsfm_lab.finance.backtest import backtest_run
from tsfm_lab.finance.contracts import FinanceReportError
from tsfm_lab.finance.reports import save_report
from tsfm_lab.models.baselines import LastValue


@pytest.fixture
def saved_experiment(tmp_path):
    times = pd.date_range("2026-01-01", periods=40, freq="h", tz="UTC")
    dataset = Dataset(
        times,
        np.arange(40, dtype=float) * 0.12345678901234567,
        "1h",
        target_name="PRIVATE_TARGET",
        series_id="PRIVATE_SERIES",
        units="°C",
        provenance={"source_path": "/private/PRIVATE_DATA.csv"},
    )
    workflow = {
        "schema_version": 1,
        "profile": "weather",
        "options": {
            "variable": "PRIVATE_VARIABLE",
            "station": "PRIVATE_STATION",
            "measurement_semantics": "other",
            "description": "PRIVATE_DESCRIPTION",
        },
    }
    result = run_experiment(
        dataset,
        LastValue(),
        EvaluationSpec(
            4, 14, times[20], times[-1] + pd.Timedelta("1h"), seasonal_period=45
        ),
        workflow=workflow,
    )
    payload = result.to_dict()
    payload["name"] = "PRIVATE_RESULT_NAME"
    payload["configuration"]["output"] = {"directory": "/private/PRIVATE_OUTPUT"}
    payload["configuration"]["runtime"]["environment"] = "PRIVATE_RUNTIME"
    payload["configuration"]["runtime"]["deterministic"] = {"path": "PRIVATE_FLAG"}
    payload["provenance"]["environment"]["PRIVATE_ENVIRONMENT"] = "secret"
    store = ExperimentStore(tmp_path / "PRIVATE_WORKSPACE")
    run_id = store.create("PRIVATE_NAME", payload["configuration"], "PRIVATE_YAML")
    store.complete(run_id, payload, result.predictions)
    return store, run_id


def test_experiment_projection_preserves_exact_values_units_reasons_and_periods(
    saved_experiment,
):
    store, run_id = saved_experiment
    source, _ = store.load_result(run_id)
    context = experiment_evidence(store, run_id)
    data = context["data"]
    assert context["schema_version"] == 1 and context["kind"] == "experiment"
    assert data["metrics"]["overall"] == source["metrics"]["overall"]
    assert data["metrics"]["overall"]["mase"] == {
        "value": None,
        "reason": "insufficient_scale_history",
    }
    assert data["metrics"]["units"] == {
        "mae": "°C",
        "rmse": "°C",
        "mase": "dimensionless",
    }
    assert data["configuration"]["evaluation"] == source["configuration"]["evaluation"]
    assert data["configuration"]["workflow"] == {
        "profile": "weather",
        "options": {"measurement_semantics": "other"},
    }
    assert data["counts"] == source["counts"]
    assert data["scale_history"] == source["provenance"]["scale_history"]
    assert data["metrics"]["by_lead"] == {
        str(lead): source["metrics"]["by_lead"][str(lead)] for lead in range(1, 13)
    }
    assert data["metrics"]["included_leads"] == 12
    assert data["metrics"]["total_leads"] == 14
    assert context["omitted"]["by_lead_count"] == 2
    assert "PRIVATE" not in bounded_json(context)
    assert run_id not in bounded_json(context)
    assert json.loads(bounded_json(context)) == context


def test_returned_evidence_is_detached_and_does_not_rewrite_records(saved_experiment):
    store, run_id = saved_experiment
    before = {path.name: path.read_bytes() for path in (store.root / run_id).iterdir()}
    context = experiment_evidence(store, run_id)
    context["data"]["metrics"]["overall"]["mae"]["value"] = 999
    assert (
        experiment_evidence(store, run_id)["data"]["metrics"]["overall"]["mae"]["value"]
        != 999
    )
    assert before == {
        path.name: path.read_bytes() for path in (store.root / run_id).iterdir()
    }


def test_full_prediction_validation_is_mandatory(saved_experiment):
    store, run_id = saved_experiment
    artifact = store.root / run_id / "predictions.csv"
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    # Lifecycle-only metadata remains readable; assistant evidence must reject it.
    assert store.result_summary(run_id)["status"] == "completed"
    with pytest.raises(StoreError, match="checksum|size"):
        experiment_evidence(store, run_id)


def test_pending_configuration_cannot_be_presented_as_a_validated_result(tmp_path):
    store = ExperimentStore(tmp_path)
    run_id = store.create("pending", {}, "")
    with pytest.raises(StoreError, match="completed"):
        experiment_evidence(store, run_id)


def test_legacy_profile_and_custom_adapter_names_stay_local(tmp_path):
    fixture = price_fixture(tmp_path / "prices")
    context = experiment_evidence(ExperimentStore(fixture.workspace), fixture.run_id)
    assert context["data"]["configuration"]["workflow"] == {"profile": "generic"}
    assert context["data"]["configuration"]["model"] == {"catalog_supported": False}
    assert "price_fixture" not in bounded_json(context)
    assert "toy_asset" not in bounded_json(context)


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_results_keep_forecast_options_and_saved_verification_without_paths(
    tmp_path, timesfm_checkpoint, timesfm_runtime, monkeypatch, legacy
):
    from tsfm_lab.models.checkpoints import inspect_local_checkpoint
    from tsfm_lab.models.timesfm import TimesFM

    times = pd.date_range("2026-01-01", periods=16, freq="h", tz="UTC")
    dataset = Dataset(times, np.arange(100.0, 116.0), "1h", units="USD")
    model = TimesFM(inspect_local_checkpoint(timesfm_checkpoint))
    result = run_experiment(dataset, model, EvaluationSpec(4, 2, times[8], times[12]))
    payload = result.to_dict()
    if legacy:
        del payload["provenance"]["project_verification"]
    store = ExperimentStore(tmp_path / "workspace")
    run_id = store.create("PRIVATE_NAME", {}, "PRIVATE_YAML")
    store.complete(run_id, payload, result.predictions)
    monkeypatch.setattr(
        TimesFM, "load", lambda *args: pytest.fail("Inference during explanation")
    )
    context = experiment_evidence(store, run_id)
    configuration = context["data"]["configuration"]
    assert (
        configuration["runtime"]["forecast_options"]
        == result.configuration["runtime"]["forecast_options"]
    )
    assert configuration["model"]["checkpoint"]["source"] == "local"
    assert configuration["model"]["checkpoint"]["resolved_revision"] is None
    if legacy:
        assert configuration["model"]["project_verification"] is None
    else:
        assert (
            configuration["model"]["project_verification"]["status"] == "not_recorded"
        )
        assert (
            configuration["model"]["project_verification"][
                "selected_checkpoint_check_count"
            ]
            == 0
        )
    text = bounded_json(context)
    assert str(timesfm_checkpoint) not in text
    assert '"files"' not in text and '"fingerprint"' not in text


def test_catalog_model_variants_keep_individual_and_representative_checks_separate():
    context = model_evidence("chronos_bolt")
    variants = {item["variant"]: item for item in context["data"]["variants"]}
    assert variants["base"]["project_inference_status"] == "not_recorded"
    assert "individual inference not run" in variants["base"]["verification_label"]
    assert variants["tiny"]["project_inference_status"] == "recorded"
    checks = context["data"]["adapter_representative_checks"]
    assert {check["repository"] for check in checks} == {
        "amazon/chronos-bolt-tiny",
        "amazon/chronos-bolt-small",
    }
    assert all(
        set(check) == {"repository", "revision", "device", "dtype"} for check in checks
    )
    text = bounded_json(context)
    assert "environment" not in text.replace("source_yaml_and_environment", "")
    assert "docs/" not in text and "config.json" not in text
    selected = model_evidence("ibm-granite/granite-timeseries-ttm-r2", "1024-96-r2")
    data = selected["data"]
    assert (
        data["selected_variant"]["min_context"]
        == data["selected_variant"]["max_context"]
        == 1024
    )
    assert data["point_statistic"] == "native_point"
    assert data["capabilities"]["max_context"] == 1024


def test_model_evidence_never_loads_optional_runtimes_or_probes_paths(monkeypatch):
    original = builtins.__import__

    def no_optional(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch",
            "chronos",
            "timesfm",
            "tirex",
            "tsfm_public",
            "huggingface_hub",
        }:
            raise AssertionError(f"Unexpected optional import: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_optional)
    monkeypatch.setattr(
        Path, "open", lambda *args, **kwargs: pytest.fail("Model info probed files")
    )
    assert model_evidence("last_value")["data"]["variants"] == []
    assert model_evidence("ttm_r2")["data"]["supported_tasks"] == ["native_point"]
    with pytest.raises(ValueError, match="Unsupported model"):
        model_evidence("/private/arbitrary")


@pytest.mark.parametrize("single_bar", [False, True])
def test_saved_finance_projection_recalculates_without_parent_or_source_access(
    tmp_path, monkeypatch, single_bar
):
    fixture = price_fixture(
        tmp_path / "prices", closes=[100, 101, 102] if single_bar else None
    )
    report = backtest_run(
        fixture.run_id,
        fixture.workspace,
        fixture.dataset,
        fixture.ohlc,
        fixture.rules,
        target_is_price=True,
    )
    path = tmp_path / "PRIVATE_REPORT.json"
    save_report(report, path)
    original_open = Path.open

    def only_report(source, *args, **kwargs):
        assert source == path, f"Unexpected file read: {source}"
        return original_open(source, *args, **kwargs)

    monkeypatch.setattr(Path, "open", only_report)
    monkeypatch.setattr(
        ExperimentStore, "load_result", lambda *args: pytest.fail("Parent store opened")
    )
    context = finance_evidence(path)
    data = context["data"]
    for name, source in (
        ("strategy", report["evaluation"]["strategy"]),
        *report["evaluation"]["benchmarks"].items(),
    ):
        assert data["capital_metrics"][name] == source["metrics"]
    if single_bar:
        assert data["capital_metrics"]["strategy"]["volatility_unannualized"] == {
            "value": None,
            "reason": "insufficient_returns",
        }
    assert data["rules"] == report["rules"]
    assert data["resolved_policy"] == report["resolved_policy"]
    assert data["rule_units"]["fee_bps"] == "basis_points"
    assert data["rule_units"]["signal_latency_seconds"] == "seconds"
    assert data["metric_units"]["final_capital"] == "toy_quote"
    assert data["metric_units"]["volatility_unannualized"] == "dimensionless"
    text = bounded_json(context)
    for excluded in (
        "PRIVATE",
        str(fixture.ohlc_path),
        fixture.run_id,
        "forecast_implied_returns",
        "signal_positions",
        '"curve"',
        '"origin_close"',
    ):
        assert excluded not in text
    assert len(text.encode("utf-8")) <= 16000


def test_finance_fingerprint_and_recalculation_are_mandatory(tmp_path):
    fixture = price_fixture(tmp_path / "prices")
    report = backtest_run(
        fixture.run_id,
        fixture.workspace,
        fixture.dataset,
        fixture.ohlc,
        fixture.rules,
        target_is_price=True,
    )
    path = tmp_path / "report.json"
    report["evaluation"]["strategy"]["metrics"]["final_capital"] += 1
    path.write_text(json.dumps(report))
    with pytest.raises(FinanceReportError, match="fingerprint"):
        finance_evidence(path)


def test_fact_ids_are_stable_json_pointers_and_keep_native_scalar_values():
    context = {
        "schema_version": 1,
        "kind": "test",
        "data": {"a/b~c": [None, True, 1.2345678901234567]},
        "glossary": {"units": "°C"},
        "omitted": {"secret": True},
    }
    assert fact_map(context) == {
        "/data/a~1b~0c/0": None,
        "/data/a~1b~0c/1": True,
        "/data/a~1b~0c/2": 1.2345678901234567,
        "/glossary/units": "°C",
    }
    assert fact_map(json.loads(bounded_json(context))) == fact_map(context)


def test_bounded_json_counts_utf8_bytes_without_rounding_or_truncation():
    context = {"data": {"unit": "°C", "exact": 1.2345678901234567}}
    original = copy.deepcopy(context)
    text = bounded_json(context)
    assert json.loads(text) == original
    assert bounded_json(context, max_bytes=len(text.encode("utf-8"))) == text
    with pytest.raises(EvidenceError, match="byte limit"):
        bounded_json(context, max_bytes=len(text.encode("utf-8")) - 1)
    assert context == original


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), (1,), Path("/private"), {1: "value"}]
)
def test_non_native_or_nonfinite_context_is_rejected(value):
    with pytest.raises(EvidenceError):
        bounded_json({"data": value})


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5])
def test_byte_limits_are_strict_positive_integers(limit):
    with pytest.raises(EvidenceError):
        bounded_json({}, max_bytes=limit)
