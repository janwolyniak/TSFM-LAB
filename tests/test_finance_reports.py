"""Self-contained recalculation, strict decoding, and no-replace finance outputs."""

import builtins
import copy
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from tsfm_lab import __version__
from tsfm_lab.contracts import Dataset, EvaluationSpec
from tsfm_lab.data import canonical_fingerprint
from tsfm_lab.evaluation.runner import _environment, run_experiment
from tsfm_lab.experiments import ExperimentStore, _json
from tsfm_lab.finance import reports
from tsfm_lab.finance.analysis import EXECUTION_POLICY, evaluate_cases
from tsfm_lab.finance.contracts import POLICY, BacktestRules, FinanceReportError
from tsfm_lab.finance.ohlc import OHLCData, ohlc_fingerprint
from tsfm_lab.models.baselines import SeasonalNaive


@pytest.fixture
def payload(tmp_path, request):
    options = getattr(request, "param", {})
    timestamps = pd.date_range("2026-01-01", periods=7, freq="h", tz="UTC")
    values = np.array(
        options.get("values", [100.0, 110.0, 105.0, 95.0, 100.0, 110.0, 105.0])
    )
    dataset = Dataset(
        timestamps,
        values,
        "1h",
        target_name="close",
        units="USD",
        series_id="asset",
        provenance={"source_path": "/unavailable/prices.csv"},
    )
    opening = np.r_[99.0, values[:-1]]
    ohlc = OHLCData(
        timestamps - pd.Timedelta(minutes=59),
        timestamps,
        opening,
        np.maximum(opening, values) + 1,
        np.minimum(opening, values) - 1,
        values,
        "1h",
        units="USD",
        series_id="asset",
        provenance={"source_path": "/unavailable/ohlc.csv"},
    )
    spec = EvaluationSpec(
        4,
        1,
        timestamps[4],
        timestamps[-1] + pd.Timedelta("1h"),
        stride=1,
        seasonal_period=options.get("seasonal_period", 2),
        metrics=options.get("metrics", ("mae", "rmse", "mase")),
    )
    result = run_experiment(dataset, SeasonalNaive(2), spec)
    store = ExperimentStore(tmp_path / "parent")
    run_id = store.create("finance fixture", result.configuration, "fixture: true\n")
    store.complete(run_id, result.to_dict(), result.predictions)
    parent, frame = store.load_result(run_id)
    cases = []
    for index, forecast in enumerate(frame.itertuples(), start=4):
        cases.append(
            {
                "request_id": forecast.request_id,
                "series_id": "asset",
                "origin": forecast.origin.isoformat(),
                "target_timestamp": forecast.target_timestamp.isoformat(),
                "lead": 1,
                "point_statistic": forecast.point_statistic,
                "origin_close": float(values[index - 1]),
                "forecast_close": float(forecast.prediction),
                "actual_close": float(forecast.actual),
                "open_timestamp": ohlc.open_timestamps[index].isoformat(),
                "close_timestamp": ohlc.close_timestamps[index].isoformat(),
                "open": float(opening[index]),
                "high": float(ohlc.highs[index]),
                "low": float(ohlc.lows[index]),
                "close": float(values[index]),
            }
        )
    rules = BacktestRules(1000, 10, 1, 2, 3, 0)
    return {
        "schema_version": 1,
        "policy": POLICY,
        "report_id": str(uuid4()),
        "created_at": "2026-01-01T12:00:00+00:00",
        "implementation": {
            "package_version": __version__,
            "git_revision": None,
            "git_dirty": None,
            "environment": _environment(None),
        },
        "parent": {
            "run_id": run_id,
            "result": parent,
            "artifacts": store.get_manifest(run_id)["artifacts"],
        },
        "original_dataset": {
            "canonical_sha256": canonical_fingerprint(dataset),
            "metadata": copy.deepcopy(parent["configuration"]["dataset"]),
            "provenance": dataset.provenance,
            "observations": 7,
        },
        "execution_data": {
            "canonical_sha256": ohlc_fingerprint(ohlc),
            "provenance": ohlc.provenance,
            "observations": 7,
        },
        "rules": asdict(rules),
        "resolved_policy": copy.deepcopy(EXECUTION_POLICY),
        "cases": cases,
        "evaluation": evaluate_cases(cases, rules, frequency="1h", series_id="asset"),
    }


def resign(report):
    body = {key: value for key, value in report.items() if key != "fingerprint"}
    raw = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    report["fingerprint"] = hashlib.sha256(raw).hexdigest()


def resign_parent(value):
    parent = value["parent"]
    for name, body in (
        ("result.json", parent["result"]),
        ("metrics.json", parent["result"]["metrics"]),
    ):
        raw = _json(body)
        parent["artifacts"][name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
        }


@pytest.mark.parametrize("statistic", ["deterministic", "median"])
def test_resigned_native_parent_task_is_ineligible_even_with_old_output_labels(
    payload, statistic
):
    sealed = reports.seal_report(payload)
    sealed["parent"]["result"]["configuration"]["evaluation"][
        "point_statistic"
    ] = "native_point"
    for case in sealed["cases"]:
        case["point_statistic"] = statistic
    resign_parent(sealed)
    resign(sealed)
    with pytest.raises(FinanceReportError, match="median task"):
        reports.validate_report(sealed)


def test_seal_save_load_recalculate_and_preserve_detached_unicode_payload(
    payload, tmp_path
):
    payload["original_dataset"]["provenance"]["note"] = "Zażółć gęślą"
    sealed = reports.seal_report(payload)
    before = copy.deepcopy(sealed)
    path = tmp_path / "derived/research.json"
    identity = reports.save_report(sealed, path)
    raw = path.read_bytes()
    assert identity == {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    assert "Zażółć" in raw.decode()
    assert reports.load_report(path) == sealed
    assert reports.seal_report(sealed) == sealed
    checked = reports.validate_report(sealed)
    checked["cases"][0]["forecast_close"] = 999
    payload["cases"][0]["forecast_close"] = 777
    assert sealed == before


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "policy",
        "report_id",
        "created_at",
        "implementation",
        "parent",
        "original_dataset",
        "execution_data",
        "rules",
        "resolved_policy",
        "cases",
        "evaluation",
    ],
)
def test_required_fields_cannot_be_omitted(payload, field):
    del payload[field]
    with pytest.raises(FinanceReportError):
        reports.seal_report(payload)


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"policy": "unknown"},
        {"report_id": "not-a-uuid"},
        {"created_at": "2026-01-01"},
        {"extra": 1},
        {"cases": []},
    ],
)
def test_scalar_schema_and_complete_coverage_are_strict(payload, changes):
    payload.update(changes)
    with pytest.raises(FinanceReportError):
        reports.seal_report(payload)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("rules", "fee_bps", True),
        ("rules", "threshold_bps", -1),
        ("rules", "new_option", 0),
        ("implementation", "git_dirty", 1),
        ("implementation", "package_version", ""),
        ("implementation", "environment", {}),
        ("original_dataset", "observations", True),
        ("execution_data", "canonical_sha256", "A" * 64),
        ("execution_data", "observations", 6),
        ("original_dataset", "canonical_sha256", "a" * 64),
    ],
)
def test_invalid_rules_provenance_and_input_identities_fail(
    payload, section, field, value
):
    payload[section][field] = value
    with pytest.raises(FinanceReportError):
        reports.seal_report(payload)


def test_parent_artifact_checksums_and_parent_grid_counts_are_verified(payload):
    corrupted = copy.deepcopy(payload)
    corrupted["parent"]["artifacts"]["result.json"]["sha256"] = "f" * 64
    with pytest.raises(FinanceReportError, match="Copied parent"):
        reports.seal_report(corrupted)
    payload["parent"]["result"]["counts"]["successful_windows"] += 1
    resign_parent(payload)
    with pytest.raises(FinanceReportError, match="counts"):
        reports.seal_report(payload)


def test_resealed_report_still_requires_explicit_price_quote_units(payload):
    report = reports.seal_report(payload)
    report["parent"]["result"]["configuration"]["dataset"]["units"] = "unspecified"
    report["original_dataset"]["metadata"]["units"] = "unspecified"
    resign_parent(report)
    resign(report)
    with pytest.raises(FinanceReportError, match="price quote units"):
        reports.validate_report(report)
    with pytest.raises(FinanceReportError, match="price quote units"):
        reports.seal_report(report)


def test_saved_cases_are_bound_to_the_parent_evaluation_grid(payload):
    for case in payload["cases"]:
        for name in ("origin", "target_timestamp", "open_timestamp", "close_timestamp"):
            case[name] = (pd.Timestamp(case[name]) + pd.Timedelta("1h")).isoformat()
    payload["evaluation"] = evaluate_cases(
        payload["cases"],
        BacktestRules(**payload["rules"]),
        frequency="1h",
        series_id="asset",
    )
    with pytest.raises(FinanceReportError, match="parent origin grid"):
        reports.seal_report(payload)


def test_resealing_never_repairs_fabricated_capital_or_metric_values(payload):
    sealed = reports.seal_report(payload)
    sealed["evaluation"]["strategy"]["metrics"]["final_capital"] += 1
    resign(sealed)
    with pytest.raises(FinanceReportError, match="recalculated"):
        reports.validate_report(sealed)
    with pytest.raises(FinanceReportError, match="recalculated"):
        reports.seal_report(sealed)


def test_altered_forecasts_and_recalculated_finance_cannot_keep_parent_metrics(payload):
    report = reports.seal_report(payload)
    report["cases"][0]["forecast_close"] += 2
    report["evaluation"] = evaluate_cases(
        report["cases"],
        BacktestRules(**report["rules"]),
        frequency="1h",
        series_id="asset",
    )
    resign(report)
    with pytest.raises(FinanceReportError, match="parent lead-1 metrics"):
        reports.validate_report(report)
    with pytest.raises(FinanceReportError, match="parent lead-1 metrics"):
        reports.seal_report(report)


def test_unknown_parent_metric_semantics_cannot_be_silently_reinterpreted(payload):
    payload["parent"]["result"]["provenance"]["metric_version"] = "future_metrics_v2"
    resign_parent(payload)
    with pytest.raises(FinanceReportError, match="metric version"):
        reports.seal_report(payload)


@pytest.mark.parametrize(
    "payload,reason",
    [
        ({"metrics": ("rmse",)}, "not_requested"),
        ({"values": [100.0] * 7}, "zero_scale"),
        ({"seasonal_period": 8}, "insufficient_scale_history"),
    ],
    indirect=["payload"],
)
def test_parent_forecast_consistency_preserves_metric_subsets_and_undefined_mase(
    payload, reason
):
    assert payload["parent"]["result"]["metrics"]["scale_reason"] == reason
    assert (
        reports.validate_report(reports.seal_report(payload))["cases"]
        == payload["cases"]
    )


def test_counts_booleans_and_resolved_policy_types_are_not_numeric_aliases(payload):
    payload["evaluation"]["cases"] = float(len(payload["cases"]))
    with pytest.raises(FinanceReportError, match="recalculated"):
        reports.seal_report(payload)
    payload["evaluation"]["cases"] = len(payload["cases"])
    payload["resolved_policy"]["cash_interest"] = False
    with pytest.raises(FinanceReportError, match="Resolved policy"):
        reports.seal_report(payload)


def test_numeric_recalculation_tolerance_is_relative_and_never_repairs_values(payload):
    metric = payload["evaluation"]["strategy"]["metrics"]
    metric["final_capital"] *= 1 + 1e-13
    sealed = reports.seal_report(payload)
    assert (
        sealed["evaluation"]["strategy"]["metrics"]["final_capital"]
        == metric["final_capital"]
    )
    payload["evaluation"]["benchmarks"]["cash"]["metrics"]["fees"] = 1e-200
    with pytest.raises(FinanceReportError, match="recalculated"):
        reports.seal_report(payload)


def test_fingerprint_detects_mutation_before_recalculation(payload):
    report = reports.seal_report(payload)
    report["rules"]["threshold_bps"] += 1
    with pytest.raises(FinanceReportError, match="fingerprint"):
        reports.validate_report(report)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"outer":{"a":1,"a":2}}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":-Infinity}',
        b'{"x":1e999}',
        b"[]",
        b"\xff",
        b"",
    ],
)
def test_load_rejects_duplicate_nonfinite_and_malformed_json(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_bytes(raw)
    with pytest.raises(FinanceReportError):
        reports.load_report(path)


def test_file_limit_applies_before_read_decode_and_before_any_write(
    payload, tmp_path, monkeypatch
):
    report = reports.seal_report(payload)
    monkeypatch.setattr(reports, "MAX_REPORT_BYTES", 32)
    source = tmp_path / "large.json"
    source.write_bytes(b" " * 33)
    with pytest.raises(FinanceReportError, match="64 MiB"):
        reports.load_report(source)
    target = tmp_path / "absent/output.json"
    with pytest.raises(FinanceReportError, match="64 MiB"):
        reports.save_report(report, target)
    assert not target.parent.exists()


@pytest.mark.parametrize("kind", ["regular", "symlink", "dangling_symlink"])
def test_existing_destinations_are_never_replaced(payload, tmp_path, kind, symlinks):
    target, source = tmp_path / "target.json", tmp_path / "user.txt"
    source.write_text("user-owned")
    if kind == "regular":
        target.write_text("existing")
    else:
        symlinks(target, source if kind == "symlink" else tmp_path / "missing")
    with pytest.raises(FileExistsError):
        reports.save_report(reports.seal_report(payload), target)
    assert source.read_text() == "user-owned"
    assert (
        target.is_symlink() if kind != "regular" else target.read_text() == "existing"
    )


def test_failed_atomic_publication_removes_only_its_own_temporary_file(
    payload, tmp_path, monkeypatch
):
    target = tmp_path / "exports/report.json"

    def failure(*args, **kwargs):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(reports.os, "link", failure)
    with pytest.raises(OSError, match="simulated"):
        reports.save_report(reports.seal_report(payload), target)
    assert not target.exists() and list(target.parent.iterdir()) == []


def test_publication_race_preserves_the_concurrent_file(payload, tmp_path, monkeypatch):
    target = tmp_path / "report.json"
    original = os.link

    def race(source, destination):
        Path(destination).write_bytes(b"concurrent user output")
        original(source, destination)

    monkeypatch.setattr(reports.os, "link", race)
    with pytest.raises(FileExistsError):
        reports.save_report(reports.seal_report(payload), target)
    assert target.read_bytes() == b"concurrent user output"
    assert not list(tmp_path.glob(".tsfm-finance-*"))


def test_reload_needs_no_source_files_git_lookup_or_optional_runtime(
    payload, tmp_path, monkeypatch
):
    report = reports.seal_report(payload)
    target = tmp_path / "portable.json"
    reports.save_report(report, target)
    original_open, original_import = Path.open, builtins.__import__

    def only_report(path, *args, **kwargs):
        if path != target:
            raise AssertionError(f"Unexpected source read: {path}")
        return original_open(path, *args, **kwargs)

    def no_model(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "chronos", "timesfm", "huggingface_hub"}:
            raise AssertionError(f"Unexpected model runtime: {name}")
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Reload must use only embedded inputs")

    monkeypatch.setattr(Path, "open", only_report)
    monkeypatch.setattr(builtins, "__import__", no_model)
    monkeypatch.setattr(ExperimentStore, "load_result", forbidden)
    monkeypatch.setattr("tsfm_lab.evaluation.runner._source_revision", forbidden)
    assert reports.load_report(target) == report
