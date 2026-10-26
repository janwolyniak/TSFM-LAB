"""Project evidence is historical provenance, separate from checkpoint identity."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsfm_lab.artifact_validation import validate_predictions
from tsfm_lab.contracts import Dataset, EvaluationSpec
from tsfm_lab.evaluation.runner import run_experiment
from tsfm_lab.experiments import ExperimentStore, StoreError
from tsfm_lab.finance.backtest import backtest_run
from tsfm_lab.finance.contracts import BacktestRules
from tsfm_lab.finance.ohlc import OHLCData
from tsfm_lab.finance.reports import save_report
from tsfm_lab.models import verification
from tsfm_lab.models.baselines import LastValue
from tsfm_lab.models.checkpoints import (
    inspect_local_checkpoint,
    validate_checkpoint_record,
)
from tsfm_lab.models.timesfm import TimesFM


@pytest.fixture
def checkpoint(timesfm_checkpoint):
    return inspect_local_checkpoint(timesfm_checkpoint)


def historical_snapshot(record):
    snapshot = verification.checkpoint_verification(record)
    evidence = {
        "id": "historical-synthetic-fixture-no-current-registry-entry",
        "adapter_id": "timesfm_2p5",
        "repository": "google/timesfm-2.5-200m-pytorch",
        "revision": "a" * 40,
        "files": {
            name: copy.deepcopy(record["files"][name])
            for name in ("config.json", "model.safetensors")
        },
        "source": {
            "package_version": None,
            "git_revision": None,
            "git_dirty": None,
        },
        "runtime": {"python": "3.13.6", "os": "fixture", "machine": "arm64"},
        "device": "cpu",
        "dtype": "float32",
        "checks": ["synthetic aligned finite outputs"],
        "reference": "docs/historical-fixture.md",
    }
    snapshot["project_inference_status"] = "recorded"
    snapshot["selected_checkpoint_checks"] = [evidence]
    snapshot["adapter_representative_checks"] = [copy.deepcopy(evidence)]
    return snapshot


def experiment(model):
    timestamps = pd.date_range("2026-01-01", periods=16, freq="h", tz="UTC")
    dataset = Dataset(
        timestamps,
        np.arange(100.0, 116.0),
        "1h",
        target_name="close",
        units="USD",
    )
    spec = EvaluationSpec(4, 2, timestamps[8], timestamps[12], stride=1)
    return dataset, spec, run_experiment(dataset, model, spec)


def test_metadata_snapshot_preserves_original_schema_one_checkpoint_identity(
    checkpoint,
):
    record = checkpoint.record()
    before = copy.deepcopy(record)
    fields = (
        "files",
        "architecture",
        "pipeline_class",
        "model_type",
        "native_context_length",
        "native_prediction_length",
        "quantiles",
        "license",
    )
    original = {
        "serialization": "tsfm_checkpoint_v1",
        **{name: record[name] for name in fields},
    }
    raw = json.dumps(original, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(raw).hexdigest() == record["fingerprint"]
    snapshot = verification.checkpoint_verification(record)
    assert record == before == validate_checkpoint_record(record)
    assert "project_verification" not in record and record["schema_version"] == 1
    assert snapshot["checkpoint_fingerprint"] == record["fingerprint"]
    assert snapshot["compatibility"] == {
        "status": "checked",
        "scope": "metadata_and_loader_profile",
    }
    assert snapshot["project_inference_status"] == "not_recorded"
    assert snapshot["selected_checkpoint_checks"] == []
    assert (
        verification.verification_label(None) == "Project verification was not recorded"
    )


def test_historical_snapshot_validation_never_consults_current_evidence(
    checkpoint, monkeypatch
):
    record = checkpoint.record()
    snapshot = historical_snapshot(record)

    class NoCurrentEvidence:
        def __iter__(self):
            raise AssertionError(
                "Historical validation cannot consult today's evidence"
            )

    monkeypatch.setattr(verification, "PROJECT_CHECKS", NoCurrentEvidence())
    checked = verification.validate_verification_record(snapshot, record, "timesfm_2p5")
    assert checked == snapshot
    checked["selected_checkpoint_checks"][0]["runtime"]["python"] = "changed"
    assert snapshot["selected_checkpoint_checks"][0]["runtime"]["python"] == "3.13.6"
    monkeypatch.setattr(verification, "PROJECT_CHECKS", ())
    assert (
        verification.validate_verification_record(snapshot, record, "timesfm_2p5")
        == snapshot
    )


def test_writer_matches_exact_files_and_returns_fresh_evidence(checkpoint, monkeypatch):
    record = checkpoint.record()
    evidence = historical_snapshot(record)["selected_checkpoint_checks"][0]
    before = copy.deepcopy(evidence)
    monkeypatch.setattr(verification, "PROJECT_CHECKS", (evidence,))
    snapshot = verification.checkpoint_verification(record)
    assert snapshot["project_inference_status"] == "recorded"
    assert snapshot["selected_checkpoint_checks"] == [before]
    assert verification.verification_label(snapshot) == (
        "Project inference check recorded for these config/weight files"
    )
    snapshot["selected_checkpoint_checks"][0]["runtime"]["python"] = "changed"
    assert evidence == before
    assert verification.checkpoint_verification(record)[
        "selected_checkpoint_checks"
    ] == [before]


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema_version",), True),
        (("adapter_id",), "chronos_bolt"),
        (("checkpoint_fingerprint",), "0" * 64),
        (("compatibility", "status"), "unchecked"),
        (("compatibility", "scope"), "full_scientific_validation"),
        (("project_inference_status",), "not_recorded"),
        (("selected_checkpoint_checks",), []),
        (("selected_checkpoint_checks", 0, "adapter_id"), "chronos_bolt"),
        (("selected_checkpoint_checks", 0, "revision"), "main"),
        (("selected_checkpoint_checks", 0, "files", "config.json", "sha256"), "0" * 64),
        (("selected_checkpoint_checks", 0, "files", "model.safetensors", "size"), True),
        (("selected_checkpoint_checks", 0, "source", "git_dirty"), 0),
        (("selected_checkpoint_checks", 0, "runtime", "python"), None),
        (("selected_checkpoint_checks", 0, "checks"), []),
        (("selected_checkpoint_checks", 0, "reference"), ""),
        (("adapter_representative_checks", 0, "adapter_id"), "chronos_bolt"),
    ],
)
def test_inconsistent_or_malformed_historical_snapshots_fail(checkpoint, path, value):
    record = checkpoint.record()
    snapshot = historical_snapshot(record)
    selected = snapshot
    for key in path[:-1]:
        selected = selected[key]
    selected[path[-1]] = value
    with pytest.raises(ValueError):
        verification.validate_verification_record(snapshot, record, "timesfm_2p5")


@pytest.mark.parametrize("value", [np.int64(1), float("nan"), float("inf")])
def test_snapshots_require_native_finite_json(checkpoint, value):
    record = checkpoint.record()
    snapshot = historical_snapshot(record)
    snapshot["schema_version"] = value
    with pytest.raises(ValueError):
        verification.validate_verification_record(snapshot, record, "timesfm_2p5")


def test_snapshot_size_is_bounded_and_check_ids_are_unique(checkpoint):
    record = checkpoint.record()
    snapshot = historical_snapshot(record)
    snapshot["adapter_representative_checks"] *= 2
    with pytest.raises(ValueError, match="unique"):
        verification.validate_verification_record(snapshot, record, "timesfm_2p5")
    snapshot = historical_snapshot(record)
    snapshot["adapter_representative_checks"][0]["reference"] = "x" * (1024 * 1024)
    with pytest.raises(ValueError, match="limit"):
        verification.validate_verification_record(snapshot, record, "timesfm_2p5")


def test_verification_hook_runs_once_before_loading_and_owns_its_snapshot(
    checkpoint, timesfm_runtime, monkeypatch
):
    model = TimesFM(checkpoint)
    snapshot = verification.checkpoint_verification(checkpoint.record())
    before, options, calls = copy.deepcopy(snapshot), model.configuration(), []

    def hook():
        assert not timesfm_runtime.load_calls
        calls.append("snapshot")
        return snapshot

    original_load = model.load

    def load():
        snapshot["compatibility"]["scope"] = "adapter changed its original buffer"
        original_load()

    monkeypatch.setattr(model, "verification_metadata", hook)
    monkeypatch.setattr(model, "load", load)
    _, _, result = experiment(model)
    assert calls == ["snapshot"]
    assert result.provenance["project_verification"] == before
    assert result.configuration["model"]["options"] == options
    assert (
        result.provenance["project_verification"]["project_inference_status"]
        == "not_recorded"
    )


def test_invalid_supplied_verification_fails_before_model_allocation(
    checkpoint, timesfm_runtime, monkeypatch
):
    model = TimesFM(checkpoint)
    monkeypatch.setattr(
        model, "verification_metadata", lambda: {"schema_version": True}
    )
    with pytest.raises(ValueError):
        experiment(model)
    assert not timesfm_runtime.load_calls


@pytest.mark.parametrize(
    "hook", [lambda: {}, {}], ids=["missing-checkpoint", "noncallable"]
)
def test_invalid_hook_contract_raises_value_error_before_loading(monkeypatch, hook):
    model, loads = LastValue(), []
    monkeypatch.setattr(model, "verification_metadata", hook, raising=False)
    monkeypatch.setattr(model, "load", lambda: loads.append("allocated"))
    with pytest.raises(ValueError, match="verification|checkpoint"):
        experiment(model)
    assert loads == []


def test_baseline_without_hook_and_legacy_checkpoint_records_stay_unchanged(
    checkpoint, timesfm_runtime
):
    _, _, baseline = experiment(LastValue())
    assert "project_verification" not in baseline.provenance
    _, _, result = experiment(TimesFM(checkpoint))
    legacy = json.loads(json.dumps(result.to_dict()))
    del legacy["provenance"]["project_verification"]
    before = copy.deepcopy(legacy)
    validate_predictions(result.predictions, legacy)
    assert legacy == before and "project_verification" not in legacy["provenance"]


def test_saved_result_validation_checks_the_optional_snapshot(
    checkpoint, timesfm_runtime
):
    _, _, result = experiment(TimesFM(checkpoint))
    payload = json.loads(json.dumps(result.to_dict()))
    payload["provenance"]["project_verification"]["checkpoint_fingerprint"] = "0" * 64
    with pytest.raises(StoreError):
        validate_predictions(result.predictions, payload)


def test_saved_forecasts_comparison_and_finance_reload_without_registry_or_runtime(
    checkpoint, timesfm_checkpoint, timesfm_runtime, tmp_path
):
    dataset, spec, result = experiment(TimesFM(checkpoint))
    store = ExperimentStore(tmp_path / "workspace")
    identifiers = []
    for legacy in (False, True):
        payload = result.to_dict()
        if legacy:
            del payload["provenance"]["project_verification"]
        identifier = store.create(
            "Historical verification fixture", {}, "fixture: true\n"
        )
        store.complete(identifier, payload, result.predictions)
        identifiers.append(identifier)
    baseline = run_experiment(dataset, LastValue(), spec)
    identifier = store.create("Baseline", {}, "fixture: true\n")
    store.complete(identifier, baseline.to_dict(), baseline.predictions)
    identifiers.append(identifier)
    bars = OHLCData(
        dataset.timestamps - pd.Timedelta(minutes=59),
        dataset.timestamps,
        dataset.values - 0.2,
        dataset.values + 1,
        dataset.values - 1,
        dataset.values,
        "1h",
        units="USD",
    )
    report = backtest_run(
        identifiers[0],
        store.workspace,
        dataset,
        bars,
        BacktestRules(1000, 0, 0, 0, 0, 0),
        target_is_price=True,
    )
    target = tmp_path / "report.json"
    save_report(report, target)
    before = (store.root / identifiers[1] / "result.json").read_bytes()
    timesfm_checkpoint.rename(tmp_path / "unavailable-checkpoint")
    program = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'timesfm', 'chronos', 'transformers', 'huggingface_hub', 'safetensors'}:
            raise AssertionError('Unexpected optional runtime: ' + fullname)
sys.meta_path.insert(0, Block())
from tsfm_lab.models import verification
class NoEvidence:
    def __iter__(self):
        raise AssertionError('Reload consulted current project evidence')
verification.PROJECT_CHECKS = NoEvidence()
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.comparison import compare_runs
from tsfm_lab.finance.reports import load_report
workspace, report, *identifiers = sys.argv[1:]
store = ExperimentStore(workspace)
new, _ = store.load_result(identifiers[0])
old, _ = store.load_result(identifiers[1])
assert new['provenance']['project_verification']['project_inference_status'] == 'not_recorded'
assert 'project_verification' not in old['provenance']
assert compare_runs(identifiers, workspace)['compatible']
assert load_report(report)['evaluation']['cases'] == 3
"""
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(store.workspace),
            str(target),
            *identifiers,
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    assert completed.returncode == 0, completed.stderr
    assert (store.root / identifiers[1] / "result.json").read_bytes() == before
