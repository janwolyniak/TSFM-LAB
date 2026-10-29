"""Future records are strict, lossless, atomic, and independent of source/model files."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import pytest

import tsfm_lab.future_store as persistence
from tsfm_lab.application import load_dataset
from tsfm_lab.experiments import ExperimentStore, StoreError
from tsfm_lab.future import run_future
from tsfm_lab.future_config import future_config_from_mapping
from tsfm_lab.future_outputs import build_outputs
from tsfm_lab.future_store import FUTURE_ARTIFACTS, FutureStore
from tsfm_lab.future_validation import UNAVAILABLE, build_result, resolved_configuration
from tsfm_lab.models.baselines import LastValue

_NS = {"s": "http://www.w3.org/2000/svg"}


def make_record(
    tmp_path,
    *,
    values=None,
    context=4,
    horizon=3,
    frequency="1h",
    times=None,
    name="Future baseline",
):
    values = (
        np.arange(8.0) - 4 if values is None else np.asarray(values, dtype="float64")
    )
    times = (
        pd.date_range("2026-01-01", periods=len(values), freq=frequency, tz="UTC")
        if times is None
        else times
    )
    source = tmp_path / "source.csv"
    pd.DataFrame({"timestamp": times, "target": values}).to_csv(source, index=False)
    config = future_config_from_mapping(
        {
            "schema_version": 2,
            "mode": "future_forecast",
            "name": name,
            "model": {"source": "builtin", "adapter_id": "last_value"},
            "dataset": {
                "path": str(source),
                "timestamp_column": "timestamp",
                "target_column": "target",
                "frequency": frequency,
                "units": "original_units",
                "series_id": "NA",
            },
            "forecast": {"context_length": context, "horizon": horizon},
            "output": {"directory": str(tmp_path / "results")},
        },
        base_directory=tmp_path,
    )
    dataset = load_dataset(config.dataset)
    evaluated = run_future(
        dataset, LastValue(), config.forecast, runtime=config.runtime
    )
    store = FutureStore(config.output_directory)
    run_id = store.create(
        resolved_configuration(config, evaluated.request.origin), config.source_yaml
    )
    result, predictions, observations = build_result(
        config,
        dataset,
        evaluated,
        run_id,
        started_at="2026-10-29T10:00:00Z",
        completed_at="2026-10-29T10:00:01Z",
        elapsed_seconds=1.0,
    )
    return SimpleNamespace(
        store=store,
        config=config,
        source=source,
        run_id=run_id,
        result=result,
        predictions=predictions,
        observations=observations,
    )


@pytest.fixture
def record(tmp_path):
    return make_record(tmp_path)


def complete(record):
    return record.store.complete(
        record.run_id, record.result, record.predictions, record.observations
    )


def rehash(record, name, raw):
    manifest = record.store.get_manifest(record.run_id)
    directory = record.store.root / record.run_id
    (directory / name).write_bytes(raw)
    manifest["artifacts"][name] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))


def test_future_kind_exists_at_creation_and_completed_contract_has_exact_eight(record):
    manifest = record.store.get_manifest(record.run_id)
    assert manifest["schema_version"] == 3 and manifest["kind"] == "future_forecast"
    assert manifest["status"] == "pending"
    assert set(manifest["artifacts"]) == {"configuration.json", "input.yaml"}
    directory = record.store.root / record.run_id
    assert (directory / "input.yaml").read_text() == record.config.source_yaml
    assert (
        json.loads((directory / "configuration.json").read_text())["forecast"]["origin"]
        != "latest"
    )
    manifest = complete(record)
    assert (
        manifest["status"] == "completed"
        and set(manifest["artifacts"]) == FUTURE_ARTIFACTS
    )
    generic = ExperimentStore(record.store.workspace)
    result, predictions, observations = generic.load_future_result(record.run_id)
    assert result["evaluation"] == UNAVAILABLE
    assert "actual" not in predictions and len(predictions) == 3
    assert len(observations) == 4 and set(observations.series_id) == {"NA"}
    assert generic.list_runs()[0]["status"] == "completed"
    summary = generic.result_summary(record.run_id)
    assert summary["mode"] == "future_forecast" and summary["horizon"] == 3
    assert "metrics" not in summary and summary["evaluation"]["metrics"] is None
    assert summary["outputs"] == generic.output_paths(record.run_id)
    assert set(summary["outputs"]) == FUTURE_ARTIFACTS
    with pytest.raises(StoreError, match="Historical.*future"):
        generic.load_result(record.run_id)


def test_resolved_configuration_preserves_an_explicit_origin(record):
    origin = pd.Timestamp(record.result["request"]["origin"])
    explicit = replace(
        record.config, forecast=replace(record.config.forecast, origin=origin)
    )
    assert (
        resolved_configuration(explicit, origin)["forecast"]["origin"]
        == origin.isoformat()
    )
    with pytest.raises(StoreError, match="Explicit future origin"):
        resolved_configuration(explicit, origin + pd.Timedelta("1h"))


def test_lifecycle_refuses_evaluation_backward_moves_and_terminal_rewrites(record):
    generic = ExperimentStore(record.store.workspace)
    with pytest.raises(StoreError, match="no evaluation"):
        generic.set_status(record.run_id, "evaluating")
    generic.set_status(record.run_id, "preparing")
    with pytest.raises(StoreError, match="forward"):
        generic.set_status(record.run_id, "pending")
    generic.set_status(
        record.run_id,
        "failed",
        {"type": "FixtureFailure", "message": "stopped", "stage": "preparing"},
    )
    assert generic.list_runs()[0]["kind"] == "future_forecast"
    with pytest.raises(StoreError, match="Terminal"):
        generic.set_status(record.run_id, "running")
    with pytest.raises(StoreError, match="active"):
        complete(record)
    with pytest.raises(StoreError, match="completed"):
        generic.output_paths(record.run_id)


@pytest.mark.parametrize(
    "values", [[np.nextafter(0.1, 1), -0.0, 1.2345678901234567, -3.5], [-0.0] * 4]
)
def test_csv_roundtrip_preserves_every_float64_bit_including_negative_zero(
    tmp_path, values
):
    record = make_record(tmp_path, values=values, context=4, horizon=2)
    complete(record)
    _, predictions, observations = record.store.load_future_result(record.run_id)
    np.testing.assert_array_equal(
        observations.observed_value.to_numpy().view("uint64"),
        record.observations.observed_value.to_numpy().view("uint64"),
    )
    np.testing.assert_array_equal(
        predictions.prediction.to_numpy().view("uint64"),
        record.predictions.prediction.to_numpy().view("uint64"),
    )


def test_single_observation_and_forecast_need_no_holdout(tmp_path):
    record = make_record(tmp_path, values=[-7], context=1, horizon=1)
    complete(record)
    _, predictions, observations = record.store.load_future_result(record.run_id)
    assert observations.timestamp.iloc[-1] < predictions.target_timestamp.iloc[0]
    assert predictions.prediction.iloc[0] == -7


def test_future_outputs_safe_sampling_and_exact_unavailable_reason(tmp_path):
    record = make_record(
        tmp_path,
        values=np.arange(501.0) - 250,
        context=501,
        horizon=501,
        name='</title><script>alert("x")</script> & Łódź',
    )
    complete(record)
    paths = record.store.output_paths(record.run_id)
    root = ET.fromstring(Path(paths["forecast.svg"]).read_bytes())
    text = " ".join(root.itertext())
    assert not root.findall(".//s:script", _NS) and not root.findall(".//s:image", _NS)
    assert root.find("s:title", _NS).text == record.config.name
    assert len(root.findall(".//s:circle[@data-role='observations-point']", _NS)) == 400
    assert len(root.findall(".//s:circle[@data-role='predictions-point']", _NS)) == 400
    assert root.find(".//s:line[@data-role='origin-boundary']", _NS) is not None
    assert "Sampled display" in text and "future_actuals_absent" in text
    assert "original units: original_units" in text
    assert len(pd.read_csv(paths["observations.csv"])) == 501
    assert len(pd.read_csv(paths["predictions.csv"])) == 501
    assert (
        json.loads(Path(paths["provenance.json"]).read_text())["artifact_contract"]
        == "future_point_outputs_v1"
    )


def test_long_valid_nanosecond_span_does_not_overflow_chart_interpolation(tmp_path):
    times = pd.to_datetime([-5_000_000_000_000_000_000, 0], unit="ns", utc=True)
    record = make_record(
        tmp_path,
        values=[-1, 1],
        context=2,
        horizon=1,
        frequency="5000000000000000000ns",
        times=times,
    )
    complete(record)
    assert (
        record.store.load_future_result(record.run_id)[1].target_timestamp.iloc[0].value
        == 5_000_000_000_000_000_000
    )


@pytest.mark.parametrize("name", sorted(FUTURE_ARTIFACTS))
@pytest.mark.parametrize("update_checksum", [False, True])
def test_every_future_artifact_is_checked_against_independent_record_facts(
    record, name, update_checksum
):
    complete(record)
    directory = record.store.root / record.run_id
    if name.endswith(".csv"):
        raw = b"wrong,columns\n1,2\n"
    elif name.endswith(".json"):
        raw = b'{"invented": true}'
    elif name == "forecast.svg":
        raw = b'<svg xmlns="http://www.w3.org/2000/svg"><script/></svg>'
    else:
        raw = b"schema_version: 999\n"
    if update_checksum:
        rehash(record, name, raw)
    else:
        (directory / name).write_bytes(raw)
    with pytest.raises(StoreError):
        record.store.load_future_result(record.run_id)
    assert record.store.list_runs()[0]["status"] == "corrupt"


@pytest.mark.parametrize(
    "name", ["manifest.json", "configuration.json", "result.json", "evaluation.json"]
)
def test_duplicate_future_json_fields_are_not_accepted_after_rehash(record, name):
    complete(record)
    directory = record.store.root / record.run_id
    raw = (directory / name).read_bytes()
    duplicate = (
        b'{"schema_version": 3,'
        if name == "manifest.json"
        else b'{"schema_version": 2,'
    )
    raw = duplicate + raw.lstrip()[1:]
    if name == "manifest.json":
        (directory / name).write_bytes(raw)
    else:
        rehash(record, name, raw)
    with pytest.raises(StoreError, match="Duplicate"):
        record.store.load_future_result(record.run_id)


@pytest.mark.parametrize(
    "fault", ["version", "kind", "unknown", "missing", "escape", "size_bool"]
)
def test_future_manifest_version_kind_and_allowlist_remain_strict(record, fault):
    complete(record)
    manifest = record.store.get_manifest(record.run_id)
    if fault == "version":
        manifest["schema_version"] = 4
    elif fault == "kind":
        manifest["kind"] = "historical_evaluation"
    elif fault == "unknown":
        manifest["extra"] = True
    elif fault == "missing":
        del manifest["artifacts"]["observations.csv"]
    elif fault == "escape":
        manifest["artifacts"]["../outside"] = {"size": 1, "sha256": "a" * 64}
    else:
        manifest["artifacts"]["predictions.csv"]["size"] = True
    (record.store.root / record.run_id / "manifest.json").write_text(
        json.dumps(manifest)
    )
    with pytest.raises(StoreError):
        record.store.get_manifest(record.run_id)


@pytest.mark.parametrize(
    "fault",
    [
        "actual",
        "order",
        "extra",
        "bool_lead",
        "wrong_lead",
        "target",
        "origin",
        "context",
        "nonfinite",
        "request_id",
        "statistic",
        "metrics",
        "count_bool",
        "configuration",
    ],
)
def test_inconsistent_future_metadata_and_rows_never_complete(record, fault):
    result, predictions, observations = (
        copy.deepcopy(record.result),
        record.predictions.copy(),
        record.observations.copy(),
    )
    if fault == "actual":
        predictions["actual"] = 0
    elif fault == "order":
        predictions = predictions[list(reversed(predictions.columns))]
    elif fault == "extra":
        predictions = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    elif fault == "bool_lead":
        predictions["lead"] = True
    elif fault == "wrong_lead":
        predictions.loc[0, "lead"] = 2
    elif fault == "target":
        predictions["target_timestamp"] += pd.Timedelta("1h")
    elif fault == "origin":
        predictions["origin"] += pd.Timedelta("1h")
    elif fault == "context":
        observations.loc[0, "observed_value"] += 1
    elif fault == "nonfinite":
        predictions.loc[0, "prediction"] = np.inf
    elif fault == "request_id":
        predictions["request_id"] = "a" * 64
    elif fault == "statistic":
        result["output"]["point_statistic"] = "mean"
        predictions["point_statistic"] = "mean"
    elif fault == "metrics":
        result["evaluation"]["metrics"] = {"mae": 0}
    elif fault == "count_bool":
        result["counts"]["forecast_requests"] = True
    else:
        result["configuration"]["name"] = "different"
    with pytest.raises(StoreError):
        record.store.complete(record.run_id, result, predictions, observations)
    assert record.store.get_manifest(record.run_id)["status"] == "pending"


@pytest.mark.parametrize(
    "name",
    [
        "result.json",
        "predictions.csv",
        "observations.csv",
        "evaluation.json",
        "forecast.svg",
        "provenance.json",
    ],
)
@pytest.mark.parametrize("cancel", [False, True])
def test_failure_or_cancellation_at_every_output_write_remains_identifiable(
    record, monkeypatch, name, cancel
):
    original = persistence._atomic
    cancelled = False

    def atomic(path, content):
        nonlocal cancelled
        if path.name == name and not cancel:
            raise OSError("injected future output failure")
        original(path, content)
        if path.name == name:
            cancelled = True

    monkeypatch.setattr(persistence, "_atomic", atomic)
    with pytest.raises(KeyboardInterrupt if cancel else OSError):
        record.store.complete(
            record.run_id,
            record.result,
            record.predictions,
            record.observations,
            cancel_check=lambda: cancelled,
        )
    manifest = record.store.get_manifest(record.run_id)
    assert manifest["schema_version"] == 3 and manifest["kind"] == "future_forecast"
    assert manifest["status"] == "saving"
    assert set(manifest["artifacts"]) == {"configuration.json", "input.yaml"}
    with pytest.raises(StoreError, match="completed"):
        record.store.load_future_result(record.run_id)
    assert not list((record.store.root / record.run_id).glob("*.tmp"))


def test_cancellation_at_final_guard_does_not_publish_completion(record, monkeypatch):
    original = record.store._future_result
    cancelled = False

    def checked(*args):
        nonlocal cancelled
        result = original(*args)
        cancelled = True
        return result

    monkeypatch.setattr(record.store, "_future_result", checked)
    with pytest.raises(KeyboardInterrupt):
        record.store.complete(
            record.run_id,
            record.result,
            record.predictions,
            record.observations,
            cancel_check=lambda: cancelled,
        )
    assert record.store.get_manifest(record.run_id)["status"] == "saving"


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed_used", True),
        ("deterministic", False),
        ("torch_compile", True),
        ("forecast_options", {}),
    ],
)
def test_future_runtime_metadata_cannot_claim_unsupported_execution(
    record, field, value
):
    result = copy.deepcopy(record.result)
    result["provenance"]["runtime"][field] = value
    with pytest.raises(StoreError, match="runtime"):
        record.store.complete(
            record.run_id, result, record.predictions, record.observations
        )
    assert record.store.get_manifest(record.run_id)["status"] == "pending"


def test_saved_future_reload_does_not_need_source_or_optional_runtimes(
    record, tmp_path
):
    complete(record)
    record.source.unlink()
    program = """
import importlib.abc
import sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'torch','chronos','transformers','timesfm','tsfm_public','huggingface_hub','keyring','httpx','textual'} or fullname.startswith('tsfm_lab.assistant'):
            raise AssertionError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, Block())
from tsfm_lab.experiments import ExperimentStore
store = ExperimentStore(Path(sys.argv[1]))
result, predictions, observations = store.load_future_result(sys.argv[2])
assert result['evaluation']['metrics'] is None
assert len(predictions) == 3 and len(observations) == 4
assert len(store.output_paths(sys.argv[2])) == 8
"""
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-c", program, str(record.store.workspace), record.run_id],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(root / "src")),
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_future_chart_generation_is_deterministic_and_does_not_mutate_rows(record):
    original_result = copy.deepcopy(record.result)
    original_predictions, original_observations = (
        record.predictions.copy(),
        record.observations.copy(),
    )
    assert build_outputs(
        record.result, record.predictions, record.observations
    ) == build_outputs(record.result, record.predictions, record.observations)
    assert record.result == original_result
    pd.testing.assert_frame_equal(record.predictions, original_predictions)
    pd.testing.assert_frame_equal(record.observations, original_observations)
