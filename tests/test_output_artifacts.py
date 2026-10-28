"""Guided outputs remain safe, complete, and reloadable without inference."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import pytest

import tsfm_lab.experiments as persistence
import tsfm_lab.output_artifacts as outputs
from tsfm_lab.contracts import Dataset, EvaluationSpec
from tsfm_lab.evaluation.runner import run_experiment
from tsfm_lab.experiments import ExperimentStore, StoreError
from tsfm_lab.models.baselines import LastValue

_NAMES = {
    "configuration.json",
    "input.yaml",
    "result.json",
    "metrics.json",
    "predictions.csv",
}
_NS = {"s": "http://www.w3.org/2000/svg"}


@pytest.fixture
def payload():
    times = pd.date_range("2026-01-01", periods=16, freq="h", tz="UTC")
    dataset = Dataset(
        times, np.arange(16.0) - 8, "1h", target_name="temperature", units="degC"
    )
    evaluated = run_experiment(
        dataset, LastValue(), EvaluationSpec(4, 3, times[6], times[-1], stride=1)
    )
    result = evaluated.to_dict()
    result["name"] = "Original-unit evaluation"
    return result, evaluated.predictions


@pytest.fixture
def store(tmp_path):
    return ExperimentStore(tmp_path / "results")


def _create(store, result):
    return store.create(result["name"], result["configuration"], "# original input\n")


def _complete(store, payload, **kwargs):
    result, frame = payload
    run_id = _create(store, result)
    store.complete(run_id, result, frame, **kwargs)
    return run_id


def _replace_artifact(store, run_id, name, content):
    directory = store.root / run_id
    manifest = store.get_manifest(run_id)
    (directory / name).write_bytes(content)
    manifest["artifacts"][name] = {
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))


def test_deterministic_chart_preserves_units_timestamps_and_window_trajectories(
    payload,
):
    result, frame = payload
    original_result, original_frame = copy.deepcopy(result), frame.copy(deep=True)
    built = outputs.build_output_artifacts(result, frame)
    assert set(built) == {"forecast.svg", "provenance.json"}
    assert built == outputs.build_output_artifacts(result, frame)
    assert result == original_result
    pd.testing.assert_frame_equal(frame, original_frame)
    root = ET.fromstring(built["forecast.svg"])
    actuals = root.findall(".//s:circle[@data-role='observed-point']", _NS)
    assert len(actuals) == frame.target_timestamp.nunique()
    assert len({point.attrib["data-timestamp"] for point in actuals}) == len(actuals)
    windows = root.findall(".//s:g[@data-role='forecast-window']", _NS)
    assert len(windows) == frame.origin.nunique()
    text = " ".join(root.itertext())
    for expected in (
        "original units: degC",
        "UTC",
        "Evaluation start (inclusive)",
        "Evaluation end (exclusive)",
        "Requested task: median",
        "saved statistic: deterministic",
        "historical context prefix is not saved",
        "predictions.csv",
    ):
        assert expected in text
    assert any(float(point.attrib["cy"]) > 320 for point in actuals)
    provenance = json.loads(built["provenance.json"])
    for name in ("configuration", "provenance", "metrics", "counts", "timing_seconds"):
        assert provenance[name] == json.loads(json.dumps(result[name]))
    assert provenance["chart"]["history_prefix_available"] is False
    assert provenance["chart"]["sampling"]["applied"] is False


def test_svg_treats_unusual_names_units_and_control_characters_as_data(payload):
    result, frame = copy.deepcopy(payload[0]), payload[1]
    result["name"] = '</title><script>alert("title")</script> & Łódź'
    result["configuration"]["dataset"][
        "units"
    ] = '</text><image href="https://example.com/pixel"/> & µg/m³\x00'
    built = outputs.build_output_artifacts(result, frame)
    root = ET.fromstring(built["forecast.svg"])
    assert root.find("s:title", _NS).text == result["name"]
    assert not root.findall(".//s:script", _NS)
    assert not root.findall(".//s:image", _NS)
    assert "href" not in {key for node in root.iter() for key in node.attrib}
    assert b"&lt;script&gt;" in built["forecast.svg"]
    assert b"\x00" not in built["forecast.svg"]
    assert (
        json.loads(built["provenance.json"])["configuration"]["dataset"]["units"]
        == result["configuration"]["dataset"]["units"]
    )


def test_missing_metric_reason_and_native_task_are_visible():
    times = pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC")
    evaluated = run_experiment(
        Dataset(times, np.full(12, -5.0), "1h", units="degC"),
        LastValue(),
        EvaluationSpec(4, 2, times[6], times[-1], point_statistic="native_point"),
    )
    built = outputs.build_output_artifacts(evaluated.to_dict(), evaluated.predictions)
    text = " ".join(ET.fromstring(built["forecast.svg"]).itertext())
    assert "MASE unavailable: zero_scale" in text
    assert "Requested task: native_point · saved statistic: deterministic" in text
    assert "-5" in text
    assert json.loads(built["provenance.json"])["metrics"]["overall"]["mase"] == {
        "value": None,
        "reason": "zero_scale",
    }


def test_unrequested_metrics_have_reasons_without_invented_values():
    times = pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC")
    evaluated = run_experiment(
        Dataset(times, np.arange(12.0), "1h", units="original_units"),
        LastValue(),
        EvaluationSpec(4, 2, times[6], times[-1], metrics=("mae",)),
    )
    built = outputs.build_output_artifacts(evaluated.to_dict(), evaluated.predictions)
    text = " ".join(ET.fromstring(built["forecast.svg"]).itertext())
    assert "RMSE unavailable: not_requested" in text
    assert "MASE unavailable: not_requested" in text
    assert set(json.loads(built["provenance.json"])["metrics"]["overall"]) == {"mae"}


def test_sampling_is_bounded_disclosed_and_keeps_full_prediction_csv(store):
    times = pd.date_range("2026-01-01", periods=500, freq="h", tz="UTC")
    evaluated = run_experiment(
        Dataset(times, np.arange(500.0) - 250, "1h", units="original_units"),
        LastValue(),
        EvaluationSpec(8, 129, times[20], times[-1] + pd.Timedelta("1h"), stride=8),
    )
    result = evaluated.to_dict()
    result["name"] = "Sample disclosure"
    run_id = _complete(store, (result, evaluated.predictions), include_outputs=True)
    paths = store.output_paths(run_id)
    root = ET.fromstring(Path(paths["forecast.svg"]).read_bytes())
    assert len(root.findall(".//s:circle[@data-role='observed-point']", _NS)) == 400
    windows = root.findall(".//s:g[@data-role='forecast-window']", _NS)
    assert len(windows) == 32
    assert all(
        len(window.find("s:polyline", _NS).attrib["points"].split()) == 128
        for window in windows
    )
    text = " ".join(root.itertext())
    assert (
        "Sampled display" in text and "Full forecasts remain in predictions.csv" in text
    )
    sampling = json.loads(Path(paths["provenance.json"]).read_text())["chart"][
        "sampling"
    ]
    assert sampling["applied"] is True
    assert sampling["forecast_cases_total"] == len(evaluated.predictions)
    assert len(pd.read_csv(paths["predictions.csv"])) == len(evaluated.predictions)
    points = root.findall(".//s:circle[@data-role='observed-point']", _NS)
    assert (
        points[0].attrib["data-timestamp"]
        == evaluated.predictions.target_timestamp.min().isoformat()
    )
    assert (
        points[-1].attrib["data-timestamp"]
        == evaluated.predictions.target_timestamp.max().isoformat()
    )


def test_new_provenance_does_not_copy_credential_fields(payload):
    result = copy.deepcopy(payload[0])
    result["provenance"]["OPENAI_API_KEY"] = "fake-secret-value"
    result["provenance"]["environment"]["AWS_SECRET_ACCESS_KEY"] = "fake-secret-value"
    result["provenance"]["environment"]["HF_TOKEN"] = "fake-secret-value"
    result["configuration"]["runtime"]["Authorization"] = "fake-secret-value"
    result["configuration"]["model"]["options"]["nested"] = {
        "refresh-token": "fake-secret-value",
        "normalization": "recorded_choice",
    }
    document = outputs.build_output_artifacts(result, payload[1])["provenance.json"]
    assert b"fake-secret-value" not in document
    assert json.loads(document)["configuration"]["model"]["options"]["nested"] == {
        "normalization": "recorded_choice"
    }


def test_subminute_ticks_show_seconds_and_normalize_boundaries_to_utc():
    times = pd.date_range("2026-01-01", periods=12, freq="500ms", tz="UTC")
    evaluated = run_experiment(
        Dataset(times, np.arange(12.0), "500ms", units="original_units"),
        LastValue(),
        EvaluationSpec(3, 2, times[4], times[-1]),
    )
    result = evaluated.to_dict()
    for key in ("evaluation_start", "evaluation_end"):
        result["configuration"]["evaluation"][key] = (
            pd.Timestamp(result["configuration"]["evaluation"][key])
            .tz_convert("Europe/Warsaw")
            .isoformat()
        )
    built = outputs.build_output_artifacts(result, evaluated.predictions)
    text = " ".join(ET.fromstring(built["forecast.svg"]).itertext())
    assert "00:00:02" in text and "00:00:05.500000" in text
    assert "Evaluation: 2026-01-01T00:00:02+00:00" in text


@pytest.mark.parametrize("include_outputs,version", [(False, 1), (True, 2)])
def test_manifest_versions_keep_exact_artifact_allowlists_and_checked_paths(
    store, payload, include_outputs, version
):
    run_id = _complete(store, payload, include_outputs=include_outputs)
    manifest = store.get_manifest(run_id)
    expected = _NAMES | (
        {"forecast.svg", "provenance.json"} if include_outputs else set()
    )
    assert manifest["schema_version"] == version
    assert manifest["status"] == "completed"
    assert set(manifest["artifacts"]) == expected
    paths = store.output_paths(run_id)
    assert set(paths) == expected
    for name, path in paths.items():
        path = Path(path)
        assert path.is_absolute() and path == store.root / run_id / name
        raw = path.read_bytes()
        assert manifest["artifacts"][name] == {
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    _, frame = store.load_result(run_id)
    assert len(frame) == len(payload[1])


@pytest.mark.parametrize("artifact", ["forecast.svg", "provenance.json"])
@pytest.mark.parametrize("rehash", [False, True])
def test_changed_outputs_are_rejected_even_with_recomputed_checksum(
    store, payload, artifact, rehash
):
    run_id = _complete(store, payload, include_outputs=True)
    if artifact == "forecast.svg":
        content = (
            b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
        )
    else:
        content = b'{"schema_version": 1, "configuration": {"invented": true}}'
    if rehash:
        _replace_artifact(store, run_id, artifact, content)
    else:
        (store.root / run_id / artifact).write_bytes(content)
    with pytest.raises(StoreError, match="checksum|size|disagree"):
        store.load_result(run_id)
    with pytest.raises(StoreError):
        store.output_paths(run_id)
    assert store.list_runs()[0]["status"] == "corrupt"


@pytest.mark.parametrize(
    "change", ["missing_output", "v1_with_outputs", "unknown_file", "pending_v2"]
)
def test_versioned_manifest_allowlists_remain_strict(store, payload, change):
    run_id = _complete(store, payload, include_outputs=True)
    manifest = store.get_manifest(run_id)
    if change == "missing_output":
        del manifest["artifacts"]["forecast.svg"]
    elif change == "v1_with_outputs":
        manifest["schema_version"] = 1
    elif change == "unknown_file":
        manifest["artifacts"]["../outside"] = {"size": 0, "sha256": "0" * 64}
    else:
        manifest["status"] = "saving"
    (store.root / run_id / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(StoreError, match="allowlist|completed publication"):
        store.get_manifest(run_id)


def test_builder_failure_does_not_publish_completed_or_v2_record(
    store, payload, monkeypatch
):
    def fail(*args):
        raise RuntimeError("injected output generation failure")

    monkeypatch.setattr(outputs, "build_output_artifacts", fail)
    run_id = _create(store, payload[0])
    with pytest.raises(RuntimeError, match="injected"):
        store.complete(run_id, *payload, include_outputs=True)
    assert store.get_manifest(run_id)["status"] == "saving"
    assert store.get_manifest(run_id)["schema_version"] == 1
    with pytest.raises(StoreError, match="completed"):
        store.output_paths(run_id)


def test_invalid_builder_outputs_fail_before_publication(store, payload, monkeypatch):
    monkeypatch.setattr(
        outputs, "build_output_artifacts", lambda *args: {"forecast.svg": b"invalid"}
    )
    run_id = _create(store, payload[0])
    with pytest.raises(StoreError, match="require"):
        store.complete(run_id, *payload, include_outputs=True)
    assert store.get_manifest(run_id)["status"] == "saving"


@pytest.mark.parametrize("cancel", [False, True])
def test_output_write_failure_and_cancellation_never_complete(
    store, payload, monkeypatch, cancel
):
    cancelled = False
    original = persistence._atomic

    def writing(path, content):
        nonlocal cancelled
        if path.name == "forecast.svg" and not cancel:
            raise OSError("injected chart write failure")
        original(path, content)
        if path.name == "forecast.svg":
            cancelled = True

    monkeypatch.setattr(persistence, "_atomic", writing)
    run_id = _create(store, payload[0])
    with pytest.raises(KeyboardInterrupt if cancel else OSError):
        store.complete(
            run_id, *payload, include_outputs=True, cancel_check=lambda: cancelled
        )
    manifest = store.get_manifest(run_id)
    assert manifest["schema_version"] == 1 and manifest["status"] == "saving"
    assert set(manifest["artifacts"]) == {"configuration.json", "input.yaml"}
    assert not list((store.root / run_id).glob("*.tmp"))
    with pytest.raises(StoreError, match="completed"):
        store.load_result(run_id)


def test_cancellation_after_all_output_checks_still_blocks_completion(
    store, payload, monkeypatch
):
    cancelled = False
    original = store._result

    def checked(*args):
        nonlocal cancelled
        result = original(*args)
        cancelled = True
        return result

    monkeypatch.setattr(store, "_result", checked)
    run_id = _create(store, payload[0])
    with pytest.raises(KeyboardInterrupt):
        store.complete(
            run_id, *payload, include_outputs=True, cancel_check=lambda: cancelled
        )
    assert store.get_manifest(run_id)["status"] == "saving"


def test_reload_needs_no_source_runtime_weights_or_assistant_packages(store, payload):
    result = copy.deepcopy(payload[0])
    result["provenance"]["dataset"]["source_path"] = "/unavailable/original-data.csv"
    run_id = _complete(store, (result, payload[1]), include_outputs=True)
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
result, frame = store.load_result(sys.argv[2])
paths = store.output_paths(sys.argv[2])
assert len(frame) > 0 and Path(paths['forecast.svg']).is_file()
"""
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-c", program, str(store.workspace), run_id],
        env=dict(os.environ, PYTHONPATH=str(root / "src")),
        cwd=store.workspace,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_legacy_release_records_keep_reloading_and_truthfully_omit_new_outputs():
    fixture = Path(__file__).parent / "fixtures/release_legacy_v1/workspace"
    store = ExperimentStore(fixture)
    records = store.list_runs()
    assert records and all(record["status"] == "completed" for record in records)
    for record in records:
        paths = store.output_paths(record["run_id"])
        assert set(paths) == _NAMES
        assert "forecast.svg" not in paths and "provenance.json" not in paths
