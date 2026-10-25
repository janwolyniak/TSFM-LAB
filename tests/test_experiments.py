"""Persistence is atomic, inspectable after failure, and validates artifact identity."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd
import pytest

import tsfm_lab.experiments as persistence
from tsfm_lab.contracts import Dataset, EvaluationSpec
from tsfm_lab.evaluation.metrics import score_with_scale
from tsfm_lab.evaluation.runner import run_experiment
from tsfm_lab.experiments import ExperimentStore, StoreError
from tsfm_lab.models.baselines import LastValue


@pytest.fixture
def payload():
    times = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    dataset = Dataset(times, np.arange(1.0, 9.0), "1h", series_id="NA")
    result = run_experiment(
        dataset, LastValue(), EvaluationSpec(3, 2, times[4], times[-1])
    )
    return result.to_dict(), result.predictions


@pytest.fixture
def store(tmp_path):
    return ExperimentStore(tmp_path / "workspace")


def create(store):
    return store.create(
        "first experiment", {"dataset": "../data.csv"}, "# original\nname: naïve\n"
    )


def rewrite_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_constructor_and_empty_listing_do_not_create_directories(store):
    assert not store.workspace.exists()
    assert store.list_runs() == []
    assert not store.workspace.exists()


def test_create_preserves_input_and_allocates_distinct_uuid_runs(store):
    first, second = create(store), create(store)
    assert first != second and str(UUID(first)) == first
    directory = store.root / first
    assert (
        directory / "input.yaml"
    ).read_bytes() == "# original\nname: naïve\n".encode()
    assert json.loads((directory / "configuration.json").read_text()) == {
        "dataset": "../data.csv"
    }
    manifest = store.get_manifest(first)
    assert manifest["status"] == "pending"
    assert set(manifest["artifacts"]) == {"configuration.json", "input.yaml"}
    assert [run["run_id"] for run in store.list_runs()] == [second, first]


def test_states_move_forward_and_terminal_records_cannot_change(store):
    run_id = create(store)
    for status in ("preparing", "loading", "running", "evaluating", "saving"):
        assert store.set_status(run_id, status)["status"] == status
    with pytest.raises(ValueError, match="forward"):
        store.set_status(run_id, "running")
    with pytest.raises(ValueError, match="complete"):
        store.set_status(run_id, "completed")
    assert (
        store.set_status(run_id, "failed", {"message": "stopped"})["error"]["message"]
        == "stopped"
    )
    with pytest.raises(ValueError, match="Terminal"):
        store.set_status(run_id, "cancelled")


def test_completed_run_round_trips_values_timestamps_and_string_identities(
    store, payload
):
    result, frame = payload
    frame = frame.copy()
    frame["adapter_id"] = "001"
    result["configuration"]["model"]["adapter_id"] = "001"
    frame["request_id"] = frame.origin.map(
        {origin: f"00{i}" for i, origin in enumerate(frame.origin.unique())}
    )
    frame["prediction"] = [np.nextafter(0.1, 1), -0.0, 1.2345678901234567, -3.5]
    saved = result["metrics"]
    result["metrics"] = asdict(
        score_with_scale(
            frame.actual.to_numpy().reshape(-1, 2),
            frame.prediction.to_numpy().reshape(-1, 2),
            scale=saved["scale"],
            scale_reason=saved["scale_reason"],
            scale_count=saved["scale_count"],
        )
    )
    run_id = create(store)
    completed = store.complete(run_id, result, frame)
    assert completed["status"] == "completed"
    loaded, predictions = store.load_result(run_id)
    assert "predictions" not in loaded
    assert loaded["metrics"] == json.loads(json.dumps(result["metrics"]))
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), predictions)
    assert np.signbit(predictions.prediction.iloc[1])
    assert "NA" in set(predictions.series_id)
    assert set(predictions.adapter_id) == {"001"}
    assert store.list_runs()[0]["status"] == "completed"
    with pytest.raises(ValueError, match="Terminal"):
        store.complete(run_id, result, frame)


def test_completed_manifest_is_written_after_verified_outputs(
    store, payload, monkeypatch
):
    run_id = create(store)
    events = []
    original = persistence.os.replace

    def recording(source, destination):
        destination = Path(destination)
        if destination.name == "manifest.json":
            events.append(json.loads(Path(source).read_text())["status"])
        else:
            events.append(destination.name)
        original(source, destination)

    monkeypatch.setattr(persistence.os, "replace", recording)
    store.complete(run_id, *payload)
    assert events == [
        "saving",
        "result.json",
        "metrics.json",
        "predictions.csv",
        "completed",
    ]
    assert not list((store.root / run_id).glob("*.tmp"))


def test_interrupted_output_write_leaves_incomplete_history_and_no_temporary_files(
    store, payload, monkeypatch
):
    run_id = create(store)
    original = persistence.os.replace

    def fail(source, destination):
        if Path(destination).name == "metrics.json":
            raise OSError("injected write failure")
        original(source, destination)

    monkeypatch.setattr(persistence.os, "replace", fail)
    with pytest.raises(OSError, match="injected"):
        store.complete(run_id, *payload)
    assert store.get_manifest(run_id)["status"] == "saving"
    assert store.list_runs()[0]["status"] == "saving"
    assert not list((store.root / run_id).glob("*.tmp"))
    with pytest.raises(StoreError, match="completed"):
        store.load_result(run_id)


@pytest.mark.parametrize(
    "artifact", ["predictions.csv", "result.json", "configuration.json", "input.yaml"]
)
def test_changed_artifact_is_rejected_and_corrupt_completed_run_remains_visible(
    store, payload, artifact
):
    run_id = create(store)
    store.complete(run_id, *payload)
    with (store.root / run_id / artifact).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(StoreError, match="checksum|size"):
        store.load_result(run_id)
    listing = store.list_runs()
    assert len(listing) == 1 and listing[0]["status"] == "corrupt"
    assert "checksum" in listing[0]["error"]["message"]


@pytest.mark.parametrize(
    "run_id", ["../outside", "/tmp", "", "123", "00000000000000000000000000000000"]
)
def test_invalid_run_identity_never_resolves_a_path(store, run_id):
    with pytest.raises(StoreError, match="canonical UUID"):
        store.get_manifest(run_id)
    assert not store.root.exists()


@pytest.mark.parametrize("artifact", ["../outside", "/tmp/outside", "unknown.json"])
def test_manifest_cannot_redirect_artifact_reads(store, artifact):
    run_id = create(store)
    path = store.root / run_id / "manifest.json"
    manifest = store.get_manifest(run_id)
    manifest["artifacts"][artifact] = {"size": 0, "sha256": "0" * 64}
    rewrite_json(path, manifest)
    with pytest.raises(StoreError, match="allowlist"):
        store.get_manifest(run_id)


def test_existing_run_uuid_is_never_overwritten(store, monkeypatch):
    run_id = create(store)
    before = (store.root / run_id / "manifest.json").read_bytes()
    monkeypatch.setattr(persistence, "uuid4", lambda: UUID(run_id))
    with pytest.raises(FileExistsError):
        create(store)
    assert (store.root / run_id / "manifest.json").read_bytes() == before


def test_symlinked_artifact_cannot_escape_workspace(store, payload, tmp_path):
    run_id = create(store)
    store.complete(run_id, *payload)
    artifact = store.root / run_id / "predictions.csv"
    outside = tmp_path / "outside.csv"
    outside.write_bytes(artifact.read_bytes())
    artifact.unlink()
    artifact.symlink_to(outside)
    with pytest.raises(StoreError, match="regular file"):
        store.load_result(run_id)
    assert store.list_runs()[0]["status"] == "corrupt"


def test_symlinked_run_or_experiments_directory_is_rejected(store, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    store.workspace.mkdir()
    store.root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(StoreError, match="symbolic links"):
        create(store)
    assert list(outside.iterdir()) == []


def test_invalid_or_missing_manifest_stays_visible_in_history(store):
    run_id = create(store)
    (store.root / run_id / "manifest.json").write_text("{broken")
    listing = store.list_runs()
    assert len(listing) == 1 and listing[0]["status"] == "corrupt"


def test_corrupt_completed_run_keeps_its_name_and_chronological_position(
    store, payload
):
    first, second = create(store), create(store)
    store.complete(second, *payload)
    (store.root / second / "metrics.json").write_text("{}")
    listing = store.list_runs()
    assert [record["run_id"] for record in listing] == [second, first]
    assert listing[0]["name"] == "first experiment"
    assert listing[0]["status"] == "corrupt"


@pytest.mark.parametrize(
    "fault", ["columns", "nonfinite", "lead", "duplicate", "time", "naive", "counts"]
)
def test_invalid_predictions_cannot_complete(store, payload, fault):
    result, frame = payload
    result = json.loads(json.dumps(result))
    frame = frame.copy()
    if fault == "columns":
        frame = frame.drop(columns="actual")
    elif fault == "nonfinite":
        frame.loc[0, "prediction"] = np.inf
    elif fault == "lead":
        frame["lead"] = frame.lead.astype(float) + 0.5
    elif fault == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif fault == "time":
        frame["target_timestamp"] = frame.origin
    elif fault == "naive":
        frame["origin"] = frame.origin.dt.tz_localize(None)
    else:
        result["counts"]["forecast_cases"] = 999
    run_id = create(store)
    with pytest.raises(StoreError):
        store.complete(run_id, result, frame)
    assert store.get_manifest(run_id)["status"] == "pending"


def test_readable_csv_with_recomputed_checksum_still_requires_valid_prediction_schema(
    store, payload
):
    run_id = create(store)
    store.complete(run_id, *payload)
    path = store.root / run_id
    raw = b"wrong,columns\n1,2\n"
    (path / "predictions.csv").write_bytes(raw)
    manifest = store.get_manifest(run_id)
    manifest["artifacts"]["predictions.csv"] = {
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    rewrite_json(path / "manifest.json", manifest)
    with pytest.raises(StoreError, match="columns"):
        store.load_result(run_id)


@pytest.mark.parametrize(
    "fault",
    [
        "adapter",
        "series",
        "reversed_targets",
        "shifted_targets",
        "origin",
        "statistic",
        "missing_window",
    ],
)
def test_prediction_identity_must_match_the_full_configured_grid(store, payload, fault):
    result, frame = payload
    result = json.loads(json.dumps(result))
    frame = frame.copy()
    if fault == "adapter":
        frame["adapter_id"] = "another_model"
    elif fault == "series":
        frame["series_id"] = "another_series"
    elif fault == "reversed_targets":
        for origin in frame.origin.unique():
            mask = frame.origin == origin
            frame.loc[mask, "target_timestamp"] = (
                frame.loc[mask, "target_timestamp"].iloc[::-1].to_numpy()
            )
    elif fault == "shifted_targets":
        frame["target_timestamp"] += pd.Timedelta("1D")
    elif fault == "origin":
        frame["origin"] += pd.Timedelta("1h")
        frame["target_timestamp"] += pd.Timedelta("1h")
    elif fault == "statistic":
        frame["point_statistic"] = "mean"
    else:
        frame = frame.loc[frame.origin == frame.origin.min()].copy()
        result["counts"].update(
            attempted_windows=1,
            successful_windows=1,
            forecast_cases=2,
            unique_target_timestamps=2,
        )
    run_id = create(store)
    with pytest.raises(StoreError):
        store.complete(run_id, result, frame)
    assert store.get_manifest(run_id)["status"] == "pending"


_BAD_METADATA = [
    (("schema_version",), True),
    (("configuration",), []),
    (("configuration", "model"), {}),
    (("configuration", "runtime", "batch_size"), True),
    (("configuration", "evaluation", "horizon"), 2.5),
    (("configuration", "evaluation", "metrics"), []),
    (("provenance",), []),
    (("provenance", "environment"), {}),
    (("provenance", "dataset", "canonical_sha256"), "x" * 64),
    (("provenance", "scale_history", "observations"), 99),
    (("counts",), []),
    (("counts", "forecast_cases"), True),
    (("metrics", "overall", "mae", "value"), -1),
    (("metrics", "overall", "mae", "value"), True),
    (("metrics", "overall", "mae", "value"), None),
    (("metrics", "overall", "mae", "reason"), "zero_scale"),
    (("metrics", "by_lead"), {}),
    (("metrics", "scale_reason"), "unknown"),
    (("timing_seconds",), []),
    (("timing_seconds", "load"), -1),
]


def change_metadata(result, path, value):
    for key in path[:-1]:
        result = result[key]
    result[path[-1]] = value


def replace_artifact(store, run_id, name, raw):
    directory = store.root / run_id
    (directory / name).write_bytes(raw)
    manifest = store.get_manifest(run_id)
    manifest["artifacts"][name] = {
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    rewrite_json(directory / "manifest.json", manifest)


@pytest.mark.parametrize("path,value", _BAD_METADATA)
@pytest.mark.parametrize("boundary", ["complete", "load"])
def test_malformed_result_metadata_always_raises_store_error(
    store, payload, path, value, boundary
):
    original, frame = payload
    result = json.loads(json.dumps(original))
    run_id = create(store)
    if boundary == "load":
        store.complete(run_id, original, frame)
    change_metadata(result, path, value)
    with pytest.raises(StoreError):
        if boundary == "complete":
            store.complete(run_id, result, frame)
        else:
            replace_artifact(store, run_id, "result.json", json.dumps(result).encode())
            replace_artifact(
                store, run_id, "metrics.json", json.dumps(result["metrics"]).encode()
            )
            store.load_result(run_id)


@pytest.mark.parametrize(
    "missing", ["configuration", "provenance", "metrics", "counts", "timing_seconds"]
)
def test_completed_results_require_all_core_metadata(store, payload, missing):
    result, frame = payload
    result = json.loads(json.dumps(result))
    del result[missing]
    with pytest.raises(StoreError):
        store.complete(create(store), result, frame)


@pytest.mark.parametrize("boundary", ["complete", "load"])
def test_stored_scores_must_agree_with_saved_predictions(store, payload, boundary):
    result, frame = payload
    changed = frame.copy()
    changed["prediction"] += 0.5
    run_id = create(store)
    if boundary == "complete":
        with pytest.raises(StoreError, match="metric value"):
            store.complete(run_id, result, changed)
    else:
        store.complete(run_id, result, frame)
        replace_artifact(
            store, run_id, "predictions.csv", changed.to_csv(index=False).encode()
        )
        with pytest.raises(StoreError, match="metric value"):
            store.load_result(run_id)
        assert store.list_runs()[0]["status"] == "corrupt"


def test_cancellation_between_artifact_writes_cannot_complete(
    store, payload, monkeypatch
):
    cancelled = False
    original = persistence._atomic

    def write_then_cancel(path, content):
        nonlocal cancelled
        original(path, content)
        if path.name == "result.json":
            cancelled = True

    monkeypatch.setattr(persistence, "_atomic", write_then_cancel)
    run_id = create(store)
    with pytest.raises(KeyboardInterrupt):
        store.complete(run_id, *payload, cancel_check=lambda: cancelled)
    assert store.get_manifest(run_id)["status"] == "saving"
    assert not (store.root / run_id / "metrics.json").exists()


def test_cancellation_at_final_completion_guard_leaves_saving_record(
    store, payload, monkeypatch
):
    cancelled = False
    original = store._result

    def checked_then_cancel(*args):
        nonlocal cancelled
        result = original(*args)
        cancelled = True
        return result

    monkeypatch.setattr(store, "_result", checked_then_cancel)
    run_id = create(store)
    with pytest.raises(KeyboardInterrupt):
        store.complete(run_id, *payload, cancel_check=lambda: cancelled)
    assert store.get_manifest(run_id)["status"] == "saving"
    with pytest.raises(StoreError, match="completed"):
        store.load_result(run_id)


def test_process_summary_does_not_parse_prediction_artifacts(
    store, payload, monkeypatch
):
    run_id = create(store)
    store.complete(run_id, *payload)
    monkeypatch.setattr(
        store,
        "load_result",
        lambda *args: pytest.fail("Prediction parsing on process-monitor path"),
    )
    summary = store.result_summary(run_id)
    assert summary["status"] == "completed"
    assert summary["metrics"] == payload[0]["metrics"]["overall"]
    assert summary["counts"] == payload[0]["counts"]
