"""TOTO 2 uses the existing offline setup, managed lifecycle and saved contracts."""

import copy
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_checkpoint_preparation import FakeHub
from test_future_setup import Questions
from toto2_helpers import (
    CONFIG,
    REPOSITORY,
    REVISION,
    install_runtime,
    mapping,
    write_checkpoint,
)
from toto2_helpers import fake_toto as shared_toto

from tsfm_lab.application import RunExecutionError, run_config
from tsfm_lab.artifact_validation import validate_predictions
from tsfm_lab.comparison import compare_runs
from tsfm_lab.config import config_from_mapping
from tsfm_lab.contracts import Dataset
from tsfm_lab.execution import ManagedRun, _worker_main
from tsfm_lab.experiments import ExperimentStore, StoreError
from tsfm_lab.future import preflight_context
from tsfm_lab.future_application import run_future_config
from tsfm_lab.future_config import FutureForecastSpec, future_config_from_mapping
from tsfm_lab.future_store import FutureStore
from tsfm_lab.future_validation import validate_future
from tsfm_lab.guided import execute_guided, review_guided
from tsfm_lab.models import checkpoints, toto2, toto2_profile
from tsfm_lab.models.catalog import catalog_info
from tsfm_lab.repl.wizard import build_experiment

fake_toto = shared_toto


def test_catalog_has_explicit_support_limits_and_no_inherited_inference():
    entry = catalog_info(REPOSITORY)
    assert entry["reference_revision"] == REVISION
    assert entry["project_inference_status"] == "recorded"
    assert entry["capabilities"]["min_context"] == 32
    assert entry["capabilities"]["max_context"] == 2048
    assert entry["capabilities"]["max_horizon"] == 128
    assert entry["point_statistic"] == "median"
    assert "Python 3.12 or 3.13" in " ".join(entry["runtime_notices"])
    assert "not upstream architectural limits" in " ".join(entry["runtime_notices"])


@pytest.mark.parametrize("future", [False, True])
def test_checked_historical_future_reload_export_and_assistant_facts(
    tmp_path, fake_toto, future
):
    write_checkpoint(tmp_path / "weights")
    raw = mapping(tmp_path, future=future)
    config = (
        future_config_from_mapping(raw, base_directory=tmp_path)
        if future
        else config_from_mapping(raw, base_directory=tmp_path)
    )
    completed = (
        run_future_config(config)
        if future
        else run_config(config, include_outputs=True)
    )
    store = ExperimentStore(tmp_path / "results")
    if future:
        result, predictions, _ = FutureStore(tmp_path / "results").load_future_result(
            completed["run_id"]
        )
        assert result["evaluation"]["status"] == "unavailable"
        assert len(predictions) == 8
    else:
        result, predictions = store.load_result(completed["run_id"])
        assert len(predictions) == 48
        baseline = copy.deepcopy(raw)
        baseline["model"] = {"source": "builtin", "adapter_id": "last_value"}
        other = run_config(config_from_mapping(baseline, base_directory=tmp_path))
        assert compare_runs(
            [completed["run_id"], other["run_id"]], tmp_path / "results"
        )["compatible"]
        from tsfm_lab.assistant.evidence import experiment_evidence

        facts = experiment_evidence(store, completed["run_id"])
        assert (
            facts["data"]["configuration"]["runtime"]["forecast_options"]
            == toto2_profile.FORECAST_OPTIONS
        )
    assert set(predictions.point_statistic) == {"median"}
    if not future:
        assert (
            result["provenance"]["project_verification"]["project_inference_status"]
            == "not_recorded"
        )
    else:
        assert (
            result["provenance"]["model"]["options"]["checkpoint"]["architecture"]
            == "Toto2Model"
        )
    shutil.rmtree(tmp_path / "weights")
    # Simulated 3.11 reload must accept a saved 3.12 execution without model imports.
    program = """
import importlib.abc,sys
from types import SimpleNamespace
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','toto2','safetensors','gluonts','lightning','dd_unit_scaling','huggingface_hub'}:
            raise AssertionError('Unexpected optional import '+fullname)
sys.meta_path.insert(0,Block())
from tsfm_lab.models import toto2_profile
toto2_profile.sys=SimpleNamespace(version_info=(3,11))
from tsfm_lab.exports import export_predictions
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.run_modes import read_completed
workspace,run_id,output=sys.argv[1:]
result,frame=read_completed(ExperimentStore(workspace),run_id)
assert len(frame)>0
assert export_predictions(run_id,workspace,output)['rows']==len(frame)
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(tmp_path / "results"),
            completed["run_id"],
            str(tmp_path / "export.csv"),
        ],
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
    )


@pytest.mark.parametrize("future", [False, True])
@pytest.mark.parametrize("mutation", ["python", "options", "minimum"])
def test_saved_toto_runtime_semantics_cannot_be_forged(
    tmp_path, fake_toto, future, mutation
):
    write_checkpoint(tmp_path / "weights")
    raw = mapping(tmp_path, future=future)
    if future:
        done = run_future_config(
            future_config_from_mapping(raw, base_directory=tmp_path)
        )
        result, frame, observations = FutureStore(
            tmp_path / "results"
        ).load_future_result(done["run_id"])
        model, runtime = result["provenance"]["model"], result["provenance"]["runtime"]
    else:
        done = run_config(config_from_mapping(raw, base_directory=tmp_path))
        result, frame = ExperimentStore(tmp_path / "results").load_result(
            done["run_id"]
        )
        model, runtime = (
            result["configuration"]["model"],
            result["configuration"]["runtime"],
        )
    if mutation == "python":
        result["provenance"]["environment"]["python"] = "3.11.13"
    elif mutation == "options":
        runtime["forecast_options"]["quantile_real_cap_k"] = 1e4
    else:
        model["capabilities"]["min_context"] = 1
    with pytest.raises(StoreError):
        (
            validate_future(result, frame, observations)
            if future
            else validate_predictions(frame, result)
        )


def test_historical_context_only_scaling_and_future_values_do_not_enter_native_input(
    tmp_path, fake_toto
):
    write_checkpoint(tmp_path / "weights")
    raw = mapping(tmp_path)
    raw["evaluation"].update(
        evaluation_start="2026-01-02T08:00Z", evaluation_end="2026-01-02T16:00Z"
    )
    config = config_from_mapping(raw, base_directory=tmp_path)
    first = run_config(config)
    original = fake_toto.targets[0].copy()
    data = pd.read_csv(raw["dataset"]["path"])
    data.loc[32:, "demand"] = 1e12
    data.to_csv(raw["dataset"]["path"], index=False)
    second = run_config(config)
    np.testing.assert_array_equal(original, fake_toto.targets[1])
    store = ExperimentStore(tmp_path / "results")
    np.testing.assert_array_equal(
        store.load_result(first["run_id"])[1].prediction,
        store.load_result(second["run_id"])[1].prediction,
    )


@pytest.mark.parametrize("future", [False, True])
def test_cancellation_or_native_failure_never_publishes_complete_result(
    tmp_path, fake_toto, future
):
    write_checkpoint(tmp_path / "weights")
    raw = mapping(tmp_path, future=future)
    config = (
        future_config_from_mapping(raw, base_directory=tmp_path)
        if future
        else config_from_mapping(raw, base_directory=tmp_path)
    )
    run = run_future_config if future else run_config
    with pytest.raises(KeyboardInterrupt):
        run(config, should_cancel=lambda: bool(fake_toto.forecasts))
    fake_toto.forecasts.clear()
    fake_toto.forecast_error = RuntimeError("synthetic native failure")
    with pytest.raises(RunExecutionError):
        run(config)
    records = ExperimentStore(tmp_path / "results").list_runs()
    assert {record["status"] for record in records} == {"cancelled", "failed"}
    assert not any(
        (
            tmp_path / "results" / "experiments" / record["run_id"] / "predictions.csv"
        ).exists()
        for record in records
    )


def test_wizard_reprompts_patch_context_and_preserves_numerical_identity(
    tmp_path, fake_toto, monkeypatch
):
    checkpoint = write_checkpoint(tmp_path / "weights")
    raw = mapping(tmp_path, local=False)
    monkeypatch.setattr(
        "tsfm_lab.repl.wizard.build_model", lambda *a, **kw: toto2.Toto2(checkpoint)
    )
    monkeypatch.setattr(
        "tsfm_lab.tui.forms.build_model", lambda *a, **kw: toto2.Toto2(checkpoint)
    )
    questions = Questions(
        {
            "Context length": ["33", "32"],
            "Forecast mode": "Future forecasting",
            "Run experiment?": "yes",
        }
    )
    emitted = []
    config = build_experiment(
        questions,
        emitted.append,
        workspace=tmp_path / "results",
        demo_path=Path(raw["dataset"]["path"]),
        adapter="toto_2",
    )
    assert config is not None, "\n".join(emitted)
    assert config.model["revision"] == REVISION
    assert config.forecast.context_length == 32
    assert "multiple of 32" in "\n".join(emitted)
    reloaded = future_config_from_mapping(
        config.resolved_dict(), base_directory=config.source_path.parent
    )
    assert reloaded.resolved_dict() == config.resolved_dict()


def test_future_float32_preflight_and_historical_review_fail_before_hub(
    tmp_path, monkeypatch
):
    raw = mapping(tmp_path, local=False)
    path = Path(raw["dataset"]["path"])
    data = pd.read_csv(path)
    data.loc[0, "demand"] = 1e100
    data.to_csv(path, index=False)
    monkeypatch.setattr(checkpoints, "_hub", lambda: pytest.fail("Hub import/network"))
    with pytest.raises(ValueError, match="representable as float32"):
        review_guided(config_from_mapping(raw, base_directory=tmp_path))
    dataset = Dataset(
        pd.date_range("2026-01-01", periods=32, freq="h", tz="UTC"),
        np.full(32, 1e100),
        "1h",
    )
    with pytest.raises(ValueError, match="representable as float32"):
        preflight_context(dataset, FutureForecastSpec(32, 4), "toto_2")


def test_guided_fake_preparation_download_cached_replay_and_stage_progress(
    tmp_path, fake_toto, monkeypatch
):
    raw = mapping(tmp_path, local=False)
    cache = tmp_path / "cache"
    raw["model"]["cache_dir"] = str(cache)
    hub = FakeHub(cache)
    hub.revision = REVISION
    hub.payloads["config.json"] = CONFIG.read_bytes()
    hub.snapshot = (
        lambda revision=REVISION: cache
        / "models--Datadog--Toto-2.0-4m"
        / "snapshots"
        / revision
    )
    monkeypatch.setattr("test_checkpoint_preparation.REPOSITORY", REPOSITORY)
    monkeypatch.setattr(checkpoints, "_hub", lambda: hub)

    def cached(repository, **options):
        assert (
            repository == REPOSITORY
            and options["revision"] == REVISION
            and options["local_files_only"] is True
        )
        return str(hub.snapshot())

    hub.snapshot_download = cached
    monkeypatch.setattr(
        "tsfm_lab.guided.runtime_diagnostics",
        lambda: [
            {
                "adapter_id": "toto_2",
                "status": "available",
                "label": "fake TOTO",
                "detail": "fixture",
            }
        ],
    )
    config = config_from_mapping(raw, base_directory=tmp_path)
    reviewed = review_guided(config)
    assert not reviewed.checkpoint.cached
    assert [name for name, _ in hub.events] == ["metadata", "config.json"]
    stages = []
    first = execute_guided(
        reviewed, progress=lambda event: stages.append(event.get("stage"))
    )
    assert {
        "cache",
        "downloading",
        "verifying",
        "loading",
        "running",
        "evaluating",
        "saving",
    } <= set(stages)
    monkeypatch.setattr(
        hub, "model_info", lambda *a, **kw: pytest.fail("cached metadata network")
    )
    reviewed_again = review_guided(reviewed.config, offline=True)
    assert reviewed_again.checkpoint.cached
    second = execute_guided(reviewed_again)
    store = ExperimentStore(tmp_path / "results")
    np.testing.assert_array_equal(
        store.load_result(first["run_id"])[1].prediction,
        store.load_result(second["run_id"])[1].prediction,
    )


def fake_worker(config, events, cancellation):
    with pytest.MonkeyPatch.context() as monkeypatch:
        install_runtime(monkeypatch)
        if config.name == "cancel-native":
            _, runtime = toto2._runtime()

            def waiting(self, *args, **kwargs):
                events.send({"type": "native_started"})
                while not cancellation.is_set():
                    time.sleep(0.005)
                raise KeyboardInterrupt

            monkeypatch.setattr(runtime.Model, "forecast", waiting)
        _worker_main(config, events, cancellation)


@pytest.mark.parametrize("cancel", [False, True])
def test_managed_fake_toto_reaps_worker_and_reports_truthful_completion(
    tmp_path, cancel
):
    write_checkpoint(tmp_path / "weights")
    raw = mapping(tmp_path)
    raw["name"] = "cancel-native" if cancel else "complete-native"
    worker = ManagedRun(
        config_from_mapping(raw, base_directory=tmp_path), _target=fake_worker
    )
    worker.start()
    events, deadline = [], time.monotonic() + 10
    try:
        while worker.is_running and time.monotonic() < deadline:
            new = worker.poll()
            events.extend(new)
            if cancel and any(event["type"] == "native_started" for event in new):
                worker.request_cancel()
            time.sleep(0.005)
        events.extend(worker.poll())
        assert not worker.is_running
        assert events[-1]["type"] == ("cancelled" if cancel else "completed"), events
        record = ExperimentStore(tmp_path / "results").get_manifest(
            events[-1]["run_id"]
        )
        assert record["status"] == events[-1]["type"]
    finally:
        worker.close()
