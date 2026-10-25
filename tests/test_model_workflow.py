"""Checkpoint selection and saved model identity work without model runtimes."""

import hashlib
import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from tsfm_lab.application import run_from_config
from tsfm_lab.cli import app
from tsfm_lab.comparison import compare_runs
from tsfm_lab.config import ConfigError
from tsfm_lab.contracts import Dataset, EvaluationSpec, ModelCapabilities
from tsfm_lab.evaluation.runner import run_experiment
from tsfm_lab.experiments import ExperimentStore, StoreError
from tsfm_lab.models import checkpoints
from tsfm_lab.models.baselines import LastValue

SHA = "a" * 40
REPOSITORY = "amazon/chronos-bolt-tiny"
runner = CliRunner()


@pytest.fixture
def checkpoint_directory(tmp_path):
    directory = tmp_path / "local-checkpoint"
    directory.mkdir()
    config = {
        "architectures": ["ChronosBoltModelForForecasting"],
        "model_type": "t5",
        "chronos_pipeline_class": "ChronosBoltPipeline",
        "chronos_config": {
            "context_length": 32,
            "prediction_length": 8,
            "quantiles": [0.1, 0.5, 0.9],
        },
    }
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "model.safetensors").write_bytes(
        b"fake fixture weights; never loaded by a framework"
    )
    return directory


@pytest.fixture
def fake_runtime(monkeypatch):
    class FakeChronos(LastValue):
        adapter_id = "chronos_bolt"

        def __init__(self, checkpoint, **runtime):
            self.checkpoint = checkpoint
            self.source = checkpoint.source
            self.runtime = runtime

        def metadata(self):
            config = self.checkpoint.config["chronos_config"]
            return replace(
                super().metadata(),
                family="chronos",
                capabilities=ModelCapabilities(
                    max_context=config["context_length"],
                    max_horizon=config["prediction_length"],
                ),
            )

        def configuration(self):
            return {"checkpoint": self.checkpoint.record()}

        def runtime_configuration(self):
            return {"device": "cpu", "dtype": "float32", "batch_size": 1}

        def environment_metadata(self):
            return {
                name: "fake-runtime-version"
                for name in (
                    "torch",
                    "chronos-forecasting",
                    "transformers",
                    "huggingface-hub",
                    "safetensors",
                    "accelerate",
                )
            }

        def predict(self, requests):
            return tuple(
                replace(value, point_statistic="median")
                for value in super().predict(requests)
            )

    monkeypatch.setattr("tsfm_lab.models.chronos_bolt.ChronosBolt", FakeChronos)
    return FakeChronos


def local_config(config_factory, checkpoint_directory, **changes):
    return config_factory(
        model={
            "source": "local",
            "adapter_id": "chronos_bolt",
            "path": str(checkpoint_directory),
        },
        **changes,
    )


def test_direct_python_model_result_has_complete_environment_and_can_be_saved(
    checkpoint_directory, fake_runtime, tmp_path
):
    times = pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC")
    result = run_experiment(
        Dataset(times, np.arange(8), "1h"),
        fake_runtime(checkpoints.inspect_local_checkpoint(checkpoint_directory)),
        EvaluationSpec(3, 2, times[4], times[7]),
    )
    assert result.provenance["environment"]["torch"] == "fake-runtime-version"
    store = ExperimentStore(tmp_path / "python-workspace")
    run_id = store.create("Python model run", {}, "# Python API")
    store.complete(run_id, result.to_dict(), result.predictions)
    loaded, frame = store.load_result(run_id)
    assert loaded["configuration"]["model"]["adapter_id"] == "chronos_bolt"
    assert len(frame) == 4


def test_local_model_records_checkpoint_and_actual_runtime(
    config_factory, checkpoint_directory, fake_runtime, tmp_path
):
    value = run_from_config(local_config(config_factory, checkpoint_directory))
    result, forecasts = ExperimentStore(tmp_path / "workspace").load_result(
        value["run_id"]
    )
    model = result["configuration"]["model"]
    assert model["source"] == "local"
    assert (
        model["options"]["checkpoint"]["files"]["model.safetensors"]["sha256"]
        == hashlib.sha256(
            (checkpoint_directory / "model.safetensors").read_bytes()
        ).hexdigest()
    )
    assert result["configuration"]["runtime"]["dtype"] == "float32"
    assert (
        result["provenance"]["environment"]["chronos-forecasting"]
        == "fake-runtime-version"
    )
    assert set(forecasts.point_statistic) == {"median"}


def test_cached_model_run_never_downloads_or_resolves_online(
    config_factory, checkpoint_directory, fake_runtime, tmp_path, monkeypatch
):
    cache = tmp_path / "hub-cache"
    snapshot = cache / "models--amazon--chronos-bolt-tiny" / "snapshots" / SHA
    shutil.copytree(checkpoint_directory, snapshot)
    calls = []

    def snapshot_download(repository, **kwargs):
        calls.append(kwargs)
        assert repository == REPOSITORY
        assert kwargs["revision"] == SHA and kwargs["local_files_only"] is True
        return str(snapshot)

    def no_api(**kwargs):
        raise AssertionError("offline runs must never call metadata")

    monkeypatch.setattr(
        checkpoints,
        "_hub",
        lambda: SimpleNamespace(snapshot_download=snapshot_download, HfApi=no_api),
    )
    path = config_factory(
        model={
            "source": "huggingface",
            "adapter_id": "chronos_bolt",
            "repository": REPOSITORY,
            "revision": SHA,
            "cache_dir": str(cache),
        }
    )
    value = run_from_config(path)
    assert len(calls) == 1
    result, _ = ExperimentStore(tmp_path / "workspace").load_result(value["run_id"])
    identity = result["configuration"]["model"]["options"]["checkpoint"]
    assert identity["requested_revision"] == SHA == identity["resolved_revision"]


def test_baseline_and_median_model_compare_under_same_task(
    config_factory, checkpoint_directory, fake_runtime, tmp_path
):
    baseline = run_from_config(config_factory())
    model = run_from_config(local_config(config_factory, checkpoint_directory))
    value = compare_runs([baseline["run_id"], model["run_id"]], tmp_path / "workspace")
    assert value["compatible"]
    assert [run["runtime"]["dtype"] for run in value["runs"]] == ["float64", "float32"]


def test_model_native_limit_failure_creates_no_record(
    config_factory, checkpoint_directory, fake_runtime, tmp_path
):
    path = local_config(
        config_factory, checkpoint_directory, evaluation={"context_length": 33}
    )
    with pytest.raises(ConfigError):
        run_from_config(path)
    assert not (tmp_path / "workspace").exists()


def test_completed_model_record_reloads_without_checkpoint_files(
    config_factory, checkpoint_directory, fake_runtime, tmp_path
):
    value = run_from_config(local_config(config_factory, checkpoint_directory))
    checkpoint_directory.rename(
        checkpoint_directory.with_name("checkpoint-unavailable")
    )
    result, frame = ExperimentStore(tmp_path / "workspace").load_result(value["run_id"])
    assert result["status"] == "completed" and len(frame) == 16


@pytest.mark.parametrize(
    "fault",
    ["source", "revision", "fingerprint", "capabilities", "dtype", "environment"],
)
def test_corrupt_checkpoint_metadata_is_rejected_even_with_new_artifact_hash(
    config_factory, checkpoint_directory, fake_runtime, tmp_path, fault
):
    value = run_from_config(local_config(config_factory, checkpoint_directory))
    directory = tmp_path / "workspace" / "experiments" / value["run_id"]
    result = json.loads((directory / "result.json").read_text())
    model = result["configuration"]["model"]
    record = model["options"]["checkpoint"]
    if fault == "source":
        model["source"] = "huggingface"
    elif fault == "revision":
        record["resolved_revision"] = "invented"
    elif fault == "fingerprint":
        record["files"]["config.json"]["sha256"] = "0" * 64
    elif fault == "capabilities":
        model["capabilities"]["max_horizon"] = 64
    elif fault == "dtype":
        result["configuration"]["runtime"]["dtype"] = "float64"
    else:
        del result["provenance"]["environment"]["torch"]
    raw = json.dumps(result).encode()
    (directory / "result.json").write_bytes(raw)
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["artifacts"]["result.json"] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(StoreError):
        ExperimentStore(tmp_path / "workspace").load_result(value["run_id"])


def test_model_info_is_offline_and_cache_inspection_is_read_only(tmp_path):
    info = runner.invoke(app, ["models", "info", REPOSITORY, "--json"])
    assert info.exit_code == 0, info.output
    assert json.loads(info.stdout)["point_statistic"] == "median"
    path = tmp_path / "absent-cache"
    cache = runner.invoke(app, ["models", "cache", "--cache-dir", str(path), "--json"])
    assert cache.exit_code == 0 and json.loads(cache.stdout) == []
    assert not path.exists()


def test_invalid_model_selection_is_rejected_before_pull(monkeypatch, tmp_path):
    def no_pull(*args, **kwargs):
        raise AssertionError("invalid selection must not download")

    monkeypatch.setattr("tsfm_lab.model_commands.pull_checkpoint", no_pull)
    for identifier in ("unsupported/repo", "chronos_bolt", "last_value"):
        result = runner.invoke(app, ["models", "pull", identifier, "--json"])
        assert result.exit_code == 2, result.output
        assert result.stdout == ""


def test_explicit_pull_command_passes_revision_and_cache(
    monkeypatch, checkpoint_directory, tmp_path
):
    checkpoint = checkpoints.inspect_local_checkpoint(checkpoint_directory)
    calls = []

    def pull(repository, revision, **kwargs):
        calls.append((repository, revision, kwargs))
        return checkpoint

    monkeypatch.setattr("tsfm_lab.model_commands.pull_checkpoint", pull)
    result = runner.invoke(
        app,
        [
            "models",
            "pull",
            REPOSITORY,
            "--revision",
            "selected-tag",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls == [(REPOSITORY, "selected-tag", {"cache_dir": tmp_path / "cache"})]
    assert json.loads(result.stdout)["fingerprint"] == checkpoint.fingerprint
    assert "Resolving" not in result.stdout
