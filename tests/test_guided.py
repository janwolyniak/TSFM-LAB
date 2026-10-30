"""Review ordering, frozen choices, and offline guided coordination failures."""

import json
import pickle
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from test_checkpoint_preparation import FakeHub
from test_chronos_bolt import fake_runtime as fake_runtime

import tsfm_lab.guided as guided
from tsfm_lab.application import run_config
from tsfm_lab.config import ConfigError, load_config
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.models.preparation import CheckpointPlan


def pretrained(config, tmp_path, *, revision="a" * 40):
    return replace(
        config,
        runtime={**config.runtime, "dtype": "float32"},
        model={
            "source": "huggingface",
            "adapter_id": "chronos_bolt",
            "repository": "amazon/chronos-bolt-tiny",
            "revision": revision,
            "offline": revision != "main",
            "cache_dir": str(tmp_path / "cache"),
            "options": {},
        },
    )


def checkpoint(tmp_path, *, context=2048):
    return CheckpointPlan(
        "amazon/chronos-bolt-tiny",
        "main",
        "a" * 40,
        tmp_path / "cache",
        False,
        None,
        (),
        b"{}",
        json.dumps(
            {
                "architecture": "ChronosBoltModelForForecasting",
                "native_context_length": context,
                "native_prediction_length": 64,
            }
        ).encode(),
    )


def available(*args):
    return [
        {
            "adapter_id": "chronos_bolt",
            "label": "Chronos-Bolt",
            "status": "available",
            "detail": "fake reviewed runtime",
        }
    ]


def prep_record(config):
    paths = list((config.output_directory / "preparations").glob("*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text())


def test_baseline_review_is_detached_pickleable_and_has_no_work_or_optional_runtime(
    config_factory, monkeypatch
):
    config = load_config(config_factory())

    def forbidden(*args, **kwargs):
        raise AssertionError("Optional model work before baseline confirmation")

    monkeypatch.setattr(guided, "runtime_diagnostics", forbidden)
    monkeypatch.setattr(guided, "plan_checkpoint", forbidden)
    plan = guided.review_guided(config)
    assert plan.checkpoint is None
    assert "Built-in baseline" in plan.summary()
    assert not config.output_directory.exists()
    snapshot = plan.config
    snapshot.model["adapter_id"] = "not-a-model"
    assert plan.config.model["adapter_id"] == "last_value"
    config.dataset["target_column"] = "changed-after-review"
    assert plan.config.dataset["target_column"] == "demand"
    assert pickle.loads(pickle.dumps(plan)).config == plan.config
    with pytest.raises(FrozenInstanceError):
        plan.dataset_sha256 = "bad"


def test_bad_dataset_and_settings_fail_before_runtime_or_hub(
    config_factory, tmp_path, monkeypatch
):
    config = pretrained(load_config(config_factory()), tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Expensive work preceded validation")

    monkeypatch.setattr(guided, "runtime_diagnostics", forbidden)
    monkeypatch.setattr(guided, "plan_checkpoint", forbidden)
    dataset = Path(config.dataset["path"])
    original = dataset.read_text()
    dataset.write_text("timestamp,demand\n2026-01-01T00:00:00Z,1\n")
    with pytest.raises(ValueError):
        guided.review_guided(config)
    dataset.write_text(original)
    with pytest.raises(ConfigError, match="supported limits"):
        guided.review_guided(
            replace(config, evaluation=replace(config.evaluation, horizon=65))
        )
    assert not config.output_directory.exists()


def test_missing_runtime_prevents_even_checkpoint_planning(
    config_factory, tmp_path, monkeypatch
):
    config = pretrained(load_config(config_factory()), tmp_path)
    monkeypatch.setattr(
        guided,
        "runtime_diagnostics",
        lambda: [
            available()[0]
            | {"status": "missing", "detail": "Install optional runtime explicitly"}
        ],
    )
    monkeypatch.setattr(
        guided,
        "plan_checkpoint",
        lambda *args, **kwargs: pytest.fail("Unexpected Hub access"),
    )
    with pytest.raises(ConfigError, match="runtime is unavailable"):
        guided.review_guided(config)
    assert not config.output_directory.exists()


@pytest.mark.parametrize("offline", [False, True])
def test_review_pins_exact_revision_and_checks_native_metadata(
    config_factory, tmp_path, monkeypatch, offline
):
    config = pretrained(load_config(config_factory()), tmp_path, revision="main")
    calls = []
    monkeypatch.setattr(guided, "runtime_diagnostics", available)

    def planned(repository, revision, **kwargs):
        calls.append((repository, revision, kwargs["offline"]))
        return checkpoint(tmp_path)

    monkeypatch.setattr(guided, "plan_checkpoint", planned)
    plan = guided.review_guided(config, offline=offline)
    assert calls == [("amazon/chronos-bolt-tiny", "main", offline)]
    assert plan.config.model["revision"] == "a" * 40
    assert plan.config.model["offline"] is True
    assert config.model["revision"] == "main"
    assert "size unknown" in plan.summary()
    assert not config.output_directory.exists()
    monkeypatch.setattr(
        guided,
        "plan_checkpoint",
        lambda *args, **kwargs: checkpoint(tmp_path, context=4),
    )
    with pytest.raises(ConfigError, match="reviewed checkpoint"):
        guided.review_guided(config)


def test_baseline_guided_outputs_match_old_numerical_contract(config_factory):
    config = load_config(config_factory())
    expected = run_config(config)
    plan = guided.review_guided(config)
    events, windows = [], []
    result = guided.execute_guided(
        plan,
        progress=events.append,
        window_progress=lambda done, total: windows.append((done, total)),
    )
    assert result["metrics"] == expected["metrics"]
    assert result["counts"] == expected["counts"]
    assert {
        "predictions.csv",
        "result.json",
        "forecast.svg",
        "provenance.json",
    } <= result["outputs"].keys()
    assert all(Path(path).is_file() for path in result["outputs"].values())
    assert windows[0] == (0, 4) and windows[-1] == (4, 4)
    assert [item["stage"] for item in events] == [
        "cache",
        "preparing",
        "loading",
        "running",
        "evaluating",
        "saving",
    ]
    record = prep_record(config)
    assert record["status"] == "completed"
    assert record["run_id"] == result["run_id"]
    assert record["history"][-1]["stage"] == "completed"
    assert (
        ExperimentStore(config.output_directory).get_manifest(expected["run_id"])[
            "schema_version"
        ]
        == 1
    )
    assert (
        ExperimentStore(config.output_directory).get_manifest(result["run_id"])[
            "schema_version"
        ]
        == 2
    )


def test_changed_data_after_review_requires_new_confirmation_before_core_work(
    config_factory, monkeypatch
):
    config = load_config(config_factory())
    plan = guided.review_guided(config)
    path = Path(config.dataset["path"])
    data = pd.read_csv(path)
    data.loc[0, "demand"] += 1
    data.to_csv(path, index=False)
    monkeypatch.setattr(
        guided,
        "_runtime_imports",
        lambda *args: pytest.fail("Runtime entered for stale review"),
    )
    with pytest.raises(ConfigError, match="changed"):
        guided.execute_guided(plan)
    assert prep_record(config)["status"] == "failed"
    assert not (config.output_directory / "experiments").exists()


def test_guided_forecast_inputs_remain_isolated_from_future_actuals(config_factory):
    config = load_config(
        config_factory(evaluation={"evaluation_end": "2026-01-01T20:00:00Z"})
    )
    before = guided.execute_guided(guided.review_guided(config))
    path = Path(config.dataset["path"])
    data = pd.read_csv(path)
    data.loc[16:19, "demand"] += 1000
    data.to_csv(path, index=False)
    after = guided.execute_guided(guided.review_guided(config))
    store = ExperimentStore(config.output_directory)
    _, original = store.load_result(before["run_id"])
    _, perturbed = store.load_result(after["run_id"])
    assert original.prediction.tolist() == perturbed.prediction.tolist()
    assert original.actual.tolist() != perturbed.actual.tolist()
    assert before["metrics"]["mae"] != after["metrics"]["mae"]


@pytest.mark.parametrize("failure", ["runtime", "download", "cancel"])
def test_injected_preparation_failures_never_publish_a_run(
    config_factory, tmp_path, monkeypatch, failure
):
    config = pretrained(load_config(config_factory()), tmp_path)
    monkeypatch.setattr(guided, "runtime_diagnostics", available)
    monkeypatch.setattr(
        guided, "plan_checkpoint", lambda *args, **kwargs: checkpoint(tmp_path)
    )
    plan = guided.review_guided(config)
    calls = []

    def runtime(config):
        calls.append("runtime")
        if failure == "runtime":
            raise ImportError("Incompatible runtime API")

    def prepare(plan, *, progress, should_cancel):
        calls.append("download")
        progress({"type": "stage", "stage": "downloading"})
        progress({"type": "transfer", "completed_bytes": 8, "total_bytes": None})
        raise (
            KeyboardInterrupt
            if failure == "cancel"
            else RuntimeError("Injected transfer failure")
        )

    monkeypatch.setattr(guided, "_runtime_imports", runtime)
    monkeypatch.setattr(guided, "prepare_checkpoint", prepare)
    monkeypatch.setattr(
        guided,
        "run_config",
        lambda *args, **kwargs: pytest.fail("Failed preparation started inference"),
    )
    with pytest.raises((ImportError, RuntimeError, KeyboardInterrupt)):
        guided.execute_guided(plan)
    record = prep_record(config)
    assert calls == (["runtime"] if failure == "runtime" else ["runtime", "download"])
    assert record["status"] == ("cancelled" if failure == "cancel" else "failed")
    assert record["error"]["stage"] == (
        "cache" if failure == "runtime" else "downloading"
    )
    assert not (config.output_directory / "experiments").exists()


def test_data_change_during_preparation_is_rechecked_before_model_allocation(
    config_factory, tmp_path, monkeypatch
):
    config = pretrained(load_config(config_factory()), tmp_path)
    monkeypatch.setattr(guided, "runtime_diagnostics", available)
    monkeypatch.setattr(
        guided, "plan_checkpoint", lambda *args, **kwargs: checkpoint(tmp_path)
    )
    plan = guided.review_guided(config)
    monkeypatch.setattr(guided, "_runtime_imports", lambda config: None)

    def prepare(plan, **kwargs):
        path = Path(config.dataset["path"])
        data = pd.read_csv(path)
        data.loc[0, "demand"] += 99
        data.to_csv(path, index=False)
        return SimpleNamespace(fingerprint="a" * 64)

    monkeypatch.setattr(guided, "prepare_checkpoint", prepare)
    monkeypatch.setattr(
        "tsfm_lab.application.build_model",
        lambda *args: pytest.fail("Changed data allocated model"),
    )
    with pytest.raises(ConfigError, match="dataset changed"):
        guided.execute_guided(plan)
    assert prep_record(config)["status"] == "failed"
    assert not (config.output_directory / "experiments").exists()


def test_prepared_checkpoint_identity_is_rechecked_before_allocation(
    config_factory, tmp_path, monkeypatch
):
    config = pretrained(load_config(config_factory()), tmp_path)
    monkeypatch.setattr(guided, "runtime_diagnostics", available)
    monkeypatch.setattr(
        guided, "plan_checkpoint", lambda *args, **kwargs: checkpoint(tmp_path)
    )
    plan = guided.review_guided(config)
    monkeypatch.setattr(guided, "_runtime_imports", lambda config: None)
    monkeypatch.setattr(
        guided,
        "prepare_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(fingerprint="a" * 64),
    )
    changed = SimpleNamespace(
        configuration=lambda: {"checkpoint": {"fingerprint": "b" * 64}}
    )
    monkeypatch.setattr("tsfm_lab.application.build_model", lambda *args: changed)
    with pytest.raises(ConfigError, match="Checkpoint changed after preparation"):
        guided.execute_guided(plan)
    assert prep_record(config)["status"] == "failed"
    assert not (config.output_directory / "experiments").exists()


@pytest.mark.usefixtures("fake_runtime")
def test_empty_cache_guided_handoff_then_offline_replay_uses_fake_hub_and_runtime(
    config_factory, tmp_path, monkeypatch
):
    from tsfm_lab.models import checkpoints as cp

    config = pretrained(load_config(config_factory()), tmp_path, revision="main")
    hub = FakeHub(tmp_path / "cache")

    def snapshot(repository, **options):
        assert options["local_files_only"] is True
        assert options["revision"] == "a" * 40
        return str(hub.snapshot())

    hub.snapshot_download = snapshot
    monkeypatch.setattr(cp, "_hub", lambda: hub)
    monkeypatch.setattr(guided, "runtime_diagnostics", available)
    plan = guided.review_guided(config)
    assert not (hub.snapshot() / "model.safetensors").exists()
    first = guided.execute_guided(plan)
    assert first["status"] == "completed"
    assert first["counts"]["forecast_cases"] == 16
    transfers = len(hub.events)
    replay = guided.review_guided(plan.config, offline=True)
    assert replay.checkpoint.cached is True
    second = guided.execute_guided(replay)
    assert second["metrics"] == first["metrics"]
    assert len(hub.events) == transfers
