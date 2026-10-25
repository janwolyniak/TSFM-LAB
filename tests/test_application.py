import json

import pytest

from tsfm_lab import application
from tsfm_lab.application import RunExecutionError, run_config, run_from_config
from tsfm_lab.config import ConfigError, load_config
from tsfm_lab.experiments import ExperimentStore


def test_saved_experiment_has_input_snapshot_resolved_paths_and_exact_scores(
    config_factory, tmp_path, monkeypatch
):
    path = config_factory()
    monkeypatch.chdir(tmp_path.parent)
    stages = []
    response = run_from_config(path, progress=stages.append)
    assert stages == ["preparing", "loading", "running", "evaluating", "saving"]
    assert response["metrics"]["mae"]["value"] == 2.5
    assert response["counts"]["forecast_cases"] == 16
    directory = tmp_path / "workspace" / "experiments" / response["run_id"]
    assert (directory / "input.yaml").read_text() == path.read_text()
    config = json.loads((directory / "configuration.json").read_text())
    assert config["dataset"]["path"] == str(tmp_path / "seasonal.csv")
    record, forecasts = ExperimentStore(tmp_path / "workspace").load_result(
        response["run_id"]
    )
    assert record["configuration"]["runtime"]["seed_used"] is False
    assert record["provenance"]["environment"]["PyYAML"]
    assert len(forecasts) == 16


@pytest.mark.parametrize(
    "changes",
    [
        {"dataset": {"path": "missing.csv"}},
        {"evaluation": {"horizon": 100}},
        {"model": {"source": "huggingface"}},
    ],
)
def test_invalid_inputs_fail_before_creating_workspace(
    config_factory, tmp_path, changes
):
    path = config_factory(**changes)
    with pytest.raises(ConfigError):
        run_from_config(path)
    assert not (tmp_path / "workspace").exists()


@pytest.mark.parametrize("cancelled", [False, True])
def test_execution_failure_or_cancellation_retains_stage_record(
    config_factory, tmp_path, monkeypatch, cancelled
):
    def broken(*args, progress, **kwargs):
        progress("loading")
        progress("running")
        if cancelled:
            raise KeyboardInterrupt()
        raise RuntimeError("synthetic inference failure")

    monkeypatch.setattr(application, "run_experiment", broken)
    expected = KeyboardInterrupt if cancelled else RunExecutionError
    with pytest.raises(expected):
        run_from_config(config_factory())
    records = ExperimentStore(tmp_path / "workspace").list_runs()
    assert len(records) == 1
    assert records[0]["status"] == ("cancelled" if cancelled else "failed")
    assert records[0]["error"]["stage"] == "running"
    with pytest.raises(ValueError, match="completed"):
        ExperimentStore(tmp_path / "workspace").load_result(records[0]["run_id"])


def test_failed_artifact_save_never_marks_a_run_completed(
    config_factory, tmp_path, monkeypatch
):
    def broken(*args, **kwargs):
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(ExperimentStore, "complete", broken)
    with pytest.raises(RunExecutionError, match="saving"):
        run_from_config(config_factory())
    (record,) = ExperimentStore(tmp_path / "workspace").list_runs()
    assert record["status"] == "failed"


def test_repeated_runs_get_new_ids_and_reproducible_scores(config_factory):
    config = config_factory("seasonal_naive")
    first, second = run_from_config(config), run_from_config(config)
    assert first["run_id"] != second["run_id"]
    assert first["metrics"] == second["metrics"]
    assert first["metrics"]["mae"]["value"] == 1


def test_created_callback_exposes_pending_identity_before_inference(config_factory):
    config = load_config(config_factory())
    identities = []

    def created(run_id):
        identities.append(run_id)
        assert (
            ExperimentStore(config.output_directory).get_manifest(run_id)["status"]
            == "pending"
        )

    result = run_config(config, on_created=created)
    assert identities == [result["run_id"]]


def test_cancellation_before_preflight_creates_no_artifacts(config_factory):
    config = load_config(config_factory())
    with pytest.raises(KeyboardInterrupt):
        run_config(config, should_cancel=lambda: True)
    assert not config.output_directory.exists()


def test_cancellation_immediately_after_creation_retains_cancelled_record(
    config_factory,
):
    config = load_config(config_factory())
    cancelled = False

    def created(run_id):
        nonlocal cancelled
        cancelled = True

    with pytest.raises(KeyboardInterrupt):
        run_config(config, on_created=created, should_cancel=lambda: cancelled)
    (record,) = ExperimentStore(config.output_directory).list_runs()
    assert record["status"] == "cancelled"
    assert record["error"]["stage"] == "pending"


def test_cancellation_after_completed_manifest_preserves_truth(
    config_factory, monkeypatch
):
    import tsfm_lab.experiments as records

    original = records._atomic

    def committed_then_interrupted(path, content):
        original(path, content)
        if (
            path.name == "manifest.json"
            and json.loads(content)["status"] == "completed"
        ):
            raise KeyboardInterrupt("late interrupt")

    monkeypatch.setattr(records, "_atomic", committed_then_interrupted)
    config = load_config(config_factory())
    result = run_config(config)
    assert result["status"] == "completed"
    assert (
        ExperimentStore(config.output_directory).get_manifest(result["run_id"])[
            "status"
        ]
        == "completed"
    )
