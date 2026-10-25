import json

import pytest

from tsfm_lab import application
from tsfm_lab.application import RunExecutionError, run_from_config
from tsfm_lab.config import ConfigError
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
