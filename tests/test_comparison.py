import pytest

from tsfm_lab.application import run_from_config
from tsfm_lab.comparison import ComparisonError, compare_runs
from tsfm_lab.experiments import ExperimentStore, StoreError


def test_distinct_baselines_compare_on_the_same_task(config_factory, tmp_path):
    a = run_from_config(config_factory())
    b = run_from_config(config_factory("seasonal_naive"))
    value = compare_runs([a["run_id"], b["run_id"]], tmp_path / "workspace")
    assert value["compatible"] is True
    assert [r["metrics"]["mae"]["value"] for r in value["runs"]] == [2.5, 1.0]
    assert value["task"]["evaluation"]["context_length"] == 8
    assert value["comparison_policy"] == "controlled_point_v1"
    assert "mixed_point_statistics" not in value
    assert "point_semantics" not in value
    assert all("output_statistics" not in run for run in value["runs"])


@pytest.mark.parametrize(
    "changes",
    [
        {"evaluation": {"context_length": 9}},
        {"evaluation": {"stride": 2}},
        {"evaluation": {"seasonal_period": 1}},
        {"evaluation": {"horizon": 2}},
        {"evaluation": {"metrics": ["mae"]}},
        {"evaluation": {"point_statistic": "native_point"}},
        {"dataset": {"units": "other_units"}},
    ],
)
def test_different_tasks_are_rejected(config_factory, tmp_path, changes):
    a = run_from_config(config_factory())
    b = run_from_config(config_factory(**changes))
    with pytest.raises(ComparisonError, match="different"):
        compare_runs([a["run_id"], b["run_id"]], tmp_path / "workspace")


def test_equivalent_frequency_spelling_does_not_change_task(config_factory, tmp_path):
    a = run_from_config(config_factory())
    b = run_from_config(config_factory(dataset={"frequency": "60min"}))
    assert compare_runs([a["run_id"], b["run_id"]], tmp_path / "workspace")[
        "compatible"
    ]


def test_reused_ids_or_single_record_are_rejected(config_factory, tmp_path):
    a = run_from_config(config_factory())["run_id"]
    for ids in ([a], [a, a]):
        with pytest.raises(ComparisonError):
            compare_runs(ids, tmp_path / "workspace")


def test_actual_values_and_forecast_case_keys_are_verified(
    config_factory, tmp_path, monkeypatch
):
    a = run_from_config(config_factory())
    b = run_from_config(config_factory())
    original = ExperimentStore.load_result

    def changed(self, run_id):
        result, frame = original(self, run_id)
        if run_id == b["run_id"]:
            frame.loc[0, "actual"] += 1
        return result, frame

    monkeypatch.setattr(ExperimentStore, "load_result", changed)
    with pytest.raises(ComparisonError, match="actuals"):
        compare_runs([a["run_id"], b["run_id"]], tmp_path / "workspace")


def test_corrupt_artifacts_cannot_enter_comparison(config_factory, tmp_path):
    a = run_from_config(config_factory())
    b = run_from_config(config_factory())
    path = tmp_path / "workspace" / "experiments" / b["run_id"] / "predictions.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(StoreError):
        compare_runs([a["run_id"], b["run_id"]], tmp_path / "workspace")
