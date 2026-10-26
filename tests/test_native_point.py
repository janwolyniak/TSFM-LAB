"""Model-native point accuracy retains output meanings and validates cadence early."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tsfm_lab.application import run_from_config
from tsfm_lab.artifact_validation import StoreError, validate_predictions
from tsfm_lab.comparison import ComparisonError, compare_runs
from tsfm_lab.config import ConfigError
from tsfm_lab.contracts import Dataset, EvaluationSpec
from tsfm_lab.evaluation.runner import run_experiment
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.models.baselines import LastValue, SeasonalNaive
from tsfm_lab.models.checkpoints import inspect_local_checkpoint
from tsfm_lab.point_tasks import output_statistics, validate_model_frequency


class LabelledForecast(LastValue):
    """A deterministic fixture whose declared output semantics can be varied."""

    adapter_id = "point_fixture"

    def __init__(self, statistic):
        self.statistic = statistic
        self.unloaded = False

    def metadata(self):
        return replace(super().metadata(), family="test_fixture")

    def configuration(self):
        return {"fixture_output_statistic": self.statistic}

    def predict(self, requests):
        return tuple(
            replace(output, point_statistic=self.statistic)
            for output in super().predict(requests)
        )

    def unload(self):
        self.unloaded = True


@pytest.fixture
def data_and_spec():
    times = pd.date_range("2026-01-01", periods=16, freq="h", tz="UTC")
    dataset = Dataset(times, 10.0 + np.arange(16), "1h", units="fixture_units")
    spec = EvaluationSpec(
        4,
        2,
        times[8],
        times[-1] + pd.Timedelta("1h"),
        stride=2,
        point_statistic="native_point",
    )
    return dataset, spec


def persist(workspace, dataset, spec, statistic):
    result = run_experiment(dataset, LabelledForecast(statistic), spec)
    store = ExperimentStore(workspace)
    run_id = store.create("Point fixture", result.configuration, "# Direct fixture")
    store.complete(run_id, result.to_dict(), result.predictions)
    return run_id, result


@pytest.mark.parametrize(
    "statistic", ["deterministic", "median", "mean", "native_point"]
)
def test_native_task_retains_actual_output_labels_and_saved_scores(
    tmp_path, data_and_spec, statistic
):
    dataset, spec = data_and_spec
    run_id, original = persist(tmp_path, dataset, spec, statistic)
    saved, frame = ExperimentStore(tmp_path).load_result(run_id)
    assert saved["configuration"]["evaluation"]["point_statistic"] == "native_point"
    assert set(frame.point_statistic) == {statistic}
    np.testing.assert_array_equal(frame.prediction, original.predictions.prediction)
    assert saved["metrics"]["overall"]["mae"]["value"] == 1.5


@pytest.mark.parametrize("statistic", ["mean", "native_point"])
def test_median_task_cannot_accept_a_different_native_output(data_and_spec, statistic):
    dataset, spec = data_and_spec
    model = LabelledForecast(statistic)
    with pytest.raises(ValueError, match="requested task"):
        run_experiment(dataset, model, replace(spec, point_statistic="median"))
    assert model.unloaded


@pytest.mark.parametrize("model", [LastValue(), SeasonalNaive(2)])
def test_deterministic_baselines_are_valid_for_both_tasks(data_and_spec, model):
    dataset, spec = data_and_spec
    generic = run_experiment(dataset, model, spec)
    median = run_experiment(dataset, model, replace(spec, point_statistic="median"))
    np.testing.assert_array_equal(
        generic.predictions.prediction, median.predictions.prediction
    )
    assert set(generic.predictions.point_statistic) == {"deterministic"}
    assert generic.scores == median.scores


def test_native_comparison_preserves_mixed_semantics_and_uses_its_own_policy(
    tmp_path, data_and_spec
):
    dataset, spec = data_and_spec
    statistics = ["native_point", "median", "mean", "deterministic"]
    identifiers = [persist(tmp_path, dataset, spec, item)[0] for item in statistics]
    result = compare_runs(identifiers, tmp_path)
    assert result["comparison_policy"] == "controlled_native_point_v1"
    assert result["mixed_point_statistics"] is True
    assert [run["output_statistics"] for run in result["runs"]] == [
        [item] for item in statistics
    ]
    assert "common conditional mean or median" in result["point_semantics"]


def test_native_comparison_does_not_invent_mixed_labels(tmp_path, data_and_spec):
    dataset, spec = data_and_spec
    identifiers = [
        persist(tmp_path, dataset, spec, "native_point")[0] for _ in range(2)
    ]
    result = compare_runs(identifiers, tmp_path)
    assert result["mixed_point_statistics"] is False
    assert all(run["output_statistics"] == ["native_point"] for run in result["runs"])


def test_different_evaluation_tasks_are_not_silently_combined(tmp_path, data_and_spec):
    dataset, spec = data_and_spec
    generic = persist(tmp_path, dataset, spec, "median")[0]
    median = persist(
        tmp_path, dataset, replace(spec, point_statistic="median"), "median"
    )[0]
    with pytest.raises(ComparisonError, match="different task settings"):
        compare_runs([generic, median], tmp_path)


def test_saved_median_record_cannot_smuggle_a_native_output(data_and_spec):
    dataset, spec = data_and_spec
    original = run_experiment(dataset, LabelledForecast("median"), spec)
    saved = json.loads(json.dumps(original.to_dict()))
    saved["configuration"]["evaluation"]["point_statistic"] = "median"
    frame = original.predictions.copy()
    frame["point_statistic"] = "native_point"
    with pytest.raises(StoreError, match="statistic"):
        validate_predictions(frame, saved)


@pytest.mark.parametrize("label", ["mean", "median", "native_point"])
def test_known_baselines_cannot_relabel_their_native_task_output(data_and_spec, label):
    dataset, spec = data_and_spec
    result = run_experiment(dataset, LastValue(), spec)
    saved = json.loads(json.dumps(result.to_dict()))
    frame = result.predictions.copy()
    frame["point_statistic"] = label
    with pytest.raises(StoreError, match="adapter semantics"):
        validate_predictions(frame, saved)


def checkpoint_result(tmp_path, data_and_spec, task):
    config = {
        "model_type": "t5",
        "architectures": ["ChronosBoltModelForForecasting"],
        "chronos_pipeline_class": "ChronosBoltPipeline",
        "chronos_config": {
            "context_length": 64,
            "prediction_length": 8,
            "quantiles": [0.1, 0.5, 0.9],
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(b"inert fixture weights")
    record = inspect_local_checkpoint(tmp_path).record()
    dataset, spec = data_and_spec
    result = run_experiment(dataset, LastValue(), replace(spec, point_statistic=task))
    saved = json.loads(json.dumps(result.to_dict()))
    model = saved["configuration"]["model"]
    model.update(
        adapter_id="chronos_bolt",
        family="chronos",
        source="local",
        options={"checkpoint": record},
    )
    model["capabilities"].update(max_context=64, max_horizon=8)
    saved["configuration"]["runtime"].update(dtype="float32")
    saved["provenance"]["environment"].update(
        {
            name: "fixture"
            for name in (
                "chronos-forecasting",
                "torch",
                "transformers",
                "huggingface-hub",
                "safetensors",
                "accelerate",
            )
        }
    )
    frame = result.predictions.copy()
    frame["adapter_id"] = "chronos_bolt"
    return saved, frame


@pytest.mark.parametrize("label", ["mean", "native_point", "deterministic"])
def test_known_checkpoint_output_semantics_bind_in_new_task(
    tmp_path, data_and_spec, label
):
    saved, frame = checkpoint_result(tmp_path, data_and_spec, "native_point")
    frame["point_statistic"] = label
    with pytest.raises(StoreError, match="adapter semantics"):
        validate_predictions(frame, saved)
    frame["point_statistic"] = "median"
    assert set(validate_predictions(frame, saved).point_statistic) == {"median"}


def test_legacy_median_record_validation_is_not_tightened(tmp_path, data_and_spec):
    saved, frame = checkpoint_result(tmp_path, data_and_spec, "median")
    assert set(frame.point_statistic) == {"deterministic"}
    assert set(validate_predictions(frame, saved).point_statistic) == {"deterministic"}


def test_output_statistics_are_immutable_and_unknown_tasks_fail():
    assert output_statistics("median") == frozenset({"deterministic", "median"})
    assert output_statistics("native_point") == frozenset(
        {"deterministic", "median", "mean", "native_point"}
    )
    for task in ("mean", "point", "", None):
        with pytest.raises(ValueError, match="point task"):
            output_statistics(task)


@pytest.mark.parametrize("invalid", [None, "not callable", 1])
def test_optional_frequency_hook_must_be_callable(invalid):
    model = LastValue()
    model.validate_frequency = invalid
    with pytest.raises(ValueError, match="callable"):
        validate_model_frequency(model, "1h")


@pytest.mark.parametrize("returned", [False, True, "accepted", {}])
def test_frequency_hook_must_raise_to_refuse_or_return_none(returned):
    model = LastValue()
    model.validate_frequency = lambda frequency: returned
    with pytest.raises(ValueError, match="return None"):
        validate_model_frequency(model, "1h")


def test_frequency_hook_receives_only_cadence_and_legacy_models_need_no_hook():
    seen = []
    model = LastValue()
    model.validate_frequency = lambda frequency: seen.append(frequency)
    validate_model_frequency(model, "60min")
    assert seen == ["60min"]
    assert validate_model_frequency(LastValue(), "1h") is None


def test_direct_runner_refuses_frequency_before_model_load(data_and_spec):
    dataset, spec = data_and_spec

    class Restricted(LastValue):
        def validate_frequency(self, frequency):
            assert frequency == "1h"
            raise ValueError("Unsupported fixture cadence")

        def load(self):
            pytest.fail("Frequency refusal must precede loading")

    with pytest.raises(ValueError, match="cadence"):
        run_experiment(dataset, Restricted(), spec)


def test_application_refuses_frequency_before_creating_run(
    config_factory, tmp_path, monkeypatch
):
    class Restricted(LastValue):
        def validate_frequency(self, frequency):
            assert frequency == "1h"
            raise ValueError("Unsupported fixture cadence")

        def load(self):
            pytest.fail("Frequency refusal must precede loading")

    monkeypatch.setattr("tsfm_lab.application.build_model", lambda *args: Restricted())
    created = []
    with pytest.raises(ConfigError, match="cadence"):
        run_from_config(config_factory(), on_created=created.append)
    assert created == [] and not (tmp_path / "workspace").exists()
