import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tsfm_lab.contracts import Dataset, EvaluationSpec, ModelCapabilities
from tsfm_lab.evaluation.runner import run_experiment
from tsfm_lab.models.baselines import LastValue, SeasonalNaive


@pytest.fixture
def dataset():
    return Dataset(
        timestamps=pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC"),
        values=np.arange(1.0, 9.0),
        frequency="1h",
        target_name="demand",
        units="synthetic_units",
    )


@pytest.fixture
def spec(dataset):
    return EvaluationSpec(
        3, 2, dataset.timestamps[4], dataset.timestamps[-1] + pd.Timedelta("1h")
    )


class RecordingModel(LastValue):
    def __init__(self):
        self.requests = []
        self.loaded = False
        self.unloaded = False

    def load(self):
        self.loaded = True

    def predict(self, requests):
        self.requests.extend(requests)
        return super().predict(requests)

    def unload(self):
        self.unloaded = True


def test_hand_calculated_end_to_end_alignment_and_metrics(dataset, spec):
    result = run_experiment(dataset, LastValue(), spec)
    assert result.scores.overall["mae"].value == 1.5
    assert result.scores.overall["rmse"].value == pytest.approx(np.sqrt(2.5))
    assert result.scores.overall["mase"].value == 1.5
    assert result.scores.by_lead[1]["mae"].value == 1
    assert result.scores.by_lead[2]["mae"].value == 2
    assert result.counts == dict(
        attempted_windows=3,
        successful_windows=3,
        failed_windows=0,
        forecast_cases=6,
        unique_target_timestamps=4,
    )
    assert result.predictions.actual.tolist() == [5, 6, 6, 7, 7, 8]
    assert result.predictions.prediction.tolist() == [4, 4, 5, 5, 6, 6]
    assert result.provenance["scale_history"]["observations"] == 4
    assert result.configuration["dataset"]["units"] == "synthetic_units"
    json.loads(json.dumps(result.to_dict(), allow_nan=False))


def test_requests_contain_only_history_and_resources_are_released(dataset, spec):
    model = RecordingModel()
    run_experiment(dataset, model, spec)
    assert model.loaded and model.unloaded
    np.testing.assert_array_equal(model.requests[0].context_values, [2, 3, 4])
    assert not hasattr(model.requests[0], "actual")
    assert model.requests[0].context_values.base is None


def test_future_perturbation_does_not_change_the_first_forecast(dataset, spec):
    changed = dataset.values.copy()
    changed[4:] += 1000
    other = replace(dataset, values=changed)
    a = run_experiment(dataset, LastValue(), spec).predictions
    b = run_experiment(other, LastValue(), spec).predictions
    assert a.prediction.iloc[:2].equals(b.prediction.iloc[:2])
    assert not a.prediction.iloc[2:].equals(b.prediction.iloc[2:])


@pytest.mark.parametrize("failure", ["load", "predict"])
def test_unload_runs_even_when_loading_or_inference_fails(dataset, spec, failure):
    class Broken(RecordingModel):
        def load(self):
            if failure == "load":
                raise RuntimeError("load failed")
            super().load()

        def predict(self, requests):
            raise RuntimeError("predict failed")

    model = Broken()
    with pytest.raises(RuntimeError, match="failed"):
        run_experiment(dataset, model, spec)
    assert model.unloaded


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "origin",
        "series",
        "request",
        "timestamp",
        "statistic",
        "model",
        "nonfinite",
        "length",
    ],
)
def test_malformed_outputs_fail_instead_of_dropping_forecasts(dataset, spec, fault):
    class Broken(RecordingModel):
        def predict(self, requests):
            (output,) = super().predict(requests)
            if fault == "missing":
                return ()
            if fault == "duplicate":
                return (output, output)
            edits = {
                "origin": {"origin": output.origin - pd.Timedelta("1h")},
                "series": {"series_id": "wrong"},
                "request": {"request_id": "wrong"},
                "timestamp": {
                    "target_timestamps": output.target_timestamps + pd.Timedelta("1h")
                },
                "statistic": {"point_statistic": "mean"},
                "model": {"adapter_id": "wrong"},
                "nonfinite": {"point": np.array([np.nan, 1.0])},
                "length": {"point": np.array([1.0])},
            }
            return (replace(output, **edits[fault]),)

    model = Broken()
    with pytest.raises(ValueError):
        run_experiment(dataset, model, spec)
    assert model.unloaded


def test_adapter_capabilities_fail_before_loading(dataset, spec):
    class Limited(RecordingModel):
        def metadata(self):
            return replace(
                super().metadata(), capabilities=ModelCapabilities(max_horizon=1)
            )

    model = Limited()
    with pytest.raises(ValueError, match="horizon"):
        run_experiment(dataset, model, spec)
    assert not model.loaded


def test_invalid_bounds_fail_before_loading(dataset, spec):
    model = RecordingModel()
    with pytest.raises(ValueError, match="history"):
        run_experiment(dataset, model, replace(spec, context_length=5))
    assert not model.loaded


def test_output_snapshots_survive_adapter_buffer_reuse(dataset, spec):
    class ReusesBuffer(LastValue):
        def __init__(self):
            self.previous = None

        def predict(self, requests):
            if self.previous is not None:
                self.previous.point.setflags(write=True)
                self.previous.point[:] = -999
            (self.previous,) = super().predict(requests)
            return (self.previous,)

    result = run_experiment(dataset, ReusesBuffer(), spec)
    assert result.predictions.prediction.tolist() == [4, 4, 5, 5, 6, 6]
    assert result.scores.overall["mae"].value == 1.5


def test_mutating_request_timestamps_cannot_relabel_ground_truth(dataset, spec):
    class MutatesDates(RecordingModel):
        def predict(self, requests):
            requests[0].target_timestamps.values[:] += np.timedelta64(1, "D")
            return super().predict(requests)

    original = dataset.timestamps.copy(deep=True)
    model = MutatesDates()
    with pytest.raises(ValueError, match="timestamps"):
        run_experiment(dataset, model, spec)
    assert model.unloaded
    assert dataset.timestamps.equals(original)


def test_repeated_baseline_runs_reproduce_numerical_result(dataset, spec):
    a = run_experiment(dataset, SeasonalNaive(2), spec)
    b = run_experiment(dataset, SeasonalNaive(2), spec)
    pd.testing.assert_frame_equal(a.predictions, b.predictions)
    assert a.scores == b.scores
    assert a.configuration == b.configuration


def test_configuration_is_detached_before_loading(dataset, spec):
    class MutatesOptions(LastValue):
        def __init__(self):
            self.options = {"nested": {"setting": "before"}}

        def configuration(self):
            return self.options

        def load(self):
            self.options["nested"]["setting"] = "during"

    model = MutatesOptions()
    result = run_experiment(dataset, model, spec)
    model.options["nested"]["setting"] = "after"
    assert result.configuration["model"]["options"] == {"nested": {"setting": "before"}}


def test_unrepresentable_fingerprint_timestamps_fail_before_loading():
    times = pd.date_range("2500-01-01", periods=8, freq="h", unit="s", tz="UTC")
    dataset = Dataset(times, np.arange(8), "1h")
    spec = EvaluationSpec(3, 2, times[4], times[-1])
    model = RecordingModel()
    with pytest.raises(ValueError, match="nanoseconds"):
        run_experiment(dataset, model, spec)
    assert not model.loaded


def test_zero_scale_serializes_as_null_with_reason(dataset, spec):
    constant = replace(dataset, values=np.ones(8))
    result = run_experiment(constant, LastValue(), spec).to_dict()
    assert result["metrics"]["overall"]["mase"] == {
        "value": None,
        "reason": "zero_scale",
    }
    assert '"value": null' in json.dumps(result, allow_nan=False)


def test_window_progress_reports_the_planned_total_and_each_completed_window(
    dataset, spec
):
    progress = []
    run_experiment(
        dataset,
        LastValue(),
        spec,
        window_progress=lambda done, total: progress.append((done, total)),
    )
    assert progress == [(0, 3), (1, 3), (2, 3), (3, 3)]


def test_cancellation_before_loading_never_initializes_model(dataset, spec):
    model = RecordingModel()
    with pytest.raises(KeyboardInterrupt, match="cancellation"):
        run_experiment(dataset, model, spec, should_cancel=lambda: True)
    assert not model.loaded


def test_cancellation_after_native_predict_stops_before_accepting_forecasts(
    dataset, spec
):
    cancellation = False

    class CancellingModel(RecordingModel):
        def predict(self, requests):
            nonlocal cancellation
            result = super().predict(requests)
            cancellation = True
            return result

    model = CancellingModel()
    with pytest.raises(KeyboardInterrupt, match="cancellation"):
        run_experiment(dataset, model, spec, should_cancel=lambda: cancellation)
    assert model.loaded and model.unloaded
    assert len(model.requests) == 1


def test_final_window_cancellation_prevents_metric_calculation(
    dataset, spec, monkeypatch
):
    import tsfm_lab.evaluation.runner as runner_module

    cancellation = False

    def progress(done, total):
        nonlocal cancellation
        if done == total:
            cancellation = True

    monkeypatch.setattr(
        runner_module,
        "score_forecasts",
        lambda *args, **kwargs: pytest.fail("Metrics after cancellation"),
    )
    with pytest.raises(KeyboardInterrupt):
        run_experiment(
            dataset,
            LastValue(),
            spec,
            window_progress=progress,
            should_cancel=lambda: cancellation,
        )
