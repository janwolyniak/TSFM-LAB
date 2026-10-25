"""TimesFM adapter contracts use inert weights and an independent fake runtime."""

import gc
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsfm_lab.contracts import Dataset, EvaluationSpec, ForecastRequest
from tsfm_lab.evaluation.runner import run_experiment
from tsfm_lab.models import timesfm as adapter
from tsfm_lab.models.checkpoints import CheckpointError, inspect_local_checkpoint
from tsfm_lab.models.timesfm import FORECAST_OPTIONS, TimesFM


def request(identifier="first", *, length=16, horizon=4, value=2.0):
    times = pd.date_range("2026-01-01", periods=length + horizon, freq="h", tz="UTC")
    return ForecastRequest(
        identifier,
        "series",
        times[length - 1],
        times[:length],
        np.full(length, value),
        times[length:],
    )


def model(path, **options):
    return TimesFM(inspect_local_checkpoint(path), **options)


def test_import_does_not_load_any_optional_runtime():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tsfm_lab.models.timesfm; "
            "assert not {'torch', 'timesfm', 'huggingface_hub'} & sys.modules.keys()",
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_metadata_and_owned_configuration_are_explicit(timesfm_checkpoint):
    value = model(timesfm_checkpoint, seed=23)
    assert value.metadata().family == "timesfm"
    assert value.metadata().capabilities.max_context == 16256
    assert value.metadata().capabilities.max_horizon == 128
    assert value.source == "local"
    runtime = value.runtime_configuration()
    assert runtime["seed"] == 23 and runtime["seed_used"] is False
    assert runtime["torch_compile"] is False
    assert runtime["forecast_options"] == FORECAST_OPTIONS
    runtime["forecast_options"]["return_backcast"] = True
    identity = value.configuration()
    identity["checkpoint"]["fingerprint"] = "changed"
    assert value.runtime_configuration()["forecast_options"]["return_backcast"] is False
    assert value.configuration()["checkpoint"]["fingerprint"] != "changed"


@pytest.mark.parametrize(
    "options", [{"device": "cuda"}, {"dtype": "float64"}, {"seed": True}, {"seed": -1}]
)
def test_unsupported_runtime_is_rejected(timesfm_checkpoint, options):
    with pytest.raises(ValueError):
        model(timesfm_checkpoint, **options)


def test_runtime_version_and_distribution_metadata(timesfm_checkpoint, timesfm_runtime):
    value = model(timesfm_checkpoint)
    assert value.environment_metadata() == timesfm_runtime.versions
    for name, invalid in [
        ("timesfm", "2.0.0"),
        ("torch", "2.9.0"),
        ("huggingface-hub", "1.0.0"),
        ("safetensors", "0.7.0"),
    ]:
        original = timesfm_runtime.versions[name]
        timesfm_runtime.versions[name] = invalid
        with pytest.raises(ImportError, match="tsfm-lab\\[timesfm\\]"):
            value.load()
        timesfm_runtime.versions[name] = original
    assert not timesfm_runtime.pipelines


def test_missing_runtime_has_actionable_error(timesfm_checkpoint, monkeypatch):
    def absent(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(adapter, "version", absent)
    with pytest.raises(ImportError, match="optional tsfm-lab\\[timesfm\\]"):
        model(timesfm_checkpoint).load()


def test_native_limits_fail_before_loading(timesfm_checkpoint, timesfm_runtime):
    value = model(timesfm_checkpoint)
    for length, horizon in [(16257, 4), (16, 129)]:
        with pytest.raises(ValueError):
            value.validate(
                EvaluationSpec(
                    length, horizon, "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"
                )
            )
    assert not timesfm_runtime.load_calls


def test_local_loading_is_exact_cpu_idempotent_and_owned(
    timesfm_checkpoint, timesfm_runtime
):
    first = model(timesfm_checkpoint)
    first.load()
    first.load()
    second = model(timesfm_checkpoint)
    second.load()
    assert len(timesfm_runtime.load_calls) == 2
    assert timesfm_runtime.load_calls[0] == (
        str(timesfm_checkpoint / "model.safetensors"),
        {"torch_compile": False},
    )
    assert timesfm_runtime.casts == [{"device": "cpu", "dtype": "float32"}] * 4
    assert first._pipeline is not second._pipeline
    assert first._pipeline.model.device.name == "cpu"
    assert first._pipeline.model.device_count == 1
    first.unload()
    first.unload()
    gc.collect()
    assert timesfm_runtime.pipelines[0]() is None
    assert second._pipeline is not None
    second.unload()


def test_load_failure_releases_partial_model(timesfm_checkpoint, timesfm_runtime):
    timesfm_runtime.load_error = ValueError("incompatible weight keys")
    value = model(timesfm_checkpoint)
    with pytest.raises(ValueError, match="incompatible weight"):
        value.load()
    # A retained exception's traceback owns its upstream model frame until released.
    timesfm_runtime.load_error = None
    gc.collect()
    assert value._pipeline is None and value._torch is None
    assert timesfm_runtime.pipelines[0]() is None


def test_altered_weights_fail_before_runtime_allocation(
    timesfm_checkpoint, timesfm_runtime
):
    value = model(timesfm_checkpoint)
    (timesfm_checkpoint / "model.safetensors").write_bytes(b"changed bytes")
    with pytest.raises(CheckpointError, match="changed"):
        value.load()
    assert not timesfm_runtime.pipelines


def test_default_dtype_is_not_silently_changed(timesfm_checkpoint, timesfm_runtime):
    value = model(timesfm_checkpoint)
    timesfm_runtime.default_dtype = "float64"
    with pytest.raises(ValueError, match="default Torch dtype"):
        value.load()
    assert not timesfm_runtime.pipelines
    timesfm_runtime.default_dtype = "float32"
    value.load()
    timesfm_runtime.default_dtype = "float64"
    with pytest.raises(ValueError, match="default Torch dtype"):
        value.predict((request(),))
    assert timesfm_runtime.default_dtype == "float64"
    assert not timesfm_runtime.predictions


def test_predictions_use_median_not_mean_and_preserve_history(
    timesfm_checkpoint, timesfm_runtime
):
    value = model(timesfm_checkpoint)
    value.load()
    item = request(length=33)
    outputs = value.predict((item,))
    value.predict((item,))
    assert len(timesfm_runtime.compilations) == 1
    config = timesfm_runtime.compilations[0]
    assert config.max_context == 64 and config.max_horizon == 128
    assert vars(config) == {
        "max_context": 64,
        "max_horizon": 128,
        "per_core_batch_size": 1,
        **FORECAST_OPTIONS,
    }
    output = outputs[0]
    np.testing.assert_array_equal(output.point, [3, 4, 5, 6])
    assert output.point.dtype == np.float64
    assert output.point_statistic == "median" and output.adapter_id == "timesfm_2p5"
    assert output.target_timestamps.equals(item.target_timestamps)
    np.testing.assert_array_equal(item.context_values, np.full(33, 2.0))
    assert timesfm_runtime.predictions[0][1][0].dtype == np.float32
    value.predict((request(length=16),))
    assert timesfm_runtime.compilations[-1].max_context == 32


@pytest.mark.parametrize(
    "requests",
    [
        (),
        (None,),
        (request(), request()),
        (request(horizon=129),),
        (request(length=16257),),
        (request(value=1e100),),
        (request(), request("second", length=17)),
        (request(), request("second", horizon=5)),
    ],
)
def test_bad_requests_fail_before_forecasting(
    timesfm_checkpoint, timesfm_runtime, requests
):
    value = model(timesfm_checkpoint)
    value.load()
    with pytest.raises(ValueError):
        value.predict(requests)
    assert not timesfm_runtime.predictions


@pytest.mark.parametrize(
    "response",
    [
        None,
        [],
        (np.zeros((1, 4)),),
        (np.zeros((1, 4)), np.zeros((1, 4, 9))),
        (np.zeros((1, 4)), np.full((1, 4, 10), np.nan)),
        (np.zeros((1, 4)), np.ones((1, 4, 10))),
        (np.full((1, 4), np.inf), np.zeros((1, 4, 10))),
    ],
)
def test_bad_forecast_outputs_are_rejected(
    timesfm_checkpoint, timesfm_runtime, response
):
    value = model(timesfm_checkpoint)
    value.load()
    timesfm_runtime.response = response if response is not None else "invalid"
    with pytest.raises(ValueError):
        value.predict((request(),))


def test_prediction_requires_load(timesfm_checkpoint):
    with pytest.raises(RuntimeError, match="Load TimesFM"):
        model(timesfm_checkpoint).predict((request(),))


def test_maximum_native_patch_and_smallest_context_are_accepted(
    timesfm_checkpoint, timesfm_runtime
):
    value = model(timesfm_checkpoint)
    value.load()
    maximum = value.predict((request(length=16256, horizon=128),))[0]
    assert len(maximum.point) == 128
    assert timesfm_runtime.compilations[-1].max_context == 16256
    assert timesfm_runtime.compilations[-1].max_horizon == 128
    minimum = value.predict((request(length=1, horizon=1),))[0]
    assert len(minimum.point) == 1
    assert timesfm_runtime.compilations[-1].max_context == 32


def test_runner_unloads_after_failure_and_earlier_forecasts_ignore_future(
    timesfm_checkpoint, timesfm_runtime
):
    times = pd.date_range("2026-01-01", periods=16, freq="h", tz="UTC")
    values = np.arange(16, dtype=float)
    spec = EvaluationSpec(4, 2, times[8], times[12], stride=2)
    first = run_experiment(
        Dataset(times, values, "1h"), model(timesfm_checkpoint), spec
    )
    altered = values.copy()
    altered[10:] += 100
    second = run_experiment(
        Dataset(times, altered, "1h"), model(timesfm_checkpoint), spec
    )
    np.testing.assert_array_equal(
        first.predictions.prediction[:2], second.predictions.prediction[:2]
    )
    failing = model(timesfm_checkpoint)
    timesfm_runtime.forecast_error = ValueError("inference failed")
    with pytest.raises(ValueError, match="inference failed"):
        run_experiment(Dataset(times, values, "1h"), failing, spec)
    assert failing._pipeline is None and failing._torch is None
