import copy
import importlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tsfm_lab.contracts import EvaluationSpec, ForecastRequest
from tsfm_lab.models import chronos_bolt as adapter
from tsfm_lab.models.checkpoints import CheckpointError, inspect_local_checkpoint
from tsfm_lab.models.chronos_bolt import ChronosBolt


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


@pytest.fixture
def checkpoint(tmp_path):
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
    (tmp_path / "model.safetensors").write_bytes(b"synthetic bytes for mocked runtime")
    return inspect_local_checkpoint(tmp_path)


@pytest.fixture
def fake_runtime(monkeypatch):
    state = SimpleNamespace(
        load_calls=[],
        tensor_calls=[],
        prediction_calls=[],
        evaluations=0,
        inference=False,
        response=None,
        loading_info={
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        },
    )

    class Model:
        _keys_to_ignore_on_load_missing = [
            "input_patch_embedding",
            "output_patch_embedding",
        ]
        _keys_to_ignore_on_load_unexpected = ["lm_head.weight"]

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert cls._keys_to_ignore_on_load_missing == []
            assert cls._keys_to_ignore_on_load_unexpected == []
            state.load_calls.append((path, kwargs))
            return cls(), copy.deepcopy(state.loading_info)

        def eval(self):
            state.evaluations += 1
            return self

    class Pipeline:
        def __init__(self, model):
            self.model = model

        def predict_quantiles(self, *, inputs, prediction_length, quantile_levels):
            assert state.inference
            state.prediction_calls.append((inputs, prediction_length, quantile_levels))
            if state.response is not None:
                return state.response
            values = np.arange(len(inputs) * prediction_length, dtype=np.float32)
            values = values.reshape(len(inputs), prediction_length, 1) + 10
            return FakeTensor(values), FakeTensor(
                np.full((len(inputs), prediction_length), 999.0)
            )

    def as_tensor(values, *, dtype, device):
        assert dtype == "float32" and device == "cpu"
        state.tensor_calls.append(values)
        return FakeTensor(np.array(values, dtype=np.float32))

    @contextmanager
    def inference_mode():
        state.inference = True
        try:
            yield
        finally:
            state.inference = False

    modules = {
        "torch": SimpleNamespace(
            float32="float32", as_tensor=as_tensor, inference_mode=inference_mode
        ),
        "chronos": SimpleNamespace(ChronosBoltPipeline=Pipeline),
        "chronos.chronos_bolt": SimpleNamespace(ChronosBoltModelForForecasting=Model),
    }
    state.versions = {
        "chronos-forecasting": "2.2.0",
        "torch": "2.8.0",
        "transformers": "4.55.0",
        "huggingface-hub": "0.35.3",
        "safetensors": "0.6.2",
        "accelerate": "1.10.0",
    }
    monkeypatch.setattr(adapter, "version", lambda name: state.versions[name])
    monkeypatch.setattr(
        adapter, "importlib", SimpleNamespace(import_module=modules.__getitem__)
    )
    return state


def request(request_id="first", *, length=16, horizon=4, value=1.0):
    timestamps = pd.date_range(
        "2026-01-01", periods=length + horizon, freq="h", tz="UTC"
    )
    return ForecastRequest(
        request_id,
        "series",
        timestamps[length - 1],
        timestamps[:length],
        np.full(length, value),
        timestamps[length:],
    )


def spec(**changes):
    options = dict(
        context_length=16,
        horizon=4,
        evaluation_start="2026-01-02T00:00:00Z",
        evaluation_end="2026-01-03T00:00:00Z",
    )
    return EvaluationSpec(**(options | changes))


def test_importing_adapter_does_not_import_torch_or_chronos():
    root = Path(__file__).resolve().parents[1]
    program = "import sys; import tsfm_lab.models.chronos_bolt; assert 'torch' not in sys.modules; assert 'chronos' not in sys.modules"
    subprocess.run(
        [sys.executable, "-c", program],
        cwd=root,
        check=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )


def test_metadata_configuration_and_runtime_are_explicit(checkpoint):
    model = ChronosBolt(checkpoint, seed=42)
    info = model.metadata()
    assert info.adapter_id == "chronos_bolt" and info.family == "chronos"
    assert info.capabilities.max_context == 64
    assert info.capabilities.max_horizon == 8
    assert info.capabilities.devices == ("cpu",)
    assert model.source == "local"
    assert model.runtime_configuration() == {
        "device": "cpu",
        "dtype": "float32",
        "batch_size": 1,
        "seed": 42,
        "seed_used": False,
        "deterministic": True,
    }
    identity = model.configuration()
    identity["checkpoint"]["fingerprint"] = "modified"
    assert model.configuration()["checkpoint"]["fingerprint"] == checkpoint.fingerprint


def test_environment_metadata_reads_versions_without_runtime_imports(
    checkpoint, fake_runtime, monkeypatch
):
    def forbidden(name):
        pytest.fail(f"Environment metadata must not import {name}")

    monkeypatch.setattr(adapter, "importlib", SimpleNamespace(import_module=forbidden))
    assert ChronosBolt(checkpoint).environment_metadata() == fake_runtime.versions
    assert fake_runtime.load_calls == []


@pytest.mark.parametrize(
    "options",
    [
        {"device": "cuda"},
        {"device": "mps"},
        {"dtype": "float64"},
        {"seed": True},
        {"seed": -1},
        {"seed": 1.0},
    ],
)
def test_unsupported_runtime_rejected(checkpoint, options):
    with pytest.raises(ValueError):
        ChronosBolt(checkpoint, **options)


@pytest.mark.parametrize("changes", [{"context_length": 65}, {"horizon": 9}])
def test_native_limits_rejected_before_loading(checkpoint, changes):
    with pytest.raises(ValueError, match="native"):
        ChronosBolt(checkpoint).validate(spec(**changes))


def test_load_is_local_safe_strict_and_idempotent(checkpoint, fake_runtime):
    model = ChronosBolt(checkpoint)
    model.load()
    model.load()
    assert fake_runtime.evaluations == 1
    assert fake_runtime.load_calls == [
        (
            str(checkpoint.path),
            {
                "device_map": "cpu",
                "torch_dtype": "float32",
                "local_files_only": True,
                "trust_remote_code": False,
                "use_safetensors": True,
                "output_loading_info": True,
                "ignore_mismatched_sizes": False,
            },
        )
    ]
    model.unload()
    model.unload()
    with pytest.raises(RuntimeError, match="Load"):
        model.predict((request(),))
    model.load()
    assert len(fake_runtime.load_calls) == 2


@pytest.mark.parametrize(
    "field", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]
)
def test_incomplete_or_incompatible_weights_are_rejected(
    checkpoint, fake_runtime, field
):
    fake_runtime.loading_info[field] = ["input_patch_embedding.hidden_layer.weight"]
    model = ChronosBolt(checkpoint)
    with pytest.raises(ValueError, match="incomplete|incompatible"):
        model.load()
    assert fake_runtime.evaluations == 0
    with pytest.raises(RuntimeError, match="Load"):
        model.predict((request(),))


def test_changed_checkpoint_fails_before_runtime_loading(checkpoint, fake_runtime):
    model = ChronosBolt(checkpoint)
    (checkpoint.path / "model.safetensors").write_bytes(b"changed weights")
    with pytest.raises(CheckpointError):
        model.load()
    assert fake_runtime.load_calls == []


def test_mutated_in_memory_config_is_rejected(checkpoint, fake_runtime):
    model = ChronosBolt(checkpoint)
    checkpoint.config["auto_map"] = {"AutoModel": "remote.py"}
    with pytest.raises(CheckpointError):
        model.load()
    assert fake_runtime.load_calls == []


def test_prediction_uses_authoritative_median_and_preserves_identity(
    checkpoint, fake_runtime
):
    model = ChronosBolt(checkpoint)
    model.load()
    requests = (request("one"), request("two", length=32))
    outputs = model.predict(requests)
    for index, (original, forecast) in enumerate(zip(requests, outputs, strict=True)):
        assert forecast.request_id == original.request_id
        assert forecast.origin == original.origin
        assert forecast.series_id == original.series_id
        assert forecast.target_timestamps.equals(original.target_timestamps)
        assert forecast.point_statistic == "median"
        assert forecast.point.dtype == np.float64
        np.testing.assert_array_equal(forecast.point, np.arange(4) + 10 + index * 4)
        assert not np.shares_memory(
            fake_runtime.tensor_calls[index], original.context_values
        )
    assert fake_runtime.prediction_calls[0][1:] == (4, [0.5])
    assert not fake_runtime.inference


@pytest.mark.parametrize(
    "requests",
    [
        (),
        (request(), request()),
        (request(), request("other", horizon=3)),
        (request(length=65),),
        (request(horizon=9),),
        (request(value=1e100),),
        (object(),),
    ],
)
def test_invalid_requests_fail_before_inference(checkpoint, fake_runtime, requests):
    model = ChronosBolt(checkpoint)
    model.load()
    with pytest.raises(ValueError):
        model.predict(requests)
    assert fake_runtime.prediction_calls == []


@pytest.mark.parametrize(
    "quantiles",
    [
        np.zeros((1, 4)),
        np.zeros((1, 1, 4)),
        np.zeros((2, 4, 1)),
        np.full((1, 4, 1), np.nan),
        np.full((1, 4, 1), np.inf),
        np.ones((1, 4, 1), dtype=complex),
    ],
)
def test_bad_quantile_shape_or_values_are_rejected(checkpoint, fake_runtime, quantiles):
    fake_runtime.response = (FakeTensor(quantiles), FakeTensor(np.zeros((1, 4))))
    model = ChronosBolt(checkpoint)
    model.load()
    with pytest.raises(ValueError, match="shape"):
        model.predict((request(),))


@pytest.mark.parametrize(
    "name,value",
    [
        ("chronos-forecasting", "2.3.0"),
        ("torch", "2.9.0"),
        ("transformers", "5.0.0"),
        ("huggingface-hub", "1.0.0"),
    ],
)
def test_untested_runtime_versions_fail_clearly(checkpoint, fake_runtime, name, value):
    fake_runtime.versions[name] = value
    with pytest.raises(ImportError, match="tsfm-lab\\[chronos\\]"):
        ChronosBolt(checkpoint).load()


def test_missing_runtime_has_actionable_error(checkpoint, monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(adapter, "version", missing)
    with pytest.raises(ImportError, match="optional tsfm-lab\\[chronos\\]"):
        ChronosBolt(checkpoint).load()
