"""Small disposable configurations for the complete shell workflow."""

import json
import shutil
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import yaml


@pytest.fixture
def config_factory(tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples/datasets/seasonal.csv"
    shutil.copyfile(source, tmp_path / "seasonal.csv")
    directory = tmp_path / "configs"
    directory.mkdir()

    def create(adapter="last_value", **changes):
        fields = {
            "schema_version": 1,
            "name": f"demo-{adapter}",
            "model": {"source": "builtin", "adapter_id": adapter},
            "dataset": {
                "path": "../seasonal.csv",
                "timestamp_column": "timestamp",
                "target_column": "demand",
                "frequency": "1h",
                "units": "synthetic_units",
            },
            "evaluation": {
                "context_length": 8,
                "horizon": 4,
                "stride": 4,
                "evaluation_start": "2026-01-01T16:00:00Z",
                "evaluation_end": "2026-01-02T08:00:00Z",
                "seasonal_period": 4,
            },
            "output": {"directory": "../workspace"},
        }
        if adapter == "seasonal_naive":
            fields["model"]["options"] = {"seasonal_period": 4}
        for key, value in changes.items():
            if isinstance(value, dict) and isinstance(fields.get(key), dict):
                fields[key].update(value)
            else:
                fields[key] = value
        path = directory / f"{uuid4().hex}.yaml"
        path.write_text(yaml.safe_dump(fields, sort_keys=False), encoding="utf-8")
        return path

    return create


@pytest.fixture
def timesfm_checkpoint(tmp_path):
    """A real 2.5 config with inert weights for fake-runtime contract tests."""
    directory = tmp_path / "timesfm-checkpoint"
    directory.mkdir()
    configuration = {
        "architectures": ["TimesFmModelForPrediction"],
        "context_length": 16384,
        "head_dim": 80,
        "hidden_size": 1280,
        "horizon_length": 128,
        "intermediate_size": 1280,
        "model_type": "timesfm",
        "num_attention_heads": 16,
        "num_hidden_layers": 20,
        "patch_length": 32,
        "quantile_horizon_length": 1024,
        "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "rms_norm_eps": 1e-6,
        "torch_compile": False,
    }
    (directory / "config.json").write_text(json.dumps(configuration))
    (directory / "model.safetensors").write_bytes(b"inert mocked TimesFM weights")
    return directory


@pytest.fixture
def timesfm_runtime(monkeypatch):
    """Exercise the adapter contract without importing or allocating a real model."""
    from tsfm_lab.models import timesfm as adapter

    state = SimpleNamespace(
        versions={
            "timesfm": "3.0.2",
            "torch": "2.8.0",
            "huggingface-hub": "0.34.4",
            "safetensors": "0.6.2",
        },
        load_calls=[],
        casts=[],
        compilations=[],
        predictions=[],
        pipelines=[],
        evaluations=0,
        inference=False,
        response=None,
        load_error=None,
        forecast_error=None,
        default_dtype="float32",
    )

    class Device:
        def __init__(self, name):
            assert name == "cpu"
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, *arguments):
            return None

    class Model:
        def to(self, **options):
            state.casts.append(options)
            return self

        def eval(self):
            state.evaluations += 1

    class Pipeline:
        def __init__(self, *, torch_compile):
            assert torch_compile is False
            self.model = Model()
            state.pipelines.append(weakref.ref(self))

        def load_checkpoint(self, path, **options):
            state.load_calls.append((path, options))
            if state.load_error:
                raise state.load_error

        def compile(self, config):
            state.compilations.append(config)

        def forecast(self, *, horizon, inputs):
            assert state.inference
            state.predictions.append((horizon, [item.copy() for item in inputs]))
            if state.forecast_error:
                raise state.forecast_error
            if state.response is not None:
                return state.response
            medians = np.stack(
                [item[-1] + np.arange(1, horizon + 1) for item in inputs]
            )
            quantiles = np.repeat(medians[:, :, None], 10, axis=2)
            quantiles[:, :, 0] += 100
            return medians, quantiles

    @contextmanager
    def inference_mode():
        state.inference = True
        try:
            yield
        finally:
            state.inference = False

    modules = {
        "torch": SimpleNamespace(
            device=Device,
            float32="float32",
            get_default_dtype=lambda: state.default_dtype,
            inference_mode=inference_mode,
        ),
        "timesfm": SimpleNamespace(
            TimesFM_2p5_200M_torch=Pipeline, ForecastConfig=SimpleNamespace
        ),
    }
    monkeypatch.setattr(adapter, "version", lambda name: state.versions[name])
    monkeypatch.setattr(
        adapter, "importlib", SimpleNamespace(import_module=modules.__getitem__)
    )
    return state
