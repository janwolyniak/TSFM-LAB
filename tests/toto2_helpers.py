"""Inert official config and a fake TOTO runtime; never imports model packages."""

import json
import platform
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tsfm_lab.models import toto2, toto2_profile
from tsfm_lab.models.checkpoints import inspect_local_checkpoint

REPOSITORY = "Datadog/Toto-2.0-4m"
REVISION = "8306a9801cf98c0f5ffe4b2dcc8f496e616d84d9"
CONFIG = Path(__file__).parent / "fixtures" / "toto2" / "config.json"
VERSIONS = {
    "toto-2": "2.0.0",
    "torch": "2.8.0",
    "huggingface-hub": "0.35.3",
    "safetensors": "0.6.2",
    "numpy": "2.1.3",
    "gluonts": "0.16.2",
    "dd-unit-scaling": "0.1.0",
    "unit-scaling": "0.3.5",
    "einops": "0.8.2",
    "jaxtyping": "0.3.11",
    "lightning": "2.4.0",
    "pytorch-lightning": "2.4.0",
    "scipy": "1.15.3",
    "matplotlib": "3.10.8",
}


def write_checkpoint(path, config=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_bytes(
        CONFIG.read_bytes() if config is None else json.dumps(config).encode()
    )
    (path / "model.safetensors").write_bytes(
        b"inert TOTO 2 weights; never native model data"
    )
    return inspect_local_checkpoint(path)


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def install_runtime(monkeypatch):
    # Simulate a supported execution environment even when core tests use 3.11.
    monkeypatch.setattr(toto2_profile, "sys", SimpleNamespace(version_info=(3, 12)))
    monkeypatch.setattr(platform, "python_version", lambda: "3.12.0")
    state = SimpleNamespace(
        versions=dict(VERSIONS),
        imports=[],
        constructors=[],
        weight_reads=[],
        load_calls=[],
        forecasts=[],
        targets=[],
        models=[],
        casts=[],
        evaluations=0,
        dtype="float32",
        devices=[],
        inference=False,
        response=None,
        forecast_error=None,
        load_error=None,
        missing=[],
        unexpected=[],
        knots=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )

    class Model:
        def __init__(self, config):
            assert state.devices == ["cpu"]
            state.constructors.append(vars(config))
            state.models.append(weakref.ref(self))
            self.output_head = SimpleNamespace(knots=list(state.knots))

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            pytest.fail("TOTO Hub mixin loader must never be called")

        def load_state_dict(self, weights, *, strict):
            state.load_calls.append((weights, strict))
            if state.load_error:
                raise state.load_error
            return SimpleNamespace(
                missing_keys=state.missing, unexpected_keys=state.unexpected
            )

        def to(self, **options):
            state.casts.append(options)
            return self

        def eval(self):
            state.evaluations += 1
            return self

        def forecast(self, inputs, *, horizon, **options):
            assert state.inference and state.devices == ["cpu"]
            state.forecasts.append((inputs, horizon, options))
            if state.forecast_error:
                raise state.forecast_error
            if state.response is not None:
                return state.response
            last = inputs["target"].values[0, 0, -1]
            values = last + np.arange(9)[:, None, None, None] + np.arange(horizon)
            return self._clamp_nonfinite(Tensor(values.astype(np.float32)))

    def load_file(path, *, device):
        assert device == "cpu"
        state.weight_reads.append(path)
        return {"safe_parameter": Tensor(np.zeros(1))}

    def tensor(values, *, dtype, device):
        assert dtype == "float32" and device == "cpu"
        state.targets.append(values)
        return Tensor(np.asarray(values, dtype=np.float32))

    @contextmanager
    def device(name):
        state.devices.append(name)
        try:
            yield
        finally:
            state.devices.pop()

    @contextmanager
    def inference_mode():
        state.inference = True
        try:
            yield
        finally:
            state.inference = False

    def finite(tensor):
        return SimpleNamespace(
            all=lambda: SimpleNamespace(
                item=lambda: bool(np.isfinite(tensor.values).all())
            )
        )

    modules = {
        "torch": SimpleNamespace(
            float32="float32",
            bool="bool",
            long="int64",
            device=device,
            inference_mode=inference_mode,
            as_tensor=tensor,
            get_default_dtype=lambda: state.dtype,
            isfinite=finite,
            ones_like=lambda tensor, **kw: Tensor(
                np.ones_like(tensor.values, dtype=bool)
            ),
            zeros=lambda shape, **kw: Tensor(np.zeros(shape, dtype=np.int64)),
        ),
        "toto2": SimpleNamespace(Toto2Model=Model),
        "toto2.configuration": SimpleNamespace(Toto2ModelConfig=SimpleNamespace),
        "safetensors.torch": SimpleNamespace(load_file=load_file),
    }

    def imported(name):
        state.imports.append(name)
        return modules[name]

    monkeypatch.setattr(toto2, "version", state.versions.__getitem__)
    monkeypatch.setattr(toto2, "importlib", SimpleNamespace(import_module=imported))
    return state


@pytest.fixture
def fake_toto(monkeypatch):
    return install_runtime(monkeypatch)


def mapping(tmp_path, *, future=False, local=True):
    count = 80
    timestamps = pd.date_range("2026-01-01", periods=count, freq="h", tz="UTC")
    source = tmp_path / "series.csv"
    pd.DataFrame({"timestamp": timestamps, "demand": np.arange(count) / 10 - 8}).to_csv(
        source, index=False
    )
    model = {
        "source": "local",
        "adapter_id": "toto_2",
        "path": str(tmp_path / "weights"),
    }
    if not local:
        model = {
            "source": "huggingface",
            "adapter_id": "toto_2",
            "repository": REPOSITORY,
            "revision": REVISION,
        }
    raw = {
        "schema_version": 1,
        "model": model,
        "dataset": {
            "path": str(source),
            "timestamp_column": "timestamp",
            "target_column": "demand",
            "frequency": "1h",
            "units": "synthetic_units",
        },
        "evaluation": {
            "context_length": 32,
            "horizon": 8,
            "stride": 8,
            "evaluation_start": timestamps[32].isoformat(),
            "evaluation_end": (timestamps[-1] + pd.Timedelta("1h")).isoformat(),
        },
        "output": {"directory": str(tmp_path / "results")},
    }
    if future:
        raw.pop("evaluation")
        raw.update(
            schema_version=2,
            mode="future_forecast",
            forecast={"context_length": 32, "horizon": 8},
        )
    return raw
