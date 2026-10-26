"""Chronos-2's numerical and safe-loading contract without optional runtimes."""

import copy
import importlib
import json
import os
import subprocess
import sys
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tsfm_lab.contracts import EvaluationSpec, ForecastRequest
from tsfm_lab.models import chronos2 as adapter
from tsfm_lab.models.checkpoints import CheckpointError, inspect_local_checkpoint
from tsfm_lab.models.chronos2 import Chronos2

# Exact profile fields from the Apache-2.0 official Small config at:
# https://huggingface.co/autogluon/chronos-2-small/blob/ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a/config.json
SMALL_CONFIG = {
    "architectures": ["Chronos2Model"],
    "chronos_config": {
        "context_length": 8192,
        "input_patch_size": 16,
        "input_patch_stride": 16,
        "max_output_patches": 64,
        "output_patch_size": 16,
        "quantiles": [
            0.01,
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.99,
        ],
        "time_encoding_scale": 8192,
        "use_arcsinh": True,
        "use_reg_token": True,
    },
    "chronos_pipeline_class": "Chronos2Pipeline",
    "d_ff": 2048,
    "d_kv": 64,
    "d_model": 512,
    "dense_act_fn": "relu",
    "dropout_rate": 0.1,
    "feed_forward_proj": "relu",
    "initializer_factor": 0.05,
    "is_gated_act": False,
    "layer_norm_epsilon": 1e-6,
    "model_type": "t5",
    "num_heads": 8,
    "num_layers": 6,
    "pad_token_id": 0,
    "reg_token_id": 1,
    "rope_theta": 10000.0,
    "torch_dtype": "float32",
    "transformers_version": "4.49.0",
    "use_cache": True,
    "vocab_size": 2,
}


def write_checkpoint(directory, configuration):
    directory.mkdir(exist_ok=True)
    (directory / "config.json").write_text(json.dumps(configuration))
    (directory / "model.safetensors").write_bytes(b"inert mocked Chronos-2 weights")
    return inspect_local_checkpoint(directory)


@pytest.fixture
def checkpoint(tmp_path):
    return write_checkpoint(tmp_path, copy.deepcopy(SMALL_CONFIG))


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
def fake_runtime(monkeypatch):
    state = SimpleNamespace(
        load_calls=[],
        tensor_calls=[],
        prediction_calls=[],
        pipelines=[],
        evaluations=0,
        inference=False,
        response=None,
        prediction_error=None,
        loading_info={
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        },
        default_dtype="float32",
        device_scopes=[],
        quantiles=SMALL_CONFIG["chronos_config"]["quantiles"],
    )

    class Model:
        _keys_to_ignore_on_load_missing = ["forecasting_head"]
        _keys_to_ignore_on_load_unexpected = ["other_head"]

        @classmethod
        def from_pretrained(cls, path, **options):
            assert state.device_scopes == ["cpu"]
            assert cls._keys_to_ignore_on_load_missing == []
            assert cls._keys_to_ignore_on_load_unexpected == []
            state.load_calls.append((path, options))
            return cls(), copy.deepcopy(state.loading_info)

        def eval(self):
            state.evaluations += 1
            return self

    class Pipeline:
        def __init__(self, *, model):
            self.model = model
            state.pipelines.append(weakref.ref(self))

        def predict(self, **options):
            assert state.inference
            assert state.device_scopes == ["cpu"]
            state.prediction_calls.append(options)
            if state.prediction_error:
                raise state.prediction_error
            if state.response is not None:
                return state.response
            horizon = options["prediction_length"]
            return [
                FakeTensor(
                    np.stack(
                        [
                            item.values[-1] + np.arange(horizon) + 100 * level
                            for level in state.quantiles
                        ]
                    )[None, :, :]
                )
                for item in options["inputs"]
            ]

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

    @contextmanager
    def device(name):
        assert name == "cpu"
        state.device_scopes.append(name)
        try:
            yield
        finally:
            state.device_scopes.pop()

    modules = {
        "torch": SimpleNamespace(
            float32="float32",
            as_tensor=as_tensor,
            inference_mode=inference_mode,
            get_default_dtype=lambda: state.default_dtype,
            device=device,
        ),
        "chronos": SimpleNamespace(Chronos2Pipeline=Pipeline),
        "chronos.chronos2": SimpleNamespace(Chronos2Model=Model),
    }
    state.versions = {
        "chronos-forecasting": "2.2.0",
        "torch": "2.8.0",
        "transformers": "4.55.0",
        "huggingface-hub": "0.35.3",
        "safetensors": "0.6.2",
        "accelerate": "1.12.0",
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
    return EvaluationSpec(
        **{
            "context_length": 16,
            "horizon": 4,
            "evaluation_start": "2026-01-02T00:00:00Z",
            "evaluation_end": "2026-01-03T00:00:00Z",
            **changes,
        }
    )


def test_metadata_and_imports_need_no_optional_runtime(checkpoint, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tsfm_lab.models.chronos2; assert 'torch' not in sys.modules; assert 'chronos' not in sys.modules",
        ],
        cwd=root,
        check=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    monkeypatch.setattr(adapter, "_runtime", lambda: pytest.fail("Runtime import"))
    model = Chronos2(checkpoint, seed=42)
    metadata = model.metadata()
    assert metadata.adapter_id == "chronos_2" and metadata.family == "chronos"
    assert metadata.capabilities.max_context == 8192
    assert metadata.capabilities.max_horizon == 1024
    assert metadata.capabilities.devices == ("cpu",)
    assert model.source == "local"
    runtime = model.runtime_configuration()
    assert runtime == {
        "device": "cpu",
        "dtype": "float32",
        "batch_size": 1,
        "seed": 42,
        "seed_used": False,
        "deterministic": True,
        "forecast_options": {"cross_learning": False, "limit_prediction_length": True},
    }
    runtime["forecast_options"]["cross_learning"] = True
    assert model.runtime_configuration()["forecast_options"]["cross_learning"] is False
    record = model.configuration()
    record["checkpoint"]["fingerprint"] = "changed"
    assert model.configuration()["checkpoint"]["fingerprint"] == checkpoint.fingerprint
    model.validate(spec(context_length=8192, horizon=1024))


def test_environment_versions_do_not_import_runtime(
    checkpoint, fake_runtime, monkeypatch
):
    monkeypatch.setattr(
        adapter,
        "importlib",
        SimpleNamespace(import_module=lambda name: pytest.fail(name)),
    )
    assert Chronos2(checkpoint).environment_metadata() == fake_runtime.versions


@pytest.mark.parametrize("stage", ["load", "predict"])
def test_nonfloat32_global_default_is_rejected_without_mutation(
    checkpoint, fake_runtime, stage
):
    model = Chronos2(checkpoint)
    if stage == "predict":
        model.load()
    fake_runtime.default_dtype = "float64"
    with pytest.raises(ValueError, match="default.*dtype|default.*float32"):
        model.load() if stage == "load" else model.predict((request(),))
    assert fake_runtime.default_dtype == "float64"
    assert fake_runtime.device_scopes == []
    assert fake_runtime.prediction_calls == []


@pytest.mark.parametrize(
    "options",
    [
        {"device": "cuda"},
        {"device": "mps"},
        {"dtype": "float64"},
        {"seed": True},
        {"seed": -1},
        {"seed": 0.5},
    ],
)
def test_unsupported_runtime_rejected(checkpoint, options):
    with pytest.raises(ValueError):
        Chronos2(checkpoint, **options)


def test_another_valid_checkpoint_family_is_rejected(tmp_path):
    checkpoint = write_checkpoint(
        tmp_path,
        {
            "model_type": "t5",
            "architectures": ["ChronosBoltModelForForecasting"],
            "chronos_pipeline_class": "ChronosBoltPipeline",
            "chronos_config": {
                "context_length": 64,
                "prediction_length": 8,
                "quantiles": [0.1, 0.5, 0.9],
            },
        },
    )
    with pytest.raises(ValueError):
        Chronos2(checkpoint)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("chronos_config", "context_length", 8193),
        ("chronos_config", "input_patch_size", 8),
        ("chronos_config", "input_patch_stride", 8),
        ("chronos_config", "max_output_patches", 65),
        ("chronos_config", "output_patch_size", 32),
        ("chronos_config", "time_encoding_scale", 4096),
        ("chronos_config", "use_arcsinh", False),
        ("chronos_config", "use_reg_token", False),
        ("chronos_config", "input_patch_size", True),
        ("chronos_config", "quantiles", [0.1, 0.5, 0.9]),
        ("chronos_config", "unknown_future_option", True),
        (None, "d_model", 768),
        (None, "num_layers", 7),
        (None, "d_model", 512.0),
        (None, "d_kv", 32),
        (None, "rope_theta", 20000.0),
        (None, "dense_act_fn", "gelu"),
        (None, "pad_token_id", 42),
        (None, "model_type", "other"),
        (None, "chronos_pipeline_class", "ChronosBoltPipeline"),
        (None, "architectures", ["RemoteChronos2Model"]),
        (None, "hidden_size", 32),
        (None, "num_hidden_layers", 1),
        (None, "num_attention_heads", 1),
        (None, "n_positions", 4),
    ],
)
def test_unreviewed_config_changes_fail_before_weight_access(
    tmp_path, monkeypatch, section, key, value
):
    from tsfm_lab.models import checkpoints

    configuration = copy.deepcopy(SMALL_CONFIG)
    target = configuration if section is None else configuration[section]
    target[key] = value
    monkeypatch.setattr(
        checkpoints,
        "_file",
        lambda *args, **kwargs: pytest.fail("Weight access before profile validation"),
    )
    with pytest.raises(CheckpointError):
        write_checkpoint(tmp_path, configuration)


@pytest.mark.parametrize(
    "changes",
    [{"context_length": 8193}, {"horizon": 1025}, {"point_statistic": "mean"}],
)
def test_incompatible_tasks_fail_before_load(checkpoint, changes):
    with pytest.raises(ValueError):
        Chronos2(checkpoint).validate(spec(**changes))


def test_load_is_strict_local_evaluated_and_idempotent(checkpoint, fake_runtime):
    model = Chronos2(checkpoint)
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
    assert fake_runtime.pipelines[-1]() is None
    with pytest.raises(RuntimeError):
        model.predict((request(),))
    model.load()
    assert len(fake_runtime.load_calls) == 2


@pytest.mark.parametrize(
    "information",
    [
        None,
        {},
        {"missing_keys": []},
        {
            "missing_keys": (),
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        },
        *[
            {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
                key: ["bad tensor"],
            }
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ],
    ],
)
def test_incomplete_weights_and_loading_diagnostics_fail_closed(
    checkpoint, fake_runtime, information
):
    fake_runtime.loading_info = information
    model = Chronos2(checkpoint)
    with pytest.raises(ValueError):
        model.load()
    assert fake_runtime.evaluations == 0
    with pytest.raises(RuntimeError):
        model.predict((request(),))


@pytest.mark.parametrize("mutation", ["weights", "configuration"])
def test_changed_checkpoint_fails_before_runtime_import(
    checkpoint, monkeypatch, mutation
):
    model = Chronos2(checkpoint)
    if mutation == "weights":
        (checkpoint.path / "model.safetensors").write_bytes(b"changed")
    else:
        checkpoint.config["auto_map"] = {"AutoModel": "remote.py"}
    monkeypatch.setattr(
        adapter, "_runtime", lambda: pytest.fail("Runtime imported too early")
    )
    with pytest.raises(CheckpointError):
        model.load()


@pytest.mark.parametrize("standard", [False, True])
def test_median_grid_identity_and_independent_context_are_preserved(
    tmp_path, fake_runtime, standard
):
    config = copy.deepcopy(SMALL_CONFIG)
    if standard:
        config.update(d_model=768, d_ff=3072, num_heads=12, num_layers=12)
        config["chronos_config"]["quantiles"] = [
            0.01,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
            0.99,
        ]
    fake_runtime.quantiles = config["chronos_config"]["quantiles"]
    checkpoint = write_checkpoint(tmp_path, config)
    model = Chronos2(checkpoint)
    model.load()
    requests = (request("one", value=2.0), request("two", length=32, value=7.0))
    outputs = model.predict(requests)
    for index, (original, output) in enumerate(zip(requests, outputs, strict=True)):
        assert output.request_id == original.request_id
        assert output.series_id == original.series_id
        assert output.origin == original.origin
        assert output.target_timestamps.equals(original.target_timestamps)
        assert output.adapter_id == "chronos_2"
        assert output.point_statistic == "median" and output.point.dtype == np.float64
        np.testing.assert_array_equal(
            output.point, original.context_values[-1] + np.arange(4) + 50
        )
        assert not np.shares_memory(
            fake_runtime.tensor_calls[index], original.context_values
        )
    call = fake_runtime.prediction_calls[0]
    assert {key: value for key, value in call.items() if key != "inputs"} == {
        "prediction_length": 4,
        "context_length": 8192,
        "batch_size": 1,
        "cross_learning": False,
        "limit_prediction_length": True,
    }
    assert not fake_runtime.inference


@pytest.mark.parametrize(
    "requests",
    [
        (),
        (object(),),
        (request(), request()),
        (request(), request("other", horizon=3)),
        (request(length=8193),),
        (request(horizon=1025),),
        (request(value=1e100),),
    ],
)
def test_invalid_requests_fail_before_inference(checkpoint, fake_runtime, requests):
    model = Chronos2(checkpoint)
    model.load()
    with pytest.raises(ValueError):
        model.predict(requests)
    assert fake_runtime.prediction_calls == []


@pytest.mark.parametrize(
    "response",
    [
        [],
        [object()],
        (FakeTensor(np.zeros((1, 13, 4))),),
        [FakeTensor(np.zeros((13, 4)))],
        [FakeTensor(np.zeros((2, 13, 4)))],
        [FakeTensor(np.zeros((1, 12, 4)))],
        [FakeTensor(np.zeros((1, 13, 5)))],
        [FakeTensor(np.full((1, 13, 4), np.nan))],
        [FakeTensor(np.full((1, 13, 4), np.inf))],
        [FakeTensor(np.ones((1, 13, 4), dtype=complex))],
    ],
)
def test_malformed_native_outputs_are_rejected(checkpoint, fake_runtime, response):
    fake_runtime.response = response
    model = Chronos2(checkpoint)
    model.load()
    with pytest.raises(ValueError):
        model.predict((request(),))


def test_prediction_failure_exits_inference_scope_and_unloads(checkpoint, fake_runtime):
    fake_runtime.prediction_error = RuntimeError("synthetic prediction failure")
    model = Chronos2(checkpoint)
    model.load()
    try:
        with pytest.raises(RuntimeError, match="synthetic"):
            model.predict((request(),))
        assert not fake_runtime.inference
    finally:
        model.unload()
    fake_runtime.prediction_error.__traceback__ = None
    fake_runtime.prediction_error = None
    assert fake_runtime.pipelines[-1]() is None


def test_native_context_and_horizon_are_preserved_without_extension(
    checkpoint, fake_runtime
):
    model = Chronos2(checkpoint)
    model.load()
    output = model.predict((request(length=8192, horizon=1024),))[0]
    call = fake_runtime.prediction_calls[0]
    assert call["inputs"][0].values.shape == (8192,)
    assert call["prediction_length"] == 1024 and output.point.shape == (1024,)
    assert call["limit_prediction_length"] is True


@pytest.mark.parametrize(
    "name,value",
    [
        ("chronos-forecasting", "2.3.0"),
        ("torch", "2.9.0"),
        ("transformers", "5.0.0"),
        ("huggingface-hub", "1.0.0"),
        ("safetensors", "0.7.0"),
        ("accelerate", "1.13.0"),
    ],
)
def test_untested_runtime_versions_fail_clearly(checkpoint, fake_runtime, name, value):
    fake_runtime.versions[name] = value
    with pytest.raises(ImportError, match="tsfm-lab\\[chronos\\]"):
        Chronos2(checkpoint).load()


def test_missing_runtime_has_actionable_error(checkpoint, monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(adapter, "version", missing)
    with pytest.raises(ImportError, match="optional tsfm-lab\\[chronos\\]"):
        Chronos2(checkpoint).load()
