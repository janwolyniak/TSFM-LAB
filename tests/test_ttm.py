"""TTM's causal scaling and strict loading contract without optional runtimes."""

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
from tsfm_lab.models import ttm as adapter
from tsfm_lab.models.checkpoints import (
    CheckpointError,
    inspect_local_checkpoint,
    validate_checkpoint_record,
)
from tsfm_lab.models.ttm import TTM

# Apache-2.0 R2 main config, pinned before its model weights were inspected:
# https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/blob/d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f/config.json
MAIN_CONFIG = {
    "adaptive_patching_levels": 3,
    "architectures": ["TinyTimeMixerForPrediction"],
    "categorical_vocab_size_list": None,
    "context_length": 512,
    "d_model": 192,
    "d_model_scale": 3,
    "decoder_adaptive_patching_levels": 0,
    "decoder_d_model": 128,
    "decoder_d_model_scale": 2,
    "decoder_mode": "common_channel",
    "decoder_num_layers": 2,
    "decoder_raw_residual": False,
    "distribution_output": "student_t",
    "dropout": 0.4,
    "enable_forecast_channel_mixing": False,
    "exogenous_channel_indices": None,
    "expansion_factor": 2,
    "fcm_context_length": 1,
    "fcm_gated_attn": True,
    "fcm_mix_layers": 3,
    "fcm_prepend_past": True,
    "fcm_use_mixer": True,
    "frequency_token_vocab_size": 5,
    "gated_attn": True,
    "head_dropout": 0.4,
    "init_embed": "pytorch",
    "init_linear": "pytorch",
    "init_processing": True,
    "init_std": 0.02,
    "loss": "mse",
    "mode": "common_channel",
    "model_type": "tinytimemixer",
    "norm_eps": 1e-5,
    "norm_mlp": "LayerNorm",
    "num_input_channels": 1,
    "num_layers": 2,
    "num_parallel_samples": 100,
    "num_patches": 8,
    "patch_last": True,
    "patch_length": 64,
    "patch_stride": 64,
    "positional_encoding_type": "sincos",
    "post_init": False,
    "prediction_channel_indices": None,
    "prediction_filter_length": None,
    "prediction_length": 96,
    "resolution_prefix_tuning": False,
    "scaling": "std",
    "self_attn": False,
    "self_attn_heads": 1,
    "stride_ratio": 1,
    "torch_dtype": "float32",
    "transformers_version": "4.37.2",
    "use_decoder": True,
    "use_positional_encoding": False,
}


def configuration(context=512):
    config = copy.deepcopy(MAIN_CONFIG)
    config["context_length"] = context
    if context in {1024, 1536}:
        config.update(
            d_model=384,
            decoder_d_model=256,
            frequency_token_vocab_size=8,
            patch_length=128,
            patch_stride=128,
        )
    if context == 1536:
        config["num_patches"] = 12
    return config


def write_checkpoint(directory, config=None):
    directory.mkdir(exist_ok=True, parents=True)
    (directory / "config.json").write_text(json.dumps(config or configuration()))
    (directory / "model.safetensors").write_bytes(b"inert TTM fixture; no model")
    return inspect_local_checkpoint(directory)


@pytest.fixture
def checkpoint(tmp_path):
    return write_checkpoint(tmp_path)


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
        models=[],
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
        versions={
            "granite-tsfm": "0.3.4",
            "torch": "2.8.0",
            "transformers": "4.56.0",
            "huggingface-hub": "0.35.3",
            "safetensors": "0.6.2",
            "accelerate": "1.12.0",
            "tokenizers": "0.22.0",
            "scikit-learn": "1.7.2",
            "filelock": "3.20.3",
        },
    )

    class Model:
        _keys_to_ignore_on_load_missing = ["head"]
        _keys_to_ignore_on_load_unexpected = ["other_head"]

        @classmethod
        def from_pretrained(cls, path, **options):
            assert state.device_scopes == ["cpu"]
            assert cls._keys_to_ignore_on_load_missing == []
            assert cls._keys_to_ignore_on_load_unexpected == []
            state.load_calls.append((path, options))
            model = cls()
            state.models.append(weakref.ref(model))
            return model, copy.deepcopy(state.loading_info)

        def eval(self):
            state.evaluations += 1
            return self

        def __call__(self, **options):
            assert state.inference and state.device_scopes == ["cpu"]
            state.prediction_calls.append(options)
            if state.prediction_error:
                raise state.prediction_error
            if state.response is not None:
                return state.response
            return SimpleNamespace(
                prediction_outputs=FakeTensor(
                    (0.25 + np.arange(96, dtype=np.float32) / 10)[None, :, None]
                )
            )

    def as_tensor(values, *, dtype, device):
        assert dtype == "float32" and device == "cpu"
        state.tensor_calls.append(values)
        return FakeTensor(np.array(values, dtype=np.float32))

    @contextmanager
    def device(name):
        state.device_scopes.append(name)
        try:
            yield
        finally:
            state.device_scopes.pop()

    @contextmanager
    def inference_mode():
        state.inference = True
        try:
            yield
        finally:
            state.inference = False

    modules = {
        "torch": SimpleNamespace(
            float32="float32",
            as_tensor=as_tensor,
            device=device,
            inference_mode=inference_mode,
            get_default_dtype=lambda: state.default_dtype,
        ),
        "tsfm_public.models.tinytimemixer": SimpleNamespace(
            TinyTimeMixerForPrediction=Model
        ),
    }
    monkeypatch.setattr(adapter, "version", state.versions.__getitem__)
    monkeypatch.setattr(
        adapter, "importlib", SimpleNamespace(import_module=modules.__getitem__)
    )
    return state


def request(request_id="first", *, values=None, length=512, horizon=4):
    timestamps = pd.date_range(
        "2026-01-01", periods=length + horizon, freq="h", tz="UTC"
    )
    return ForecastRequest(
        request_id,
        "series",
        timestamps[length - 1],
        timestamps[:length],
        np.arange(length, dtype=np.float64) if values is None else values,
        timestamps[length:],
    )


def spec(**changes):
    return EvaluationSpec(
        **{
            "context_length": 512,
            "horizon": 4,
            "evaluation_start": "2026-02-01T00:00:00Z",
            "evaluation_end": "2026-02-02T00:00:00Z",
            "point_statistic": "native_point",
            **changes,
        }
    )


def test_metadata_and_imports_need_no_optional_runtime(checkpoint, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tsfm_lab.models.ttm; "
            "assert not {'torch','tsfm_public','transformers','sklearn','datasets'} "
            ".intersection(sys.modules)",
        ],
        cwd=root,
        check=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    monkeypatch.setattr(adapter, "_runtime", lambda: pytest.fail("Runtime imported"))
    model = TTM(checkpoint, seed=42)
    metadata = model.metadata()
    assert metadata.adapter_id == "ttm_r2" and metadata.family == "ttm"
    assert metadata.capabilities.min_context == metadata.capabilities.max_context == 512
    assert metadata.capabilities.max_horizon == 96
    assert metadata.capabilities.devices == ("cpu",)
    assert model.source == "local"
    runtime = model.runtime_configuration()
    assert runtime["seed"] == 42 and runtime["seed_used"] is False
    assert runtime["torch_compile"] is False
    assert runtime["forecast_options"]["native_scaling"] == "std"
    runtime["forecast_options"]["constant_scale"] = 100
    assert model.runtime_configuration()["forecast_options"]["constant_scale"] == 1
    recorded = model.configuration()
    recorded["checkpoint"]["point_forecast"]["training_loss"] = "mae"
    assert (
        model.configuration()["checkpoint"]["point_forecast"]["training_loss"] == "mse"
    )
    model.validate(spec(horizon=96))


@pytest.mark.parametrize("context", [512, 1024, 1536])
def test_schema_two_binds_reviewed_profile_and_point_contract(tmp_path, context):
    checkpoint = write_checkpoint(tmp_path, configuration(context))
    record = checkpoint.record()
    assert validate_checkpoint_record(record) == record
    assert record["schema_version"] == 2
    assert record["quantiles"] == []
    assert record["native_context_length"] == context
    assert record["native_prediction_length"] == 96
    assert record["point_forecast"]["statistic"] == "native_point"
    assert record["point_forecast"]["training_loss"] == "mse"
    assert record["point_forecast"]["profile"] == f"{context}-96-r2"
    model = TTM(checkpoint)
    model.validate(spec(context_length=context))
    assert model.metadata().capabilities.min_context == context
    record["point_forecast"]["inverse_transform"] = "none"
    with pytest.raises(CheckpointError):
        validate_checkpoint_record(record)


@pytest.mark.parametrize(
    "key,value",
    [
        ("loss", "mae"),
        ("scaling", None),
        ("prediction_length", 192),
        ("prediction_filter_length", 4),
        ("resolution_prefix_tuning", True),
        ("num_input_channels", 2),
        ("context_length", 512.0),
        ("d_model", 384),
        ("unreviewed_option", True),
    ],
)
def test_unreviewed_config_fails_before_weight_access(
    tmp_path, monkeypatch, key, value
):
    from tsfm_lab.models import checkpoints

    config = configuration()
    config[key] = value
    monkeypatch.setattr(
        checkpoints, "_file", lambda *args, **kwargs: pytest.fail("Weight access")
    )
    with pytest.raises(CheckpointError):
        write_checkpoint(tmp_path, config)


@pytest.mark.parametrize(
    "options",
    [{"device": "mps"}, {"dtype": "float64"}, {"seed": True}, {"seed": -1}],
)
def test_unsupported_runtime_is_rejected(checkpoint, options):
    with pytest.raises(ValueError):
        TTM(checkpoint, **options)


@pytest.mark.parametrize(
    "changes",
    [
        {"context_length": 511},
        {"context_length": 513},
        {"horizon": 97},
        {"point_statistic": "median"},
        {"point_statistic": "mean"},
    ],
)
def test_incompatible_tasks_are_rejected(checkpoint, changes):
    with pytest.raises(ValueError):
        TTM(checkpoint).validate(spec(**changes))


@pytest.mark.parametrize("frequency", ["1min", "15min", "1h"])
def test_minute_to_hour_frequency_is_supported(checkpoint, frequency):
    TTM(checkpoint).validate_frequency(frequency)


@pytest.mark.parametrize("frequency", ["30s", "2h", "D", "ME"])
def test_unreviewed_frequency_is_rejected(checkpoint, frequency):
    with pytest.raises(ValueError):
        TTM(checkpoint).validate_frequency(frequency)


def test_loading_is_strict_local_evaluated_and_idempotent(checkpoint, fake_runtime):
    model = TTM(checkpoint)
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
    assert model._model is model._torch is None
    assert fake_runtime.models[-1]() is None
    with pytest.raises(RuntimeError):
        model.predict((request(),))


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
                **dict.fromkeys(
                    [
                        "missing_keys",
                        "unexpected_keys",
                        "mismatched_keys",
                        "error_msgs",
                    ],
                    [],
                ),
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
def test_incomplete_weights_fail_closed(checkpoint, fake_runtime, information):
    fake_runtime.loading_info = information
    model = TTM(checkpoint)
    with pytest.raises(ValueError):
        model.load()
    assert model._model is model._torch is None
    assert fake_runtime.evaluations == 0 and fake_runtime.models[-1]() is None


@pytest.mark.parametrize("mutation", ["weights", "configuration"])
def test_checkpoint_changes_fail_before_runtime_import(
    checkpoint, monkeypatch, mutation
):
    model = TTM(checkpoint)
    if mutation == "weights":
        (checkpoint.path / "model.safetensors").write_bytes(b"changed")
    else:
        checkpoint.config["loss"] = "mae"
    monkeypatch.setattr(adapter, "_runtime", lambda: pytest.fail("Runtime imported"))
    with pytest.raises(CheckpointError):
        model.load()


def test_external_scaling_is_float64_past_only_with_native_prefix_inverse(
    checkpoint, fake_runtime
):
    model = TTM(checkpoint)
    model.load()
    history = 1e12 + np.arange(512, dtype=np.float64) / 8
    original = request(values=history)
    other = request("other", values=4 * history - 17)
    before = history.copy()
    outputs = model.predict((original, other))
    for index, (item, output) in enumerate(
        zip((original, other), outputs, strict=True)
    ):
        location, scale = item.context_values.mean(), item.context_values.std(ddof=0)
        normalized = ((item.context_values - location) / scale).astype(np.float32)
        tensor = fake_runtime.tensor_calls[index]
        assert tensor.shape == (1, 512, 1) and tensor.dtype == np.float32
        np.testing.assert_array_equal(tensor[0, :, 0], normalized)
        assert not np.shares_memory(tensor, item.context_values)
        native_prefix = (0.25 + np.arange(96, dtype=np.float32) / 10)[:4]
        np.testing.assert_array_equal(
            output.point, native_prefix.astype(np.float64) * scale + location
        )
        assert (
            output.point.dtype == np.float64
            and output.point_statistic == "native_point"
        )
        assert (
            output.request_id == item.request_id and output.series_id == item.series_id
        )
        assert output.origin == item.origin and output.adapter_id == "ttm_r2"
        assert output.target_timestamps.equals(item.target_timestamps)
        call = fake_runtime.prediction_calls[index]
        assert set(call) == {"past_values", "return_dict"}
        assert call["return_dict"] is True
    np.testing.assert_array_equal(history, before)
    assert not fake_runtime.inference and fake_runtime.device_scopes == []


def test_constant_context_uses_unit_external_scale(checkpoint, fake_runtime):
    model = TTM(checkpoint)
    model.load()
    result = model.predict((request(values=np.full(512, 23.0), horizon=96),))[0]
    np.testing.assert_array_equal(fake_runtime.tensor_calls[0], np.zeros((1, 512, 1)))
    native = 0.25 + np.arange(96, dtype=np.float32) / 10
    np.testing.assert_array_equal(result.point, native.astype(np.float64) + 23)
    assert result.point.shape == (96,)


def test_runner_future_actuals_do_not_enter_normalization(checkpoint, fake_runtime):
    from tsfm_lab.contracts import Dataset
    from tsfm_lab.evaluation.runner import run_experiment

    timestamps = pd.date_range("2026-01-01", periods=517, freq="h", tz="UTC")
    values = 10 + np.sin(np.arange(517) / 4) + np.arange(517) / 200
    changed = values.copy()
    changed[512:] = 1e12
    evaluation = spec(evaluation_start=timestamps[512], evaluation_end=timestamps[516])
    model = TTM(checkpoint)
    first = run_experiment(Dataset(timestamps, values, "1h"), model, evaluation)
    second = run_experiment(Dataset(timestamps, changed, "1h"), model, evaluation)
    assert (
        first.counts["successful_windows"] == second.counts["successful_windows"] == 1
    )
    np.testing.assert_array_equal(
        first.predictions.prediction, second.predictions.prediction
    )
    np.testing.assert_array_equal(
        fake_runtime.tensor_calls[0], fake_runtime.tensor_calls[1]
    )
    assert model._model is model._torch is None
    assert all(reference() is None for reference in fake_runtime.models)


@pytest.mark.parametrize(
    "requests",
    [
        (),
        (object(),),
        (request(), request()),
        (request(), request("other", horizon=3)),
        (request(length=511),),
        (request(length=513),),
        (request(horizon=97),),
        (request(values=np.full(512, 1e308)),),
        (request(values=np.full(512, 1e-300) * np.arange(512)),),
    ],
)
def test_invalid_requests_fail_before_native_inference(
    checkpoint, fake_runtime, requests
):
    model = TTM(checkpoint)
    model.load()
    with pytest.raises(ValueError):
        model.predict(requests)
    assert fake_runtime.prediction_calls == []


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_history_is_rejected(checkpoint, fake_runtime, value):
    model = TTM(checkpoint)
    model.load()
    item = request()
    object.__setattr__(item, "context_values", np.full(512, value))
    with pytest.raises(ValueError, match="finite"):
        model.predict((item,))
    assert fake_runtime.prediction_calls == []


@pytest.mark.parametrize(
    "native",
    [
        np.zeros((96,)),
        np.zeros((1, 4, 1)),
        np.zeros((2, 96, 1)),
        np.zeros((1, 96, 2)),
        np.full((1, 96, 1), np.nan),
        np.full((1, 96, 1), np.inf),
        np.ones((1, 96, 1), dtype=complex),
        np.full((1, 96, 1), "1"),
    ],
)
def test_full_native_tensor_contract_is_enforced(checkpoint, fake_runtime, native):
    fake_runtime.response = SimpleNamespace(prediction_outputs=FakeTensor(native))
    model = TTM(checkpoint)
    model.load()
    with pytest.raises(ValueError):
        model.predict((request(),))
    assert not fake_runtime.inference and fake_runtime.device_scopes == []


@pytest.mark.parametrize(
    "response", [object(), SimpleNamespace(prediction_outputs=None)]
)
def test_missing_native_tensor_fails_clearly(checkpoint, fake_runtime, response):
    fake_runtime.response = response
    model = TTM(checkpoint)
    model.load()
    with pytest.raises(ValueError, match="prediction_outputs"):
        model.predict((request(),))


def test_nonfinite_external_inverse_is_rejected(checkpoint, fake_runtime):
    fake_runtime.response = SimpleNamespace(
        prediction_outputs=FakeTensor(np.full((1, 96, 1), 1e308))
    )
    model = TTM(checkpoint)
    model.load()
    with pytest.raises(ValueError, match="inverse"):
        model.predict((request(),))


@pytest.mark.parametrize("stage", ["load", "predict"])
def test_nonfloat32_default_is_rejected_without_mutation(
    checkpoint, fake_runtime, stage
):
    model = TTM(checkpoint)
    if stage == "predict":
        model.load()
    fake_runtime.default_dtype = "float64"
    with pytest.raises(ValueError, match="default.*dtype"):
        model.load() if stage == "load" else model.predict((request(),))
    assert fake_runtime.default_dtype == "float64"
    assert fake_runtime.prediction_calls == [] and fake_runtime.device_scopes == []


def test_prediction_error_exits_scopes_and_model_can_be_released(
    checkpoint, fake_runtime
):
    model = TTM(checkpoint)
    model.load()
    fake_runtime.prediction_error = RuntimeError("synthetic native failure")
    with pytest.raises(RuntimeError, match="synthetic"):
        model.predict((request(),))
    assert not fake_runtime.inference and fake_runtime.device_scopes == []
    model.unload()
    fake_runtime.prediction_error.__traceback__ = None
    assert fake_runtime.models[-1]() is None


def test_environment_metadata_needs_no_optional_import(
    checkpoint, fake_runtime, monkeypatch
):
    monkeypatch.setattr(
        adapter,
        "importlib",
        SimpleNamespace(import_module=lambda name: pytest.fail(name)),
    )
    assert TTM(checkpoint).environment_metadata() == fake_runtime.versions


@pytest.mark.parametrize(
    "name,value",
    [
        ("granite-tsfm", "0.3.5"),
        ("torch", "2.9.0"),
        ("transformers", "4.55.4"),
        ("huggingface-hub", "1.0.0"),
        ("tokenizers", "0.23.0"),
        ("scikit-learn", "1.8.0"),
        ("filelock", "3.20.2"),
    ],
)
def test_unreviewed_runtime_is_rejected(checkpoint, fake_runtime, name, value):
    fake_runtime.versions[name] = value
    with pytest.raises(ImportError, match="requirements-ttm"):
        TTM(checkpoint).load()


def test_missing_runtime_gives_install_guidance(checkpoint, monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(adapter, "version", missing)
    with pytest.raises(ImportError, match="optional tsfm-lab\\[ttm\\]"):
        TTM(checkpoint).load()


@pytest.mark.ttm_smoke
def test_ttm_local_cpu_smoke(record_property, monkeypatch):
    """Explicit local weights only; verify actual runner causality and release."""
    import gc
    import socket

    path = os.environ.get("TSFM_TTM_CHECKPOINT")
    if not path:
        pytest.skip("Set TSFM_TTM_CHECKPOINT to an authorized local TTM checkpoint")
    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append("network")
        raise AssertionError("Network access is prohibited during TTM smoke")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")
    torch = importlib.import_module("torch")
    dynamo = importlib.import_module("torch._dynamo.convert_frame")
    compiler_attempts = []

    def forbidden_compile(*args, **kwargs):
        compiler_attempts.append("Dynamo compile")
        raise AssertionError("Compilation is prohibited during TTM smoke")

    monkeypatch.setattr(dynamo, "_compile", forbidden_compile)
    from tsfm_lab.contracts import Dataset
    from tsfm_lab.evaluation.runner import run_experiment

    checkpoint = inspect_local_checkpoint(path)
    context = checkpoint.record()["native_context_length"]
    stamps = pd.date_range("2026-01-01", periods=context + 5, freq="h", tz="UTC")
    values = 10 + np.sin(np.arange(context + 5) / 4) + np.arange(context + 5) / 200
    evaluation = spec(
        context_length=context,
        evaluation_start=stamps[context],
        evaluation_end=stamps[context + 4],
    )
    model = TTM(checkpoint)
    references, strict_model = [], adapter._strict_model

    def observed_load(*args, **kwargs):
        native = strict_model(*args, **kwargs)
        references.append(weakref.ref(native))
        return native

    monkeypatch.setattr(adapter, "_strict_model", observed_load)
    dtype, precision = torch.get_default_dtype(), torch.get_float32_matmul_precision()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        dataset = Dataset(stamps, values, "1h", units="synthetic_units")
        first = run_experiment(dataset, model, evaluation)
        replay = run_experiment(dataset, model, evaluation)
        future = values.copy()
        future[context:] += 1000
        causal = run_experiment(
            Dataset(stamps, future, "1h", units="synthetic_units"), model, evaluation
        )
        assert first.counts["successful_windows"] == 1
        assert first.counts["forecast_cases"] == 4
        assert set(first.predictions.point_statistic) == {"native_point"}
        assert (
            first.predictions.target_timestamp.tolist()
            == stamps[context : context + 4].tolist()
        )
        assert np.isfinite(first.predictions.prediction).all()
        np.testing.assert_array_equal(
            first.predictions.prediction, replay.predictions.prediction
        )
        np.testing.assert_array_equal(
            first.predictions.prediction, causal.predictions.prediction
        )
        gc.collect()
        assert model._model is model._torch is None
        assert len(references) == 3 and all(item() is None for item in references)
        assert attempts == compiler_attempts == []
        assert torch.get_default_dtype() == dtype
        assert torch.get_float32_matmul_precision() == precision
    finally:
        model.unload()
        torch.set_num_threads(threads)
    record_property("checkpoint_fingerprint", checkpoint.fingerprint)
    record_property("repeat_and_future_isolation", "exactly equal forecasts")
    record_property("native_models_released", len(references))
    record_property("network_attempts", attempts)
    record_property("compiler_attempts", compiler_attempts)
    for name, installed in model.environment_metadata().items():
        record_property(name, installed)
