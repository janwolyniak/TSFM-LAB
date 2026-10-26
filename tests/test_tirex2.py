"""TiRex-2 contract tests with inert weights and a fake optional runtime."""

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
from tsfm_lab.models import tirex2 as adapter
from tsfm_lab.models.checkpoints import CheckpointError, inspect_local_checkpoint
from tsfm_lab.models.tirex2 import TiRex2

# Reviewed Apache-2.0 main checkpoint configuration:
# https://huggingface.co/NX-AI/TiRex-2/blob/30788a311699acba7e6393ad9700ea7b7cb9bb2d/config.json
MAIN_CONFIG = {
    "num_blocks": 12,
    "embedding_dim": 512,
    "input_patch_size": 32,
    "output_patch_size": 32,
    "input_ff_dim": 2048,
    "h_expand": 4,
    "context_len": 2048,
    "future_len": 320,
    "dropout": 0.1,
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "act_func": "torch.nn.SiLU",
    "use_qk_norm": True,
    "tokenizer_cfg": {
        "input_patch_size": 32,
        "output_patch_size": 32,
        "input_patch_stride": 32,
    },
    "scaler_cfg": {"use_arcsinh": True, "binaryaware": True},
    "stack_out_norm_config": {"eps": 1e-7},
    "tta_diff": True,
    "stack_config": {
        "templates": {
            "slstm_attn": {
                "time_mixer": {
                    "model_type": "bi-slstm",
                    "embedding_dim": 512,
                    "num_heads": 4,
                    "rnn_backend": "flashrnn-cuda",
                    "num_slstm_heads": 4,
                },
                "variate_mixer": {"embedding_dim": 512, "num_heads": 4},
                "dropout": 0.1,
                "eps": 1e-7,
            },
            "mlstm_attn": {
                "time_mixer": {
                    "model_type": "bi-mlstm",
                    "embedding_dim": 512,
                    "num_heads": 4,
                    "rnn_backend": "flashrnn-cuda",
                },
                "variate_mixer": {"embedding_dim": 512, "num_heads": 4},
                "dropout": 0.1,
                "eps": 1e-7,
            },
        },
        "recipe": ["mlstm_attn", "slstm_attn"] * 6,
    },
}


def configuration(gifteval=False):
    config = copy.deepcopy(MAIN_CONFIG)
    if gifteval:
        config.update(
            context_len=8192,
            future_len=928,
            dropout=0.0,
            use_qk_norm=False,
            tta_sign_flip=True,
            tta_diff=False,
        )
        config["scaler_cfg"] = {"use_arcsinh": False, "binaryaware": False}
        for template in config["stack_config"]["templates"].values():
            template["dropout"] = 0.0
    return config


def write_checkpoint(directory, config=None):
    directory.mkdir(exist_ok=True, parents=True)
    (directory / "config.json").write_text(json.dumps(config or configuration()))
    (directory / "model.safetensors").write_bytes(
        b"inert TiRex-2 fixture; no serialized model"
    )
    return inspect_local_checkpoint(directory)


@pytest.fixture
def checkpoint(tmp_path):
    return write_checkpoint(tmp_path)


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __getitem__(self, key):
        return FakeTensor(self.values[key])

    def unsqueeze(self, axis):
        return FakeTensor(np.expand_dims(self.values, axis))

    @property
    def shape(self):
        return self.values.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


@pytest.fixture
def fake_runtime(monkeypatch):
    state = SimpleNamespace(
        constructors=[],
        files=[],
        loads=[],
        expansions=[],
        casts=[],
        tensor_calls=[],
        prediction_calls=[],
        models=[],
        evaluations=0,
        inference=False,
        response=None,
        prediction_error=None,
        load_error=None,
        alias_error=None,
        default_dtype="float32",
        device_scopes=[],
        disable_calls=[],
        compiler_disabled=False,
        compiled_block_calls=0,
        eager_block_calls=0,
        block_objects=[],
        block_member_ids=[],
        block_count=2,
        disable_error=None,
        loading_info=SimpleNamespace(missing_keys=[], unexpected_keys=[]),
        versions={
            "tirex-2": "0.4.0",
            "torch": "2.8.0",
            "huggingface-hub": "0.35.3",
            "safetensors": "0.6.2",
            "flashrnn": "1.0.8",
            "mlstm-kernels": "2.0.6",
            "ninja": "1.13.2",
            "numpy": "2.1.3",
            "narwhals": "2.13.0",
            "einops": "0.8.1",
            "matplotlib": "3.10.8",
        },
    )

    class Module:
        def __call__(self, *args, **kwargs):
            return self._call_impl(*args, **kwargs)

        def _call_impl(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    class ResidualBlock(Module):
        def __init__(self):
            self.parameter = object()
            self.buffer = object()
            self.hook = object()
            state.block_objects.append(weakref.ref(self))
            state.block_member_ids.append(
                (id(self.parameter), id(self.buffer), id(self.hook))
            )

        def __call__(self, *args, **kwargs):
            state.compiled_block_calls += 1
            raise AssertionError("Compiled residual-block class entry")

        def _call_impl(self, *args, **kwargs):
            state.compiled_block_calls += 1
            raise AssertionError("Compiled residual-block implementation")

        def forward(self, values):
            state.eager_block_calls += 1
            return values

    state.residual_class = ResidualBlock

    class Model:
        def __init__(self, **options):
            assert state.device_scopes == ["cpu"]
            assert options.get("device") == "cpu"
            assert options.get("matmul_precision") is None
            state.constructors.append(copy.deepcopy(options))
            state.models.append(weakref.ref(self))
            self.blocks = [ResidualBlock() for _ in range(state.block_count)]

        def modules(self):
            return iter((self, *self.blocks))

        def load_state_dict(self, weights, *, strict):
            assert strict is True
            assert weights["reverse"] is weights["forward"]
            state.loads.append((weights, strict))
            if state.load_error:
                raise state.load_error
            return state.loading_info

        def to(self, **options):
            state.casts.append(options)
            return self

        def eval(self):
            state.evaluations += 1
            return self

        def predict(self, inputs, prediction_length, **options):
            assert state.inference and state.device_scopes == ["cpu"]
            assert state.compiler_disabled
            assert len(inputs) == 1
            series = inputs[0]
            assert series.past_covariates is None and series.future_covariates is None
            target = series.target.values
            for block in self.blocks:
                target = block(target)
            assert target.ndim == 2 and target.shape[0] == 1
            state.prediction_calls.append((inputs, prediction_length, options))
            if state.prediction_error:
                raise state.prediction_error
            if state.response is not None:
                return state.response
            levels = np.arange(1, 10, dtype=np.float32) * 10
            return [
                FakeTensor(
                    target[0, -1] + levels[None, :, None] + np.arange(prediction_length)
                )
            ]

    def load_file(path, *, device):
        assert device == "cpu"
        state.files.append((path, device))
        return {"forward": object()}

    def expand(model, weights):
        assert isinstance(model, Model)
        state.expansions.append(tuple(weights))
        if state.alias_error:
            raise state.alias_error
        return {**weights, "reverse": weights["forward"]}

    def as_tensor(values, *, dtype, device):
        assert dtype == "float32" and device == "cpu"
        state.tensor_calls.append(values)
        return FakeTensor(np.array(values, dtype=np.float32))

    @contextmanager
    def device(name):
        assert name == "cpu"
        state.device_scopes.append(name)
        try:
            yield name
        finally:
            state.device_scopes.pop()

    @contextmanager
    def inference_mode():
        state.inference = True
        try:
            yield
        finally:
            state.inference = False

    @contextmanager
    def fork_rng(*args, **kwargs):
        yield

    def disable(function, *, recursive):
        state.disable_calls.append(recursive)
        if state.disable_error:
            raise state.disable_error

        def uncompiled(*args, **kwargs):
            state.compiler_disabled = True
            try:
                return function(*args, **kwargs)
            finally:
                state.compiler_disabled = False

        return uncompiled

    modules = {
        "torch": SimpleNamespace(
            float32="float32",
            as_tensor=as_tensor,
            device=device,
            get_default_dtype=lambda: state.default_dtype,
            inference_mode=inference_mode,
            random=SimpleNamespace(fork_rng=fork_rng),
            compiler=SimpleNamespace(disable=disable),
            nn=SimpleNamespace(Module=Module),
        ),
        "tirex2": SimpleNamespace(TiRex2=Model, TimeseriesType=SimpleNamespace),
        "tirex2.model": SimpleNamespace(TiRex2=Model),
        "tirex2.model.tirex2": SimpleNamespace(TiRex2=Model),
        "tirex2.model.types": SimpleNamespace(TimeseriesType=SimpleNamespace),
        "safetensors.torch": SimpleNamespace(load_file=load_file),
        "tirex2._state_dict": SimpleNamespace(expand_shared_duplicates=expand),
        "tirex2.model.component.residual_block": SimpleNamespace(
            ResidualBlock=ResidualBlock
        ),
    }
    monkeypatch.setattr(adapter, "version", lambda name: state.versions[name])
    monkeypatch.setattr(
        adapter, "importlib", SimpleNamespace(import_module=modules.__getitem__)
    )
    return state


def request(request_id="first", *, length=16, horizon=4, value=1.0):
    stamps = pd.date_range("2026-01-01", periods=length + horizon, freq="h", tz="UTC")
    return ForecastRequest(
        request_id,
        "series",
        stamps[length - 1],
        stamps[:length],
        np.full(length, value),
        stamps[length:],
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


def test_import_metadata_and_configuration_need_no_optional_runtime(
    checkpoint, monkeypatch
):
    root = Path(__file__).resolve().parents[1]
    program = "import sys; import tsfm_lab.models.tirex2; assert not {'torch','tirex2','flashrnn','mlstm_kernels'}.intersection(sys.modules)"
    subprocess.run(
        [sys.executable, "-c", program],
        cwd=root,
        check=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    monkeypatch.setattr(
        adapter, "_runtime", lambda: pytest.fail("Optional runtime import")
    )
    model = TiRex2(checkpoint, seed=42)
    info = model.metadata()
    assert (info.adapter_id, info.family) == ("tirex_2", "tirex")
    assert (info.capabilities.max_context, info.capabilities.max_horizon) == (2048, 320)
    assert info.capabilities.devices == ("cpu",) and model.source == "local"
    config = model.configuration()
    config["checkpoint"]["fingerprint"] = "changed"
    assert model.configuration()["checkpoint"]["fingerprint"] == checkpoint.fingerprint
    runtime = model.runtime_configuration()
    assert (
        runtime["device"] == "cpu"
        and runtime["dtype"] == "float32"
        and runtime["batch_size"] == 1
    )
    assert (
        runtime["seed"] == 42
        and runtime["seed_used"] is False
        and runtime["deterministic"] is True
    )
    assert runtime["torch_compile"] is False
    assert runtime["forecast_options"] == {
        "tta_sign_flip": False,
        "tta_diff": True,
        "pad_context": True,
    }
    runtime["forecast_options"]["tta_diff"] = False
    assert model.runtime_configuration()["forecast_options"]["tta_diff"] is True


@pytest.mark.parametrize(
    "options",
    [
        {"device": "cuda"},
        {"device": "mps"},
        {"dtype": "float64"},
        {"seed": True},
        {"seed": -1},
        {"seed": 0.1},
    ],
)
def test_unsupported_runtime_rejected(checkpoint, options):
    with pytest.raises(ValueError):
        TiRex2(checkpoint, **options)


def test_environment_versions_are_available_without_runtime_imports(
    checkpoint, fake_runtime, monkeypatch
):
    monkeypatch.setattr(
        adapter,
        "importlib",
        SimpleNamespace(import_module=lambda name: pytest.fail(name)),
    )
    installed = TiRex2(checkpoint).environment_metadata()
    assert {"torch", "tirex-2", "huggingface-hub", "safetensors"} <= set(installed)
    assert all(
        value == fake_runtime.versions[name] for name, value in installed.items()
    )
    assert fake_runtime.constructors == []


def test_a_valid_checkpoint_from_another_family_is_rejected(tmp_path):
    other = {
        "model_type": "t5",
        "architectures": ["ChronosBoltModelForForecasting"],
        "chronos_pipeline_class": "ChronosBoltPipeline",
        "chronos_config": {
            "context_length": 64,
            "prediction_length": 8,
            "quantiles": [0.1, 0.5, 0.9],
        },
    }
    with pytest.raises(ValueError):
        TiRex2(write_checkpoint(tmp_path, other))


@pytest.mark.parametrize(
    "path,value",
    [
        (("act_func",), "malicious_module.Activation"),
        (("act_func",), "torch.nn.ReLU"),
        (("num_blocks",), 13),
        (("embedding_dim",), True),
        (("context_len",), 2049),
        (("future_len",), 321),
        (("input_patch_size",), 16),
        (("output_patch_size",), 16),
        (("quantiles",), [0.1, 0.5, 0.9]),
        (("dropout",), 0),
        (("scaler_cfg", "use_arcsinh"), False),
        (("scaler_cfg", "binaryaware"), False),
        (("tokenizer_cfg", "input_patch_stride"), 16),
        (
            ("stack_config", "templates", "mlstm_attn", "time_mixer", "embedding_dim"),
            1024,
        ),
        (
            ("stack_config", "templates", "slstm_attn", "time_mixer", "rnn_backend"),
            "remote",
        ),
        (("stack_config", "recipe"), ["slstm_attn"] * 12),
        (
            (
                "stack_config",
                "templates",
                "mlstm_attn",
                "variate_mixer",
                "use_flex_attention",
            ),
            True,
        ),
        (("tta_diff",), False),
        (("tta_diff",), 1),
        (("scaler_cfg", "binaryaware"), 1),
        (("input_patch_size",), 32.0),
        (("tta_sign_flip",), True),
        (("matmul_precision",), "high"),
        (("device",), "cuda"),
        (("compile",), True),
        (("auto_map",), {"RemoteModel": "remote.py"}),
        (("unknown",), False),
    ],
)
def test_unreviewed_config_rejected_before_weights_or_runtime(
    tmp_path, monkeypatch, path, value
):
    from tsfm_lab.models import checkpoints

    config = configuration()
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    monkeypatch.setattr(
        checkpoints,
        "_file",
        lambda *a, **k: pytest.fail("Weight access before profile validation"),
    )
    with pytest.raises(CheckpointError):
        write_checkpoint(tmp_path, config)


@pytest.mark.parametrize("gifteval", [False, True])
def test_profile_limits_and_median_semantics_preserve_independent_history(
    tmp_path, fake_runtime, gifteval
):
    config = configuration(gifteval)
    checkpoint = write_checkpoint(tmp_path, config)
    original = copy.deepcopy(checkpoint.config)
    model = TiRex2(checkpoint)
    context, horizon = (8192, 928) if gifteval else (2048, 320)
    assert (
        model.metadata().capabilities.max_context,
        model.metadata().capabilities.max_horizon,
    ) == (context, horizon)
    model.validate(spec(context_length=context, horizon=horizon))
    model.load()
    requests = (request("first", value=2.0), request("second", length=32, value=7.0))
    outputs = model.predict(requests)
    assert len(fake_runtime.prediction_calls) == 2
    for index, (original_request, output) in enumerate(
        zip(requests, outputs, strict=True)
    ):
        assert (
            output.request_id == original_request.request_id
            and output.series_id == original_request.series_id
        )
        assert (
            output.origin == original_request.origin
            and output.target_timestamps.equals(original_request.target_timestamps)
        )
        assert (
            output.adapter_id == "tirex_2"
            and output.point_statistic == "median"
            and output.point.dtype == np.float64
        )
        np.testing.assert_array_equal(
            output.point, original_request.context_values[-1] + 50 + np.arange(4)
        )
        assert not np.shares_memory(
            fake_runtime.tensor_calls[index], original_request.context_values
        )
        items, h, flags = fake_runtime.prediction_calls[index]
        assert h == 4 and flags == {
            "tta_sign_flip": gifteval,
            "tta_diff": not gifteval,
            "pad_context": True,
        }
        np.testing.assert_array_equal(
            items[0].target.values[0], original_request.context_values
        )
    assert checkpoint.config == original
    assert not fake_runtime.inference and fake_runtime.device_scopes == []


def test_strict_safe_load_restores_aliases_before_loading_and_is_idempotent(
    checkpoint, fake_runtime
):
    model = TiRex2(checkpoint)
    model.load()
    model.load()
    assert (
        len(fake_runtime.constructors)
        == len(fake_runtime.files)
        == len(fake_runtime.expansions)
        == len(fake_runtime.loads)
        == fake_runtime.evaluations
        == 1
    )
    assert fake_runtime.files[0] == (str(checkpoint.path / "model.safetensors"), "cpu")
    assert fake_runtime.loads[0][1] is True
    assert fake_runtime.disable_calls == [True]
    assert all(
        block().__class__ is not fake_runtime.residual_class
        for block in fake_runtime.block_objects
    )
    model.unload()
    model.unload()
    assert fake_runtime.models[-1]() is None
    with pytest.raises(RuntimeError):
        model.predict((request(),))
    model.load()
    assert len(fake_runtime.constructors) == 2


def test_cpu_eager_blocks_keep_parameters_hooks_and_global_class(
    checkpoint, fake_runtime
):
    compiled_call = fake_runtime.residual_class.__call__
    compiled_impl = fake_runtime.residual_class._call_impl
    original_forward = fake_runtime.residual_class.forward
    model = TiRex2(checkpoint)
    model.load()
    for index, reference in enumerate(fake_runtime.block_objects):
        block = reference()
        assert (id(block.parameter), id(block.buffer), id(block.hook)) == (
            fake_runtime.block_member_ids[index]
        )
        assert block.forward.__func__ is original_forward
    assert fake_runtime.residual_class.__call__ is compiled_call
    assert fake_runtime.residual_class._call_impl is compiled_impl
    assert fake_runtime.residual_class.forward is original_forward
    model.predict((request(),))
    assert fake_runtime.eager_block_calls == 2
    assert fake_runtime.compiled_block_calls == 0
    del block
    model.unload()
    assert all(reference() is None for reference in fake_runtime.block_objects)


@pytest.mark.parametrize("stage", ["alias", "load"])
def test_loading_failure_drops_model_and_does_not_predict(
    checkpoint, fake_runtime, stage
):
    error = ValueError("synthetic incomplete or conflicting weights")
    setattr(fake_runtime, "alias_error" if stage == "alias" else "load_error", error)
    model = TiRex2(checkpoint)
    with pytest.raises(ValueError):
        model.load()
    with pytest.raises(RuntimeError):
        model.predict((request(),))
    error.__traceback__ = None
    setattr(fake_runtime, "alias_error" if stage == "alias" else "load_error", None)
    assert fake_runtime.models[-1]() is None


@pytest.mark.parametrize(
    "information",
    [
        None,
        SimpleNamespace(missing_keys=["missing"], unexpected_keys=[]),
        SimpleNamespace(missing_keys=[], unexpected_keys=["extra"]),
        SimpleNamespace(missing_keys=(), unexpected_keys=[]),
        SimpleNamespace(missing_keys=[]),
    ],
)
def test_strict_load_diagnostics_cannot_be_ignored(
    checkpoint, fake_runtime, information
):
    fake_runtime.loading_info = information
    with pytest.raises(ValueError):
        TiRex2(checkpoint).load()
    assert fake_runtime.evaluations == 0


@pytest.mark.parametrize("count", [1, 3])
def test_an_unreviewed_compiled_block_layout_fails_closed(
    checkpoint, fake_runtime, count
):
    fake_runtime.block_count = count
    with pytest.raises(ValueError, match="two"):
        TiRex2(checkpoint).load()
    assert fake_runtime.evaluations == 0


def test_disabled_prediction_wrapper_failure_releases_loaded_weights(
    checkpoint, fake_runtime
):
    error = RuntimeError("synthetic disable failure")
    fake_runtime.disable_error = error
    model = TiRex2(checkpoint)
    with pytest.raises(RuntimeError, match="disable"):
        model.load()
    assert model._model is None and model._predict is None
    error.__traceback__ = None
    fake_runtime.disable_error = None
    assert fake_runtime.models[-1]() is None


@pytest.mark.parametrize("kind", ["file", "config"])
def test_checkpoint_changes_fail_before_runtime(checkpoint, monkeypatch, kind):
    model = TiRex2(checkpoint)
    if kind == "file":
        (checkpoint.path / "model.safetensors").write_bytes(b"modified")
    else:
        checkpoint.config["act_func"] = "untrusted.Activation"
    monkeypatch.setattr(adapter, "_runtime", lambda: pytest.fail("Runtime import"))
    with pytest.raises(CheckpointError):
        model.load()


@pytest.mark.parametrize("stage", ["load", "predict"])
def test_default_dtype_is_required_without_mutating_it(checkpoint, fake_runtime, stage):
    model = TiRex2(checkpoint)
    if stage == "predict":
        model.load()
    fake_runtime.default_dtype = "float64"
    with pytest.raises(ValueError, match="default|float32"):
        model.load() if stage == "load" else model.predict((request(),))
    assert fake_runtime.default_dtype == "float64" and fake_runtime.device_scopes == []
    assert fake_runtime.prediction_calls == []


@pytest.mark.parametrize(
    "requests",
    [
        (),
        (object(),),
        (request(), request()),
        (request(), request("other", horizon=3)),
        (request(length=2049),),
        (request(horizon=321),),
        (request(value=1e100),),
    ],
)
def test_invalid_request_fails_before_upstream_truncation(
    checkpoint, fake_runtime, requests
):
    model = TiRex2(checkpoint)
    model.load()
    with pytest.raises(ValueError):
        model.predict(requests)
    assert fake_runtime.prediction_calls == []


@pytest.mark.parametrize(
    "response",
    [
        [],
        [object()],
        (FakeTensor(np.zeros((1, 9, 4))),),
        [FakeTensor(np.zeros((9, 4)))],
        [FakeTensor(np.zeros((2, 9, 4)))],
        [FakeTensor(np.zeros((1, 8, 4)))],
        [FakeTensor(np.zeros((1, 9, 5)))],
        [FakeTensor(np.full((1, 9, 4), np.nan))],
        [FakeTensor(np.full((1, 9, 4), np.inf))],
        [FakeTensor(np.ones((1, 9, 4), dtype=complex))],
    ],
)
def test_malformed_prediction_rejected(checkpoint, fake_runtime, response):
    fake_runtime.response = response
    model = TiRex2(checkpoint)
    model.load()
    with pytest.raises(ValueError):
        model.predict((request(),))


@pytest.mark.parametrize("gifteval", [False, True])
def test_native_endpoints_are_preserved_without_extension(
    tmp_path, fake_runtime, gifteval
):
    model = TiRex2(write_checkpoint(tmp_path, configuration(gifteval)))
    model.load()
    context, horizon = (8192, 928) if gifteval else (2048, 320)
    result = model.predict((request(length=context, horizon=horizon),))[0]
    items, h, _ = fake_runtime.prediction_calls[0]
    assert items[0].target.values.shape == (1, context) and h == horizon
    assert result.point.shape == (horizon,)


@pytest.mark.parametrize(
    "changes", [{"context_length": 2049}, {"horizon": 321}, {"point_statistic": "mean"}]
)
def test_incompatible_spec_fails_before_load(checkpoint, changes):
    with pytest.raises(ValueError):
        TiRex2(checkpoint).validate(spec(**changes))


@pytest.mark.parametrize(
    "name,value",
    [
        ("tirex-2", "0.5.0"),
        ("torch", "2.9.0"),
        ("huggingface-hub", "0.34.4"),
        ("safetensors", "0.7.0"),
        ("numpy", "2.1.2"),
        ("numpy", "3.0.0"),
        ("flashrnn", "1.0.9"),
        ("mlstm-kernels", "2.0.5"),
        ("ninja", "1.13.1"),
        ("narwhals", "2.12.0"),
    ],
)
def test_untested_runtime_versions_fail_clearly(checkpoint, fake_runtime, name, value):
    fake_runtime.versions[name] = value
    with pytest.raises(ImportError, match="TiRex|tirex"):
        TiRex2(checkpoint).load()


def test_missing_runtime_error_is_actionable(checkpoint, monkeypatch):
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(adapter, "version", missing)
    with pytest.raises(ImportError, match="optional|tirex"):
        TiRex2(checkpoint).load()


def test_prediction_exception_exits_scopes_and_allows_cleanup(checkpoint, fake_runtime):
    fake_runtime.prediction_error = RuntimeError("synthetic prediction error")
    model = TiRex2(checkpoint)
    model.load()
    with pytest.raises(RuntimeError):
        model.predict((request(),))
    assert fake_runtime.inference is False and fake_runtime.device_scopes == []
    model.unload()
    fake_runtime.prediction_error.__traceback__ = None
    fake_runtime.prediction_error = None
    assert fake_runtime.models[-1]() is None
