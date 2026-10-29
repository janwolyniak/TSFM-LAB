"""TOTO 2 strict loading, bounded medians and fail-closed numerical behavior."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from toto2_helpers import CONFIG, Tensor, write_checkpoint
from toto2_helpers import fake_toto as shared_toto

from tsfm_lab.contracts import EvaluationSpec, ForecastRequest
from tsfm_lab.models import toto2, toto2_profile
from tsfm_lab.models.checkpoints import CheckpointError, validate_checkpoint_record

fake_toto = shared_toto


@pytest.fixture
def checkpoint(tmp_path):
    return write_checkpoint(tmp_path / "weights")


def request(*, context=32, horizon=8, values=None, name="request"):
    stamps = pd.date_range("2026-01-01", periods=context + horizon, freq="h", tz="UTC")
    return ForecastRequest(
        name,
        "series",
        stamps[context - 1],
        stamps[:context],
        np.arange(context) if values is None else values,
        stamps[context:],
    )


def spec(context=32, horizon=8, task="median"):
    return EvaluationSpec(
        context, horizon, "2026-01-01T00:00Z", "2026-01-02T00:00Z", point_statistic=task
    )


def test_configuration_identity_record_and_import_isolation(checkpoint):
    assert (
        hashlib.sha256(CONFIG.read_bytes()).hexdigest()
        == "7a926d130e401ab0c5fdb3564f46c8d917bd05c7b3ae26b9c22d2da2ef01d2d8"
    )
    record = checkpoint.record()
    assert validate_checkpoint_record(record) == record
    assert record["schema_version"] == 1
    assert (record["architecture"], record["model_type"]) == ("Toto2Model", "toto2")
    assert record["quantiles"] == toto2_profile.QUANTILES
    source = Path(__file__).resolve().parents[1] / "src"
    program = """
import sys
from tsfm_lab.models.catalog import catalog_info
from tsfm_lab.models.toto2 import Toto2
assert catalog_info('toto_2')['point_statistic']=='median'
assert not {'torch','toto2','gluonts','lightning','dd_unit_scaling','safetensors'}.intersection(sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", program],
        check=True,
        env={**os.environ, "PYTHONPATH": str(source)},
    )


@pytest.mark.parametrize(
    "mutation", ["unknown", "patch", "dimension", "type", "architecture"]
)
def test_complete_unreviewed_configs_fail_before_weight_access(
    tmp_path, monkeypatch, mutation
):
    config = json.loads(CONFIG.read_bytes())
    config[
        {
            "unknown": "unreviewed",
            "patch": "patch_size",
            "dimension": "d_model",
            "type": "num_heads",
            "architecture": "architectures",
        }[mutation]
    ] = {
        "unknown": True,
        "patch": 16,
        "dimension": 512,
        "type": 4.0,
        "architecture": ["Toto2Model"],
    }[
        mutation
    ]
    monkeypatch.setattr(
        "tsfm_lab.models.checkpoints._file",
        lambda *a, **kw: pytest.fail("weight access"),
    )
    with pytest.raises(CheckpointError):
        write_checkpoint(tmp_path / "weights", config)


def test_loading_bypasses_hub_and_requires_strict_safe_weights(checkpoint, fake_toto):
    model = toto2.Toto2(checkpoint, seed=8)
    model.load()
    model.load()
    assert fake_toto.constructors == [json.loads(CONFIG.read_bytes())]
    assert fake_toto.weight_reads == [str(checkpoint.path / "model.safetensors")]
    assert fake_toto.load_calls[0][1] is True
    assert fake_toto.casts == [{"device": "cpu", "dtype": "float32"}]
    assert fake_toto.evaluations == 1
    options = model.runtime_configuration()
    assert options["seed"] == 8 and options["seed_used"] is False
    assert options["torch_compile"] is False
    options["forecast_options"]["decode_block_size"] = 0
    assert model.runtime_configuration()["forecast_options"]["decode_block_size"] == 32
    model.unload()
    assert fake_toto.models[0]() is None


@pytest.mark.parametrize("failure", ["missing", "unexpected", "shape", "knots"])
def test_incompatible_native_weights_or_runtime_knots_release_model(
    checkpoint, fake_toto, failure
):
    if failure == "shape":
        fake_toto.load_error = RuntimeError("shape mismatch")
    elif failure == "knots":
        fake_toto.knots = [0.5]
    else:
        setattr(fake_toto, failure, ["wrong_parameter"])
    model = toto2.Toto2(checkpoint)
    with pytest.raises((ValueError, RuntimeError)):
        model.load()
    assert model._model is model._torch is None
    if fake_toto.load_error:
        fake_toto.load_error.__traceback__ = None
    assert fake_toto.models[0]() is None


@pytest.mark.parametrize("context,horizon", [(32, 1), (64, 33), (2048, 128)])
def test_original_units_median_inputs_masks_and_fixed_decoding(
    checkpoint, fake_toto, context, horizon
):
    model = toto2.Toto2(checkpoint)
    model.validate(spec(context, horizon))
    model.load()
    item = request(context=context, horizon=horizon, values=np.arange(context) / 4 - 9)
    output = model.predict((item,))[0]
    expected = np.float32(item.context_values[-1]) + 4 + np.arange(horizon)
    np.testing.assert_array_equal(
        output.point, expected.astype(np.float32).astype(np.float64)
    )
    inputs, actual_horizon, options = fake_toto.forecasts[0]
    assert inputs["target"].values.shape == (1, 1, context)
    np.testing.assert_array_equal(
        inputs["target"].values[0, 0], item.context_values.astype(np.float32)
    )
    assert not np.shares_memory(fake_toto.targets[0], item.context_values)
    assert (
        inputs["target_mask"].values.all()
        and inputs["target_mask"].values.dtype == bool
    )
    assert inputs["series_ids"].values.tolist() == [[0]]
    assert actual_horizon == horizon
    assert options == {
        "decode_block_size": 32,
        "has_missing_values": False,
        "scaler_fallback_min_obs": 0,
        "quantile_real_cap_k": 0.0,
    }
    assert output.point_statistic == "median"
    assert output.origin == item.origin and output.target_timestamps.equals(
        item.target_timestamps
    )
    assert not fake_toto.inference and fake_toto.devices == []


@pytest.mark.parametrize("context,horizon", [(31, 8), (33, 8), (2049, 8), (32, 129)])
def test_unsupported_requests_fail_before_tensor_or_forecast_allocation(
    checkpoint, fake_toto, context, horizon
):
    model = toto2.Toto2(checkpoint)
    with pytest.raises(ValueError):
        model.validate(spec(context, horizon))
    model.load()
    with pytest.raises(ValueError):
        model.predict((request(context=context, horizon=horizon),))
    assert fake_toto.forecasts == fake_toto.targets == []


@pytest.mark.parametrize(
    "response",
    [
        None,
        np.zeros((1, 9, 1, 8)),
        np.full((9, 1, 1, 8), np.inf),
        np.full((9, 1, 1, 8), np.nan),
        np.zeros((9, 1, 1, 8), dtype=complex),
        np.repeat(np.arange(9, 0, -1)[:, None, None, None], 8, axis=-1),
    ],
)
def test_full_native_output_is_validated_including_nonmedian_quantiles(
    checkpoint, fake_toto, response
):
    fake_toto.response = object() if response is None else Tensor(response)
    model = toto2.Toto2(checkpoint)
    model.load()
    with pytest.raises(ValueError):
        model.predict((request(),))


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_upstream_nonfinite_repair_is_overridden_to_refuse(
    checkpoint, fake_toto, value
):
    model = toto2.Toto2(checkpoint)
    model.load()
    original = Tensor(np.zeros((9, 1, 1, 8)))
    assert model._model._clamp_nonfinite(original) is original
    original.values[0, 0, 0, 0] = value
    with pytest.raises(ValueError, match="nonfinite"):
        model._model._clamp_nonfinite(original)


@pytest.mark.parametrize("stage", ["load", "predict"])
def test_default_dtype_is_refused_without_mutation(checkpoint, fake_toto, stage):
    model = toto2.Toto2(checkpoint)
    if stage == "predict":
        model.load()
    fake_toto.dtype = "float64"
    with pytest.raises(ValueError, match="default Torch dtype"):
        model.load() if stage == "load" else model.predict((request(),))
    assert fake_toto.dtype == "float64" and fake_toto.forecasts == []


@pytest.mark.parametrize(
    "package,value",
    [
        ("toto-2", "2.0.1"),
        ("gluonts", "0.17.0"),
        ("torch", "2.10.0"),
        ("numpy", "2.2.0"),
        ("scipy", "1.16.0"),
    ],
)
def test_incompatible_runtime_stops_before_optional_import(
    checkpoint, fake_toto, package, value
):
    fake_toto.versions[package] = value
    with pytest.raises(ImportError, match="requirements-toto2"):
        toto2.Toto2(checkpoint).load()
    assert fake_toto.imports == []


@pytest.mark.parametrize("python", [(3, 11), (3, 14)])
def test_python_restriction_stops_load_and_transfer_before_optional_imports(
    checkpoint, monkeypatch, python
):
    monkeypatch.setattr(toto2_profile, "sys", SimpleNamespace(version_info=python))
    monkeypatch.setattr(toto2, "version", lambda name: pytest.fail("package metadata"))
    from tsfm_lab.models import checkpoints
    from tsfm_lab.models.preparation import plan_checkpoint

    monkeypatch.setattr(checkpoints, "_hub", lambda: pytest.fail("network/import"))
    for action in (
        toto2.Toto2(checkpoint).load,
        lambda: checkpoints.resolve_checkpoint(
            "Datadog/Toto-2.0-4m", "main", offline=False
        ),
        lambda: checkpoints.pull_checkpoint("Datadog/Toto-2.0-4m"),
        lambda: plan_checkpoint("Datadog/Toto-2.0-4m", "main"),
    ):
        with pytest.raises(ImportError, match="Python 3.12 or 3.13"):
            action()


def test_checkpoint_mutation_precedes_runtime_import(checkpoint, monkeypatch):
    monkeypatch.setattr(toto2_profile, "sys", SimpleNamespace(version_info=(3, 12)))
    model = toto2.Toto2(checkpoint)
    (checkpoint.path / "model.safetensors").write_bytes(b"changed")
    monkeypatch.setattr(toto2, "_runtime", lambda: pytest.fail("runtime"))
    with pytest.raises(CheckpointError):
        model.load()


def test_missing_optional_runtime_is_actionable_without_imports(
    checkpoint, fake_toto, monkeypatch
):
    def missing(name):
        raise toto2.PackageNotFoundError(name)

    monkeypatch.setattr(toto2, "version", missing)
    with pytest.raises(ImportError, match="optional tsfm-lab"):
        toto2.Toto2(checkpoint).load()
    assert fake_toto.imports == []


def test_configuration_and_runtime_snapshots_are_detached(checkpoint):
    model = toto2.Toto2(checkpoint)
    snapshot = copy.deepcopy(model.configuration())
    snapshot["checkpoint"]["quantiles"] = [0.5]
    assert model.configuration()["checkpoint"]["quantiles"] == toto2_profile.QUANTILES
