"""Opt-in TiRex-2 CPU inference from authorized existing files; never downloads."""

import importlib
import os
import socket
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest


@pytest.mark.tirex2_smoke
def test_tirex2_local_or_cached_cpu_smoke(record_property, monkeypatch):
    local = os.environ.get("TSFM_TIREX2_CHECKPOINT")
    cache = os.environ.get("TSFM_TIREX2_CACHE")
    revision = os.environ.get("TSFM_TIREX2_REVISION")
    if not local and not (cache and revision):
        pytest.skip("Set TSFM_TIREX2_CHECKPOINT or exact TSFM_TIREX2_CACHE/REVISION")
    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append("network")
        raise AssertionError("Network access is prohibited during model smoke")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)

    torch = importlib.import_module("torch")
    dynamo = importlib.import_module("torch._dynamo.convert_frame")
    compiler_attempts = []

    def forbidden_compile(*args, **kwargs):
        compiler_attempts.append("torch._dynamo.convert_frame._compile")
        raise AssertionError("Compilation is prohibited during CPU model smoke")

    # Guard actual compiler entry before importing the optional upstream model.
    monkeypatch.setattr(dynamo, "_compile", forbidden_compile)
    initial_dtype = torch.get_default_dtype()
    initial_matmul_precision = torch.get_float32_matmul_precision()

    from tsfm_lab.contracts import Dataset, EvaluationSpec
    from tsfm_lab.evaluation.runner import run_experiment
    from tsfm_lab.models.checkpoints import inspect_local_checkpoint, resolve_checkpoint
    from tsfm_lab.models.tirex2 import TiRex2

    checkpoint = (
        inspect_local_checkpoint(local)
        if local
        else resolve_checkpoint(
            "NX-AI/TiRex-2", revision, cache_dir=cache, offline=True
        )
    )
    stamps = pd.date_range("2026-01-01", periods=37, freq="h", tz="UTC")
    values = 10 + np.sin(np.arange(37) / 4) + np.arange(37) / 20
    spec = EvaluationSpec(32, 4, stamps[32], stamps[36])
    model = TiRex2(checkpoint)
    dataset = Dataset(stamps, values, "1h", units="synthetic_units")
    result = run_experiment(dataset, model, spec)
    replay = run_experiment(dataset, model, spec)
    future = values.copy()
    future[32:] += 1000
    causal = run_experiment(
        Dataset(stamps, future, "1h", units="synthetic_units"), model, spec
    )
    assert (
        result.counts["successful_windows"] == 1
        and result.counts["forecast_cases"] == 4
    )
    assert set(result.predictions.point_statistic) == {"median"}
    assert np.isfinite(result.predictions.prediction).all()
    assert result.predictions.target_timestamp.tolist() == stamps[32:36].tolist()
    np.testing.assert_array_equal(
        result.predictions.prediction, replay.predictions.prediction
    )
    np.testing.assert_array_equal(
        result.predictions.prediction, causal.predictions.prediction
    )
    assert (
        model._model is None
        and model._torch is None
        and model._series_type is None
        and model._predict is None
    )
    assert attempts == []
    assert compiler_attempts == []
    assert torch.get_default_dtype() == initial_dtype
    assert torch.get_float32_matmul_precision() == initial_matmul_precision
    record_property("checkpoint_fingerprint", checkpoint.fingerprint)
    record_property("checkpoint_revision", checkpoint.resolved_revision or "local")
    record_property("repeat_and_future_isolation", "exactly equal forecasts")
    record_property("network_attempts", str(attempts))
    record_property("compiler_attempts", str(compiler_attempts))
    record_property("default_dtype_unchanged", True)
    record_property("matmul_precision_unchanged", True)
    for name in (
        "tirex-2",
        "torch",
        "huggingface-hub",
        "safetensors",
        "flashrnn",
        "mlstm-kernels",
    ):
        record_property(name, version(name))
