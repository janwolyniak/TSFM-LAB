"""Opt-in Chronos-2 CPU inference from already-present weights; never downloads."""

import os
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest


@pytest.mark.chronos2_smoke
def test_chronos2_local_or_cached_cpu_smoke(record_property):
    local = os.environ.get("TSFM_CHRONOS2_CHECKPOINT")
    cache = os.environ.get("TSFM_CHRONOS2_CACHE")
    revision = os.environ.get("TSFM_CHRONOS2_REVISION")
    if not local and not (cache and revision):
        pytest.skip(
            "Set TSFM_CHRONOS2_CHECKPOINT or exact TSFM_CHRONOS2_CACHE/REVISION"
        )

    from tsfm_lab.contracts import Dataset, EvaluationSpec
    from tsfm_lab.evaluation.runner import run_experiment
    from tsfm_lab.models.checkpoints import inspect_local_checkpoint, resolve_checkpoint
    from tsfm_lab.models.chronos2 import Chronos2

    checkpoint = (
        inspect_local_checkpoint(local)
        if local
        else resolve_checkpoint(
            "autogluon/chronos-2-small", revision, cache_dir=cache, offline=True
        )
    )
    timestamps = pd.date_range("2026-01-01", periods=37, freq="h", tz="UTC")
    values = 10 + np.sin(np.arange(37) / 4) + np.arange(37) / 20
    spec = EvaluationSpec(32, 4, timestamps[32], timestamps[36])
    dataset = Dataset(timestamps, values, "1h", units="synthetic_units")
    model = Chronos2(checkpoint)
    result = run_experiment(dataset, model, spec)
    replay = run_experiment(dataset, model, spec)
    changed_future = values.copy()
    changed_future[32:] += 1000
    causal = run_experiment(
        Dataset(timestamps, changed_future, "1h", units="synthetic_units"), model, spec
    )
    assert result.counts["successful_windows"] == 1
    assert result.counts["forecast_cases"] == 4
    assert set(result.predictions.point_statistic) == {"median"}
    assert np.isfinite(result.predictions.prediction).all()
    assert result.predictions.target_timestamp.tolist() == timestamps[32:36].tolist()
    np.testing.assert_array_equal(
        result.predictions.prediction, replay.predictions.prediction
    )
    np.testing.assert_array_equal(
        result.predictions.prediction, causal.predictions.prediction
    )
    assert model._pipeline is None and model._torch is None
    record_property("checkpoint_fingerprint", checkpoint.fingerprint)
    record_property("checkpoint_revision", checkpoint.resolved_revision or "local")
    record_property("repeat_and_future_isolation", "exactly equal forecasts")
    for name in ("chronos-forecasting", "torch", "transformers", "huggingface-hub"):
        record_property(name, version(name))
