"""Opt-in CPU inference using existing local weights; never downloads a model."""

import os
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest


@pytest.mark.chronos_smoke
def test_chronos_bolt_local_cpu_smoke(record_property):
    path = os.environ.get("TSFM_CHRONOS_CHECKPOINT")
    if not path:
        pytest.skip(
            "Set TSFM_CHRONOS_CHECKPOINT to an authorized local Bolt checkpoint"
        )

    from tsfm_lab.contracts import Dataset, EvaluationSpec
    from tsfm_lab.evaluation.runner import run_experiment
    from tsfm_lab.models.checkpoints import inspect_local_checkpoint
    from tsfm_lab.models.chronos_bolt import ChronosBolt

    checkpoint = inspect_local_checkpoint(path)
    timestamps = pd.date_range("2026-01-01", periods=37, freq="h", tz="UTC")
    values = 10 + np.sin(np.arange(37) / 4) + np.arange(37) / 20
    result = run_experiment(
        Dataset(timestamps, values, "1h", units="synthetic_units"),
        ChronosBolt(checkpoint),
        EvaluationSpec(32, 4, timestamps[32], timestamps[36]),
    )
    assert result.counts["successful_windows"] == 1
    assert result.counts["forecast_cases"] == 4
    assert set(result.predictions.point_statistic) == {"median"}
    assert np.isfinite(result.predictions.prediction).all()
    assert result.predictions.target_timestamp.tolist() == timestamps[32:36].tolist()
    record_property("checkpoint_fingerprint", checkpoint.fingerprint)
    record_property("checkpoint_revision", checkpoint.resolved_revision or "local")
    for name in ("chronos-forecasting", "torch", "transformers", "huggingface-hub"):
        record_property(name, version(name))
