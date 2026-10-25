"""Opt-in CPU integration using existing local or cached TimesFM 2.5 weights."""

import os
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest


@pytest.mark.timesfm_smoke
def test_timesfm_local_cpu_smoke(record_property):
    path = os.environ.get("TSFM_TIMESFM_CHECKPOINT")
    revision = os.environ.get("TSFM_TIMESFM_REVISION")
    if not path and not revision:
        pytest.skip(
            "Set TSFM_TIMESFM_CHECKPOINT or TSFM_TIMESFM_REVISION for existing weights"
        )
    if path and revision:
        pytest.fail("Choose a standalone checkpoint or an exact cached revision")

    from tsfm_lab.contracts import Dataset, EvaluationSpec
    from tsfm_lab.evaluation.runner import run_experiment
    from tsfm_lab.models.checkpoints import inspect_local_checkpoint, resolve_checkpoint
    from tsfm_lab.models.timesfm import TimesFM

    checkpoint = (
        inspect_local_checkpoint(path)
        if path
        else resolve_checkpoint(
            "google/timesfm-2.5-200m-pytorch",
            revision,
            cache_dir=os.environ.get("TSFM_TIMESFM_CACHE_DIR"),
            offline=True,
        )
    )
    times = pd.date_range("2026-01-01", periods=37, freq="h", tz="UTC")
    values = 10 + np.sin(np.arange(37) / 4) + np.arange(37) / 20
    result = run_experiment(
        Dataset(times, values, "1h", units="synthetic_units"),
        TimesFM(checkpoint),
        EvaluationSpec(32, 4, times[32], times[36]),
    )
    assert result.counts["successful_windows"] == 1
    assert result.counts["forecast_cases"] == 4
    assert set(result.predictions.point_statistic) == {"median"}
    assert np.isfinite(result.predictions.prediction).all()
    assert result.predictions.target_timestamp.tolist() == times[32:36].tolist()
    record_property("checkpoint_fingerprint", checkpoint.fingerprint)
    record_property("checkpoint_revision", checkpoint.resolved_revision or "local")
    for name in ("timesfm", "torch", "huggingface-hub", "safetensors"):
        record_property(name, version(name))
