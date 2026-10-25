import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsfm_lab.contracts import Dataset, EvaluationSpec, ForecastRequest
from tsfm_lab.evaluation.runner import run_experiment

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "custom_adapter.py"
_specification = importlib.util.spec_from_file_location(
    "tsfm_custom_adapter_example", EXAMPLE
)
assert _specification is not None and _specification.loader is not None
_example = importlib.util.module_from_spec(_specification)
_specification.loader.exec_module(_example)
LinearDrift = _example.LinearDrift


def trend():
    timestamps = pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC")
    dataset = Dataset(timestamps, 5 + 2 * np.arange(12), "1h", units="synthetic_units")
    evaluation = EvaluationSpec(
        context_length=4,
        horizon=3,
        stride=2,
        evaluation_start=timestamps[6],
        evaluation_end=timestamps[-1] + pd.Timedelta("1h"),
    )
    return dataset, evaluation


def test_manual_nonlinear_context_uses_endpoint_drift_and_preserves_identity():
    timestamps = pd.date_range("2026-01-01", periods=7, freq="h", tz="UTC")
    request = ForecastRequest(
        "origin-one",
        "sensor",
        timestamps[3],
        timestamps[:4],
        [1, 2, 4, 7],
        timestamps[4:],
    )
    (forecast,) = LinearDrift().predict((request,))
    # The average historical increase is two, although the last increase is three.
    np.testing.assert_array_equal(forecast.point, [9, 11, 13])
    assert forecast.request_id == "origin-one" and forecast.series_id == "sensor"
    assert forecast.origin == timestamps[3]
    assert forecast.target_timestamps.equals(timestamps[4:])
    assert forecast.point_statistic == "deterministic"
    assert forecast.point.dtype == np.float64
    assert not forecast.point.flags.writeable


def test_known_linear_trend_has_exact_aligned_forecasts_and_zero_errors():
    dataset, evaluation = trend()
    result = run_experiment(dataset, LinearDrift(), evaluation)
    np.testing.assert_array_equal(
        result.predictions.prediction, [17, 19, 21, 21, 23, 25]
    )
    assert (
        result.predictions.origin.tolist()
        == [dataset.timestamps[5]] * 3 + [dataset.timestamps[7]] * 3
    )
    assert result.predictions.target_timestamp.tolist() == [
        *dataset.timestamps[6:9],
        *dataset.timestamps[8:11],
    ]
    assert result.counts["successful_windows"] == 2
    assert result.counts["forecast_cases"] == 6
    assert result.counts["unique_target_timestamps"] == 5
    assert all(metric.value == 0 for metric in result.scores.overall.values())
    assert result.scores.scale == 2
    assert result.configuration["model"]["source"] == "local_python"
    assert result.configuration["model"]["options"]["fit_scope"] == "request_context"


def test_future_changes_leave_an_earlier_forecast_unchanged():
    dataset, evaluation = trend()
    altered = dataset.values.copy()
    altered[6:] += 100
    changed = Dataset(
        dataset.timestamps, altered, dataset.frequency, units=dataset.units
    )
    original = run_experiment(dataset, LinearDrift(), evaluation)
    modified = run_experiment(changed, LinearDrift(), evaluation)
    np.testing.assert_array_equal(
        original.predictions.prediction[:3], modified.predictions.prediction[:3]
    )
    assert not np.array_equal(
        original.predictions.actual[:3], modified.predictions.actual[:3]
    )
    assert not np.array_equal(
        original.predictions.prediction[3:], modified.predictions.prediction[3:]
    )
    assert original.scores.overall["mae"].value == 0
    assert modified.scores.overall["mae"].value > 0


def test_insufficient_context_fails_before_loading(monkeypatch):
    dataset, evaluation = trend()
    model = LinearDrift()
    monkeypatch.setattr(
        model, "load", lambda: pytest.fail("Invalid context must fail before load")
    )
    too_short = EvaluationSpec(
        1, 3, evaluation.evaluation_start, evaluation.evaluation_end
    )
    with pytest.raises(ValueError, match="minimum"):
        run_experiment(dataset, model, too_short)


def test_example_script_runs_offline_with_only_core_dependencies(tmp_path):
    root = EXAMPLE.parents[1]
    program = """
import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path,target=None):
  if fullname.split('.')[0] in {'torch','chronos','huggingface_hub','typer','rich','textual','prompt_toolkit'}:
   raise ModuleNotFoundError(fullname,name=fullname)
sys.meta_path.insert(0,Block())
runpy.run_path(sys.argv[1],run_name='__main__')
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(EXAMPLE)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["purpose"].startswith("Educational")
    result = payload["experiment"]
    assert result["configuration"]["model"]["adapter_id"] == "linear_drift_example"
    assert result["configuration"]["dataset"]["target_name"] == "demand"
    assert result["configuration"]["dataset"]["units"] == "synthetic_units"
    assert result["counts"]["forecast_cases"] == len(result["predictions"]) == 16
    assert completed.stderr == ""
    assert list(tmp_path.iterdir()) == []
