"""Synthetic compatibility fixtures for the isolated fev research companion."""

import math

import numpy as np
import pandas as pd
from study_support import (
    SOURCE,
    convert,
    expect_error,
    last_value,
    row,
    seasonal,
    task,
)

from tsfm_lab.contracts import EvaluationSpec
from tsfm_lab.data import load_csv
from tsfm_lab.evaluation.metrics import score_forecasts
from tsfm_lab.evaluation.runner import run_experiment
from tsfm_lab.models.baselines import LastValue, SeasonalNaive


def baseline_equivalence():
    dataset = load_csv(
        SOURCE / "examples/datasets/seasonal.csv",
        timestamp_column="timestamp",
        target_column="demand",
        frequency="1h",
        units="synthetic_units",
    )
    spec = EvaluationSpec(
        context_length=8,
        horizon=4,
        stride=4,
        evaluation_start=dataset.timestamps[16],
        evaluation_end=dataset.timestamps[-1] + pd.Timedelta("1h"),
        seasonal_period=4,
    )
    candidate = task(
        "bundled_series",
        [row("series_0", dataset.values)],
        horizon=4,
        num_windows=4,
        initial_cutoff=16,
        window_step_size=4,
        min_context_length=8,
        max_context_length=8,
        seasonality=4,
        extra_metrics=["RMSE", "MASE"],
    )
    results = {}
    for name, adapter, forecast, expected in [
        ("last_value", LastValue(), last_value, (2.5, math.sqrt(8.5), 2.5)),
        ("seasonal_naive", SeasonalNaive(4), seasonal, (1.0, 1.0, 1.0)),
    ]:
        native = convert(candidate, forecast)
        core = run_experiment(dataset, adapter, spec)
        columns = [
            "series_id",
            "origin",
            "target_timestamp",
            "lead",
            "actual",
            "prediction",
        ]
        left = core.predictions[columns].copy()
        right = pd.DataFrame(native["cases"])[columns]
        for key in ["origin", "target_timestamp"]:
            left[key] = pd.to_datetime(left[key], utc=True)
            right[key] = pd.to_datetime(right[key], utc=True)
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True), right.reset_index(drop=True), check_dtype=False
        )
        assert len(right) == 16
        for key, value in zip(["MAE", "RMSE", "MASE"], expected, strict=True):
            assert math.isclose(native["summary"][key], value, rel_tol=1e-12)
            assert math.isclose(
                core.scores.overall[key.lower()].value, value, rel_tol=1e-12
            )
        results[name] = {
            "fev": native,
            "tsfm_metrics": {k: v.value for k, v in core.scores.overall.items()},
            "case_keys_and_predictions_equal": True,
        }
    return results


def changing_scale_and_pooling():
    values = np.array([0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66], dtype=float)
    candidate = task(
        "changing_scale",
        [row("growing", values)],
        horizon=1,
        num_windows=3,
        initial_cutoff=4,
        window_step_size=2,
        min_context_length=4,
        max_context_length=4,
        seasonality=1,
        extra_metrics=["MASE", "RMSE"],
    )
    native = convert(candidate)
    actual = np.array([x["actual"] for x in native["cases"]]).reshape(3, 1)
    predicted = np.array([x["prediction"] for x in native["cases"]]).reshape(3, 1)
    scores = score_forecasts(actual, predicted, values[:4], seasonal_period=1)
    scales = [
        float(np.mean(np.abs(np.diff(window[0])))) for window in native["histories"]
    ]
    assert scales == [2.0, 4.0, 6.0]
    assert math.isclose(native["summary"]["MASE"], 29 / 18)
    assert scores.scale == 2 and scores.overall["mase"].value == 3
    assert native["summary"]["RMSE"] == 6
    assert math.isclose(scores.overall["rmse"].value, math.sqrt(116 / 3))
    return {
        "fev": native,
        "rolling_context_scales": scales,
        "tsfm_fixed_scale": scores.scale,
        "tsfm_mase": scores.overall["mase"].value,
        "fev_window_averaged_rmse": native["summary"]["RMSE"],
        "tsfm_pooled_rmse": scores.overall["rmse"].value,
    }


def unequal_series_weighting():
    candidate = task(
        "unequal_series",
        [row("long", [0] * 10), row("short", [0] * 7)],
        horizon=2,
        num_windows=2,
        initial_cutoff=5,
        window_step_size=2,
        max_context_length=4,
    )

    def forecast(past, horizon):
        return np.repeat(10.0 if past["id"] == "short" else 0.0, horizon)

    native = convert(candidate, forecast, require_all_items=False)
    assert native["item_ids"] == [["long", "short"], ["long"]]
    frame = pd.DataFrame(native["cases"])
    frame["error"] = abs(frame.actual - frame.prediction)
    pooled = float(frame.error.mean())
    macro = float(frame.groupby("series_id").error.mean().mean())
    assert len(frame) == 6 and math.isclose(pooled, 10 / 3)
    assert macro == 5 and native["summary"]["MAE"] == 2.5
    guard = expect_error(lambda: convert(candidate, forecast))
    return {
        "fev": native,
        "case_weighted_mae": pooled,
        "series_weighted_mae": macro,
        "fev_equal_window_mae": native["summary"]["MAE"],
        "strict_bridge_coverage_guard": guard,
    }
