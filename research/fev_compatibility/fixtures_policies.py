"""Synthetic compatibility fixtures for the isolated fev research companion."""

import math

import numpy as np
import pandas as pd
from study_support import (
    clean_json,
    convert,
    expect_error,
    row,
    task,
)

from tsfm_lab.contracts import Dataset
from tsfm_lab.evaluation.metrics import score_forecasts


def undefined_and_nonfinite():
    candidate = task(
        "zero_scale_panel",
        [row("constant", [1] * 8), row("trend", range(8))],
        horizon=2,
        initial_cutoff=6,
        max_context_length=4,
        seasonality=1,
        extra_metrics=["MASE"],
    )
    native = convert(candidate)
    constant = score_forecasts([[1.0, 1.0]], [[1.0, 1.0]], [1.0] * 6, seasonal_period=1)
    assert (
        constant.overall["mase"].value is None and constant.scale_reason == "zero_scale"
    )
    assert native["summary"]["MASE"] == 1.5
    bad_forecasts = [
        expect_error(
            lambda value=value: candidate.clean_and_validate_predictions(
                [{"predictions": [value, 1.0]}]
            )
        )
        for value in [float("nan"), float("inf")]
    ]
    constant_task = task(
        "zero_scale_only",
        [row("constant", [1] * 8)],
        horizon=2,
        initial_cutoff=6,
        max_context_length=4,
        seasonality=1,
        extra_metrics=["MASE"],
    )
    all_undefined = convert(constant_task)
    assert math.isnan(all_undefined["summary"]["MASE"])
    insufficient = task(
        "insufficient_scale",
        [row("short_scale", [1.0, 2.0, 3.0, 4.0])],
        horizon=2,
        initial_cutoff=1,
        seasonality=4,
        extra_metrics=["MASE"],
    )
    insufficient_native = convert(insufficient)
    strict_insufficient = score_forecasts(
        [[2.0, 3.0]], [[1.0, 1.0]], [1.0], seasonal_period=4
    )
    assert math.isnan(insufficient_native["summary"]["MASE"])
    assert strict_insufficient.scale_reason == "insufficient_scale_history"
    invalid = task(
        "nonfinite_truth",
        [row("nan_target", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, float("nan")])],
        horizon=2,
        initial_cutoff=6,
        max_context_length=4,
    )
    native_invalid = convert(invalid)
    assert native_invalid["summary"]["MAE"] == 1.0
    strict_input = expect_error(
        lambda: Dataset(
            pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC"),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, float("nan")],
            "1h",
        )
    )
    return {
        "zero_scale_panel": native,
        "constant_tsfm_mase": None,
        "constant_tsfm_reason": constant.scale_reason,
        "all_undefined_fev_mase": None,
        "all_undefined_native_value": "NaN",
        "insufficient_fev_mase": None,
        "insufficient_fev_native_value": "NaN",
        "insufficient_tsfm_reason": strict_insufficient.scale_reason,
        "nonfinite_forecast_rejections": bad_forecasts,
        "nonfinite_actual_native_summary": native_invalid,
        "strict_tsfm_input_rejection": strict_input,
    }


def overlapping_identities():
    candidate = task(
        "overlap",
        [row("overlap", range(12))],
        horizon=3,
        num_windows=3,
        initial_cutoff=6,
        window_step_size=1,
        min_context_length=4,
        max_context_length=4,
    )
    native = convert(candidate)
    keys = [
        (x["series_id"], x["origin"], x["target_timestamp"], x["lead"])
        for x in native["cases"]
    ]
    expected = [
        (
            "overlap",
            pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=start - 1),
            pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=start + lead - 1),
            lead,
        )
        for start in [6, 7, 8]
        for lead in [1, 2, 3]
    ]
    canonical = [(a, b.isoformat(), c.isoformat(), d) for a, b, c, d in expected]
    assert keys == canonical and len(set(keys)) == 9
    unique = len({x["target_timestamp"] for x in native["cases"]})
    assert unique == 5
    return {
        "fev": native,
        "forecast_cases": len(keys),
        "unique_target_timestamps": unique,
        "expected_keys": canonical,
        "overlap_preserved": True,
    }


def covariate_availability():
    base = row(
        "covariates",
        range(12),
        calendar=list(range(12)),
        measured=list(range(100, 112)),
    )
    options = dict(
        horizon=2,
        initial_cutoff=6,
        min_context_length=4,
        max_context_length=4,
        known_dynamic_columns=["calendar"],
        past_dynamic_columns=["measured"],
    )
    first = task("covariates", [base], **options)
    past, future = first.get_window(0, num_proc=1).get_input_data()
    changed = {**base, "measured": base["measured"][:6] + [9999] * 6}
    second = task("covariates_changed", [changed], **options)
    new_past, new_future = second.get_window(0, num_proc=1).get_input_data()
    assert clean_json(past.to_dict()) == clean_json(new_past.to_dict())
    assert clean_json(future.to_dict()) == clean_json(new_future.to_dict())
    assert "measured" not in future.column_names and "target" not in future.column_names
    misclassified = task(
        "covariates_misclassified",
        [base],
        horizon=2,
        initial_cutoff=6,
        max_context_length=4,
        known_dynamic_columns=["calendar", "measured"],
    )
    _, unsafe_future = misclassified.get_window(0, num_proc=1).get_input_data()
    assert "measured" in unsafe_future.column_names

    def bridge_guard():
        available = {"calendar": "known", "measured": "past_only"}
        for name in misclassified.known_dynamic_columns:
            if available[name] != "known":
                raise ValueError(f"{name} was unavailable at the forecast origin")

    error = expect_error(bridge_guard)
    return {
        "past_fields": past.column_names,
        "future_fields": future.column_names,
        "past_input": past.to_dict(),
        "future_input": future.to_dict(),
        "future_measured_perturbation_leaves_inputs_equal": True,
        "fev_accepts_declared_future_measured_column": True,
        "prototype_availability_guard": error,
    }


def quantile_normalization():
    candidate = task(
        "quantiles",
        [row("quantile", [0.0, 1.0, 1.0, 1.0])],
        horizon=2,
        initial_cutoff=2,
        quantile_levels=[0.9, 0.1, 0.5],
        extra_metrics=["MQL"],
    )
    predictions = [
        {
            "predictions": [0.5, 0.5],
            "0.1": [0.0, 0.0],
            "0.5": [0.5, 0.5],
            "0.9": [2.0, 2.0],
        }
    ]
    summary = candidate.evaluation_summary(
        [predictions], model_name="hand_quantiles", per_quantile_scores=True
    )
    assert candidate.quantile_levels == [0.1, 0.5, 0.9]
    pinball = [0.1, 0.25, 0.1]
    assert math.isclose(summary["MQL"], 2 * np.mean(pinball), abs_tol=1e-12)
    bad = [{key: value for key, value in predictions[0].items() if key != "0.9"}]
    missing = expect_error(lambda: candidate.clean_and_validate_predictions(bad))
    crossing = [
        {
            "predictions": [0.5, 0.5],
            "0.1": [2.0, 2.0],
            "0.5": [0.5, 0.5],
            "0.9": [0.0, 0.0],
        }
    ]
    candidate.clean_and_validate_predictions(crossing)

    def crossing_guard():
        matrix = np.array([crossing[0][str(q)] for q in candidate.quantile_levels]).T
        if (np.diff(matrix, axis=1) < 0).any():
            raise ValueError("Crossing quantiles require an explicit policy")

    return {
        "fev_summary": summary,
        "ordinary_pinball_per_level": pinball,
        "ordinary_mean_pinball": float(np.mean(pinball)),
        "fev_mql": summary["MQL"],
        "unordered_input_levels_sorted_by_fev": True,
        "missing_quantile_rejection": missing,
        "crossing_quantiles_pass_native_validation": True,
        "prototype_crossing_guard": expect_error(crossing_guard),
    }
