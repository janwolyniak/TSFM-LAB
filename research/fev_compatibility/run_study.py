"""Run the isolated, synthetic fev compatibility study into a fresh directory."""

import argparse
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def concise_results(fixtures):
    data = {name: value["evidence"] for name, value in fixtures.items()}
    baseline = data["baseline_equivalence"]
    scale = data["changing_scale_and_pooling"]
    unequal = data["unequal_series_weighting"]
    invalid = data["undefined_and_nonfinite"]
    overlap = data["overlapping_identities"]
    covariates = data["covariate_availability"]
    quantiles = data["quantile_normalization"]
    return {
        "baseline_equivalence": {
            name: {
                "case_keys_and_predictions_equal": result[
                    "case_keys_and_predictions_equal"
                ],
                "forecast_cases": len(result["fev"]["cases"]),
                "fev_num_forecasts": result["fev"]["summary"]["num_forecasts"],
                "metrics": result["tsfm_metrics"],
            }
            for name, result in baseline.items()
        },
        "changing_scale_and_pooling": {
            "rolling_context_scales": scale["rolling_context_scales"],
            "tsfm_fixed_scale": scale["tsfm_fixed_scale"],
            "fev_mase": scale["fev"]["summary"]["MASE"],
            "tsfm_mase": scale["tsfm_mase"],
            "fev_window_averaged_rmse": scale["fev_window_averaged_rmse"],
            "tsfm_pooled_rmse": scale["tsfm_pooled_rmse"],
        },
        "unequal_series_weighting": {
            "item_ids_by_window": unequal["fev"]["item_ids"],
            "forecast_cases": len(unequal["fev"]["cases"]),
            "fev_equal_window_mae": unequal["fev_equal_window_mae"],
            "case_weighted_mae": unequal["case_weighted_mae"],
            "series_weighted_mae": unequal["series_weighted_mae"],
            "prototype_coverage_guard_rejected_missing_item": True,
        },
        "undefined_and_nonfinite": {
            "fev_mase_after_undefined_constant_item_excluded": invalid[
                "zero_scale_panel"
            ]["summary"]["MASE"],
            "undefined_constant_items_excluded_by_fev": 1,
            "defined_items_in_zero_scale_panel": 1,
            "tsfm_constant_mase": {
                "value": None,
                "reason": invalid["constant_tsfm_reason"],
                "series_count": 1,
            },
            "fev_all_undefined_mase": {
                "value": None,
                "reason": "zero_scale_for_all_items",
                "native_value": "NaN",
                "valid_items": 0,
                "excluded_items": 1,
            },
            "tsfm_insufficient_history_mase": {
                "value": None,
                "reason": invalid["insufficient_tsfm_reason"],
                "series_count": 1,
            },
            "fev_insufficient_history_mase": {
                "value": None,
                "reason": "insufficient_scale_history_for_all_items",
                "native_value": "NaN",
                "valid_items": 0,
                "excluded_items": 1,
            },
            "nonfinite_forecasts_rejected_by_fev_task": len(
                invalid["nonfinite_forecast_rejections"]
            ),
            "fev_mae_with_one_nan_actual": invalid["nonfinite_actual_native_summary"][
                "summary"
            ]["MAE"],
            "nonfinite_actual_fixture_total_cases": 2,
            "nonfinite_actual_fixture_scored_cases": 1,
            "nonfinite_actual_rejected_by_tsfm_dataset": True,
        },
        "overlapping_identities": {
            "forecast_cases": overlap["forecast_cases"],
            "unique_target_timestamps": overlap["unique_target_timestamps"],
            "overlap_preserved": overlap["overlap_preserved"],
        },
        "covariate_availability": {
            "past_fields": covariates["past_fields"],
            "future_fields": covariates["future_fields"],
            "future_measured_perturbation_leaves_inputs_equal": True,
            "fev_accepts_declared_future_measured_column": True,
            "prototype_availability_guard_rejected_misclassification": True,
        },
        "quantile_normalization": {
            "ordinary_mean_pinball": quantiles["ordinary_mean_pinball"],
            "fev_mql": quantiles["fev_mql"],
            "unordered_input_levels_sorted_by_fev": True,
            "missing_quantile_column_rejected_by_fev": True,
            "crossing_quantiles_pass_native_validation": True,
            "prototype_crossing_guard_rejected_crossing": True,
        },
    }


def source_identities(source):
    paths = [
        "examples/datasets/seasonal.csv",
        "src/tsfm_lab/contracts.py",
        "src/tsfm_lab/data.py",
        "src/tsfm_lab/evaluation/metrics.py",
        "src/tsfm_lab/evaluation/runner.py",
        "src/tsfm_lab/models/baselines.py",
    ]
    return {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in paths
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh output directory"
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[2]
    os.environ.update(
        TSFM_FEV_STUDY_OUTPUT=str(output),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        HF_HOME=str(output / "hf-cache"),
        HF_DATASETS_CACHE=str(output / "datasets-cache"),
        HF_HUB_DISABLE_TELEMETRY="1",
        DO_NOT_TRACK="1",
    )
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source / "src"))

    from fixtures_baselines import (
        baseline_equivalence,
        changing_scale_and_pooling,
        unequal_series_weighting,
    )
    from fixtures_policies import (
        covariate_availability,
        overlapping_identities,
        quantile_normalization,
        undefined_and_nonfinite,
    )
    from study_support import dependency_closure, identities, write_json

    if importlib.metadata.version("fev") != "0.10.0":
        parser.error("This recorded study requires fev==0.10.0")
    start = time.perf_counter()
    evidence = {
        "status": "running",
        "date_utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "fev_version": importlib.metadata.version("fev"),
        "datasets_version": importlib.metadata.version("datasets"),
        "fixtures": {},
        "runtime_network_policy": "IP socket connections blocked; local IPC permitted; Hub/datasets offline",
    }
    try:
        for check in [
            baseline_equivalence,
            changing_scale_and_pooling,
            unequal_series_weighting,
            undefined_and_nonfinite,
            overlapping_identities,
            covariate_availability,
            quantile_normalization,
        ]:
            tick = time.perf_counter()
            result = check()
            evidence["fixtures"][check.__name__] = {
                "status": "passed",
                "elapsed_seconds": time.perf_counter() - tick,
                "evidence": result,
            }
            print(f"PASS {check.__name__}", flush=True)
        evidence["dependency_closure"] = dependency_closure()
        evidence["inputs"] = identities()
        evidence["tsfm_source_sha256"] = source_identities(source)
        pip_check = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            check=False,
        )
        evidence["pip_check_including_inherited_packages"] = {
            "exit_code": pip_check.returncode,
            "output": pip_check.stdout.strip(),
        }
        evidence["status"] = "passed"
    except Exception as exc:
        evidence["status"] = "failed"
        evidence["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        evidence["elapsed_seconds"] = time.perf_counter() - start
        write_json("results-full.json", evidence)
    summary = {
        key: value
        for key, value in evidence.items()
        if key not in {"fixtures", "inputs", "dependency_closure"}
    }
    summary["fixture_groups_passed"] = len(evidence["fixtures"])
    summary["results"] = concise_results(evidence["fixtures"])
    summary["relevant_dependency_closure_satisfied"] = True
    summary["relevant_dependency_count"] = len(evidence["dependency_closure"])
    summary["prototype_guards_are_user_defined"] = [
        "complete item coverage",
        "declared covariate availability",
        "quantile crossing policy",
    ]
    write_json("summary.json", summary)
    print(f'All seven fixture groups passed in {evidence["elapsed_seconds"]:.3f}s')


if __name__ == "__main__":
    main()
