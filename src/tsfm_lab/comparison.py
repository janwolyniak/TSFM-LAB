"""Refuse controlled comparisons when forecast tasks or observed cases differ."""

from pathlib import Path

from tsfm_lab.contracts import fixed_interval
from tsfm_lab.experiments import ExperimentStore, StoreError


class ComparisonError(ValueError):
    """The selected records do not define the same forecasting task."""


def _signature(result: dict) -> dict:
    try:
        configuration = result["configuration"]
        provenance = result["provenance"]
        dataset = dict(configuration["dataset"])
        dataset["frequency"] = fixed_interval(dataset["frequency"]).value
        evaluation = dict(configuration["evaluation"])
        evaluation["metrics"] = sorted(evaluation["metrics"])
        fingerprint = provenance["dataset"]["canonical_sha256"]
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("invalid canonical dataset fingerprint")
        return {
            "dataset_fingerprint": fingerprint,
            "dataset": dataset,
            "evaluation": evaluation,
            "metric_version": provenance["metric_version"],
            "scale_history": provenance["scale_history"],
            "scale": result["metrics"]["scale"],
            "scale_reason": result["metrics"]["scale_reason"],
            "scale_count": result["metrics"]["scale_count"],
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise StoreError(f"Invalid comparison metadata: {exc}") from exc


def compare_runs(run_ids: list[str], workspace: str | Path) -> dict:
    if len(run_ids) < 2 or len(set(run_ids)) != len(run_ids):
        raise ComparisonError("Choose at least two distinct completed experiments")
    store = ExperimentStore(workspace)
    loaded = [(run_id, *store.load_result(run_id)) for run_id in run_ids]
    first_signature = _signature(loaded[0][1])
    keys = ["series_id", "origin", "target_timestamp", "lead"]
    reference = loaded[0][2].sort_values(keys)[keys + ["actual"]].reset_index(drop=True)
    for run_id, result, predictions in loaded:
        signature = _signature(result)
        differences = [
            key for key in first_signature if signature[key] != first_signature[key]
        ]
        if differences:
            raise ComparisonError(
                f"Experiment {run_id} has different task settings: {', '.join(differences)}"
            )
        actual = predictions.sort_values(keys)[keys + ["actual"]].reset_index(drop=True)
        if not reference.equals(actual):
            raise ComparisonError(
                f"Experiment {run_id} has different forecast cases or actuals"
            )
        if not set(predictions.point_statistic) <= {
            "deterministic",
            signature["evaluation"]["point_statistic"],
        }:
            raise ComparisonError(
                f"Experiment {run_id} has an incompatible point statistic"
            )
    return {
        "compatible": True,
        "comparison_policy": "controlled_point_v1",
        "task": first_signature,
        "runs": [
            {
                "run_id": run_id,
                "name": result.get("name", run_id),
                "model": result["configuration"]["model"],
                "metrics": result["metrics"]["overall"],
                "by_lead": result["metrics"]["by_lead"],
                "runtime": result["configuration"]["runtime"],
                "timing_seconds": result["timing_seconds"],
            }
            for run_id, result, _ in loaded
        ],
    }
