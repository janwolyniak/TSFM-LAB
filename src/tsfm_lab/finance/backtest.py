"""Build derived OHLC reports from checked saved forecasts and explicit inputs."""

import copy
from dataclasses import asdict
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np

from tsfm_lab import __version__
from tsfm_lab.contracts import Dataset
from tsfm_lab.data import canonical_fingerprint
from tsfm_lab.evaluation.runner import _environment, _source_revision
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.finance.analysis import EXECUTION_POLICY, evaluate_cases
from tsfm_lab.finance.contracts import POLICY, BacktestRules, FinanceError
from tsfm_lab.finance.ohlc import (
    OHLCData,
    ohlc_fingerprint,
    validate_price_alignment,
)


def _selected_cases(result, forecasts, dataset, ohlc):
    if result["configuration"]["evaluation"]["stride"] != 1:
        raise FinanceError("The next-bar OHLC policy requires forecast stride 1")
    selected = forecasts.loc[forecasts.lead == 1]
    if len(selected) != result["counts"]["successful_windows"] or selected.empty:
        raise FinanceError("Every completed origin must retain its lead-1 forecast")
    origins = dataset.timestamps.get_indexer(selected.origin)
    targets = dataset.timestamps.get_indexer(selected.target_timestamp)
    if np.any(origins < 0) or np.any(targets != origins + 1):
        raise FinanceError(
            "Forecast origins/targets are missing from the original data"
        )
    if not np.array_equal(selected.actual.to_numpy(), dataset.values[targets]):
        raise FinanceError("Saved actual prices disagree with the original dataset")
    cases = []
    for row, origin, target in zip(
        selected.itertuples(), origins, targets, strict=True
    ):
        cases.append(
            {
                "request_id": row.request_id,
                "series_id": row.series_id,
                "origin": row.origin.isoformat(),
                "target_timestamp": row.target_timestamp.isoformat(),
                "lead": 1,
                "point_statistic": row.point_statistic,
                "origin_close": float(dataset.values[origin]),
                "forecast_close": float(row.prediction),
                "actual_close": float(row.actual),
                "open_timestamp": ohlc.open_timestamps[target].isoformat(),
                "close_timestamp": ohlc.close_timestamps[target].isoformat(),
                "open": float(ohlc.opens[target]),
                "high": float(ohlc.highs[target]),
                "low": float(ohlc.lows[target]),
                "close": float(ohlc.closes[target]),
            }
        )
    return cases


def backtest_run(
    run_id,
    workspace,
    dataset: Dataset,
    ohlc: OHLCData,
    rules: BacktestRules,
    *,
    target_is_price: bool,
) -> dict:
    """Evaluate saved lead-1 signals; this never constructs or executes a model."""
    if target_is_price is not True:
        raise FinanceError("Explicitly declare that the forecast target is a price")
    if not isinstance(dataset, Dataset) or not isinstance(ohlc, OHLCData):
        raise FinanceError(
            "Validated original Dataset and OHLCData objects are required"
        )
    if not isinstance(rules, BacktestRules):
        raise FinanceError("Explicit validated BacktestRules are required")
    if dataset.units == "unspecified":
        raise FinanceError("Declare price quote units in the original forecast dataset")
    # Revalidate snapshots in case trusted calling Python changed a timestamp buffer.
    dataset = Dataset(**vars(dataset))
    ohlc = OHLCData(**vars(ohlc))
    validate_price_alignment(dataset, ohlc)
    store = ExperimentStore(workspace)
    result, forecasts = store.load_result(run_id)
    if result["configuration"]["evaluation"]["point_statistic"] != "median":
        raise FinanceError("Finance reports require the saved median task")
    fingerprint = canonical_fingerprint(dataset)
    if fingerprint != result["provenance"]["dataset"]["canonical_sha256"]:
        raise FinanceError("The original dataset identity does not match the saved run")
    cases = _selected_cases(result, forecasts, dataset, ohlc)
    evaluation = evaluate_cases(
        cases, rules, frequency=dataset.frequency, series_id=dataset.series_id
    )
    payload = {
        "schema_version": 1,
        "policy": POLICY,
        "report_id": str(uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "implementation": {
            "package_version": __version__,
            **_source_revision(),
            "environment": _environment(None),
        },
        "parent": {
            "run_id": run_id,
            "result": copy.deepcopy(result),
            "artifacts": copy.deepcopy(store.get_manifest(run_id)["artifacts"]),
        },
        "original_dataset": {
            "canonical_sha256": fingerprint,
            "metadata": copy.deepcopy(result["configuration"]["dataset"]),
            "provenance": copy.deepcopy(dataset.provenance),
            "observations": len(dataset.values),
        },
        "execution_data": {
            "canonical_sha256": ohlc_fingerprint(ohlc),
            "provenance": copy.deepcopy(ohlc.provenance),
            "observations": len(ohlc.closes),
        },
        "rules": asdict(rules),
        "resolved_policy": copy.deepcopy(EXECUTION_POLICY),
        "cases": cases,
        "evaluation": evaluation,
    }
    from tsfm_lab.finance.reports import seal_report

    return seal_report(payload)
