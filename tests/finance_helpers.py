"""Original synthetic prices and saved forecast fixtures for finance tests."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from tsfm_lab.contracts import EvaluationSpec
from tsfm_lab.data import load_csv
from tsfm_lab.evaluation.runner import run_experiment
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.finance.contracts import BacktestRules
from tsfm_lab.finance.ohlc import load_ohlc_csv
from tsfm_lab.models.baselines import LastValue


def price_fixture(
    root, *, closes=None, opens=None, bump=0.02, stride=1, horizon=1, units="toy_quote"
):
    root.mkdir(parents=True, exist_ok=True)
    closes = np.asarray(
        closes if closes is not None else [100, 102, 101, 103, 100, 105, 104, 106],
        dtype=float,
    )
    opens = closes.copy() if opens is None else np.asarray(opens, dtype=float)
    times = pd.date_range("2026-01-01", periods=len(closes), freq="h", tz="UTC")
    table = pd.DataFrame(
        {
            "open_timestamp": times - pd.Timedelta(minutes=45),
            "close_timestamp": times,
            "open": opens,
            "high": np.maximum(opens, closes) * 1.02,
            "low": np.minimum(opens, closes) * 0.98,
            "close": closes,
        }
    )
    path = root / "ohlc.csv"
    table.to_csv(path, index=False)
    dataset_options = {
        "timestamp_column": "close_timestamp",
        "target_column": "close",
        "frequency": "1h",
        "units": units,
        "series_id": "toy_asset",
    }
    dataset = load_csv(path, **dataset_options)
    bars = load_ohlc_csv(path, frequency="1h", units=units, series_id="toy_asset")

    class FixtureForecast(LastValue):
        adapter_id = "price_fixture"

        def metadata(self):
            return replace(super().metadata(), family="test_fixture")

        def predict(self, requests):
            outputs = super().predict(requests)
            values = []
            for request, output in zip(requests, outputs, strict=True):
                index = times.get_indexer([request.origin])[0]
                change = bump if np.isscalar(bump) else bump[index]
                point = np.full(
                    len(output.point), request.context_values[-1] * (1 + change)
                )
                values.append(replace(output, point=point))
            return tuple(values)

    spec = EvaluationSpec(
        2, horizon, times[2], times[-1] + pd.Timedelta("1h"), stride=stride
    )
    result = run_experiment(dataset, FixtureForecast(), spec)
    workspace = root / "workspace"
    store = ExperimentStore(workspace)
    run_id = store.create(
        "Synthetic direct-Python price fixture", {}, "# Direct Python fixture"
    )
    store.complete(run_id, result.to_dict(), result.predictions)
    source_config = root / "source.yaml"
    source_config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "model": {"source": "builtin", "adapter_id": "last_value"},
                "dataset": {"path": str(path), **dataset_options},
                "evaluation": {
                    "context_length": 2,
                    "horizon": 1,
                    "stride": 1,
                    "evaluation_start": times[2].isoformat(),
                    "evaluation_end": (times[-1] + pd.Timedelta("1h")).isoformat(),
                },
            }
        )
    )
    rules = BacktestRules(
        initial_capital=1000,
        threshold_bps=100,
        fee_bps=5,
        slippage_bps=2,
        short_financing_bps=3,
        signal_latency_seconds=60,
    )
    return SimpleNamespace(
        run_id=run_id,
        workspace=workspace,
        dataset=dataset,
        ohlc=bars,
        rules=rules,
        source_config=source_config,
        ohlc_path=path,
        result=result,
    )
