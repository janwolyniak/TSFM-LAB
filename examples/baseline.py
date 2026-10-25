"""Run two offline baselines on an original synthetic series."""

import json
from pathlib import Path

import pandas as pd

from tsfm_lab.contracts import EvaluationSpec
from tsfm_lab.data import load_csv
from tsfm_lab.evaluation.runner import run_experiment
from tsfm_lab.models.baselines import LastValue, SeasonalNaive


def main() -> None:
    dataset = load_csv(
        Path(__file__).parent / "datasets" / "seasonal.csv",
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
    results = [
        run_experiment(dataset, model, spec).to_dict()
        for model in (LastValue(), SeasonalNaive(4))
    ]
    print(json.dumps({"experiments": results}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
