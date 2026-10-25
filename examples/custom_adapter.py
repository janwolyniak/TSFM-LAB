"""Educational endpoint drift using only each forecast request's history."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsfm_lab.contracts import (
    EvaluationSpec,
    ForecastOutput,
    ForecastRequest,
    ModelCapabilities,
    ModelMetadata,
)
from tsfm_lab.data import load_csv
from tsfm_lab.evaluation.runner import ExperimentResult, run_experiment


class LinearDrift:
    """Extrapolate the average change between the context's first and last values."""

    adapter_id = "linear_drift_example"

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            adapter_id=self.adapter_id,
            adapter_version="1",
            family="example",
            capabilities=ModelCapabilities(min_context=2),
        )

    def configuration(self) -> dict[str, str]:
        return {"method": "endpoint_drift", "fit_scope": "request_context"}

    def validate(self, spec: EvaluationSpec) -> None:
        if spec.context_length < 2:
            raise ValueError(
                "Linear drift requires at least two historical observations"
            )

    def load(self) -> None:
        """This arithmetic example has no model resources to initialize."""

    def predict(
        self, requests: tuple[ForecastRequest, ...]
    ) -> tuple[ForecastOutput, ...]:
        outputs = []
        for request in requests:
            values = request.context_values
            if len(values) < 2:
                raise ValueError(
                    "Linear drift requires at least two historical observations"
                )
            with np.errstate(over="raise", invalid="raise"):
                try:
                    drift = (values[-1] - values[0]) / (len(values) - 1)
                    leads = np.arange(
                        1, len(request.target_timestamps) + 1, dtype=np.float64
                    )
                    point = values[-1] + drift * leads
                except FloatingPointError as exc:
                    raise ValueError(
                        "Linear drift exceeds finite float64 arithmetic"
                    ) from exc
            outputs.append(
                ForecastOutput(
                    request_id=request.request_id,
                    series_id=request.series_id,
                    origin=request.origin,
                    target_timestamps=request.target_timestamps,
                    point=point,
                    adapter_id=self.adapter_id,
                    point_statistic="deterministic",
                )
            )
        return tuple(outputs)

    def unload(self) -> None:
        """This arithmetic example has no model resources to release."""


def run_example() -> ExperimentResult:
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
    return run_experiment(dataset, LinearDrift(), spec)


def main() -> None:
    print(
        json.dumps(
            {
                "purpose": "Educational adapter arithmetic on original synthetic data.",
                "experiment": run_example().to_dict(),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
