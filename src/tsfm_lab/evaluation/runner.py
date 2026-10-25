"""Synchronous baseline evaluation without terminal or persistence concerns."""

import json
import platform
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from tsfm_lab import __version__
from tsfm_lab.contracts import (
    Dataset,
    EvaluationSpec,
    ForecastOutput,
    ForecastRequest,
    ModelMetadata,
)
from tsfm_lab.data import canonical_fingerprint
from tsfm_lab.evaluation.metrics import Scores, score_forecasts
from tsfm_lab.evaluation.windows import evaluation_bounds, iter_windows
from tsfm_lab.models.base import ForecastModel


@dataclass(frozen=True)
class ExperimentResult:
    configuration: dict
    predictions: pd.DataFrame
    scores: Scores
    provenance: dict
    counts: dict[str, int]
    timing_seconds: dict[str, float]

    def to_dict(self) -> dict:
        """Return a JSON-compatible numerical result, without writing files."""
        rows = self.predictions.copy()
        for column in ("origin", "target_timestamp"):
            rows[column] = rows[column].map(lambda value: value.isoformat())
        return {
            "schema_version": 1,
            "status": "completed",
            "configuration": self.configuration,
            "predictions": rows.to_dict(orient="records"),
            "metrics": asdict(self.scores),
            "provenance": self.provenance,
            "counts": self.counts,
            "timing_seconds": self.timing_seconds,
        }


def _source_revision() -> dict:
    """Read only this source checkout; never misidentify an enclosing Git repo."""
    root = Path(__file__).resolve().parents[3]
    if not (root / ".git").exists():
        return {"git_revision": None, "git_dirty": None}
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"git_revision": None, "git_dirty": None}
    return {"git_revision": revision, "git_dirty": bool(status)}


def _check_capabilities(metadata: ModelMetadata, spec: EvaluationSpec) -> None:
    caps = metadata.capabilities
    if spec.context_length < caps.min_context:
        raise ValueError("context_length is below the adapter minimum")
    if caps.max_context is not None and spec.context_length > caps.max_context:
        raise ValueError("context_length exceeds the adapter maximum")
    if caps.max_horizon is not None and spec.horizon > caps.max_horizon:
        raise ValueError("horizon exceeds the adapter maximum")
    if caps.target_count != 1 or "point" not in caps.output_types:
        raise ValueError("the runner requires a univariate point forecast adapter")
    if "cpu" not in caps.devices:
        raise ValueError("the initial runner requires a CPU adapter")


def _aligned_output(
    output: ForecastOutput,
    request: ForecastRequest,
    metadata: ModelMetadata,
    spec: EvaluationSpec,
) -> ForecastOutput:
    if not isinstance(output, ForecastOutput):
        raise ValueError("adapter must return ForecastOutput objects")
    if (
        output.request_id != request.request_id
        or output.series_id != request.series_id
        or output.origin != request.origin
        or output.adapter_id != metadata.adapter_id
        or not output.target_timestamps.equals(request.target_timestamps)
    ):
        raise ValueError(
            "adapter output identity or timestamps do not match the request"
        )
    if output.point_statistic not in {"deterministic", spec.point_statistic}:
        raise ValueError(
            "adapter output point statistic does not match the requested task"
        )
    # Snapshot values before the adapter can reuse a mutable working buffer.
    return ForecastOutput(**{**vars(output), "point": output.point.copy()})


def _prediction_rows(
    actual_values, request: ForecastRequest, output: ForecastOutput
) -> list[dict]:
    return [
        {
            "request_id": request.request_id,
            "series_id": request.series_id,
            "adapter_id": output.adapter_id,
            "origin": request.origin,
            "target_timestamp": timestamp,
            "lead": lead,
            "actual": float(actual),
            "prediction": float(predicted),
            "point_statistic": output.point_statistic,
        }
        for lead, (timestamp, actual, predicted) in enumerate(
            zip(request.target_timestamps, actual_values, output.point, strict=True),
            start=1,
        )
    ]


def _configuration(dataset, metadata, model, spec) -> dict:
    evaluation = asdict(spec)
    for name in ("evaluation_start", "evaluation_end"):
        evaluation[name] = evaluation[name].isoformat()
    options = getattr(model, "configuration", lambda: {})()
    return {
        "model": {
            **asdict(metadata),
            "source": getattr(
                model,
                "source",
                "builtin" if metadata.family == "baseline" else "local_python",
            ),
            "options": options,
        },
        "dataset": {
            "series_id": dataset.series_id,
            "target_name": dataset.target_name,
            "units": dataset.units,
            "frequency": dataset.frequency,
        },
        "evaluation": evaluation,
        "runtime": getattr(
            model,
            "runtime_configuration",
            lambda: {"device": "cpu", "dtype": "float64", "batch_size": 1},
        )(),
    }


def _environment(model) -> dict:
    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "os": platform.platform(),
        "machine": platform.machine(),
        **getattr(model, "environment_metadata", lambda: {})(),
    }


def run_experiment(
    dataset: Dataset,
    model: ForecastModel,
    spec: EvaluationSpec,
    *,
    progress: Callable[[str], None] | None = None,
    window_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> ExperimentResult:
    """Run every complete window; any inference/alignment failure aborts the run."""
    start, end = evaluation_bounds(dataset, spec)
    total = 1 + (end - start - spec.horizon) // spec.stride

    def check_cancel() -> None:
        if should_cancel and should_cancel():
            raise KeyboardInterrupt("Experiment cancellation requested")

    check_cancel()
    metadata = model.metadata()
    _check_capabilities(metadata, spec)
    model.validate(spec)
    configuration = json.loads(
        json.dumps(_configuration(dataset, metadata, model, spec), allow_nan=False)
    )
    fingerprint = canonical_fingerprint(dataset)
    rows, actual, predicted = [], [], []
    load_time, inference_time = 0.0, 0.0
    try:
        if progress:
            progress("loading")
        check_cancel()
        before = perf_counter()
        model.load()
        load_time = perf_counter() - before
        check_cancel()
        if progress:
            progress("running")
        if window_progress:
            window_progress(0, total)
        for completed, window in enumerate(iter_windows(dataset, spec), start=1):
            check_cancel()
            # pandas timestamp buffers remain mutable even inside frozen records.
            expected = ForecastRequest(**vars(window.request))
            before = perf_counter()
            check_cancel()
            outputs = model.predict((window.request,))
            inference_time += perf_counter() - before
            check_cancel()
            if len(outputs) != 1:
                raise ValueError("adapter must return exactly one output per request")
            output = _aligned_output(outputs[0], expected, metadata, spec)
            rows.extend(_prediction_rows(window.actual, expected, output))
            actual.append(window.actual)
            predicted.append(output.point)
            if window_progress:
                window_progress(completed, total)
    finally:
        model.unload()
    check_cancel()
    if progress:
        progress("evaluating")
    check_cancel()
    before = perf_counter()
    scores = score_forecasts(
        np.stack(actual),
        np.stack(predicted),
        dataset.values[:start],
        seasonal_period=spec.seasonal_period,
        metrics=spec.metrics,
    )
    evaluation_time = perf_counter() - before
    check_cancel()
    predictions = pd.DataFrame(rows)
    return ExperimentResult(
        configuration=configuration,
        predictions=predictions,
        scores=scores,
        provenance={
            "source": {"package_version": __version__, **_source_revision()},
            "dataset": {
                **dataset.provenance,
                "canonical_sha256": fingerprint,
            },
            "environment": _environment(model),
            "metric_version": "point_metrics_v1",
            "scale_history": {
                "start": dataset.timestamps[0].isoformat(),
                "end_exclusive": dataset.timestamps[start].isoformat(),
                "observations": start,
                "seasonal_period": spec.seasonal_period,
            },
        },
        counts={
            "attempted_windows": len(actual),
            "successful_windows": len(actual),
            "failed_windows": 0,
            "forecast_cases": len(rows),
            "unique_target_timestamps": predictions.target_timestamp.nunique(),
        },
        timing_seconds={
            "load": load_time,
            "inference": inference_time,
            "evaluation": evaluation_time,
        },
    )
