"""Shared application functions for shell and future interactive callers."""

from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

from tsfm_lab.config import ConfigError, load_config
from tsfm_lab.data import canonical_fingerprint, load_csv
from tsfm_lab.evaluation.runner import run_experiment
from tsfm_lab.evaluation.windows import evaluation_bounds
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.models.factory import build_model


class RunExecutionError(RuntimeError):
    """A validated experiment failed after its record was created."""


def load_dataset(options: dict):
    """Choose a strict local loader without importing optional Parquet eagerly."""
    arguments = {key: value for key, value in options.items() if key != "format"}
    if options.get("format", Path(options["path"]).suffix.lower()[1:]) == "csv":
        return load_csv(**arguments)
    from tsfm_lab.parquet import load_parquet

    return load_parquet(**arguments)


def run_from_config(
    path: str | Path, *, progress: Callable[[str], None] | None = None
) -> dict:
    """Validate before creating a run, then persist success, failure, or cancellation."""
    config = load_config(path)
    try:
        dataset = load_dataset(config.dataset)
        model = build_model(config.model, config.runtime)
        evaluation_bounds(dataset, config.evaluation)
        model.validate(config.evaluation)
        canonical_fingerprint(dataset)
    except (OSError, ValueError, ImportError) as exc:
        raise ConfigError(str(exc)) from exc
    store = ExperimentStore(config.output_directory)
    run_id = store.create(config.name, config.resolved_dict(), config.source_yaml)
    stage = "pending"

    def update(next_stage: str) -> None:
        nonlocal stage
        store.set_status(run_id, next_stage)
        stage = next_stage
        if progress:
            progress(next_stage)

    try:
        update("preparing")
        result = run_experiment(dataset, model, config.evaluation, progress=update)
        payload = result.to_dict()
        payload.update(run_id=run_id, name=config.name)
        payload["configuration"]["runtime"].update(
            seed=config.runtime["seed"], seed_used=False, deterministic=True
        )
        payload["provenance"]["environment"]["PyYAML"] = version("PyYAML")
        if config.dataset["format"] == "parquet":
            payload["provenance"]["environment"]["pyarrow"] = version("pyarrow")
        stage = "saving"
        if progress:
            progress(stage)
        store.complete(run_id, payload, result.predictions)
    except (Exception, KeyboardInterrupt) as exc:
        status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        try:
            store.set_status(
                run_id,
                status,
                error={"type": type(exc).__name__, "message": str(exc), "stage": stage},
            )
        except (OSError, ValueError) as record_error:
            exc.add_note(f"Could not persist {status} status: {record_error}")
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise RunExecutionError(
            f"Experiment {run_id} failed during {stage}: {exc}"
        ) from exc
    return {
        "run_id": run_id,
        "name": config.name,
        "status": "completed",
        "workspace": str(config.output_directory),
        "directory": str(config.output_directory / "experiments" / run_id),
        "metrics": payload["metrics"]["overall"],
        "counts": payload["counts"],
    }
