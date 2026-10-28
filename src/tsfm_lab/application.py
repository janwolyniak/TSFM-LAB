"""Shared application functions for shell and interactive callers."""

from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

from tsfm_lab.config import ConfigError, ExperimentConfig, load_config
from tsfm_lab.data import canonical_fingerprint, load_csv
from tsfm_lab.evaluation.runner import run_experiment
from tsfm_lab.evaluation.windows import evaluation_bounds
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.models.factory import build_model
from tsfm_lab.point_tasks import validate_model_frequency
from tsfm_lab.workflows import validate_workflow_dataset


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
    path: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
    window_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_created: Callable[[str], None] | None = None,
) -> dict:
    """Load YAML and delegate to the same application used by managed workers."""
    return run_config(
        load_config(path),
        progress=progress,
        window_progress=window_progress,
        should_cancel=should_cancel,
        on_created=on_created,
    )


def run_config(
    config: ExperimentConfig,
    *,
    progress: Callable[[str], None] | None = None,
    window_progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_created: Callable[[str], None] | None = None,
    include_outputs: bool = False,
    expected_dataset_sha256: str | None = None,
    expected_checkpoint_fingerprint: str | None = None,
) -> dict:
    """Validate before creating a run, then persist success, failure, or cancellation."""

    def check_cancel() -> None:
        if should_cancel and should_cancel():
            raise KeyboardInterrupt("Experiment cancellation requested")

    check_cancel()
    try:
        dataset = load_dataset(config.dataset)
        if (
            expected_dataset_sha256 is not None
            and canonical_fingerprint(dataset) != expected_dataset_sha256
        ):
            raise ConfigError(
                "The dataset changed after review; review the experiment again"
            )
        validate_workflow_dataset(config.workflow, dataset)
        model = build_model(config.model, config.runtime)
        if expected_checkpoint_fingerprint is not None:
            checkpoint = model.configuration().get("checkpoint", {})
            if checkpoint.get("fingerprint") != expected_checkpoint_fingerprint:
                raise ConfigError(
                    "Checkpoint changed after preparation; review the experiment again"
                )
        evaluation_bounds(dataset, config.evaluation)
        model.validate(config.evaluation)
        validate_model_frequency(model, dataset.frequency)
        canonical_fingerprint(dataset)
    except (OSError, ValueError, ImportError) as exc:
        raise ConfigError(str(exc)) from exc
    check_cancel()
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
        if on_created:
            on_created(run_id)
        check_cancel()
        update("preparing")
        result = run_experiment(
            dataset,
            model,
            config.evaluation,
            workflow=config.workflow,
            progress=update,
            window_progress=window_progress,
            should_cancel=should_cancel,
        )
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
        check_cancel()
        store.complete(
            run_id,
            payload,
            result.predictions,
            cancel_check=should_cancel,
            include_outputs=include_outputs,
        )
    except (Exception, KeyboardInterrupt) as exc:
        status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        try:
            current = store.get_manifest(run_id)["status"]
            if current == "completed":
                summary = store.result_summary(run_id)
                if include_outputs:
                    summary["outputs"] = store.output_paths(run_id)
                return summary
            if current not in {"failed", "cancelled"}:
                store.set_status(
                    run_id,
                    status,
                    error={
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "stage": stage,
                    },
                )
        except (OSError, ValueError) as record_error:
            exc.add_note(f"Could not persist {status} status: {record_error}")
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise RunExecutionError(
            f"Experiment {run_id} failed during {stage}: {exc}"
        ) from exc
    summary = {
        "run_id": run_id,
        "name": config.name,
        "status": "completed",
        "workspace": str(config.output_directory),
        "directory": str(config.output_directory / "experiments" / run_id),
        "metrics": payload["metrics"]["overall"],
        "counts": payload["counts"],
    }
    if include_outputs:
        summary["outputs"] = store.output_paths(run_id)
    return summary
