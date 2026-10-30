"""Future point forecasting with explicit preparation and separate checked outputs."""

from datetime import UTC, datetime
from time import monotonic

from tsfm_lab.application import RunExecutionError, load_dataset
from tsfm_lab.config import ConfigError
from tsfm_lab.data import canonical_fingerprint
from tsfm_lab.future import prepare_future_request, resolve_future, run_future
from tsfm_lab.future_config import (
    FutureForecastConfig,
    future_config_from_mapping,
    load_future_config,
)
from tsfm_lab.future_store import FutureStore
from tsfm_lab.future_validation import build_result, resolved_configuration
from tsfm_lab.models.factory import build_model
from tsfm_lab.models.settings import validate_settings
from tsfm_lab.point_tasks import validate_model_frequency
from tsfm_lab.workflows import validate_workflow_dataset


def run_future_from_config(path, **options):
    return run_future_config(load_future_config(path), **options)


def run_future_config(
    config,
    *,
    progress=None,
    window_progress=None,
    should_cancel=None,
    on_created=None,
    include_outputs=True,
    expected_dataset_sha256=None,
    expected_checkpoint_fingerprint=None,
    expected_source_sha256=None,
):
    """Validate before model access; publish a complete future record last."""
    if not isinstance(config, FutureForecastConfig):
        raise ConfigError("Future execution requires FutureForecastConfig")
    if config._record_only:
        raise ConfigError(
            "Saved future configurations are read-only; load a native configuration before running"
        )

    def check_cancel():
        if should_cancel and should_cancel():
            raise KeyboardInterrupt("Future forecast cancellation requested")

    started_at, started = datetime.now(UTC).isoformat(), monotonic()
    check_cancel()
    try:
        future_config_from_mapping(
            config.resolved_dict(), base_directory=config.source_path.parent
        )
        dataset = load_dataset(config.dataset)
        validate_workflow_dataset(config.workflow, dataset)
        if (
            expected_dataset_sha256 is not None
            and canonical_fingerprint(dataset) != expected_dataset_sha256
        ):
            raise ConfigError(
                "The dataset changed after review; review the forecast again"
            )
        if (
            expected_source_sha256 is not None
            and dataset.provenance["source_sha256"] != expected_source_sha256
        ):
            raise ConfigError(
                "The source bytes changed after review; review the forecast again"
            )
        resolved = resolve_future(config, dataset)
        prepare_future_request(
            dataset, resolved.forecast, adapter_id=config.model["adapter_id"]
        )
        validate_settings(config.model, resolved.forecast, dataset.frequency)
        check_cancel()
        model = build_model(config.model, config.runtime)
        model.validate(resolved.forecast)
        validate_model_frequency(model, dataset.frequency)
        checkpoint = model.configuration().get("checkpoint", {})
        if (
            expected_checkpoint_fingerprint is not None
            and checkpoint.get("fingerprint") != expected_checkpoint_fingerprint
        ):
            raise ConfigError(
                "Checkpoint changed after preparation; review the forecast again"
            )
    except (OSError, ValueError, ImportError) as exc:
        raise ConfigError(str(exc)) from exc
    check_cancel()
    store = FutureStore(config.output_directory)
    run_id = store.create(
        resolved_configuration(config, resolved.forecast.origin), config.source_yaml
    )
    stage = "pending"

    def update(value):
        nonlocal stage
        store.set_status(run_id, value)
        stage = value
        if progress:
            progress(value)

    try:
        if on_created:
            on_created(run_id)
        check_cancel()
        update("preparing")
        if window_progress:
            window_progress(0, 1)
        forecast = run_future(
            dataset,
            model,
            resolved.forecast,
            runtime=config.runtime,
            progress=update,
            should_cancel=should_cancel,
        )
        if window_progress:
            window_progress(1, 1)
        check_cancel()
        payload, predictions, observations = build_result(
            config,
            dataset,
            forecast,
            run_id,
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
            elapsed_seconds=monotonic() - started,
        )
        update("saving")
        check_cancel()
        store.complete(
            run_id, payload, predictions, observations, cancel_check=should_cancel
        )
    except (Exception, KeyboardInterrupt) as exc:
        status = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        try:
            current = store.get_manifest(run_id)["status"]
            if current == "completed":
                return store.result_summary(run_id)
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
            f"Future forecast {run_id} failed during {stage}: {exc}"
        ) from exc
    return store.result_summary(run_id)
