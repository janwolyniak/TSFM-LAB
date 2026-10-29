"""Review and execute an explicitly confirmed local forecasting preparation plan."""

import copy
import hashlib
import importlib
import json
from dataclasses import dataclass, field, replace

from tsfm_lab.application import load_dataset, run_config
from tsfm_lab.config import ConfigError, ExperimentConfig, config_from_mapping
from tsfm_lab.data import canonical_fingerprint
from tsfm_lab.evaluation.windows import evaluation_bounds
from tsfm_lab.models.checkpoint_profiles import checkpoint_adapter
from tsfm_lab.models.factory import build_model
from tsfm_lab.models.preparation import (
    CheckpointPlan,
    plan_checkpoint,
    prepare_checkpoint,
)
from tsfm_lab.models.registry import adapter_profile
from tsfm_lab.models.settings import validate_settings
from tsfm_lab.point_tasks import validate_model_frequency
from tsfm_lab.run_modes import forecast_spec, is_future, run_validated
from tsfm_lab.runtime_diagnostics import runtime_diagnostics
from tsfm_lab.workflows import validate_workflow_dataset


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _cancel(should_cancel):
    if should_cancel and should_cancel():
        raise KeyboardInterrupt("Guided preparation cancellation requested")


@dataclass(frozen=True, init=False)
class GuidedRunPlan:
    _config: ExperimentConfig = field(repr=False)
    checkpoint: CheckpointPlan | None
    dataset_sha256: str
    configuration_sha256: str
    local_checkpoint_sha256: str | None
    source_sha256: str | None

    def __init__(
        self,
        config,
        checkpoint,
        dataset_sha256,
        local_checkpoint_sha256=None,
        source_sha256=None,
    ):
        object.__setattr__(self, "_config", copy.deepcopy(config))
        object.__setattr__(self, "checkpoint", checkpoint)
        object.__setattr__(self, "dataset_sha256", dataset_sha256)
        object.__setattr__(
            self, "configuration_sha256", _digest(config.resolved_dict())
        )
        object.__setattr__(self, "local_checkpoint_sha256", local_checkpoint_sha256)
        object.__setattr__(self, "source_sha256", source_sha256)

    @property
    def config(self):
        return copy.deepcopy(self._config)

    @property
    def plan_identity(self):
        return _digest(
            {
                "configuration": self.configuration_sha256,
                "dataset": self.dataset_sha256,
                "source": self.source_sha256,
                "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
                "local_checkpoint": self.local_checkpoint_sha256,
            }
        )

    def summary(self):
        if self.checkpoint is None:
            model = (
                "Built-in baseline"
                if self.config.model["source"] == "builtin"
                else "Verified local checkpoint"
            )
            return f"{model} · results: {self.config.output_directory}"
        checkpoint = self.checkpoint
        size = (
            f"{checkpoint.download_bytes} bytes"
            if checkpoint.download_bytes is not None
            else "size unknown"
        )
        return f"{checkpoint.repository} @ {checkpoint.resolved_revision} · {checkpoint.cache_status} · download {size} · results: {self.config.output_directory}"


def _runtime_metadata(adapter):
    status = next(
        item for item in runtime_diagnostics() if item["adapter_id"] == adapter
    )
    if status["status"] != "available":
        raise ConfigError(
            f"{status['label']} runtime is unavailable: {status['detail']}"
        )


def _native_bounds(config, metadata):
    adapter = config.model["adapter_id"]
    if checkpoint_adapter(metadata["architecture"]) != adapter:
        raise ConfigError("Checkpoint architecture disagrees with the selected model")
    spec = forecast_spec(config)
    context, horizon = spec.context_length, spec.horizon
    native_context = metadata["native_context_length"]
    if (adapter == "ttm_r2" and context != native_context) or context > native_context:
        raise ConfigError("Context length is unsupported by the reviewed checkpoint")
    if horizon > metadata["native_prediction_length"]:
        raise ConfigError("Forecast horizon is unsupported by the reviewed checkpoint")
    if adapter == "ttm_r2":
        from tsfm_lab.models.ttm_profile import validate_frequency

        validate_frequency(
            config.dataset["frequency"], metadata["point_forecast"]["frequency_policy"]
        )


def _review_data(config):
    if is_future(config):
        from tsfm_lab.future import prepare_future_request
        from tsfm_lab.future_config import future_config_from_mapping

        future_config_from_mapping(
            config.resolved_dict(), base_directory=config.source_path.parent
        )
        dataset = load_dataset(config.dataset)
        validate_workflow_dataset(config.workflow, dataset)
        prepare_future_request(
            dataset, config.forecast, adapter_id=config.model["adapter_id"]
        )
        validate_settings(config.model, config.forecast, dataset.frequency)
        return canonical_fingerprint(dataset)
    config_from_mapping(
        config.resolved_dict(), base_directory=config.source_path.parent
    )
    dataset = load_dataset(config.dataset)
    validate_workflow_dataset(config.workflow, dataset)
    evaluation_bounds(dataset, config.evaluation)
    validate_settings(config.model, config.evaluation, dataset.frequency)
    return canonical_fingerprint(dataset)


def review_guided(config, *, offline=False, progress=None, should_cancel=None):
    """Validate data and installed metadata before reviewing any Hub checkpoint.

    Review never starts inference, transfers weights, or creates a run/preparation
    record. Automatic mode can resolve public metadata and a small configuration.
    """
    if not isinstance(config, ExperimentConfig) and not is_future(config):
        raise ConfigError(
            "Guided preparation requires a validated experiment configuration"
        )
    _cancel(should_cancel)
    snapshot = copy.deepcopy(config)
    fingerprint = _review_data(snapshot)
    if is_future(snapshot):
        from tsfm_lab.future import resolve_future

        snapshot = resolve_future(snapshot, load_dataset(snapshot.dataset))
    _cancel(should_cancel)
    checkpoint, local_identity = None, None
    selection = snapshot.model
    if selection["source"] != "builtin":
        _runtime_metadata(selection["adapter_id"])
        if selection["source"] == "local":
            model = build_model(selection, snapshot.runtime)
            model.validate(forecast_spec(snapshot))
            validate_model_frequency(model, snapshot.dataset["frequency"])
            local_identity = _digest(model.configuration()["checkpoint"])
        else:
            checkpoint = plan_checkpoint(
                selection["repository"],
                selection["revision"],
                cache_dir=selection.get("cache_dir"),
                offline=offline,
                progress=progress,
                should_cancel=should_cancel,
            )
            _native_bounds(snapshot, checkpoint.metadata)
            snapshot = replace(
                snapshot,
                model={
                    **selection,
                    "revision": checkpoint.resolved_revision,
                    "offline": True,
                    "cache_dir": str(checkpoint.cache_dir),
                },
            )
            validate_settings(
                snapshot.model, forecast_spec(snapshot), snapshot.dataset["frequency"]
            )
    _cancel(should_cancel)
    source_sha256 = (
        load_dataset(snapshot.dataset).provenance["source_sha256"]
        if is_future(snapshot)
        else None
    )
    return GuidedRunPlan(
        snapshot, checkpoint, fingerprint, local_identity, source_sha256
    )


def _runtime_imports(config):
    if config.model["source"] != "builtin":
        profile = adapter_profile(config.model["adapter_id"])
        importlib.import_module(profile.module)._runtime()


def validate_reviewed_checkpoint(plan, checkpoint, prepared_fingerprint=None):
    """Bind late completion to reviewed known bytes and the prepared receipt."""
    if plan.config.model["source"] == "builtin":
        return
    if not isinstance(checkpoint, dict):
        raise ConfigError("Future completion lacks its reviewed checkpoint")
    if plan.local_checkpoint_sha256 is not None:
        if _digest(checkpoint) != plan.local_checkpoint_sha256:
            raise ConfigError(
                "Future completion checkpoint differs from reviewed local bytes"
            )
        return
    reviewed = plan.checkpoint
    if reviewed is None or (
        checkpoint["repository"] != reviewed.repository
        or checkpoint["resolved_revision"] != reviewed.resolved_revision
        or checkpoint["fingerprint"] != prepared_fingerprint
    ):
        raise ConfigError(
            "Future completion checkpoint differs from approved preparation"
        )
    files = checkpoint["files"]
    if set(files) != {item.name for item in reviewed.expected_files}:
        raise ConfigError("Future checkpoint file set differs from review")
    for expected in reviewed.expected_files:
        actual = files[expected.name]
        if (expected.size is not None and actual["size"] != expected.size) or (
            expected.sha256 is not None and actual["sha256"] != expected.sha256
        ):
            raise ConfigError("Future checkpoint files differ from reviewed identities")


def execute_guided(
    plan,
    *,
    progress=None,
    window_progress=None,
    should_cancel=None,
    on_created=None,
    operation_id=None,
):
    """Execute one reviewed, explicitly authorized preparation and historical run."""
    from tsfm_lab.guided_execution import PreparationRecord

    if not isinstance(plan, GuidedRunPlan):
        raise ConfigError("Expected a reviewed guided run plan")
    config = plan.config
    record = PreparationRecord(config, plan, operation_id)

    def stage(value):
        event = {"stage": value} if isinstance(value, str) else dict(value)
        record.update(event)
        if progress:
            progress(
                event
                | {
                    "operation_id": record.operation_id,
                    "elapsed_seconds": record.elapsed,
                }
            )

    def created(identifier):
        record.link_run(identifier)
        if on_created:
            on_created(identifier)

    try:
        stage("cache")
        _cancel(should_cancel)
        if (
            _digest(config.resolved_dict()) != plan.configuration_sha256
            or _review_data(config) != plan.dataset_sha256
            or (
                plan.source_sha256 is not None
                and load_dataset(config.dataset).provenance["source_sha256"]
                != plan.source_sha256
            )
        ):
            raise ConfigError(
                "Reviewed configuration or dataset changed; review the experiment again"
            )
        _runtime_imports(config)
        _cancel(should_cancel)
        checkpoint_fingerprint = None
        if plan.checkpoint is not None:
            checkpoint = prepare_checkpoint(
                plan.checkpoint, progress=stage, should_cancel=should_cancel
            )
            checkpoint_fingerprint = checkpoint.fingerprint
        elif plan.local_checkpoint_sha256 is not None:
            stage("verifying")
            model = build_model(config.model, config.runtime)
            if (
                _digest(model.configuration()["checkpoint"])
                != plan.local_checkpoint_sha256
            ):
                raise ConfigError(
                    "Local checkpoint changed after review; review the experiment again"
                )
            checkpoint_fingerprint = model.configuration()["checkpoint"]["fingerprint"]
        _cancel(should_cancel)
        if is_future(config) and checkpoint_fingerprint is not None:
            record.record_checkpoint(checkpoint_fingerprint)
        extra = (
            {"expected_source_sha256": plan.source_sha256} if is_future(config) else {}
        )
        runner = run_validated if is_future(config) else run_config
        result = runner(
            config,
            progress=stage,
            window_progress=window_progress,
            should_cancel=should_cancel,
            on_created=created,
            include_outputs=True,
            expected_dataset_sha256=plan.dataset_sha256,
            expected_checkpoint_fingerprint=checkpoint_fingerprint,
            **extra,
        )
        record.finish("completed")
        return result | {
            "operation_id": record.operation_id,
            "preparation_path": str(record.path),
        }
    except (Exception, KeyboardInterrupt) as exc:
        record.finish(
            "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed", exc
        )
        raise
