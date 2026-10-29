"""Recovery accepts complete artifacts only when they match reviewed identities."""

import copy
from pathlib import Path

import pandas as pd
import pytest
import yaml
from test_checkpoint_preparation import FakeHub

from tsfm_lab.application import load_dataset
from tsfm_lab.data import canonical_fingerprint
from tsfm_lab.execution import _reconcile
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.future import resolve_future
from tsfm_lab.future_application import run_future_config
from tsfm_lab.future_config import future_config_from_mapping
from tsfm_lab.guided import GuidedRunPlan, review_guided, validate_reviewed_checkpoint
from tsfm_lab.guided_execution import ManagedGuidedRun, PreparationRecord
from tsfm_lab.models import checkpoints as cp
from tsfm_lab.models.preparation import plan_checkpoint, prepare_checkpoint


@pytest.fixture
def config(config_factory):
    path = config_factory()
    fields = yaml.safe_load(path.read_text())
    fields.pop("evaluation")
    fields.update(
        schema_version=2,
        mode="future_forecast",
        forecast={"context_length": 8, "horizon": 4},
    )
    return future_config_from_mapping(fields, base_directory=path.parent)


def local_config(config, checkpoint):
    fields = config.resolved_dict()
    fields["model"] = {
        "source": "local",
        "adapter_id": "timesfm_2p5",
        "path": str(checkpoint),
    }
    fields["runtime"]["dtype"] = "float32"
    return future_config_from_mapping(fields, base_directory=config.source_path.parent)


def settle(plan, run_id, fingerprint=None):
    """Simulate the terminal IPC event after the managed child has been reaped."""
    worker = ManagedGuidedRun(plan)
    worker._record = PreparationRecord(plan.config, plan, worker.operation_id)
    worker._record.update({"stage": "verifying"})
    if fingerprint is not None:
        worker._record.record_checkpoint(fingerprint)
    worker._record.link_run(run_id)
    try:
        event = worker._settle(
            {
                "type": "failed",
                "run_id": None,
                "message": "IPC dropped before the final reply",
            }
        )
        record = PreparationRecord.load(worker._record.path).data
        assert event["run_id"] == run_id
        assert worker.run_id == run_id
        return event, record
    finally:
        worker.close()


@pytest.mark.parametrize("change_weights", [False, True])
def test_late_local_completion_must_match_reviewed_bytes(
    config, timesfm_checkpoint, timesfm_runtime, monkeypatch, change_weights
):
    config = local_config(config, timesfm_checkpoint)
    monkeypatch.setattr("tsfm_lab.guided._runtime_metadata", lambda adapter: None)
    plan = review_guided(config)
    reviewed_fingerprint = cp.inspect_local_checkpoint(timesfm_checkpoint).fingerprint
    if change_weights:
        (timesfm_checkpoint / "model.safetensors").write_bytes(
            b"different inert weights; no deserialization"
        )
    # This is a valid independent run; recovery must still enforce older consent.
    summary = run_future_config(config)
    store = ExperimentStore(config.output_directory)
    saved, _, _ = store.load_future_result(summary["run_id"])
    assert bool(timesfm_runtime.predictions)
    assert (
        saved["provenance"]["model"]["options"]["checkpoint"]["fingerprint"]
        != reviewed_fingerprint
    ) is change_weights
    event, record = settle(plan, summary["run_id"], reviewed_fingerprint)
    assert (
        event["type"]
        == record["status"]
        == ("failed" if change_weights else "completed")
    )
    if change_weights:
        assert "reviewed local bytes" in event["message"]
    else:
        assert event["result"]["evaluation"]["status"] == "unavailable"
    # Rejecting a recovery claim must preserve the independently valid result.
    assert store.get_manifest(summary["run_id"])["status"] == "completed"


def test_baseline_late_completion_recovers_its_run_id_and_outputs(config):
    plan = review_guided(config)
    summary = run_future_config(config)
    event, record = settle(plan, summary["run_id"])
    assert event["type"] == record["status"] == "completed"
    assert "observations.csv" in event["result"]["outputs"]


def test_explicit_origin_cannot_be_substituted_by_another_valid_completed_record(
    config,
):
    observed = load_dataset(config.dataset)
    fields = config.resolved_dict()
    fields["forecast"]["origin"] = observed.timestamps[-1].isoformat()
    explicit = future_config_from_mapping(
        fields, base_directory=config.source_path.parent
    )
    path = Path(config.dataset["path"])
    source = pd.read_csv(path)
    source.loc[len(source)] = [
        (observed.timestamps[-1] + pd.Timedelta("1h")).isoformat(),
        float(observed.values[-1] + 1),
    ]
    source.to_csv(path, index=False)
    summary = run_future_config(config)
    event = _reconcile(
        explicit, summary["run_id"], {"type": "completed"}, True, 0, "saving", None
    )
    assert event["type"] == "failed"
    assert "Explicit future origin" in event["message"]
    assert (
        ExperimentStore(config.output_directory).get_manifest(summary["run_id"])[
            "status"
        ]
        == "completed"
    )
    # A latest-origin worker may resolve to this new final timestamp.
    accepted = _reconcile(
        config, summary["run_id"], {"type": "completed"}, True, 0, "saving", None
    )
    assert accepted["type"] == "completed"


def remote_plan(config, tmp_path, monkeypatch, *, unknown_weight_hash=False):
    hub = FakeHub(tmp_path / "hub-cache")
    if unknown_weight_hash:

        def omit_weight_identity(info):
            entry = next(
                item for item in info.siblings if item.rfilename == "model.safetensors"
            )
            entry.lfs, entry.blob_id, entry.size = None, None, None
            return info

        hub.edit_metadata = omit_weight_identity
    monkeypatch.setattr(cp, "_hub", lambda: hub)
    reviewed = plan_checkpoint("amazon/chronos-bolt-tiny", "main", cache_dir=hub.root)
    checkpoint = prepare_checkpoint(reviewed)
    fields = config.resolved_dict()
    fields["model"] = {
        "source": "huggingface",
        "adapter_id": "chronos_bolt",
        "repository": reviewed.repository,
        "revision": reviewed.resolved_revision,
        "cache_dir": str(hub.root),
        "offline": True,
    }
    fields["runtime"]["dtype"] = "float32"
    config = future_config_from_mapping(
        fields, base_directory=config.source_path.parent
    )
    dataset = load_dataset(config.dataset)
    resolved = resolve_future(config, dataset)
    plan = GuidedRunPlan(
        resolved,
        reviewed,
        canonical_fingerprint(dataset),
        source_sha256=dataset.provenance["source_sha256"],
    )
    return plan, checkpoint


def test_reviewed_raw_pretty_json_is_not_confused_with_canonical_plan_json(
    config, tmp_path, monkeypatch
):
    plan, checkpoint = remote_plan(config, tmp_path, monkeypatch)
    raw = (checkpoint.path / "config.json").read_bytes()
    assert raw != plan.checkpoint._configuration_json
    validate_reviewed_checkpoint(plan, checkpoint.record(), checkpoint.fingerprint)


@pytest.mark.parametrize("unknown_weight_hash", [False, True])
def test_remote_recovery_checks_known_manifest_or_prepared_byte_fingerprint(
    config, tmp_path, monkeypatch, unknown_weight_hash
):
    plan, checkpoint = remote_plan(
        config, tmp_path, monkeypatch, unknown_weight_hash=unknown_weight_hash
    )
    changed = copy.deepcopy(checkpoint.record())
    changed["files"]["model.safetensors"]["sha256"] = "0" * 64
    changed["fingerprint"] = cp._fingerprint(changed)
    cp.validate_checkpoint_record(changed)
    prepared = checkpoint.fingerprint if unknown_weight_hash else changed["fingerprint"]
    with pytest.raises(ValueError, match="(approved preparation|reviewed identities)"):
        validate_reviewed_checkpoint(plan, changed, prepared)
