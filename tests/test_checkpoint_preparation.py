"""Reviewed checkpoint preparation uses fake Hub metadata and inert file bytes."""

import copy
import hashlib
import json
import pickle
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from tsfm_lab.models import checkpoints as cp
from tsfm_lab.models import preparation as prep

REPOSITORY = "amazon/chronos-bolt-tiny"
SHA = "a" * 40
OTHER_SHA = "b" * 40
CONFIG = {
    "architectures": ["ChronosBoltModelForForecasting"],
    "chronos_pipeline_class": "ChronosBoltPipeline",
    "model_type": "t5",
    "chronos_config": {
        "context_length": 2048,
        "prediction_length": 64,
        "quantiles": [0.1, 0.5, 0.9],
    },
}


def git_hash(raw):
    return hashlib.sha1(
        f"blob {len(raw)}\0".encode() + raw, usedforsecurity=False
    ).hexdigest()


class FakeHub:
    def __init__(self, root):
        self.root = root
        self.revision = SHA
        self.payloads = {
            "config.json": json.dumps(CONFIG).encode(),
            "model.safetensors": b"inert reviewed model bytes; never deserialized",
            "README.md": b"---\nlicense: apache-2.0\n---\nDisposable model card\n",
        }
        self.events = []
        self.edit_metadata = lambda info: info
        self.fail_file = None
        self.download_revision = None
        self.after_download = lambda name: None

    def snapshot(self, revision=SHA):
        return self.root / "models--amazon--chronos-bolt-tiny" / "snapshots" / revision

    def HfApi(self, *, token):
        assert token is False
        return self

    def model_info(self, repository, **options):
        assert repository == REPOSITORY and options["token"] is False
        assert options["files_metadata"] is True
        self.events.append(("metadata", options))
        siblings = []
        for name, raw in self.payloads.items():
            lfs = (
                {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                if name == "model.safetensors"
                else None
            )
            siblings.append(
                SimpleNamespace(
                    rfilename=name, size=len(raw), blob_id=git_hash(raw), lfs=lfs
                )
            )
        siblings.append(SimpleNamespace(rfilename="unreviewed.py"))
        return self.edit_metadata(SimpleNamespace(sha=self.revision, siblings=siblings))

    def hf_hub_download(self, repository, filename, **options):
        assert repository == REPOSITORY and options["token"] is False
        assert filename in self.payloads
        self.events.append((filename, options))
        if self.fail_file == filename:
            raise OSError("injected file transfer failure")
        destination = (
            self.snapshot(self.download_revision or options["revision"]) / filename
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.payloads[filename])
        self.after_download(filename)
        return str(destination)

    @property
    def cache_dir(self):
        return self.root


@pytest.fixture
def hub(tmp_path, monkeypatch):
    fake = FakeHub(tmp_path / "hub cache")
    monkeypatch.setattr(cp, "_hub", lambda: fake)
    return fake


def plan(hub, **kwargs):
    return prep.plan_checkpoint(REPOSITORY, "main", cache_dir=hub.root, **kwargs)


def test_empty_cache_review_downloads_only_config_and_execution_is_pinned(hub):
    reviewed = plan(hub)
    assert [name for name, _ in hub.events] == ["metadata", "config.json"]
    assert not (hub.snapshot() / "model.safetensors").exists()
    assert reviewed.resolved_revision == SHA
    assert reviewed.requested_revision == "main"
    assert reviewed.download_bytes == sum(
        len(hub.payloads[name]) for name in ("model.safetensors", "README.md")
    )
    assert reviewed.metadata["native_context_length"] == 2048
    assert reviewed.metadata["native_prediction_length"] == 64
    hub.revision = OTHER_SHA
    events = []
    checkpoint = prep.prepare_checkpoint(reviewed, progress=events.append)
    assert checkpoint.resolved_revision == SHA
    assert checkpoint.requested_revision == "main"
    assert [name for name, _ in hub.events].count("metadata") == 1
    assert all(
        options["revision"] == SHA for name, options in hub.events if name != "metadata"
    )
    assert set(checkpoint.files) == set(hub.payloads)
    assert not (hub.snapshot() / "unreviewed.py").exists()
    transfers = [event for event in events if event["type"] == "transfer"]
    assert transfers[-1]["completed_bytes"] == reviewed.download_bytes
    assert transfers[-1]["total_bytes"] == reviewed.download_bytes
    assert all("percent" not in event for event in events)


def test_plan_is_deeply_detached_frozen_and_picklable(hub):
    reviewed = plan(hub)
    with pytest.raises(FrozenInstanceError):
        reviewed.resolved_revision = OTHER_SHA
    with pytest.raises(FrozenInstanceError):
        reviewed.expected_files[0].name = "unsafe.py"
    reviewed.configuration["auto_map"] = {"unsafe": "code"}
    reviewed.metadata["native_context_length"] = 1
    summary = reviewed.to_dict()
    summary["expected_files"][0]["size"] = 1
    assert "auto_map" not in reviewed.configuration
    assert reviewed.metadata["native_context_length"] == 2048
    assert pickle.loads(pickle.dumps(reviewed)).to_dict() == reviewed.to_dict()
    assert set(reviewed.missing_files) == {"model.safetensors", "README.md"}


def test_declining_after_review_never_downloads_weights_or_publishes_receipt(hub):
    reviewed = plan(hub)
    assert reviewed.cached is False
    assert [name for name, _ in hub.events] == ["metadata", "config.json"]
    assert not prep._receipt_path(hub.root, REPOSITORY, SHA).exists()


def test_second_and_offline_execution_reuse_verified_cache_without_hub_import(
    hub, monkeypatch
):
    prepared = prep.prepare_checkpoint(plan(hub))
    hub.events.clear()
    monkeypatch.setattr(cp, "_hub", lambda: pytest.fail("Offline replay imported Hub"))
    reviewed = prep.plan_checkpoint(REPOSITORY, SHA, cache_dir=hub.root, offline=True)
    assert reviewed.cached and reviewed.download_bytes == 0
    assert prep.prepare_checkpoint(reviewed).record() == prepared.record() | {
        "requested_revision": SHA
    }
    assert hub.events == []


def test_cached_named_revision_is_read_only_and_needs_no_resolution(hub, monkeypatch):
    prep.prepare_checkpoint(plan(hub))
    refs = hub.root / "models--amazon--chronos-bolt-tiny" / "refs"
    refs.mkdir()
    (refs / "main").write_text(SHA)
    monkeypatch.setattr(
        cp, "_hub", lambda: pytest.fail("Unexpected metadata or transfer")
    )
    assert plan(hub, offline=True).resolved_revision == SHA


def test_verified_cached_preparation_needs_no_receipt_write(hub, monkeypatch):
    prep.prepare_checkpoint(plan(hub))
    reviewed = prep.plan_checkpoint(REPOSITORY, SHA, cache_dir=hub.root, offline=True)
    monkeypatch.setattr(
        prep,
        "_save_receipt",
        lambda *args: pytest.fail("Cached replay must remain read-only"),
    )
    assert prep.prepare_checkpoint(reviewed).resolved_revision == SHA


def test_offline_missing_cache_fails_without_optional_hub_import_or_writes(
    hub, monkeypatch
):
    monkeypatch.setattr(cp, "_hub", lambda: pytest.fail("Offline metadata request"))
    with pytest.raises(cp.CheckpointError, match="not fully cached"):
        prep.plan_checkpoint(REPOSITORY, SHA, cache_dir=hub.root, offline=True)
    assert not hub.root.exists()


def test_changed_cached_bytes_fail_against_retained_receipt(hub):
    prep.prepare_checkpoint(plan(hub))
    weights = hub.snapshot() / "model.safetensors"
    weights.write_bytes(b"x" * len(weights.read_bytes()))
    hub.events.clear()
    with pytest.raises(cp.CheckpointError, match="integrity receipt"):
        prep.plan_checkpoint(REPOSITORY, SHA, cache_dir=hub.root, offline=True)
    assert hub.events == []


def test_change_after_cached_review_is_rechecked_before_returning_checkpoint(hub):
    prep.prepare_checkpoint(plan(hub))
    reviewed = prep.plan_checkpoint(REPOSITORY, SHA, cache_dir=hub.root, offline=True)
    (hub.snapshot() / "model.safetensors").write_bytes(b"corrupted")
    with pytest.raises(cp.CheckpointError, match="differs"):
        prep.prepare_checkpoint(reviewed)


def test_newly_missing_reviewed_present_weight_requires_new_review_before_transfer(hub):
    hub.snapshot().mkdir(parents=True)
    (hub.snapshot() / "model.safetensors").write_bytes(
        hub.payloads["model.safetensors"]
    )
    (hub.snapshot() / "README.md").write_bytes(hub.payloads["README.md"])
    reviewed = plan(hub)
    assert reviewed.download_bytes == 0 and reviewed.missing_files == ()
    (hub.snapshot() / "model.safetensors").unlink()
    before = list(hub.events)
    with pytest.raises(cp.CheckpointError, match="cache changed after review"):
        prep.prepare_checkpoint(reviewed)
    assert hub.events == before


def test_newly_missing_config_stops_all_preparation_downloads(hub):
    reviewed = plan(hub)
    (hub.snapshot() / "config.json").unlink()
    before = list(hub.events)
    with pytest.raises(cp.CheckpointError, match="cache changed after review"):
        prep.prepare_checkpoint(reviewed)
    assert hub.events == before


def test_file_appearing_after_review_is_verified_and_reused_without_excess_bytes(hub):
    reviewed = plan(hub)
    (hub.snapshot() / "model.safetensors").write_bytes(
        hub.payloads["model.safetensors"]
    )
    events = []
    prep.prepare_checkpoint(reviewed, progress=events.append)
    assert [name for name, _ in hub.events] == ["metadata", "config.json", "README.md"]
    transfers = [event for event in events if event["type"] == "transfer"]
    assert transfers[-1]["completed_bytes"] == len(hub.payloads["README.md"])
    assert all(
        event["completed_bytes"] <= reviewed.download_bytes for event in transfers
    )


def test_invalid_file_size_fails_before_emitting_excess_transfer_bytes(hub):
    reviewed = plan(hub)
    hub.payloads["model.safetensors"] += b"unreviewed extra data"
    events = []
    with pytest.raises(cp.CheckpointError, match="size differs"):
        prep.prepare_checkpoint(reviewed, progress=events.append)
    assert all(
        event["completed_bytes"] <= event["total_bytes"]
        for event in events
        if event["type"] == "transfer"
    )


@pytest.mark.parametrize("metadata", ["size", "hash", "git"])
def test_supplied_file_identity_is_verified(hub, metadata):
    def change(info):
        target = next(
            item
            for item in info.siblings
            if item.rfilename
            == ("README.md" if metadata == "git" else "model.safetensors")
        )
        if metadata == "size":
            target.size += 1
            target.lfs["size"] += 1
        elif metadata == "hash":
            target.lfs["sha256"] = "0" * 64
        else:
            target.blob_id = "0" * 40
        return info

    hub.edit_metadata = change
    reviewed = plan(hub)
    with pytest.raises(
        cp.CheckpointError, match="(size differs|hash differs|Git blob hash)"
    ):
        prep.prepare_checkpoint(reviewed)
    assert not prep._receipt_path(hub.root, REPOSITORY, SHA).exists()


def test_unknown_weight_size_stays_unknown_in_plan_and_measured_transfer(hub):
    def unknown(info):
        weights = next(
            item for item in info.siblings if item.rfilename == "model.safetensors"
        )
        weights.size, weights.lfs, weights.blob_id = None, None, None
        return info

    hub.edit_metadata = unknown
    reviewed = plan(hub)
    assert reviewed.download_bytes is None and reviewed.total_bytes is None
    events = []
    prep.prepare_checkpoint(reviewed, progress=events.append)
    transfers = [event for event in events if event["type"] == "transfer"]
    assert transfers[-1]["completed_bytes"] == sum(
        len(hub.payloads[name]) for name in ("model.safetensors", "README.md")
    )
    assert all(event["total_bytes"] is None for event in transfers)


@pytest.mark.parametrize(
    "alteration",
    [
        "remote_code",
        "redirect",
        "family",
        "unsafe_weights",
        "peft",
        "oversized",
        "unknown_config_size",
    ],
)
def test_unsafe_or_unbounded_preparation_fails_before_weights(hub, alteration):
    config = copy.deepcopy(CONFIG)
    if alteration == "remote_code":
        config["auto_map"] = {"Model": "unreviewed.py"}
    elif alteration == "redirect":
        config["configuration_files"] = ["other.json"]
    elif alteration == "family":
        config["architectures"] = ["UnknownArchitecture"]
    hub.payloads["config.json"] = json.dumps(config).encode()

    def change(info):
        if alteration == "unsafe_weights":
            info.siblings = [
                item for item in info.siblings if item.rfilename != "model.safetensors"
            ]
        elif alteration == "peft":
            info.siblings.append(SimpleNamespace(rfilename="adapter_config.json"))
        elif alteration in {"oversized", "unknown_config_size"}:
            target = next(
                item for item in info.siblings if item.rfilename == "config.json"
            )
            target.size = 1024 * 1024 + 1 if alteration == "oversized" else None
        return info

    hub.edit_metadata = change
    with pytest.raises(cp.CheckpointError):
        plan(hub)
    assert "model.safetensors" not in [name for name, _ in hub.events]


def test_pinned_config_change_after_review_stops_before_weight_download(hub):
    reviewed = plan(hub)
    config = copy.deepcopy(CONFIG)
    config["chronos_config"]["context_length"] = 1024
    (hub.snapshot() / "config.json").write_text(json.dumps(config))
    with pytest.raises(cp.CheckpointError, match="(hash|size|config)"):
        prep.prepare_checkpoint(reviewed)
    assert "model.safetensors" not in [name for name, _ in hub.events]


def test_cancellation_before_and_after_config_has_no_weight_transfer(hub):
    with pytest.raises(KeyboardInterrupt):
        plan(hub, should_cancel=lambda: True)
    assert hub.events == []
    cancelled = False

    def after(name):
        nonlocal cancelled
        cancelled = True

    hub.after_download = after
    with pytest.raises(KeyboardInterrupt):
        plan(hub, should_cancel=lambda: cancelled)
    assert [name for name, _ in hub.events] == ["metadata", "config.json"]


def test_cancellation_between_files_leaves_no_verified_receipt(hub):
    reviewed = plan(hub)
    cancelled = False

    def after(name):
        nonlocal cancelled
        cancelled = name == "README.md"

    hub.after_download = after
    with pytest.raises(KeyboardInterrupt):
        prep.prepare_checkpoint(reviewed, should_cancel=lambda: cancelled)
    assert not (hub.snapshot() / "model.safetensors").exists()
    assert not prep._receipt_path(hub.root, REPOSITORY, SHA).exists()


def test_injected_transfer_failure_retains_no_verified_receipt_and_retry_is_explicit(
    hub,
):
    reviewed = plan(hub)
    hub.fail_file = "model.safetensors"
    with pytest.raises(OSError, match="injected"):
        prep.prepare_checkpoint(reviewed)
    assert not prep._receipt_path(hub.root, REPOSITORY, SHA).exists()
    hub.fail_file = None
    assert prep.prepare_checkpoint(reviewed).resolved_revision == SHA


def test_wrong_revision_download_path_is_rejected(hub):
    hub.download_revision = OTHER_SHA
    with pytest.raises(cp.CheckpointError, match="exact SHA"):
        plan(hub)


def test_exact_revision_metadata_cannot_substitute_another_commit(hub):
    hub.revision = OTHER_SHA
    with pytest.raises(cp.CheckpointError, match="different exact"):
        prep.plan_checkpoint(REPOSITORY, SHA, cache_dir=hub.root)
    assert [name for name, _ in hub.events] == ["metadata"]


def test_arbitrary_repository_or_path_revision_fails_before_hub(hub, monkeypatch):
    monkeypatch.setattr(cp, "_hub", lambda: pytest.fail("Unreviewed request"))
    for repository, revision in (("unknown/model", SHA), (REPOSITORY, "../../outside")):
        with pytest.raises(cp.CheckpointError):
            prep.plan_checkpoint(repository, revision, cache_dir=hub.root)


def test_manifest_dataclass_refuses_unsafe_file_names():
    with pytest.raises(cp.CheckpointError, match="Unreviewed"):
        prep.CheckpointFile("unreviewed.py", 10)


def test_cancellation_is_checked_inside_checkpoint_hashing(tmp_path):
    source = tmp_path / "model.safetensors"
    source.write_bytes(b"inert weights")
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks == 2

    with pytest.raises(KeyboardInterrupt):
        cp._file(source, tmp_path, hub=False, should_cancel=cancelled)
    assert checks == 2


def test_hub_blob_name_checks_detect_corruption_without_a_prior_receipt(hub):
    path = hub.snapshot()
    blobs = path.parent.parent / "blobs"
    path.mkdir(parents=True)
    blobs.mkdir()
    for name, raw in hub.payloads.items():
        digest = (
            hashlib.sha256(raw).hexdigest()
            if name == "model.safetensors"
            else git_hash(raw)
        )
        blob = blobs / digest
        blob.write_bytes(b"changed bytes" if name == "model.safetensors" else raw)
        (path / name).symlink_to(blob)
    with pytest.raises(cp.CheckpointError, match="blob hash mismatch"):
        prep.plan_checkpoint(REPOSITORY, SHA, cache_dir=hub.root, offline=True)
    assert hub.events == []
