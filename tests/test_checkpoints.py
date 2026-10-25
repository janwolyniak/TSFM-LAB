"""Small fake checkpoints and fake Hub clients never contact model services."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import tsfm_lab.models.checkpoints as checkpoints
from tsfm_lab.models.checkpoints import (
    CheckpointError,
    cached_checkpoints,
    inspect_local_checkpoint,
    pull_checkpoint,
    resolve_checkpoint,
    validate_checkpoint_record,
    verify_checkpoint,
)

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
CARD = "---\nlicense: apache-2.0\n---\n# Disposable test checkpoint\n"


def write_checkpoint(path, *, config=None, weights=True, card=True):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(CONFIG if config is None else config))
    if weights:
        (path / "model.safetensors").write_bytes(
            b"disposable test bytes, not model weights"
        )
    if card:
        (path / "README.md").write_text(CARD)
    return path


class FakeHub:
    def __init__(self, root):
        self.root = root
        self.config = copy.deepcopy(CONFIG)
        self.names = [
            "config.json",
            "model.safetensors",
            "README.md",
            "unsafe.py",
            "pytorch_model.bin",
        ]
        self.events = []
        self.returned_sha = SHA
        self.wrong_snapshot = None
        self.wrong_config = False

    def path(self, sha=SHA):
        return self.root / "models--amazon--chronos-bolt-tiny" / "snapshots" / sha

    def api(self, *, token):
        assert token is False
        return self

    def model_info(self, repository, **kwargs):
        assert repository == REPOSITORY and kwargs["token"] is False
        self.events.append(("metadata", kwargs))
        return SimpleNamespace(
            sha=self.returned_sha,
            siblings=[SimpleNamespace(rfilename=name) for name in self.names],
        )

    def download_config(self, repository, filename, **kwargs):
        assert repository == REPOSITORY and filename == "config.json"
        assert kwargs["token"] is False
        self.events.append(("config", kwargs))
        path = self.path(OTHER_SHA if self.wrong_config else kwargs["revision"])
        path.mkdir(parents=True, exist_ok=True)
        (path / filename).write_text(json.dumps(self.config))
        return str(path / filename)

    def snapshot(self, repository, **kwargs):
        assert repository == REPOSITORY and kwargs["token"] is False
        self.events.append(("snapshot", kwargs))
        if self.wrong_snapshot is not None:
            return str(self.wrong_snapshot)
        path = self.path(kwargs["revision"])
        if kwargs["local_files_only"]:
            if not path.exists():
                raise OSError("Checkpoint is not cached")
        else:
            path.mkdir(parents=True, exist_ok=True)
            for name in kwargs["allow_patterns"]:
                if name == "config.json":
                    (path / name).write_text(json.dumps(self.config))
                elif name == "model.safetensors":
                    (path / name).write_bytes(b"explicitly downloaded test bytes")
                elif name == "README.md":
                    (path / name).write_text(CARD)
                else:
                    raise AssertionError("Unexpected download")
        return str(path)

    def client(self):
        return SimpleNamespace(
            HfApi=self.api,
            hf_hub_download=self.download_config,
            snapshot_download=self.snapshot,
            cache_dir=self.root,
        )


@pytest.fixture
def hub(tmp_path, monkeypatch):
    fake = FakeHub(tmp_path / "cache")
    monkeypatch.setattr(checkpoints, "_hub", fake.client)
    return fake


def test_local_identity_records_consumed_files_native_bounds_and_license(tmp_path):
    path = write_checkpoint(tmp_path / "checkpoint")
    checkpoint = inspect_local_checkpoint(path)
    record = checkpoint.record()
    assert record["source"] == "local"
    assert record["path"] == str(path.resolve())
    assert record["repository"] is None and record["resolved_revision"] is None
    assert record["native_context_length"] == 2048
    assert record["native_prediction_length"] == 64
    assert record["license"] == "apache-2.0"
    assert set(record["files"]) == {"config.json", "model.safetensors", "README.md"}
    assert (
        record["files"]["model.safetensors"]["sha256"]
        == hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest()
    )
    assert "config" not in record
    assert validate_checkpoint_record(record) == record
    assert verify_checkpoint(checkpoint).record() == record


def test_record_fingerprint_is_independent_of_source_paths(tmp_path):
    a = inspect_local_checkpoint(write_checkpoint(tmp_path / "a"))
    b = inspect_local_checkpoint(write_checkpoint(tmp_path / "b"))
    assert a.fingerprint == b.fingerprint
    (b.path / "model.safetensors").write_bytes(b"different weights")
    changed = inspect_local_checkpoint(b.path)
    assert changed.fingerprint != a.fingerprint
    with pytest.raises(CheckpointError, match="changed"):
        verify_checkpoint(b)


@pytest.mark.parametrize(
    "field,value",
    [
        ("architectures", ["OtherModel"]),
        ("chronos_pipeline_class", "ChronosPipeline"),
        ("model_type", "another_type"),
        ("auto_map", {}),
        ("trust_remote_code", True),
    ],
)
def test_unsupported_architecture_and_remote_code_fail(tmp_path, field, value):
    config = copy.deepcopy(CONFIG)
    config[field] = value
    with pytest.raises(CheckpointError):
        inspect_local_checkpoint(
            write_checkpoint(tmp_path / "checkpoint", config=config)
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("context_length", True),
        ("context_length", 0),
        ("context_length", 2049),
        ("prediction_length", 65),
        ("prediction_length", 0),
        ("quantiles", []),
        ("quantiles", [0.1, 0.9]),
        ("quantiles", [0.5, 0.1]),
        ("quantiles", [0.5, 0.5]),
        ("quantiles", [False, 0.5]),
        ("quantiles", [0, 0.5, 1]),
    ],
)
def test_native_limits_and_quantiles_are_validated(tmp_path, field, value):
    config = copy.deepcopy(CONFIG)
    config["chronos_config"][field] = value
    with pytest.raises(CheckpointError):
        inspect_local_checkpoint(
            write_checkpoint(tmp_path / "checkpoint", config=config)
        )


def test_nested_remote_code_policy_is_rejected(tmp_path):
    config = copy.deepcopy(CONFIG)
    config["chronos_config"]["unrelated"] = {"auto_map": {"AutoModel": "code.Model"}}
    with pytest.raises(CheckpointError, match="remote code"):
        inspect_local_checkpoint(
            write_checkpoint(tmp_path / "checkpoint", config=config)
        )


def test_unsafe_serialization_is_never_a_weights_fallback(tmp_path):
    path = write_checkpoint(tmp_path / "checkpoint", weights=False)
    (path / "pytorch_model.bin").write_bytes(b"unsafe")
    with pytest.raises(CheckpointError, match="model.safetensors"):
        inspect_local_checkpoint(path)


def test_missing_or_invalid_json_config_is_rejected(tmp_path):
    path = write_checkpoint(tmp_path / "checkpoint")
    for raw in (b"[]", b"{broken", b"null", b'{"x": NaN}'):
        (path / "config.json").write_bytes(raw)
        with pytest.raises(CheckpointError):
            inspect_local_checkpoint(path)


def test_local_consumed_symlink_cannot_escape_selected_directory(tmp_path):
    path = write_checkpoint(tmp_path / "checkpoint")
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    (path / "model.safetensors").unlink()
    (path / "model.safetensors").symlink_to(outside)
    with pytest.raises(CheckpointError, match="escapes"):
        inspect_local_checkpoint(path)


def test_unknown_local_license_is_explicitly_absent(tmp_path):
    checkpoint = inspect_local_checkpoint(
        write_checkpoint(tmp_path / "checkpoint", card=False)
    )
    assert checkpoint.license is None


def test_mutated_or_forged_resolved_config_fails_reverification(tmp_path):
    checkpoint = inspect_local_checkpoint(write_checkpoint(tmp_path / "checkpoint"))
    checkpoint.config["auto_map"] = {"AutoModel": "external.Model"}
    with pytest.raises(CheckpointError, match="changed"):
        verify_checkpoint(checkpoint)
    with pytest.raises(CheckpointError):
        verify_checkpoint(replace(checkpoint, config={}))


def test_persisted_identity_validation_performs_no_file_or_hub_access(
    tmp_path, monkeypatch
):
    record = inspect_local_checkpoint(
        write_checkpoint(tmp_path / "checkpoint")
    ).record()
    monkeypatch.setattr(checkpoints, "_hub", lambda: pytest.fail("Hub access"))
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("File access"))
    assert validate_checkpoint_record(record) == record


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("source", "unknown"),
        ("path", "relative/path"),
        ("native_context_length", 128),
        ("license", "changed-license"),
        ("files", {"../outside": {"sha256": "a" * 64, "size": 10}}),
        ("fingerprint", "0" * 64),
    ],
)
def test_modified_checkpoint_records_fail_pure_validation(tmp_path, field, value):
    record = inspect_local_checkpoint(
        write_checkpoint(tmp_path / "checkpoint")
    ).record()
    record[field] = value
    with pytest.raises(CheckpointError):
        validate_checkpoint_record(record)


def test_offline_resolution_uses_exact_sha_and_never_metadata_or_downloads(hub):
    write_checkpoint(hub.path())
    checkpoint = resolve_checkpoint(REPOSITORY, SHA, cache_dir=hub.root)
    assert checkpoint.resolved_revision == SHA
    assert [event for event, _ in hub.events] == ["snapshot"]
    assert hub.events[0][1]["local_files_only"] is True


@pytest.mark.parametrize("revision", ["main", "tag", "a" * 7, "A" * 40, "", None])
def test_offline_resolution_rejects_moving_refs_before_hub_access(
    monkeypatch, revision
):
    monkeypatch.setattr(checkpoints, "_hub", lambda: pytest.fail("Hub access"))
    with pytest.raises(CheckpointError):
        resolve_checkpoint(REPOSITORY, revision)


def test_online_resolution_resolves_ref_but_still_cannot_download(hub):
    write_checkpoint(hub.path())
    checkpoint = resolve_checkpoint(
        REPOSITORY, "main", cache_dir=hub.root, offline=False
    )
    assert (
        checkpoint.requested_revision == "main" and checkpoint.resolved_revision == SHA
    )
    assert [event for event, _ in hub.events] == ["metadata", "snapshot"]
    assert hub.events[1][1]["revision"] == SHA
    assert hub.events[1][1]["local_files_only"] is True


def test_missing_cache_never_triggers_a_download(hub):
    with pytest.raises(OSError, match="not cached"):
        resolve_checkpoint(REPOSITORY, SHA, cache_dir=hub.root)
    assert hub.events[-1][1]["local_files_only"] is True
    assert not hub.root.exists()


def test_incomplete_cached_snapshot_never_triggers_a_download(hub):
    write_checkpoint(hub.path(), weights=False)
    with pytest.raises(CheckpointError, match="model.safetensors"):
        resolve_checkpoint(REPOSITORY, SHA, cache_dir=hub.root)
    assert [event for event, _ in hub.events] == ["snapshot"]


def test_cached_checkpoint_without_model_card_does_not_invent_license(hub):
    write_checkpoint(hub.path(), card=False)
    checkpoint = resolve_checkpoint(REPOSITORY, SHA, cache_dir=hub.root)
    assert checkpoint.license is None
    assert set(checkpoint.files) == {"config.json", "model.safetensors"}


def test_wrong_cached_snapshot_sha_is_rejected(hub):
    hub.wrong_snapshot = write_checkpoint(hub.path(OTHER_SHA))
    with pytest.raises(CheckpointError, match="exact SHA"):
        resolve_checkpoint(REPOSITORY, SHA, cache_dir=hub.root)


def test_explicit_pull_pins_every_download_and_limits_consumed_files(hub):
    checkpoint = pull_checkpoint(REPOSITORY, cache_dir=hub.root)
    assert (
        checkpoint.resolved_revision == SHA and checkpoint.requested_revision == "main"
    )
    assert [event for event, _ in hub.events] == ["metadata", "config", "snapshot"]
    assert hub.events[0][1]["files_metadata"] is True
    assert all(
        options["revision"] == SHA
        for event, options in hub.events
        if event != "metadata"
    )
    assert set(hub.events[-1][1]["allow_patterns"]) == {
        "config.json",
        "model.safetensors",
        "README.md",
    }
    assert not (checkpoint.path / "unsafe.py").exists()
    assert not (checkpoint.path / "pytorch_model.bin").exists()


def test_invalid_remote_config_stops_before_weights(hub):
    hub.config["auto_map"] = {"AutoModel": "external.Model"}
    with pytest.raises(CheckpointError, match="remote code"):
        pull_checkpoint(REPOSITORY, cache_dir=hub.root)
    assert [event for event, _ in hub.events] == ["metadata", "config"]
    assert not (hub.path() / "model.safetensors").exists()


def test_remote_unsafe_weights_only_fail_before_any_file_download(hub):
    hub.names = ["config.json", "pytorch_model.bin"]
    with pytest.raises(CheckpointError, match="unsafe weights"):
        pull_checkpoint(REPOSITORY, cache_dir=hub.root)
    assert [event for event, _ in hub.events] == ["metadata"]


def test_wrong_config_revision_is_rejected_before_weights(hub):
    hub.wrong_config = True
    with pytest.raises(CheckpointError, match="exact SHA"):
        pull_checkpoint(REPOSITORY, cache_dir=hub.root)
    assert [event for event, _ in hub.events] == ["metadata", "config"]


@pytest.mark.parametrize(
    "repository", ["unknown/model", "amazon/chronos-2", "amazon/chronos-bolt-base"]
)
def test_unknown_repositories_fail_before_optional_imports(monkeypatch, repository):
    monkeypatch.setattr(checkpoints, "_hub", lambda: pytest.fail("Hub access"))
    with pytest.raises(CheckpointError, match="curated"):
        pull_checkpoint(repository)
    with pytest.raises(CheckpointError, match="curated"):
        resolve_checkpoint(repository, SHA)


def test_hub_blob_symlinks_are_supported_but_arbitrary_external_files_are_not(
    hub, tmp_path
):
    path = write_checkpoint(hub.path())
    blobs = path.parent.parent / "blobs"
    blobs.mkdir()
    for name in ("config.json", "model.safetensors"):
        contents = (path / name).read_bytes()
        target = blobs / hashlib.sha256(contents).hexdigest()
        target.write_bytes(contents)
        (path / name).unlink()
        (path / name).symlink_to(target)
    checkpoint = resolve_checkpoint(REPOSITORY, SHA, cache_dir=hub.root)
    assert verify_checkpoint(checkpoint).fingerprint == checkpoint.fingerprint
    (path / "model.safetensors").unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    (path / "model.safetensors").symlink_to(outside)
    with pytest.raises(CheckpointError, match="escapes"):
        resolve_checkpoint(REPOSITORY, SHA, cache_dir=hub.root)


def test_cache_listing_is_local_and_reports_incomplete_known_repositories(hub):
    complete = write_checkpoint(hub.path())
    incomplete = write_checkpoint(hub.path(OTHER_SHA), weights=False)
    write_checkpoint(hub.root / "models--unknown--repo" / "snapshots" / SHA)
    entries = cached_checkpoints(hub.root)
    assert [entry["status"] for entry in entries] == ["cached", "incomplete"]
    assert entries[0]["bytes"] == sum(
        (complete / name).stat().st_size
        for name in ("config.json", "model.safetensors", "README.md")
    )
    assert entries[1]["path"] == str(incomplete)
    assert hub.events == []


def test_empty_explicit_cache_listing_needs_no_optional_imports(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "_hub", lambda: pytest.fail("Hub import"))
    assert cached_checkpoints(tmp_path / "absent") == []
    assert not (tmp_path / "absent").exists()


def test_missing_optional_hub_has_install_extra_guidance(monkeypatch):
    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(checkpoints.importlib, "import_module", missing)
    with pytest.raises(ImportError, match=r"tsfm-lab\[hub\]"):
        checkpoints._hub()


@pytest.mark.parametrize(
    "directive", ["transformers_weights", "configuration_files", "quantization_config"]
)
@pytest.mark.parametrize("nested", [False, True])
def test_loader_redirects_and_quantization_are_rejected_at_any_depth(
    tmp_path, directive, nested
):
    config = copy.deepcopy(CONFIG)
    loading = {directive: "../unrecorded.safetensors"}
    if nested:
        config["nested"] = [loading]
    else:
        config.update(loading)
    path = write_checkpoint(tmp_path / "checkpoint", config=config)
    (tmp_path / "unrecorded.safetensors").write_bytes(b"unrecorded weights")
    with pytest.raises(CheckpointError, match="loading directive"):
        inspect_local_checkpoint(path)


@pytest.mark.parametrize(
    "directive", ["transformers_weights", "configuration_files", "quantization_config"]
)
def test_pull_rejects_loader_directives_before_downloading_weights(hub, directive):
    hub.config["nested"] = [{directive: "unrecorded"}]
    with pytest.raises(CheckpointError, match="loading directive"):
        pull_checkpoint(REPOSITORY, cache_dir=hub.root)
    assert [event for event, _ in hub.events] == ["metadata", "config"]
    assert not (hub.path() / "model.safetensors").exists()


@pytest.mark.parametrize(
    "sidecar",
    [
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "adapter_model.safetensors.index.json",
    ],
)
@pytest.mark.parametrize("symlink", [False, True])
def test_added_peft_sidecars_invalidate_checkpoint_even_if_consumed_bytes_match(
    tmp_path, sidecar, symlink
):
    path = write_checkpoint(tmp_path / "checkpoint")
    checkpoint = inspect_local_checkpoint(path)
    before = copy.deepcopy(checkpoint.record())
    if symlink:
        (path / sidecar).symlink_to(tmp_path / "unavailable-sidecar")
    else:
        (path / sidecar).write_bytes(b"unrecorded adapter")
    assert checkpoint.record() == before
    with pytest.raises(CheckpointError, match="PEFT adapter sidecar"):
        inspect_local_checkpoint(path)
    with pytest.raises(CheckpointError, match="PEFT adapter sidecar"):
        verify_checkpoint(checkpoint)


def test_remote_peft_sidecar_metadata_is_rejected_before_file_downloads(hub):
    hub.names.append("adapter_config.json")
    with pytest.raises(CheckpointError, match="PEFT adapter sidecar"):
        pull_checkpoint(REPOSITORY, cache_dir=hub.root)
    assert [event for event, _ in hub.events] == ["metadata"]


def test_peft_sidecar_in_cached_snapshot_is_reported_as_incomplete(hub):
    path = write_checkpoint(hub.path())
    (path / "adapter_model.safetensors").write_bytes(b"unrecorded adapter")
    with pytest.raises(CheckpointError, match="PEFT adapter sidecar"):
        resolve_checkpoint(REPOSITORY, SHA, cache_dir=hub.root)
    assert cached_checkpoints(hub.root)[0]["status"] == "incomplete"


def test_oversized_config_is_bounded_before_any_weights_are_hashed(
    tmp_path, monkeypatch
):
    path = write_checkpoint(tmp_path / "checkpoint")
    raw = (path / "config.json").read_bytes() + b" " * (1024 * 1024)
    (path / "config.json").write_bytes(raw)
    monkeypatch.setattr(
        checkpoints,
        "_file",
        lambda *args, **kwargs: pytest.fail(
            "Weight or card hashing before config validation"
        ),
    )
    with pytest.raises(CheckpointError, match="1 MiB"):
        inspect_local_checkpoint(path)


def test_deeply_nested_json_becomes_a_checkpoint_error(tmp_path):
    path = write_checkpoint(tmp_path / "checkpoint")
    (path / "config.json").write_bytes(b'{"nested":' * 2000 + b"0" + b"}" * 2000)
    with pytest.raises(CheckpointError, match="nesting is too deep"):
        inspect_local_checkpoint(path)


def test_config_hash_binds_the_exact_validated_bytes_with_bounded_reads(
    tmp_path, monkeypatch
):
    path = write_checkpoint(tmp_path / "checkpoint")
    raw = b" \n" + (path / "config.json").read_bytes() + b"\n "
    (path / "config.json").write_bytes(raw)
    monkeypatch.setattr(
        Path, "read_bytes", lambda self: pytest.fail("Unbounded file read")
    )
    checkpoint = inspect_local_checkpoint(path)
    assert checkpoint.files["config.json"] == {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
    }


def test_oversized_remote_config_stops_before_weights(hub):
    hub.config["padding"] = "x" * (1024 * 1024)
    with pytest.raises(CheckpointError, match="1 MiB"):
        pull_checkpoint(REPOSITORY, cache_dir=hub.root)
    assert [event for event, _ in hub.events] == ["metadata", "config"]


def test_remote_config_size_metadata_prevents_oversized_config_download(
    hub, monkeypatch
):
    original = hub.model_info

    def metadata(*args, **kwargs):
        info = original(*args, **kwargs)
        for entry in info.siblings:
            if entry.rfilename == "config.json":
                entry.size = 1024 * 1024 + 1
        return info

    monkeypatch.setattr(hub, "model_info", metadata)
    with pytest.raises(CheckpointError, match="1 MiB"):
        pull_checkpoint(REPOSITORY, cache_dir=hub.root)
    assert [event for event, _ in hub.events] == ["metadata"]
