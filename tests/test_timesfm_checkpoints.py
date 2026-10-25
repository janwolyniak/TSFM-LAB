"""Family-specific checkpoint identity and download boundaries use fake weights."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tsfm_lab.models.checkpoints as checkpoints
from tsfm_lab.models.checkpoint_profiles import checkpoint_adapter
from tsfm_lab.models.checkpoints import (
    CheckpointError,
    cached_checkpoints,
    inspect_local_checkpoint,
    pull_checkpoint,
    resolve_checkpoint,
    validate_checkpoint_record,
    verify_checkpoint,
)

REPOSITORY = "google/timesfm-2.5-200m-pytorch"
SHA = "a" * 40
CONFIG = {
    "architectures": ["TimesFmModelForPrediction"],
    "context_length": 16384,
    "head_dim": 80,
    "hidden_size": 1280,
    "horizon_length": 128,
    "intermediate_size": 1280,
    "model_type": "timesfm",
    "num_attention_heads": 16,
    "num_hidden_layers": 20,
    "patch_length": 32,
    "quantile_horizon_length": 1024,
    "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "rms_norm_eps": 1e-6,
    "torch_compile": False,
}


def write_checkpoint(path, config=None):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(CONFIG if config is None else config))
    (path / "model.safetensors").write_bytes(b"synthetic weights, never deserialized")
    return path


class FakeHub:
    def __init__(self, root):
        self.root = root
        self.events = []
        self.config = copy.deepcopy(CONFIG)

    def path(self, repository=REPOSITORY):
        return (
            self.root / ("models--" + repository.replace("/", "--")) / "snapshots" / SHA
        )

    def api(self, *, token):
        assert token is False
        return self

    def model_info(self, repository, **options):
        self.events.append(("metadata", repository, options))
        return SimpleNamespace(
            sha=SHA,
            siblings=[
                SimpleNamespace(rfilename=name, size=100)
                for name in ["config.json", "model.safetensors", "unreviewed.py"]
            ],
        )

    def config_file(self, repository, filename, **options):
        self.events.append(("config", repository, options))
        path = self.path(repository)
        path.mkdir(parents=True, exist_ok=True)
        (path / filename).write_text(json.dumps(self.config))
        return str(path / filename)

    def snapshot(self, repository, **options):
        self.events.append(("snapshot", repository, options))
        path = self.path(repository)
        if not options["local_files_only"]:
            write_checkpoint(path, self.config)
        return str(path)

    def client(self):
        return SimpleNamespace(
            HfApi=self.api,
            hf_hub_download=self.config_file,
            snapshot_download=self.snapshot,
            cache_dir=self.root,
        )


@pytest.fixture
def hub(tmp_path, monkeypatch):
    hub = FakeHub(tmp_path / "hub-cache")
    monkeypatch.setattr(checkpoints, "_hub", hub.client)
    return hub


def test_timesfm_identity_retains_native_limits_and_explicit_runtime_class(tmp_path):
    checkpoint = inspect_local_checkpoint(write_checkpoint(tmp_path / "model"))
    record = checkpoint.record()
    assert record["schema_version"] == 1
    assert record["architecture"] == "TimesFmModelForPrediction"
    assert record["pipeline_class"] == "TimesFM_2p5_200M_torch"
    assert record["model_type"] == "timesfm"
    assert record["native_context_length"] == 16384
    assert record["native_prediction_length"] == 128
    assert record["quantiles"] == CONFIG["quantiles"]
    assert record["license"] is None
    assert checkpoint_adapter(record["architecture"]) == "timesfm_2p5"
    assert validate_checkpoint_record(record) == record
    assert verify_checkpoint(checkpoint).record() == record


@pytest.mark.parametrize(
    "field,value",
    [
        ("architectures", ["TimesFm3Model"]),
        ("model_type", "t5"),
        ("context_length", 8192),
        ("horizon_length", 1024),
        ("quantile_horizon_length", 128),
        ("hidden_size", 2560),
        ("num_hidden_layers", 50),
        ("patch_length", 64),
        ("rms_norm_eps", 0.0),
        ("torch_compile", True),
        ("torch_compile", 0),
        ("chronos_config", {}),
        ("chronos_pipeline_class", "ChronosBoltPipeline"),
        ("quantiles", [0.1, 0.5, 0.9]),
        ("quantiles", [False, 0.5]),
    ],
)
def test_timesfm_requires_the_fixed_supported_model_profile(tmp_path, field, value):
    config = copy.deepcopy(CONFIG)
    config[field] = value
    with pytest.raises(CheckpointError):
        inspect_local_checkpoint(write_checkpoint(tmp_path / "model", config))


@pytest.mark.parametrize("field", list(CONFIG))
def test_timesfm_rejects_missing_required_native_configuration(tmp_path, field):
    config = copy.deepcopy(CONFIG)
    del config[field]
    with pytest.raises(CheckpointError):
        inspect_local_checkpoint(write_checkpoint(tmp_path / "model", config))


def test_timesfm_record_reload_is_pure_and_preserves_consumed_file_identity(
    tmp_path, monkeypatch
):
    record = inspect_local_checkpoint(write_checkpoint(tmp_path / "model")).record()
    monkeypatch.setattr(checkpoints, "_hub", lambda: pytest.fail("Unexpected Hub"))
    monkeypatch.setattr(
        Path, "open", lambda *a, **k: pytest.fail("Unexpected file read")
    )
    assert validate_checkpoint_record(record) == record
    mutated = copy.deepcopy(record)
    mutated["files"]["model.safetensors"]["sha256"] = "0" * 64
    with pytest.raises(CheckpointError, match="fingerprint"):
        validate_checkpoint_record(mutated)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pipeline_class", "ChronosBoltPipeline"),
        ("model_type", "t5"),
        ("native_context_length", 16256),
        ("native_prediction_length", 1024),
        ("quantiles", [0.1, 0.5, 0.9]),
    ],
)
def test_forged_timesfm_metadata_fails_even_with_recomputed_fingerprint(
    tmp_path, field, value
):
    record = inspect_local_checkpoint(write_checkpoint(tmp_path / "model")).record()
    record[field] = value
    record["fingerprint"] = checkpoints._fingerprint(record)
    with pytest.raises(CheckpointError):
        validate_checkpoint_record(record)


def test_timesfm_offline_resolution_and_cache_inventory_never_resolve_metadata(hub):
    write_checkpoint(hub.path())
    checkpoint = resolve_checkpoint(REPOSITORY, SHA, cache_dir=hub.root)
    assert [event[0] for event in hub.events] == ["snapshot"]
    assert hub.events[0][2]["local_files_only"] is True
    assert checkpoint.resolved_revision == SHA
    entries = cached_checkpoints(hub.root)
    assert len(entries) == 1
    assert entries[0]["status"] == "cached"
    assert entries[0]["repository"] == REPOSITORY


def test_timesfm_explicit_pull_validates_pinned_config_before_weights(hub):
    checkpoint = pull_checkpoint(REPOSITORY, "main", cache_dir=hub.root)
    assert [event[0] for event in hub.events] == ["metadata", "config", "snapshot"]
    assert checkpoint.resolved_revision == SHA
    assert hub.events[-1][2]["allow_patterns"] == ["config.json", "model.safetensors"]
    assert all(event[2]["token"] is False for event in hub.events)
    assert all(event[2]["revision"] == SHA for event in hub.events[1:])
    assert not (hub.path() / "unreviewed.py").exists()


def test_valid_timesfm_config_cannot_be_pulled_through_a_bolt_repository(hub):
    with pytest.raises(CheckpointError, match="repository and architecture"):
        pull_checkpoint("amazon/chronos-bolt-tiny", cache_dir=hub.root)
    assert [event[0] for event in hub.events] == ["metadata", "config"]
    assert not (hub.path("amazon/chronos-bolt-tiny") / "model.safetensors").exists()


def test_cached_timesfm_config_with_wrong_family_repository_is_incomplete(hub):
    write_checkpoint(hub.path("amazon/chronos-bolt-tiny"))
    with pytest.raises(CheckpointError, match="repository and architecture"):
        resolve_checkpoint("amazon/chronos-bolt-tiny", SHA, cache_dir=hub.root)
    assert cached_checkpoints(hub.root)[0]["status"] == "incomplete"


def test_timesfm_hub_record_rejects_a_repository_from_the_other_family(hub):
    write_checkpoint(hub.path())
    record = resolve_checkpoint(REPOSITORY, SHA, cache_dir=hub.root).record()
    record["repository"] = "amazon/chronos-bolt-tiny"
    record["path"] = str(hub.path(record["repository"]))
    with pytest.raises(CheckpointError, match="repository and architecture"):
        validate_checkpoint_record(record)


def test_wrong_timesfm_remote_profile_fails_before_weights(hub):
    hub.config["patch_length"] = 16
    with pytest.raises(CheckpointError, match="patch_length"):
        pull_checkpoint(REPOSITORY, cache_dir=hub.root)
    assert [event[0] for event in hub.events] == ["metadata", "config"]


def test_timesfm_nested_loading_redirect_remains_unsupported(hub):
    hub.config["extra"] = {"configuration_files": ["other_config.json"]}
    with pytest.raises(CheckpointError, match="loading directive"):
        pull_checkpoint(REPOSITORY, cache_dir=hub.root)
    assert [event[0] for event in hub.events] == ["metadata", "config"]
