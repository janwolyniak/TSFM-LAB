"""Curated variants resolve through safe profiles without loading model weights."""

import copy
import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tsfm_lab.models import checkpoints, factory, registry, verification
from tsfm_lab.models.catalog import catalog_info, checkpoint_info

BASE = "amazon/chronos-bolt-base"
REVISION = "5d9f166d69f47aef3401367a7b842e78fe97b121"
CONFIG_SHA = "46f7e7e695569f67b06178faa4f236c281392945aa2d1bdf844442e57155530b"
WEIGHT_SHA = "31f875483a3215bc6880a0837ea608a13ce55f88ad90538c3cdd0b29aeb60b36"
# Exact metadata only, retrieved from the official pinned revision:
# https://huggingface.co/amazon/chronos-bolt-base/blob/5d9f166d69f47aef3401367a7b842e78fe97b121/config.json
# Apache-2.0 license declaration:
# https://huggingface.co/amazon/chronos-bolt-base/blob/5d9f166d69f47aef3401367a7b842e78fe97b121/README.md
BASE_CONFIG = b"""{
  "_name_or_path": "autogluon/chronos-bolt-base",
  "architectures": [
    "ChronosBoltModelForForecasting"
  ],
  "chronos_config": {
    "context_length": 2048,
    "input_patch_size": 16,
    "input_patch_stride": 16,
    "prediction_length": 64,
    "quantiles": [
      0.1,
      0.2,
      0.3,
      0.4,
      0.5,
      0.6,
      0.7,
      0.8,
      0.9
    ],
    "use_reg_token": true
  },
  "chronos_pipeline_class": "ChronosBoltPipeline",
  "classifier_dropout": 0.0,
  "d_ff": 3072,
  "d_kv": 64,
  "d_model": 768,
  "decoder_start_token_id": 0,
  "dense_act_fn": "relu",
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "relu",
  "initializer_factor": 0.05,
  "is_encoder_decoder": true,
  "is_gated_act": false,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "n_positions": 512,
  "num_decoder_layers": 12,
  "num_heads": 12,
  "num_layers": 12,
  "pad_token_id": 0,
  "reg_token_id": 1,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "torch_dtype": "float32",
  "transformers_version": "4.46.1",
  "use_cache": true,
  "vocab_size": 2
}
"""
INERT_WEIGHTS = b"inert Base fixture; not a serialized model"
CARD = b"---\nlicense: apache-2.0\n---\n# Disposable Base test checkpoint\n"
RUNTIME = {"device": "cpu", "dtype": "float32", "seed": 17}


def write_base(path):
    path.mkdir(parents=True)
    (path / "config.json").write_bytes(BASE_CONFIG)
    (path / "model.safetensors").write_bytes(INERT_WEIGHTS)
    (path / "README.md").write_bytes(CARD)
    return path


@pytest.fixture
def base_cache(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    path = write_base(
        root / "models--amazon--chronos-bolt-base" / "snapshots" / REVISION
    )
    events = []

    def metadata(repository, **options):
        assert repository == BASE and options == {"revision": "main", "token": False}
        events.append("metadata")
        return SimpleNamespace(sha=REVISION)

    def api(*, token):
        assert token is False
        return SimpleNamespace(model_info=metadata)

    def snapshot(repository, **options):
        assert repository == BASE
        assert options == {
            "revision": REVISION,
            "cache_dir": root,
            "local_files_only": True,
            "token": False,
            "allow_patterns": ["README.md", "config.json", "model.safetensors"],
        }
        events.append("cached_files")
        return str(path)

    def download(*args, **kwargs):
        pytest.fail("Catalog selection must not download a checkpoint")

    monkeypatch.setattr(
        checkpoints,
        "_hub",
        lambda: SimpleNamespace(
            HfApi=api,
            snapshot_download=snapshot,
            hf_hub_download=download,
            cache_dir=root,
        ),
    )
    return SimpleNamespace(root=root, path=path, events=events)


def capture_adapter(
    monkeypatch, module="tsfm_lab.models.chronos_bolt", name="ChronosBolt"
):
    imports = []

    class Adapter:
        def __init__(self, checkpoint, **runtime):
            self.checkpoint, self.runtime = checkpoint, runtime

        def load(self):
            pytest.fail("Selection must not load weights")

        def predict(self, *args):
            pytest.fail("Selection must not run inference")

    def import_module(selected):
        assert selected == module
        imports.append(selected)
        return SimpleNamespace(**{name: Adapter})

    monkeypatch.setattr(
        factory, "importlib", SimpleNamespace(import_module=import_module)
    )
    return Adapter, imports


def test_base_reference_is_pinned_and_has_no_individual_inference_claim():
    info = catalog_info(BASE)
    variant = checkpoint_info(BASE)
    assert info["display_name"] == "Chronos-Bolt"
    assert info["variant_display_name"] == variant["display_name"] == "Base"
    assert info["license"] == "apache-2.0"
    assert variant["reference_revision"] == REVISION
    assert variant["weight_bytes"] == 821203576
    assert variant["parameter_count"] == 205292928
    assert variant["project_inference_status"] == "not_recorded"
    assert variant["verification_label"] == (
        "Metadata/API compatibility checked; individual inference not run"
    )
    assert {item["repository"] for item in variant["representative_checks"]} == {
        "amazon/chronos-bolt-tiny",
        "amazon/chronos-bolt-small",
    }


def test_nested_variant_metadata_is_detached_from_catalog_and_project_evidence():
    before = checkpoint_info(BASE)
    altered = catalog_info("chronos_bolt")
    base = next(item for item in altered["variants"] if item["repository"] == BASE)
    base["representative_checks"][0]["runtime"]["torch"] = "changed"
    base["representative_checks"][0]["files"]["config.json"]["size"] = 1
    altered["runtime_requirements"]["torch"] = "changed"
    assert checkpoint_info(BASE) == before
    assert catalog_info("chronos_bolt")["runtime_requirements"]["torch"] == ">=2.8,<2.9"


@pytest.mark.parametrize("source", ["local", "cached", "moving_reference"])
def test_base_selection_uses_standard_factory_and_preserves_schema_one(
    source, base_cache, tmp_path, monkeypatch
):
    adapter, imports = capture_adapter(monkeypatch)
    if source == "local":
        selection = {
            "source": "local",
            "adapter_id": "chronos_bolt",
            "path": write_base(tmp_path / "standalone"),
        }
    else:
        selection = {
            "source": "huggingface",
            "adapter_id": "chronos_bolt",
            "repository": BASE,
            "revision": "main" if source == "moving_reference" else REVISION,
            "cache_dir": base_cache.root,
            "offline": source != "moving_reference",
        }
    model = factory.build_model(selection, RUNTIME)
    record = model.checkpoint.record()
    assert isinstance(model, adapter) and model.runtime == RUNTIME
    assert imports == ["tsfm_lab.models.chronos_bolt"]
    assert (
        base_cache.events
        == {
            "local": [],
            "cached": ["cached_files"],
            "moving_reference": ["metadata", "cached_files"],
        }[source]
    )
    assert len(BASE_CONFIG) == 1123
    assert hashlib.sha256(BASE_CONFIG).hexdigest() == CONFIG_SHA
    assert record["schema_version"] == 1
    assert record["files"]["config.json"] == {"sha256": CONFIG_SHA, "size": 1123}
    assert record["files"]["model.safetensors"]["sha256"] != WEIGHT_SHA
    assert record["native_context_length"] == 2048
    assert record["native_prediction_length"] == 64
    assert record["quantiles"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert (
        record["fingerprint"]
        == "cbe06842f001d1b70c0b9aa1d39053409a8b05b1daef4a9beb496f65344db0f8"
    )
    assert checkpoints.validate_checkpoint_record(record) == record
    assert "project_verification" not in record
    snapshot = verification.checkpoint_verification(record)
    assert snapshot["selected_checkpoint_checks"] == []
    assert snapshot["project_inference_status"] == "not_recorded"


def test_base_cache_inventory_records_inert_bytes_without_hub_access(base_cache):
    entries = checkpoints.cached_checkpoints(base_cache.root)
    assert len(entries) == 1 and base_cache.events == []
    record = entries[0]
    assert record["repository"] == BASE and record["resolved_revision"] == REVISION
    assert record["status"] == "cached" and record["license"] == "apache-2.0"
    assert record["files"]["model.safetensors"]["size"] == len(INERT_WEIGHTS)
    assert record["bytes"] == len(BASE_CONFIG) + len(INERT_WEIGHTS) + len(CARD)


def test_base_repository_rejects_another_family_before_weights_or_adapter_import(
    base_cache, timesfm_checkpoint, monkeypatch
):
    (base_cache.path / "config.json").write_bytes(
        (timesfm_checkpoint / "config.json").read_bytes()
    )
    monkeypatch.setattr(
        checkpoints, "_file", lambda *args, **kwargs: pytest.fail("Weights")
    )
    _, imports = capture_adapter(monkeypatch)
    with pytest.raises(
        checkpoints.CheckpointError, match="repository and architecture"
    ):
        factory.build_model(
            {
                "source": "huggingface",
                "adapter_id": "chronos_bolt",
                "repository": BASE,
                "revision": REVISION,
                "cache_dir": base_cache.root,
                "offline": True,
            },
            RUNTIME,
        )
    assert imports == []


@pytest.mark.parametrize(
    "filename,field,value",
    [
        ("config.json", "sha256", "0" * 64),
        ("model.safetensors", "sha256", WEIGHT_SHA),
        ("model.safetensors", "size", 821203576),
    ],
)
def test_repository_and_revision_do_not_replace_actual_tested_file_identity(
    base_cache, monkeypatch, filename, field, value
):
    record = checkpoints.resolve_checkpoint(
        BASE, REVISION, cache_dir=base_cache.root, offline=True
    ).record()
    evidence = {
        "id": "synthetic-base-file-matching-fixture-no-real-inference",
        "adapter_id": "chronos_bolt",
        "repository": BASE,
        "revision": REVISION,
        "files": {
            name: copy.deepcopy(record["files"][name])
            for name in ("config.json", "model.safetensors")
        },
        "source": {"package_version": None, "git_revision": None, "git_dirty": None},
        "runtime": {"python": "fixture"},
        "device": "cpu",
        "dtype": "float32",
        "checks": ["inert file-matching fixture; no model inference"],
        "reference": "docs/model-catalog.md",
    }
    evidence["files"][filename][field] = value
    monkeypatch.setattr(verification, "PROJECT_CHECKS", (evidence,))
    snapshot = verification.checkpoint_verification(record)
    assert snapshot["adapter_representative_checks"] == [evidence]
    assert snapshot["selected_checkpoint_checks"] == []
    assert snapshot["project_inference_status"] == "not_recorded"


def test_factory_dispatches_a_registered_profile_without_a_hardcoded_adapter_branch(
    base_cache, monkeypatch
):
    profile = replace(
        registry.adapter_profile("chronos_bolt"),
        adapter_id="registered_fixture",
        module="fixture.future_adapter",
        class_name="FutureAdapter",
    )
    monkeypatch.setattr(
        registry, "ADAPTER_PROFILES", (*registry.ADAPTER_PROFILES, profile)
    )
    adapter, imports = capture_adapter(monkeypatch, profile.module, profile.class_name)
    model = factory.build_model(
        {"source": "local", "adapter_id": profile.adapter_id, "path": base_cache.path},
        RUNTIME,
    )
    assert isinstance(model, adapter) and model.runtime == RUNTIME
    assert model.checkpoint.config["d_model"] == 768
    assert imports == [profile.module] and base_cache.events == []


def test_factory_rejects_an_unregistered_adapter_before_checkpoint_or_module_access(
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        pytest.fail("Unregistered adapters must fail before checkpoint/module access")

    monkeypatch.setattr(checkpoints, "inspect_local_checkpoint", forbidden)
    monkeypatch.setattr(factory, "importlib", SimpleNamespace(import_module=forbidden))
    with pytest.raises(ValueError, match="Unsupported checkpoint adapter"):
        factory.build_model(
            {"source": "local", "adapter_id": "unregistered", "path": "/unused"},
            RUNTIME,
        )
