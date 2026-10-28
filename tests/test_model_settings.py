"""The shared settings contract rejects unsupported choices before preparation."""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from tsfm_lab.config import ConfigError, config_from_mapping, load_config
from tsfm_lab.contracts import EvaluationSpec
from tsfm_lab.models.registry import ADAPTER_PROFILES
from tsfm_lab.models.settings import model_settings, validate_settings

VARIANTS = [
    (profile.adapter_id, variant)
    for profile in ADAPTER_PROFILES
    for variant in profile.variants
]


def spec(context=8, horizon=4, task="median"):
    return EvaluationSpec(
        context,
        horizon,
        "2026-01-01T00:00:00Z",
        "2026-01-02T00:00:00Z",
        point_statistic=task,
    )


def checkpoint_model(adapter, variant):
    return {
        "source": "huggingface",
        "adapter_id": adapter,
        "repository": variant.repository,
        "revision": variant.reference_revision,
        "options": {},
    }


@pytest.mark.parametrize(
    "adapter,label",
    [("last_value", "Last value"), ("seasonal_naive", "Seasonal naive")],
)
def test_baselines_have_meaningful_defaults_and_no_checkpoint_options(adapter, label):
    settings = model_settings(adapter)
    assert settings.family_label == "Baselines"
    assert settings.generation_label == "Deterministic"
    assert settings.variant_label == label
    assert settings.variant_kind == "baseline"
    assert settings.default_context == 8
    assert settings.default_horizon == 4
    assert settings.supported_tasks == ("median", "native_point")
    with pytest.raises(FrozenInstanceError):
        settings.default_context = 20
    with pytest.raises(ValueError, match="do not accept checkpoint"):
        model_settings(adapter, repository="example/model")


@pytest.mark.parametrize("adapter,variant", VARIANTS)
def test_every_reference_sha_and_explicit_variant_resolve_to_same_description(
    adapter, variant
):
    explicit = model_settings(
        adapter, repository=variant.repository, variant=variant.variant
    )
    by_revision = model_settings(
        adapter, repository=variant.repository, revision=variant.reference_revision
    )
    assert explicit == by_revision
    assert explicit.variant_label == variant.display_name
    validate_settings(
        checkpoint_model(adapter, variant),
        spec(
            explicit.default_context,
            explicit.default_horizon,
            explicit.supported_tasks[0],
        ),
        "1h",
    )


@pytest.mark.parametrize("adapter,variant", VARIANTS)
def test_variant_limits_reject_unsupported_horizon_and_context_before_loading(
    adapter, variant
):
    settings = model_settings(
        adapter, repository=variant.repository, variant=variant.variant
    )
    caps = settings.capabilities
    selection = checkpoint_model(adapter, variant)
    boundary = spec(caps.max_context, caps.max_horizon, settings.supported_tasks[0])
    validate_settings(selection, boundary, "1h")
    with pytest.raises(ValueError, match="horizon is outside"):
        validate_settings(
            selection, replace(boundary, horizon=caps.max_horizon + 1), "1h"
        )
    with pytest.raises(ValueError, match="context_length is outside"):
        validate_settings(
            selection, replace(boundary, context_length=caps.max_context + 1), "1h"
        )


@pytest.mark.parametrize(
    "profile_name,context",
    [("512-96-r2", 512), ("1024-96-r2", 1024), ("1536-96-r2", 1536)],
)
def test_ttm_profile_has_fixed_context_required_scaling_and_native_point_task(
    profile_name, context
):
    repository = "ibm-granite/granite-timeseries-ttm-r2"
    settings = model_settings("ttm_r2", repository=repository, variant=profile_name)
    assert (
        settings.capabilities.min_context
        == settings.capabilities.max_context
        == context
    )
    assert settings.default_context == context
    assert settings.supported_tasks == ("native_point",)
    assert settings.variant_kind == "context_horizon"
    assert "Required context-only population z-score" in settings.preprocessing
    variant = next(
        variant
        for adapter, variant in VARIANTS
        if adapter == "ttm_r2" and variant.variant == profile_name
    )
    model = checkpoint_model("ttm_r2", variant)
    with pytest.raises(ValueError, match="requires exactly"):
        validate_settings(model, spec(context - 1, task="native_point"), "1h")
    with pytest.raises(ValueError, match="unsupported by this model"):
        validate_settings(model, spec(context, task="median"), "1h")
    for frequency in ("1min", "15min", "1h"):
        validate_settings(model, spec(context, task="native_point"), frequency)
    for frequency in ("30s", "1D"):
        with pytest.raises(ValueError, match="one minute to one hour"):
            validate_settings(model, spec(context, task="native_point"), frequency)


def test_unknown_revision_and_local_checkpoint_do_not_claim_known_native_profile():
    repository = "ibm-granite/granite-timeseries-ttm-r2"
    local = model_settings("ttm_r2")
    unknown = model_settings(
        "ttm_r2", repository=repository, revision="a" * 40, variant="1024-96-r2"
    )
    assert local == unknown
    assert local.variant_label == "Unresolved checkpoint"
    assert local.capabilities.min_context == 512
    assert local.capabilities.max_context == 1536
    validate_settings(
        {"adapter_id": "ttm_r2", "source": "local", "path": "/not/read"},
        spec(1024, task="native_point"),
        "1h",
    )


def test_training_profiles_are_not_sizes_and_tirex_native_limits_differ():
    ordinary = model_settings("tirex_2", repository="NX-AI/TiRex-2", variant="main")
    gift = model_settings(
        "tirex_2", repository="NX-AI/TiRex-2-gifteval-zs", variant="gifteval-zs"
    )
    assert ordinary.variant_kind == gift.variant_kind == "training_profile"
    assert ordinary.capabilities.max_context == 2048
    assert ordinary.capabilities.max_horizon == 320
    assert gift.capabilities.max_context == 8192
    assert gift.capabilities.max_horizon == 928
    for settings in (ordinary, gift):
        assert "Fixed native profile preprocessing" in settings.preprocessing
        assert "inverse to original units" in settings.preprocessing
        assert "differencing" not in settings.preprocessing


def test_invalid_repository_variant_and_conflicting_known_revision_are_rejected():
    with pytest.raises(ValueError, match="curated repository"):
        model_settings("chronos_bolt", repository="amazon/chronos-2")
    with pytest.raises(ValueError, match="before its checkpoint profile"):
        model_settings("chronos_bolt", variant="tiny")
    with pytest.raises(ValueError, match="curated checkpoint profile"):
        model_settings(
            "chronos_bolt", repository="amazon/chronos-bolt-tiny", variant="base"
        )
    repository = "ibm-granite/granite-timeseries-ttm-r2"
    with pytest.raises(ValueError, match="disagrees"):
        model_settings(
            "ttm_r2",
            repository=repository,
            variant="512-96-r2",
            revision="1a382618f3c5b63836281ad632e54724e55e85a5",
        )


def test_baseline_season_is_validated_without_conflating_mase_period():
    model = {"adapter_id": "seasonal_naive", "options": {"seasonal_period": 24}}
    validate_settings(model, replace(spec(24), seasonal_period=500), "1D")
    with pytest.raises(ValueError, match="complete baseline season"):
        validate_settings(model, spec(23), "1D")


def test_config_entrypoint_rejects_fixed_ttm_constraints_and_unsupported_transforms(
    tmp_path,
):
    variant = next(variant for adapter, variant in VARIANTS if adapter == "ttm_r2")
    mapping = {
        "schema_version": 1,
        "model": checkpoint_model("ttm_r2", variant),
        "dataset": {
            "path": "unread.csv",
            "timestamp_column": "timestamp",
            "target_column": "target",
            "frequency": "1h",
        },
        "evaluation": {
            "context_length": 512,
            "horizon": 4,
            "evaluation_start": "2026-01-01T00:00:00Z",
            "evaluation_end": "2026-01-02T00:00:00Z",
            "point_statistic": "native_point",
        },
    }
    original = config_from_mapping(mapping, base_directory=tmp_path)
    assert original.model == mapping["model"] | {"offline": True}
    assert "settings" not in original.resolved_dict()
    assert not (tmp_path / "unread.csv").exists()
    mapping["evaluation"]["context_length"] = 513
    with pytest.raises(ConfigError, match="requires exactly 512"):
        config_from_mapping(mapping, base_directory=tmp_path)
    mapping["evaluation"]["context_length"] = 512
    mapping["dataset"]["frequency"] = "1D"
    with pytest.raises(ConfigError, match="one minute to one hour"):
        config_from_mapping(mapping, base_directory=tmp_path)
    mapping["dataset"]["frequency"] = "1h"
    mapping["model"]["options"] = {"scaling": False}
    with pytest.raises(ConfigError, match="Unknown model.options"):
        config_from_mapping(mapping, base_directory=tmp_path)


def test_existing_yaml_keeps_schema_and_numerical_settings():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "examples/configurations/last-value.yaml")
    resolved = config.resolved_dict()
    assert set(resolved) == {
        "schema_version",
        "name",
        "model",
        "dataset",
        "evaluation",
        "runtime",
        "output",
    }
    assert resolved["schema_version"] == 1
    assert resolved["evaluation"]["context_length"] == 8
    assert resolved["evaluation"]["horizon"] == 4
    assert resolved["model"] == {
        "source": "builtin",
        "adapter_id": "last_value",
        "options": {},
    }


def test_settings_can_be_imported_and_validated_without_optional_model_dependencies():
    code = """
import importlib.abc
import socket
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'torch', 'chronos', 'timesfm', 'tirex2', 'tsfm_public', 'transformers', 'huggingface_hub'}:
            raise AssertionError('Unexpected optional import: ' + fullname)
def forbidden(*args, **kwargs):
    raise AssertionError('Unexpected network access')
sys.meta_path.insert(0, Block())
socket.socket = socket.create_connection = forbidden
from tsfm_lab.models.settings import model_settings, validate_settings
from tsfm_lab.contracts import EvaluationSpec
settings = model_settings('ttm_r2', repository='ibm-granite/granite-timeseries-ttm-r2', variant='512-96-r2')
assert settings.default_context == 512
validate_settings({'adapter_id': 'ttm_r2'}, EvaluationSpec(512, 4, '2026-01-01T00:00:00Z', '2026-01-02T00:00:00Z', point_statistic='native_point'), '1h')
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, PYTHONPATH=str(root / "src")),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
