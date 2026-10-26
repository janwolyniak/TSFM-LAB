"""Offline configuration profiles for the explicitly supported model families."""

import math

from tsfm_lab.models.registry import ADAPTER_PROFILES

CHECKPOINT_REPOSITORIES = {
    profile.adapter_id: tuple(
        dict.fromkeys(variant.repository for variant in profile.variants)
    )
    for profile in ADAPTER_PROFILES
}
ADAPTER_LIMITS = {
    profile.adapter_id: (profile.max_context, profile.max_horizon)
    for profile in ADAPTER_PROFILES
}
_IDENTITIES = {
    profile.adapter_id: (
        profile.architecture,
        profile.pipeline_class,
        profile.model_type,
    )
    for profile in ADAPTER_PROFILES
}
_TIMESFM_CONFIG = {
    "context_length": 16384,
    "head_dim": 80,
    "hidden_size": 1280,
    "horizon_length": 128,
    "intermediate_size": 1280,
    "num_attention_heads": 16,
    "num_hidden_layers": 20,
    "patch_length": 32,
    "quantile_horizon_length": 1024,
    "rms_norm_eps": 1e-6,
    "torch_compile": False,
}
_TIMESFM_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def checkpoint_adapter(architecture: str) -> str:
    """Recognize an architecture without loading its runtime or checkpoint files."""
    for adapter, identity in _IDENTITIES.items():
        if architecture == identity[0]:
            return adapter
    raise ValueError("Unsupported checkpoint architecture")


def repository_adapter(repository: str) -> str:
    for adapter, repositories in CHECKPOINT_REPOSITORIES.items():
        if repository in repositories:
            return adapter
    raise ValueError("Only curated checkpoint repositories are supported")


def _quantiles(values, *, exact=False):
    if (
        not isinstance(values, list)
        or not values
        or any(
            type(q) not in (int, float) or not math.isfinite(q) or not 0 < q < 1
            for q in values
        )
        or values != sorted(set(values))
        or 0.5 not in values
        or (exact and values != _TIMESFM_QUANTILES)
    ):
        raise ValueError("Quantiles must be ordered, unique, finite, and include 0.5")


def _bolt_bounds(context, horizon):
    if type(context) is not int or not 0 < context <= 2048:
        raise ValueError("Supported Bolt context length must be in 1..2048")
    if type(horizon) is not int or not 0 < horizon <= 64:
        raise ValueError("Supported Bolt prediction length must be in 1..64")


def config_metadata(config: dict) -> dict:
    """Validate runtime-compatible architecture and return native checkpoint fields."""
    if "stack_config" in config or "act_func" in config:
        from tsfm_lab.models.tirex2_profile import QUANTILES, validated_profile

        profile = validated_profile(config)
        architecture, pipeline, model_type = _IDENTITIES["tirex_2"]
        return {
            "architecture": architecture,
            "pipeline_class": pipeline,
            "model_type": model_type,
            "native_context_length": profile["context"],
            "native_prediction_length": profile["horizon"],
            "quantiles": QUANTILES.copy(),
        }
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ValueError("Checkpoint must declare one supported architecture")
    adapter = checkpoint_adapter(architectures[0])
    architecture, pipeline, model_type = _IDENTITIES[adapter]
    if config.get("model_type") != model_type:
        raise ValueError("Checkpoint model_type disagrees with its architecture")
    if adapter == "ttm_r2":
        from tsfm_lab.models.ttm_profile import validated_profile

        profile = validated_profile(config)
        return {
            "architecture": architecture,
            "pipeline_class": pipeline,
            "model_type": model_type,
            "native_context_length": profile["context"],
            "native_prediction_length": profile["horizon"],
            "quantiles": [],
            "point_forecast": profile["point_forecast"],
        }
    if adapter in {"chronos_bolt", "chronos_2"}:
        if config.get("chronos_pipeline_class") != pipeline:
            raise ValueError("Checkpoint is not a supported Chronos pipeline")
        options = config["chronos_config"]
        if not isinstance(options, dict):
            raise ValueError("chronos_config must be an object")
        if adapter == "chronos_bolt":
            context, horizon = options["context_length"], options["prediction_length"]
            _bolt_bounds(context, horizon)
            quantiles = options["quantiles"]
        else:
            from tsfm_lab.models.chronos2_profile import validate_config

            context, horizon, quantiles = validate_config(config)
    else:
        for key, expected in _TIMESFM_CONFIG.items():
            if type(config.get(key)) is not type(expected) or config[key] != expected:
                raise ValueError(f"Unsupported TimesFM 2.5 configuration: {key}")
        if "chronos_pipeline_class" in config or "chronos_config" in config:
            raise ValueError("TimesFM checkpoint cannot declare Chronos configuration")
        context, horizon = config["context_length"], config["horizon_length"]
        quantiles = config["quantiles"]
    _quantiles(quantiles, exact=adapter == "timesfm_2p5")
    return {
        "architecture": architecture,
        "pipeline_class": pipeline,
        "model_type": model_type,
        "native_context_length": context,
        "native_prediction_length": horizon,
        "quantiles": quantiles,
    }


def validate_record_profile(record: dict) -> str:
    """Validate persisted native metadata without depending on config-file access."""
    adapter = checkpoint_adapter(record["architecture"])
    _, pipeline, model_type = _IDENTITIES[adapter]
    if record["pipeline_class"] != pipeline or record["model_type"] != model_type:
        raise ValueError("Unsupported checkpoint record architecture")
    if (record["schema_version"] == 2) != (adapter == "ttm_r2"):
        raise ValueError("Checkpoint schema disagrees with its output representation")
    if adapter == "ttm_r2":
        from tsfm_lab.models.ttm_profile import validate_native

        validate_native(record)
        if record["source"] == "huggingface":
            if repository_adapter(record["repository"]) != adapter:
                raise ValueError("Checkpoint repository and architecture disagree")
        return adapter
    context, horizon = (
        record["native_context_length"],
        record["native_prediction_length"],
    )
    if adapter == "chronos_bolt":
        _bolt_bounds(context, horizon)
    elif adapter == "chronos_2":
        from tsfm_lab.models.chronos2_profile import validate_native

        validate_native(context, horizon, record["quantiles"])
    elif adapter == "tirex_2":
        from tsfm_lab.models.tirex2_profile import validate_native

        validate_native(context, horizon, record["quantiles"])
    elif (
        type(context) is not int
        or context != _TIMESFM_CONFIG["context_length"]
        or type(horizon) is not int
        or horizon != _TIMESFM_CONFIG["horizon_length"]
    ):
        raise ValueError("Unsupported TimesFM checkpoint native limits")
    _quantiles(record["quantiles"], exact=adapter == "timesfm_2p5")
    if record["source"] == "huggingface":
        if repository_adapter(record["repository"]) != adapter:
            raise ValueError("Checkpoint repository and architecture disagree")
    return adapter
