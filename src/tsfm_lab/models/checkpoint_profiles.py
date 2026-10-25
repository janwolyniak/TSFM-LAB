"""Offline configuration profiles for the explicitly supported model families."""

import math

CHECKPOINT_REPOSITORIES = {
    "chronos_bolt": ("amazon/chronos-bolt-tiny", "amazon/chronos-bolt-small"),
    "timesfm_2p5": ("google/timesfm-2.5-200m-pytorch",),
}
ADAPTER_LIMITS = {"chronos_bolt": (2048, 64), "timesfm_2p5": (16256, 128)}
_IDENTITIES = {
    "chronos_bolt": ("ChronosBoltModelForForecasting", "ChronosBoltPipeline", "t5"),
    "timesfm_2p5": (
        "TimesFmModelForPrediction",
        "TimesFM_2p5_200M_torch",
        "timesfm",
    ),
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
    """Validate runtime-compatible architecture and return schema-1 native fields."""
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ValueError("Checkpoint must declare one supported architecture")
    adapter = checkpoint_adapter(architectures[0])
    architecture, pipeline, model_type = _IDENTITIES[adapter]
    if config.get("model_type") != model_type:
        raise ValueError("Checkpoint model_type disagrees with its architecture")
    if adapter == "chronos_bolt":
        if config.get("chronos_pipeline_class") != pipeline:
            raise ValueError("Checkpoint is not a supported Chronos-Bolt pipeline")
        options = config["chronos_config"]
        if not isinstance(options, dict):
            raise ValueError("chronos_config must be an object")
        context, horizon = options["context_length"], options["prediction_length"]
        _bolt_bounds(context, horizon)
        quantiles = options["quantiles"]
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
    context, horizon = (
        record["native_context_length"],
        record["native_prediction_length"],
    )
    if adapter == "chronos_bolt":
        _bolt_bounds(context, horizon)
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
