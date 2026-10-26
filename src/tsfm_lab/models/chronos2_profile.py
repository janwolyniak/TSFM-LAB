"""Native profiles from the pinned Small and standard Chronos-2 configurations."""

SMALL_QUANTILES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
STANDARD_QUANTILES = [
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
]
_CORE = {
    "d_kv": 64,
    "dense_act_fn": "relu",
    "dropout_rate": 0.1,
    "feed_forward_proj": "relu",
    "initializer_factor": 0.05,
    "is_gated_act": False,
    "layer_norm_epsilon": 1e-6,
    "pad_token_id": 0,
    "reg_token_id": 1,
    "rope_theta": 10000.0,
    "torch_dtype": "float32",
    "use_cache": True,
    "vocab_size": 2,
}
_FORECAST = {
    "context_length": 8192,
    "input_patch_size": 16,
    "input_patch_stride": 16,
    "max_output_patches": 64,
    "output_patch_size": 16,
    "time_encoding_scale": 8192,
    "use_arcsinh": True,
    "use_reg_token": True,
}


def _fixed(values, expected):
    if not isinstance(values, dict):
        raise ValueError("Chronos-2 configuration must be an object")
    for key, target in expected.items():
        if type(values.get(key)) is not type(target) or values[key] != target:
            raise ValueError(f"Unsupported Chronos-2 configuration: {key}")


def validate_config(config):
    """Reject unreviewed shape, normalization, patch, or quantile substitutions."""
    allowed = set(_CORE) | {
        "architectures",
        "chronos_pipeline_class",
        "model_type",
        "chronos_config",
        "d_model",
        "d_ff",
        "num_heads",
        "num_layers",
        "transformers_version",
        "_name_or_path",
        "_commit_hash",
    }
    # PretrainedConfig aliases can otherwise override validated dimensions later.
    if set(config) - allowed:
        raise ValueError("Unsupported Chronos-2 configuration fields")
    _fixed(config, _CORE)
    options = config.get("chronos_config")
    _fixed(options, _FORECAST)
    if set(options) != set(_FORECAST) | {"quantiles"}:
        raise ValueError("Unsupported Chronos-2 forecasting configuration fields")
    dimensions = tuple(
        config.get(key) for key in ("d_model", "d_ff", "num_heads", "num_layers")
    )
    if any(type(value) is not int for value in dimensions):
        raise ValueError("Chronos-2 dimensions must be integers")
    profiles = {
        (512, 2048, 8, 6): SMALL_QUANTILES,
        (768, 3072, 12, 12): STANDARD_QUANTILES,
    }
    if dimensions not in profiles or options.get("quantiles") != profiles[dimensions]:
        raise ValueError("Unsupported Chronos-2 dimensions or quantile profile")
    return 8192, 1024, options["quantiles"]


def validate_native(context, horizon, quantiles):
    """Validate persisted native fields without access to files or runtimes."""
    if (
        type(context) is not int
        or context != 8192
        or type(horizon) is not int
        or horizon != 1024
        or quantiles not in (SMALL_QUANTILES, STANDARD_QUANTILES)
    ):
        raise ValueError("Unsupported Chronos-2 checkpoint native profile")
