"""Exact TOTO 2 4M configuration and conservative project integration bounds."""

import sys

import numpy as np

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MIN_CONTEXT, MAX_CONTEXT, MAX_HORIZON, PATCH_SIZE = 32, 2048, 128, 32
CONFIG = {
    "attn_bias": True,
    "d_ff": 688,
    "d_model": 256,
    "dropout_p": 0.0,
    "heads_per_group": 1,
    "layer_group_size": 4,
    "mlp_bias": False,
    "norm_eps": 0.0001,
    "norm_include_weight": False,
    "num_groups": 4,
    "num_heads": 4,
    "num_layers": 4,
    "num_output_patches": 1,
    "num_variate_layers_per_group": 1,
    "patch_size": 32,
    "per_dim_scale": True,
    "pre_norm": True,
    "qk_dim": 64,
    "qk_norm": False,
    "qk_norm_include_weight": False,
    "residual_attn_ratio": 5.136215466577748,
    "residual_mult": 0.75,
    "use_xpos": True,
    "v_dim": 64,
    "variate_layer_first": False,
}
FORECAST_OPTIONS = {
    "decode_block_size": 32,
    "has_missing_values": False,
    "scaler_fallback_min_obs": 0,
    "quantile_real_cap_k": 0.0,
    "normalization": "native_causal_std_asinh_inverse_v1",
    "quantile_policy": "native_sorted_knots_v1",
    "nonfinite_policy": "reject_v1",
    "bounds_policy": "project_32_2048_context_128_horizon_v1",
}


def ensure_supported_python():
    if not (3, 12) <= sys.version_info[:2] <= (3, 13):
        raise ImportError(
            "TOTO 2 requires Python 3.12 or 3.13; other families remain available"
        )


def validate_config(config):
    if (
        not isinstance(config, dict)
        or set(config) != set(CONFIG)
        or any(
            type(config[key]) is not type(value) or config[key] != value
            for key, value in CONFIG.items()
        )
    ):
        raise ValueError("Unsupported TOTO 2 4M configuration")


def validate_native(context, horizon, quantiles):
    # These schema-1 fields encode our bounded integration, not upstream limits.
    if (
        type(context) is not int
        or context != MAX_CONTEXT
        or type(horizon) is not int
        or horizon != MAX_HORIZON
        or type(quantiles) is not list
        or quantiles != QUANTILES
        or any(type(value) is not float for value in quantiles)
    ):
        raise ValueError("Unsupported TOTO 2 project integration profile")


def validate_bounds(context, horizon):
    if (
        type(context) is not int
        or not MIN_CONTEXT <= context <= MAX_CONTEXT
        or context % PATCH_SIZE
    ):
        raise ValueError(
            "TOTO 2 context_length must be a multiple of 32 between 32 and 2048; no padding is applied"
        )
    if type(horizon) is not int or not 1 <= horizon <= MAX_HORIZON:
        raise ValueError("TOTO 2 horizon must be between 1 and 128")


def validate_values(values):
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or np.any(np.abs(values) > np.finfo(np.float32).max)
    ):
        raise ValueError("TOTO 2 history must be finite and representable as float32")


def validate_runtime_options(options):
    if (
        not isinstance(options, dict)
        or set(options) != set(FORECAST_OPTIONS)
        or any(
            type(options[key]) is not type(value) or options[key] != value
            for key, value in FORECAST_OPTIONS.items()
        )
    ):
        raise ValueError(
            "TOTO 2 runtime requires its recorded forecast and numerical policies"
        )
