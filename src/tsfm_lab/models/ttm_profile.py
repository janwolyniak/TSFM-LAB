"""Reviewed R2 MSE profiles and point-only checkpoint semantics, without runtimes."""

import copy
import json

import pandas as pd

from tsfm_lab.contracts import fixed_interval

_BASE = {
    "adaptive_patching_levels": 3,
    "architectures": ["TinyTimeMixerForPrediction"],
    "categorical_vocab_size_list": None,
    "context_length": 512,
    "d_model": 192,
    "d_model_scale": 3,
    "decoder_adaptive_patching_levels": 0,
    "decoder_d_model": 128,
    "decoder_d_model_scale": 2,
    "decoder_mode": "common_channel",
    "decoder_num_layers": 2,
    "decoder_raw_residual": False,
    "distribution_output": "student_t",
    "dropout": 0.4,
    "enable_forecast_channel_mixing": False,
    "exogenous_channel_indices": None,
    "expansion_factor": 2,
    "fcm_context_length": 1,
    "fcm_gated_attn": True,
    "fcm_mix_layers": 3,
    "fcm_prepend_past": True,
    "fcm_use_mixer": True,
    "frequency_token_vocab_size": 5,
    "gated_attn": True,
    "head_dropout": 0.4,
    "init_embed": "pytorch",
    "init_linear": "pytorch",
    "init_processing": True,
    "init_std": 0.02,
    "loss": "mse",
    "mode": "common_channel",
    "model_type": "tinytimemixer",
    "norm_eps": 1e-05,
    "norm_mlp": "LayerNorm",
    "num_input_channels": 1,
    "num_layers": 2,
    "num_parallel_samples": 100,
    "num_patches": 8,
    "patch_last": True,
    "patch_length": 64,
    "patch_stride": 64,
    "positional_encoding_type": "sincos",
    "post_init": False,
    "prediction_channel_indices": None,
    "prediction_filter_length": None,
    "prediction_length": 96,
    "resolution_prefix_tuning": False,
    "scaling": "std",
    "self_attn": False,
    "self_attn_heads": 1,
    "stride_ratio": 1,
    "torch_dtype": "float32",
    "transformers_version": "4.37.2",
    "use_decoder": True,
    "use_positional_encoding": False,
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _configuration(context):
    config = copy.deepcopy(_BASE)
    if context not in (512, 1024, 1536):
        raise ValueError("Unsupported TTM R2 context profile")
    config["context_length"] = context
    if context != 512:
        config.update(
            d_model=384,
            decoder_d_model=256,
            frequency_token_vocab_size=8,
            patch_length=128,
            patch_stride=128,
        )
    if context == 1536:
        config["num_patches"] = 12
    return config


def point_metadata(context):
    if type(context) is not int or context not in (512, 1024, 1536):
        raise ValueError("Unsupported TTM R2 context profile")
    return {
        "statistic": "native_point",
        "training_loss": "mse",
        "release": "r2",
        "profile": f"{context}-96-r2",
        "frequency_policy": "minute_hour_fixed_v1",
        "normalization": "external_context_zscore_population_v1_then_native_std",
        "inverse_transform": "native_then_external_v1",
        "horizon_policy": "native_prefix_v1",
    }


def validated_profile(config):
    try:
        identity = _canonical(config)
        for context in (512, 1024, 1536):
            if identity == _canonical(_configuration(context)):
                return {
                    "context": context,
                    "horizon": 96,
                    "point_forecast": point_metadata(context),
                }
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Invalid TTM configuration") from exc
    raise ValueError("Unsupported TTM R2 MSE configuration profile")


def validate_native(record):
    context = record["native_context_length"]
    if (
        type(record["native_prediction_length"]) is not int
        or record["native_prediction_length"] != 96
        or type(record["quantiles"]) is not list
        or record["quantiles"]
        or _canonical(record["point_forecast"]) != _canonical(point_metadata(context))
    ):
        raise ValueError("Unsupported TTM point checkpoint profile")


def validate_frequency(frequency):
    interval = fixed_interval(frequency)
    if not pd.Timedelta(minutes=1) <= interval <= pd.Timedelta(hours=1):
        raise ValueError(
            "TTM R2 requires a regular frequency from one minute to one hour"
        )


def forecast_options_for_record(record):
    validate_native(record)
    return {
        "output": "prediction_outputs",
        "horizon_policy": "native_prefix_v1",
        "external_scaling": "context_zscore_population_v1",
        "constant_scale": 1.0,
        "native_scaling": "std",
        "native_inverse": True,
    }


def validate_runtime_options(options, record):
    if _canonical(options) != _canonical(forecast_options_for_record(record)):
        raise ValueError(
            "TTM runtime requires its recorded forecast and scaling options"
        )
