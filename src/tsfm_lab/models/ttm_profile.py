"""Reviewed TTM profiles and point-only checkpoint semantics, without runtimes."""

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


_PROFILES = (
    (512, 96),
    (1024, 96),
    (1536, 96),
    (512, 192),
    (512, 336),
    (512, 720),
    (52, 16),
    (90, 30),
)


def _configuration(context, horizon=96):
    config = copy.deepcopy(_BASE)
    if (context, horizon) not in _PROFILES:
        raise ValueError("Unsupported TTM context/horizon profile")
    config["context_length"] = context
    if context in (1024, 1536):
        config.update(
            d_model=384,
            decoder_d_model=256,
            frequency_token_vocab_size=8,
            patch_length=128,
            patch_stride=128,
        )
    if context == 1536:
        config["num_patches"] = 12
    if horizon != 96:
        config.update(
            prediction_length=horizon,
            fcm_prepend_past_offset=None,
            mask_value=0,
            masked_context_length=None,
            transformers_version="4.38.0",
        )
    if context in (52, 90):
        config.update(
            adaptive_patching_levels=0,
            decoder_num_layers=4,
            dropout=0.3,
            frequency_token_vocab_size=10,
            head_dropout=0.3,
            resolution_prefix_tuning=True,
        )
        if context == 52:
            config.update(
                d_model=20,
                d_model_scale=5,
                decoder_d_model=16,
                decoder_d_model_scale=4,
                expansion_factor=3,
                num_layers=12,
                num_patches=14,
                patch_length=4,
                patch_stride=4,
            )
        else:
            config.update(
                d_model=90,
                d_model_scale=10,
                decoder_d_model=18,
                num_layers=20,
                num_patches=11,
                patch_length=9,
                patch_stride=9,
            )
    return config


def point_metadata(context, horizon=96):
    if (
        type(context) is not int
        or type(horizon) is not int
        or (context, horizon) not in _PROFILES
    ):
        raise ValueError("Unsupported TTM context/horizon profile")
    prefix = context in (52, 90)
    release = "r2.1" if prefix else "r2"
    result = {
        "statistic": "native_point",
        "training_loss": "mse",
        "release": release,
        "profile": f"{context}-{horizon}{'-ft' if prefix else ''}-{release}",
        "frequency_policy": (
            ("weekly_fixed_v1" if context == 52 else "daily_fixed_v1")
            if prefix
            else "minute_hour_fixed_v1"
        ),
        "normalization": "external_context_zscore_population_v1_then_native_std",
        "inverse_transform": "native_then_external_v1",
        "horizon_policy": "native_prefix_v1",
    }
    if prefix:
        result["frequency_token"] = 9 if context == 52 else 8
    return result


def validated_profile(config):
    try:
        identity = _canonical(config)
        for context, horizon in _PROFILES:
            if identity == _canonical(_configuration(context, horizon)):
                return {
                    "context": context,
                    "horizon": horizon,
                    "point_forecast": point_metadata(context, horizon),
                }
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Invalid TTM configuration") from exc
    raise ValueError("Unsupported TTM MSE configuration profile")


def validate_native(record):
    context = record["native_context_length"]
    horizon = record["native_prediction_length"]
    if (
        type(record["quantiles"]) is not list
        or record["quantiles"]
        or _canonical(record["point_forecast"])
        != _canonical(point_metadata(context, horizon))
    ):
        raise ValueError("Unsupported TTM point checkpoint profile")


def profile_frequency(profile):
    if profile == "52-16-ft-r2.1":
        return "weekly_fixed_v1"
    if profile == "90-30-ft-r2.1":
        return "daily_fixed_v1"
    return "minute_hour_fixed_v1"


def validate_frequency(frequency, policy="minute_hour_fixed_v1"):
    interval = fixed_interval(frequency)
    if policy in ("daily_fixed_v1", "weekly_fixed_v1"):
        days = 1 if policy == "daily_fixed_v1" else 7
        if interval != pd.Timedelta(days=days):
            raise ValueError(f"TTM {policy} requires exactly {days} fixed UTC day(s)")
        return
    if policy != "minute_hour_fixed_v1":
        raise ValueError("Unsupported TTM frequency policy")
    if not pd.Timedelta(minutes=1) <= interval <= pd.Timedelta(hours=1):
        raise ValueError(
            "TTM R2 requires a regular frequency from one minute to one hour"
        )


def forecast_options_for_record(record):
    validate_native(record)
    options = {
        "output": "prediction_outputs",
        "horizon_policy": "native_prefix_v1",
        "external_scaling": "context_zscore_population_v1",
        "constant_scale": 1.0,
        "native_scaling": "std",
        "native_inverse": True,
    }
    if "frequency_token" in record["point_forecast"]:
        options["frequency_token"] = record["point_forecast"]["frequency_token"]
    return options


def validate_runtime_options(options, record):
    if _canonical(options) != _canonical(forecast_options_for_record(record)):
        raise ValueError(
            "TTM runtime requires its recorded forecast and scaling options"
        )
