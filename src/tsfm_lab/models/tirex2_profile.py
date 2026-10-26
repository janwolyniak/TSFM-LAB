"""Reviewed native TiRex-2 configurations without optional runtime imports."""

import copy
import json

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _configuration(gifteval):
    dropout = 0.0 if gifteval else 0.1
    templates = {}
    for name, kind in (("slstm_attn", "bi-slstm"), ("mlstm_attn", "bi-mlstm")):
        mixer = {
            "model_type": kind,
            "embedding_dim": 512,
            "num_heads": 4,
            "rnn_backend": "flashrnn-cuda",
        }
        if kind == "bi-slstm":
            mixer["num_slstm_heads"] = 4
        templates[name] = {
            "time_mixer": mixer,
            "variate_mixer": {"embedding_dim": 512, "num_heads": 4},
            "dropout": dropout,
            "eps": 1e-7,
        }
    config = {
        "num_blocks": 12,
        "embedding_dim": 512,
        "input_patch_size": 32,
        "output_patch_size": 32,
        "input_ff_dim": 2048,
        "h_expand": 4,
        "context_len": 8192 if gifteval else 2048,
        "future_len": 928 if gifteval else 320,
        "dropout": dropout,
        "quantiles": QUANTILES.copy(),
        "act_func": "torch.nn.SiLU",
        "use_qk_norm": not gifteval,
        "tokenizer_cfg": {
            "input_patch_size": 32,
            "output_patch_size": 32,
            "input_patch_stride": 32,
        },
        "scaler_cfg": {"use_arcsinh": not gifteval, "binaryaware": not gifteval},
        "stack_out_norm_config": {"eps": 1e-7},
        "tta_diff": not gifteval,
        "stack_config": {
            "templates": templates,
            "recipe": ["mlstm_attn", "slstm_attn"] * 6,
        },
    }
    if gifteval:
        config["tta_sign_flip"] = True
    return config


def _canonical(config):
    # JSON comparison also distinguishes bool/int and integer/float substitutions.
    return json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _options(gifteval):
    return {"tta_sign_flip": gifteval, "tta_diff": not gifteval, "pad_context": True}


def validated_profile(config):
    """Reject any unreviewed activation, device, kernel, shape, or preprocessing."""
    try:
        identity = _canonical(config)
        for gifteval in (False, True):
            reviewed = _configuration(gifteval)
            if identity == _canonical(reviewed):
                return {
                    "name": "gifteval" if gifteval else "main",
                    "context": reviewed["context_len"],
                    "horizon": reviewed["future_len"],
                    "forecast_options": _options(gifteval),
                }
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Invalid TiRex-2 configuration") from exc
    raise ValueError("Unsupported TiRex-2 configuration profile")


def validate_native(context, horizon, quantiles):
    if (
        type(context) is not int
        or type(horizon) is not int
        or (context, horizon) not in {(2048, 320), (8192, 928)}
        or _canonical(quantiles) != _canonical(QUANTILES)
    ):
        raise ValueError("Unsupported TiRex-2 checkpoint native profile")


def forecast_options_for_record(record):
    """Recover the fixed profile flags without reading old checkpoint files."""
    validate_native(
        record["native_context_length"],
        record["native_prediction_length"],
        record["quantiles"],
    )
    return copy.deepcopy(_options(record["native_context_length"] == 8192))
