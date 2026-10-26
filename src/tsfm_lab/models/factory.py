"""Resolve explicitly selected models without implicit checkpoint downloads."""

import importlib

from tsfm_lab.models.baselines import LastValue, SeasonalNaive
from tsfm_lab.models.registry import adapter_profile


def build_model(selection: dict, runtime: dict):
    if selection["source"] == "builtin":
        return (
            LastValue()
            if selection["adapter_id"] == "last_value"
            else SeasonalNaive(**selection["options"])
        )
    profile = adapter_profile(selection["adapter_id"])
    from tsfm_lab.models.checkpoints import inspect_local_checkpoint, resolve_checkpoint

    if selection["source"] == "local":
        checkpoint = inspect_local_checkpoint(selection["path"])
    else:
        checkpoint = resolve_checkpoint(
            selection["repository"],
            selection["revision"],
            cache_dir=selection.get("cache_dir"),
            offline=selection["offline"],
        )
    adapter = getattr(importlib.import_module(profile.module), profile.class_name)
    return adapter(
        checkpoint,
        device=runtime["device"],
        dtype=runtime["dtype"],
        seed=runtime["seed"],
    )
