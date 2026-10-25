"""Resolve explicitly selected models without implicit checkpoint downloads."""

from tsfm_lab.models.baselines import LastValue, SeasonalNaive


def build_model(selection: dict, runtime: dict):
    if selection["source"] == "builtin":
        return (
            LastValue()
            if selection["adapter_id"] == "last_value"
            else SeasonalNaive(**selection["options"])
        )
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
    if selection["adapter_id"] == "chronos_bolt":
        from tsfm_lab.models.chronos_bolt import ChronosBolt

        adapter = ChronosBolt
    elif selection["adapter_id"] == "timesfm_2p5":
        from tsfm_lab.models.timesfm import TimesFM

        adapter = TimesFM
    else:
        raise ValueError("Unsupported checkpoint adapter")
    return adapter(
        checkpoint,
        device=runtime["device"],
        dtype=runtime["dtype"],
        seed=runtime["seed"],
    )
