"""Shared point-task eligibility and metadata-only frequency validation."""


def output_statistics(task: str) -> frozenset[str]:
    """Keep median claims strict and model-native point accuracy explicit."""
    if task == "median":
        return frozenset({"deterministic", "median"})
    if task == "native_point":
        return frozenset({"deterministic", "median", "mean", "native_point"})
    raise ValueError("unsupported point task; choose median or native_point")


def validate_model_frequency(model, frequency: str) -> None:
    """Pass only declared cadence to an optional validation hook before loading."""
    if not isinstance(frequency, str) or not frequency.strip():
        raise ValueError("Model frequency validation requires a nonempty string")
    missing = object()
    hook = getattr(model, "validate_frequency", missing)
    if hook is missing:
        return
    if not callable(hook):
        raise ValueError("validate_frequency must be a callable hook")
    if hook(frequency) is not None:
        raise ValueError("validate_frequency must return None or raise on refusal")
