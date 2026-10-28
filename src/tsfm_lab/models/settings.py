"""Metadata-only model settings shared by guided, assisted, and scriptable setup."""

from dataclasses import dataclass

from tsfm_lab.contracts import (
    EvaluationSpec,
    ModelCapabilities,
    fixed_interval,
    positive_int,
)
from tsfm_lab.models.registry import adapter_profile

_LABELS = {
    "chronos_bolt": ("Chronos", "Bolt", "size"),
    "chronos_2": ("Chronos", "2", "size"),
    "timesfm_2p5": ("TimesFM", "2.5", "size"),
    "tirex_2": ("TiRex", "2", "training_profile"),
    "ttm_r2": ("IBM Granite TTM", "R2", "context_horizon"),
}
_PREPROCESSING = {
    "chronos_bolt": "Required native context normalization and inverse; no external transform option.",
    "chronos_2": "Required native context normalization and inverse; no external transform option.",
    "timesfm_2p5": "Required native input normalization and inverse; fixed forecast options.",
    "tirex_2": "Fixed native profile preprocessing and inverse to original units; no external transform option.",
    "ttm_r2": "Required context-only population z-score, native std scaling, and inverse to original units.",
}


@dataclass(frozen=True)
class ModelSettings:
    adapter_id: str
    family_label: str
    generation_label: str
    variant_label: str
    variant_kind: str
    capabilities: ModelCapabilities
    default_context: int
    default_horizon: int
    supported_tasks: tuple[str, ...]
    preprocessing: str
    frequency_policy: str


def _selected_variant(profile, repository, variant, revision):
    candidates = profile.variants
    if repository is not None:
        candidates = tuple(item for item in candidates if item.repository == repository)
        if not candidates:
            raise ValueError(
                "Choose a curated repository belonging to the requested adapter"
            )
    selected = None
    if variant is not None:
        if repository is None:
            raise ValueError("Select a repository before its checkpoint profile")
        selected = next((item for item in candidates if item.variant == variant), None)
        if selected is None:
            raise ValueError("Select a curated checkpoint profile for this repository")
    if revision is not None:
        exact = next(
            (item for item in candidates if item.reference_revision == revision), None
        )
        if exact is not None and selected is not None and exact != selected:
            raise ValueError("Checkpoint revision disagrees with the selected profile")
        return exact
    if selected is not None:
        return selected
    return candidates[0] if repository is not None and len(candidates) == 1 else None


def model_settings(
    adapter_id: str, *, repository=None, variant=None, revision=None
) -> ModelSettings:
    """Describe curated bounds without reading files, importing runtimes, or networking.

    Unknown revisions and local sources retain adapter-wide bounds. Their exact
    native requirements must still pass checkpoint validation during preparation.
    """
    if adapter_id in {"last_value", "seasonal_naive"}:
        if any(item is not None for item in (repository, variant, revision)):
            raise ValueError("Built-in baselines do not accept checkpoint selections")
        return ModelSettings(
            adapter_id,
            "Baselines",
            "Deterministic",
            "Last value" if adapter_id == "last_value" else "Seasonal naive",
            "baseline",
            ModelCapabilities(),
            8,
            4,
            ("median", "native_point"),
            "None; original input units are preserved.",
            "any_fixed",
        )
    profile = adapter_profile(adapter_id)
    selected = _selected_variant(profile, repository, variant, revision)
    low = (selected.min_context if selected else None) or profile.min_context
    high = (selected.max_context if selected else None) or profile.max_context
    horizon = (selected.max_horizon if selected else None) or profile.max_horizon
    family, generation, kind = _LABELS[adapter_id]
    return ModelSettings(
        adapter_id,
        family,
        generation,
        selected.display_name if selected else "Unresolved checkpoint",
        kind,
        ModelCapabilities(min_context=low, max_context=high, max_horizon=horizon),
        max(low, min(8, high)),
        min(4, horizon),
        profile.supported_tasks,
        _PREPROCESSING[adapter_id],
        profile.frequency_policy,
    )


def validate_settings(
    model: dict, evaluation: EvaluationSpec, frequency: str | None = None
) -> None:
    """Validate settings metadata; checkpoint/runtime/data checks remain mandatory."""
    settings = model_settings(
        model["adapter_id"],
        repository=model.get("repository"),
        revision=model.get("revision"),
    )
    if evaluation.point_statistic not in settings.supported_tasks:
        raise ValueError("The requested forecast task is unsupported by this model")
    caps = settings.capabilities
    for key, low, high in (
        ("context_length", caps.min_context, caps.max_context),
        ("horizon", 1, caps.max_horizon),
    ):
        value = getattr(evaluation, key)
        if value < low or (high is not None and value > high):
            exact = (
                f"requires exactly {low} history observations; "
                if key == "context_length" and low == high
                else ""
            )
            raise ValueError(
                f"{key} is outside the selected model's supported limits; {exact}"
                f"context_length <= {caps.max_context} and horizon <= {caps.max_horizon}"
            )
    if model["adapter_id"] == "seasonal_naive":
        period = positive_int(
            model.get("options", {}).get("seasonal_period"), "seasonal_period"
        )
        if evaluation.context_length < period:
            raise ValueError("context_length must contain a complete baseline season")
    if frequency is not None:
        fixed_interval(frequency)
        if settings.frequency_policy == "minutely_hourly":
            from tsfm_lab.models.ttm_profile import validate_frequency

            validate_frequency(frequency)
