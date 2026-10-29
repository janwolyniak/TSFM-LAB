"""Small explicit model catalog without optional inference or Hub imports."""

import copy
from dataclasses import asdict

from tsfm_lab import __version__
from tsfm_lab.contracts import ModelCapabilities
from tsfm_lab.models.baselines import LastValue, SeasonalNaive
from tsfm_lab.models.checkpoint_profiles import (
    ADAPTER_LIMITS,
    CHECKPOINT_REPOSITORIES,
)
from tsfm_lab.models.registry import ADAPTER_PROFILES, checkpoint_variant

CURATED_REPOSITORIES = tuple(
    repository
    for repositories in CHECKPOINT_REPOSITORIES.values()
    for repository in repositories
)


def checkpoint_info(repository: str, variant: str | None = None) -> dict:
    """Describe a curated reference without inspecting local files or runtimes."""
    from tsfm_lab.models._verification_evidence import PROJECT_CHECKS

    profile, variant = checkpoint_variant(repository, variant)
    checks = [
        item for item in PROJECT_CHECKS if item["adapter_id"] == profile.adapter_id
    ]
    recorded = any(
        item["repository"] == repository
        and item["revision"] == variant.reference_revision
        for item in checks
    )
    information = {
        **asdict(variant),
        "max_context": variant.max_context or profile.max_context,
        "max_horizon": variant.max_horizon or profile.max_horizon,
        "min_context": variant.min_context or profile.min_context,
        "project_inference_status": "recorded" if recorded else "not_recorded",
        "verification_label": (
            "Reference checkpoint inference check recorded"
            if recorded
            else "Metadata/API compatibility checked; individual inference not run"
        ),
        "representative_checks": copy.deepcopy(checks),
    }
    if profile.adapter_id == "ttm_r2":
        from tsfm_lab.models.ttm_profile import profile_frequency

        information["frequency_policy"] = profile_frequency(variant.variant)
    return information


def catalog_entries() -> list[dict]:
    """Return fresh JSON-compatible metadata for implemented adapters."""
    entries = []
    for model, name in (
        (LastValue(), "Last value"),
        (SeasonalNaive(1), "Seasonal naive"),
    ):
        entries.append(
            {
                **asdict(model.metadata()),
                "display_name": name,
                "source": "builtin",
                "license": "Apache-2.0",
                "dtypes": ["float64"],
                "point_statistic": "deterministic",
                "supported_tasks": ["median", "native_point"],
                "frequency_policy": "any_fixed",
                "repositories": [],
                "variants": [],
                "runtime_requirements": {},
                "runtime_notices": [],
            }
        )
    for profile in ADAPTER_PROFILES:
        adapter = profile.adapter_id
        context, horizon = ADAPTER_LIMITS[adapter]
        entries.append(
            {
                "adapter_id": adapter,
                "adapter_version": __version__,
                "family": profile.family,
                "display_name": profile.display_name,
                "source": "huggingface",
                "license": "apache-2.0",
                "dtypes": ["float32"],
                "point_statistic": profile.point_statistic,
                "supported_tasks": list(profile.supported_tasks),
                "frequency_policy": profile.frequency_policy,
                "capabilities": asdict(
                    ModelCapabilities(
                        min_context=profile.min_context,
                        max_context=context,
                        max_horizon=horizon,
                    )
                ),
                "repositories": list(CHECKPOINT_REPOSITORIES[adapter]),
                "variants": [
                    checkpoint_info(variant.repository, variant.variant)
                    for variant in profile.variants
                ],
                "runtime_requirements": dict(profile.runtime_requirements),
                "runtime_notices": list(profile.runtime_notices),
            }
        )
    # Tuples in typed metadata become ordinary JSON arrays at this boundary.
    for entry in entries:
        for key in ("output_types", "devices"):
            entry["capabilities"][key] = list(entry["capabilities"][key])
    return entries


def catalog_info(identifier: str, variant: str | None = None) -> dict:
    """Look up an adapter or curated repository without probing arbitrary models."""
    for entry in catalog_entries():
        if identifier == entry["adapter_id"]:
            if variant is not None:
                raise ValueError("Select a repository before its checkpoint profile")
            return entry
        if identifier in entry["repositories"]:
            variant = checkpoint_info(identifier, variant)
            return {
                **entry,
                **{
                    key: value
                    for key, value in variant.items()
                    if key != "display_name"
                },
                "variant_display_name": variant["display_name"],
                "capabilities": {
                    **entry["capabilities"],
                    "min_context": variant["min_context"],
                    "max_context": variant["max_context"],
                    "max_horizon": variant["max_horizon"],
                },
            }
    raise ValueError(
        f"Unsupported model {identifier!r}; use a catalog adapter or curated repository"
    )
