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


def checkpoint_info(repository: str) -> dict:
    """Describe a curated reference without inspecting local files or runtimes."""
    from tsfm_lab.models._verification_evidence import PROJECT_CHECKS

    profile, variant = checkpoint_variant(repository)
    checks = [
        item for item in PROJECT_CHECKS if item["adapter_id"] == profile.adapter_id
    ]
    recorded = any(
        item["repository"] == repository
        and item["revision"] == variant.reference_revision
        for item in checks
    )
    return {
        **asdict(variant),
        "project_inference_status": "recorded" if recorded else "not_recorded",
        "verification_label": (
            "Reference checkpoint inference check recorded"
            if recorded
            else "Metadata/API compatibility checked; individual inference not run"
        ),
        "representative_checks": copy.deepcopy(checks),
    }


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
                "repositories": [],
                "variants": [],
                "runtime_requirements": {},
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
                "point_statistic": "median",
                "capabilities": asdict(
                    ModelCapabilities(max_context=context, max_horizon=horizon)
                ),
                "repositories": list(CHECKPOINT_REPOSITORIES[adapter]),
                "variants": [
                    checkpoint_info(repository)
                    for repository in CHECKPOINT_REPOSITORIES[adapter]
                ],
                "runtime_requirements": dict(profile.runtime_requirements),
            }
        )
    # Tuples in typed metadata become ordinary JSON arrays at this boundary.
    for entry in entries:
        for key in ("output_types", "devices"):
            entry["capabilities"][key] = list(entry["capabilities"][key])
    return entries


def catalog_info(identifier: str) -> dict:
    """Look up an adapter or curated repository without probing arbitrary models."""
    for entry in catalog_entries():
        if identifier == entry["adapter_id"]:
            return entry
        if identifier in entry["repositories"]:
            variant = checkpoint_info(identifier)
            return {
                **entry,
                **{
                    key: value
                    for key, value in variant.items()
                    if key != "display_name"
                },
                "variant_display_name": variant["display_name"],
            }
    raise ValueError(
        f"Unsupported model {identifier!r}; use a catalog adapter or curated repository"
    )
