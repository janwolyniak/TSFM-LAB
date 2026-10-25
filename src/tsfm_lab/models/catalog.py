"""Small explicit model catalog without optional inference or Hub imports."""

from dataclasses import asdict

from tsfm_lab import __version__
from tsfm_lab.contracts import ModelCapabilities
from tsfm_lab.models.baselines import LastValue, SeasonalNaive
from tsfm_lab.models.checkpoint_profiles import (
    ADAPTER_LIMITS,
    CHECKPOINT_REPOSITORIES,
)

CURATED_REPOSITORIES = tuple(
    repository
    for repositories in CHECKPOINT_REPOSITORIES.values()
    for repository in repositories
)


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
            }
        )
    for adapter, family, name in (
        ("chronos_bolt", "chronos", "Chronos-Bolt"),
        ("timesfm_2p5", "timesfm", "TimesFM 2.5"),
    ):
        context, horizon = ADAPTER_LIMITS[adapter]
        entries.append(
            {
                "adapter_id": adapter,
                "adapter_version": __version__,
                "family": family,
                "display_name": name,
                "source": "huggingface",
                "license": "apache-2.0",
                "dtypes": ["float32"],
                "point_statistic": "median",
                "capabilities": asdict(
                    ModelCapabilities(max_context=context, max_horizon=horizon)
                ),
                "repositories": list(CHECKPOINT_REPOSITORIES[adapter]),
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
            return {**entry, "repository": identifier}
    raise ValueError(
        f"Unsupported model {identifier!r}; use a catalog adapter or curated repository"
    )
