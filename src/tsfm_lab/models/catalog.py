"""Small explicit model catalog without optional inference or Hub imports."""

from dataclasses import asdict

from tsfm_lab import __version__
from tsfm_lab.contracts import ModelCapabilities
from tsfm_lab.models.baselines import LastValue, SeasonalNaive

CURATED_REPOSITORIES = (
    "amazon/chronos-bolt-tiny",
    "amazon/chronos-bolt-small",
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
    entries.append(
        {
            "adapter_id": "chronos_bolt",
            "adapter_version": __version__,
            "family": "chronos",
            "display_name": "Chronos-Bolt",
            "source": "huggingface",
            "license": "apache-2.0",
            "dtypes": ["float32"],
            "point_statistic": "median",
            "capabilities": asdict(ModelCapabilities(max_context=2048, max_horizon=64)),
            "repositories": list(CURATED_REPOSITORIES),
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
