"""Model catalog and explicit checkpoint-management commands."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from tsfm_lab.cli_support import emit, guard, table
from tsfm_lab.config import ConfigError
from tsfm_lab.models.catalog import catalog_entries, catalog_info
from tsfm_lab.models.checkpoints import cached_checkpoints, pull_checkpoint

models = typer.Typer(
    help="Inspect adapters and manage explicit model downloads.", no_args_is_help=True
)
JsonOutput = Annotated[bool, typer.Option("--json")]
CacheDirectory = Annotated[Path | None, typer.Option("--cache-dir")]


def _selection(identifier):
    try:
        return catalog_info(identifier)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


@models.command("list")
@guard
def list_models(json_output: JsonOutput = False):
    """List supported adapters without importing model runtimes or downloading."""
    items = catalog_entries()
    if json_output:
        emit(items)
    else:
        table(
            "Available models",
            ["Adapter", "Family", "Device"],
            [
                [
                    item["adapter_id"],
                    item["family"],
                    ", ".join(item["capabilities"]["devices"]),
                ]
                for item in items
            ],
        )


@models.command("info")
@guard
def model_info(identifier: str, json_output: JsonOutput = False):
    """Show curated adapter/checkpoint metadata without network access."""
    value = _selection(identifier)
    if json_output:
        emit(value)
    else:
        table(
            "Model information",
            ["Field", "Value"],
            [
                [key, str(value[key])]
                for key in (
                    "adapter_id",
                    "family",
                    "license",
                    "point_statistic",
                    "repositories",
                )
                if key in value
            ],
        )


@models.command("pull")
@guard
def pull_model(
    repository: str,
    revision: str = typer.Option("main", "--revision"),
    cache_dir: CacheDirectory = None,
    json_output: JsonOutput = False,
):
    """Explicitly download a curated Bolt checkpoint at a resolved exact revision."""
    selected = _selection(repository)
    if repository not in selected["repositories"]:
        raise ConfigError("Select a curated Hugging Face repository, not an adapter ID")
    console = Console(stderr=True, markup=False, highlight=False)
    console.print(f"Resolving and pulling {repository} at {revision}")
    checkpoint = pull_checkpoint(repository, revision, cache_dir=cache_dir)
    value = checkpoint.record()
    if json_output:
        emit(value)
    else:
        table(
            "Cached checkpoint",
            ["Field", "Value"],
            [
                [key, str(value.get(key))]
                for key in (
                    "repository",
                    "requested_revision",
                    "resolved_revision",
                    "path",
                    "license",
                    "fingerprint",
                )
            ],
        )


@models.command("cache")
@guard
def list_cached_models(
    cache_dir: CacheDirectory = None, json_output: JsonOutput = False
):
    """Inspect curated cached snapshots without network access or deletion."""
    values = cached_checkpoints(cache_dir=cache_dir)
    if json_output:
        emit(values)
    else:
        table(
            "Cached checkpoints",
            ["Repository", "Revision", "Status"],
            [
                [
                    str(value.get("repository", "")),
                    str(value.get("resolved_revision", "")),
                    str(value.get("status", "cached")),
                ]
                for value in values
            ],
        )
