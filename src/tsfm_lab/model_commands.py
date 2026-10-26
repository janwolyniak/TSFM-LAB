"""Model catalog and explicit checkpoint-management commands."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from tsfm_lab.cli_support import emit, guard, table
from tsfm_lab.config import ConfigError
from tsfm_lab.models.catalog import catalog_entries, catalog_info, checkpoint_info
from tsfm_lab.models.checkpoints import cached_checkpoints, pull_checkpoint
from tsfm_lab.models.verification import checkpoint_verification, verification_label

models = typer.Typer(
    help="Inspect adapters and manage explicit model downloads.", no_args_is_help=True
)
JsonOutput = Annotated[bool, typer.Option("--json")]
CacheDirectory = Annotated[Path | None, typer.Option("--cache-dir")]


def _selection(identifier, variant=None):
    try:
        return catalog_info(identifier, variant)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _variant_information(variant):
    rows = [
        ["Repository", variant["repository"]],
        ["Variant", variant["variant"]],
        ["Reference revision", variant["reference_revision"]],
    ]
    for key, label in (
        ("parameter_count", "Parameters"),
        ("weight_bytes", "Weight bytes"),
        ("min_context", "Minimum context"),
        ("max_context", "Maximum context"),
        ("max_horizon", "Maximum horizon"),
    ):
        if variant[key] is not None:
            rows.append([label, f"{variant[key]:,}"])
    table(variant["display_name"], ["Field", "Value"], rows)
    console = Console(markup=False, highlight=False)
    console.print("Reference checkpoint:")
    console.print(variant["verification_label"])
    checks = variant["representative_checks"]
    if checks:
        table(
            "Adapter representative checks",
            ["Checkpoint", "Revision", "Device / dtype"],
            [
                [
                    item["repository"],
                    item["revision"][:12],
                    f"{item['device']} / {item['dtype']}",
                ]
                for item in checks
            ],
        )


def _selected_verification(record):
    snapshot = checkpoint_verification(record)
    Console(markup=False, highlight=False).print(
        f"Selected checkpoint: {verification_label(snapshot)}"
    )


def _runtime_notices(value):
    if value.get("runtime_notices"):
        console = Console(markup=False, highlight=False)
        console.print(f"{value['display_name']} optional runtime notices:")
        for notice in value["runtime_notices"]:
            console.print(notice)


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
            ["Adapter", "Family", "Device", "Variants"],
            [
                [
                    item["adapter_id"],
                    item["family"],
                    ", ".join(item["capabilities"]["devices"]),
                    ", ".join(
                        variant["variant"] for variant in item.get("variants", [])
                    )
                    or "built-in",
                ]
                for item in items
            ],
        )
        for item in items:
            _runtime_notices(item)


@models.command("info")
@guard
def model_info(
    identifier: str,
    json_output: JsonOutput = False,
    variant: str | None = typer.Option(None, "--variant"),
):
    """Show curated adapter/checkpoint metadata without network access."""
    value = _selection(identifier, variant)
    selected = checkpoint_info(identifier, variant) if "repository" in value else None
    if json_output:
        emit({**value, "selected_checkpoint": selected} if selected else value)
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
                    "supported_tasks",
                    "frequency_policy",
                    "repositories",
                )
                if key in value
            ],
        )
        _runtime_notices(value)
        for variant in [selected] if selected else value.get("variants", []):
            _variant_information(variant)


@models.command("pull")
@guard
def pull_model(
    repository: str,
    revision: str | None = typer.Option(None, "--revision"),
    variant: str | None = typer.Option(None, "--variant"),
    cache_dir: CacheDirectory = None,
    json_output: JsonOutput = False,
):
    """Explicitly download a curated checkpoint at a resolved exact revision."""
    selected = _selection(repository)
    if repository not in selected["repositories"]:
        raise ConfigError("Select a curated Hugging Face repository, not an adapter ID")
    if revision is not None and variant is not None:
        raise ConfigError("Choose --revision or --variant, not both")
    if variant is not None:
        revision = checkpoint_info(repository, variant)["reference_revision"]
    revision = revision if revision is not None else "main"
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
        _selected_verification(value)


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
            ["Repository", "Revision", "Cache", "Project inference"],
            [
                [
                    str(value.get("repository", "")),
                    str(value.get("resolved_revision", ""))[:12],
                    str(value.get("status", "cached")),
                    (
                        verification_label(
                            checkpoint_verification(
                                {
                                    key: item
                                    for key, item in value.items()
                                    if key not in {"status", "bytes"}
                                }
                            )
                        )
                        if value.get("status", "cached") == "cached"
                        else "Unavailable for incomplete files"
                    ),
                ]
                for value in values
            ],
        )
