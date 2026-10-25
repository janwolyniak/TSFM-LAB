"""Scriptable terminal commands backed by the shared application functions."""

import json
import sys
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from tsfm_lab import __version__
from tsfm_lab.application import load_dataset, run_from_config
from tsfm_lab.comparison import ComparisonError, compare_runs
from tsfm_lab.config import ConfigError
from tsfm_lab.experiments import ExperimentStore, StoreError
from tsfm_lab.exports import ExportError, export_predictions
from tsfm_lab.models.baselines import LastValue, SeasonalNaive

app = typer.Typer(
    help="Evaluate time-series forecasts locally.", invoke_without_command=True
)
models = typer.Typer(help="Inspect available model adapters.", no_args_is_help=True)
datasets = typer.Typer(help="Inspect a local dataset.", no_args_is_help=True)
experiments = typer.Typer(help="Inspect saved experiments.", no_args_is_help=True)
app.add_typer(models, name="models")
app.add_typer(datasets, name="datasets")
app.add_typer(experiments, name="experiments")

Workspace = Annotated[Path, typer.Option("--workspace", "-w")]
JsonOutput = Annotated[bool, typer.Option("--json")]


def _guard(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ConfigError, ComparisonError, ExportError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise typer.Exit(2) from exc
        except (StoreError, OSError, ImportError, RuntimeError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise typer.Exit(1) from exc
        except KeyboardInterrupt as exc:
            print("Experiment cancelled.", file=sys.stderr)
            raise typer.Exit(130) from exc

    return wrapped


def _emit(value: dict | list) -> None:
    print(json.dumps(value, indent=2, allow_nan=False))


def _table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    table = Table(title=title)
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*row)
    Console(markup=False, highlight=False).print(table)


def _metric(value: dict) -> str:
    return (
        f"{value['value']:.6g}"
        if value["value"] is not None
        else f"undefined ({value['reason']})"
    )


@app.callback()
def home(
    ctx: typer.Context, version: bool = typer.Option(False, "--version", is_eager=True)
):
    if version:
        typer.echo(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("run")
@_guard
def run(config: Path, json_output: bool = typer.Option(False, "--json")):
    """Execute a validated YAML configuration and save its artifacts."""
    console = Console(stderr=True, markup=False, highlight=False)
    result = run_from_config(
        config, progress=lambda stage: console.print(f"Stage: {stage}")
    )
    if json_output:
        _emit(result)
    else:
        console.print(f"Completed experiment {result['run_id']}")
        _table(
            result["name"],
            ["Metric", "Value"],
            [[key.upper(), _metric(value)] for key, value in result["metrics"].items()],
        )
        typer.echo(f"Saved to {result['directory']}")


@models.command("list")
@_guard
def list_models(json_output: bool = typer.Option(False, "--json")):
    """List the implemented built-in adapters; no downloads are performed."""
    items = [asdict(model.metadata()) for model in (LastValue(), SeasonalNaive(1))]
    if json_output:
        _emit(items)
    else:
        _table(
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


@datasets.command("inspect")
@_guard
def inspect_dataset(
    path: Path,
    timestamp_column: str = typer.Option(..., "--timestamp-column"),
    target_column: str = typer.Option(..., "--target-column"),
    frequency: str = typer.Option(..., "--frequency"),
    timezone: str | None = typer.Option(None, "--timezone"),
    units: str = typer.Option("unspecified", "--units"),
    series_id: str = typer.Option("series_0", "--series-id"),
    json_output: bool = typer.Option(False, "--json"),
):
    """Validate a CSV or Parquet series and report its identity."""
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".parquet", ".pq"}:
        raise ConfigError("Dataset format must be CSV or Parquet")
    try:
        dataset = load_dataset(
            dict(
                path=path,
                format="csv" if suffix == ".csv" else "parquet",
                timestamp_column=timestamp_column,
                target_column=target_column,
                frequency=frequency,
                timezone=timezone,
                units=units,
                series_id=series_id,
            )
        )
    except (ValueError, OSError) as exc:
        raise ConfigError(str(exc)) from exc
    value = {
        "series_id": dataset.series_id,
        "target": dataset.target_name,
        "units": dataset.units,
        "frequency": dataset.frequency,
        "observations": len(dataset.values),
        "start": dataset.timestamps[0].isoformat(),
        "end": dataset.timestamps[-1].isoformat(),
        "provenance": dataset.provenance,
    }
    if json_output:
        _emit(value)
    else:
        _table(
            "Dataset",
            ["Field", "Value"],
            [[key, str(item)] for key, item in value.items() if key != "provenance"],
        )


@experiments.command("list")
@_guard
def list_experiments(
    workspace: Workspace = Path("results"),
    json_output: JsonOutput = False,
):
    """List completed, failed, cancelled, and incomplete records."""
    items = ExperimentStore(workspace).list_runs()
    if json_output:
        _emit(items)
    else:
        _table(
            "Experiment history",
            ["ID", "Name", "Status"],
            [[item["run_id"], item.get("name", ""), item["status"]] for item in items],
        )


@experiments.command("show")
@_guard
def show_experiment(
    run_id: str,
    workspace: Workspace = Path("results"),
    json_output: JsonOutput = False,
):
    """Inspect a record, verifying completed artifacts before displaying scores."""
    store = ExperimentStore(workspace)
    manifest = store.get_manifest(run_id)
    value = {"manifest": manifest}
    if manifest["status"] == "completed":
        value["result"], _ = store.load_result(run_id)
    if json_output:
        _emit(value)
    else:
        _table(
            "Experiment",
            ["Field", "Value"],
            [
                [key, str(manifest.get(key, ""))]
                for key in ("run_id", "name", "status", "created_at")
            ],
        )
        if "result" in value:
            _table(
                "Metrics",
                ["Metric", "Value"],
                [
                    [key.upper(), _metric(metric)]
                    for key, metric in value["result"]["metrics"]["overall"].items()
                ],
            )
        elif manifest.get("error"):
            typer.echo(str(manifest["error"]))


@app.command("compare")
@_guard
def compare(
    run_ids: Annotated[list[str], typer.Argument()],
    workspace: Workspace = Path("results"),
    json_output: JsonOutput = False,
):
    """Compare completed experiments only when forecasting tasks match."""
    value = compare_runs(run_ids, workspace)
    if json_output:
        _emit(value)
    else:
        metric_names = list(value["runs"][0]["metrics"])
        _table(
            "Controlled comparison",
            ["Experiment", "Model", *[name.upper() for name in metric_names]],
            [
                [
                    run["run_id"],
                    run["model"]["adapter_id"],
                    *[_metric(run["metrics"][name]) for name in metric_names],
                ]
                for run in value["runs"]
            ],
        )


@app.command("export")
@_guard
def export(
    run_id: str,
    output: Annotated[Path, typer.Option("--output", "-o")],
    format: str | None = typer.Option(None, "--format"),
    workspace: Workspace = Path("results"),
    json_output: JsonOutput = False,
):
    """Export verified predictions to a new CSV, Parquet, or JSON file."""
    value = export_predictions(run_id, workspace, output, format=format)
    if json_output:
        _emit(value)
    else:
        typer.echo(f"Exported {value['rows']} predictions to {value['path']}")
