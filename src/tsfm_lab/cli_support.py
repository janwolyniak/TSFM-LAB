"""Shared terminal formatting and exit-code handling."""

import json
import sys
from functools import wraps

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from tsfm_lab.comparison import ComparisonError
from tsfm_lab.config import ConfigError
from tsfm_lab.experiments import StoreError
from tsfm_lab.exports import ExportError


def guard(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except typer.Exit:
            raise
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


def emit(value: dict | list) -> None:
    print(json.dumps(value, indent=2, allow_nan=False))


def table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    view = Table(title=Text(title))
    for column in columns:
        view.add_column(Text(column))
    for row in rows:
        view.add_row(*(Text(value) for value in row))
    Console(markup=False, highlight=False).print(view)
