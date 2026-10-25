"""Lazy terminal launch boundary for scriptable and numerical callers."""

import sys
from pathlib import Path

from tsfm_lab.config import ConfigError


def interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def launch_tui(workspace: str | Path = Path("results")) -> None:
    try:
        from tsfm_lab.tui.app import LabApp
    except ModuleNotFoundError as exc:
        if exc.name not in {"textual", "mdit_py_plugins", "linkify_it"}:
            raise
        raise ConfigError(
            "The interactive terminal requires the optional tsfm-lab[tui] dependencies. "
            "Shell commands remain available."
        ) from exc
    LabApp(workspace=Path(workspace)).run()
