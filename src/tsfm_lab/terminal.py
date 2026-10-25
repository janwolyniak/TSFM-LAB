"""Lazy terminal launch boundary for scriptable and numerical callers."""

import sys
from pathlib import Path

from tsfm_lab.config import ConfigError


def interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def launch_tui(workspace: str | Path = Path("results")) -> None:
    try:
        from tsfm_lab.repl.session import TerminalSession
    except ModuleNotFoundError as exc:
        if exc.name != "prompt_toolkit":
            raise
        raise ConfigError(
            "The interactive terminal requires the optional tsfm-lab[cli] dependencies. "
            "Shell commands remain available."
        ) from exc
    TerminalSession(workspace=Path(workspace)).run()
