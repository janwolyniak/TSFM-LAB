"""Lazy terminal launch boundary for scriptable and numerical callers."""

import sys
from pathlib import Path

from tsfm_lab.config import ConfigError


def interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def launch_tui(workspace: str | Path | None = None) -> None:
    try:
        from tsfm_lab.repl.session import TerminalSession
    except ModuleNotFoundError as exc:
        if exc.name != "prompt_toolkit":
            raise
        raise ConfigError(
            "The interactive terminal requires the optional tsfm-lab[cli] dependencies. "
            "Shell commands remain available."
        ) from exc
    from tsfm_lab.preferences import PreferenceStore

    TerminalSession(workspace=workspace, preference_store=PreferenceStore()).run()
