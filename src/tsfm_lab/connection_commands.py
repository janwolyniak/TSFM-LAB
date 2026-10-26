"""Optional connection management without credential command arguments."""

from pathlib import Path
from typing import Annotated

import typer

from tsfm_lab.cli_support import emit, guard
from tsfm_lab.config import ConfigError
from tsfm_lab.connections.manager import ConnectionManager
from tsfm_lab.connections.state import StateStore
from tsfm_lab.repl.text import literal
from tsfm_lab.terminal import interactive_terminal

connections = typer.Typer(
    help="Manage optional assistant connections.", no_args_is_help=True
)
StateDirectory = Annotated[Path | None, typer.Option("--state-directory")]


@connections.command("list")
@guard
def list_connections(
    state_directory: StateDirectory = None,
    json_output: bool = typer.Option(False, "--json"),
):
    """Read nonsecret metadata only; no vault access or network requests."""
    items = ConnectionManager(StateStore(state_directory)).list()
    if json_output:
        emit(items)
    else:
        for item in items:
            typer.echo(
                literal(
                    f"{item['label']} · {item['provider']} · {item['usage']} · model {item['model'] or 'not selected'} · {item['status']}"
                )
            )
        if not items:
            typer.echo(
                "No connections. Use tsfm connections manage or /connections in the terminal."
            )


@connections.command("manage")
@guard
def manage_connections(
    state_directory: StateDirectory = None,
):
    """Open a persistent menu with hidden credential prompts."""
    if not interactive_terminal():
        raise ConfigError(
            "Connection setup needs terminal input and output for hidden credential prompts."
        )
    from rich.console import Console
    from rich.text import Text

    from tsfm_lab.connections.ui import ConnectionMenu
    from tsfm_lab.repl.prompt import TerminalInput

    console = Console(markup=False, highlight=False)
    ConnectionMenu(
        TerminalInput(lambda: []),
        lambda value: console.print(Text(literal(value))),
        manager=ConnectionManager(StateStore(state_directory)),
    ).run()
