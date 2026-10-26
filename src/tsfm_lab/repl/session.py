"""A persistent local command prompt with inline results and guided setup."""

import shlex
from contextlib import ExitStack
from difflib import get_close_matches
from importlib.resources import as_file, files
from pathlib import Path

from rich.console import Console
from rich.text import Text

from tsfm_lab import __version__
from tsfm_lab.config import ConfigError, load_config
from tsfm_lab.repl.progress import execute
from tsfm_lab.repl.prompt import COMMANDS, TerminalInput
from tsfm_lab.repl.records import RecordCommands
from tsfm_lab.repl.text import literal
from tsfm_lab.repl.wizard import build_experiment
from tsfm_lab.workflows import PROFILE_NAMES, workflow_summary


class TerminalSession:
    def __init__(
        self,
        workspace=Path("results"),
        *,
        console=None,
        editor=None,
        runner=execute,
        connection_manager=None,
    ):
        self.workspace = Path(workspace).expanduser().resolve()
        self.console = console or Console(markup=False, highlight=False)
        self.editor = editor
        self.runner = runner
        self.records = RecordCommands(self.console, self.workspace)
        self.demo_path = None
        self.ttm_demo_path = None
        self.profile_demo_paths = {}
        self.last_run = None
        self.known_ids = []
        self.closed = False
        self.connection_manager = connection_manager

    def emit(self, value, style=None):
        self.console.print(Text(literal(value), style=style))

    def _welcome(self):
        self.emit(f"\nTSFM LAB  {__version__}", "bold cyan")
        self.emit("Forecasting, in your terminal.")
        self.emit(f"Workspace  {self.workspace}", "dim")
        self.emit(
            "\n/demo to try it · /new to set up a run · /help for commands\n", "dim"
        )

    def run(self):
        if self.editor is None:
            self.editor = TerminalInput(lambda: list(self.known_ids))
        with ExitStack() as resources:
            self.demo_path = resources.enter_context(
                as_file(files("tsfm_lab.resources").joinpath("seasonal.csv"))
            )
            self.ttm_demo_path = resources.enter_context(
                as_file(files("tsfm_lab.resources").joinpath("ttm-seasonal.csv"))
            )
            self.profile_demo_paths = {
                profile: resources.enter_context(
                    as_file(files("tsfm_lab.resources").joinpath(f"{profile}-demo.csv"))
                )
                for profile in PROFILE_NAMES
                if profile != "generic"
            }
            self._welcome()
            while not self.closed:
                try:
                    self.dispatch(self.editor.read())
                except EOFError:
                    self.closed = True
                except KeyboardInterrupt:
                    self.emit("\nInput cancelled. Use /exit or ctrl+d to leave.", "dim")
            self.emit("Session closed. Results remain in your workspace.", "dim")

    def dispatch(self, line):
        if not line.strip():
            return
        try:
            words = shlex.split(line)
            command, arguments = words[0], words[1:]
            command = command if command.startswith("/") else f"/{command}"
            handlers = {
                "/help": self._help,
                "/demo": self._demo,
                "/new": self._new,
                "/run": self._run,
                "/models": self._models,
                "/connections": self._connections,
                "/history": self._history,
                "/show": self._show,
                "/compare": self._compare,
                "/export": self._export,
                "/workspace": self._workspace,
                "/status": self._status,
                "/exit": self._exit,
                "/quit": self._exit,
            }
            if command not in handlers:
                suggestion = get_close_matches(command, COMMANDS, n=1)
                hint = (
                    f" Did you mean {suggestion[0]}?"
                    if suggestion
                    else " Type /help for commands."
                )
                raise ConfigError(f"Unknown command: {words[0]}.{hint}")
            handlers[command](arguments)
        except (EOFError, KeyboardInterrupt):
            self.emit("Command cancelled. Back at the prompt.", "dim")
        except (OSError, ValueError, ImportError, RuntimeError) as exc:
            self.emit(f"Error: {exc}", "red")
        self.console.print()

    @staticmethod
    def _count(arguments, minimum, maximum, usage):
        if not minimum <= len(arguments) <= maximum:
            raise ConfigError(f"Usage: {usage}")

    def _help(self, arguments):
        self._count(arguments, 0, 0, "/help")
        for command, description in COMMANDS.items():
            self.emit(f"  {command:<12} {description}")
        self.emit(
            "\n/show ID · /compare ID ID · /export ID PATH [csv|json|parquet]", "dim"
        )
        self.emit(
            "Use latest or a unique run ID prefix. Quote paths containing spaces.",
            "dim",
        )
        self.emit(
            "Tab completes · ↑/↓ history · ctrl+c cancels setup or an active run · ctrl+d exits",
            "dim",
        )

    def _demo(self, arguments):
        self._setup(arguments, demo=True)

    def _new(self, arguments):
        self._setup(arguments, demo=False)

    def _setup(self, arguments, *, demo):
        self._count(arguments, 0, 1, "/demo [adapter]" if demo else "/new [adapter]")
        self.emit(
            (
                "Original seasonal demo"
                if demo
                else "New experiment · /back revisits a step · /cancel stops setup"
            ),
            "bold",
        )
        if not demo:
            self.emit(
                "Enter /connections at any setup question to manage optional assistant connections.",
                "dim",
            )
        config = build_experiment(
            self._setup_ask,
            self.emit,
            workspace=self.workspace,
            demo_path=self.demo_path,
            ttm_demo_path=self.ttm_demo_path,
            profile_demo_paths=self.profile_demo_paths,
            adapter=arguments[0] if arguments else None,
            demo=demo,
        )
        if config is not None:
            self._execute(config)
        else:
            self.emit("Setup cancelled.", "dim")

    def _run(self, arguments):
        self._count(arguments, 1, 1, "/run CONFIG.yaml")
        self._execute(load_config(arguments[0]))

    def _execute(self, config):
        if self.workspace != config.output_directory:
            self.known_ids = []
            self.last_run = None
            self.emit(f"Workspace  {config.output_directory}", "dim")
        self.workspace = config.output_directory
        self.records.workspace = self.workspace
        spec = config.evaluation
        self.emit(f"\n{config.name} · {config.model['adapter_id']}", "bold")
        self.emit(f"Usage  {workflow_summary(config.workflow)}")
        self.emit(
            f"Target {config.dataset['target_column']} · {config.dataset['units']} · {config.dataset['frequency']} · horizon {spec.horizon}"
        )
        event = self.runner(config, self.console)
        run_id = event.get("run_id") or event.get("result", {}).get("run_id")
        if run_id:
            self.last_run = run_id
            self.known_ids = list(dict.fromkeys([run_id, *self.known_ids]))
        if event["type"] == "completed":
            self.emit(f"Completed · {run_id}", "green")
            self.records.show(run_id)
            self.emit(
                "Next: /compare latest OTHER_ID · /export latest forecasts.csv", "dim"
            )
        else:
            message = (
                "Experiment cancelled"
                if event["type"] == "cancelled"
                else event.get("message", event.get("error", "Experiment failed"))
            )
            self.emit(message, "yellow" if event["type"] == "cancelled" else "red")
            if event.get("log_tail"):
                self.emit(event["log_tail"], "dim")
            if run_id:
                self.emit(f"Saved record: {run_id}", "dim")

    def _models(self, arguments):
        self._count(arguments, 0, 0, "/models")
        self.records.models()

    def _connections(self, arguments):
        self._count(arguments, 0, 0, "/connections")
        from tsfm_lab.connections.manager import ConnectionManager
        from tsfm_lab.connections.ui import ConnectionMenu

        if self.connection_manager is None:
            self.connection_manager = ConnectionManager()
        ConnectionMenu(self.editor, self.emit, manager=self.connection_manager).run()

    def _setup_ask(self, label, default="", choices=()):
        while True:
            answer = self.editor.ask(label, default, choices)
            if answer.strip() != "/connections":
                return answer
            self._connections([])

    def _history(self, arguments):
        self._count(arguments, 0, 0, "/history")
        records = self.records.history()
        self.known_ids = [record["run_id"] for record in records]

    def _show(self, arguments):
        self._count(arguments, 1, 1, "/show ID")
        self.records.show(arguments[0])

    def _compare(self, arguments):
        self._count(arguments, 2, 100, "/compare ID ID [ID…]")
        self.records.compare(arguments)

    def _export(self, arguments):
        self._count(arguments, 2, 3, "/export ID PATH [csv|json|parquet]")
        self.records.export(
            arguments[0], arguments[1], arguments[2] if len(arguments) == 3 else None
        )

    def _workspace(self, arguments):
        self._count(arguments, 0, 1, "/workspace [DIRECTORY]")
        if arguments:
            if not arguments[0].strip() or "://" in arguments[0]:
                raise ConfigError("Enter a local workspace directory")
            self.workspace = Path(arguments[0]).expanduser().resolve()
            self.records.workspace = self.workspace
            self.known_ids = []
            self.last_run = None
        self.emit(f"Workspace  {self.workspace}")

    def _status(self, arguments):
        self._count(arguments, 0, 0, "/status")
        self.emit(f"Workspace  {self.workspace}")
        self.emit(f"Last run   {self.last_run or 'none in this session'}")
        self.emit(
            "Local commands · Forecasts run locally · Optional assistant connections",
            "dim",
        )
        if self.connection_manager is not None:
            from tsfm_lab.connections.contracts import ConnectionError

            try:
                item = self.connection_manager.status()
                self.emit(
                    f"Assistant {item['provider']} · {item['usage']} · model {item['model'] or 'not selected'}"
                )
            except ConnectionError:
                self.emit("No active assistant connection.", "dim")

    def _exit(self, arguments):
        self._count(arguments, 0, 0, "/exit")
        self.closed = True
