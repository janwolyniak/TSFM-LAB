"""A persistent local command prompt with inline results and guided setup."""

from contextlib import ExitStack
from difflib import get_close_matches
from importlib.resources import as_file, files
from pathlib import Path

from rich.console import Console
from rich.text import Text

from tsfm_lab import __version__
from tsfm_lab.config import ConfigError, load_run_config
from tsfm_lab.local_paths import active_path_text
from tsfm_lab.repl.arguments import split_arguments
from tsfm_lab.repl.progress import execute, execute_guided
from tsfm_lab.repl.prompt import COMMANDS, TerminalInput
from tsfm_lab.repl.records import RecordCommands
from tsfm_lab.repl.text import literal
from tsfm_lab.repl.wizard import build_experiment
from tsfm_lab.workflows import PROFILE_NAMES, workflow_summary


class TerminalSession:
    def __init__(
        self,
        workspace=None,
        *,
        console=None,
        editor=None,
        runner=execute,
        connection_manager=None,
        preference_store=None,
        guided_runner=None,
    ):
        self.preference_store = preference_store
        self.preferences = preference_store.load() if preference_store else None
        if workspace is not None:
            active_path_text(str(workspace))
        self._workspace_override = (
            Path(workspace).expanduser().resolve() if workspace is not None else None
        )
        selected = (
            workspace
            if workspace is not None
            else (
                self.preferences.results_directory
                if self.preferences
                else Path("results")
            )
        )
        self.workspace = Path(selected).expanduser().resolve()
        self.console = console or Console(markup=False, highlight=False)
        self.editor = editor
        self.runner = runner
        self.guided_runner = guided_runner or (
            execute_guided
            if runner is execute
            else lambda plan, console: runner(plan.config, console)
        )
        self.records = RecordCommands(self.console, self.workspace)
        self.demo_path = None
        self.ttm_demo_path = None
        self.profile_demo_paths = {}
        self.last_run = None
        self.known_ids = []
        self.closed = False
        self.connection_manager = connection_manager
        self.assistant_service = None

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
            if self.preferences and self.preferences.onboarding_status == "new":
                self._configure_preferences(first_launch=True)
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
            words = split_arguments(line)
            command, arguments = words[0], words[1:]
            command = command if command.startswith("/") else f"/{command}"
            handlers = {
                "/help": self._help,
                "/demo": self._demo,
                "/new": self._new,
                "/setup": self._setup_preferences,
                "/settings": self._settings,
                "/run": self._run,
                "/models": self._models,
                "/connections": self._connections,
                "/explain": self._explain,
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
        method = "guided"
        if not demo:
            if arguments and arguments[0] in {"guided", "describe"}:
                method, arguments = arguments[0], []
            elif not arguments:
                method = self._setup_ask(
                    "Setup method", "guided", ("guided", "describe")
                )
                if method in {"/cancel", "/back"}:
                    return
                if method not in {"guided", "describe"}:
                    raise ConfigError(
                        "Choose guided or describe to start an experiment"
                    )
        if method == "describe":
            self._assisted_setup()
            return
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
            preferences=self.preferences,
            guided_preparation=True,
        )
        if config is not None:
            self._execute(config)
        else:
            self.emit("Setup cancelled.", "dim")

    def _run(self, arguments):
        self._count(arguments, 1, 1, "/run CONFIG.yaml")
        self._execute(load_run_config(arguments[0]))

    def _execute(self, config):
        from tsfm_lab.guided import GuidedRunPlan
        from tsfm_lab.preferences import validate_directory

        plan = config if isinstance(config, GuidedRunPlan) else None
        if plan is not None:
            config = plan.config
        validate_directory(config.output_directory)
        if self.workspace != config.output_directory:
            self.known_ids = []
            self.last_run = None
            self.emit(f"Workspace  {config.output_directory}", "dim")
        self.workspace = config.output_directory
        self.records.workspace = self.workspace
        from tsfm_lab.run_modes import forecast_spec, is_future

        spec = forecast_spec(config)
        self.emit(f"\n{config.name} · {config.model['adapter_id']}", "bold")
        self.emit(f"Usage  {workflow_summary(config.workflow)}")
        self.emit(
            f"Target {config.dataset['target_column']} · {config.dataset['units']} · {config.dataset['frequency']} · horizon {spec.horizon}"
        )
        event = (
            self.guided_runner(plan, self.console)
            if plan is not None
            else self.runner(config, self.console)
        )
        if event.get("preparation_path"):
            self.emit(f"Preparation record: {event['preparation_path']}", "dim")
        run_id = event.get("run_id") or event.get("result", {}).get("run_id")
        if run_id:
            self.last_run = run_id
            self.known_ids = list(dict.fromkeys([run_id, *self.known_ids]))
        if event["type"] == "completed":
            self.emit(f"Completed · {run_id}", "green")
            self.records.show(run_id)
            self._saved_paths(run_id)
            self.emit(
                (
                    "Next: /export latest forecasts.csv"
                    if is_future(config)
                    else "Next: /compare latest OTHER_ID · /export latest forecasts.csv"
                ),
                "dim",
            )
            self.emit("Optional: /explain run latest", "dim")
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

    def _assistant(self):
        from tsfm_lab.assistant.service import AssistantService
        from tsfm_lab.assistant.ui import AssistantFlow
        from tsfm_lab.connections.manager import ConnectionManager

        if self.connection_manager is None:
            self.connection_manager = ConnectionManager()
        if self.assistant_service is None:
            self.assistant_service = AssistantService(self.connection_manager)
        return AssistantFlow(
            self.editor,
            self.emit,
            self.assistant_service,
            connections=lambda: self._connections([]),
        )

    def _assisted_setup(self):
        def review(resolution):
            options = (
                {}
                if resolution is None
                else {
                    "seed_fields": resolution.seed_fields,
                    "explicit_fields": resolution.explicit_fields,
                }
            )
            return build_experiment(
                self._setup_ask,
                self.emit,
                workspace=self.workspace,
                demo_path=self.demo_path,
                ttm_demo_path=self.ttm_demo_path,
                profile_demo_paths=self.profile_demo_paths,
                preferences=self.preferences,
                guided_preparation=True,
                **options,
            )

        config = self._assistant().describe(review, default_dataset=self.demo_path)
        if config is not None:
            self._execute(config)
        else:
            self.emit("Assisted setup finished without starting an experiment.", "dim")

    def _explain(self, arguments):
        self._count(
            arguments,
            2,
            4,
            "/explain model ID [ID ...] | /explain run ID | /explain finance REPORT.json",
        )
        from tsfm_lab.assistant.evidence import (
            bounded_json,
            experiment_evidence,
            finance_evidence,
            model_evidence,
        )
        from tsfm_lab.experiments import ExperimentStore

        kind, selected = arguments[0], arguments[1:]
        if kind == "model":
            contexts = [model_evidence(identifier) for identifier in selected]
            context = contexts[0]
            if len(contexts) > 1:
                context = {
                    **context,
                    "kind": "model_comparison",
                    "data": {"models": [item["data"] for item in contexts]},
                }
            question = "Explain these model capabilities and verification limits without ranking forecast quality."
        elif kind == "run" and len(selected) == 1:
            identifier = self.records.resolve_id(selected[0])
            context = experiment_evidence(ExperimentStore(self.workspace), identifier)
            question = (
                "Explain this future forecast's origin, point semantics, units, and unavailable accuracy."
                if context.get("kind") == "future_forecast"
                else (
                    "Explain the historical evaluation, its scores, units, and limitations."
                )
            )
        elif kind == "finance" and len(selected) == 1:
            context = finance_evidence(Path(selected[0]))
            question = "Explain these saved hypothetical finance results and explicit assumptions without giving trading advice."
        else:
            raise ConfigError(
                "Use /explain model ID, /explain run ID, or /explain finance REPORT.json"
            )
        bounded_json(context)
        self._assistant().explain(context, default_question=question)

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
        saved = self.records.show(arguments[0])
        if saved["manifest"]["status"] == "completed":
            self._saved_paths(saved["manifest"]["run_id"])

    def _saved_paths(self, run_id):
        from tsfm_lab.experiments import ExperimentStore

        paths = ExperimentStore(self.workspace).output_paths(run_id)
        for filename, label in (
            ("predictions.csv", "Forecast CSV"),
            ("result.json", "Results JSON"),
            ("metrics.json", "Evaluation JSON"),
            ("evaluation.json", "Evaluation availability JSON"),
            ("observations.csv", "Consumed observations CSV"),
            ("provenance.json", "Provenance JSON"),
            ("forecast.svg", "Forecast chart"),
        ):
            if filename in paths:
                self.emit(f"{label}: {paths[filename]}", "dim")

    def _compare(self, arguments):
        self._count(arguments, 2, 100, "/compare ID ID [ID…]")
        self.records.compare(arguments)

    def _export(self, arguments):
        self._count(arguments, 2, 3, "/export ID PATH [csv|json|parquet]")
        path = Path(arguments[1]).expanduser()
        selected = arguments[2] if len(arguments) == 3 else None
        if self.preferences is not None:
            if not path.is_absolute():
                path = self.preferences.workspace / path
            if selected is None and not path.suffix:
                selected = self.preferences.export_format
        self.records.export(arguments[0], path, selected)

    def _setup_preferences(self, arguments):
        self._count(arguments, 0, 0, "/setup")
        self._configure_preferences()

    def _settings(self, arguments):
        self._count(arguments, 0, 0, "/settings")
        from tsfm_lab.repl.onboarding import show_preferences

        self._load_preferences()
        show_preferences(self.emit, self.preferences, self.preference_store.path)
        if self._setup_ask("Edit preferences?", "no", ("yes", "no")) in {"yes", "y"}:
            self._configure_preferences(extras=False)

    def _load_preferences(self):
        if self.preference_store is None:
            from tsfm_lab.preferences import PreferenceStore

            self.preference_store = PreferenceStore()
            self.preferences = self.preference_store.load()

    def _configure_preferences(self, *, first_launch=False, extras=True):
        from tsfm_lab.repl.onboarding import configure_preferences

        self._load_preferences()
        try:
            saved = configure_preferences(
                self.editor.ask,
                self.emit,
                self.preferences,
                self.preference_store,
                first_launch=first_launch,
            )
            if saved is None:
                return
            self.preferences = saved
            self.workspace = self._workspace_override or saved.results_directory
            self.records.workspace = self.workspace
            self.last_run, self.known_ids = None, []
            self.emit(f"Results directory  {self.workspace}")
            if self._workspace_override:
                self.emit(
                    "This session uses the explicit --workspace directory.", "dim"
                )
            if extras and saved.onboarding_status == "completed":
                if self._setup_ask(
                    "Set up an optional assistant?", "no", ("yes", "no")
                ) in {"yes", "y"}:
                    self._connections([])
                if self._setup_ask(
                    "Try the offline baseline demo?", "no", ("yes", "no")
                ) in {"yes", "y"}:
                    self._demo([])
        except (EOFError, KeyboardInterrupt):
            self.emit("Setup finished. Use /setup to revisit preferences.", "dim")
        except (OSError, ValueError, ImportError, RuntimeError) as exc:
            self.emit(f"Setup needs attention: {exc}. Use /setup to retry.", "red")

    def _workspace(self, arguments):
        self._count(arguments, 0, 1, "/workspace [DIRECTORY]")
        if arguments:
            if not arguments[0].strip() or "://" in arguments[0]:
                raise ConfigError("Enter a local workspace directory")
            active_path_text(arguments[0])
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
