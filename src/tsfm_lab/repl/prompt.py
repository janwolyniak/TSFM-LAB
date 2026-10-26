"""Small line editor: completion and history without taking over the screen."""

import os
import shlex
from collections.abc import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.clipboard import DummyClipboard
from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
    DummyCompleter,
    PathCompleter,
    WordCompleter,
)
from prompt_toolkit.document import Document
from prompt_toolkit.history import DummyHistory, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from tsfm_lab.models.catalog import catalog_entries
from tsfm_lab.repl.text import literal

COMMANDS = {
    "/help": "Commands and shortcuts",
    "/demo": "Try the original seasonal dataset",
    "/new": "Set up an experiment one question at a time",
    "/run": "Run a saved YAML configuration",
    "/models": "Browse the local model catalog",
    "/connections": "Manage optional assistant connections",
    "/history": "List saved experiments",
    "/show": "Read a saved result",
    "/compare": "Compare matching experiments",
    "/export": "Export saved predictions to a new file",
    "/workspace": "Show or change the results directory",
    "/status": "Show session context",
    "/exit": "Return to your shell",
}


class CommandCompleter(Completer):
    def __init__(self, run_ids: Callable[[], list[str]] = lambda: []):
        self.run_ids = run_ids
        self.paths = PathCompleter(expanduser=True)

    def get_completions(self, document, complete_event):
        line = document.text_before_cursor
        words = line.split()
        if not line.strip() or (len(words) == 1 and not line.endswith(" ")):
            for command, description in COMMANDS.items():
                if command.startswith(line):
                    yield Completion(command, -len(line), display_meta=description)
            return
        command, word = words[0], document.get_word_before_cursor(WORD=True)
        if command in {"/show", "/compare", "/export"}:
            # Export's second argument is a path rather than another run ID.
            position = len(words) if line.endswith(" ") else len(words) - 1
            if command != "/export" or position == 1:
                for run_id in ["latest", *self.run_ids()]:
                    if run_id.startswith(word):
                        yield Completion(run_id, -len(word))
                return
        if command in {"/new", "/demo"}:
            for entry in catalog_entries():
                adapter = entry["adapter_id"]
                if adapter.startswith(word):
                    yield Completion(adapter, -len(word))
        elif command in {"/run", "/workspace", "/export"}:
            count = 2 if command == "/export" else 1
            parts = line.split(maxsplit=count)
            raw = parts[count] if len(parts) > count else ""
            try:
                parsed = shlex.split(raw)
            except ValueError:
                quote = raw[0] if raw.startswith(("'", '"')) else ""
                try:
                    parsed = shlex.split(raw + quote)
                except ValueError:
                    return
            path = parsed[0] if parsed else ""
            for completion in self.paths.get_completions(
                Document(path), complete_event
            ):
                candidate = (
                    path[: len(path) + completion.start_position] + completion.text
                )
                yield Completion(
                    shlex.quote(candidate), -len(raw), display=completion.display
                )


class TerminalInput:
    def __init__(self, run_ids: Callable[[], list[str]]):
        style = Style.from_dict(
            {}
            if "NO_COLOR" in os.environ
            else {"prompt": "ansicyan bold", "hint": "ansibrightblack"}
        )
        completer = CommandCompleter(run_ids)
        commands = KeyBindings()

        @commands.add("tab")
        def complete(event):
            buffer = event.current_buffer
            options = list(
                completer.get_completions(
                    buffer.document, CompleteEvent(completion_requested=True)
                )
            )
            if len(options) == 1:
                buffer.apply_completion(options[0])
            elif buffer.complete_state is not None:
                buffer.complete_next()
            else:
                buffer.start_completion(select_first=True)

        self.session = PromptSession(
            history=InMemoryHistory(),
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
            key_bindings=commands,
            complete_while_typing=True,
            reserve_space_for_menu=3,
            style=style,
            bottom_toolbar=[
                (
                    "class:hint",
                    " /help commands   ↑ history   tab complete   ctrl+d exit",
                )
            ],
        )
        keys = KeyBindings()

        @keys.add("escape")
        def cancel(event):
            event.app.exit(exception=KeyboardInterrupt())

        self.questions = PromptSession(
            history=DummyHistory(), style=style, key_bindings=keys
        )
        self.secrets = PromptSession(
            history=DummyHistory(),
            style=style,
            key_bindings=keys,
            completer=DummyCompleter(),
            auto_suggest=None,
            clipboard=DummyClipboard(),
        )

    def read(self) -> str:
        return self.session.prompt([("class:prompt", "› ")])

    def ask_secret(self, label: str) -> str:
        try:
            return self.secrets.prompt(
                [("class:hint", literal(label)), ("class:prompt", " › ")],
                is_password=True,
                enable_history_search=False,
            )
        finally:
            self.secrets.default_buffer.reset(append_to_history=False)

    def ask(self, label: str, default: str = "", choices=()) -> str:
        suffix = f" [{default}]" if default else ""
        value = self.questions.prompt(
            [("class:hint", literal(f"  {label}{suffix}")), ("class:prompt", " › ")],
            completer=WordCompleter(list(choices)) if choices else DummyCompleter(),
        )
        return value.strip() or default
