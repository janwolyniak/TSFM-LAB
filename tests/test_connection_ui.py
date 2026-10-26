"""Exercise real connection UI and manager with fake keys and in-memory replies."""

import io
import json

import pytest
from rich.console import Console
from typer.testing import CliRunner

from tsfm_lab.cli import app
from tsfm_lab.connections.manager import ConnectionManager
from tsfm_lab.connections.state import StateStore
from tsfm_lab.connections.transport import HTTPResponse
from tsfm_lab.connections.ui import ConnectionMenu
from tsfm_lab.repl.prompt import TerminalInput
from tsfm_lab.repl.session import TerminalSession

FAKE_KEY = "fake-ui-key-never-real-123456"


class Editor:
    def __init__(self, answers, *, secrets=(FAKE_KEY,), lines=()):
        self.answers, self.secrets, self.lines = (
            iter(answers),
            iter(secrets),
            iter(lines),
        )
        self.hidden_calls, self.questions = [], []

    def ask(self, label, default="", choices=()):
        self.questions.append(label)
        answer = next(self.answers, default)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def ask_secret(self, label):
        self.hidden_calls.append(label)
        return next(self.secrets)

    def read(self):
        return next(self.lines, "/exit")


class FakeTransport:
    def __init__(self):
        self.requests = []

    def send(self, request, **kwargs):
        self.requests.append(request)
        if request.method == "GET":
            payload = {"data": [{"id": "fake-assistant-model"}]}
        else:
            payload = {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "Connection confirmed."}
                        ],
                    }
                ],
            }
        return HTTPResponse(200, json.dumps(payload).encode())


def menu(tmp_path, answers):
    editor = Editor(answers)
    transport = FakeTransport()
    manager = ConnectionManager(
        StateStore(tmp_path.resolve() / "connections"), transport=transport
    )
    output = []
    ConnectionMenu(editor, output.append, manager=manager).run()
    return editor, manager, transport, "\n".join(output)


def test_add_select_discover_preview_send_and_disconnect(tmp_path):
    editor, manager, transport, output = menu(
        tmp_path,
        [
            "add",
            "openai_api",
            "My API",
            "session",
            "select",
            "1",
            "models",
            "fake-assistant-model",
            "test",
            "yes",
            "disconnect",
            "1",
            "yes",
            "back",
        ],
    )
    assert len(editor.hidden_calls) == 1
    assert len(transport.requests) == 2
    request = transport.requests[1]
    assert (
        request.method == "POST"
        and request.url == "https://api.openai.com/v1/responses"
    )
    assert "OpenAI API billing" in output and "Exact outgoing" in output
    assert output.index("Exact outgoing") < output.index("Connection-check reply")
    assert FAKE_KEY not in output
    assert FAKE_KEY not in manager.state.path.read_text()
    assert manager.session._secrets == {}
    assert manager.list()[0]["credentials_saved"] is False


def test_no_confirmation_or_interrupt_does_not_send_inference(tmp_path):
    for choice in ("no", KeyboardInterrupt()):
        location = tmp_path / str(type(choice).__name__)
        _, _, transport, output = menu(
            location,
            [
                "add",
                "openai_api",
                "API",
                "session",
                "select",
                "1",
                "models",
                "fake-assistant-model",
                "test",
                choice,
                "back",
            ],
        )
        assert len(transport.requests) == 1
        assert transport.requests[0].method == "GET"
        assert "cancelled" in output


def test_status_is_nonsecret_and_does_not_read_the_vault_or_network(
    tmp_path, monkeypatch
):
    manager = ConnectionManager(StateStore(tmp_path.resolve() / "connections"))
    manager.add_key("openai_api", "fake account", FAKE_KEY, storage="session")

    def forbidden(*args, **kwargs):
        raise AssertionError("No protected storage or HTTP access permitted")

    monkeypatch.setattr(manager.session, "get", forbidden)
    monkeypatch.setattr(manager.vault, "get", forbidden)
    monkeypatch.setattr(manager.transport, "send", forbidden)
    output = []
    ConnectionMenu(Editor(["status", "back"]), output.append, manager=manager).run()
    assert "fake account" in "\n".join(output)
    assert FAKE_KEY not in "\n".join(output)


def test_secret_prompt_is_separate_hidden_and_has_no_history(monkeypatch):
    sessions = []

    class Prompt:
        def __init__(self, **options):
            self.options = options
            self.calls = []
            self.resets = []
            self.default_buffer = type(
                "Buffer", (), {"reset": lambda _, **kwargs: self.resets.append(kwargs)}
            )()
            sessions.append(self)

        def prompt(self, *args, **kwargs):
            self.calls.append(kwargs)
            return FAKE_KEY

    monkeypatch.setattr("tsfm_lab.repl.prompt.PromptSession", Prompt)
    editor = TerminalInput(lambda: [])
    assert editor.ask_secret("key") == FAKE_KEY
    assert len(sessions) == 3
    assert sessions[2].options["history"].__class__.__name__ == "DummyHistory"
    assert sessions[2].options["auto_suggest"] is None
    assert sessions[2].options["clipboard"].__class__.__name__ == "DummyClipboard"
    assert sessions[2].calls == [{"is_password": True, "enable_history_search": False}]
    assert not sessions[0].calls and not sessions[1].calls
    assert sessions[2].resets == [{"append_to_history": False}]


def test_connections_reachable_during_setup_and_session_credentials_survive_menu(
    tmp_path, monkeypatch
):
    editor = Editor(
        ["/connections", "add", "openai_api", "API", "session", "back", "generic"],
        lines=["/new", "/connections", "status", "/exit"],
    )
    # The second menu uses the same editor answer stream, so close it explicitly.
    editor.lines = iter(["/new", "/connections", "/exit"])
    editor.answers = iter(
        [
            "/connections",
            "add",
            "openai_api",
            "API",
            "session",
            "back",
            "generic",
            "status",
            "back",
        ]
    )
    manager = ConnectionManager(
        StateStore(tmp_path.resolve() / "connections"), transport=FakeTransport()
    )

    def build(ask, emit, **kwargs):
        assert ask("Usage") == "generic"
        assert manager.list()[0]["credentials_saved"]
        return None

    monkeypatch.setattr("tsfm_lab.repl.session.build_experiment", build)
    output = io.StringIO()
    TerminalSession(
        tmp_path / "results",
        console=Console(file=output, color_system=None),
        editor=editor,
        connection_manager=manager,
    ).run()
    assert "API" in output.getvalue()
    assert len(editor.hidden_calls) == 1
    assert not (tmp_path / "results").exists()


def test_connection_cli_is_safe_for_noninteractive_metadata(tmp_path, monkeypatch):
    directory = tmp_path.resolve() / "settings"
    result = CliRunner().invoke(
        app, ["connections", "list", "--state-directory", str(directory), "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []
    assert not directory.exists()
    monkeypatch.setattr(
        "tsfm_lab.connection_commands.interactive_terminal", lambda: False
    )
    result = CliRunner().invoke(
        app, ["connections", "manage", "--state-directory", str(directory)]
    )
    assert result.exit_code == 2 and "hidden credential" in result.stderr
    assert not directory.exists()


def test_unexpected_ui_errors_are_canned_and_terminal_controls_literal(tmp_path):
    class BrokenManager:
        def list(self):
            raise RuntimeError(FAKE_KEY)

    output = []
    ConnectionMenu(
        Editor(["status", "back"]), output.append, manager=BrokenManager()
    ).run()
    assert FAKE_KEY not in "\n".join(output)
    output = []
    item = {
        "label": "fake\x1b]52;c;INJECT\x07",
        "provider": "provider",
        "selected": True,
        "usage": "usage",
        "storage": "session",
        "model": None,
        "status": "saved",
        "account": None,
        "workspace_id": None,
    }
    ConnectionMenu(
        Editor(["status", "back"]),
        output.append,
        manager=type("Manager", (), {"list": lambda self: [item]})(),
    ).run()
    assert "\x1b" not in "\n".join(output) and "\\x1b" in "\n".join(output)


def test_connected_credentials_do_not_enter_records_or_exports(
    tmp_path, config_factory
):
    from tsfm_lab.application import run_from_config
    from tsfm_lab.exports import export_predictions

    manager = ConnectionManager(
        StateStore(tmp_path.resolve() / "connections"), transport=FakeTransport()
    )
    item = manager.add_key("openai_api", "API", FAKE_KEY, storage="session")
    manager.select(item["id"])
    result = run_from_config(config_factory())
    for suffix in ("csv", "json"):
        export_predictions(
            result["run_id"], result["workspace"], tmp_path / f"forecast.{suffix}"
        )
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert FAKE_KEY.encode() not in path.read_bytes()


@pytest.mark.parametrize(
    "account,workspace,expected",
    [
        (
            {"subject": "fake-subject", "email": "fake@example.invalid"},
            None,
            "Account fake@example.invalid",
        ),
        (None, "workspace-fixture", "Anthropic workspace workspace-fixture"),
    ],
)
def test_preview_shows_bound_account_or_workspace_without_internal_nonce(
    account, workspace, expected
):
    from tsfm_lab.connections.contracts import ConnectionError

    class Manager:
        def preview_test(self):
            return {
                "id": "internal-nonce-not-for-display",
                "connection_label": "Work connection",
                "provider": "Fake provider",
                "usage": "Selected usage",
                "model": "fake-model",
                "account": account,
                "workspace_id": workspace,
                "instructions": "Exact instructions",
                "text": "Exact message",
                "data_included": "Fixed check only",
                "local_limits": {"seconds": 30, "received_bytes": 262144},
                "output_tokens": None,
            }

        def submit_test(self, preview, *, confirmed):
            assert confirmed is False
            raise ConnectionError("permission")

    output = []
    ConnectionMenu(
        Editor(["test", "no", "back"]), output.append, manager=Manager()
    ).run()
    value = "\n".join(output)
    assert expected in value and "Work connection" in value
    assert (
        "Instructions: Exact instructions" in value
        and "Message: Exact message" in value
    )
    assert "internal-nonce-not-for-display" not in value
