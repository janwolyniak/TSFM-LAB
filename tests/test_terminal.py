"""TTY routing must never open a full-screen UI for scripts."""

import pytest
from typer.testing import CliRunner

from tsfm_lab.cli import app
from tsfm_lab.terminal import interactive_terminal


@pytest.mark.parametrize(
    "stdin,stdout,expected",
    [
        (True, True, True),
        (False, True, False),
        (True, False, False),
        (False, False, False),
    ],
)
def test_interactive_requires_both_terminal_streams(
    monkeypatch, stdin, stdout, expected
):
    monkeypatch.setattr("sys.stdin.isatty", lambda: stdin)
    monkeypatch.setattr("sys.stdout.isatty", lambda: stdout)
    assert interactive_terminal() is expected


@pytest.mark.parametrize("arguments", [[], ["ui", "--workspace", "custom-runs"]])
def test_tty_launches_ui_only_when_requested(monkeypatch, arguments):
    launched = []
    monkeypatch.setattr("tsfm_lab.cli.interactive_terminal", lambda: True)
    monkeypatch.setattr("tsfm_lab.cli.launch_tui", lambda *args: launched.append(args))
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert len(launched) == 1
    if arguments:
        assert str(launched[0][0]) == "custom-runs"


@pytest.mark.parametrize(
    "arguments", [["--help"], ["--version"], ["models", "list"], ["ui", "--help"]]
)
def test_script_commands_never_open_ui_even_in_tty(monkeypatch, arguments):
    def forbidden(*args):
        raise AssertionError("Unexpected full-screen UI")

    monkeypatch.setattr("tsfm_lab.cli.interactive_terminal", lambda: True)
    monkeypatch.setattr("tsfm_lab.cli.launch_tui", forbidden)
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output


def test_noninteractive_ui_reports_actionable_error(monkeypatch):
    monkeypatch.setattr("tsfm_lab.cli.interactive_terminal", lambda: False)
    result = CliRunner().invoke(app, ["ui"])
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "terminal input and output" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("arguments", [[], ["--version"], ["models", "list", "--json"]])
def test_script_commands_ignore_even_corrupt_interactive_preferences(
    arguments, tmp_path, monkeypatch
):
    monkeypatch.setenv("TSFM_LAB_SETTINGS_DIR", str(tmp_path))
    path = tmp_path / "preferences.json"
    path.write_bytes(b"invalid preferences")
    result = CliRunner().invoke(app, arguments, input="")
    assert result.exit_code == 0, result.output
    assert "First-launch" not in result.output
    assert path.read_bytes() == b"invalid preferences"
