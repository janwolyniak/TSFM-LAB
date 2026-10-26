import io
from pathlib import Path

import pandas as pd
import pytest
from rich.console import Console

from tsfm_lab.application import run_config, run_from_config
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.repl.session import TerminalSession


class Editor:
    def __init__(self, lines, answers=()):
        self.lines, self.answers = iter(lines), iter(answers)

    def read(self):
        value = next(self.lines, EOFError())
        if isinstance(value, BaseException):
            raise value
        return value

    def ask(self, label, default="", choices=()):
        value = next(self.answers, default)
        if isinstance(value, BaseException):
            raise value
        return value


def session(tmp_path, editor, runner=None):
    output = io.StringIO()
    options = {"runner": runner} if runner else {}
    return (
        TerminalSession(
            tmp_path,
            console=Console(file=output, width=80, color_system=None),
            editor=editor,
            **options,
        ),
        output,
    )


def synchronous(config, console):
    result = run_config(config)
    return {"type": "completed", "run_id": result["run_id"], "result": result}


def test_local_demo_history_export_and_metrics_match_shell(tmp_path, config_factory):
    workspace = tmp_path / "prompt"
    target = tmp_path / "exported forecasts.json"
    editor = Editor(
        [
            "/models",
            "/demo",
            "/history",
            "/show latest",
            f'/export latest "{target}"',
            "/exit",
        ],
        ["yes"],
    )
    app, output = session(workspace, editor, synchronous)
    app.run()
    assert app.closed
    assert target.exists()
    store = ExperimentStore(workspace)
    result, frame = store.load_result(app.last_run)
    shell = run_from_config(config_factory())
    shell_result, shell_frame = ExperimentStore(tmp_path / "workspace").load_result(
        shell["run_id"]
    )
    pd.testing.assert_frame_equal(frame, shell_frame)
    assert result["metrics"] == shell_result["metrics"]
    assert "Completed" in output.getvalue()
    assert "2.5" in output.getvalue()
    assert "last_value" in output.getvalue()


def test_errors_recover_and_existing_export_is_preserved(tmp_path):
    target = tmp_path / "user.csv"
    target.write_text("preserve me")
    editor = Editor(
        [
            "/new",
            '/workspace "https://remote"',
            "/compare",
            "/modelz",
            '/run "unfinished',
            "/exit",
        ],
        [KeyboardInterrupt()],
    )
    app, output = session(tmp_path / "runs", editor)
    app.run()
    text = output.getvalue()
    assert "cancelled" in text
    assert "local workspace" in text
    assert "Usage: /compare" in text
    assert "Did you mean /models" in text
    assert "No closing quotation" in text
    assert not (tmp_path / "runs").exists()
    assert target.read_text() == "preserve me"


def test_idle_interrupt_then_eof_closes_without_creating_workspace(tmp_path):
    app, output = session(tmp_path / "runs", Editor([KeyboardInterrupt(), EOFError()]))
    app.run()
    assert app.closed
    assert "Input cancelled" in output.getvalue()


def test_setup_keeps_both_packaged_datasets_available_for_the_session(
    tmp_path, monkeypatch
):
    calls = []

    def setup(*args, **kwargs):
        assert kwargs["demo_path"].is_file()
        assert kwargs["ttm_demo_path"].is_file()
        assert len(pd.read_csv(kwargs["demo_path"])) == 32
        assert len(pd.read_csv(kwargs["ttm_demo_path"])) == 1600
        calls.append(kwargs)
        return None

    monkeypatch.setattr("tsfm_lab.repl.session.build_experiment", setup)
    app, _ = session(tmp_path / "runs", Editor(["/new ttm_r2", "/demo", "/exit"]))
    app.run()
    assert len(calls) == 2
    assert calls[0]["adapter"] == "ttm_r2"
    assert calls[1]["demo"] is True
    assert not (tmp_path / "runs").exists()
    assert not app.workspace.exists()


@pytest.mark.parametrize("width", [20, 80, 120])
def test_workspace_switch_resets_session_ids_without_creating_directory(
    tmp_path, width
):
    app, output = session(tmp_path / "old", Editor([]))
    app.console.width = width
    app.last_run, app.known_ids = "old-id", ["old-id"]
    app.dispatch(f'/workspace "{tmp_path / "new directory"}"')
    assert app.records.workspace == tmp_path / "new directory"
    assert app.last_run is None
    assert app.known_ids == []
    assert not app.workspace.exists()
    rendered = "".join(output.getvalue().split())
    assert "".join(str(app.workspace).split()) in rendered


def test_saved_config_respects_its_output_directory(config_factory, tmp_path):
    app, output = session(tmp_path / "initial", Editor([]), synchronous)
    app.dispatch(f'/run "{config_factory()}"')
    assert app.workspace == tmp_path / "workspace"
    assert app.records.workspace == app.workspace
    assert app.last_run
    assert not (tmp_path / "initial").exists()
    assert "Completed" in output.getvalue()


def test_cancelled_and_failed_events_return_to_prompt_without_scores(tmp_path):
    for kind in ("cancelled", "failed"):
        app, output = session(
            tmp_path,
            Editor([]),
            lambda *args, event_type=kind: {
                "type": event_type,
                "message": "native failure",
                "log_tail": "[red]literal[/red]",
            },
        )
        from tsfm_lab.config import load_config

        config = load_config(
            Path(__file__).parents[1] / "examples/configurations/last-value.yaml"
        )
        app._execute(config)
        assert "Completed" not in output.getvalue()
        assert "[red]literal[/red]" in output.getvalue()
        assert "MAE" not in output.getvalue()
