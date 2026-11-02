"""Interactive writers must validate raw aliases before native expansion."""

import asyncio
import io
from functools import partial

import pytest
from rich.console import Console
from test_home_path_portability import _windows_expansion
from test_onboarding import Editor

from tsfm_lab import preferences
from tsfm_lab.local_paths import expand_local_user
from tsfm_lab.repl import onboarding, records, session


@pytest.fixture
def windows_homes(monkeypatch, tmp_path):
    """Exercise the actual shared guard with Windows' invented-home behavior."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(onboarding, "runtime_diagnostics", lambda: [])
    _windows_expansion(monkeypatch, tmp_path)
    guarded = partial(expand_local_user, windows=True)
    for module in (preferences, onboarding, records, session):
        monkeypatch.setattr(module, "expand_local_user", guarded)
    return tmp_path / "unknown"


@pytest.mark.parametrize(
    "label", ["Workspace directory", "Results directory", "Model cache directory"]
)
def test_setup_rejects_raw_unknown_home_before_saving(windows_homes, tmp_path, label):
    store = preferences.PreferenceStore(tmp_path / "settings")
    original = store.load()
    store.save(original)
    before = store.path.read_bytes()
    editor = Editor(answers={label: ["~unknown/new directory", "/cancel"]})
    messages = []
    assert (
        onboarding.configure_preferences(editor.ask, messages.append, original, store)
        is None
    )
    assert any("named user's home" in message for message in messages)
    assert store.path.read_bytes() == before
    assert not windows_homes.exists()


@pytest.mark.parametrize(
    "constructor", [session.TerminalSession, records.RecordCommands]
)
def test_prompt_constructors_reject_raw_unknown_home(windows_homes, constructor):
    console = Console(file=io.StringIO())
    with pytest.raises(ValueError, match="named user's home"):
        if constructor is records.RecordCommands:
            constructor(console, "~unknown/results")
        else:
            constructor("~unknown/results", console=console)
    assert not windows_homes.exists()


def test_workspace_command_rejects_alias_without_changing_session(
    windows_homes, tmp_path
):
    output = io.StringIO()
    app = session.TerminalSession(
        tmp_path / "results", console=Console(file=output, width=120)
    )
    original = app.workspace
    app.last_run, app.known_ids = "previous-run", ["previous-run"]
    app.dispatch('/workspace "~unknown/results"')
    assert "named user's home" in output.getvalue()
    assert app.workspace == app.records.workspace == original
    assert app.last_run == "previous-run"
    assert app.known_ids == ["previous-run"]
    assert not original.exists()
    assert not windows_homes.exists()


def test_export_rejects_raw_alias_before_record_or_destination_access(
    windows_homes, tmp_path, monkeypatch
):
    app = session.TerminalSession(
        tmp_path / "results", console=Console(file=io.StringIO())
    )

    def forbidden_export(*args):
        pytest.fail("An invalid destination must be rejected before exporting")

    monkeypatch.setattr(app.records, "export", forbidden_export)
    with pytest.raises(ValueError, match="named user's home"):
        app._export(["latest", "~unknown/forecast.csv"])
    assert not app.workspace.exists()
    assert not windows_homes.exists()


def test_changed_record_workspace_keeps_raw_alias_guard(windows_homes, tmp_path):
    commands = records.RecordCommands(Console(file=io.StringIO()), tmp_path / "results")
    commands.workspace = "~unknown/results"
    with pytest.raises(ValueError, match="named user's home"):
        commands.history()
    assert not windows_homes.exists()


def test_valid_alias_and_relative_preferences_keep_the_workspace_base(
    windows_homes, tmp_path
):
    (tmp_path / "known").mkdir()
    store = preferences.PreferenceStore(tmp_path / "settings")
    editor = Editor(
        answers={
            "Workspace directory": ["~known/new work"],
            "Results directory": ["forecasts"],
            "Model cache directory": ["weights"],
        }
    )
    saved = onboarding.configure_preferences(
        editor.ask, lambda _: None, store.load(), store
    )
    assert saved.workspace == tmp_path / "known/new work"
    assert saved.results_directory == saved.workspace / "forecasts"
    assert saved.cache_directory == saved.workspace / "weights"
    assert not saved.workspace.exists()
    app = session.TerminalSession(
        "relative results", console=Console(file=io.StringIO())
    )
    assert app.workspace == tmp_path / "relative results"


def test_tui_initial_workspace_rejects_raw_unknown_home(windows_homes, monkeypatch):
    pytest.importorskip("textual")
    from tsfm_lab.tui import app

    monkeypatch.setattr(
        app, "expand_local_user", partial(expand_local_user, windows=True)
    )
    with pytest.raises(ValueError, match="named user's home"):
        app.LabApp("~unknown/results")
    assert not windows_homes.exists()


def test_tui_workspace_edit_rejects_raw_unknown_home(
    windows_homes, tmp_path, monkeypatch
):
    pytest.importorskip("textual")
    from textual.widgets import Input, Label

    from tsfm_lab.tui import app

    monkeypatch.setattr(
        app, "expand_local_user", partial(expand_local_user, windows=True)
    )

    async def exercise():
        initial = tmp_path / "results"
        instance = app.LabApp(initial)
        async with instance.run_test() as pilot:
            await pilot.pause()
            instance.screen.query_one("#workspace-path", Input).value = (
                "~unknown/results"
            )
            instance.action_new_experiment()
            assert isinstance(instance.screen, app.HomeScreen)
            assert instance.workspace == initial
            assert "named user's home" in str(
                instance.screen.query_one("#home-status", Label).render()
            )
        assert not initial.exists()

    asyncio.run(exercise())
    assert not windows_homes.exists()
