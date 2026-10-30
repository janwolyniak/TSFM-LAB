"""Offline first-launch, preference precedence, and terminal handoff checks."""

import csv
import io
import json
import shutil
from dataclasses import replace
from importlib.resources import files
from pathlib import Path

import pytest
from rich.console import Console

from tsfm_lab.application import run_config
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.preferences import Preferences, PreferenceStore
from tsfm_lab.repl.onboarding import configure_preferences
from tsfm_lab.repl.session import TerminalSession
from tsfm_lab.repl.wizard import _Guide


class Editor:
    def __init__(self, lines=(), answers=None):
        self.lines = iter(lines)
        self.answers = {key: iter(value) for key, value in (answers or {}).items()}
        self.questions = []

    def read(self):
        return next(self.lines, "/exit")

    def ask(self, label, default="", choices=()):
        self.questions.append((label, default, choices))
        value = next(self.answers.get(label, iter(())), default)
        if isinstance(value, BaseException):
            raise value
        return value


def synchronous(config, console):
    result = run_config(config)
    return {"type": "completed", "run_id": result["run_id"], "result": result}


def session(store, editor, **kwargs):
    output = io.StringIO()
    app = TerminalSession(
        preference_store=store,
        editor=editor,
        console=Console(file=output, width=120, color_system=None),
        runner=synchronous,
        **kwargs,
    )
    return app, output


@pytest.fixture(autouse=True)
def isolated_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr("tsfm_lab.repl.onboarding.runtime_diagnostics", lambda: [])


@pytest.fixture
def store(tmp_path):
    return PreferenceStore(tmp_path / "settings")


def test_first_launch_skip_is_saved_once_and_can_be_completed_later(store):
    editor = Editor(["/exit"], {"First-launch setup": ["skip"]})
    app, output = session(store, editor)
    app.run()
    assert store.load().onboarding_status == "skipped"
    assert "Setup skipped" in output.getvalue()
    assert not app.workspace.exists()
    later = Editor(["/setup", "/exit"])
    app, _ = session(store, later)
    app.run()
    assert store.load().onboarding_status == "completed"
    assert "First-launch setup" not in [item[0] for item in later.questions]
    assert "Workspace directory" in [item[0] for item in later.questions]
    assert not app.workspace.exists()


@pytest.mark.parametrize("cancellation", ["/cancel", KeyboardInterrupt(), EOFError()])
def test_cancelled_edit_preserves_saved_file_and_does_not_create_chosen_dirs(
    store, tmp_path, cancellation
):
    original = replace(store.load(), onboarding_status="completed")
    store.save(original)
    before = store.path.read_bytes()
    editor = Editor(
        answers={
            "Workspace directory": [str(tmp_path / "new work")],
            "Results directory": [cancellation],
        }
    )
    assert configure_preferences(editor.ask, lambda _: None, original, store) is None
    assert store.path.read_bytes() == before
    assert not (tmp_path / "new work").exists()


def test_back_navigation_retains_compatible_choices_and_saves_atomically(
    store, tmp_path
):
    original = store.load()
    editor = Editor(
        answers={
            "Workspace directory": [str(tmp_path / "Łódź work")],
            "Results directory": ["saved forecasts"],
            "Model cache directory": ["/back", "weights"],
            "Default category": ["temperature"],
            "Default export format": ["json"],
        }
    )
    saved = configure_preferences(editor.ask, lambda _: None, original, store)
    assert saved == store.load()
    assert saved.results_directory == tmp_path / "Łódź work" / "saved forecasts"
    assert saved.cache_directory == tmp_path / "Łódź work" / "weights"
    assert saved.category == "temperature"
    assert not saved.workspace.exists()


def test_settings_view_then_edit_changes_future_sessions_without_moving_records(
    store, tmp_path
):
    old = replace(store.load(), onboarding_status="completed")
    store.save(old)
    old.results_directory.mkdir()
    marker = old.results_directory / "retained.txt"
    marker.write_text("existing user results")
    editor = Editor(
        ["/settings", "/exit"],
        {
            "Edit preferences?": ["yes"],
            "Results directory": [str(tmp_path / "new results")],
            "Default export format": ["json"],
        },
    )
    app, output = session(store, editor)
    app.run()
    assert "Preferences" in output.getvalue()
    assert app.workspace == tmp_path / "new results"
    assert marker.read_text() == "existing user results"
    assert not app.workspace.exists()
    restarted, _ = session(store, Editor())
    assert restarted.workspace == app.workspace
    assert restarted.preferences.export_format == "json"


def test_explicit_workspace_takes_priority_without_changing_saved_default(
    store, tmp_path
):
    store.save(replace(store.load(), onboarding_status="completed"))
    saved = store.load().results_directory
    override = tmp_path / "explicit output"
    app, output = session(store, Editor(["/setup", "/exit"]), workspace=override)
    app.run()
    assert app.workspace == override
    assert store.load().results_directory == saved
    assert "explicit --workspace" in output.getvalue()


def test_yaml_output_overrides_preferences_and_does_not_rewrite_them(
    store, config_factory, tmp_path
):
    store.save(replace(store.load(), onboarding_status="completed"))
    before = store.path.read_bytes()
    app, _ = session(store, Editor([f'/run "{config_factory()}"', "/exit"]))
    app.run()
    assert app.workspace == tmp_path / "workspace"
    assert app.last_run
    assert store.path.read_bytes() == before
    assert not store.load().results_directory.exists()


def test_defaults_reach_guided_category_cache_and_relative_dataset_validation(
    store, tmp_path
):
    root = tmp_path / "selected work"
    root.mkdir()
    source = Path(str(files("tsfm_lab.resources").joinpath("seasonal.csv")))
    shutil.copy(source, root / "custom.csv")
    prefs = Preferences(
        root, root / "outputs", tmp_path / "weights", category="temperature"
    )
    guide = _Guide(
        lambda *args, **kwargs: "",
        lambda _: None,
        prefs.results_directory,
        source,
        None,
        preferences=prefs,
    )
    assert guide.fields["usage_profile"] == "temperature"
    assert guide.fields["cache_dir"] == str(prefs.cache_directory)
    guide._data_path("custom.csv")
    assert guide._resolved_data_path("custom.csv") == root / "custom.csv"
    assisted = _Guide(
        lambda *args, **kwargs: "",
        lambda _: None,
        prefs.results_directory,
        source,
        None,
        preferences=prefs,
        seed_fields={
            "usage_profile": "generic",
            "cache_dir": str(tmp_path / "explicit cache"),
        },
    )
    assert assisted.fields["usage_profile"] == "generic"
    assert assisted.fields["cache_dir"] == str(tmp_path / "explicit cache")


def test_offline_demo_exports_use_preference_with_explicit_overrides(store):
    store.save(
        replace(store.load(), onboarding_status="completed", export_format="json")
    )
    editor = Editor(
        [
            "/demo",
            "/export latest default-output",
            "/export latest explicit.csv",
            "/export latest chosen-format csv",
            "/exit",
        ],
        {"Run experiment?": ["yes"]},
    )
    app, output = session(store, editor)
    app.run()
    root = store.load().workspace
    assert len(json.loads((root / "default-output").read_text())) == 16
    for path in (root / "explicit.csv", root / "chosen-format"):
        rows = list(csv.DictReader(io.StringIO(path.read_text())))
        assert len(rows) == 16
        assert rows[0]["prediction"] == "15.0"
    result, _ = ExperimentStore(app.workspace).load_result(app.last_run)
    assert result["metrics"]["overall"]["mae"]["value"] == 2.5
    assert "Completed" in output.getvalue()


def test_optional_assistant_setup_reuses_connection_menu_and_keeps_secrets_out(
    store, monkeypatch
):
    seen = []
    secret = "fake-secret-never-in-preferences"
    editor = Editor(["/exit"], {"Set up an optional assistant?": ["yes"]})
    editor.ask_secret = lambda label: secret
    manager = object()

    class Menu:
        def __init__(self, received_editor, emit, *, manager):
            seen.append((received_editor, manager))

        def run(self):
            assert editor.ask_secret("API key") == secret

    monkeypatch.setattr("tsfm_lab.connections.ui.ConnectionMenu", Menu)
    app, output = session(store, editor, connection_manager=manager)
    app.run()
    assert seen == [(editor, manager)]
    assert store.load().onboarding_status == "completed"
    assert secret not in store.path.read_text()
    assert secret not in output.getvalue()


def test_skip_remains_available_when_unused_default_cache_is_unusable(store, tmp_path):
    (tmp_path / "cache").write_text("preexisting file blocks this cache directory")
    app, output = session(store, Editor(["/exit"], {"First-launch setup": ["skip"]}))
    app.run()
    assert store.load().onboarding_status == "skipped"
    assert "Setup skipped" in output.getvalue()
    assert (tmp_path / "cache").is_file()


def test_unwritable_output_is_rejected_before_starting_runner(
    store, config_factory, tmp_path
):
    store.save(replace(store.load(), onboarding_status="completed"))
    config_path = config_factory()
    (tmp_path / "workspace").write_text("preexisting file")
    app, output = session(store, Editor([f'/run "{config_path}"', "/exit"]))
    calls = []
    app.runner = lambda *args: calls.append(args)
    app.run()
    assert calls == []
    assert "not writable" in output.getvalue()
    assert (tmp_path / "workspace").read_text() == "preexisting file"


def test_first_launch_can_finish_with_offline_baseline_demo(store):
    editor = Editor(
        ["/exit"],
        {
            "Try the offline baseline demo?": ["yes"],
            "Run experiment?": ["yes"],
        },
    )
    app, output = session(store, editor)
    app.run()
    assert store.load().onboarding_status == "completed"
    result, predictions = ExperimentStore(app.workspace).load_result(app.last_run)
    assert len(predictions) == 16
    assert result["configuration"]["model"]["source"] == "builtin"
    assert app.connection_manager is None
    assert "Completed" in output.getvalue()
