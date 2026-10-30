"""Transactional, optional first-launch questions over nonsecret preferences."""

from dataclasses import replace
from pathlib import Path

from tsfm_lab.config import ConfigError
from tsfm_lab.preferences import validate_directory
from tsfm_lab.repl.wizard import _Back, _Cancel, _question
from tsfm_lab.runtime_diagnostics import runtime_diagnostics
from tsfm_lab.workflows import PROFILE_NAMES


def show_preferences(emit, preferences, path):
    emit(f"Preferences  {path}")
    emit(f"Workspace    {preferences.workspace}")
    emit(f"Results      {preferences.results_directory}")
    emit(f"Model cache  {preferences.cache_directory}")
    emit(f"Category     {preferences.category}")
    emit(f"Export       {preferences.export_format}")


def configure_preferences(ask, emit, preferences, store, *, first_launch=False):
    """Save only reviewed choices; back/cancel never save a partial form."""
    try:
        if first_launch:
            emit("First launch · optional setup · manual forecasting needs no API key.")
            answer = _question(
                ask, emit, "First-launch setup", "setup", ("setup", "skip")
            )
            if answer == "skip":
                skipped = store.skip(preferences)
                emit("Setup skipped. Use /setup whenever you are ready.")
                return skipped
        emit("Preferences · /back revisits a choice · /cancel discards edits.")
        emit(
            "Directory changes apply to subsequent work; existing files stay in place."
        )
        emit("Runtime diagnostics (installed package metadata only):")
        for item in runtime_diagnostics():
            emit(f"  {item['label']}: {item['status']} · {item['detail']}")
        return _edit(ask, emit, preferences, store)
    except (_Cancel, _Back, EOFError, KeyboardInterrupt):
        emit("Setup cancelled. Preferences are unchanged; use /setup later.")
        return None


def _edit(ask, emit, preferences, store):
    draft = preferences
    phase = 0
    questions = (
        ("workspace", "Workspace directory", ()),
        ("results_directory", "Results directory", ()),
        ("cache_directory", "Model cache directory", ()),
        ("category", "Default category", PROFILE_NAMES),
        ("export_format", "Default export format", ("csv", "json", "parquet")),
    )
    while True:
        try:
            if phase == len(questions):
                show_preferences(emit, draft, store.path)
                if (
                    _question(ask, emit, "Save preferences?", "yes", ("yes", "no"))
                    == "no"
                ):
                    return None
                saved = replace(draft, onboarding_status="completed")
                store.save(saved)
                emit("Preferences saved.")
                return saved
            field, label, choices = questions[phase]
            directory = field.endswith("directory") or field == "workspace"

            def resolve(value, base=draft.workspace):
                if "://" in value or any(ord(char) < 32 for char in value):
                    raise ConfigError(
                        "Choose a local directory without control characters"
                    )
                path = Path(value).expanduser()
                if not path.is_absolute():
                    path = base / path
                return validate_directory(path)

            value = _question(
                ask,
                emit,
                label,
                str(getattr(draft, field)),
                choices,
                validate=resolve if directory else None,
            )
            value = resolve(value) if directory else value
            changes = {field: value}
            if (
                field == "workspace"
                and draft.results_directory == draft.workspace / "results"
            ):
                changes["results_directory"] = value / "results"
            draft = replace(draft, **changes)
            phase += 1
        except _Back:
            phase = max(0, phase - 1)
