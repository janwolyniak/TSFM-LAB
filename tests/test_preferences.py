"""Nonsecret preference persistence and filesystem failure boundaries."""

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from tsfm_lab.config import ConfigError
from tsfm_lab.preferences import Preferences, PreferenceStore, validate_directory


@pytest.fixture
def preferences(tmp_path):
    return Preferences(
        tmp_path / "project Łódź",
        tmp_path / "saved results",
        tmp_path / "model cache",
        category="temperature",
        export_format="json",
        onboarding_status="completed",
    )


def test_missing_preferences_do_not_create_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = PreferenceStore(tmp_path / "settings")
    value = store.load()
    assert value.workspace == tmp_path
    assert value.results_directory == tmp_path / "results"
    assert value.onboarding_status == "new"
    assert not store.directory.exists()
    assert not value.results_directory.exists()


def test_roundtrip_is_absolute_and_does_not_create_selected_dirs(
    tmp_path, preferences, monkeypatch
):
    store = PreferenceStore(tmp_path / "settings")
    store.save(preferences)
    other = tmp_path / "another launch"
    other.mkdir()
    monkeypatch.chdir(other)
    assert PreferenceStore(store.directory).load() == preferences
    assert not preferences.workspace.exists()
    assert not preferences.results_directory.exists()
    assert not preferences.cache_directory.exists()
    fields = json.loads(store.path.read_text())
    assert fields["schema_version"] == 1
    assert set(fields) == set(Preferences.__dataclass_fields__) | {"schema_version"}
    with pytest.raises(FrozenInstanceError):
        preferences.category = "finance"


def test_relative_preferences_resolve_when_created(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    value = Preferences(Path("work"), Path("runs"), Path("cache"))
    assert value.workspace == tmp_path / "work"
    assert value.results_directory == tmp_path / "runs"


@pytest.mark.parametrize("status", ["new", "skipped", "completed"])
def test_onboarding_status_survives_restart(tmp_path, preferences, status):
    store = PreferenceStore(tmp_path / "settings")
    store.save(replace(preferences, onboarding_status=status))
    assert store.load().onboarding_status == status


def test_skip_persists_without_probing_unused_cache(tmp_path, preferences):
    blocker = tmp_path / "unusable-cache-parent"
    blocker.write_text("Preserve existing file")
    value = replace(preferences, cache_directory=blocker / "hub")
    store = PreferenceStore(tmp_path / "settings")
    skipped = store.skip(value)
    assert store.load() == skipped
    assert skipped.onboarding_status == "skipped"
    assert blocker.read_text() == "Preserve existing file"
    assert not preferences.workspace.exists()


@pytest.mark.parametrize(
    "change",
    [
        {"category": "unsupported"},
        {"category": []},
        {"export_format": "xlsx"},
        {"onboarding_status": "pending"},
        {"workspace": "https://example.com/work"},
        {"cache_directory": "bad\npath"},
        {"results_directory": ""},
    ],
)
def test_invalid_preference_choices_are_rejected(preferences, change):
    with pytest.raises(ConfigError):
        replace(preferences, **change)


@pytest.mark.parametrize(
    "payload",
    [
        b"not JSON with fake-secret",
        b'{"schema_version": 1, "api_key": "fake-secret"}',
        b'{"schema_version": 1, "schema_version": 1}',
        b"\xff",
        b" " * (64 * 1024 + 1),
        b"[" * 2000 + b"]" * 2000,
    ],
    ids=(
        "not_json",
        "unknown_secret",
        "duplicate_keys",
        "non_utf8",
        "oversized",
        "deep",
    ),
)
def test_invalid_file_is_preserved_without_echoing_its_content(tmp_path, payload):
    store = PreferenceStore(tmp_path)
    store.path.write_bytes(payload)
    with pytest.raises(ConfigError, match="Invalid preferences") as error:
        store.load()
    assert "fake-secret" not in str(error.value)
    assert store.path.read_bytes() == payload


@pytest.mark.parametrize(
    "change", [{"schema_version": True}, {"workspace": "relative"}]
)
def test_persisted_schema_and_absolute_paths_are_strict(tmp_path, preferences, change):
    store = PreferenceStore(tmp_path)
    store.save(preferences)
    fields = json.loads(store.path.read_text())
    fields.update(change)
    store.path.write_text(json.dumps(fields))
    with pytest.raises(ConfigError, match="Invalid preferences"):
        store.load()


def test_file_in_directory_path_fails_without_changing_existing_data(tmp_path):
    existing = tmp_path / "user-file"
    existing.write_text("preserve me")
    for selected in (existing, existing / "child"):
        with pytest.raises(ConfigError, match="not writable"):
            validate_directory(selected)
    assert existing.read_text() == "preserve me"


def test_unwritable_probe_has_actionable_error(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr("tsfm_lab.preferences.tempfile.TemporaryFile", denied)
    with pytest.raises(ConfigError, match="Choose another directory"):
        validate_directory(tmp_path / "new results")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("root_only", [False, True])
def test_missing_drive_root_fails_without_looping_or_creating_paths(
    tmp_path, monkeypatch, root_only
):
    selected = Path(tmp_path.anchor) if root_only else tmp_path / "offline drive/runs"
    probes = []

    def unavailable(path):
        probes.append(path)
        assert len(probes) <= len(selected.parents) + 1, "Root traversal repeated"
        return False

    def forbidden(*args, **kwargs):
        raise AssertionError("Unavailable roots must fail before a write probe")

    with monkeypatch.context() as patch:
        patch.setattr("tsfm_lab.preferences._local_path", lambda _: selected)
        patch.setattr(Path, "exists", unavailable)
        patch.setattr("tsfm_lab.preferences.tempfile.TemporaryFile", forbidden)
        with pytest.raises(ConfigError, match="Choose another directory"):
            validate_directory(selected)
    assert probes[-1] == Path(selected.anchor)
    assert list(tmp_path.iterdir()) == []


def test_failed_validation_preserves_previous_preferences(
    tmp_path, preferences, monkeypatch
):
    store = PreferenceStore(tmp_path / "settings")
    store.save(preferences)
    original = store.path.read_bytes()
    blocked = tmp_path / "file"
    blocked.write_text("existing")
    with pytest.raises(ConfigError):
        store.save(replace(preferences, cache_directory=blocked))
    assert store.path.read_bytes() == original


def test_atomic_replace_failure_preserves_previous_preferences(
    tmp_path, preferences, monkeypatch
):
    store = PreferenceStore(tmp_path / "settings")
    store.save(preferences)
    original = store.path.read_bytes()

    def denied(*args, **kwargs):
        raise PermissionError("replace denied")

    monkeypatch.setattr("tsfm_lab.preferences.os.replace", denied)
    with pytest.raises(ConfigError, match="previous preferences were preserved"):
        store.save(replace(preferences, category="weather"))
    assert store.path.read_bytes() == original
    assert list(store.directory.iterdir()) == [store.path]


def test_settings_directory_can_be_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("TSFM_LAB_SETTINGS_DIR", str(tmp_path / "isolated settings"))
    assert PreferenceStore().path == tmp_path / "isolated settings" / "preferences.json"


@pytest.mark.parametrize(
    "environment,expected",
    [
        ({}, ".cache/huggingface/hub"),
        ({"XDG_CACHE_HOME": "xdg"}, "xdg/huggingface/hub"),
        ({"HF_HOME": "hf-home"}, "hf-home/hub"),
        ({"HUGGINGFACE_HUB_CACHE": "legacy"}, "legacy"),
        ({"HF_HUB_CACHE": "hub", "HUGGINGFACE_HUB_CACHE": "legacy"}, "hub"),
    ],
)
def test_cache_defaults_match_hub_environment_precedence(
    tmp_path, monkeypatch, environment, expected
):
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    for name, value in environment.items():
        monkeypatch.setenv(name, str(tmp_path / value))
    assert Preferences.defaults(tmp_path).cache_directory == tmp_path / expected
