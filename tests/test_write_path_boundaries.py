"""Raw named-home spellings are checked before export or model-cache writes."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from test_checkpoint_preparation import FakeHub as PreparationHub
from test_checkpoints import FakeHub as PullHub

from tsfm_lab import exports
from tsfm_lab.application import run_from_config
from tsfm_lab.finance import reports
from tsfm_lab.local_paths import expand_local_user
from tsfm_lab.models import checkpoints as cp
from tsfm_lab.models import preparation as prep

ALIAS = "~definitely-missing-tsfm-write-user"
REPOSITORY = "amazon/chronos-bolt-tiny"
CACHE_ENVIRONMENT = (
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_HOME",
    "XDG_CACHE_HOME",
)


@pytest.fixture
def missing_windows_home(tmp_path, monkeypatch):
    invented = tmp_path / "invented-user-home"
    original = Path.expanduser

    def windows_expand(path):
        value = str(path)
        if value == ALIAS:
            return invented
        if value.startswith(ALIAS + "/"):
            return invented / value[len(ALIAS) + 1 :]
        return original(path)

    def guarded(value):
        return expand_local_user(value, windows=True)

    monkeypatch.setattr(Path, "expanduser", windows_expand)
    for module in (exports, reports, cp):
        monkeypatch.setattr(module, "expand_local_user", guarded)
    for name in CACHE_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    return invented


@pytest.mark.parametrize("kind", ["predictions", "finance"])
def test_unknown_named_home_is_rejected_before_any_export_io(
    missing_windows_home, tmp_path, monkeypatch, kind
):
    def forbidden(*args, **kwargs):
        pytest.fail("Export source read before rejecting an unavailable named home")

    monkeypatch.setattr(exports, "read_completed", forbidden)
    if kind == "predictions":
        with pytest.raises(exports.ExportError, match="named user's home"):
            exports.export_predictions(
                "unused", tmp_path / "workspace", ALIAS + "/out.csv"
            )
    else:
        with pytest.raises(reports.FinanceReportError, match="named user's home"):
            reports._publish_bytes(b"synthetic output", ALIAS + "/report.json")
    assert not missing_windows_home.exists()
    assert not (tmp_path / "workspace").exists()


@pytest.mark.parametrize("operation", ["pull", "plan"])
def test_unknown_cache_home_is_rejected_before_hub_access(
    missing_windows_home, monkeypatch, operation
):
    monkeypatch.setattr(
        cp, "_hub", lambda: pytest.fail("Hub access before path validation")
    )
    with pytest.raises(cp.CheckpointError, match="named user's home"):
        if operation == "pull":
            cp.pull_checkpoint(REPOSITORY, cache_dir=ALIAS)
        else:
            prep.plan_checkpoint(REPOSITORY, "main", cache_dir=ALIAS)
    assert not missing_windows_home.exists()


@pytest.mark.parametrize("name", CACHE_ENVIRONMENT)
@pytest.mark.parametrize("operation", ["pull", "plan"])
def test_active_raw_cache_environment_is_checked_before_sdk_expansion(
    missing_windows_home, monkeypatch, name, operation
):
    monkeypatch.setenv(name, ALIAS)
    monkeypatch.setattr(
        cp, "_hub", lambda: pytest.fail("SDK constants before raw validation")
    )
    with pytest.raises(cp.CheckpointError, match="named user's home"):
        if operation == "pull":
            cp.pull_checkpoint(REPOSITORY)
        else:
            prep.plan_checkpoint(REPOSITORY, "main")
    assert not missing_windows_home.exists()


@pytest.mark.parametrize("selected_value", ["selected cache", ""])
def test_unused_cache_environment_does_not_override_selected_sdk_root(
    missing_windows_home, tmp_path, monkeypatch, selected_value
):
    selected = tmp_path / "selected cache"
    monkeypatch.setenv("HF_HUB_CACHE", selected_value)
    monkeypatch.setenv("HF_HOME", ALIAS)
    monkeypatch.setattr(cp, "_hub", lambda: SimpleNamespace(cache_dir=selected))
    assert cp._write_cache_directory(None) == selected
    assert not selected.exists() and not missing_windows_home.exists()


def test_explicit_cache_overrides_unavailable_environment_home(
    missing_windows_home, tmp_path, monkeypatch
):
    selected = tmp_path / "explicit cache"
    monkeypatch.setenv("HF_HUB_CACHE", ALIAS)
    monkeypatch.setattr(
        cp, "_hub", lambda: pytest.fail("SDK defaults for explicit cache")
    )
    assert cp._write_cache_directory(selected) == selected
    assert not selected.exists() and not missing_windows_home.exists()


@pytest.mark.parametrize("kind", ["predictions", "finance"])
def test_ordinary_relative_nested_exports_keep_their_current_directory_base(
    config_factory, tmp_path, monkeypatch, kind
):
    run = run_from_config(config_factory()) if kind == "predictions" else None
    monkeypatch.chdir(tmp_path)
    target = Path("nested exports") / ("out.json" if run else "report.json")
    if run:
        result = exports.export_predictions(
            run["run_id"], tmp_path / "workspace", target
        )
    else:
        result = reports._publish_bytes(b"synthetic output", target)
    assert result["path"] == str(tmp_path / target)
    assert target.is_file()


@pytest.mark.parametrize("operation", ["pull", "plan"])
def test_ordinary_relative_nested_cache_uses_the_current_directory(
    tmp_path, monkeypatch, operation
):
    selected = tmp_path / "nested cache"
    monkeypatch.chdir(tmp_path)
    hub = PullHub(selected) if operation == "pull" else PreparationHub(selected)
    monkeypatch.setattr(cp, "_hub", hub.client if operation == "pull" else lambda: hub)
    if operation == "pull":
        result = cp.pull_checkpoint(REPOSITORY, cache_dir=Path("nested cache"))
        assert result.path == hub.path()
    else:
        result = prep.plan_checkpoint(
            REPOSITORY, "main", cache_dir=Path("nested cache")
        )
        assert result.cache_dir == selected
    assert selected.is_dir()
