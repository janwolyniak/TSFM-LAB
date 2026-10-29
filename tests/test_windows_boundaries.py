"""Simulated Windows policies and portable orchestration never install runtimes."""

import json
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_future_store import make_record
from test_runtime_diagnostics import VERSIONS

import tsfm_lab.execution as execution_module
from tsfm_lab import platform_runtime, runtime_diagnostics
from tsfm_lab.connections import state
from tsfm_lab.connections.contracts import ConnectionError
from tsfm_lab.execution import ManagedRun
from tsfm_lab.future_validation import validate_configuration
from tsfm_lab.guided import GuidedRunPlan, execute_guided
from tsfm_lab.guided_execution import ManagedGuidedRun, PreparationRecord
from tsfm_lab.models import tirex2
from tsfm_lab.models.registry import adapter_profile


def test_redirected_windows_cli_output_is_utf8_without_changing_console_streams(
    monkeypatch,
):
    from tsfm_lab import __main__ as entry

    calls = []
    pipe = SimpleNamespace(
        isatty=lambda: False, reconfigure=lambda **kw: calls.append(kw)
    )
    console = SimpleNamespace(
        isatty=lambda: True,
        reconfigure=lambda **kw: pytest.fail("console reconfiguration"),
    )
    monkeypatch.setattr(
        entry, "sys", SimpleNamespace(platform="win32", stdout=pipe, stderr=console)
    )
    entry._configure_windows_output()
    assert calls == [{"encoding": "utf-8"}]


def test_forced_stop_escalates_to_kill_when_a_process_resists_termination(
    config_factory, monkeypatch
):
    from tsfm_lab.config import load_config

    worker = ManagedRun(load_config(config_factory()), grace_seconds=0)
    calls = []
    worker._started = True
    worker._requested_at = 0.0
    worker._process = SimpleNamespace(
        events=SimpleNamespace(error=False),
        control=SimpleNamespace(error=False),
        is_alive=lambda: True,
        terminate=lambda: calls.append("terminate"),
        kill=lambda: calls.append("kill"),
    )
    clock = iter([1.0, 1.3])
    monkeypatch.setattr(execution_module, "monotonic", lambda: next(clock))
    assert worker.poll() == [] and worker.poll() == []
    assert calls == ["terminate", "kill"]
    worker._closed = True


def test_windows_byte_lock_retries_without_touching_the_locked_byte(
    tmp_path, monkeypatch
):
    calls = []

    def locking(descriptor, mode, count):
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 0 and count == 1
        calls.append(mode)
        if mode == 1 and len(calls) < 3:
            raise OSError("contended")

    fake_os = SimpleNamespace(
        **{
            **vars(os),
            "name": "nt",
            "write": lambda *a: pytest.fail("write before locking"),
        }
    )
    monkeypatch.setattr(state, "os", fake_os)
    monkeypatch.setitem(
        sys.modules, "msvcrt", SimpleNamespace(locking=locking, LK_NBLCK=1, LK_UNLCK=2)
    )
    monkeypatch.setattr(state, "sleep", lambda seconds: None)
    store = state.StateStore(tmp_path / "state")
    with store.locked() as value:
        assert value["generation"] == 0
    assert calls == [1, 1, 1, 2]
    assert (store.directory / ".lock").read_bytes() == b""


def test_windows_lock_contention_timeout_closes_descriptor_and_releases_thread_lock(
    tmp_path, monkeypatch
):
    clock = iter([0, 0, 6])
    descriptors = []

    def locking(descriptor, mode, count):
        descriptors.append(descriptor)
        raise OSError("contended")

    monkeypatch.setattr(state, "os", SimpleNamespace(**{**vars(os), "name": "nt"}))
    monkeypatch.setitem(
        sys.modules, "msvcrt", SimpleNamespace(locking=locking, LK_NBLCK=1, LK_UNLCK=2)
    )
    monkeypatch.setattr(state, "monotonic", lambda: next(clock))
    monkeypatch.setattr(state, "sleep", lambda seconds: None)
    with pytest.raises(ConnectionError) as error:
        with state.StateStore(tmp_path / "state").locked():
            pytest.fail("lock acquired")
    assert error.value.code == "state"
    with pytest.raises(OSError):
        os.fstat(descriptors[-1])
    assert state._THREAD_LOCK.acquire(blocking=False)
    state._THREAD_LOCK.release()


@pytest.mark.parametrize(
    "found,expected",
    [("3.4.0.post21", "available"), ("3.5.0.post1", "incompatible"), (None, "missing")],
)
def test_windows_tirex_candidate_metadata_is_execution_only(
    monkeypatch, found, expected
):
    before = adapter_profile("tirex_2").runtime_requirements
    monkeypatch.setattr(platform_runtime, "sys", SimpleNamespace(platform="win32"))

    def version(name):
        if name == "triton-windows":
            if found is None:
                raise runtime_diagnostics.PackageNotFoundError(name)
            return found
        return VERSIONS[name]

    monkeypatch.setattr(runtime_diagnostics, "version", version)
    entry = next(
        item
        for item in runtime_diagnostics.runtime_diagnostics()
        if item["adapter_id"] == "tirex_2"
    )
    assert entry["status"] == expected
    assert adapter_profile("tirex_2").runtime_requirements == before
    assert "triton-windows" not in dict(before)


def test_windows_tirex_runtime_refuses_wrong_candidate_before_optional_imports(
    monkeypatch,
):
    monkeypatch.setattr(platform_runtime, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(
        tirex2, "version", {**VERSIONS, "triton-windows": "3.5.0.post1"}.__getitem__
    )
    monkeypatch.setattr(
        tirex2,
        "importlib",
        SimpleNamespace(import_module=lambda _: pytest.fail("optional runtime import")),
    )
    with pytest.raises(ImportError, match="candidate pin"):
        tirex2._runtime()


def test_record_only_configuration_never_reaches_preparation_or_worker_io(
    tmp_path, monkeypatch
):
    config = validate_configuration(make_record(tmp_path).result["configuration"])
    # Construct a deliberately stale/internal plan to exercise execution's own
    # boundary, independently of the normal constructor's earlier refusal.
    plan = object.__new__(GuidedRunPlan)
    object.__setattr__(plan, "_config", config)
    monkeypatch.setattr(Path, "resolve", lambda *a, **k: pytest.fail("path resolution"))
    monkeypatch.setattr(
        "tsfm_lab.guided.load_dataset", lambda *a: pytest.fail("dataset access")
    )
    monkeypatch.setattr(
        "tsfm_lab.guided_execution.PreparationRecord",
        lambda *a, **k: pytest.fail("preparation allocation"),
    )
    for action in (
        lambda: GuidedRunPlan(config, None, "0" * 64),
        lambda: execute_guided(plan),
        lambda: ManagedRun(config),
        lambda: ManagedGuidedRun(plan),
        lambda: PreparationRecord(config, plan),
    ):
        with pytest.raises(ValueError, match="read-only"):
            action()


def test_ci_helper_selects_native_paths_and_verifies_the_exact_wheel(
    tmp_path, monkeypatch
):
    script = (
        Path(__file__).resolve().parents[1] / "scripts/check_installed_distribution.py"
    )
    definitions = runpy.run_path(str(script), run_name="ci_test")
    assert (
        definitions["environment_python"](tmp_path, windows=True)
        == tmp_path / "Scripts/python.exe"
    )
    assert (
        definitions["environment_python"](tmp_path, windows=False)
        == tmp_path / "bin/python"
    )
    function = definitions["check_installation"]
    calls = []
    wheel = tmp_path / "candidate.whl"

    def run(arguments, **options):
        calls.append((arguments, options))
        if str(arguments[1]).endswith("build_distribution.py"):
            return SimpleNamespace(stdout=json.dumps({"wheel": {"path": str(wheel)}}))
        return SimpleNamespace(stdout='{"status":"passed"}')

    monkeypatch.setitem(function.__globals__, "_run", run)
    monkeypatch.setattr(
        function.__globals__["venv"],
        "EnvBuilder",
        lambda **kw: SimpleNamespace(create=lambda path: None),
    )
    result = function(tmp_path / "qualification")
    assert result["verification"]["status"] == "passed"
    final, options = calls[-1]
    assert final[-2:] == ["--wheel", wheel]
    assert "-I" in final
    assert options["environment"]["PATH"].split(os.pathsep)[0] == str(
        Path(final[0]).parent
    )
    assert "PYTHONPATH" not in options["environment"]
