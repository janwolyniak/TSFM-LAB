"""Real interprocess StateStore locking: MSVCRT on Windows, flock on POSIX."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _wait_for(path, *, process=None, root=None, seconds=15):
    deadline = time.monotonic() + seconds
    while not path.exists():
        if root is not None and (root / "stop").exists():
            raise RuntimeError("Disposable lock probe stopped")
        if process is not None and process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1)
            raise AssertionError(("Lock probe exited before signaling", stdout, stderr))
        if time.monotonic() >= deadline:
            raise AssertionError(f"Lock probe timed out waiting for {path.name}")
        time.sleep(0.01)


def _locked_update(store, root, role, *, hold=False):
    """A second exclusive marker independently detects overlapping critical work."""
    with store.locked() as value:
        assert (store.directory / ".lock").stat().st_size == 0
        marker = root / "critical-section"
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        try:
            (root / f"{role}-acquired").touch()
            if hold:
                _wait_for(root / "release", root=root)
            generation = value["generation"]
            assert generation == (0 if hold else 1)
            store.write({**value, "generation": generation + 1})
            return generation
        finally:
            marker.unlink()


def _probe(role, directory, expect_timeout):
    # The child imports the actual source API, without pytest's monkeypatches or
    # inherited import-path configuration. It has no providers or credentials.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from tsfm_lab.connections.contracts import ConnectionError
    from tsfm_lab.connections.state import StateStore

    root = Path(directory)
    store = StateStore(root / "state")
    report = {
        "pid": os.getpid(),
        "backend": "msvcrt" if os.name == "nt" else "flock",
        "platform": sys.platform,
    }
    if role == "holder":
        report["generation_read"] = _locked_update(store, root, role, hold=True)
    else:
        (root / "contender-ready").touch()
        if expect_timeout:
            started = time.monotonic()
            try:
                with store.locked():
                    raise AssertionError("Contender acquired a held lock")
            except ConnectionError as exc:
                # An initialization write against a locked Windows byte would
                # fail immediately with storage, rather than retry to state.
                assert exc.code == "state"
                elapsed = time.monotonic() - started
                assert elapsed >= 4.5
                report["timeout_code"] = exc.code
                report["timeout_seconds"] = elapsed
            (root / "contender-timed-out").touch()
            _wait_for(root / "retry", root=root)
        report["generation_read"] = _locked_update(store, root, role)
    print(json.dumps(report), flush=True)


def _spawn(root, role, expect_timeout):
    environment = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONNOUSERSITE": "1",
    }
    return subprocess.Popen(
        [
            sys.executable,
            "-I",
            str(Path(__file__).resolve()),
            "--lock-probe",
            role,
            str(root),
            "timeout" if expect_timeout else "retry",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        cwd=root,
        env=environment,
    )


def _finished(process):
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, (process.returncode, stdout, stderr)
    return json.loads(stdout)


def _cleanup(processes, root):
    # Every process and path belongs to this test. No child launches descendants.
    for name in ("stop", "release", "retry"):
        (root / name).touch()
    for process in processes:
        try:
            process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=2)


def _exercise_native_lock(tmp_path, expect_timeout, record_property):
    from tsfm_lab.connections.state import StateStore

    root = (tmp_path / "native lock Łódź").resolve()
    root.mkdir(mode=0o700)
    store = StateStore(root / "state")
    store.write(store.read())
    original = store.path.read_bytes()
    processes = []
    try:
        holder = _spawn(root, "holder", expect_timeout)
        processes.append(holder)
        _wait_for(root / "holder-acquired", process=holder)
        contender = _spawn(root, "contender", expect_timeout)
        processes.append(contender)
        _wait_for(root / "contender-ready", process=contender)

        # Keep both live long enough to expose an unprotected second acquisition.
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            assert holder.poll() is None
            assert contender.poll() is None
            assert not (root / "contender-acquired").exists()
            assert store.path.read_bytes() == original
            time.sleep(0.01)
        if expect_timeout:
            _wait_for(root / "contender-timed-out", process=contender, seconds=10)
            assert not (root / "contender-acquired").exists()
            assert holder.poll() is None

        (root / "release").touch()
        holder_report = _finished(holder)
        if expect_timeout:
            (root / "retry").touch()
        contender_report = _finished(contender)
        assert {holder_report["pid"], contender_report["pid"]} == {
            holder.pid,
            contender.pid,
        }
        assert holder.pid != contender.pid and os.getpid() not in {
            holder.pid,
            contender.pid,
        }
        assert holder_report["generation_read"] == 0
        assert contender_report["generation_read"] == 1
        for report in (holder_report, contender_report):
            assert report["platform"] == sys.platform
            assert report["backend"] == ("msvcrt" if os.name == "nt" else "flock")
        if expect_timeout:
            assert contender_report["timeout_code"] == "state"
        assert (root / "contender-acquired").is_file()
        assert not (root / "critical-section").exists()
        assert (store.directory / ".lock").read_bytes() == b""
        with store.locked() as value:
            assert value["generation"] == 2
        record_property("lock_backend", contender_report["backend"])
        record_property("native_platform", sys.platform)
        record_property("independent_child_pids", [holder.pid, contender.pid])
    finally:
        _cleanup(processes, root)


def test_native_interprocess_state_lock_excludes_and_retries(tmp_path, record_property):
    _exercise_native_lock(tmp_path, False, record_property)


def test_native_interprocess_state_lock_times_out_then_recovers(
    tmp_path, record_property
):
    _exercise_native_lock(tmp_path, True, record_property)


if __name__ == "__main__":
    assert len(sys.argv) == 5 and sys.argv[1] == "--lock-probe"
    _probe(sys.argv[2], sys.argv[3], sys.argv[4] == "timeout")
