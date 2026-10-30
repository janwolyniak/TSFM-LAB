"""Actual Windows console Ctrl+C, isolated from pytest's own console."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Native Windows console control events")
def test_private_console_ctrl_c_cancels_and_reaps_managed_worker(
    config_factory, tmp_path
):
    config_path = config_factory()
    helper = Path(__file__).with_name("windows_console_helpers.py")
    source = Path(__file__).resolve().parents[1] / "src"
    report = tmp_path / "console-result.json"
    driver = subprocess.Popen(
        [
            sys.executable,
            str(helper),
            "driver",
            str(source),
            str(config_path),
            str(report),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    try:
        deadline = time.monotonic() + 10
        while (
            not (tmp_path / "console-ready").exists()
            and driver.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.025)
        if not (tmp_path / "console-ready").exists():
            pytest.fail("Private console driver did not become ready")
        sent = subprocess.run(
            [sys.executable, str(helper), "signal", str(driver.pid)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.DETACHED_PROCESS,
        )
        if sent.returncode == 77:
            pytest.skip(
                f"Runner lacks an attachable private Windows console: {sent.stdout.strip()}"
            )
        assert sent.returncode == 0, sent.stderr
        stdout, stderr = driver.communicate(timeout=10)
        assert driver.returncode == 0, (stdout, stderr)
        assert json.loads(report.read_text(encoding="utf-8")) == {
            "event": "cancelled",
            "status": "cancelled",
            "worker_alive": False,
            "exitcode": 0,
            "log_exists": False,
        }
    finally:
        if driver.poll() is None:
            (tmp_path / "console-stop").write_text("stop")
            try:
                driver.wait(timeout=5)
            except subprocess.TimeoutExpired:
                subprocess.run(
                    ["taskkill", "/PID", str(driver.pid), "/T", "/F"],
                    check=False,
                    capture_output=True,
                    timeout=5,
                )
        driver.communicate(timeout=5)
