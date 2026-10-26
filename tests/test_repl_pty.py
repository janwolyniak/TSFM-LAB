"""Real terminal controls and scrollback; each child has its own process group."""

import os
import select
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
import yaml

if sys.platform == "win32":
    pytest.skip("POSIX pseudo-terminal verification", allow_module_level=True)

import fcntl  # noqa: E402
import pty  # noqa: E402
import termios  # noqa: E402

from tsfm_lab.experiments import ExperimentStore  # noqa: E402


def _control_tty():
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)


class Terminal:
    def __init__(self, directory, size=(80, 24)):
        self.master, slave = pty.openpty()
        fcntl.ioctl(
            slave, termios.TIOCSWINSZ, struct.pack("HHHH", size[1], size[0], 0, 0)
        )
        source = Path(__file__).resolve().parents[1] / "src"
        environment = dict(
            os.environ,
            PYTHONPATH=str(source),
            NO_COLOR="1",
            TERM="xterm-256color",
            PROMPT_TOOLKIT_NO_CPR="1",
        )
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tsfm_lab",
                "ui",
                "--workspace",
                str(directory / "workspace"),
            ],
            cwd=directory,
            env=environment,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            start_new_session=True,
            preexec_fn=_control_tty,
        )
        os.close(slave)
        self.data = b""

    def send(self, value):
        os.write(self.master, value if isinstance(value, bytes) else value.encode())

    def expect(self, value, *, after=0, timeout=15):
        value = value.encode() if isinstance(value, str) else value
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if value in self.data[after:]:
                return
            if select.select([self.master], [], [], 0.05)[0]:
                try:
                    chunk = os.read(self.master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                self.data += chunk
            if self.process.poll() is not None:
                break
        raise AssertionError(f"Missing {value!r}: {self.data[-4000:]!r}")

    def command(self, value, response):
        start = len(self.data)
        self.send(value + "\r")
        self.expect(response, after=start)

    def close(self):
        try:
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=5)
        finally:
            os.close(self.master)


@pytest.fixture(params=[(80, 24), (120, 36)])
def terminal(tmp_path, request):
    value = Terminal(tmp_path, size=request.param)
    try:
        value.expect("›")
        yield value
    finally:
        value.close()


def test_scroll_history_completion_demo_and_normal_exit(terminal, tmp_path):
    start = len(terminal.data)
    terminal.send("/hist\t")
    terminal.expect("/history", after=start)
    terminal.command("", "No saved experiments")
    start = len(terminal.data)
    terminal.send(b"\x1b[A\r")
    terminal.expect("No saved experiments", after=start)
    terminal.command("/new", "Usage profile")
    start = len(terminal.data)
    terminal.send(b"\x03")
    terminal.expect("Command cancelled", after=start)
    assert not (tmp_path / "workspace").exists()
    terminal.command("/demo", "Run experiment?")
    terminal.command("y", "Completed ·")
    terminal.expect("2.5")
    terminal.expect("›", after=terminal.data.rfind(b"Completed"))
    terminal.command("/exit", "Session closed")
    assert terminal.process.wait(timeout=5) == 0
    records = ExperimentStore(tmp_path / "workspace").list_runs()
    assert len(records) == 1 and records[0]["status"] == "completed"
    for forbidden in (b"\x1b[?1049h", b"\x1b[?1047h", b"\x1b[?47h"):
        assert forbidden not in terminal.data
    assert b"No saved experiments" in terminal.data
    assert b"Traceback" not in terminal.data


def test_terminal_ctrl_c_cancels_worker_and_returns_to_command_prompt(
    terminal, tmp_path
):
    data = tmp_path / "many.csv"
    count = 10000
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=count, freq="h", tz="UTC"),
            "target": range(count),
        }
    ).to_csv(data, index=False)
    mapping = {
        "schema_version": 1,
        "name": "cancel-in-terminal",
        "model": {"source": "builtin", "adapter_id": "last_value"},
        "dataset": {
            "path": str(data),
            "timestamp_column": "timestamp",
            "target_column": "target",
            "frequency": "1h",
            "units": "synthetic_units",
        },
        "evaluation": {
            "context_length": 8,
            "horizon": 4,
            "stride": 1,
            "seasonal_period": 4,
            "evaluation_start": "2026-01-01T08:00:00Z",
            "evaluation_end": (
                pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=count)
            ).isoformat(),
        },
        "output": {"directory": str(tmp_path / "workspace")},
    }
    config = tmp_path / "cancel.yaml"
    config.write_text(yaml.safe_dump(mapping))
    terminal.command(f"/run {config}", "windows · ctrl+c")
    start = len(terminal.data)
    terminal.send(b"\x03")
    terminal.expect("Cancellation requested", after=start)
    terminal.expect("Experiment cancelled", after=start)
    terminal.expect("›", after=start)
    records = ExperimentStore(tmp_path / "workspace").list_runs()
    assert len(records) == 1 and records[0]["status"] == "cancelled"
    terminal.command("/status", "Local commands")
    terminal.command("/exit", "Session closed")
    assert terminal.process.wait(timeout=5) == 0
    assert b"Traceback" not in terminal.data


def test_idle_ctrl_d_exits_without_fullscreen_or_result_files(terminal, tmp_path):
    terminal.send(b"\x04")
    terminal.expect("Session closed")
    assert terminal.process.wait(timeout=5) == 0
    assert not (tmp_path / "workspace").exists()
