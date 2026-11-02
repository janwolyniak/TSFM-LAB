"""Private clean-environment launcher; startup pickle is trusted controller input only."""

import importlib
import json
import os
import pickle
import queue
import signal
import struct
import subprocess
import sys
import threading
from pathlib import Path

MAX_HEADER = 65536
MAX_STARTUP = 4 * 1024 * 1024
MAX_EVENT = 65536
MAX_QUEUED = 128
ENVIRONMENT = frozenset(
    {
        "PATH",
        "SystemRoot",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        # CPython 3.11 needs these for native/WOW64 Windows machine metadata.
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_ARCHITEW6432",
        "HOME",
        "USERPROFILE",
        "LOCALAPPDATA",
        "APPDATA",
        "PROGRAMDATA",
        "TEMP",
        "TMP",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "HF_HUB_DISABLE_XET",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    }
)
BOOTSTRAP = """
import json, os, struct, sys
source = os.fdopen(os.dup(sys.stdin.fileno()), "rb", buffering=0)
def exact(size):
    data = bytearray()
    while len(data) < size:
        chunk = source.read(size - len(data))
        if not chunk:
            raise RuntimeError('Incomplete private worker startup')
        data.extend(chunk)
    return bytes(data)
size = struct.unpack('!I', exact(4))[0]
if not 0 < size <= 65536:
    raise RuntimeError('Invalid private worker header')
header = json.loads(exact(size))
if set(header) != {'paths', 'log_path'} or not isinstance(header['paths'], list):
    raise RuntimeError('Invalid private worker paths')
if any(not isinstance(path, str) or not path for path in header['paths']):
    raise RuntimeError('Invalid private worker paths')
sys.path[:0] = header['paths']
from tsfm_lab.worker_entry import run_child
run_child(header, source)
"""


def worker_environment(environ=None):
    """Copy declared nonsecret settings; never mutate the parent's environment."""
    source = os.environ if environ is None else environ
    result = {key: value for key, value in source.items() if key in ENVIRONMENT}
    result["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    return result


def frame(raw, limit):
    if not 0 < len(raw) <= limit:
        raise ValueError("Private worker message exceeds its size limit")
    return struct.pack("!I", len(raw)) + raw


def exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            raise EOFError("Private worker stream ended")
        data.extend(chunk)
    return bytes(data)


class Events:
    def __init__(self, stream):
        self.stream = stream
        self.items = queue.Queue(maxsize=MAX_QUEUED)
        self.finished, self.closed = threading.Event(), threading.Event()
        self.error = False
        self.thread = threading.Thread(
            target=self._read, daemon=True, name="tsfm-worker-events"
        )

    def _put(self, value):
        while not self.closed.is_set():
            try:
                self.items.put(value, timeout=0.05)
                return
            except queue.Full:
                continue

    def _read(self):
        try:
            while not self.closed.is_set():
                prefix = self.stream.read(4)
                if not prefix:
                    break
                if len(prefix) < 4:
                    prefix += exact(self.stream, 4 - len(prefix))
                size = struct.unpack("!I", prefix)[0]
                if not 0 < size <= MAX_EVENT:
                    raise ValueError("Invalid event size")
                event = json.loads(exact(self.stream, size))
                if not isinstance(event, dict) or not isinstance(
                    event.get("type"), str
                ):
                    raise ValueError("Invalid event")
                self._put(event)
        except (OSError, ValueError, EOFError, UnicodeError):
            if not self.closed.is_set():
                self.error = True
                self._put(
                    {
                        "type": "failed",
                        "message": "Invalid private worker event stream",
                        "run_id": None,
                    }
                )
        finally:
            self.finished.set()

    def poll(self):
        return not self.items.empty() or self.finished.is_set()

    def recv(self):
        try:
            return self.items.get_nowait()
        except queue.Empty:
            raise EOFError("Private worker stream ended") from None

    def close(self):
        self.closed.set()
        self.stream.close()
        if self.thread.ident is not None:
            self.thread.join(timeout=0.25)


class Control:
    def __init__(self, stream, startup):
        self.stream, self.startup = stream, startup
        self.requested, self.closed = threading.Event(), threading.Event()
        self.error = False
        self.thread = threading.Thread(
            target=self._write, daemon=True, name="tsfm-worker-control"
        )

    def _write(self):
        try:
            remaining = memoryview(self.startup)
            while remaining and not self.closed.is_set():
                count = self.stream.write(remaining)
                if not count:
                    raise BrokenPipeError("Worker startup ended")
                remaining = remaining[count:]
            self.startup = b""
            self.stream.flush()
            self.requested.wait()
            if not self.closed.is_set():
                self.stream.write(b"C")
                self.stream.flush()
        except (OSError, ValueError):
            if not self.closed.is_set():
                self.error = True
        finally:
            self.startup = b""

    def set(self):
        self.requested.set()

    def close(self):
        self.closed.set()
        self.requested.set()
        self.stream.close()
        if self.thread.ident is not None:
            self.thread.join(timeout=0.25)


class Process:
    """The small process interface retained by ManagedRun's private test hooks."""

    def __init__(self, process, events, control):
        self.process, self.events, self.control = process, events, control

    @property
    def pid(self):
        return self.process.pid

    @property
    def exitcode(self):
        return self.process.poll()

    def is_alive(self):
        return self.process.poll() is None

    def join(self, timeout=None):
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

    def terminate(self):
        self.process.terminate()

    def kill(self):
        self.process.kill()

    def close(self):
        self.control.close()
        self.events.close()


def launch_worker(config, target, log_path, output):
    """Receive only validated configuration and a trusted top-level Python hook."""
    payload = frame(pickle.dumps((config, target), protocol=5), MAX_STARTUP)
    paths = [str(Path(__file__).resolve().parents[1])]
    module = importlib.import_module(target.__module__)
    if getattr(module, "__file__", None):
        paths.append(str(Path(module.__file__).resolve().parent))
    header = {"paths": list(dict.fromkeys(paths)), "log_path": str(log_path)}
    startup = frame(json.dumps(header).encode("utf-8"), MAX_HEADER) + payload
    process = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", BOOTSTRAP],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=output,
        env=worker_environment(),
        close_fds=True,
        bufsize=0,
    )
    events = control = None
    try:
        events = Events(process.stdout)
        control = Control(process.stdin, startup)
        # Keep both wrappers reachable if thread startup is interrupted.
        events.thread.start()
        control.thread.start()
        return Process(process, events, control), events, control
    except BaseException:
        process.kill()
        process.wait(timeout=1.0)
        if control is not None:
            control.close()
        if events is not None:
            events.close()
        process.stdin.close()
        process.stdout.close()
        raise


class EventWriter:
    def __init__(self, stream):
        self.stream = stream

    def send(self, event):
        raw = json.dumps(event, allow_nan=False, separators=(",", ":")).encode("utf-8")
        remaining = memoryview(frame(raw, MAX_EVENT))
        while remaining:
            count = self.stream.write(remaining)
            if not count:
                raise BrokenPipeError("Worker event stream ended")
            remaining = remaining[count:]
        self.stream.flush()

    def close(self):
        self.stream.close()


def run_child(header, source):
    """Never expose this trusted-pickle protocol as a user file or public socket."""
    events = EventWriter(os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0))
    cancellation = threading.Event()
    reader_stop = threading.Event()
    reader = None

    def cancel_reader():
        try:
            if os.name == "nt":
                import _winapi
                import msvcrt

                handle = msvcrt.get_osfhandle(source.fileno())
                # A pending CRT pipe read can block native DLL initialization
                # (observed during NumPy import). Only read an available byte.
                while not reader_stop.is_set():
                    if _winapi.PeekNamedPipe(handle)[0]:
                        source.read(1)
                        break
                    reader_stop.wait(0.01)
            else:
                source.read(1)
        except OSError:
            # A closed controller pipe has the same cancellation meaning as EOF.
            pass
        finally:
            cancellation.set()

    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(header["log_path"], flags), "ab", buffering=0) as output:
            os.dup2(output.fileno(), 1)
            os.dup2(output.fileno(), 2)
            sys.stdout = open(1, "w", encoding="utf-8", buffering=1, closefd=False)
            sys.stderr = open(2, "w", encoding="utf-8", buffering=1, closefd=False)
            size = struct.unpack("!I", exact(source, 4))[0]
            if not 0 < size <= MAX_STARTUP:
                raise ValueError("Invalid private worker startup size")
            payload = exact(source, size)
            reader = threading.Thread(
                target=cancel_reader, daemon=True, name="tsfm-worker-cancel"
            )
            reader.start()
            config, target = pickle.loads(payload)
            if cancellation.is_set():
                events.send({"type": "cancelled", "run_id": None})
            else:
                target(config, events, cancellation)
    except BaseException as exc:
        try:
            events.send(
                {
                    "type": "failed",
                    "message": f"Worker setup or execution failed: {exc}",
                    "run_id": None,
                }
            )
        except (OSError, ValueError):
            pass
    finally:
        reader_stop.set()
        if os.name == "nt" and reader is not None and reader.ident is not None:
            reader.join(timeout=0.25)
        events.close()
