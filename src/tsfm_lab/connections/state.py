"""Atomic private nonsecret connection metadata, outside experiment workspaces."""

import json
import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from time import monotonic, sleep
from uuid import UUID, uuid4

from tsfm_lab.connections.contracts import KINDS, ConnectionError
from tsfm_lab.connections.transport import decode_json

_THREAD_LOCK = RLock()
_FIELDS = {
    "id",
    "kind",
    "label",
    "storage",
    "credential_ref",
    "model",
    "account",
    "workspace_id",
    "registration",
}


def identifier(value):
    try:
        if type(value) is not str or str(UUID(value)) != value:
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        raise ConnectionError("invalid") from None
    return value


def text(value, *, maximum=200):
    if (
        type(value) is not str
        or not value.strip()
        or len(value) > maximum
        or any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value)
    ):
        raise ConnectionError("invalid")
    return value


def account(value, *, registration=False):
    if value is None:
        return
    fields = {"issuer", "client_id"}
    if not registration:
        fields.add("subject")
    if type(value) is not dict or not fields <= set(value) <= fields | (
        {"email"} if not registration else set()
    ):
        raise ConnectionError("invalid")
    if (
        value["issuer"] != "https://auth.openai.com"
        or value["client_id"] == "dynamic_agent_client"
    ):
        raise ConnectionError("invalid")
    for item in value.values():
        text(item, maximum=512)


def validate_state(value):
    if type(value) is not dict or set(value) != {
        "schema_version",
        "host_id",
        "generation",
        "selected",
        "profiles",
        "retired_refs",
    }:
        raise ConnectionError("state")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or type(value["generation"]) is not int
        or value["generation"] < 0
    ):
        raise ConnectionError("state")
    if not isinstance(value["host_id"], str) or not value["host_id"].startswith(
        "urn:uuid:"
    ):
        raise ConnectionError("state")
    host = identifier(value["host_id"][9:])
    if UUID(host).version != 4:
        raise ConnectionError("state")
    if (
        type(value["profiles"]) is not list
        or len(value["profiles"]) > 100
        or type(value["retired_refs"]) is not list
        or len(value["retired_refs"]) > 200
    ):
        raise ConnectionError("state")
    seen, references = set(), set()
    for profile in value["profiles"]:
        if type(profile) is not dict or set(profile) != _FIELDS:
            raise ConnectionError("state")
        identifier(profile["id"])
        if (
            profile["id"] in seen
            or profile["kind"] not in KINDS
            or profile["storage"] not in {"session", "vault"}
        ):
            raise ConnectionError("state")
        seen.add(profile["id"])
        text(profile["label"], maximum=80)
        if profile["credential_ref"] is not None:
            identifier(profile["credential_ref"])
            if profile["credential_ref"] in references:
                raise ConnectionError("state")
            references.add(profile["credential_ref"])
        for key in ("model", "workspace_id"):
            if profile[key] is not None:
                text(profile[key])
        account(profile["account"])
        account(profile["registration"], registration=True)
        if profile["kind"] != "chatgpt_plan" and (
            profile["account"] is not None or profile["registration"] is not None
        ):
            raise ConnectionError("state")
        if profile["kind"] != "anthropic_api" and profile["workspace_id"] is not None:
            raise ConnectionError("state")
    if value["selected"] is not None and value["selected"] not in seen:
        raise ConnectionError("state")
    for item in value["retired_refs"]:
        if (
            type(item) is not dict
            or set(item) != {"reference", "storage"}
            or item["storage"] not in {"session", "vault"}
        ):
            raise ConnectionError("state")
        identifier(item["reference"])
        if item["reference"] in references:
            raise ConnectionError("state")
        references.add(item["reference"])
    return value


def default_directory():
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
        return base / "TSFM-LAB" / "connections"
    return Path.home() / ".config" / "tsfm-lab" / "connections"


class StateStore:
    def __init__(self, directory=None):
        self.directory = (
            Path(directory) if directory is not None else default_directory()
        )
        self.path = self.directory / "connections.json"
        self._new_host = f"urn:uuid:{uuid4()}"

    def _directory(self):
        if any(path.is_symlink() for path in (self.directory, *self.directory.parents)):
            raise ConnectionError("storage")
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name != "nt" and self.directory.stat().st_mode & 0o077:
            raise ConnectionError("storage")

    @contextmanager
    def locked(self):
        if not _THREAD_LOCK.acquire(timeout=5):
            raise ConnectionError("state")
        try:
            self._directory()
            path = self.directory / ".lock"
            flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
            descriptor = None
            locked = False
            try:
                if path.is_symlink():
                    raise ConnectionError("storage")
                descriptor = os.open(path, flags, 0o600)
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ConnectionError("storage")
                if os.name == "nt":
                    import msvcrt

                    os.write(descriptor, b"0")
                    os.lseek(descriptor, 0, 0)
                else:
                    import fcntl
                end = monotonic() + 5
                while not locked:
                    try:
                        if os.name == "nt":
                            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                        else:
                            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        locked = True
                    except OSError:
                        if monotonic() >= end:
                            raise ConnectionError("state") from None
                        sleep(0.01)
                yield self.read()
            except ConnectionError:
                raise
            except Exception:
                raise ConnectionError("storage") from None
            finally:
                if descriptor is not None:
                    if locked:
                        if os.name == "nt":
                            os.lseek(descriptor, 0, 0)
                            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                        else:
                            fcntl.flock(descriptor, fcntl.LOCK_UN)
                    os.close(descriptor)
        finally:
            _THREAD_LOCK.release()

    def read(self):
        try:
            if (
                any(
                    path.is_symlink()
                    for path in (self.directory, *self.directory.parents)
                )
                or self.path.is_symlink()
            ):
                raise ConnectionError("storage")
            if not self.path.exists():
                return {
                    "schema_version": 1,
                    "host_id": self._new_host,
                    "generation": 0,
                    "selected": None,
                    "profiles": [],
                    "retired_refs": [],
                }
            if os.name != "nt" and self.directory.stat().st_mode & 0o077:
                raise ConnectionError("storage")
            descriptor = os.open(
                self.path,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
            )
            with os.fdopen(descriptor, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or (
                    os.name != "nt" and info.st_mode & 0o077
                ):
                    raise ConnectionError("storage")
                raw = stream.read(262145)
            if len(raw) > 262144:
                raise ConnectionError("state")
            return validate_state(decode_json(raw))
        except ConnectionError:
            raise
        except Exception:
            raise ConnectionError("storage") from None

    def write(self, value):
        temporary = None
        try:
            validate_state(value)
            self._directory()
            if self.path.is_symlink():
                raise ConnectionError("storage")
            raw = (
                json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + "\n"
            ).encode()
            if len(raw) > 262144:
                raise ConnectionError("bounds")
            with tempfile.NamedTemporaryFile(
                dir=self.directory, prefix=".connections-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except ConnectionError:
            raise
        except Exception:
            raise ConnectionError("storage") from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
