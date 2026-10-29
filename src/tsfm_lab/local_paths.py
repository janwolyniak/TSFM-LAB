"""Native execution paths and detached cross-platform path provenance."""

import os
import re
from pathlib import PurePosixPath, PureWindowsPath


def _text(value):
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value)
    ):
        raise ValueError("Choose a local path without control characters")
    return value


def active_path_text(value, *, windows=None):
    """Validate spelling before native Path resolution; never interpret URLs."""
    text = _text(value)
    windows = os.name == "nt" if windows is None else windows
    if text.startswith(("//", "\\\\")):
        raise ValueError(
            "Network and device paths are not supported; choose a local disk"
        )
    drive = re.match(r"^[A-Za-z]:", text)
    if drive:
        if not windows or len(text) < 3 or text[2] not in "\\/":
            raise ValueError(
                "Choose an absolute native drive path, not a drive-relative path"
            )
    elif re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", text):
        raise ValueError("Choose a local path, not a URL")
    elif windows and text.startswith(("/", "\\")):
        raise ValueError("Choose a drive-qualified absolute path or a relative path")
    if windows:
        _windows_components(PureWindowsPath(text))
    return text


def _windows_components(path):
    parts = path.parts[1:] if path.anchor else path.parts
    for part in parts:
        if part in {".", ".."}:
            continue
        if (
            any(char in '<>:"|?*' for char in part)
            or part.endswith((".", " "))
            or re.match(
                r"^(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)",
                part,
                re.IGNORECASE,
            )
            or part.upper() in {"CONIN$", "CONOUT$"}
        ):
            raise ValueError(
                "Windows path contains an alternate stream, device name or ambiguous component"
            )


def recorded_path(value):
    """Validate absolute provenance lexically, without probing another filesystem.

    The returned pure path cannot resolve, open, or otherwise use the recorded
    destination. Native execution must still go through active configuration.
    """
    text = _text(value)
    if text.startswith(("//", "\\\\")):
        raise ValueError("Recorded paths must identify an ordinary local disk")
    if re.match(r"^[A-Za-z]:", text):
        path = PureWindowsPath(text)
        if not path.is_absolute() or not re.fullmatch(r"[A-Za-z]:", path.drive):
            raise ValueError("Recorded drive paths must be absolute")
        _windows_components(path)
    else:
        path = PurePosixPath(text)
        if not path.is_absolute():
            raise ValueError("Recorded paths must be absolute")
    if ".." in path.parts or str(path) != text:
        raise ValueError("Recorded paths must be canonical without traversal")
    return path


def recorded_path_resolver(value, label, directory):
    """Private strict-configuration resolver; directory is deliberately unused."""
    try:
        return recorded_path(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {label}: {exc}") from exc


def checkpoint_record_path(value):
    """Keep legacy absolute checkpoint spellings as inert provenance.

    Older readers accepted UNC locations and redundant separators. Location is
    not part of the numerical checkpoint fingerprint, so display/reload preserves
    that original string. Execution paths use the stricter native policy.
    """
    text = _text(value)
    path = (
        PureWindowsPath(text)
        if re.match(r"^[A-Za-z]:", text) or text.startswith(("//", "\\\\"))
        else PurePosixPath(text)
    )
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(
            "Checkpoint provenance requires an absolute path without traversal"
        )
    return path
