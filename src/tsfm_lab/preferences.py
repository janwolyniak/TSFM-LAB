"""Versioned nonsecret terminal preferences, separate from experiment records."""

import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from tsfm_lab.config import ConfigError
from tsfm_lab.workflows import PROFILE_NAMES

_PATH_FIELDS = ("workspace", "results_directory", "cache_directory")
_MAX_BYTES = 64 * 1024


def _local_path(value) -> Path:
    if not isinstance(value, (str, Path)):
        raise ConfigError("Choose a local directory")
    text = str(value)
    if (
        not text.strip()
        or "://" in text
        or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in text)
    ):
        raise ConfigError("Choose a local directory without control characters")
    try:
        return Path(value).expanduser().resolve()
    except (OSError, ValueError, RuntimeError) as exc:
        raise ConfigError("Cannot resolve the local directory") from exc


def validate_directory(path: Path) -> Path:
    """Probe an existing directory/ancestor without creating the selected path."""
    selected = _local_path(path)
    try:
        ancestor = selected
        while not ancestor.exists():
            ancestor = ancestor.parent
        if not ancestor.is_dir():
            raise OSError("A path component is a file")
        with tempfile.TemporaryFile(dir=ancestor):
            pass
    except OSError as exc:
        raise ConfigError(
            f"Directory is not writable: {selected}. Choose another directory "
            "or grant write access to its existing parent."
        ) from exc
    return selected


def _cache_directory() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return _local_path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return _local_path(os.environ["HUGGINGFACE_HUB_CACHE"])
    home = os.environ.get("HF_HOME")
    if home:
        return _local_path(home) / "hub"
    cache = os.environ.get("XDG_CACHE_HOME")
    root = _local_path(cache) if cache else Path.home() / ".cache"
    return root / "huggingface" / "hub"


@dataclass(frozen=True)
class Preferences:
    workspace: Path
    results_directory: Path
    cache_directory: Path
    category: str = "generic"
    export_format: str = "csv"
    onboarding_status: str = "new"

    def __post_init__(self):
        for field in _PATH_FIELDS:
            object.__setattr__(self, field, _local_path(getattr(self, field)))
        if not isinstance(self.category, str) or self.category not in PROFILE_NAMES:
            raise ConfigError("Choose a supported default category")
        if not isinstance(self.export_format, str) or self.export_format not in {
            "csv",
            "json",
            "parquet",
        }:
            raise ConfigError("Choose CSV, JSON, or Parquet as the export format")
        if not isinstance(
            self.onboarding_status, str
        ) or self.onboarding_status not in {
            "new",
            "skipped",
            "completed",
        }:
            raise ConfigError("Invalid onboarding status")

    @classmethod
    def defaults(cls, base_directory: Path | None = None):
        workspace = _local_path(
            base_directory if base_directory is not None else Path.cwd()
        )
        return cls(workspace, workspace / "results", _cache_directory())


def _settings_directory() -> Path:
    if os.environ.get("TSFM_LAB_SETTINGS_DIR"):
        return _local_path(os.environ["TSFM_LAB_SETTINGS_DIR"])
    if os.name == "nt":
        return (
            _local_path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "TSFM-LAB"
        )
    return Path.home() / ".config" / "tsfm-lab"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate preference field")
        result[key] = value
    return result


class PreferenceStore:
    def __init__(self, directory: Path | None = None):
        self.directory = (
            _local_path(directory) if directory is not None else _settings_directory()
        )
        self.path = self.directory / "preferences.json"

    def load(self) -> Preferences:
        try:
            with self.path.open("rb") as source:
                payload = source.read(_MAX_BYTES + 1)
        except FileNotFoundError:
            return Preferences.defaults()
        except OSError as exc:
            raise ConfigError(
                f"Cannot read preferences at {self.path}. Check access permissions."
            ) from exc
        try:
            if len(payload) > _MAX_BYTES:
                raise ValueError("Preferences exceed size limit")
            fields = json.loads(payload, object_pairs_hook=_unique_object)
            required = set(Preferences.__dataclass_fields__) | {"schema_version"}
            if (
                type(fields) is not dict
                or set(fields) != required
                or type(fields["schema_version"]) is not int
                or fields["schema_version"] != 1
            ):
                raise ValueError("Unsupported preference schema")
            for field in _PATH_FIELDS:
                if (
                    not isinstance(fields[field], str)
                    or not Path(fields[field]).is_absolute()
                ):
                    raise ValueError("Persisted directories must be absolute")
            return Preferences(
                **{key: fields[key] for key in required - {"schema_version"}}
            )
        except (ValueError, TypeError, UnicodeError, RuntimeError) as exc:
            raise ConfigError(
                f"Invalid preferences at {self.path}. Restore this file from a known "
                "valid copy or choose another TSFM_LAB_SETTINGS_DIR."
            ) from exc

    def save(self, preferences: Preferences) -> None:
        if not isinstance(preferences, Preferences):
            raise ConfigError("Expected validated nonsecret preferences")
        for field in _PATH_FIELDS:
            validate_directory(getattr(preferences, field))
        self._write(preferences)

    def skip(self, preferences: Preferences) -> Preferences:
        """Record a dismissed offer without probing unused work/cache paths."""
        if not isinstance(preferences, Preferences):
            raise ConfigError("Expected validated nonsecret preferences")
        skipped = replace(preferences, onboarding_status="skipped")
        self._write(skipped)
        return skipped

    def _write(self, preferences):
        fields = {"schema_version": 1, **asdict(preferences)}
        for field in _PATH_FIELDS:
            fields[field] = str(fields[field])
        temporary = None
        try:
            validate_directory(self.directory)
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.directory, delete=False
            ) as target:
                temporary = Path(target.name)
                json.dump(fields, target, ensure_ascii=False, indent=2)
                target.write("\n")
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, self.path)
        except OSError as exc:
            raise ConfigError(
                f"Cannot save preferences at {self.path}. Check access permissions; "
                "previous preferences were preserved."
            ) from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
