"""Pinned, safe Chronos-Bolt checkpoints; only explicit pulls download artifacts."""

import hashlib
import importlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from tsfm_lab.models.catalog import CURATED_REPOSITORIES

_FILES = {"config.json", "model.safetensors", "README.md"}
_REQUIRED = {"config.json", "model.safetensors"}
_SHA = re.compile(r"[0-9a-f]{40}")
_HEX = re.compile(r"[0-9a-f]{64}")
_ARCHITECTURE = "ChronosBoltModelForForecasting"
_PIPELINE = "ChronosBoltPipeline"
_MAX_CONFIG_BYTES = 1024 * 1024
_LOADING_DIRECTIVES = {
    "transformers_weights",
    "configuration_files",
    "quantization_config",
}


class CheckpointError(ValueError):
    """Checkpoint identity, contents, or supported loading policy is invalid."""


def _json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _hub():
    try:
        hub = importlib.import_module("huggingface_hub")
        constants = importlib.import_module("huggingface_hub.constants")
    except ImportError as exc:
        raise ImportError(
            "Hugging Face support requires the optional tsfm-lab[hub] dependencies"
        ) from exc
    return SimpleNamespace(
        HfApi=hub.HfApi,
        snapshot_download=hub.snapshot_download,
        hf_hub_download=hub.hf_hub_download,
        cache_dir=constants.HF_HUB_CACHE,
    )


def _repository(repository):
    if repository not in CURATED_REPOSITORIES:
        raise CheckpointError(
            "Only curated Chronos-Bolt tiny/small repositories are supported"
        )


def _revision(revision):
    if not isinstance(revision, str) or not revision.strip():
        raise CheckpointError("Checkpoint revision must be a nonempty string")


def _sha(value):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise CheckpointError(
            "An exact lowercase 40-character checkpoint SHA is required"
        )
    return value


def _bounds(context, horizon, quantiles):
    if type(context) is not int or not 0 < context <= 2048:
        raise CheckpointError("Supported Bolt context length must be in 1..2048")
    if type(horizon) is not int or not 0 < horizon <= 64:
        raise CheckpointError("Supported Bolt prediction length must be in 1..64")
    if (
        not isinstance(quantiles, list)
        or not quantiles
        or any(
            type(q) not in (int, float) or not math.isfinite(q) or not 0 < q < 1
            for q in quantiles
        )
        or quantiles != sorted(set(quantiles))
        or 0.5 not in quantiles
    ):
        raise CheckpointError(
            "Bolt quantiles must be ordered, unique, inside (0,1), and include 0.5"
        )


def _security(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _LOADING_DIRECTIVES:
                raise CheckpointError(
                    f"Checkpoint loading directive {key!r} is unsupported; "
                    "only the recorded config.json and model.safetensors may be loaded"
                )
            if key in {"auto_map", "custom_pipelines"} or (
                key == "trust_remote_code" and child is not False
            ):
                raise CheckpointError(
                    "Checkpoint remote code configuration is not supported"
                )
            _security(child)
    elif isinstance(value, list):
        for child in value:
            _security(child)


def _config(raw: bytes) -> dict:
    try:
        if len(raw) > _MAX_CONFIG_BYTES:
            raise CheckpointError("Checkpoint config.json exceeds the 1 MiB size limit")
        config = json.loads(raw)
        _json(config)
        if not isinstance(config, dict):
            raise CheckpointError("Checkpoint config must be a JSON object")
        _security(config)
        if (
            config.get("architectures") != [_ARCHITECTURE]
            or config.get("chronos_pipeline_class") != _PIPELINE
            or config.get("model_type") != "t5"
        ):
            raise CheckpointError(
                "Checkpoint is not a supported Chronos-Bolt architecture"
            )
        options = config["chronos_config"]
        if not isinstance(options, dict):
            raise CheckpointError("chronos_config must be an object")
        _bounds(
            options["context_length"],
            options["prediction_length"],
            options["quantiles"],
        )
        return config
    except RecursionError as exc:
        raise CheckpointError("Checkpoint configuration nesting is too deep") from exc
    except (ValueError, TypeError, KeyError, UnicodeError) as exc:
        raise CheckpointError(f"Invalid checkpoint configuration: {exc}") from exc


def _safe_file(path: Path, root: Path, *, hub: bool) -> Path:
    resolved = path.resolve(strict=True)
    allowed = resolved.is_relative_to(root) or (
        hub and resolved.is_relative_to(root.parent.parent / "blobs")
    )
    if not allowed or not resolved.is_file():
        raise CheckpointError(
            f"Checkpoint file escapes its permitted directory: {path.name}"
        )
    return resolved


def _config_file(path: Path, root: Path, *, hub: bool) -> tuple[dict, dict]:
    try:
        resolved = _safe_file(path, root, hub=hub)
        with resolved.open("rb") as stream:
            raw = stream.read(_MAX_CONFIG_BYTES + 1)
        config = _config(raw)
        return config, {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
    except OSError as exc:
        raise CheckpointError(
            "Checkpoint config.json is missing or unreadable"
        ) from exc


def _file(path: Path, root: Path, *, hub: bool) -> tuple[Path, dict]:
    try:
        resolved = _safe_file(path, root, hub=hub)
        digest, size = hashlib.sha256(), 0
        with resolved.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        if not size:
            raise CheckpointError(f"Checkpoint file is empty: {path.name}")
        return resolved, {"sha256": digest.hexdigest(), "size": size}
    except OSError as exc:
        raise CheckpointError(
            f"Checkpoint file missing or unreadable: {path.name}"
        ) from exc


def _fingerprint(record: dict) -> str:
    fields = (
        "files",
        "architecture",
        "pipeline_class",
        "model_type",
        "native_context_length",
        "native_prediction_length",
        "quantiles",
        "license",
    )
    payload = {
        "serialization": "tsfm_checkpoint_v1",
        **{key: record[key] for key in fields},
    }
    return hashlib.sha256(_json(payload)).hexdigest()


@dataclass(frozen=True)
class ResolvedCheckpoint:
    source: str
    path: Path
    repository: str | None
    requested_revision: str | None
    resolved_revision: str | None
    files: dict
    fingerprint: str
    config: dict
    license: str | None

    def record(self) -> dict:
        """Return consumed-file identity and native capabilities, excluding raw config."""
        options = self.config["chronos_config"]
        return json.loads(
            _json(
                {
                    "schema_version": 1,
                    "source": self.source,
                    "path": str(self.path),
                    "repository": self.repository,
                    "requested_revision": self.requested_revision,
                    "resolved_revision": self.resolved_revision,
                    "files": self.files,
                    "fingerprint": self.fingerprint,
                    "license": self.license,
                    "architecture": self.config["architectures"][0],
                    "pipeline_class": self.config["chronos_pipeline_class"],
                    "model_type": self.config["model_type"],
                    "native_context_length": options["context_length"],
                    "native_prediction_length": options["prediction_length"],
                    "quantiles": options["quantiles"],
                }
            )
        )


def validate_checkpoint_record(record: dict) -> dict:
    """Validate persisted identity without Hub, Torch, filesystem access, or secrets."""
    try:
        if (
            not isinstance(record, dict)
            or type(record["schema_version"]) is not int
            or record["schema_version"] != 1
        ):
            raise CheckpointError("Invalid checkpoint record schema")
        fields = {
            "schema_version",
            "source",
            "path",
            "repository",
            "requested_revision",
            "resolved_revision",
            "files",
            "fingerprint",
            "license",
            "architecture",
            "pipeline_class",
            "model_type",
            "native_context_length",
            "native_prediction_length",
            "quantiles",
        }
        if set(record) != fields or ".." in Path(record["path"]).parts:
            raise CheckpointError("Invalid checkpoint record fields or path")
        if (
            record["source"] not in {"local", "huggingface"}
            or not isinstance(record["path"], str)
            or not Path(record["path"]).is_absolute()
        ):
            raise CheckpointError(
                "Checkpoint record requires an absolute path and known source"
            )
        if record["source"] == "local":
            if any(
                record[key] is not None
                for key in ("repository", "requested_revision", "resolved_revision")
            ):
                raise CheckpointError("Local checkpoint cannot claim a Hub revision")
        else:
            _repository(record["repository"])
            _revision(record["requested_revision"])
            _sha(record["resolved_revision"])
            _snapshot_path(
                Path(record["path"]),
                record["repository"],
                record["resolved_revision"],
                check_directory=False,
            )
        files = record["files"]
        if not isinstance(files, dict) or not _REQUIRED <= set(files) <= _FILES:
            raise CheckpointError("Checkpoint record has invalid consumed files")
        for item in files.values():
            if (
                not isinstance(item, dict)
                or set(item) != {"sha256", "size"}
                or not isinstance(item["sha256"], str)
                or not _HEX.fullmatch(item["sha256"])
                or type(item["size"]) is not int
                or item["size"] <= 0
            ):
                raise CheckpointError("Invalid checkpoint file identity")
        if (
            record["architecture"] != _ARCHITECTURE
            or record["pipeline_class"] != _PIPELINE
            or record["model_type"] != "t5"
        ):
            raise CheckpointError("Unsupported checkpoint record architecture")
        _bounds(
            record["native_context_length"],
            record["native_prediction_length"],
            record["quantiles"],
        )
        if record["license"] is not None and (
            not isinstance(record["license"], str) or not record["license"].strip()
        ):
            raise CheckpointError("Invalid checkpoint license")
        if record["fingerprint"] != _fingerprint(record):
            raise CheckpointError("Checkpoint fingerprint does not match its manifest")
        return json.loads(_json(record))
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise CheckpointError(f"Invalid checkpoint record: {exc}") from exc


def _snapshot_path(
    path: Path, repository: str, revision: str, *, check_directory=True
) -> Path:
    path = path.expanduser().absolute()
    if (
        path.name != revision
        or path.parent.name != "snapshots"
        or path.parent.parent.name != "models--" + repository.replace("/", "--")
    ):
        raise CheckpointError(
            "Cached snapshot path does not match repository and exact SHA"
        )
    if check_directory and path.is_symlink():
        raise CheckpointError("Cached snapshot directory cannot be a symbolic link")
    return path


def _reject_adapters(names):
    for name in names:
        if name == "adapter_config.json" or name.startswith("adapter_model."):
            raise CheckpointError(
                f"PEFT adapter sidecar {name!r} is unsupported; "
                "the checkpoint must load only its recorded base weights"
            )


def _inspect(
    path: Path, source: str, repository=None, requested=None, resolved=None
) -> ResolvedCheckpoint:
    if source == "huggingface":
        _snapshot_path(Path(path), repository, resolved)
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise CheckpointError("Checkpoint directory does not exist")
    if source == "huggingface":
        _snapshot_path(path, repository, resolved)
    try:
        _reject_adapters(candidate.name for candidate in path.iterdir())
    except OSError as exc:
        raise CheckpointError("Checkpoint directory is unreadable") from exc
    config, config_identity = _config_file(
        path / "config.json", path, hub=source == "huggingface"
    )
    files, consumed = {"config.json": config_identity}, {}
    for name in sorted(_FILES - {"config.json"}):
        candidate = path / name
        if name in _REQUIRED or candidate.exists() or candidate.is_symlink():
            consumed[name], files[name] = _file(
                candidate, path, hub=source == "huggingface"
            )
    license = None
    if "README.md" in consumed:
        try:
            card = consumed["README.md"].read_text(encoding="utf-8")
            header = card.split("---", 2)[1] if card.startswith("---\n") else ""
            match = re.search(r"(?m)^license:\s*([A-Za-z0-9.+-]+)\s*$", header)
            license = match.group(1) if match else None
        except (OSError, UnicodeError, IndexError) as exc:
            raise CheckpointError("Checkpoint model card is unreadable") from exc
    checkpoint = ResolvedCheckpoint(
        source, path, repository, requested, resolved, files, "", config, license
    )
    return ResolvedCheckpoint(
        source,
        path,
        repository,
        requested,
        resolved,
        files,
        _fingerprint(checkpoint.record()),
        config,
        license,
    )


def inspect_local_checkpoint(path: str | Path) -> ResolvedCheckpoint:
    return _inspect(Path(path), "local")


def verify_checkpoint(checkpoint: ResolvedCheckpoint) -> ResolvedCheckpoint:
    """Recheck consumed bytes and security immediately before local inference."""
    if not isinstance(checkpoint, ResolvedCheckpoint):
        raise CheckpointError("Expected a resolved checkpoint")
    try:
        record = validate_checkpoint_record(checkpoint.record())
        current = _inspect(
            checkpoint.path,
            checkpoint.source,
            checkpoint.repository,
            checkpoint.requested_revision,
            checkpoint.resolved_revision,
        )
        if current.record() != record or current.config != checkpoint.config:
            raise CheckpointError("Checkpoint changed since it was resolved")
        return current
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise CheckpointError(f"Checkpoint verification failed: {exc}") from exc


def resolve_checkpoint(
    repository: str, revision: str, *, cache_dir=None, offline=True
) -> ResolvedCheckpoint:
    """Resolve metadata if permitted, then require already-cached exact files."""
    _repository(repository)
    _revision(revision)
    if type(offline) is not bool:
        raise CheckpointError("offline must be a boolean")
    if offline:
        resolved = _sha(revision)
    elif _SHA.fullmatch(revision):
        resolved = revision
    else:
        resolved = _sha(
            _hub()
            .HfApi(token=False)
            .model_info(repository, revision=revision, token=False)
            .sha
        )
    path = _hub().snapshot_download(
        repository,
        revision=resolved,
        cache_dir=cache_dir,
        local_files_only=True,
        token=False,
        allow_patterns=sorted(_FILES),
    )
    path = _snapshot_path(Path(path), repository, resolved)
    return _inspect(path, "huggingface", repository, revision, resolved)


def pull_checkpoint(
    repository: str, revision: str = "main", *, cache_dir=None
) -> ResolvedCheckpoint:
    """Explicitly download a reviewed pinned config, then only safe checkpoint files."""
    _repository(repository)
    _revision(revision)
    hub = _hub()
    info = hub.HfApi(token=False).model_info(
        repository, revision=revision, files_metadata=True, token=False
    )
    resolved = _sha(info.sha)
    if not isinstance(info.siblings, (list, tuple)) or any(
        not isinstance(getattr(entry, "rfilename", None), str)
        for entry in info.siblings
    ):
        raise CheckpointError("Repository metadata lacks a readable file manifest")
    names = {entry.rfilename for entry in info.siblings}
    _reject_adapters(names)
    if not _REQUIRED <= names:
        raise CheckpointError(
            "Repository lacks config.json and model.safetensors; unsafe weights are unsupported"
        )
    config_size = next(
        getattr(entry, "size", None)
        for entry in info.siblings
        if entry.rfilename == "config.json"
    )
    if config_size is not None and (
        type(config_size) is not int or not 0 < config_size <= _MAX_CONFIG_BYTES
    ):
        raise CheckpointError(
            "Remote config.json size is invalid or exceeds the 1 MiB limit"
        )
    config_path = hub.hf_hub_download(
        repository, "config.json", revision=resolved, cache_dir=cache_dir, token=False
    )
    config_path = Path(config_path)
    config_snapshot = _snapshot_path(config_path.parent, repository, resolved)
    if config_path.name != "config.json":
        raise CheckpointError("Hub config download returned an unexpected file")
    _config_file(config_path, config_snapshot.resolve(), hub=True)
    path = hub.snapshot_download(
        repository,
        revision=resolved,
        cache_dir=cache_dir,
        local_files_only=False,
        token=False,
        allow_patterns=sorted(_FILES & names),
    )
    return _inspect(
        _snapshot_path(Path(path), repository, resolved),
        "huggingface",
        repository,
        revision,
        resolved,
    )


def cached_checkpoints(cache_dir=None) -> list[dict]:
    """Inspect only curated cache directories without network or downloads."""
    root = Path(cache_dir if cache_dir is not None else _hub().cache_dir).expanduser()
    entries = []
    for repository in CURATED_REPOSITORIES:
        snapshots = root / ("models--" + repository.replace("/", "--")) / "snapshots"
        if not snapshots.is_dir():
            continue
        for path in sorted(snapshots.iterdir()):
            if not path.is_dir():
                continue
            try:
                _sha(path.name)
                checkpoint = _inspect(
                    path, "huggingface", repository, path.name, path.name
                )
                entries.append(
                    {
                        **checkpoint.record(),
                        "status": "cached",
                        "bytes": sum(
                            item["size"] for item in checkpoint.files.values()
                        ),
                    }
                )
            except (ValueError, OSError) as exc:
                entries.append(
                    {
                        "source": "huggingface",
                        "repository": repository,
                        "resolved_revision": path.name,
                        "path": str(path.absolute()),
                        "status": "incomplete",
                        "error": str(exc),
                    }
                )
    return entries
