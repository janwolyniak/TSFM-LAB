"""Review checkpoint metadata first, then prepare only the pinned safe files."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from tsfm_lab.models import checkpoints as cp
from tsfm_lab.models.checkpoint_profiles import config_metadata

_RECEIPT_LIMIT = 64 * 1024


def _cancel(should_cancel):
    if should_cancel and should_cancel():
        raise KeyboardInterrupt


def _emit(progress, **event):
    if progress:
        progress(event)


@dataclass(frozen=True)
class CheckpointFile:
    name: str
    size: int | None = None
    sha256: str | None = None
    git_blob_sha1: str | None = None

    def __post_init__(self):
        if self.name not in cp._FILES:
            raise cp.CheckpointError("Unreviewed checkpoint file")
        if self.size is not None and (type(self.size) is not int or self.size <= 0):
            raise cp.CheckpointError("Invalid remote checkpoint file size")
        for value, pattern in ((self.sha256, cp._HEX), (self.git_blob_sha1, cp._SHA)):
            if value is not None and (
                not isinstance(value, str) or not pattern.fullmatch(value)
            ):
                raise cp.CheckpointError("Invalid remote checkpoint file hash")


@dataclass(frozen=True)
class CheckpointPlan:
    repository: str
    requested_revision: str
    resolved_revision: str
    cache_dir: Path
    cached: bool
    download_bytes: int | None
    expected_files: tuple[CheckpointFile, ...]
    _configuration_json: bytes
    _metadata_json: bytes
    missing_files: tuple[str, ...] = ()

    @property
    def configuration(self) -> dict:
        return json.loads(self._configuration_json)

    @property
    def config(self) -> dict:
        return self.configuration

    @property
    def metadata(self) -> dict:
        return json.loads(self._metadata_json)

    @property
    def cache_status(self) -> str:
        return "cached" if self.cached else "missing"

    @property
    def total_bytes(self) -> int | None:
        sizes = [item.size for item in self.expected_files]
        return sum(sizes) if all(size is not None for size in sizes) else None

    def to_dict(self) -> dict:
        return {
            "repository": self.repository,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "cache_dir": str(self.cache_dir),
            "cached": self.cached,
            "cache_status": self.cache_status,
            "download_bytes": self.download_bytes,
            "total_bytes": self.total_bytes,
            "configuration": self.configuration,
            "metadata": self.metadata,
            "expected_files": [asdict(item) for item in self.expected_files],
            "missing_files": list(self.missing_files),
        }


def _snapshot(root, repository, revision):
    path = root / ("models--" + repository.replace("/", "--")) / "snapshots" / revision
    cp._snapshot_path(path, repository, revision)
    if not path.resolve().is_relative_to(root):
        raise cp.CheckpointError("Checkpoint cache escapes its configured directory")
    return path


def _local_revision(root, repository, revision):
    if cp._SHA.fullmatch(revision):
        return revision
    relative = Path(revision)
    if relative.is_absolute() or ".." in relative.parts or "\\" in revision:
        raise cp.CheckpointError("Invalid checkpoint revision reference")
    refs = root / ("models--" + repository.replace("/", "--")) / "refs"
    path = refs / relative
    if not path.resolve().is_relative_to(refs.resolve()):
        raise cp.CheckpointError("Checkpoint reference escapes the cache")
    try:
        with path.open("rb") as source:
            value = source.read(128).decode("ascii").strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as exc:
        raise cp.CheckpointError("Cached checkpoint reference is unreadable") from exc
    return cp._sha(value)


def _receipt_path(root, repository, revision):
    path = (
        root
        / ("models--" + repository.replace("/", "--"))
        / ".tsfm-lab"
        / f"{revision}.json"
    )
    if path.is_symlink() or not path.parent.resolve().is_relative_to(root):
        raise cp.CheckpointError("Checkpoint integrity receipt escapes its cache")
    return path


def _check_receipt(root, checkpoint):
    path = _receipt_path(root, checkpoint.repository, checkpoint.resolved_revision)
    try:
        with path.open("rb") as source:
            raw = source.read(_RECEIPT_LIMIT + 1)
    except FileNotFoundError:
        return
    try:
        if len(raw) > _RECEIPT_LIMIT:
            raise ValueError
        record = cp.validate_checkpoint_record(json.loads(raw))
        if any(
            record[key] != checkpoint.record()[key]
            for key in (
                "repository",
                "resolved_revision",
                "path",
                "files",
                "fingerprint",
            )
        ):
            raise ValueError
    except (ValueError, TypeError, RuntimeError) as exc:
        raise cp.CheckpointError(
            "Cached checkpoint differs from its verified integrity receipt"
        ) from exc


def _git_hash(path, size, should_cancel):
    digest = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            _cancel(should_cancel)
            digest.update(chunk)
    _cancel(should_cancel)
    return digest.hexdigest()


def _verify_files(path, files, expected, should_cancel):
    if set(files) != {item.name for item in expected}:
        raise cp.CheckpointError(
            "Checkpoint consumed files differ from the reviewed manifest"
        )
    for item in expected:
        _cancel(should_cancel)
        actual = files[item.name]
        if item.size is not None and actual["size"] != item.size:
            raise cp.CheckpointError(
                f"Checkpoint size differs from reviewed metadata: {item.name}"
            )
        if item.sha256 is not None and actual["sha256"] != item.sha256:
            raise cp.CheckpointError(
                f"Checkpoint hash differs from reviewed metadata: {item.name}"
            )
        resolved = cp._safe_file(path / item.name, path, hub=True)
        blob = (
            resolved.name if resolved.parent == path.parent.parent / "blobs" else None
        )
        if blob and cp._HEX.fullmatch(blob) and blob != actual["sha256"]:
            raise cp.CheckpointError(f"Cached blob hash mismatch: {item.name}")
        wanted_git = item.git_blob_sha1 or (
            blob if blob and cp._SHA.fullmatch(blob) else None
        )
        if (
            wanted_git
            and _git_hash(resolved, actual["size"], should_cancel) != wanted_git
        ):
            raise cp.CheckpointError(f"Checkpoint Git blob hash mismatch: {item.name}")


def _make_plan(
    repository, requested, resolved, root, cached, config, expected, missing
):
    sizes = [item.size for item in expected if item.name in missing]
    download = sum(sizes) if all(size is not None for size in sizes) else None
    return CheckpointPlan(
        repository,
        requested,
        resolved,
        root,
        cached,
        download,
        tuple(expected),
        cp._json(config),
        cp._json(config_metadata(config)),
        tuple(sorted(missing)),
    )


def _remote_files(info):
    siblings = getattr(info, "siblings", None)
    if not isinstance(siblings, (list, tuple)) or any(
        not isinstance(getattr(item, "rfilename", None), str) for item in siblings
    ):
        raise cp.CheckpointError("Repository metadata lacks a readable file manifest")
    names = [item.rfilename for item in siblings]
    if len(set(names)) != len(names):
        raise cp.CheckpointError("Repository metadata repeats file names")
    cp._reject_adapters(names)
    if not cp._REQUIRED <= set(names):
        raise cp.CheckpointError(
            "Repository lacks config.json and model.safetensors; unsafe weights are unsupported"
        )
    files = []
    for item in siblings:
        if item.rfilename not in cp._FILES:
            continue
        lfs = getattr(item, "lfs", None)
        size = getattr(item, "size", None)
        if lfs is not None:
            sha = (
                lfs.get("sha256")
                if isinstance(lfs, dict)
                else getattr(lfs, "sha256", None)
            )
            lfs_size = (
                lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
            )
            if size is not None and lfs_size is not None and size != lfs_size:
                raise cp.CheckpointError("Repository LFS and file sizes disagree")
            if sha is None:
                raise cp.CheckpointError(
                    "Repository LFS metadata lacks its content hash"
                )
            files.append(
                CheckpointFile(
                    item.rfilename, size if size is not None else lfs_size, sha
                )
            )
        else:
            files.append(
                CheckpointFile(
                    item.rfilename, size, git_blob_sha1=getattr(item, "blob_id", None)
                )
            )
    config = next(item for item in files if item.name == "config.json")
    if config.size is None or config.size > cp._MAX_CONFIG_BYTES:
        raise cp.CheckpointError(
            "Remote config.json needs a known size within the 1 MiB limit"
        )
    return tuple(
        sorted(files, key=lambda item: (item.name != "config.json", item.name))
    )


def _download_path(value, root, repository, resolved, name):
    path = Path(value).expanduser().absolute()
    snapshot = _snapshot(root, repository, resolved)
    if path.name != name or path.parent != snapshot:
        raise cp.CheckpointError(
            "Hub download returned an unexpected checkpoint path or exact SHA"
        )
    return path, snapshot


def plan_checkpoint(
    repository,
    revision,
    *,
    cache_dir=None,
    offline=False,
    progress=None,
    should_cancel=None,
):
    """Inspect an offline snapshot first; otherwise review only metadata and config."""
    cp._repository(repository)
    cp._revision(revision)
    if repository == "Datadog/Toto-2.0-4m":
        from tsfm_lab.models.toto2_profile import ensure_supported_python

        ensure_supported_python()
    if type(offline) is not bool:
        raise cp.CheckpointError("offline must be a boolean")
    _cancel(should_cancel)
    root = (
        Path(cache_dir if cache_dir is not None else cp._hub().cache_dir)
        .expanduser()
        .resolve()
    )
    _emit(progress, type="stage", stage="cache")
    local = _local_revision(root, repository, revision)
    if local is not None:
        path = _snapshot(root, repository, local)
        if all((path / name).exists() for name in cp._REQUIRED):
            checkpoint = cp._inspect(
                path,
                "huggingface",
                repository,
                revision,
                local,
                should_cancel=should_cancel,
            )
            expected = tuple(
                CheckpointFile(name, item["size"], item["sha256"])
                for name, item in sorted(checkpoint.files.items())
            )
            _verify_files(path, checkpoint.files, expected, should_cancel)
            _check_receipt(root, checkpoint)
            return _make_plan(
                repository,
                revision,
                local,
                root,
                True,
                checkpoint.config,
                expected,
                set(),
            )
    if offline:
        raise cp.CheckpointError(
            "Checkpoint is not fully cached; offline planning makes no network requests"
        )
    _cancel(should_cancel)
    hub = cp._hub()
    info = hub.HfApi(token=False).model_info(
        repository, revision=revision, files_metadata=True, token=False
    )
    _cancel(should_cancel)
    resolved = cp._sha(info.sha)
    if cp._SHA.fullmatch(revision) and revision != resolved:
        raise cp.CheckpointError(
            "Hub metadata returned a different exact checkpoint SHA"
        )
    expected = _remote_files(info)
    _cancel(should_cancel)
    value = hub.hf_hub_download(
        repository, "config.json", revision=resolved, cache_dir=root, token=False
    )
    _cancel(should_cancel)
    path, snapshot = _download_path(value, root, repository, resolved, "config.json")
    config, identity = cp._config_file(path, snapshot, hub=True)
    cp._match_repository(config, repository)
    config_file = next(item for item in expected if item.name == "config.json")
    _verify_files(snapshot, {"config.json": identity}, (config_file,), should_cancel)
    # Bind even hash-less metadata to the exact reviewed configuration bytes.
    expected = tuple(
        (
            CheckpointFile(
                item.name, identity["size"], identity["sha256"], item.git_blob_sha1
            )
            if item.name == "config.json"
            else item
        )
        for item in expected
    )
    missing = {item.name for item in expected if not (snapshot / item.name).exists()}
    return _make_plan(
        repository, revision, resolved, root, False, config, expected, missing
    )


def _save_receipt(plan, checkpoint):
    destination = _receipt_path(plan.cache_dir, plan.repository, plan.resolved_revision)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, mode="wb", delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(cp._json(checkpoint.record()))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_checkpoint(plan, *, progress=None, should_cancel=None):
    """Prepare the reviewed manifest at its exact SHA, without model_info resolution."""
    if not isinstance(plan, CheckpointPlan):
        raise cp.CheckpointError("Expected a reviewed checkpoint plan")
    cp._repository(plan.repository)
    cp._sha(plan.resolved_revision)
    if plan.repository == "Datadog/Toto-2.0-4m":
        from tsfm_lab.models.toto2_profile import ensure_supported_python

        ensure_supported_python()
    path = _snapshot(plan.cache_dir, plan.repository, plan.resolved_revision)
    _cancel(should_cancel)
    _emit(progress, type="stage", stage="cache")
    authorized_missing = set(plan.missing_files)
    for item in plan.expected_files:
        if not (path / item.name).exists() and item.name not in authorized_missing:
            raise cp.CheckpointError(
                "Checkpoint cache changed after review; review the missing files again"
            )
    if not plan.cached:
        _emit(progress, type="stage", stage="downloading")
        completed = 0
        _emit(
            progress,
            type="transfer",
            completed_bytes=0,
            total_bytes=plan.download_bytes,
        )
        for item in plan.expected_files:
            _cancel(should_cancel)
            target = path / item.name
            transferred = not target.exists()
            if transferred:
                if item.name not in authorized_missing:
                    raise cp.CheckpointError(
                        "Checkpoint cache changed after review; review the missing files again"
                    )
                value = cp._hub().hf_hub_download(
                    plan.repository,
                    item.name,
                    revision=plan.resolved_revision,
                    cache_dir=plan.cache_dir,
                    token=False,
                )
                _download_path(
                    value,
                    plan.cache_dir,
                    plan.repository,
                    plan.resolved_revision,
                    item.name,
                )
            _cancel(should_cancel)
            if item.name == "config.json":
                config, identity = cp._config_file(target, path, hub=True)
                _verify_files(path, {item.name: identity}, (item,), should_cancel)
                if config != plan.configuration:
                    raise cp.CheckpointError("Checkpoint config changed after review")
            if transferred:
                measured = cp._safe_file(target, path, hub=True).stat().st_size
                if item.size is not None and measured != item.size:
                    raise cp.CheckpointError(
                        f"Checkpoint size differs from reviewed metadata: {item.name}"
                    )
                completed += measured
                _emit(
                    progress,
                    type="transfer",
                    completed_bytes=completed,
                    total_bytes=plan.download_bytes,
                )
    _cancel(should_cancel)
    _emit(progress, type="stage", stage="verifying")
    checkpoint = cp._inspect(
        path,
        "huggingface",
        plan.repository,
        plan.requested_revision,
        plan.resolved_revision,
        should_cancel=should_cancel,
    )
    _verify_files(path, checkpoint.files, plan.expected_files, should_cancel)
    if (
        checkpoint.config != plan.configuration
        or config_metadata(checkpoint.config) != plan.metadata
    ):
        raise cp.CheckpointError("Checkpoint configuration changed after review")
    _cancel(should_cancel)
    if not plan.cached:
        _save_receipt(plan, checkpoint)
    return checkpoint
