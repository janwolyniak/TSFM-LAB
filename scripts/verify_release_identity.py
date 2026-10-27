"""Bind release archive, wheel, and optional installed bytes to a Git commit."""

import argparse
import hashlib
import io
import json
import runpy
import subprocess
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = runpy.run_path(str(Path(__file__).with_name("release_contract.py")))
GENERATED_SOURCE = {"PKG-INFO", "setup.cfg"} | {
    f"src/tsfm_lab.egg-info/{name}"
    for name in (
        "PKG-INFO",
        "SOURCES.txt",
        "dependency_links.txt",
        "entry_points.txt",
        "requires.txt",
        "top_level.txt",
    )
}
GENERATED_WHEEL = {
    "METADATA",
    "WHEEL",
    "entry_points.txt",
    "top_level.txt",
    "RECORD",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": _hash(raw)}


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _tar_files(raw):
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        result = {}
        seen = set()
        for member in archive.getmembers():
            name = member.name
            require(name not in seen, f"Duplicate source archive member: {name}")
            seen.add(name)
            require(
                not PurePosixPath(name).is_absolute()
                and ".." not in PurePosixPath(name).parts
                and "\\" not in name,
                f"Unsafe source archive member: {name}",
            )
            require(
                member.isfile() or member.isdir(),
                f"Unsupported source archive member: {name}",
            )
            if member.isfile():
                result[name] = archive.extractfile(member).read()
        return result


def _wheel_files(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate wheel archive member")
        result = {}
        for item in archive.infolist():
            name = item.filename
            require(
                not PurePosixPath(name).is_absolute()
                and ".." not in PurePosixPath(name).parts
                and "\\" not in name,
                f"Unsafe wheel archive member: {name}",
            )
            require(
                item.external_attr >> 16 & 0o170000 != 0o120000,
                f"Wheel archive contains a symbolic link: {name}",
            )
            if not item.is_dir():
                result[name] = archive.read(item)
        return result


def _compare(expected, actual, description):
    missing, extra = sorted(expected.keys() - actual.keys()), sorted(
        actual.keys() - expected.keys()
    )
    changed = sorted(
        name
        for name in expected.keys() & actual.keys()
        if expected[name] != actual[name]
    )
    require(
        not (missing or extra or changed),
        f"{description} mismatch: missing={missing}, extra={extra}, changed={changed}",
    )
    return len(expected)


def verify_installed(wheel, package_root):
    """Verify package and original wheel metadata without importing the package."""
    package_root = Path(package_root).absolute()
    require(
        package_root.is_dir() and not package_root.is_symlink(),
        "Installed package must be an ordinary directory",
    )
    packaged = _wheel_files(wheel)
    _, package = CONTRACT["wheel_contract"](packaged)
    expected = {
        name: raw for name, raw in packaged.items() if name.startswith("tsfm_lab/")
    }
    require(expected, "Wheel has no tsfm_lab runtime files")
    installed = {}
    for path in package_root.rglob("*"):
        require(
            not path.is_symlink(), f"Installed package contains a symbolic link: {path}"
        )
        if path.is_file():
            relative = path.relative_to(package_root)
            if "__pycache__" in relative.parts and path.suffix == ".pyc":
                continue
            installed[f"tsfm_lab/{relative.as_posix()}"] = path.read_bytes()
    count = _compare(expected, installed, "Installed runtime")
    metadata = {
        name: raw
        for name, raw in packaged.items()
        if ".dist-info/" in name and not name.endswith("/RECORD")
    }
    for name, raw in metadata.items():
        path = package_root.parent / name
        require(
            path.is_file() and not path.is_symlink() and path.read_bytes() == raw,
            f"Installed wheel metadata differs: {name}",
        )
    return {
        **package,
        "package_root": str(package_root.resolve()),
        "runtime_files": count,
        "metadata_files": len(metadata),
        "wheel": identity(wheel),
    }


def verify_release(source, wheel, revision, *, root=ROOT, installed=None):
    """Compare delivered files with immutable Git bytes, never checkout contents."""
    root, source, wheel = Path(root).resolve(), Path(source), Path(wheel)
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    committed = _tar_files(
        subprocess.run(
            ["git", "archive", "--format=tar", resolved],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout
    )
    configuration = tomllib.loads(committed["pyproject.toml"].decode())
    name, version = (
        configuration["project"]["name"],
        configuration["project"]["version"],
    )
    CONTRACT["version_contract"](committed["src/tsfm_lab/__init__.py"], version)
    helpers = runpy.run_path(str(ROOT / "scripts/build_distribution.py"))
    helpers["inspect_source"](source, name=name, version=version)
    helpers["inspect_wheel"](wheel, name=name, version=version)
    prefix = f"{name.replace('-', '_')}-{version}/"
    archived = {
        name.removeprefix(prefix): raw
        for name, raw in _tar_files(source.read_bytes()).items()
    }
    unexpected = archived.keys() - committed.keys() - GENERATED_SOURCE
    require(not unexpected, f"Uncommitted source archive files: {sorted(unexpected)}")
    if "setup.cfg" in archived and "setup.cfg" not in committed:
        require(
            archived["setup.cfg"] == b"[egg_info]\ntag_build = \ntag_date = 0\n\n",
            "Unexpected generated setup.cfg content",
        )
    expected_names = CONTRACT["source_names"](committed, configuration)
    source_count = _compare(
        {name: committed[name] for name in expected_names},
        {
            name: raw
            for name, raw in archived.items()
            if name not in GENERATED_SOURCE or name in committed
        },
        "Committed source",
    )
    require("PKG-INFO" in archived, "Source archive has no package metadata")
    for entry in ("PKG-INFO", "src/tsfm_lab.egg-info/PKG-INFO"):
        if entry in archived:
            CONTRACT["metadata_contract"](archived[entry], configuration)
    if "src/tsfm_lab.egg-info/entry_points.txt" in archived:
        CONTRACT["entry_contract"](
            archived["src/tsfm_lab.egg-info/entry_points.txt"], configuration
        )
    runtime = {
        name.removeprefix("src/"): raw
        for name, raw in committed.items()
        if name.startswith("src/tsfm_lab/")
    }
    require(runtime, "Git revision has no tsfm_lab runtime files")
    _compare(
        {f"src/{name}": raw for name, raw in runtime.items()},
        {
            name: raw
            for name, raw in archived.items()
            if name.startswith("src/tsfm_lab/")
        },
        "Source runtime",
    )
    packaged = _wheel_files(wheel)
    CONTRACT["wheel_contract"](packaged, configuration)
    runtime_count = _compare(
        runtime,
        {name: raw for name, raw in packaged.items() if name.startswith("tsfm_lab/")},
        "Wheel runtime",
    )
    metadata_prefix = f"{name.replace('-', '_')}-{version}.dist-info/"
    for entry, raw in packaged.items():
        if entry.startswith("tsfm_lab/"):
            continue
        relative = entry.removeprefix(metadata_prefix)
        require(entry.startswith(metadata_prefix), f"Unexpected wheel file: {entry}")
        if relative.startswith("licenses/"):
            original = relative.removeprefix("licenses/")
            require(committed.get(original) == raw, f"Wheel license differs: {entry}")
        else:
            require(relative in GENERATED_WHEEL, f"Unexpected wheel metadata: {entry}")
    result = {
        "status": "passed",
        "revision": resolved,
        "version": version,
        "source": identity(source),
        "wheel": identity(wheel),
        "committed_source_files": source_count,
        "runtime_files": runtime_count,
    }
    if installed is not None:
        result["installed"] = verify_installed(wheel, installed)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--installed", type=Path, help="Installed tsfm_lab package directory"
    )
    args = parser.parse_args()
    try:
        report = verify_release(
            args.source, args.wheel, args.revision, installed=args.installed
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "message": str(exc)}))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
