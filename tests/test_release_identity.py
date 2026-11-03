"""Immutable-source and installation identity checks without building or installing."""

import io
import json
import runpy
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_release_identity.py"
IDENTITY = runpy.run_path(str(SCRIPT))
BUILDER = runpy.run_path(str(ROOT / "scripts/build_distribution.py"))
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
PREFIX = f"tsfm_lab-{VERSION}"
DIST_INFO = PREFIX + ".dist-info/"


def write_source(path, files):
    with tarfile.open(path, "w:gz") as archive:
        for name, raw in files.items():
            member = tarfile.TarInfo(f"{PREFIX}/{name}")
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))


def write_wheel(path, files):
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)


@pytest.fixture
def candidate(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    committed = {
        name: (ROOT / name).read_bytes() if (ROOT / name).is_file() else b"fixture"
        for name in BUILDER["SOURCE_FILES"]
    }
    committed.update(
        {
            path.relative_to(ROOT).as_posix(): path.read_bytes()
            for path in (ROOT / "src/tsfm_lab").rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
    )
    committed["docs/extra-guide.md"] = b"tracked guide beyond the required-file list"
    committed["tests/test_additional.py"] = (
        b"# tracked test beyond required-file list\n"
    )
    for name, raw in committed.items():
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    for arguments in (
        ["init", "-q"],
        # This isolated repository has no project attributes. Keep its commit
        # and archive bytes independent of the caller's Windows EOL defaults.
        ["config", "core.autocrlf", "false"],
        ["config", "core.eol", "lf"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "release fixture",
        ],
    ):
        subprocess.run(
            ["git", *arguments], cwd=repository, check=True, capture_output=True
        )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    source_files = dict(committed)
    source_files["setup.cfg"] = b"[egg_info]\ntag_build = \ntag_date = 0\n\n"
    wheel_files = {
        name.removeprefix("src/"): raw
        for name, raw in committed.items()
        if name.startswith("src/tsfm_lab/")
    }
    for name in ("LICENSE", "NOTICE", "licenses/mlstm-kernels-2.0.6.txt"):
        wheel_files[DIST_INFO + "licenses/" + name] = committed[name]
    metadata = (
        f"Metadata-Version: 2.4\nName: tsfm-lab\nVersion: {VERSION}\n"
        "Requires-Python: >=3.11\nLicense-Expression: Apache-2.0\n"
        "License-File: LICENSE\nLicense-File: NOTICE\n"
        "License-File: licenses/mlstm-kernels-2.0.6.txt\n"
    )
    configuration = tomllib.loads(committed["pyproject.toml"].decode())
    for requirement in configuration["project"]["dependencies"]:
        metadata += f"Requires-Dist: {requirement}\n"
    # All optional requirements are represented, including self-referential extras.
    for extra, requirements in configuration["project"][
        "optional-dependencies"
    ].items():
        metadata += f"Provides-Extra: {extra}\n"
        for requirement in requirements:
            name, separator, marker = requirement.partition(";")
            selector = f"({marker.strip()}) and " if separator else ""
            metadata += f'Requires-Dist: {name.strip()}; {selector}extra == "{extra}"\n'
    wheel_files[DIST_INFO + "METADATA"] = metadata.encode()
    source_files["PKG-INFO"] = metadata.encode()
    wheel_files[DIST_INFO + "entry_points.txt"] = (
        b"[console_scripts]\ntsfm = tsfm_lab.__main__:main\n"
    )
    wheel_files[DIST_INFO + "WHEEL"] = b"Wheel-Version: 1.0\n"
    wheel_files[DIST_INFO + "RECORD"] = b"generated record"
    source, wheel = tmp_path / "source.tar.gz", tmp_path / "candidate.whl"
    write_source(source, source_files)
    write_wheel(wheel, wheel_files)
    return repository, revision, source, wheel, source_files, wheel_files


def verify(candidate):
    repository, revision, source, wheel, *_ = candidate
    return IDENTITY["verify_release"](source, wheel, revision, root=repository)


def test_complete_source_wheel_and_installed_bytes_match_commit(candidate, tmp_path):
    _, revision, source, wheel, _, wheel_files = candidate
    report = verify(candidate)
    assert report["status"] == "passed" and report["revision"] == revision
    assert report["source"] == IDENTITY["identity"](source)
    assert report["wheel"] == IDENTITY["identity"](wheel)
    installation = tmp_path / "installation"
    for name, raw in wheel_files.items():
        path = installation / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    package = installation / "tsfm_lab"
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-313.pyc").write_bytes(b"generated bytecode")
    installed = IDENTITY["verify_installed"](wheel, package)
    assert installed["runtime_files"] == report["runtime_files"]
    assert installed["metadata_files"] == len(wheel_files) - report["runtime_files"] - 1


@pytest.mark.parametrize("archive", ["source", "wheel"])
def test_same_version_changed_runtime_cannot_claim_commit_identity(candidate, archive):
    _, _, source, wheel, source_files, wheel_files = candidate
    if archive == "source":
        source_files["src/tsfm_lab/__init__.py"] += b"\n# drift\n"
        write_source(source, source_files)
    else:
        wheel_files["tsfm_lab/__init__.py"] += b"\n# drift\n"
        write_wheel(wheel, wheel_files)
    with pytest.raises(ValueError, match="mismatch.*changed"):
        verify(candidate)


@pytest.mark.parametrize("archive", ["source", "wheel"])
def test_line_ending_changes_do_not_preserve_byte_identity(candidate, archive):
    _, _, source, wheel, source_files, wheel_files = candidate
    files, name, writer, target = (
        (source_files, "src/tsfm_lab/__init__.py", write_source, source)
        if archive == "source"
        else (wheel_files, "tsfm_lab/__init__.py", write_wheel, wheel)
    )
    original = files[name]
    files[name] = original.replace(b"\n", b"\r\n")
    assert files[name] != original
    writer(target, files)
    with pytest.raises(ValueError, match="mismatch.*changed"):
        verify(candidate)


@pytest.mark.parametrize("archive", ["source", "wheel"])
def test_entire_runtime_set_is_required_beyond_builder_allowlist(candidate, archive):
    _, _, source, wheel, source_files, wheel_files = candidate
    if archive == "source":
        del source_files["src/tsfm_lab/__init__.py"]
        write_source(source, source_files)
    else:
        del wheel_files["tsfm_lab/__init__.py"]
        write_wheel(wheel, wheel_files)
    with pytest.raises(ValueError, match="mismatch.*missing"):
        verify(candidate)


@pytest.mark.parametrize("archive", ["source", "wheel"])
def test_uncommitted_files_are_rejected(candidate, archive):
    _, _, source, wheel, source_files, wheel_files = candidate
    if archive == "source":
        source_files["docs/uncommitted.md"] = b"not reviewed"
        write_source(source, source_files)
    else:
        wheel_files["tsfm_lab/uncommitted.py"] = b"not reviewed"
        write_wheel(wheel, wheel_files)
    with pytest.raises(ValueError, match="Uncommitted source|runtime mismatch.*extra"):
        verify(candidate)


def test_source_check_uses_commit_bytes_instead_of_modified_checkout(candidate):
    repository, _, source, _, source_files, _ = candidate
    path = "README.md"
    (repository / path).write_bytes(b"local change")
    assert verify(candidate)["status"] == "passed"
    source_files[path] = b"local change"
    write_source(source, source_files)
    with pytest.raises(ValueError, match="Committed source mismatch"):
        verify(candidate)


def test_generated_setup_configuration_cannot_hide_uncommitted_code(candidate):
    _, _, source, _, source_files, _ = candidate
    source_files["setup.cfg"] += b"[options]\nsetup_requires = unsafe\n"
    write_source(source, source_files)
    with pytest.raises(ValueError, match="generated setup.cfg"):
        verify(candidate)


@pytest.mark.parametrize("name", ["docs/extra-guide.md", "tests/test_additional.py"])
def test_entire_manifest_file_set_is_required(candidate, name):
    _, _, source, _, source_files, _ = candidate
    del source_files[name]
    write_source(source, source_files)
    with pytest.raises(ValueError, match="Committed source mismatch.*missing"):
        verify(candidate)


@pytest.mark.parametrize("archive", ["source", "wheel"])
def test_optional_dependency_tampering_is_rejected(candidate, archive):
    _, _, source, wheel, source_files, wheel_files = candidate
    if archive == "source":
        source_files["PKG-INFO"] = source_files["PKG-INFO"].replace(
            b"httpx==0.28.1", b"httpx>=0.1"
        )
        write_source(source, source_files)
    else:
        wheel_files[DIST_INFO + "METADATA"] = wheel_files[
            DIST_INFO + "METADATA"
        ].replace(b"httpx==0.28.1", b"httpx>=0.1")
        write_wheel(wheel, wheel_files)
    with pytest.raises(ValueError, match="dependency metadata differs"):
        verify(candidate)


@pytest.mark.parametrize("archive", ["source", "wheel"])
def test_extra_entrypoint_cannot_hide_in_generated_metadata(candidate, archive):
    _, _, source, wheel, source_files, wheel_files = candidate
    entry = b"[console_scripts]\ntsfm = tsfm_lab.__main__:main\nother = evil:main\n"
    if archive == "source":
        source_files["src/tsfm_lab.egg-info/entry_points.txt"] = entry
        write_source(source, source_files)
    else:
        wheel_files[DIST_INFO + "entry_points.txt"] = entry
        write_wheel(wheel, wheel_files)
    with pytest.raises(ValueError, match="entry point metadata differs"):
        verify(candidate)


@pytest.mark.parametrize("change", ["missing", "duplicate", "other-project"])
def test_installed_check_rejects_incomplete_or_wrong_wheel_metadata(
    candidate, tmp_path, change
):
    _, _, _, wheel, _, wheel_files = candidate
    package = tmp_path / "installation" / "tsfm_lab"
    package.mkdir(parents=True)
    if change == "missing":
        wheel_files = {
            name: raw for name, raw in wheel_files.items() if ".dist-info/" not in name
        }
    elif change == "duplicate":
        wheel_files["other-1.0.dist-info/METADATA"] = b"Name: other\nVersion: 1.0\n"
    else:
        wheel_files[DIST_INFO + "METADATA"] = wheel_files[
            DIST_INFO + "METADATA"
        ].replace(b"Name: tsfm-lab", b"Name: another-project")
    write_wheel(wheel, wheel_files)
    with pytest.raises(ValueError, match="metadata directory|versioned tsfm-lab"):
        IDENTITY["verify_installed"](wheel, package)


@pytest.mark.parametrize("change", ["changed", "missing", "extra", "metadata"])
def test_installed_byte_drift_or_extra_runtime_is_rejected(candidate, tmp_path, change):
    _, _, _, wheel, _, wheel_files = candidate
    installation = tmp_path / "installed"
    for name, raw in wheel_files.items():
        path = installation / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    package = installation / "tsfm_lab"
    if change == "changed":
        (package / "__init__.py").write_bytes(b"same version, other code")
    elif change == "missing":
        (package / "__init__.py").unlink()
    elif change == "extra":
        (package / "additional.py").write_bytes(b"unexpected")
    else:
        (installation / (DIST_INFO + "METADATA")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Installed runtime mismatch|metadata differs"):
        IDENTITY["verify_installed"](wheel, package)


def test_standalone_installed_check_rejects_foreign_wheel_files(candidate, tmp_path):
    _, _, _, wheel, _, wheel_files = candidate
    wheel_files["sitecustomize.py"] = b"unrelated startup code"
    write_wheel(wheel, wheel_files)
    package = tmp_path / "installation/tsfm_lab"
    package.mkdir(parents=True)
    with pytest.raises(ValueError, match="outside the supported package"):
        IDENTITY["verify_installed"](wheel, package)


def test_failed_cli_reports_json_without_traceback(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(SCRIPT),
            "--source",
            str(tmp_path / "missing"),
            "--wheel",
            str(tmp_path / "missing.whl"),
            "--revision",
            "does-not-exist",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1 and "Traceback" not in result.stderr
    assert json.loads(result.stdout)["status"] == "failed"
