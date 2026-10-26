"""Build and inspect a source archive and its wheel without dependency downloads."""

import argparse
import configparser
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from contextlib import redirect_stdout
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

CONNECTION_MODULES = {
    f"tsfm_lab/connections/{name}.py"
    for name in (
        "__init__",
        "contracts",
        "credentials",
        "identity",
        "manager",
        "oauth",
        "oauth_callback",
        "providers",
        "sign_in",
        "state",
        "streams",
        "transport",
        "ui",
        "vault",
    )
} | {"tsfm_lab/connection_commands.py", "tsfm_lab/worker_entry.py"}
ASSISTANT_MODULES = {
    f"tsfm_lab/assistant/{name}.py"
    for name in (
        "__init__",
        "drafts",
        "resolution",
        "evidence",
        "protocol",
        "service",
        "ui",
    )
}

SOURCE_FILES = {
    "LICENSE",
    "NOTICE",
    "licenses/mlstm-kernels-2.0.6.txt",
    "README.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "MANIFEST.in",
    "pyproject.toml",
    "requirements-dev.txt",
    "requirements-tui.txt",
    "requirements-chronos.txt",
    "requirements-timesfm.txt",
    "requirements-tirex2.txt",
    "requirements-ttm.txt",
    "docs/installation.md",
    "docs/releasing.md",
    "docs/adapters.md",
    "docs/evaluation.md",
    "docs/timesfm.md",
    "docs/chronos2.md",
    "docs/p2-chronos2-verification.md",
    "docs/tirex2.md",
    "docs/ttm.md",
    "docs/ttm-contract.md",
    "docs/p4-ttm-verification.md",
    "docs/usage-profiles.md",
    "docs/p5-profiles-verification.md",
    "docs/connections.md",
    "docs/p6-connections-verification.md",
    "src/tsfm_lab/connections/manager.py",
    "src/tsfm_lab/connections/oauth.py",
    "src/tsfm_lab/connections/transport.py",
    "src/tsfm_lab/connection_commands.py",
    "docs/p3-tirex2-verification.md",
    "docs/finance-extension.md",
    "docs/model-catalog.md",
    "docs/next-release-plan.md",
    "examples/baseline.py",
    "examples/custom_adapter.py",
    "examples/datasets/README.md",
    "examples/datasets/seasonal.csv",
    "examples/datasets/ohlc.csv",
    "examples/datasets/ttm-seasonal.csv",
    "examples/configurations/last-value.yaml",
    "examples/configurations/seasonal-naive.yaml",
    "examples/configurations/chronos-bolt-tiny.yaml",
    "examples/configurations/chronos-bolt-base.yaml",
    "examples/configurations/chronos2-small.yaml",
    "examples/configurations/timesfm-2p5.yaml",
    "examples/configurations/tirex2-main.yaml",
    "examples/configurations/ttm-512-96.yaml",
    "examples/configurations/finance-seasonal-naive.yaml",
    "tests/conftest.py",
    "src/tsfm_lab/models/chronos2_profile.py",
    "src/tsfm_lab/models/tirex2.py",
    "src/tsfm_lab/models/tirex2_profile.py",
    "src/tsfm_lab/models/ttm.py",
    "src/tsfm_lab/models/ttm_profile.py",
    "src/tsfm_lab/point_tasks.py",
    "src/tsfm_lab/workflows.py",
    "src/tsfm_lab/workflow_views.py",
    "scripts/build_distribution.py",
    "scripts/verify_distribution.py",
    "research/fev_compatibility/README.md",
    "research/fev_compatibility/run_study.py",
    "research/fev_compatibility/results.json",
}
SOURCE_FILES |= (
    {
        f"examples/datasets/{profile}-demo.csv"
        for profile in ("finance", "energy", "temperature", "weather")
    }
    | {
        f"examples/configurations/usage-{profile}.yaml"
        for profile in ("generic", "finance", "energy", "temperature", "weather")
    }
    | {"examples/datasets/finance-demo-ohlc.csv"}
)
SOURCE_FILES |= {f"src/{name}" for name in CONNECTION_MODULES}
SOURCE_FILES |= {f"src/{name}" for name in ASSISTANT_MODULES}
SOURCE_FILES |= {"docs/language-assistance.md", "docs/p7-language-verification.md"}


def _allowed(names):
    for name in names:
        parts = PurePosixPath(name).parts
        if PurePosixPath(name).is_absolute() or ".." in parts or "\\" in name:
            raise ValueError(f"Unsafe path in distribution: {name}")
        if any(
            part in {".git", ".venv", "venv", "model-cache", "results", "__pycache__"}
            for part in parts
        ):
            raise ValueError(f"Unexpected local artifact in distribution: {name}")
        if Path(name).name == "PLAN.md" or Path(name).suffix in {
            ".safetensors",
            ".pyc",
            ".pyo",
        }:
            raise ValueError(f"Unexpected local artifact in distribution: {name}")


def inspect_source(path, *, name, version):
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        roots = {member.name.split("/")[0] for member in members}
        expected = f"{name.replace('-', '_')}-{version}"
        if roots != {expected}:
            raise ValueError("Source archive has an unexpected root directory")
        _allowed({member.name[len(expected) + 1 :] for member in members})
        names = {
            member.name[len(expected) + 1 :] for member in members if member.isfile()
        }
        if missing := SOURCE_FILES - names:
            raise ValueError(f"Source archive is incomplete: {sorted(missing)}")
        if any(member.issym() or member.islnk() for member in members):
            raise ValueError("Source archive must not contain links")
        _allowed(names)
        return {"files": len(names), "required_files": len(SOURCE_FILES)}


def inspect_wheel(path, *, name, version):
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        _allowed(names)
        required = {
            "tsfm_lab/resources/seasonal.csv",
            "tsfm_lab/resources/ttm-seasonal.csv",
            "tsfm_lab/point_tasks.py",
            "tsfm_lab/workflows.py",
            "tsfm_lab/workflow_views.py",
            "tsfm_lab/models/ttm.py",
            "tsfm_lab/models/ttm_profile.py",
            "tsfm_lab/repl/session.py",
            "tsfm_lab/terminal.py",
            "tsfm_lab/finance_commands.py",
            "tsfm_lab/finance/backtest.py",
            "tsfm_lab/finance/reports.py",
            "tsfm_lab/finance/ledger.py",
            "tsfm_lab/finance/ohlc.py",
            "tsfm_lab/finance/exports.py",
            "tsfm_lab/models/registry.py",
            "tsfm_lab/models/chronos2.py",
            "tsfm_lab/models/chronos2_profile.py",
            "tsfm_lab/models/tirex2.py",
            "tsfm_lab/models/tirex2_profile.py",
            "tsfm_lab/models/verification.py",
            "tsfm_lab/models/_verification_evidence.py",
        }
        required |= {
            f"tsfm_lab/resources/{profile}-demo.csv"
            for profile in ("finance", "energy", "temperature", "weather")
        }
        required |= CONNECTION_MODULES
        required |= ASSISTANT_MODULES
        if "tsfm_lab/tui/app.py" in names:
            required.add("tsfm_lab/tui/app.tcss")
        if not required <= names:
            raise ValueError("Wheel is missing runtime modules or the bundled demo")
        metadata_files = [
            item for item in names if item.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ValueError("Wheel must contain exactly one project metadata file")
        metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
        if metadata["Name"] != name or metadata["Version"] != version:
            raise ValueError("Wheel name/version differs from pyproject.toml")
        if (
            metadata["License-Expression"] != "Apache-2.0"
            or metadata["Requires-Python"] != ">=3.11"
        ):
            raise ValueError(
                "Wheel license/Python metadata differs from the supported contract"
            )
        prefix = metadata_files[0].rsplit("/", 1)[0]
        license_files = {"LICENSE", "NOTICE", "licenses/mlstm-kernels-2.0.6.txt"}
        if not license_files <= set(metadata.get_all("License-File", [])) or any(
            f"{prefix}/licenses/{file}" not in names for file in license_files
        ):
            raise ValueError(
                "Wheel does not include its declared project/runtime license notices"
            )
        entry = configparser.ConfigParser()
        entry.read_string(archive.read(f"{prefix}/entry_points.txt").decode())
        if (
            entry.get("console_scripts", "tsfm", fallback=None)
            != "tsfm_lab.__main__:main"
        ):
            raise ValueError("Wheel does not expose the tsfm command")
        dependencies = metadata.get_all("Requires-Dist", [])
        core = set()
        for item in dependencies:
            requirement, _, marker = item.partition(";")
            # Only the backend's exact extra selector establishes optionality.
            # Other/compound markers remain core for this conservative audit.
            if re.fullmatch(r"""\s*extra\s*==\s*(["'])[A-Za-z0-9._-]+\1\s*""", marker):
                continue
            match = re.match(r"[A-Za-z0-9._-]+", requirement)
            if not match:
                raise ValueError("Invalid dependency name in wheel metadata")
            core.add(re.sub(r"[-_.]+", "-", match.group()).lower())
        if core != {"numpy", "pandas"}:
            raise ValueError("Optional dependencies leaked into the numerical core")
        return {
            "files": len(names),
            "console_script": "tsfm",
            "license": metadata["License-Expression"],
            "license_files": sorted(metadata.get_all("License-File", [])),
            "core_dependencies": sorted(core),
        }


def _identity(path):
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _source_state(root):
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        if Path(top.stdout.strip()).resolve() != root:
            return {"revision": None, "dirty": None}
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return {
            "revision": revision.stdout.strip(),
            "dirty": bool(status.stdout.strip()),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def build(out_dir: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    configuration = tomllib.loads((root / "pyproject.toml").read_text())
    name, version = (
        configuration["project"]["name"],
        configuration["project"]["version"],
    )
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if list(out_dir.iterdir()):
        raise ValueError(
            "Use an empty output directory; existing release artifacts are preserved"
        )
    # The backend has no build requirements beyond the declared setuptools>=77.
    command = "from setuptools.build_meta import build_sdist; import sys; build_sdist(sys.argv[1])"
    subprocess.run(
        [sys.executable, "-c", command, str(out_dir)],
        cwd=root,
        stdout=sys.stderr,
        check=True,
    )
    sources = list(out_dir.glob("*.tar.gz"))
    if len(sources) != 1:
        raise ValueError("Expected exactly one source archive")
    source = sources[0]
    source_checks = inspect_source(source, name=name, version=version)
    wheel_dir = out_dir / "wheels"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "--wheel-dir",
            str(wheel_dir),
            str(source),
        ],
        cwd=out_dir,
        stdout=sys.stderr,
        check=True,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected exactly one wheel built from the source archive")
    wheel = wheels[0]
    wheel_checks = inspect_wheel(wheel, name=name, version=version)
    return {
        "status": "passed",
        "source_state": _source_state(root),
        "built_from_source_archive": True,
        "source": _identity(source),
        "wheel": _identity(wheel),
        "source_checks": source_checks,
        "wheel_checks": wheel_checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            result = build(args.out_dir)
    except (
        OSError,
        ValueError,
        subprocess.CalledProcessError,
        tarfile.TarError,
        zipfile.BadZipFile,
        KeyError,
    ) as exc:
        parser.exit(1, f"Distribution build failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
