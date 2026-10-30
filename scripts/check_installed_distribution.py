"""CI build and clean wheel installation with native paths on every runner.

This entry point installs dependencies into a new disposable environment when
explicitly invoked. Routine tests mock those operations; the verifier itself
never installs dependencies.
"""

import json
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def environment_python(directory, *, windows=None):
    windows = os.name == "nt" if windows is None else windows
    return Path(directory) / ("Scripts/python.exe" if windows else "bin/python")


def _run(arguments, *, environment=None, capture=False):
    try:
        return subprocess.run(
            list(map(str, arguments)),
            cwd=ROOT,
            env=environment,
            check=True,
            text=True,
            encoding="utf-8",
            capture_output=capture,
        )
    except subprocess.CalledProcessError as exc:
        if capture:
            print(exc.stdout or "", file=sys.stderr)
            print(exc.stderr or "", file=sys.stderr)
        raise


def check_installation(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    built = _run(
        [
            sys.executable,
            ROOT / "scripts/build_distribution.py",
            "--out-dir",
            directory / "dist",
        ],
        capture=True,
    )
    artifacts = json.loads(built.stdout)
    (directory / "distributions.json").write_text(built.stdout, encoding="utf-8")
    wheel = Path(artifacts["wheel"]["path"]).resolve()
    environment = directory / "environment"
    venv.EnvBuilder(with_pip=True).create(environment)
    interpreter = environment_python(environment)
    _run(
        [
            interpreter,
            "-m",
            "pip",
            "install",
            "-c",
            ROOT / "requirements-dev.txt",
            f"{wheel}[cli,parquet]",
        ]
    )
    _run([interpreter, "-m", "pip", "check"])
    settings = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    settings["PATH"] = str(interpreter.parent) + os.pathsep + settings.get("PATH", "")
    checked = _run(
        [interpreter, "-I", ROOT / "scripts/verify_distribution.py", "--wheel", wheel],
        environment=settings,
        capture=True,
    )
    (directory / "verification.json").write_text(checked.stdout, encoding="utf-8")
    return {
        "directory": str(directory),
        "artifacts": artifacts,
        "verification": json.loads(checked.stdout),
    }


def main():
    parent = Path(
        tempfile.mkdtemp(prefix="tsfm-ci-", dir=os.environ.get("RUNNER_TEMP"))
    )
    print(json.dumps(check_installation(parent / "qualification"), allow_nan=False))


if __name__ == "__main__":
    main()
