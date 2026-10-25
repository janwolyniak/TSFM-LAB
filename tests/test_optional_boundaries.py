"""The numerical package remains usable when terminal/Parquet extras are absent."""

import os
import subprocess
import sys
from pathlib import Path


def execute(code):
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, PYTHONPATH=str(root / "src")),
        capture_output=True,
        text=True,
    )


def test_numerical_core_imports_without_optional_modules():
    completed = execute("""
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'typer', 'rich', 'yaml', 'pyarrow', 'torch', 'huggingface_hub'}:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
from tsfm_lab.evaluation.runner import run_experiment
from tsfm_lab.models.baselines import LastValue
""")
    assert completed.returncode == 0, completed.stderr


def test_console_entry_point_explains_missing_cli_extra():
    completed = execute("""
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'typer':
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
from tsfm_lab.__main__ import main
main()
""")
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "tsfm-lab[cli]" in completed.stderr
    assert "Traceback" not in completed.stderr
