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
        if fullname.split('.')[0] in {'typer', 'rich', 'yaml', 'pyarrow', 'torch', 'timesfm', 'huggingface_hub', 'textual', 'prompt_toolkit'}:
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


def test_missing_prompt_editor_is_explained_without_affecting_shell_commands():
    completed = execute("""
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'prompt_toolkit':
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, Block())
from typer.testing import CliRunner
import tsfm_lab.cli as cli
cli.interactive_terminal = lambda: True
failed = CliRunner().invoke(cli.app, ['ui'])
assert failed.exit_code == 2
assert 'tsfm-lab[cli]' in failed.stderr
assert 'Traceback' not in failed.stderr
plain = CliRunner().invoke(cli.app, ['models', 'list', '--json'])
assert plain.exit_code == 0
assert 'last_value' in plain.stdout
""")
    assert completed.returncode == 0, completed.stderr


def test_native_command_session_needs_neither_textual_nor_model_runtimes():
    completed = execute("""
import importlib.abc
import io
import sys
from rich.console import Console
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'textual', 'torch', 'chronos', 'timesfm', 'huggingface_hub'}:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, Block())
from tsfm_lab.repl.session import TerminalSession
class Editor:
    def read(self): return '/exit'
session = TerminalSession(editor=Editor(), console=Console(file=io.StringIO()))
session.run()
assert session.closed
""")
    assert completed.returncode == 0, completed.stderr
