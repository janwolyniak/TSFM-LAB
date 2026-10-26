"""The release verifier must run without test libraries and reject source leakage."""

import json
import os
import runpy
import subprocess
import sys
import venv
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_distribution.py"


@pytest.fixture
def bare_environment(tmp_path):
    environment = tmp_path / "bare-environment"
    venv.EnvBuilder(with_pip=False).create(environment)
    return environment, environment / "bin" / "python"


def invoke(interpreter, *, isolated=True, cwd=None):
    arguments = [str(interpreter), *(("-I",) if isolated else ()), str(SCRIPT)]
    result = subprocess.run(
        arguments,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "failed"
    assert report["check"] == "installed_environment"
    return report


def test_verifier_definition_import_needs_only_standard_library():
    code = "import runpy,sys; runpy.run_path(sys.argv[1],run_name='verifier_import'); print('stdlib-only')"
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", code, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "stdlib-only"


def test_missing_wheel_reports_actionable_json_without_creating_workspace(
    bare_environment, tmp_path
):
    _, interpreter = bare_environment
    before = set(tmp_path.iterdir())
    report = invoke(interpreter, cwd=tmp_path)
    assert "built wheel" in report["message"]
    assert "[cli,parquet]" in report["message"]
    assert set(tmp_path.iterdir()) == before


def test_verifier_requires_python_isolated_mode(bare_environment):
    _, interpreter = bare_environment
    report = invoke(interpreter, isolated=False)
    assert "python -I" in report["message"]


def test_site_pth_cannot_make_source_checkout_pass_as_an_installed_wheel(
    bare_environment,
):
    _, interpreter = bare_environment
    result = subprocess.run(
        [str(interpreter), "-I", "-c", "import site; print(site.getsitepackages()[0])"],
        capture_output=True,
        text=True,
        check=True,
    )
    site_packages = Path(result.stdout.strip())
    source = SCRIPT.parents[1] / "src"
    (site_packages / "source-leak.pth").write_text(str(source) + os.linesep)
    report = invoke(interpreter)
    assert "source-checkout" in report["message"]
    assert Path(report["details"]["module_origin"]).is_relative_to(source)


def test_rich_box_command_help_is_accepted_without_plain_commands_colon(tmp_path):
    verifier_class = runpy.run_path(str(SCRIPT), run_name="verifier_test")["Verifier"]
    verifier = verifier_class(Path("tsfm"), tmp_path)
    help_text = "Usage: tsfm [OPTIONS] COMMAND [ARGS]...\n╭─ Commands ─╮\n│ run models datasets experiments compare export ui │\n╰────────────╯\n"

    def command(*arguments, expected=0):
        if arguments == ("--version",):
            return subprocess.CompletedProcess(arguments, 0, "0.1.0\n", "")
        if arguments == ("ui",):
            assert expected == 2
            return subprocess.CompletedProcess(
                arguments, 2, "", "Interactive UI needs terminal input and output"
            )
        assert arguments in ((), ("--help",))
        return subprocess.CompletedProcess(arguments, 0, help_text, "")

    verifier.command = command
    verifier.entry_points({"version": "0.1.0"})
    assert verifier.passed == ["installed_entry_points_and_redirected_routing"]


def test_finance_stage_runs_real_commands_without_model_imports_or_original_sources(
    tmp_path,
):
    pytest.importorskip("typer")
    definitions = runpy.run_path(str(SCRIPT), run_name="finance_verifier_test")
    source = SCRIPT.parents[1] / "src"
    launcher = tmp_path / "tsfm"
    # The test explicitly selects source; the installed verifier itself forbids this.
    launcher.write_text(
        f"#!{sys.executable}\n"
        + definitions["_BLOCK_OPTIONAL"]
        + f"\nsys.path.insert(0, {str(source)!r})\n"
        + "import runpy\nrunpy.run_module('tsfm_lab', run_name='__main__')\n"
    )
    launcher.chmod(0o755)
    verifier = definitions["Verifier"](launcher, tmp_path)
    verifier.finance()
    assert verifier.passed == [
        "finance_backtest_self_contained_reload_export_and_overwrite_refusal"
    ]
    assert not (tmp_path / "finance-inputs").exists()
    assert (tmp_path / "finance-sources-unavailable/workspace/experiments").is_dir()
    report = json.loads((tmp_path / "finance-report.json").read_text())
    assert report["evaluation"]["cases"] == 4
    assert report["evaluation"]["signal_positions"] == [-1, 1, -1, 0]
    assert (tmp_path / "finance-equity.csv").is_file()
