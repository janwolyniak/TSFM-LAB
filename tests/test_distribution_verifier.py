"""The release verifier must run without test libraries and reject source leakage."""

import json
import os
import runpy
import subprocess
import sys
import venv
import zipfile
from pathlib import Path
from types import SimpleNamespace

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


@pytest.fixture
def installed_probe(tmp_path, monkeypatch):
    definitions = runpy.run_path(str(SCRIPT), run_name="identity_verifier_test")
    environment = tmp_path / "installed-environment"
    package = environment / "lib/site-packages/tsfm_lab"
    package.mkdir(parents=True)
    origin = package / "__init__.py"
    origin.write_text('__version__ = "0.1.0"\n')
    launcher = environment / "bin/tsfm"
    launcher.parent.mkdir()
    launcher.write_text("fixture")
    launcher.chmod(0o755)
    configuration = environment / "pyvenv.cfg"
    configuration.write_text("include-system-site-packages = false\n")
    distribution = SimpleNamespace(
        version="0.1.0",
        read_text=lambda _: None,
        locate_file=lambda name: package.parent / name,
        entry_points=[
            SimpleNamespace(
                group="console_scripts", name="tsfm", value="tsfm_lab.__main__:main"
            )
        ],
    )
    function = definitions["installed_environment"]
    globals_ = function.__globals__
    monkeypatch.setitem(
        globals_,
        "sys",
        SimpleNamespace(
            flags=SimpleNamespace(isolated=1),
            prefix=str(environment),
            base_prefix="/base",
            executable=str(environment / "bin/python"),
            version=sys.version,
        ),
    )
    monkeypatch.setattr(
        globals_["importlib"],
        "import_module",
        lambda _: SimpleNamespace(__file__=str(origin), __version__="0.1.0"),
    )
    monkeypatch.setattr(
        globals_["importlib"].metadata, "distribution", lambda _: distribution
    )
    monkeypatch.setattr(globals_["importlib"].util, "find_spec", lambda _: None)
    monkeypatch.setattr(globals_["shutil"], "which", lambda _: str(launcher))
    return function, configuration, origin, distribution


def test_installed_probe_rejects_system_site_inheritance(installed_probe):
    function, configuration, _, _ = installed_probe
    configuration.write_text("include-system-site-packages = true\n")
    with pytest.raises(RuntimeError, match="without system site packages"):
        function()


@pytest.mark.parametrize(
    "configuration_text",
    ["unrelated = true\n", "include-system-site-packages = invalid\n"],
)
def test_installed_probe_requires_explicit_clean_environment_configuration(
    installed_probe,
    configuration_text,
):
    function, configuration, _, _ = installed_probe
    configuration.write_text(configuration_text)
    with pytest.raises(RuntimeError, match="without system site packages"):
        function()


def test_installed_probe_binds_distribution_origin_and_version(installed_probe):
    function, _, _, distribution = installed_probe
    distribution.version = "0.2.0"
    with pytest.raises(RuntimeError, match="identity differs"):
        function()


def test_wheel_argument_checks_real_installed_bytes(installed_probe, tmp_path):
    function, _, origin, _ = installed_probe
    prefix = "tsfm_lab-0.1.0.dist-info/"
    files = {
        "tsfm_lab/__init__.py": origin.read_bytes(),
        prefix + "METADATA": b"Name: tsfm-lab\nVersion: 0.1.0\n",
        prefix + "WHEEL": b"Wheel-Version: 1.0\n",
        prefix + "RECORD": b"installation rewrites this",
        prefix
        + "entry_points.txt": b"[console_scripts]\ntsfm = tsfm_lab.__main__:main\n",
    }
    wheel = tmp_path / "candidate.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
            path = origin.parent.parent / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    _, evidence = function(wheel)
    assert evidence["wheel_identity"]["runtime_files"] == 1
    origin.write_text('__version__ = "0.1.0"\n# same version, different bytes\n')
    with pytest.raises(ValueError, match="Installed runtime mismatch"):
        function(wheel)


def test_legacy_stage_invokes_isolated_probe_and_checks_identity(tmp_path):
    definitions = runpy.run_path(str(SCRIPT), run_name="legacy_verifier_test")
    verifier = definitions["Verifier"](Path("tsfm"), tmp_path)

    def command(*arguments, python=False):
        assert python and arguments == (
            str(SCRIPT.parents[1] / "tests/fixtures/release_legacy_v1/verify.py"),
        )
        return subprocess.CompletedProcess(
            arguments,
            0,
            json.dumps(
                {
                    "runs": 2,
                    "unchanged_artifacts": 17,
                    "source_revision": "0ef0064ee80c83e90c6d9c927e0f5c5af4ede1a2",
                }
            ),
            "",
        )

    verifier.command = command
    verifier.legacy_records()
    assert verifier.passed == ["historical_schema_one_records_keep_meanings_and_hashes"]
