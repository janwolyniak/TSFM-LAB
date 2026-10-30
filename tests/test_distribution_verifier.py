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
    return environment, environment / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )


def invoke(interpreter, *, isolated=True, cwd=None):
    arguments = [str(interpreter), *(("-I",) if isolated else ()), str(SCRIPT)]
    result = subprocess.run(
        arguments,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
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
        encoding="utf-8",
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
        encoding="utf-8",
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
    launcher = tmp_path / "fixture_launcher.py"
    # The test explicitly selects source; the installed verifier itself forbids this.
    launcher.write_text(
        definitions["_BLOCK_OPTIONAL"]
        + f"\nsys.path.insert(0, {str(source)!r})\n"
        + "import runpy\nrunpy.run_module('tsfm_lab', run_name='__main__')\n"
    )
    verifier = definitions["Verifier"](launcher, tmp_path)
    real_command = verifier.command

    def fixture_command(*arguments, expected=0):
        return real_command(str(launcher), *arguments, expected=expected, python=True)

    verifier.command = fixture_command
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
    binaries = environment / ("Scripts" if os.name == "nt" else "bin")
    launcher = binaries / ("tsfm.exe" if os.name == "nt" else "tsfm")
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
            executable=str(binaries / ("python.exe" if os.name == "nt" else "python")),
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


@pytest.fixture
def source_verifier(tmp_path):
    """Exercise verifier stages against source without relaxing production identity."""
    definitions = runpy.run_path(str(SCRIPT), run_name="stage_verifier_test")
    source = SCRIPT.parents[1] / "src"
    launcher = tmp_path / "fixture_launcher.py"
    launcher.write_text(
        definitions["_BLOCK_OPTIONAL"]
        + f"\nsys.path.insert(0, {str(source)!r})\n"
        + "import runpy\nrunpy.run_module('tsfm_lab', run_name='__main__')\n",
        encoding="utf-8",
    )
    verifier = definitions["Verifier"](launcher, tmp_path)
    command = verifier.command

    def fixture_command(*arguments, expected=0, python=False):
        if python:
            assert arguments[0] == "-c"
            probe = f"import sys; sys.path.insert(0, {str(source)!r})\n" + arguments[1]
            return command("-c", probe, expected=expected, python=True)
        return command(str(launcher), *arguments, expected=expected, python=True)

    verifier.command = fixture_command
    return verifier


def test_installed_probe_requires_exact_wheel_even_with_matching_version(
    installed_probe,
):
    function, _, _, _ = installed_probe
    with pytest.raises(RuntimeError, match="Supply --wheel.*exact built wheel"):
        function()


def test_subprocess_text_is_explicitly_utf8(tmp_path, monkeypatch):
    definitions = runpy.run_path(str(SCRIPT), run_name="encoding_verifier_test")
    verifier = definitions["Verifier"](Path("tsfm"), tmp_path)
    received = []

    def run(arguments, **kwargs):
        received.append(kwargs)
        return subprocess.CompletedProcess(arguments, 0, "Łódź", "")

    monkeypatch.setattr(subprocess, "run", run)
    assert verifier.command("--version").stdout == "Łódź"
    assert received[0]["text"] is True and received[0]["encoding"] == "utf-8"


def test_optional_blocker_covers_new_families_and_kernel_modules():
    definitions = runpy.run_path(str(SCRIPT), run_name="block_verifier_test")
    blocked = set(definitions["_OPTIONAL"])
    assert {
        "toto2",
        "gluonts",
        "dd_unit_scaling",
        "unit_scaling",
        "triton",
        "triton_windows",
        "flashrnn",
        "mlstm_kernels",
        "lightning",
        "pytorch_lightning",
    } <= blocked
    probe = definitions["_BLOCK_OPTIONAL"] + """
from importlib import import_module
for name in blocked:
    try:
        import_module(name + '.unavailable_child')
    except ModuleNotFoundError as exc:
        assert exc.name == name
    else:
        raise AssertionError(name)
assert not blocked.intersection(sys.modules)
print(json.dumps(sorted(blocked)))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
        check=True,
    )
    assert set(json.loads(result.stdout)) == blocked


def test_installed_catalog_exact_profiles_and_metadata(source_verifier):
    source_verifier.catalog()
    assert source_verifier.passed == ["offline_catalog_without_checkpoints"]
    assert not (source_verifier.root / "empty-cache").exists()


@pytest.mark.parametrize(
    "mutation", ["missing_toto", "unexpected_ttm", "wrong_cadence"]
)
def test_catalog_refuses_missing_extra_or_drifted_profiles(tmp_path, mutation):
    from tsfm_lab.models.catalog import catalog_entries, catalog_info

    definitions = runpy.run_path(str(SCRIPT), run_name="catalog_verifier_test")
    verifier = definitions["Verifier"](Path("tsfm"), tmp_path)

    def metadata(*arguments):
        if arguments == ("models", "list"):
            entries = catalog_entries()
            if mutation == "missing_toto":
                return [entry for entry in entries if entry["adapter_id"] != "toto_2"]
            if mutation == "unexpected_ttm":
                next(entry for entry in entries if entry["adapter_id"] == "ttm_r2")[
                    "variants"
                ].append({"variant": "unreviewed"})
            return entries
        if arguments[:2] == ("models", "cache"):
            return []
        identifier = arguments[2]
        variant = arguments[4] if len(arguments) == 5 else None
        value = catalog_info(identifier, variant)
        if "repository" in value:
            value["selected_checkpoint"] = dict(value)
        if mutation == "wrong_cadence" and variant == "90-30-ft-r2.1":
            value["frequency_policy"] = "any_fixed"
        return value

    verifier.json_command = metadata
    with pytest.raises(RuntimeError, match="catalog|TTM"):
        verifier.catalog()


def test_seven_fake_routes_use_real_protocols_without_optional_dependencies(
    source_verifier,
):
    source_verifier.connections()
    assert source_verifier.passed == [
        "seven_fake_connections_drafts_and_explanations_without_provider_dependencies"
    ]
    state = json.loads(
        (source_verifier.root / "connection-metadata/connections.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["selected"] is None
    assert len(state["profiles"]) == 7
    assert all(
        profile["credential_ref"] is None and profile["model"] is None
        for profile in state["profiles"]
    )


def test_installed_onboarding_and_guided_baselines_detach_future_records(
    source_verifier,
):
    pytest.importorskip("pyarrow")
    dependencies = subprocess.run(
        [sys.executable, "-I", "-c", "import prompt_toolkit, rich, yaml, pyarrow"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
        check=False,
    )
    if dependencies.returncode:
        pytest.skip("Isolated source fixture requires installed [cli,parquet] extras")
    # Explicit disposable source; no installed resource lookup or user settings.
    source = SCRIPT.parents[1] / "src/tsfm_lab/resources/seasonal.csv"
    source_verifier.onboarding_and_future(source)
    assert source_verifier.passed == [
        "onboarding_preferences_guided_historical_and_future_baselines_detached_exports"
    ]
    assert (source_verifier.root / "future-input-unavailable.csv").is_file()
    assert not (source_verifier.root / "future-input.csv").exists()
    assert (source_verifier.root / "detached-future.parquet").is_file()
    assert (
        len(
            json.loads(
                (source_verifier.root / "detached-future.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        == 4
    )
