"""Distribution boundary regressions without executing or installing artifacts."""

import runpy
import tarfile
import zipfile
from pathlib import Path

import pytest

BUILDER = runpy.run_path(
    str(Path(__file__).parents[1] / "scripts/build_distribution.py")
)


@pytest.mark.parametrize(
    "name", ["safe/../../outside.txt", "/absolute.txt", "safe\\outside.txt"]
)
def test_archive_paths_cannot_escape_the_distribution(name):
    with pytest.raises(ValueError, match="Unsafe path"):
        BUILDER["_allowed"]({name})


def test_directory_members_are_checked_before_extracting_a_source_archive(tmp_path):
    path = tmp_path / "source.tar.gz"
    root = "tsfm_lab-0.1.0"
    with tarfile.open(path, "w:gz") as archive:
        for name in BUILDER["SOURCE_FILES"]:
            archive.addfile(tarfile.TarInfo(f"{root}/{name}"))
        directory = tarfile.TarInfo(f"{root}/../../outside")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
    with pytest.raises(ValueError, match="Unsafe path"):
        BUILDER["inspect_source"](path, name="tsfm-lab", version="0.1.0")


def wheel(tmp_path, dependency):
    path = tmp_path / "test.whl"
    prefix = "tsfm_lab-0.1.0.dist-info"
    metadata = (
        "Metadata-Version: 2.4\nName: tsfm-lab\nVersion: 0.1.0\n"
        "Requires-Python: >=3.11\nLicense-Expression: Apache-2.0\n"
        f"Requires-Dist: numpy>=1.26\nRequires-Dist: pandas>=2.2\nRequires-Dist: {dependency}\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        for name in (
            "tsfm_lab/resources/seasonal.csv",
            "tsfm_lab/repl/session.py",
            "tsfm_lab/terminal.py",
            "tsfm_lab/finance_commands.py",
            "tsfm_lab/finance/backtest.py",
            "tsfm_lab/finance/reports.py",
            "tsfm_lab/finance/ledger.py",
            "tsfm_lab/finance/ohlc.py",
            "tsfm_lab/finance/exports.py",
            f"{prefix}/licenses/LICENSE",
        ):
            archive.writestr(name, "fixture")
        archive.writestr(f"{prefix}/METADATA", metadata)
        archive.writestr(
            f"{prefix}/entry_points.txt",
            "[console_scripts]\ntsfm = tsfm_lab.__main__:main\n",
        )
    return path


@pytest.mark.parametrize(
    "dependency",
    [
        'torch; sys_platform == "darwin"',
        'torch; sys_platform == "linux" or extra == "chronos"',
    ],
)
def test_conditional_core_runtime_cannot_be_disguised_as_an_extra(tmp_path, dependency):
    with pytest.raises(ValueError, match="Optional dependencies leaked"):
        BUILDER["inspect_wheel"](
            wheel(tmp_path, dependency), name="tsfm-lab", version="0.1.0"
        )


def test_declared_extra_does_not_expand_the_numerical_core(tmp_path):
    result = BUILDER["inspect_wheel"](
        wheel(tmp_path, 'torch; extra == "chronos"'), name="tsfm-lab", version="0.1.0"
    )
    assert result["core_dependencies"] == ["numpy", "pandas"]


def test_build_preserves_preexisting_output_files(tmp_path):
    output = tmp_path / "user.txt"
    output.write_text("keep")
    with pytest.raises(ValueError, match="empty output"):
        BUILDER["build"](tmp_path)
    assert output.read_text() == "keep"
