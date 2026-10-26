"""Real saved forecasts exercise the scriptable, offline derived-report workflow."""

import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
from finance_helpers import price_fixture
from typer.testing import CliRunner

from tsfm_lab.cli import app
from tsfm_lab.finance.reports import load_report

runner = CliRunner()


def arguments(value, destination):
    args = [
        "finance",
        "backtest",
        value.run_id,
        "--workspace",
        str(value.workspace),
        "--source-config",
        str(value.source_config),
        "--ohlc",
        str(value.ohlc_path),
        "--output",
        str(destination),
        "--price-target",
        "--json",
    ]
    for name, number in asdict(value.rules).items():
        args.extend(["--" + name.replace("_", "-"), str(number)])
    return args


def test_complete_report_show_export_and_no_clobber(tmp_path):
    value = price_fixture(tmp_path / "source")
    path = tmp_path / "reports" / "report.json"
    args = arguments(value, path)
    completed = runner.invoke(app, args)
    assert completed.exit_code == 0, completed.output
    report = json.loads(completed.stdout)
    assert report == load_report(path)
    original = path.read_bytes()
    repeated = runner.invoke(app, args)
    assert repeated.exit_code != 0
    assert path.read_bytes() == original
    # The saved report is independent of source config, dataset, and parent store.
    (tmp_path / "source").rename(tmp_path / "moved-source")
    shown = runner.invoke(app, ["finance", "show", str(path), "--json"])
    assert shown.exit_code == 0, shown.output
    assert json.loads(shown.stdout) == report
    human = runner.invoke(app, ["finance", "show", str(path)])
    assert human.exit_code == 0, human.output
    assert "Hypothetical OHLC backtest" in human.stdout
    output = tmp_path / "equity.csv"
    exported = runner.invoke(
        app, ["finance", "export", str(path), "-o", str(output), "--json"]
    )
    assert exported.exit_code == 0, exported.output
    identity = json.loads(exported.stdout)
    assert identity["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert identity["rows"] == len(rows) == 6
    assert (
        float(rows[-1]["strategy_equity"])
        == report["evaluation"]["strategy"]["metrics"]["final_capital"]
    )
    assert float(rows[-1]["cash_equity"]) == value.rules.initial_capital
    assert (
        float(rows[-1]["buy_and_hold_equity"])
        == report["evaluation"]["benchmarks"]["buy_and_hold"]["metrics"][
            "final_capital"
        ]
    )
    before = output.read_bytes()
    repeated = runner.invoke(app, ["finance", "export", str(path), "-o", str(output)])
    assert repeated.exit_code != 0
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "option",
    [
        "--price-target",
        "--initial-capital",
        "--threshold-bps",
        "--fee-bps",
        "--slippage-bps",
        "--short-financing-bps",
        "--signal-latency-seconds",
    ],
)
def test_every_assumption_must_be_explicit(tmp_path, option):
    value = price_fixture(tmp_path / "source")
    path = tmp_path / "report.json"
    args = arguments(value, path)
    start = args.index(option)
    del args[start : start + (1 if option == "--price-target" else 2)]
    result = runner.invoke(app, args)
    assert result.exit_code == 2, result.output
    assert not path.exists()


@pytest.mark.parametrize(
    "name,number",
    [
        ("--fee-bps", "-1"),
        ("--signal-latency-seconds", "nan"),
        ("--initial-capital", "0"),
    ],
)
def test_bad_assumptions_fail_without_publishing(tmp_path, name, number):
    value = price_fixture(tmp_path / "source")
    path = tmp_path / "report.json"
    args = arguments(value, path)
    args[args.index(name) + 1] = number
    result = runner.invoke(app, args)
    assert result.exit_code == 2, result.output
    assert not path.exists()


def test_saved_report_needs_no_optional_models_or_yaml_for_python_reload(tmp_path):
    value = price_fixture(tmp_path / "source")
    path = tmp_path / "report.json"
    completed = runner.invoke(app, arguments(value, path))
    assert completed.exit_code == 0, completed.output
    code = """
import importlib.abc, sys
sys.path.insert(0, sys.argv[3])
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','chronos','timesfm','transformers','huggingface_hub','yaml','typer','rich','textual'}:
            raise AssertionError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, Refuse())
from tsfm_lab.finance.reports import load_report
from tsfm_lab.finance.exports import export_equity
value = load_report(sys.argv[1])
assert value['evaluation']['cases'] == 6
export_equity(value, sys.argv[2])
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(path),
            str(tmp_path / "offline.csv"),
            str(Path(__file__).resolve().parents[1] / "src"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
