import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from tsfm_lab.application import run_from_config
from tsfm_lab.cli import app

runner = CliRunner()


@pytest.mark.parametrize(
    "arguments", [[], ["--help"], ["--version"], ["models", "list"]]
)
def test_basic_commands_are_noninteractive(arguments):
    result = runner.invoke(app, arguments, input="")
    assert result.exit_code == 0, result.output


def test_model_json_is_plain_and_limited_to_implemented_adapters():
    result = runner.invoke(app, ["models", "list", "--json"])
    assert result.exit_code == 0
    assert {m["adapter_id"] for m in json.loads(result.stdout)} == {
        "last_value",
        "seasonal_naive",
        "chronos_bolt",
        "timesfm_2p5",
    }
    assert "\x1b[" not in result.stdout


def test_run_history_show_compare_work_in_json(config_factory, tmp_path):
    ids = []
    for adapter in ("last_value", "seasonal_naive"):
        result = runner.invoke(
            app, ["run", str(config_factory(adapter)), "--json"], input=""
        )
        assert result.exit_code == 0, result.output
        value = json.loads(result.stdout)
        assert value["status"] == "completed"
        assert "Stage:" not in result.stdout
        assert "Stage: saving" in result.stderr
        ids.append(value["run_id"])
    workspace = ["--workspace", str(tmp_path / "workspace"), "--json"]
    history = runner.invoke(app, ["experiments", "list", *workspace])
    assert len(json.loads(history.stdout)) == 2
    show = runner.invoke(app, ["experiments", "show", ids[0], *workspace])
    assert show.exit_code == 0, show.output
    assert (
        json.loads(show.stdout)["result"]["metrics"]["overall"]["mae"]["value"] == 2.5
    )
    comparison = runner.invoke(app, ["compare", *ids, *workspace])
    assert comparison.exit_code == 0, comparison.output
    assert json.loads(comparison.stdout)["compatible"] is True


def test_bad_configuration_has_exit_two_and_no_result(config_factory, tmp_path):
    result = runner.invoke(
        app, ["run", str(config_factory(evaluation={"horizon": 100})), "--json"]
    )
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Error:" in result.stderr
    assert not (tmp_path / "workspace").exists()


def test_incompatible_comparison_has_exit_two(config_factory, tmp_path):
    a = run_from_config(config_factory())
    b = run_from_config(config_factory(evaluation={"context_length": 9}))
    result = runner.invoke(
        app,
        [
            "compare",
            a["run_id"],
            b["run_id"],
            "--workspace",
            str(tmp_path / "workspace"),
            "--json",
        ],
    )
    assert result.exit_code == 2
    assert result.stdout == ""


def test_corrupt_saved_run_has_exit_one(config_factory, tmp_path):
    value = run_from_config(config_factory())
    path = tmp_path / "workspace" / "experiments" / value["run_id"] / "predictions.csv"
    path.write_text("corrupt")
    result = runner.invoke(
        app,
        [
            "experiments",
            "show",
            value["run_id"],
            "--workspace",
            str(tmp_path / "workspace"),
            "--json",
        ],
    )
    assert result.exit_code == 1
    assert result.stdout == ""


def test_dataset_inspection_json(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text(
        "timestamp,target\n2026-01-01T00:00:00Z,1\n2026-01-01T01:00:00Z,2\n"
    )
    result = runner.invoke(
        app,
        [
            "datasets",
            "inspect",
            str(source),
            "--timestamp-column",
            "timestamp",
            "--target-column",
            "target",
            "--frequency",
            "1h",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["observations"] == 2


@pytest.mark.parametrize("format", ["csv", "json", "parquet"])
def test_export_reloads_forecasts_without_inference(
    config_factory, tmp_path, monkeypatch, format
):
    value = run_from_config(config_factory())

    def fail(*args, **kwargs):
        raise AssertionError("export must not run inference")

    monkeypatch.setattr("tsfm_lab.cli.run_from_config", fail)
    target = tmp_path / f"export.{format}"
    result = runner.invoke(
        app,
        [
            "export",
            value["run_id"],
            "--workspace",
            str(tmp_path / "workspace"),
            "--output",
            str(target),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["rows"] == 16
    if format == "csv":
        assert len(pd.read_csv(target)) == 16
    elif format == "parquet":
        assert len(pd.read_parquet(target)) == 16
    else:
        assert len(json.loads(target.read_text())) == 16


def test_separate_process_can_reload_a_saved_experiment(config_factory, tmp_path):
    value = run_from_config(config_factory())
    environment = dict(
        os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src")
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tsfm_lab",
            "experiments",
            "show",
            value["run_id"],
            "--workspace",
            str(tmp_path / "workspace"),
            "--json",
        ],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout)["manifest"]["status"] == "completed"
    assert completed.stderr == ""


def test_empty_history_does_not_create_a_workspace(tmp_path):
    result = runner.invoke(
        app, ["experiments", "list", "--workspace", str(tmp_path / "missing"), "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []
    assert not (tmp_path / "missing").exists()


def test_invalid_export_format_is_rejected_before_reading_a_run(tmp_path):
    result = runner.invoke(
        app,
        [
            "export",
            "not-a-uuid",
            "--output",
            str(tmp_path / "new-dir" / "out.xml"),
            "--format",
            "xml",
            "--json",
        ],
    )
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Export format" in result.stderr
    assert not (tmp_path / "new-dir").exists()


def test_existing_export_destination_is_preserved(config_factory, tmp_path):
    value = run_from_config(config_factory())
    target = tmp_path / "existing.csv"
    target.write_text("existing user content")
    result = runner.invoke(
        app,
        [
            "export",
            value["run_id"],
            "--output",
            str(target),
            "--workspace",
            str(tmp_path / "workspace"),
            "--json",
        ],
    )
    assert result.exit_code == 1
    assert target.read_text() == "existing user content"


@pytest.mark.parametrize(
    "change",
    [
        {"dataset": {"frequency": "100000000000000000000000h"}},
        {"output": {"directory": "~tsfmlab-no-such-user-928374/results"}},
    ],
)
def test_path_or_frequency_overflow_is_a_configuration_error(config_factory, change):
    result = runner.invoke(app, ["run", str(config_factory(**change)), "--json"])
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Error:" in result.stderr
