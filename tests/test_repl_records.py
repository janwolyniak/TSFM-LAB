import io
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import pytest
from rich.console import Console

from tsfm_lab.application import run_from_config
from tsfm_lab.comparison import ComparisonError
from tsfm_lab.contracts import Dataset, EvaluationSpec
from tsfm_lab.evaluation.runner import run_experiment
from tsfm_lab.experiments import ExperimentStore, StoreError
from tsfm_lab.exports import ExportError
from tsfm_lab.models.baselines import LastValue
from tsfm_lab.repl.records import RecordCommands
from tsfm_lab.repl.text import literal


@pytest.fixture
def commands(tmp_path):
    output = io.StringIO()
    console = Console(file=output, width=120, color_system=None, force_terminal=False)
    return RecordCommands(console, tmp_path / "workspace"), output


def test_catalog_is_offline_compact_and_borderless(commands):
    records, output = commands
    entries = records.models()
    assert {entry["adapter_id"] for entry in entries} == {
        "last_value",
        "seasonal_naive",
        "chronos_bolt",
        "timesfm_2p5",
    }
    rendered = output.getvalue()
    assert "apache-2.0" in rendered and "2048" in rendered and "64" in rendered
    assert "cached or local checkpoint" in rendered
    assert not any(symbol in rendered for symbol in ("┌", "┐", "╭", "╮", "│", "\x1b["))


def test_catalog_variant_statuses_remain_readable_in_an_80_column_terminal(tmp_path):
    output = io.StringIO()
    records = RecordCommands(
        Console(file=output, width=80, color_system=None), tmp_path
    )
    records.models()
    rendered = output.getvalue()
    assert "amazon/chronos-bolt-base" in rendered
    assert "individual inference not run" in rendered
    assert "Adapter reference:" in rendered
    assert "Weight files:" in rendered
    assert not any(name in rendered for name in ("Chronos-2", "TiRex", "TTM"))
    assert not any(symbol in rendered for symbol in ("┌", "┐", "╭", "╮", "│", "\x1b["))


def test_empty_history_suggests_demo_without_creating_workspace(commands):
    records, output = commands
    assert records.history() == []
    assert "/demo" in output.getvalue()
    assert not records.workspace.exists()


def test_history_keeps_names_literal_and_reports_status(config_factory, commands):
    records, output = commands
    value = run_from_config(config_factory(name="[red]literal[/red]"))
    history = records.history()
    assert [item["run_id"] for item in history] == [value["run_id"]]
    assert "[red]literal[/red]" in output.getvalue()
    assert "completed" in output.getvalue()


def test_show_verified_task_precedes_scores_and_returns_full_data(
    config_factory, commands
):
    records, output = commands
    value = run_from_config(config_factory())
    shown = records.show(value["run_id"][:8])
    assert shown["result"]["metrics"]["overall"]["mae"]["value"] == 2.5
    assert len(shown["predictions"]) == 16
    rendered = output.getvalue()
    assert rendered.index("Target: demand") < rendered.index("MAE")
    assert (
        "Units: synthetic_units" in rendered and "Horizon: 4 observations" in rendered
    )
    assert "Metrics by forecast lead" in rendered and "first 12 of 16" in rendered
    assert "/export" in rendered


def test_failed_record_shows_error_without_completed_metrics(commands):
    records, output = commands
    store = ExperimentStore(records.workspace)
    run_id = store.create("[bold]failed[/bold]", {}, "")
    store.set_status(
        run_id, "failed", {"message": "synthetic failure", "stage": "loading"}
    )
    shown = records.show(run_id)
    assert set(shown) == {"manifest"}
    assert "synthetic failure (loading)" in output.getvalue()
    assert "[bold]failed[/bold]" in output.getvalue()
    assert "MAE" not in output.getvalue()


def test_corrupt_completed_record_is_not_rendered_as_valid(config_factory, commands):
    records, output = commands
    value = run_from_config(config_factory())
    path = records.workspace / "experiments" / value["run_id"] / "predictions.csv"
    path.write_text("corrupt")
    with pytest.raises(StoreError):
        records.show(value["run_id"])
    assert output.getvalue() == ""
    assert records.history()[0]["status"] == "corrupt"


def test_exact_and_unique_prefix_resolution(config_factory, commands):
    records, _ = commands
    value = run_from_config(config_factory())
    assert records.resolve_id(value["run_id"]) == value["run_id"]
    assert records.resolve_id(value["run_id"].upper()) == value["run_id"]
    assert records.resolve_id(value["run_id"][:8]) == value["run_id"]


def test_ambiguous_prefix_lists_longer_candidate_ids(commands, monkeypatch):
    import tsfm_lab.experiments as storage

    records, _ = commands
    ids = iter(
        [
            UUID("a0000000-0000-4000-8000-000000000001"),
            UUID("a0000000-0000-4000-8000-000000000002"),
        ]
    )
    monkeypatch.setattr(storage, "uuid4", lambda: next(ids))
    store = ExperimentStore(records.workspace)
    first, second = store.create("first", {}, ""), store.create("second", {}, "")
    with pytest.raises(StoreError, match="Ambiguous") as error:
        records.resolve_id("a0000000")
    assert first in str(error.value) and second in str(error.value)
    assert records.resolve_id(first[:-1] + "1") == first


@pytest.mark.parametrize("value", ["", " ", "../outside", "unknown", str(uuid4())])
def test_missing_or_invalid_id_has_actionable_error(commands, value):
    records, _ = commands
    with pytest.raises(StoreError, match="/history"):
        records.resolve_id(value)


def test_latest_selects_verified_completed_not_newer_failed_runs(
    config_factory, commands
):
    records, _ = commands
    completed = run_from_config(config_factory())["run_id"]
    store = ExperimentStore(records.workspace)
    failed = store.create("newer failure", {}, "")
    store.set_status(failed, "failed")
    assert records.resolve_id("latest") == completed


def test_latest_does_not_select_corrupt_completed_record(config_factory, commands):
    records, _ = commands
    older = run_from_config(config_factory())["run_id"]
    newer = run_from_config(config_factory())["run_id"]
    (records.workspace / "experiments" / newer / "metrics.json").write_text("{}")
    assert records.resolve_id("latest") == older


def test_latest_without_completed_results_explains_next_action(commands):
    records, _ = commands
    with pytest.raises(StoreError, match="/demo"):
        records.resolve_id("latest")


def test_prefix_does_not_resolve_non_uuid_workspace_directories(commands):
    records, _ = commands
    (records.workspace / "experiments" / "abcdef").mkdir(parents=True)
    assert records.history()[0]["status"] == "corrupt"
    with pytest.raises(StoreError, match="/history"):
        records.resolve_id("abc")


def test_comparison_uses_shared_policy_and_original_order(config_factory, commands):
    records, output = commands
    first = run_from_config(config_factory())["run_id"]
    second = run_from_config(config_factory("seasonal_naive"))["run_id"]
    compared = records.compare([first[:8], second[:8]])
    assert compared["compatible"]
    assert [item["run_id"] for item in compared["runs"]] == [first, second]
    assert [item["metrics"]["mae"]["value"] for item in compared["runs"]] == [2.5, 1]
    assert output.getvalue().index("Target: demand") < output.getvalue().index("MAE")
    assert "float64" in output.getvalue()


def test_incompatible_comparison_does_not_print_a_controlled_result(
    config_factory, commands
):
    records, output = commands
    first = run_from_config(config_factory())["run_id"]
    second = run_from_config(config_factory(evaluation={"context_length": 9}))["run_id"]
    with pytest.raises(ComparisonError):
        records.compare([first, second])
    assert output.getvalue() == ""


@pytest.mark.parametrize("format", ["csv", "json"])
def test_export_is_shared_verified_and_preserves_existing_files(
    config_factory, commands, tmp_path, format
):
    records, output = commands
    run_id = run_from_config(config_factory())["run_id"]
    destination = tmp_path / f"forecasts.{format}"
    exported = records.export(run_id[:8], destination, format)
    assert exported["rows"] == 16 and str(destination) in output.getvalue()
    if format == "csv":
        assert len(pd.read_csv(destination)) == 16
    else:
        assert len(json.loads(destination.read_text())) == 16
    original = destination.read_bytes()
    with pytest.raises(ExportError, match="already exists"):
        records.export(run_id, destination, format)
    assert destination.read_bytes() == original


def test_record_commands_do_not_run_inference(
    config_factory, commands, tmp_path, monkeypatch
):
    records, _ = commands
    run_id = run_from_config(config_factory())["run_id"]

    def forbidden(*args, **kwargs):
        raise AssertionError("Record commands must not run inference")

    monkeypatch.setattr("tsfm_lab.application.run_experiment", forbidden)
    monkeypatch.setattr("tsfm_lab.models.factory.build_model", forbidden)
    records.models()
    records.history()
    records.show("latest")
    records.export(run_id, tmp_path / "out.csv")


def test_workspace_can_change_without_reusing_previous_history(
    config_factory, commands, tmp_path
):
    records, _ = commands
    first = run_from_config(config_factory())["run_id"]
    second = run_from_config(
        config_factory(output={"directory": "../other-workspace"})
    )["run_id"]
    assert records.resolve_id("latest") == first
    records.workspace = tmp_path / "other-workspace"
    assert records.resolve_id("latest") == second
    assert [item["run_id"] for item in records.history()] == [second]


def test_record_import_and_catalog_need_no_ui_or_model_runtime():
    root = Path(__file__).resolve().parents[1]
    source = """
import importlib.abc,io,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path,target=None):
  if fullname.split('.')[0] in {'textual','torch','chronos','timesfm','huggingface_hub','safetensors','transformers','prompt_toolkit'}:
   raise ModuleNotFoundError(fullname,name=fullname)
sys.meta_path.insert(0,Block())
from rich.console import Console
from tsfm_lab.repl.records import RecordCommands
RecordCommands(Console(file=io.StringIO()),'.').models()
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_literal_preserves_unicode_and_newlines_but_escapes_all_terminal_controls():
    controls = "".join(chr(code) for code in range(0xA0) if code < 0x20 or code >= 0x7F)
    rendered = literal("Zażółć gęślą\n" + controls)
    assert rendered.startswith("Zażółć gęślą\n")
    assert r"\x1b" in rendered and r"\t" in rendered and r"\r" in rendered
    assert r"\x7f" in rendered and r"\x85" in rendered and r"\x9f" in rendered
    assert not any(
        (ord(char) < 0x20 and char != "\n") or 0x7F <= ord(char) <= 0x9F
        for char in rendered
    )
    assert literal(rendered) == rendered


def test_stored_name_and_error_cannot_change_the_terminal_transcript(commands):
    records, output = commands
    store = ExperimentStore(records.workspace)
    run_id = store.create("[red]name\x1b[3J[/red]", {}, "")
    store.set_status(
        run_id, "failed", {"message": "line one\n\x1b]0;title\x07line two\r\t\x85"}
    )
    records.history()
    records.show(run_id)
    rendered = output.getvalue()
    assert "[red]name\\x1b[3J[/red]" in rendered
    assert "line one\n\\x1b]0;title\\x07line two\\r\\t\\x85" in rendered
    assert "\x1b" not in rendered and "\x07" not in rendered and "\x85" not in rendered
    assert "MAE" not in rendered


def test_table_headers_and_values_escape_controls_and_preserve_markup(commands):
    records, output = commands
    records._table(("[bold]Target[/bold]\x1b[2J",), [("[red]value[/red]\t\x7f",)])
    rendered = output.getvalue()
    assert "[bold]Target[/bold]\\x1b[2J" in rendered
    assert "[red]value[/red]\\t\\x7f" in rendered
    assert "\x1b" not in rendered and "\t" not in rendered and "\x7f" not in rendered


def test_narrow_report_folds_nanosecond_timestamps_without_ellipsis(tmp_path):
    output = io.StringIO()
    records = RecordCommands(
        Console(file=output, width=80, color_system=None, force_terminal=False),
        tmp_path,
    )
    timestamps = pd.date_range("2026-01-01T00:00:00.000000001Z", periods=8, freq="1ns")
    result = run_experiment(
        Dataset(timestamps, np.arange(1.0, 9.0), "1ns", units="synthetic_units"),
        LastValue(),
        EvaluationSpec(
            2, 2, timestamps[4], timestamps[-1] + pd.Timedelta(1, unit="ns")
        ),
    )
    store = ExperimentStore(tmp_path)
    run_id = store.create("Nanosecond example", {}, "")
    store.complete(run_id, result.to_dict(), result.predictions)
    records.show(run_id)
    rendered = output.getvalue()
    assert "Target: target" in rendered and "Frequency: 1ns" in rendered
    assert "2026-01-01T00:00:00." in rendered
    for ending in ("000000004", "000000005", "000000008"):
        assert ending in rendered
    assert any(line.strip().split() == ["Z", "Z"] for line in rendered.splitlines())
    assert "…" not in rendered
