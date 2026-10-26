"""Profiles explain checked forecasts without rewriting records or running models."""

import asyncio
import copy
import io
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from rich.console import Console
from typer.testing import CliRunner

from tsfm_lab.application import run_from_config
from tsfm_lab.cli import app
from tsfm_lab.comparison import compare_runs
from tsfm_lab.contracts import Dataset, EvaluationSpec
from tsfm_lab.evaluation.runner import run_experiment
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.models.baselines import LastValue
from tsfm_lab.repl.records import RecordCommands
from tsfm_lab.workflow_views import history_profile, ohlc_handoff, profile_context

PROFILES = [
    ("generic", {}, "Generic"),
    ("finance", {"target_kind": "close_price"}, "Finance · close price"),
    ("energy", {"measurement": "power"}, "Energy · power"),
    ("temperature", {}, "Temperature"),
    (
        "weather",
        {
            "variable": "humidity",
            "station": "station [blue]A[/blue]",
            "measurement_semantics": "interval_mean",
        },
        "Weather · humidity · station [blue]A[/blue] · interval mean",
    ),
]
runner = CliRunner()


def workflow(profile, options=None):
    return {"schema_version": 1, "profile": profile, "options": options or {}}


def commands(workspace):
    output = io.StringIO()
    return (
        RecordCommands(Console(file=output, width=160, color_system=None), workspace),
        output,
    )


def workspace_bytes(workspace):
    return {
        str(path.relative_to(workspace)): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("profile,options,description", PROFILES)
def test_every_profile_is_explained_in_repl_cli_history_and_results(
    config_factory, tmp_path, profile, options, description
):
    value = run_from_config(config_factory(workflow=workflow(profile, options)))
    workspace = tmp_path / "workspace"
    records, output = commands(workspace)
    before = workspace_bytes(workspace)
    records.history()
    result = records.show(value["run_id"])
    assert f"Usage profile: {description}" in output.getvalue()
    assert "MAE/RMSE error units: synthetic_units" in output.getvalue()
    assert "MASE is dimensionless" in output.getvalue()
    assert result["result"]["configuration"]["workflow"] == workflow(profile, options)
    for command in (
        ["experiments", "show", value["run_id"]],
        ["experiments", "list"],
    ):
        response = runner.invoke(app, [*command, "--workspace", str(workspace)])
        assert response.exit_code == 0, response.output
        assert profile.title() in response.stdout
    assert workspace_bytes(workspace) == before


def test_legacy_generic_display_preserves_all_saved_bytes_and_json(
    config_factory, tmp_path
):
    value = run_from_config(config_factory())
    workspace = tmp_path / "workspace"
    store = ExperimentStore(workspace)
    before = workspace_bytes(workspace)
    records, output = commands(workspace)
    expected = store.list_runs()
    assert records.history() == expected
    shown = records.show(value["run_id"])
    assert "Usage profile: Generic" in output.getvalue()
    assert "workflow" not in shown["result"]["configuration"]
    response = runner.invoke(
        app, ["experiments", "list", "--workspace", str(workspace), "--json"]
    )
    assert response.exit_code == 0
    assert json.loads(response.stdout) == expected
    assert workspace_bytes(workspace) == before


def test_pending_history_uses_checked_metadata_without_opening_source_data(
    config_factory, tmp_path
):
    from tsfm_lab.config import load_config

    config = load_config(
        config_factory(workflow=workflow("energy", {"measurement": "energy"}))
    )
    store = ExperimentStore(tmp_path / "workspace")
    identifier = store.create(config.name, config.resolved_dict(), config.source_yaml)
    (tmp_path / "seasonal.csv").rename(tmp_path / "unavailable.csv")
    manifest = store.get_manifest(identifier)
    assert history_profile(store, manifest) == "Energy"
    records, output = commands(store.workspace)
    records.history()
    assert "Energy" in output.getvalue()
    configuration = store.root / identifier / "configuration.json"
    configuration.write_text("{}")
    assert history_profile(store, manifest) == "Unavailable"


def test_profile_only_differences_preserve_comparison_json_and_scores(
    config_factory, tmp_path
):
    ids = [
        run_from_config(config_factory(workflow=workflow(profile, options)))["run_id"]
        for profile, options in [("generic", {}), ("temperature", {})]
    ]
    workspace = tmp_path / "workspace"
    expected = compare_runs(ids, workspace)
    assert expected["comparison_policy"] == "controlled_point_v1"
    assert expected["runs"][0]["metrics"] == expected["runs"][1]["metrics"]
    assert all("workflow" not in row for row in expected["runs"])
    assert "workflow" not in expected["task"]
    response = runner.invoke(
        app, ["compare", *ids, "--workspace", str(workspace), "--json"]
    )
    assert response.exit_code == 0, response.output
    assert json.loads(response.stdout) == expected
    records, output = commands(workspace)
    assert records.compare(ids) == expected
    assert "Generic" in output.getvalue() and "Temperature" in output.getvalue()
    response = runner.invoke(app, ["compare", *ids, "--workspace", str(workspace)])
    assert response.exit_code == 0, response.output
    assert "Generic" in response.stdout and "Temperature" in response.stdout
    assert "Output" not in response.stdout


def test_finance_offer_has_every_explicit_input_and_never_runs_a_report(
    config_factory, tmp_path, monkeypatch
):
    value = run_from_config(
        config_factory(
            workflow=workflow("finance", {"target_kind": "close_price"}),
            evaluation={"stride": 1, "horizon": 1},
        )
    )
    workspace = tmp_path / "workspace"

    def forbidden(*args, **kwargs):
        raise AssertionError("A result view cannot execute a forecast or OHLC report")

    monkeypatch.setattr("tsfm_lab.finance.backtest.backtest_run", forbidden)
    monkeypatch.setattr("tsfm_lab.application.run_experiment", forbidden)
    records, output = commands(workspace)
    shown = records.show(value["run_id"])
    handoff = "\n".join(
        ohlc_handoff(shown["result"], shown["predictions"], workspace, value["run_id"])
    )
    assert "Optional OHLC report" in handoff
    for flag in (
        "--source-config ORIGINAL_CONFIG.yaml",
        "--ohlc OHLC.csv",
        "--price-target",
        "--initial-capital CAPITAL",
        "--threshold-bps THRESHOLD_BPS",
        "--fee-bps FEE_BPS",
        "--slippage-bps SLIPPAGE_BPS",
        "--short-financing-bps FINANCING_BPS",
        "--signal-latency-seconds LATENCY_SECONDS",
        "--output NEW_REPORT.json",
    ):
        assert flag in handoff
    assert "saved input.yaml may have relative paths" in handoff
    response = runner.invoke(
        app, ["experiments", "show", value["run_id"], "--workspace", str(workspace)]
    )
    assert response.exit_code == 0, response.output
    assert "--threshold-bps THRESHOLD_BPS" in response.stdout


@pytest.mark.parametrize(
    "changed",
    ["legacy", "generic", "return", "native_point", "stride", "negative_forecast"],
)
def test_report_offer_is_never_inferred_from_a_label_or_numeric_series(
    config_factory, tmp_path, changed
):
    value = run_from_config(
        config_factory(
            workflow=workflow("finance", {"target_kind": "close_price"}),
            evaluation={"stride": 1, "horizon": 1},
        )
    )
    result, predictions = ExperimentStore(tmp_path / "workspace").load_result(
        value["run_id"]
    )
    configuration = result["configuration"]
    if changed == "legacy":
        configuration.pop("workflow")
    elif changed == "generic":
        configuration["workflow"] = workflow("generic")
    elif changed == "return":
        configuration["workflow"] = workflow("finance", {"target_kind": "return"})
    elif changed == "native_point":
        configuration["evaluation"]["point_statistic"] = "native_point"
    elif changed == "stride":
        configuration["evaluation"]["stride"] = 4
    else:
        predictions.loc[0, "prediction"] = -1
    assert ohlc_handoff(result, predictions, tmp_path, value["run_id"]) == []


def test_profile_text_keeps_units_and_makes_control_sequences_visible():
    value = copy.deepcopy(workflow("weather", PROFILES[-1][1]))
    value["options"]["station"] = "[red]station[/red]\x1b[31m"
    configuration = {
        "workflow": value,
        "dataset": {"units": "mm"},
        "evaluation": {"metrics": ["mae"]},
    }
    text = "\n".join(profile_context(configuration))
    assert "[red]station[/red]\\x1b[31m" in text
    assert "\x1b" not in text
    assert "MAE error units: mm" in text
    assert "MASE" not in text


def test_ohlc_offer_considers_only_the_existing_reports_lead_one_prices(tmp_path):
    timestamps = pd.date_range("2026-01-01", periods=16, freq="h", tz="UTC")
    dataset = Dataset(timestamps, np.arange(16) + 10.0, "1h", units="quote_units")
    spec = EvaluationSpec(4, 2, timestamps[8], timestamps[-1] + pd.Timedelta("1h"))

    class NegativeLaterLead(LastValue):
        def predict(self, requests):
            return tuple(
                replace(item, point=np.array([item.point[0], -1.0]))
                for item in super().predict(requests)
            )

    result = run_experiment(
        dataset,
        NegativeLaterLead(),
        spec,
        workflow=workflow("finance", {"target_kind": "close_price"}),
    )
    assert (
        result.predictions.loc[result.predictions.lead == 2, "prediction"] < 0
    ).all()
    assert ohlc_handoff(result.to_dict(), result.predictions, tmp_path, "run-id")


def native_records(workspace):
    timestamps = pd.date_range("2026-01-01", periods=16, freq="h", tz="UTC")
    dataset = Dataset(timestamps, np.arange(16) + 10.0, "1h", units="synthetic_units")
    spec = EvaluationSpec(
        4,
        2,
        timestamps[8],
        timestamps[-1] + pd.Timedelta("1h"),
        stride=2,
        point_statistic="native_point",
    )

    class NativeFixture(LastValue):
        adapter_id = "profile_native_fixture"

        def metadata(self):
            return replace(super().metadata(), family="fixture")

        def predict(self, requests):
            return tuple(
                replace(item, point_statistic="native_point")
                for item in super().predict(requests)
            )

    identifiers = []
    store = ExperimentStore(workspace)
    for model in (NativeFixture(), LastValue()):
        result = run_experiment(dataset, model, spec, workflow=workflow("temperature"))
        identifier = store.create("Native fixture", result.configuration, "# Fixture")
        store.complete(identifier, result.to_dict(), result.predictions)
        identifiers.append(identifier)
    return identifiers


def test_cli_native_comparison_keeps_actual_output_disclosures(tmp_path):
    ids = native_records(tmp_path / "workspace")
    response = runner.invoke(
        app, ["compare", *ids, "--workspace", str(tmp_path / "workspace")]
    )
    assert response.exit_code == 0, response.output
    assert "Output" in response.stdout
    assert "native_point" in response.stdout and "deterministic" in response.stdout
    assert "common conditional mean or median" in response.stdout


def test_legacy_views_keep_profiles_and_native_output_labels(tmp_path):
    pytest.importorskip("textual")
    from textual.app import App
    from textual.widgets import DataTable, Static

    from tsfm_lab.tui.views import ComparisonScreen, HistoryScreen, ResultsScreen

    workspace = tmp_path / "workspace"
    ids = native_records(workspace)

    class ViewApp(App):
        def __init__(self, screen):
            super().__init__()
            self.view = screen

        def on_mount(self):
            self.push_screen(self.view)

    async def settle(pilot, condition):
        for _ in range(100):
            await pilot.pause(0.02)
            if condition():
                return
        raise AssertionError("Profile view did not finish")

    async def exercise():
        history = HistoryScreen(workspace)
        async with ViewApp(history).run_test(size=(80, 24)) as pilot:
            await settle(pilot, lambda: len(history.records) == 2)
            table = history.query_one("#history-table", DataTable)
            assert "Temperature" in " ".join(str(cell) for cell in table.get_row_at(0))
        result = ResultsScreen(workspace, ids[0])
        async with ViewApp(result).run_test(size=(80, 24)) as pilot:
            await settle(pilot, lambda: result.completed)
            summary = str(result.query_one("#result-summary", Static).render())
            assert "Usage profile: Temperature" in summary
            assert "Output: native_point" in summary
        comparison = ComparisonScreen(workspace, ids)
        async with ViewApp(comparison).run_test(size=(80, 24)) as pilot:
            table = comparison.query_one("#comparison-table", DataTable)
            await settle(pilot, lambda: table.row_count == 2)
            columns = [str(column.label) for column in table.columns.values()]
            assert "Profile" in columns and "Output" in columns
            assert "native_point" in " ".join(str(cell) for cell in table.get_row_at(0))

    asyncio.run(exercise())
