"""Profile choices guide declarations while preserving explicit numerical settings."""

import asyncio
import builtins
import json
import socket
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tsfm_lab.config import ConfigError, config_from_mapping
from tsfm_lab.models.ttm_profile import _configuration
from tsfm_lab.repl.wizard import build_experiment
from tsfm_lab.tui.forms import (
    config_from_fields,
    demo_fields,
    inspect_data,
    prepare_experiment,
    profile_example_fields,
    suggested_bounds,
)


@pytest.mark.parametrize(
    "profile", ["generic", "finance", "energy", "temperature", "weather"]
)
@pytest.mark.parametrize("context", [512, 1024, 1536])
def test_all_usage_profiles_accept_each_exact_ttm_context_without_runtime_loading(
    tmp_path, monkeypatch, profile, context
):
    resources = Path(__file__).resolve().parents[1] / "src/tsfm_lab/resources"
    domain_paths = {
        name: resources / f"{name}-demo.csv"
        for name in ("finance", "energy", "temperature", "weather")
    }
    expected = {
        "generic": ("demand", "synthetic_units"),
        "finance": ("close", "toy_quote_per_asset"),
        "energy": ("power", "kW"),
        "temperature": ("temperature", "degC"),
        "weather": ("wind_speed", "m/s"),
    }
    checkpoint = tmp_path / "inert-ttm"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(_configuration(context)))
    (checkpoint / "model.safetensors").write_bytes(b"inert setup fixture; never loaded")
    original_import = builtins.__import__
    forbidden = {
        "torch",
        "transformers",
        "tsfm_public",
        "huggingface_hub",
        "safetensors",
    }

    def no_runtime(name, *args, **kwargs):
        assert (
            name.split(".")[0] not in forbidden
        ), f"Unexpected model runtime import: {name}"
        return original_import(name, *args, **kwargs)

    def no_network(*args, **kwargs):
        raise AssertionError("Profile setup must not connect or download")

    monkeypatch.setattr(builtins, "__import__", no_runtime)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    answers = {
        "Usage profile": profile,
        "Checkpoint source": "local",
        "Local checkpoint directory": str(checkpoint),
        "Run experiment?": "yes",
    }
    calls = []

    def question(label, default="", choices=()):
        calls.append(label)
        assert len(calls) < 40, "Setup repeated an unanswered question"
        if label in answers:
            return answers[label]
        assert default, f"Supply an explicit answer for unexpected question: {label}"
        return ""

    config = build_experiment(
        question,
        lambda message: None,
        workspace=tmp_path / "runs",
        demo_path=resources / "seasonal.csv",
        ttm_demo_path=resources / "ttm-seasonal.csv",
        profile_demo_paths=domain_paths,
        adapter="ttm_r2",
    )
    assert config is not None
    assert config.model["source"] == "local"
    assert config.model["path"] == str(checkpoint)
    assert config.evaluation.context_length == context
    assert config.evaluation.point_statistic == "native_point"
    assert config.workflow["profile"] == profile
    assert (config.dataset["target_column"], config.dataset["units"]) == expected[
        profile
    ]
    expected_path = (
        resources / "ttm-seasonal.csv"
        if profile == "generic"
        else domain_paths[profile]
    )
    assert config.dataset["path"] == str(expected_path)
    frame = pd.read_csv(expected_path)
    assert len(frame) == 1600
    first_target = pd.Timestamp(config.evaluation.evaluation_start)
    timestamps = pd.to_datetime(frame.timestamp, utc=True)
    assert (timestamps < first_target).sum() >= context
    assert (timestamps >= first_target).sum() >= config.evaluation.horizon
    assert "Checkpoint profile" not in calls
    assert not (tmp_path / "runs").exists()


class Questions:
    def __init__(self, answers=None):
        self.answers = defaultdict(deque)
        self.calls = []
        for label, values in (answers or {}).items():
            self.answers[label].extend(values if isinstance(values, list) else [values])

    def __call__(self, label, default="", choices=()):
        self.calls.append((label, default, choices))
        return self.answers[label].popleft() if self.answers[label] else ""


@pytest.fixture
def examples(tmp_path):
    columns = {
        "generic": ("demand", "synthetic_units", np.arange(32) + 10.0),
        "finance": ("close", "toy_quote_per_asset", np.arange(32) + 100.0),
        "energy": ("power", "kW", np.arange(32) + 2.0),
        "temperature": ("temperature", "degC", np.arange(32) - 16.0),
        "weather": ("wind_speed", "m/s", np.arange(32) / 10),
    }
    paths = {}
    for profile, (column, _, values) in columns.items():
        path = tmp_path / f"{profile}.csv"
        pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2026-01-01", periods=32, freq="h", tz="UTC"
                ),
                column: values,
            }
        ).to_csv(path, index=False)
        paths[profile] = path
    return paths, columns


def guided(examples, tmp_path, questions, emitted=None, **kwargs):
    paths, _ = examples
    return build_experiment(
        questions,
        (emitted if emitted is not None else []).append,
        workspace=tmp_path / "runs",
        demo_path=paths["generic"],
        profile_demo_paths={
            key: path for key, path in paths.items() if key != "generic"
        },
        **kwargs,
    )


@pytest.mark.parametrize(
    "profile,options",
    [
        ("generic", {}),
        ("finance", {"target_kind": "close_price"}),
        ("energy", {"measurement": "power"}),
        ("temperature", {}),
        (
            "weather",
            {
                "variable": "wind_speed",
                "station": "synthetic-station",
                "measurement_semantics": "instantaneous",
            },
        ),
    ],
)
def test_profile_guide_matches_independent_manual_configuration(
    examples, tmp_path, profile, options
):
    paths, columns = examples
    questions = Questions({"Usage profile": profile, "Run experiment?": "yes"})
    config = guided(examples, tmp_path, questions)
    expected = config_from_mapping(
        {
            "schema_version": 1,
            "name": "New experiment",
            "workflow": {"schema_version": 1, "profile": profile, "options": options},
            "model": {"source": "builtin", "adapter_id": "last_value"},
            "dataset": {
                "path": str(paths[profile]),
                "timestamp_column": "timestamp",
                "target_column": columns[profile][0],
                "frequency": "1h",
                "timezone": "UTC",
                "units": columns[profile][1],
            },
            "evaluation": {
                "context_length": 8,
                "horizon": 4,
                "stride": 4,
                "seasonal_period": 4,
                "evaluation_start": "2026-01-01T16:00:00Z",
                "evaluation_end": "2026-01-02T08:00:00Z",
                "metrics": ["mae", "rmse", "mase"],
                "point_statistic": "median",
            },
            "output": {"directory": str(tmp_path / "runs")},
        },
        base_directory=tmp_path,
    )
    assert config.resolved_dict() == expected.resolved_dict()
    assert yaml.safe_load(config.source_yaml)["workflow"] == expected.workflow
    assert not (tmp_path / "runs").exists()


def test_quick_demo_keeps_absent_workflow_and_asks_only_confirmation(
    examples, tmp_path
):
    questions = Questions({"Run experiment?": "yes"})
    config = guided(examples, tmp_path, questions, demo=True)
    assert config.workflow is None
    assert "workflow" not in config.resolved_dict()
    assert "workflow" not in yaml.safe_load(config.source_yaml)
    assert [call[0] for call in questions.calls] == ["Run experiment?"]


def test_temperature_inspection_preserves_negative_readings(examples, tmp_path):
    paths, _ = examples
    fields = profile_example_fields("temperature", paths["temperature"])
    arguments = {"workspace": tmp_path / "runs", "base_directory": tmp_path}
    dataset = inspect_data(fields, **arguments)
    np.testing.assert_array_equal(dataset.values, np.arange(32) - 16.0)
    fields.update(suggested_bounds(dataset, fields))
    prepared = prepare_experiment(fields, **arguments)
    assert "Usage: Temperature" in prepared.summary()
    assert prepared.config.dataset["units"] == "degC"


def test_custom_weather_requests_station_semantics_and_units(examples, tmp_path):
    paths, _ = examples
    questions = Questions(
        {
            "Usage profile": "weather",
            "Dataset path": str(paths["generic"]),
            "Weather variable": "rainfall",
            "Weather station": ["", "station-7"],
            "Weather measurement semantics": "interval_total",
            "Weather measurement units": "mm/interval",
            "Run experiment?": "yes",
        }
    )
    config = guided(examples, tmp_path, questions)
    assert config.dataset["path"] == str(paths["generic"])
    assert config.dataset["units"] == "mm/interval"
    assert config.workflow["options"] == {
        "variable": "rainfall",
        "station": "station-7",
        "measurement_semantics": "interval_total",
    }
    assert sum(call[0] == "Weather station" for call in questions.calls) == 2


def test_switching_from_weather_example_requires_fresh_custom_declarations(
    examples, tmp_path
):
    questions = Questions(
        {
            "Usage profile": "weather",
            "Dataset path": ["", str(examples[0]["generic"])],
            "Target column": "demand",
            "Experiment name": ["first", "/back", "final"],
            "Weather variable": ["", "rainfall"],
            "Weather station": ["", "station-new"],
            "Weather measurement semantics": ["", "interval_total"],
            "Weather measurement units": ["", "mm"],
            "Run experiment?": ["back", "yes"],
        }
    )
    config = guided(examples, tmp_path, questions)
    assert config.workflow["options"]["station"] == "station-new"
    assert config.dataset["units"] == "mm"
    station_calls = [call for call in questions.calls if call[0] == "Weather station"]
    unit_calls = [
        call for call in questions.calls if call[0] == "Weather measurement units"
    ]
    assert [call[1] for call in station_calls] == ["synthetic-station", ""]
    assert [call[1] for call in unit_calls] == ["m/s", "unspecified"]


def test_other_measurements_require_user_descriptions(examples, tmp_path):
    questions = Questions(
        {
            "Usage profile": "energy",
            "Energy measurement": "other",
            "Energy quantity": "meter counter",
            "Power or energy units": "meter_ticks",
            "Run experiment?": "yes",
        }
    )
    config = guided(examples, tmp_path, questions)
    assert config.workflow["options"] == {
        "measurement": "other",
        "quantity": "meter counter",
    }
    assert config.dataset["units"] == "meter_ticks"


def test_revisiting_profile_data_preserves_deliberate_unit_declaration(
    examples, tmp_path
):
    questions = Questions(
        {
            "Usage profile": "energy",
            "Power or energy units": ["kilowatts", ""],
            "Experiment name": ["first", "/back", "final"],
            "Run experiment?": ["back", "yes"],
        }
    )
    config = guided(examples, tmp_path, questions)
    assert config.dataset["units"] == "kilowatts"
    assert [
        call[1] for call in questions.calls if call[0] == "Power or energy units"
    ] == ["kW", "kilowatts"]


def test_profile_back_navigation_retains_explicit_numeric_choices(examples, tmp_path):
    questions = Questions(
        {
            "Usage profile": ["energy", "temperature"],
            "Experiment name": ["first", "/back", "final"],
            "Context length": "9",
            "Forecast horizon": "2",
            "Evaluation stride": "1",
            "MASE seasonal lag": "2",
            "Dataset path": ["", "/back", ""],
            "Model adapter": ["", "/back", ""],
            "Run experiment?": ["back", "yes"],
        }
    )
    emitted = []
    config = guided(examples, tmp_path, questions, emitted)
    assert config.workflow == {
        "schema_version": 1,
        "profile": "temperature",
        "options": {},
    }
    assert config.dataset["path"] == str(examples[0]["temperature"])
    assert config.name == "final"
    assert (
        config.evaluation.context_length,
        config.evaluation.horizon,
        config.evaluation.stride,
        config.evaluation.seasonal_period,
    ) == (9, 2, 1, 2)
    assert any("24 observations per day or 168 per week" in item for item in emitted)


@pytest.mark.parametrize(
    "answers",
    [
        {"Usage profile": "/cancel"},
        {"Usage profile": "weather", "Weather station": "/cancel"},
    ],
)
def test_profile_cancellation_writes_nothing(examples, tmp_path, answers):
    assert guided(examples, tmp_path, Questions(answers)) is None
    assert not (tmp_path / "runs").exists()


def test_inactive_profile_fields_never_leak_into_resolved_options(examples, tmp_path):
    fields = profile_example_fields("temperature", examples[0]["temperature"])
    fields.update(
        finance_target_kind="close_price",
        energy_measurement="power",
        weather_variable="wind",
    )
    data = inspect_data(fields, workspace=tmp_path / "runs", base_directory=tmp_path)
    fields.update(suggested_bounds(data, fields))
    config = config_from_fields(
        fields, workspace=tmp_path / "runs", base_directory=tmp_path
    )
    assert config.workflow["options"] == {}


def test_profile_form_rejects_undeclared_measurement_before_result_writes(
    examples, tmp_path
):
    fields = demo_fields(examples[0]["generic"])
    fields["usage_profile"] = "energy"
    with pytest.raises(ConfigError, match="measurement"):
        inspect_data(fields, workspace=tmp_path / "runs", base_directory=tmp_path)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("profile", ["finance", "energy", "temperature", "weather"])
def test_legacy_tui_profile_example_and_review_match_shared_form(
    examples, tmp_path, profile
):
    pytest.importorskip("textual")
    from textual.app import App
    from textual.widgets import Input, Select

    from tsfm_lab.tui.wizard import WizardScreen

    class ProfileApp(App):
        def on_mount(self):
            self.wizard = WizardScreen(
                tmp_path / "runs",
                examples[0]["generic"],
                profile_demo_paths=examples[0],
            )
            self.push_screen(self.wizard)

    async def settle(pilot, screen):
        await pilot.pause()
        for _ in range(100):
            if not screen.busy:
                await pilot.pause()
                return
            await asyncio.sleep(0.01)
            await pilot.pause()
        raise AssertionError("Profile validation did not finish")

    async def exercise():
        app = ProfileApp()
        async with app.run_test(size=(80, 24)) as pilot:
            screen = app.wizard
            screen.query_one("#wizard-usage-profile", Select).value = profile
            await pilot.pause()
            screen.action_next()
            assert screen.step == "data"
            screen._use_demo()
            await settle(pilot, screen)
            assert (
                screen.query_one("#wizard-target-column", Input).value
                == examples[1][profile][0]
            )
            assert (
                screen.query_one("#wizard-units", Input).value
                == examples[1][profile][1]
            )
            screen.action_next()
            await settle(pilot, screen)
            assert screen.step == "evaluation"
            screen.action_next()
            await settle(pilot, screen)
            assert screen.step == "review"
            assert screen._prepared.config.workflow["profile"] == profile
            assert screen._prepared.config.evaluation.seasonal_period == 4
            assert not (tmp_path / "runs").exists()

    asyncio.run(exercise())


def test_legacy_tui_discards_example_declarations_for_a_custom_path(examples, tmp_path):
    pytest.importorskip("textual")
    from textual.app import App
    from textual.widgets import Input, Select

    from tsfm_lab.tui.wizard import WizardScreen

    class ProfileApp(App):
        def on_mount(self):
            self.wizard = WizardScreen(
                tmp_path / "runs",
                examples[0]["generic"],
                profile_demo_paths=examples[0],
            )
            self.push_screen(self.wizard)

    async def exercise():
        app = ProfileApp()
        async with app.run_test(size=(80, 24)) as pilot:
            screen = app.wizard
            screen.query_one("#wizard-usage-profile", Select).value = "weather"
            await pilot.pause()
            screen._apply_demo_fields()
            await pilot.pause()
            assert (
                screen.query_one("#wizard-weather-station", Input).value
                == "synthetic-station"
            )
            screen.query_one("#wizard-dataset-path", Input).value = str(
                examples[0]["generic"]
            )
            await pilot.pause()
            assert screen.query_one("#wizard-weather-station", Input).value == ""
            assert (
                screen.query_one("#wizard-weather-measurement-semantics", Select).value
                == ""
            )
            assert screen.query_one("#wizard-units", Input).value == "unspecified"
            screen.query_one("#wizard-dataset-path", Input).value = "\0invalid.csv"
            await pilot.pause()
            screen.query_one("#wizard-dataset-path", Input).value = str(
                examples[0]["generic"]
            )
            await pilot.pause()
            before = screen._fields()
            screen.query_one("#wizard-usage-profile", Select).value = "temperature"
            await pilot.pause()
            after = screen._fields()
            for key in (
                "dataset_path",
                "context_length",
                "horizon",
                "stride",
                "mase_period",
            ):
                assert after[key] == before[key]
            assert not (tmp_path / "runs").exists()

    asyncio.run(exercise())
