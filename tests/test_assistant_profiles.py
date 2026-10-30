"""All five assisted profiles preserve independently maintained manual examples."""

import json
import socket
from pathlib import Path

import numpy as np
import pytest

from tsfm_lab.application import load_dataset
from tsfm_lab.assistant.drafts import decode_draft, resolve_draft
from tsfm_lab.config import load_config
from tsfm_lab.data import canonical_fingerprint
from tsfm_lab.repl.wizard import build_experiment
from tsfm_lab.workflows import PROFILE_NAMES

_ROOT = Path(__file__).resolve().parents[1]
_CASES = {
    "generic": {
        "adapter": "last_value",
        "target_column": "demand",
        "units": "synthetic_units",
        "context_length": "8",
        "horizon": "4",
        "stride": "4",
        "mase_period": "4",
        "evaluation_start": "2026-01-01T16:00:00Z",
        "evaluation_end": "2026-01-02T08:00:00Z",
    },
    "finance": {
        "adapter": "seasonal_naive",
        "target_column": "close",
        "units": "toy_quote_per_asset",
        "horizon": "1",
        "stride": "1",
        "seasonal_period": "24",
        "finance_target_kind": "close_price",
    },
    "energy": {
        "adapter": "last_value",
        "target_column": "power",
        "units": "kW",
        "energy_measurement": "power",
    },
    "temperature": {
        "adapter": "last_value",
        "target_column": "temperature",
        "units": "degC",
    },
    "weather": {
        "adapter": "last_value",
        "target_column": "wind_speed",
        "units": "m/s",
        "weather_variable": "wind_speed",
        "weather_station": "synthetic-station",
        "weather_measurement_semantics": "instantaneous",
    },
}
_QUESTIONS = {
    "Usage profile": "usage_profile",
    "Forecast task": "point_statistic",
    "Baseline seasonal period": "seasonal_period",
    "Timestamp column": "timestamp_column",
    "Target column": "target_column",
    "Observation frequency": "frequency",
    "Experiment name": "name",
    "Context length": "context_length",
    "Forecast horizon": "horizon",
    "Evaluation stride": "stride",
    "MASE seasonal lag": "mase_period",
    "First target timestamp (inclusive)": "evaluation_start",
    "Target end timestamp (exclusive)": "evaluation_end",
    "Finance target kind": "finance_target_kind",
    "Energy measurement": "energy_measurement",
    "Weather variable": "weather_variable",
    "Weather station": "weather_station",
    "Weather measurement semantics": "weather_measurement_semantics",
}


def _fields(profile):
    return {
        "name": f"usage-{profile}-demo",
        "usage_profile": profile,
        "timestamp_column": "timestamp",
        "frequency": "1h",
        "timezone": "",
        "point_statistic": "median",
        "context_length": "48",
        "horizon": "4",
        "stride": "4",
        "mase_period": "24",
        "evaluation_start": "2026-01-03T00:00:00Z",
        "evaluation_end": "2026-01-03T16:00:00Z",
        **_CASES[profile],
    }


class Questions:
    def __init__(self, answers=None):
        self.answers = answers or {}
        self.calls = []

    def __call__(self, label, default="", choices=()):
        self.calls.append((label, default, choices))
        assert len(self.calls) < 50, f"Unexpected repeated question: {label}"
        return self.answers.get(label, "yes" if label == "Run experiment?" else "")


def _normalized(config, workspace):
    result = config.resolved_dict()
    result["output"] = {"directory": str(workspace)}
    return result


@pytest.mark.parametrize("profile", tuple(_CASES))
def test_assisted_and_manual_guides_match_each_golden_profile_without_execution(
    profile, tmp_path, monkeypatch
):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Profile preparation must not execute runs or use a network")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr("tsfm_lab.application.run_config", forbidden)
    monkeypatch.setattr("tsfm_lab.application.run_from_config", forbidden)
    expected = load_config(_ROOT / "examples/configurations" / f"usage-{profile}.yaml")
    path = Path(expected.dataset["path"])
    original_data = load_dataset(expected.dataset)
    fields = _fields(profile)
    wire = {
        "schema_version": 1,
        "state": "needs_input",  # The provider cannot establish local readiness.
        "intent": "historical_evaluation",
        "dataset_ref": "selected_dataset",
        "fields": fields,
        "explicit_fields": [],  # Every supplied value must still be preserved.
        "assumptions": [],
        "missing_fields": [],
    }
    assert not {"dataset_path", "checkpoint_path", "output", "runtime"} & fields.keys()
    assert str(path) not in json.dumps(wire)
    resolved = resolve_draft(
        decode_draft(json.dumps(wire)), dataset_refs={"selected_dataset": path}
    )
    assert resolved.state == "ready", (resolved.missing_fields, resolved.errors)
    assert fields.items() <= resolved.seed_fields.items()
    workspace = tmp_path / "prepared-only"
    assisted_questions = Questions()
    assisted = build_experiment(
        assisted_questions,
        lambda _message: None,
        workspace=workspace,
        demo_path=_ROOT / "examples/datasets/seasonal.csv",
        profile_demo_paths={
            name: _ROOT / "examples/datasets" / f"{name}-demo.csv"
            for name in PROFILE_NAMES
            if name != "generic"
        },
        seed_fields=resolved.seed_fields,
        explicit_fields=resolved.explicit_fields,
    )
    manual_answers = {
        label: fields[key] for label, key in _QUESTIONS.items() if key in fields
    }
    manual_answers.update(
        {
            "Model family": "Baselines",
            "Baseline": (
                "Seasonal naive"
                if fields["adapter"] == "seasonal_naive"
                else "Last value"
            ),
            "Dataset path": str(path),
            "Source timezone (optional; - leaves it unspecified)": "-",
            "Target units": fields["units"],
            "Financial target units": fields["units"],
            "Power or energy units": fields["units"],
            "Temperature units": fields["units"],
            "Weather measurement units": fields["units"],
        }
    )
    manual_questions = Questions(manual_answers)
    manual = build_experiment(
        manual_questions,
        lambda _message: None,
        workspace=workspace,
        demo_path=_ROOT / "src/tsfm_lab/resources/seasonal.csv",
    )
    assert assisted is not None and manual is not None
    golden = _normalized(expected, workspace)
    assert _normalized(assisted, workspace) == golden
    assert _normalized(manual, workspace) == golden
    for questions in (assisted_questions, manual_questions):
        profiles = next(
            call[2] for call in questions.calls if call[0] == "Usage profile"
        )
        assert set(profiles) == set(_CASES) == set(PROFILE_NAMES)
        assert questions.calls[-1] == (
            "Run experiment?",
            "no",
            ("yes", "no", "back"),
        )
        required = {
            "generic": {"Target units"},
            "finance": {"Finance target kind", "Financial target units"},
            "energy": {"Energy measurement", "Power or energy units"},
            "temperature": {"Temperature units"},
            "weather": {
                "Weather variable",
                "Weather station",
                "Weather measurement semantics",
                "Weather measurement units",
            },
        }[profile]
        assert required <= {call[0] for call in questions.calls}
    assisted_defaults = {
        label: default for label, default, _ in assisted_questions.calls
    }
    for label, key in _QUESTIONS.items():
        if key in fields:
            assert assisted_defaults[label] == fields[key]
    assert assisted.evaluation.point_statistic == "median"
    assert assisted.workflow == expected.workflow
    if profile == "finance":
        assert assisted.workflow["options"] == {"target_kind": "close_price"}
    for config in (manual, assisted):
        loaded = load_dataset(config.dataset)
        np.testing.assert_array_equal(loaded.values, original_data.values)
        assert canonical_fingerprint(loaded) == canonical_fingerprint(original_data)
        if profile == "temperature":
            assert original_data.values.min() < 0 and loaded.values.min() < 0
    assert not workspace.exists()
