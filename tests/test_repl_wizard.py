"""Inline setup preserves evaluation semantics without a screen UI or inference."""

import json
import os
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from tsfm_lab.application import run_config, run_from_config
from tsfm_lab.config import ConfigError
from tsfm_lab.contracts import ModelCapabilities
from tsfm_lab.repl import wizard
from tsfm_lab.repl.wizard import build_experiment

_RUN = "Run experiment?"
_RECOVER = "Setup needs attention: retry, back, or cancel?"


class Questions:
    def __init__(self, answers=None):
        self.answers = defaultdict(deque)
        self.calls = []
        for label, values in (answers or {}).items():
            self.answers[label].extend(values if isinstance(values, list) else [values])

    def __call__(self, label, default="", choices=()):
        self.calls.append((label, default, choices))
        answer = self.answers[label].popleft() if self.answers[label] else ""
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def count(self, label):
        return sum(call[0] == label for call in self.calls)


@pytest.fixture
def setup(config_factory, tmp_path):
    path = config_factory()
    return {
        "workspace": tmp_path / "inline-results",
        "demo_path": tmp_path / "seasonal.csv",
        "path": path,
    }


def build(setup, questions, emitted=None, **kwargs):
    return build_experiment(
        questions,
        (emitted if emitted is not None else []).append,
        workspace=setup["workspace"],
        demo_path=setup["demo_path"],
        **kwargs,
    )


def test_quick_demo_asks_only_confirmation_and_matches_configured_metrics(setup):
    questions, emitted = Questions({_RUN: "yes"}), []
    config = build(setup, questions, emitted, demo=True)
    assert [call[0] for call in questions.calls] == [_RUN]
    assert "Windows: 4 · Forecast cases: 16" in emitted[0]
    assert not setup["workspace"].exists()
    actual = run_config(config)
    expected = run_from_config(setup["path"])
    assert actual["metrics"] == expected["metrics"]
    assert actual["counts"] == expected["counts"]


@pytest.mark.parametrize("answer", ["no", "n", "", "/cancel"])
def test_negative_or_default_confirmation_creates_nothing(setup, answer):
    assert build(setup, Questions({_RUN: answer}), demo=True) is None
    assert not setup["workspace"].exists()


def test_guided_demo_skips_column_mapping_and_keeps_sensible_defaults(setup):
    questions = Questions({_RUN: "y"})
    config = build(setup, questions)
    assert config.model["adapter_id"] == "last_value"
    assert config.dataset["target_column"] == "demand"
    assert config.dataset["units"] == "synthetic_units"
    assert config.evaluation.context_length == 8
    assert config.evaluation.horizon == config.evaluation.stride == 4
    assert config.evaluation.evaluation_start == pd.Timestamp("2026-01-01T16:00Z")
    assert config.evaluation.evaluation_end == pd.Timestamp("2026-01-02T08:00Z")
    assert questions.count("Timestamp column") == 0
    assert questions.count("Target column") == 0
    assert not config.source_path.exists()
    assert not setup["workspace"].exists()


def test_own_dataset_requests_mapping_and_preserves_explicit_settings(setup, tmp_path):
    path = tmp_path / "own.csv"
    pd.DataFrame(
        {
            "when": pd.date_range("2026-02-01", periods=20, freq="30min"),
            "load": range(20),
        }
    ).to_csv(path, index=False)
    questions = Questions(
        {
            "Dataset path": str(path),
            "Timestamp column": "when",
            "Target column": "load",
            "Observation frequency": "30min",
            "Target units": "kWh",
            "Experiment name": "[red]literal name[/red]",
            "Context length": "6",
            "Forecast horizon": "2",
            "Evaluation stride": "1",
            "MASE seasonal lag": "1",
            "First target timestamp (inclusive)": "2026-02-01T04:00:00Z",
            "Target end timestamp (exclusive)": "2026-02-01T09:00:00Z",
            _RUN: "yes",
        }
    )
    emitted = []
    config = build(setup, questions, emitted)
    assert config.name == "[red]literal name[/red]"
    assert config.dataset["target_column"] == "load"
    assert config.dataset["frequency"] == "30min"
    assert config.dataset["units"] == "kWh"
    assert config.evaluation.context_length == 6
    assert config.evaluation.horizon == 2
    assert config.evaluation.stride == config.evaluation.seasonal_period == 1
    assert config.evaluation.evaluation_start == pd.Timestamp("2026-02-01T04:00:00Z")
    assert config.evaluation.evaluation_end == pd.Timestamp("2026-02-01T09:00:00Z")
    assert any("[red]literal name[/red]" in value for value in emitted)
    assert not setup["workspace"].exists()


def test_invalid_numeric_answers_reprompt_the_same_question(setup):
    questions = Questions(
        {
            "Context length": ["0", "9"],
            "Forecast horizon": ["1.5", "2"],
            "Evaluation stride": ["-1", "2"],
            "MASE seasonal lag": ["false", "1"],
            _RUN: "yes",
        }
    )
    emitted = []
    config = build(setup, questions, emitted)
    assert config.evaluation.context_length == 9
    assert config.evaluation.horizon == 2
    for label in (
        "Context length",
        "Forecast horizon",
        "Evaluation stride",
        "MASE seasonal lag",
    ):
        assert questions.count(label) == 2
    assert sum("Enter a" in message for message in emitted) >= 4


def test_invalid_bounds_reprompt_without_changing_valid_user_values(setup):
    questions = Questions(
        {
            "First target timestamp (inclusive)": [
                "2026-01-01T16:30:00Z",
                "2026-01-01T17:00:00Z",
            ],
            "Target end timestamp (exclusive)": [
                "2026-01-01T12:00:00Z",
                "2026-01-02T07:00:00Z",
            ],
            _RUN: "yes",
        }
    )
    config = build(setup, questions)
    assert questions.count("First target timestamp (inclusive)") == 2
    assert questions.count("Target end timestamp (exclusive)") == 2
    assert config.evaluation.evaluation_start == pd.Timestamp("2026-01-01T17:00:00Z")
    assert config.evaluation.evaluation_end == pd.Timestamp("2026-01-02T07:00:00Z")


def test_review_back_retains_explicit_evaluation_values(setup):
    questions = Questions(
        {
            "Experiment name": ["First name", "Final name"],
            "Context length": ["9", ""],
            "Forecast horizon": ["2", ""],
            _RUN: ["back", "yes"],
        }
    )
    config = build(setup, questions)
    assert config.name == "Final name"
    assert config.evaluation.context_length == 9
    assert config.evaluation.horizon == 2
    assert questions.count("Dataset path") == 1


def test_back_from_a_field_returns_to_previous_setup_phase(setup):
    questions = Questions({"Experiment name": ["/back", "Changed name"], _RUN: "yes"})
    config = build(setup, questions)
    assert config.name == "Changed name"
    assert questions.count("Dataset path") == 2


def test_review_validation_error_can_retry_or_abort(setup, monkeypatch):
    original = wizard.prepare_experiment
    calls = []

    def changed_once(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise ConfigError("Checkpoint metadata changed; review the settings")
        return original(*args, **kwargs)

    monkeypatch.setattr(wizard, "prepare_experiment", changed_once)
    questions = Questions({_RECOVER: "retry", _RUN: "yes"})
    config = build(setup, questions)
    assert config is not None
    assert questions.count(_RECOVER) == 1
    assert questions.count("Context length") == 2


def test_missing_data_path_reprompts_and_can_cancel(setup):
    questions = Questions({"Dataset path": ["missing.csv", "/cancel"]})
    emitted = []
    assert build(setup, questions, emitted) is None
    assert questions.count("Dataset path") == 2
    assert any("existing local CSV" in text for text in emitted)
    assert not setup["workspace"].exists()


@pytest.mark.parametrize("exception", [KeyboardInterrupt(), EOFError()])
def test_input_interruptions_propagate_to_the_terminal_owner(setup, exception):
    with pytest.raises(type(exception)):
        build(setup, Questions({"Model adapter": exception}))
    assert not setup["workspace"].exists()


def test_seasonal_adapter_argument_sets_the_default_and_keeps_lags_distinct(setup):
    questions = Questions(
        {"Baseline seasonal period": ["0", "4"], "MASE seasonal lag": "2", _RUN: "yes"}
    )
    config = build(setup, questions, adapter="seasonal_naive")
    assert questions.calls[0][1] == "seasonal_naive"
    assert questions.count("Baseline seasonal period") == 2
    assert config.model["options"]["seasonal_period"] == 4
    assert config.evaluation.seasonal_period == 2


def test_local_checkpoint_metadata_refines_limits_without_loading_runtime(
    setup, tmp_path, monkeypatch
):
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["ChronosBoltModelForForecasting"],
                "chronos_pipeline_class": "ChronosBoltPipeline",
                "model_type": "t5",
                "chronos_config": {
                    "context_length": 4,
                    "prediction_length": 2,
                    "quantiles": [0.1, 0.5, 0.9],
                },
            }
        )
    )
    (directory / "model.safetensors").write_bytes(
        b"fake bytes for metadata inspection only"
    )
    questions = Questions(
        {
            "Checkpoint source": "local",
            "Local checkpoint directory": str(directory),
            "Context length": ["5", "4"],
            "Forecast horizon": ["3", "2"],
            _RUN: "yes",
        }
    )
    from tsfm_lab.models.chronos_bolt import ChronosBolt

    monkeypatch.setattr(
        ChronosBolt,
        "load",
        lambda self: pytest.fail("No model initialization during setup"),
    )
    config = build(setup, questions, adapter="chronos_bolt")
    assert config.model["path"] == str(directory)
    assert config.runtime["dtype"] == "float32"
    assert config.evaluation.context_length == 4
    assert config.evaluation.horizon == 2
    assert questions.count("Context length") == questions.count("Forecast horizon") == 2
    assert not setup["workspace"].exists()


@pytest.mark.parametrize(
    "adapter,repository",
    [
        ("chronos_bolt", "amazon/chronos-bolt-tiny"),
        ("timesfm_2p5", "google/timesfm-2.5-200m-pytorch"),
    ],
)
def test_cached_model_requires_exact_sha_and_respects_native_limits(
    setup, monkeypatch, adapter, repository
):
    selections = []
    capabilities = ModelCapabilities(max_context=4, max_horizon=2)

    class Model:
        def metadata(self):
            return SimpleNamespace(capabilities=capabilities)

        def validate(self, spec):
            assert spec.context_length <= 4 and spec.horizon <= 2

    def model(selection, runtime):
        selections.append((selection, runtime))
        return Model()

    monkeypatch.setattr(wizard, "build_model", model)
    monkeypatch.setattr("tsfm_lab.tui.forms.build_model", model)
    questions = Questions({"Exact checkpoint SHA": ["main", "a" * 40], _RUN: "yes"})
    config = build(setup, questions, adapter=adapter)
    assert questions.count("Exact checkpoint SHA") == 2
    assert config.model["offline"] is True
    assert config.model["revision"] == "a" * 40
    assert config.model["repository"] == repository
    assert config.evaluation.context_length == 4
    assert config.evaluation.horizon == 2
    assert all(selection["source"] == "huggingface" for selection, _ in selections)
    assert not setup["workspace"].exists()


def test_selecting_timesfm_from_default_setup_uses_only_its_repository(
    setup, monkeypatch
):
    class Model:
        def metadata(self):
            return SimpleNamespace(capabilities=ModelCapabilities(max_horizon=128))

        def validate(self, spec):
            pass

    monkeypatch.setattr(wizard, "build_model", lambda *_: Model())
    monkeypatch.setattr("tsfm_lab.tui.forms.build_model", lambda *_: Model())
    questions = Questions(
        {
            "Model adapter": "timesfm_2p5",
            "Exact checkpoint SHA": "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
            _RUN: "yes",
        }
    )
    config = build(setup, questions)
    assert config.model["adapter_id"] == "timesfm_2p5"
    assert config.model["repository"] == "google/timesfm-2.5-200m-pytorch"
    repository_call = next(
        call for call in questions.calls if call[0] == "Checkpoint repository"
    )
    assert repository_call[2] == ("google/timesfm-2.5-200m-pytorch",)
    assert config.runtime["dtype"] == "float32"
    assert not setup["workspace"].exists()


@pytest.mark.parametrize("adapter", ["chronos_bolt", "timesfm_2p5"])
def test_checkpoint_quick_demo_requires_guided_selection(setup, adapter):
    with pytest.raises(ConfigError, match="guided setup"):
        build(setup, Questions(), adapter=adapter, demo=True)


def test_fresh_process_demo_setup_imports_no_torch_textual_or_chronos(setup):
    program = """
import sys
from pathlib import Path
from tsfm_lab.repl.wizard import build_experiment
config = build_experiment(lambda *args, **kwargs: 'yes', lambda text: None,
    workspace=Path(sys.argv[1]), demo_path=Path(sys.argv[2]), demo=True)
assert config is not None
assert not {'torch', 'textual', 'chronos', 'timesfm', 'huggingface_hub'}.intersection(sys.modules)
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(setup["workspace"]),
            str(setup["demo_path"]),
        ],
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
    )
    assert not setup["workspace"].exists()
