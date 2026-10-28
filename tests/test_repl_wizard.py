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
        assert len(self.calls) < 100, f"Unexpected repeated question: {label}"
        answer = self.answers[label].popleft() if self.answers[label] else ""
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def count(self, label):
        return sum(call[0] == label for call in self.calls)


@pytest.mark.parametrize(
    "answer", ["NX-AI/TiRex-2", "nx-ai/tirex-2", "Nx-Ai/TiReX-2", ""]
)
def test_choice_matching_preserves_curated_repository_spelling(answer):
    questions = Questions({"Repository": answer})
    emitted = []
    repository = wizard._question(
        questions,
        emitted.append,
        "Repository",
        default="NX-AI/TiRex-2",
        choices=("NX-AI/TiRex-2",),
    )
    assert repository == "NX-AI/TiRex-2"
    assert questions.count("Repository") == 1 and emitted == []


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
        build(setup, Questions({"Model family": exception}))
    assert not setup["workspace"].exists()


def test_seasonal_adapter_argument_sets_the_default_and_keeps_lags_distinct(setup):
    questions = Questions(
        {"Baseline seasonal period": ["0", "4"], "MASE seasonal lag": "2", _RUN: "yes"}
    )
    config = build(setup, questions, adapter="seasonal_naive")
    assert (
        next(call[1] for call in questions.calls if call[0] == "Baseline")
        == "Seasonal naive"
    )
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
    emitted = []
    config = build(setup, questions, emitted, adapter="chronos_bolt")
    assert config.model["path"] == str(directory)
    assert config.runtime["dtype"] == "float32"
    assert config.evaluation.context_length == 4
    assert config.evaluation.horizon == 2
    assert questions.count("Context length") == questions.count("Forecast horizon") == 2
    text = "\n".join(emitted)
    assert (
        "Selected checkpoint: Compatibility checked; project inference check not recorded for these files"
        in text
    )
    assert "Adapter reference:" in text
    assert not setup["workspace"].exists()


@pytest.mark.parametrize(
    "adapter,repository",
    [
        ("chronos_bolt", "amazon/chronos-bolt-tiny"),
        ("chronos_bolt", "amazon/chronos-bolt-base"),
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
    questions = Questions(
        {
            "Model size": wizard.checkpoint_info(repository)["display_name"],
            "Advanced checkpoint options?": "yes",
            "Exact checkpoint SHA": ["main", "a" * 40],
            _RUN: "yes",
        }
    )
    config = build(setup, questions, adapter=adapter)
    assert questions.count("Exact checkpoint SHA") == 2
    assert config.model["offline"] is True
    assert config.model["revision"] == "a" * 40
    assert config.model["repository"] == repository
    assert config.evaluation.context_length == 4
    assert config.evaluation.horizon == 2
    assert all(selection["source"] == "huggingface" for selection, _ in selections)
    assert not setup["workspace"].exists()


def test_base_choice_shows_compatibility_and_project_reference_evidence(
    setup, monkeypatch
):
    class Model:
        def metadata(self):
            return SimpleNamespace(capabilities=ModelCapabilities(max_horizon=64))

        def validate(self, spec):
            pass

    monkeypatch.setattr(wizard, "build_model", lambda *_: Model())
    monkeypatch.setattr("tsfm_lab.tui.forms.build_model", lambda *_: Model())
    questions, emitted = (
        Questions(
            {
                "Model size": "Base",
                "Advanced checkpoint options?": "yes",
                _RUN: "yes",
            }
        ),
        [],
    )
    config = build(setup, questions, emitted, adapter="chronos_bolt")
    assert config.model["repository"] == "amazon/chronos-bolt-base"
    reference = wizard.checkpoint_info("amazon/chronos-bolt-base")["reference_revision"]
    assert config.model["revision"] == reference
    assert (
        next(call[1] for call in questions.calls if call[0] == "Exact checkpoint SHA")
        == reference
    )
    text = "\n".join(emitted)
    assert "amazon/chronos-bolt-base" in text
    assert "individual inference not run" in text
    assert "Reference checkpoint inference check recorded" in text
    size_call = next(call for call in questions.calls if call[0] == "Model size")
    assert size_call[2] == ("Tiny", "Small", "Base")
    assert not setup["workspace"].exists()


@pytest.mark.parametrize(
    "repository,revision,context,horizon",
    [
        ("NX-AI/TiRex-2", "30788a311699acba7e6393ad9700ea7b7cb9bb2d", 2048, 320),
        (
            "NX-AI/TiRex-2-fevbench",
            "acd6153b8ee2179f38cfd3cc4fd64a3c045a424f",
            2048,
            320,
        ),
        (
            "NX-AI/TiRex-2-gifteval-zs",
            "c94872d1c8caacbe7755cbfa01bc5c4c5c5ddb7f",
            8192,
            928,
        ),
        (
            "NX-AI/TiRex-2-gifteval-pretrain",
            "54f88ac06ccfba9fe3e727aa8856b39775481f0f",
            8192,
            928,
        ),
    ],
)
def test_tirex_training_profiles_select_canonical_repository_and_pinned_sha(
    setup, monkeypatch, repository, revision, context, horizon
):
    selections = []

    class Model:
        def metadata(self):
            return SimpleNamespace(
                capabilities=ModelCapabilities(max_context=context, max_horizon=horizon)
            )

        def validate(self, spec):
            assert spec.context_length <= context and spec.horizon <= horizon

        def load(self):
            pytest.fail("Guided selection must not load model weights")

    def selected_model(selection, runtime):
        selections.append(selection)
        return Model()

    monkeypatch.setattr(wizard, "build_model", selected_model)
    monkeypatch.setattr("tsfm_lab.tui.forms.build_model", selected_model)
    questions = Questions(
        {
            "Training profile": wizard.checkpoint_info(repository)[
                "display_name"
            ].lower(),
            "Advanced checkpoint options?": "yes",
            _RUN: "yes",
        }
    )
    emitted = []
    config = build(setup, questions, emitted, adapter="tirex_2")
    assert config.model["adapter_id"] == "tirex_2"
    assert config.model["repository"] == repository
    assert config.model["revision"] == revision
    assert questions.count("Training profile") == 1
    assert questions.count("Model size") == 0
    assert all(item["repository"] == repository for item in selections)
    sha = next(call for call in questions.calls if call[0] == "Exact checkpoint SHA")
    assert sha[1] == revision
    text = "\n".join(emitted)
    assert f"Context up to {context}" in text
    assert f"Horizon up to {horizon}" in text
    assert "Built with technology from NXAI" in text
    assert "NXAI Community License" in text
    assert not setup["workspace"].exists()


@pytest.mark.parametrize("change_repository", [False, True])
def test_tirex_back_navigation_preserves_sha_override_until_profile_changes(
    setup, monkeypatch, change_repository
):
    class Model:
        def metadata(self):
            return SimpleNamespace(
                capabilities=ModelCapabilities(max_context=8192, max_horizon=928)
            )

        def validate(self, spec):
            pass

    monkeypatch.setattr(wizard, "build_model", lambda *_: Model())
    monkeypatch.setattr("tsfm_lab.tui.forms.build_model", lambda *_: Model())
    first = "NX-AI/TiRex-2"
    second = "NX-AI/TiRex-2-gifteval-zs" if change_repository else first
    override = "b" * 40
    questions = Questions(
        {
            "Training profile": [
                wizard.checkpoint_info(first)["display_name"],
                wizard.checkpoint_info(second)["display_name"].lower(),
            ],
            "Advanced checkpoint options?": ["yes", "yes"],
            "Exact checkpoint SHA": [override, ""],
            "Dataset path": ["/back", ""],
            "Context length": ["", "/back", ""],
            _RUN: "yes",
        }
    )
    config = build(setup, questions, adapter="tirex_2")
    assert config.model["repository"] == second
    assert config.model["revision"] == (
        "c94872d1c8caacbe7755cbfa01bc5c4c5c5ddb7f" if change_repository else override
    )
    assert questions.count("Model family") == 2


@pytest.mark.parametrize("variant", ["tiny", "base"])
def test_back_to_model_preserves_same_repo_override_or_defaults_new_variant(
    setup, monkeypatch, variant
):
    class Model:
        def metadata(self):
            return SimpleNamespace(capabilities=ModelCapabilities(max_horizon=64))

        def validate(self, spec):
            pass

    monkeypatch.setattr(wizard, "build_model", lambda *_: Model())
    monkeypatch.setattr("tsfm_lab.tui.forms.build_model", lambda *_: Model())
    repository = f"amazon/chronos-bolt-{variant}"
    override = "b" * 40
    questions = Questions(
        {
            "Model size": ["Tiny", wizard.checkpoint_info(repository)["display_name"]],
            "Advanced checkpoint options?": ["yes", "yes"],
            "Exact checkpoint SHA": [override, ""],
            "Dataset path": ["/back", ""],
            "Context length": ["", "/back", ""],
            _RUN: "yes",
        }
    )
    config = build(setup, questions, adapter="chronos_bolt")
    expected = (
        override
        if variant == "tiny"
        else wizard.checkpoint_info(repository)["reference_revision"]
    )
    assert config.model["repository"] == repository
    assert config.model["revision"] == expected
    sha_defaults = [
        call[1] for call in questions.calls if call[0] == "Exact checkpoint SHA"
    ]
    assert sha_defaults == [
        wizard.checkpoint_info("amazon/chronos-bolt-tiny")["reference_revision"],
        expected,
    ]
    assert questions.count(_RECOVER) == 0
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
            "Model family": "TimesFM",
            "Advanced checkpoint options?": "yes",
            "Exact checkpoint SHA": "1d952420fba87f3c6dee4f240de0f1a0fbc790e3",
            _RUN: "yes",
        }
    )
    config = build(setup, questions)
    assert config.model["adapter_id"] == "timesfm_2p5"
    assert config.model["repository"] == "google/timesfm-2.5-200m-pytorch"
    size_call = next(call for call in questions.calls if call[0] == "Model size")
    assert size_call[2] == ("2.5 200M",)
    assert config.runtime["dtype"] == "float32"
    assert not setup["workspace"].exists()


def test_family_change_preserves_compatible_explicit_model_options(setup, monkeypatch):
    class MetadataOnly:
        def metadata(self):
            return SimpleNamespace(
                capabilities=ModelCapabilities(max_context=2048, max_horizon=128)
            )

        def validate(self, spec):
            assert spec.point_statistic == "native_point"

        def load(self):
            pytest.fail("Changing setup choices must not load model weights")

    monkeypatch.setattr(wizard, "build_model", lambda *_: MetadataOnly())
    monkeypatch.setattr("tsfm_lab.tui.forms.build_model", lambda *_: MetadataOnly())
    questions = Questions(
        {
            "Model family": ["Chronos", "TimesFM"],
            "Model generation": "Bolt",
            "Forecast task": ["native_point", ""],
            "Context length": ["9", "/back", ""],
            "Forecast horizon": ["2", ""],
            "Dataset path": ["/back", ""],
            "Evaluation stride": "1",
            "MASE seasonal lag": "2",
            _RUN: "yes",
        }
    )
    config = build(setup, questions)
    assert config.model["adapter_id"] == "timesfm_2p5"
    assert config.evaluation.point_statistic == "native_point"
    assert config.evaluation.context_length == 9
    assert config.evaluation.horizon == 2
    assert config.evaluation.stride == 1
    assert config.evaluation.seasonal_period == 2
    assert not setup["workspace"].exists()


def test_new_dataset_refreshes_generated_bounds_and_preserves_numeric_edits(
    setup, tmp_path
):
    new_path = tmp_path / "different-dates.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-02-01", periods=32, freq="h", tz="UTC"),
            "demand": range(32),
        }
    ).to_csv(new_path, index=False)
    questions = Questions(
        {
            "Dataset path": ["", str(new_path)],
            "Target units": "declared_units",
            "Context length": "6",
            "Forecast horizon": "2",
            "Evaluation stride": "1",
            "MASE seasonal lag": "3",
            "Experiment name": ["first", "/back", "second"],
            _RUN: ["back", "yes"],
        }
    )
    config = build(setup, questions)
    assert config.dataset["path"] == str(new_path)
    assert config.dataset["units"] == "declared_units"
    assert config.evaluation.context_length == 6
    assert config.evaluation.horizon == 2
    assert config.evaluation.stride == 1
    assert config.evaluation.seasonal_period == 3
    assert config.evaluation.evaluation_start == pd.Timestamp("2026-02-01T16:00Z")
    assert config.evaluation.evaluation_end == pd.Timestamp("2026-02-02T08:00Z")
    assert questions.count("First target timestamp (inclusive)") == 2
    assert questions.count("Target end timestamp (exclusive)") == 2
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
