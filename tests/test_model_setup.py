"""Model-specific setup shares configuration identity and validates data first."""

import shutil
from collections import defaultdict, deque
from importlib.resources import files
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from tsfm_lab.config import config_from_mapping, load_config
from tsfm_lab.models.catalog import catalog_entries
from tsfm_lab.models.settings import model_settings, validate_settings
from tsfm_lab.repl import wizard


class Questions:
    def __init__(self, answers=None):
        self.answers = defaultdict(deque)
        self.calls = []
        for label, values in (answers or {}).items():
            self.answers[label].extend(values if isinstance(values, list) else [values])

    def __call__(self, label, default="", choices=()):
        assert len(self.calls) < 150, "Setup failed to reach review or cancellation"
        self.calls.append((label, default, choices))
        return self.answers[label].popleft() if self.answers[label] else ""


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "observations.csv"
    shutil.copyfile(files("tsfm_lab.resources").joinpath("ttm-seasonal.csv"), path)
    return path


def metadata_model(selection, _runtime):
    settings = model_settings(
        selection["adapter_id"],
        repository=selection.get("repository"),
        revision=selection.get("revision"),
    )
    return SimpleNamespace(
        metadata=lambda: SimpleNamespace(capabilities=settings.capabilities),
        validate=lambda spec: validate_settings(selection, spec),
    )


@pytest.mark.parametrize(
    "adapter", [entry["adapter_id"] for entry in catalog_entries()]
)
def test_manual_assisted_yaml_and_python_have_the_same_validated_identity(
    adapter, source, tmp_path, monkeypatch
):
    monkeypatch.setattr(wizard, "build_model", metadata_model)
    monkeypatch.setattr("tsfm_lab.tui.forms.build_model", metadata_model)
    arguments = {"workspace": tmp_path / "results", "demo_path": source}
    questions = Questions({"Run experiment?": "yes"})
    manual = wizard.build_experiment(
        questions, lambda _: None, adapter=adapter, **arguments
    )
    assert manual is not None
    spec = manual.evaluation
    fields = {
        "name": manual.name,
        "adapter": adapter,
        "source": manual.model["source"],
        "usage_profile": "generic",
        "dataset_path": str(source),
        "units": manual.dataset["units"],
        "context_length": str(spec.context_length),
        "horizon": str(spec.horizon),
        "stride": str(spec.stride),
        "mase_period": str(spec.seasonal_period),
        "point_statistic": spec.point_statistic,
        "evaluation_start": spec.evaluation_start.isoformat(),
        "evaluation_end": spec.evaluation_end.isoformat(),
    }
    if "repository" in manual.model:
        fields.update(
            repository=manual.model["repository"], revision=manual.model["revision"]
        )
    assisted = wizard.build_experiment(
        Questions({"Run experiment?": "yes"}),
        lambda _: None,
        seed_fields=fields,
        **arguments,
    )
    saved = tmp_path / "experiment.yaml"
    saved.write_text(manual.source_yaml)
    python = config_from_mapping(
        yaml.safe_load(manual.source_yaml), base_directory=tmp_path
    )
    assert (
        assisted.resolved_dict()
        == load_config(saved).resolved_dict()
        == python.resolved_dict()
        == manual.resolved_dict()
    )
    labels = [call[0] for call in questions.calls]
    assert labels.index("Context length") < labels.index("Dataset path")
    assert labels.index("Forecast horizon") < labels.index("Dataset path")
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("problem", ["gap", "cadence", "history", "horizon"])
def test_invalid_data_or_options_stop_before_checkpoint_access(
    problem, tmp_path, monkeypatch
):
    calls = []

    def forbidden(*args):
        calls.append(args)
        pytest.fail("Invalid inputs must not access a checkpoint")

    monkeypatch.setattr(wizard, "build_model", forbidden)
    monkeypatch.setattr("tsfm_lab.tui.forms.build_model", forbidden)
    count = 520 if problem == "cadence" else 3 if problem == "history" else 32
    cadence = "1D" if problem == "cadence" else "1h"
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01", periods=count, freq=cadence, tz="UTC"
            ),
            "demand": range(count),
        }
    )
    if problem == "gap":
        frame = frame.drop(index=2)
    path = tmp_path / "custom.csv"
    frame.to_csv(path, index=False)
    adapter = "ttm_r2" if problem == "cadence" else "chronos_bolt"
    answers = {
        "Dataset path": str(path),
        "Observation frequency": cadence,
        "Target units": "declared_units",
    }
    if problem == "history":
        answers["Context length"] = ["8", "/cancel"]
    if problem == "horizon":
        answers["Forecast horizon"] = ["65", "/cancel"]
    emitted = []
    result = wizard.build_experiment(
        Questions(answers),
        emitted.append,
        workspace=tmp_path / "results",
        demo_path=files("tsfm_lab.resources").joinpath("seasonal.csv"),
        adapter=adapter,
    )
    assert result is None
    assert calls == []
    assert not (tmp_path / "results").exists()
    text = "\n".join(emitted)
    expected = {
        "gap": "irregular interval",
        "cadence": "one minute to one hour",
        "history": "fit the inspected dataset",
        "horizon": "between 1 and 64",
    }
    assert expected[problem] in text


def test_switching_to_ttm_does_not_apply_previous_short_dataset_limits(
    source, tmp_path
):
    questions = Questions({"Model family": ["Baselines", "IBM Granite TTM"]})
    guide = wizard._Guide(
        questions,
        lambda _: None,
        tmp_path / "results",
        files("tsfm_lab.resources").joinpath("seasonal.csv"),
        None,
        source,
    )
    guide.model()
    guide.model_options()
    guide.data()
    assert len(guide.dataset.values) == 32
    guide.model()
    assert guide.dataset is None
    assert guide.fields["dataset_path"] == str(source)
    guide.model_options()
    assert guide.fields["context_length"] == "512"


def test_dataset_preview_keeps_negative_values_and_declared_units(tmp_path):
    source = tmp_path / "temperature.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=16, freq="h", tz="UTC"),
            "demand": range(-8, 8),
        }
    ).to_csv(source, index=False)
    emitted = []
    result = wizard.build_experiment(
        Questions({"Target units": "degC", "Run experiment?": "yes"}),
        emitted.append,
        workspace=tmp_path / "results",
        demo_path=source,
        seed_fields={"units": "degC"},
    )
    assert result.dataset["units"] == "degC"
    assert any("Dataset preview" in line for line in emitted)
    assert any("-8 degC" in line for line in emitted)
    assert any("7 degC" in line for line in emitted)


def test_invalid_seeded_evaluation_fields_can_be_corrected_after_data(source, tmp_path):
    questions = Questions(
        {
            "Evaluation stride": ["", "2"],
            "MASE seasonal lag": ["", "3"],
            "Run experiment?": "yes",
        }
    )
    result = wizard.build_experiment(
        questions,
        lambda _: None,
        workspace=tmp_path / "results",
        demo_path=source,
        seed_fields={"stride": "0", "mase_period": "0", "units": "synthetic_units"},
    )
    assert result.evaluation.stride == 2
    assert result.evaluation.seasonal_period == 3
    labels = [label for label, _, _ in questions.calls]
    assert labels.index("Dataset path") < labels.index("Evaluation stride")
