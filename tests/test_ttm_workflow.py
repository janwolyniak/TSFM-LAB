"""Pinned TTM profile selection and local artifacts without model dependencies."""

import copy
import io
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from rich.console import Console
from test_repl_wizard import Questions
from test_ttm import configuration, write_checkpoint
from typer.testing import CliRunner

from tsfm_lab.application import run_from_config
from tsfm_lab.artifact_validation import validate_predictions
from tsfm_lab.cli import app
from tsfm_lab.contracts import ForecastOutput, ModelCapabilities, ModelMetadata
from tsfm_lab.experiments import ExperimentStore, StoreError
from tsfm_lab.models.catalog import (
    CHECKPOINT_REPOSITORIES,
    CURATED_REPOSITORIES,
    catalog_info,
    checkpoint_info,
)
from tsfm_lab.models.ttm_profile import validate_frequency
from tsfm_lab.repl import wizard
from tsfm_lab.repl.records import RecordCommands
from tsfm_lab.tui import forms

REPOSITORY = "ibm-granite/granite-timeseries-ttm-r2"
REFERENCES = {
    "512-96-r2": "d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f",
    "1024-96-r2": "1a382618f3c5b63836281ad632e54724e55e85a5",
    "1536-96-r2": "f6ed09d67cf42e4877f7fa33d98326911385693d",
}


@pytest.fixture
def long_demo(tmp_path):
    path = tmp_path / "long-seasonal.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=1600, freq="h", tz="UTC"),
            "demand": 20 + np.arange(1600) % 24,
        }
    ).to_csv(path, index=False)
    return path


@pytest.mark.parametrize("name,revision", REFERENCES.items())
def test_same_repository_profiles_have_exact_limits_and_pinned_information(
    name, revision
):
    context = int(name.split("-")[0])
    entry = catalog_info(REPOSITORY, name)
    assert CHECKPOINT_REPOSITORIES["ttm_r2"] == (REPOSITORY,)
    assert CURATED_REPOSITORIES.count(REPOSITORY) == 1
    assert len(catalog_info("ttm_r2")["variants"]) == 3
    assert entry["capabilities"]["min_context"] == context
    assert entry["capabilities"]["max_context"] == context
    assert entry["capabilities"]["max_horizon"] == 96
    assert entry["point_statistic"] == "native_point"
    assert entry["supported_tasks"] == ["native_point"]
    assert entry["reference_revision"] == revision
    result = CliRunner().invoke(
        app, ["models", "info", REPOSITORY, "--variant", name, "--json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["selected_checkpoint"] == checkpoint_info(
        REPOSITORY, name
    )
    assert checkpoint_info(REPOSITORY)["variant"] == "512-96-r2"
    if context != 512:
        assert entry["parameter_count"] is None


def test_pull_variant_selects_its_exact_reference_and_default_stays_main(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "tsfm_lab.model_commands.pull_checkpoint",
        lambda repository, revision, **options: calls.append((repository, revision))
        or SimpleNamespace(record=lambda: {}),
    )
    for name, revision in REFERENCES.items():
        result = CliRunner().invoke(
            app, ["models", "pull", REPOSITORY, "--variant", name, "--json"]
        )
        assert result.exit_code == 0, result.output
        assert calls[-1] == (REPOSITORY, revision)
    result = CliRunner().invoke(app, ["models", "pull", REPOSITORY, "--json"])
    assert result.exit_code == 0, result.output
    assert calls[-1] == (REPOSITORY, "main")


@pytest.mark.parametrize(
    "options",
    [["--variant", "unknown"], ["--variant", "512-96-r2", "--revision", "main"]],
)
def test_invalid_profile_or_revision_combination_never_pulls(monkeypatch, options):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid profile selection must stop before network access")

    monkeypatch.setattr("tsfm_lab.model_commands.pull_checkpoint", forbidden)
    result = CliRunner().invoke(app, ["models", "pull", REPOSITORY, *options])
    assert result.exit_code != 0
    assert "Resolving" not in result.output


@pytest.fixture
def guided_model(monkeypatch):
    class MetadataModel:
        def __init__(self, selection):
            self.context = next(
                (
                    int(name.split("-")[0])
                    for name, revision in REFERENCES.items()
                    if revision == selection.get("revision")
                ),
                512,
            )

        def metadata(self):
            return ModelMetadata(
                "ttm_r2",
                "fixture",
                "ttm",
                ModelCapabilities(
                    min_context=self.context, max_context=self.context, max_horizon=96
                ),
            )

        def configuration(self):
            return {}

        def validate(self, spec):
            assert spec.context_length == self.context
            assert spec.point_statistic == "native_point"

        def validate_frequency(self, frequency):
            validate_frequency(frequency)

    def build(selection, runtime):
        if selection["adapter_id"] == "ttm_r2":
            return MetadataModel(selection)
        from tsfm_lab.models.baselines import LastValue

        return LastValue()

    monkeypatch.setattr(wizard, "build_model", build)
    monkeypatch.setattr(forms, "build_model", build)


@pytest.mark.parametrize("name,revision", REFERENCES.items())
def test_guided_profile_uses_native_task_exact_context_and_long_default_fixture(
    tmp_path, config_factory, long_demo, guided_model, name, revision
):
    config_factory()
    questions = Questions(
        {
            "Context/horizon profile": checkpoint_info(REPOSITORY, name)[
                "display_name"
            ],
            "Run experiment?": "yes",
        }
    )
    emitted = []
    config = wizard.build_experiment(
        questions,
        emitted.append,
        workspace=tmp_path / "runs",
        demo_path=tmp_path / "seasonal.csv",
        ttm_demo_path=long_demo,
        adapter="ttm_r2",
    )
    assert config is not None, emitted
    assert config.model["repository"] == REPOSITORY
    assert config.model["revision"] == revision
    assert config.evaluation.point_statistic == "native_point"
    assert config.evaluation.context_length == int(name.split("-")[0])
    assert config.dataset["path"] == str(long_demo)
    assert config.dataset["target_column"] == "demand"
    assert "variant" not in config.model and "variant" not in config.model["options"]
    assert questions.count("Context/horizon profile") == 1
    assert questions.count("Timestamp column") == 0
    assert "Forecast task: native_point" in "\n".join(emitted)
    assert f"Context exactly {int(name.split('-')[0])}" in "\n".join(emitted)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("change_profile", [False, True])
def test_guide_back_preserves_override_until_same_repository_profile_changes(
    tmp_path, long_demo, guided_model, change_profile
):
    override = "a" * 40
    second = "1024-96-r2" if change_profile else "512-96-r2"
    questions = Questions(
        {
            "Context/horizon profile": [
                checkpoint_info(REPOSITORY, "512-96-r2")["display_name"],
                checkpoint_info(REPOSITORY, second)["display_name"],
            ],
            "Advanced checkpoint options?": ["yes", "yes"],
            "Exact checkpoint SHA": [override, ""],
            "Dataset path": ["/back", ""],
            "Context length": ["", "/back", ""],
            "Run experiment?": "yes",
        }
    )
    config = wizard.build_experiment(
        questions,
        lambda text: None,
        workspace=tmp_path / "runs",
        demo_path=long_demo,
        adapter="ttm_r2",
    )
    assert config is not None
    assert config.model["revision"] == (
        REFERENCES[second] if change_profile else override
    )
    assert config.evaluation.context_length == (1024 if change_profile else 512)


def test_family_switch_changes_only_a_default_dataset_path(
    tmp_path, config_factory, long_demo, guided_model
):
    config_factory()
    ordinary = tmp_path / "seasonal.csv"
    questions = Questions(
        {
            "Model family": [
                "IBM Granite TTM",
                "Baselines",
                "IBM Granite TTM",
                "Baselines",
            ]
        }
    )
    guide = wizard._Guide(
        questions, lambda text: None, tmp_path / "runs", ordinary, None, long_demo
    )
    guide.model()
    assert guide.fields["dataset_path"] == str(long_demo)
    assert guide.fields["point_statistic"] == "native_point"
    guide.model()
    assert guide.fields["dataset_path"] == str(ordinary)
    assert guide.fields["point_statistic"] == "median"
    custom = tmp_path / "my-data.csv"
    guide.fields["dataset_path"] = str(custom)
    guide.model()
    guide.model()
    assert guide.fields["dataset_path"] == str(custom)


def test_forms_use_native_default_and_refuse_daily_frequency_before_validation(
    tmp_path, long_demo, guided_model, monkeypatch
):
    fields = forms.default_fields("ttm_r2")
    assert (
        fields["context_length"] == "512"
        and fields["point_statistic"] == "native_point"
    )
    fields.update(dataset_path=str(long_demo), revision=REFERENCES["512-96-r2"])
    arguments = {"workspace": tmp_path / "runs", "base_directory": tmp_path}
    dataset = forms.inspect_data(fields, **arguments)
    dataset = replace(
        dataset,
        timestamps=pd.date_range("2026-01-01", periods=1600, freq="D", tz="UTC"),
        frequency="1D",
    )
    fields["frequency"] = "1D"
    fields.update(forms.suggested_bounds(dataset, fields))
    monkeypatch.setattr(forms, "load_dataset", lambda selection: dataset)
    with pytest.raises(ValueError, match="one minute to one hour"):
        forms.prepare_experiment(fields, **arguments)


def test_record_report_separates_generic_task_from_actual_deterministic_output(
    tmp_path, config_factory
):
    complete = run_from_config(
        config_factory(evaluation={"point_statistic": "native_point"})
    )
    output = io.StringIO()
    RecordCommands(Console(file=output, width=120), tmp_path / "workspace").show(
        complete["run_id"]
    )
    assert "Forecast task: native_point" in output.getvalue()
    assert "Forecast output: deterministic" in output.getvalue()


@pytest.fixture
def fake_adapter(monkeypatch):
    from tsfm_lab.models.registry import adapter_profile
    from tsfm_lab.models.ttm import TTM

    class FakeTTM(TTM):
        def load(self):
            self._model = object()

        def predict(self, requests):
            assert self._model is not None
            return tuple(
                ForecastOutput(
                    request_id=request.request_id,
                    series_id=request.series_id,
                    origin=request.origin,
                    target_timestamps=request.target_timestamps,
                    point=np.full(
                        len(request.target_timestamps), request.context_values[-1]
                    ),
                    adapter_id=self.adapter_id,
                    point_statistic="native_point",
                )
                for request in requests
            )

        def environment_metadata(self):
            return {
                name: "fixture"
                for name, _ in adapter_profile("ttm_r2").runtime_requirements
            }

    monkeypatch.setattr("tsfm_lab.models.ttm.TTM", FakeTTM)


def saved_run(config_factory, directory, dataset, context=512, adapter="ttm_r2"):
    end = pd.read_csv(dataset)["timestamp"]
    evaluation = {
        "context_length": context,
        "horizon": 4,
        "stride": 4,
        "point_statistic": "native_point",
        "evaluation_start": end.iloc[-8],
        "evaluation_end": (
            pd.Timestamp(end.iloc[-1]) + pd.Timedelta(hours=1)
        ).isoformat(),
    }
    model = (
        {"source": "local", "adapter_id": "ttm_r2", "path": str(directory)}
        if adapter == "ttm_r2"
        else {"source": "builtin", "adapter_id": adapter}
    )
    return run_from_config(
        config_factory(
            model=model, dataset={"path": str(dataset)}, evaluation=evaluation
        )
    )


@pytest.mark.parametrize("context", [512, 1024, 1536])
def test_schema2_saved_runs_reload_compare_and_export_without_runtime_or_checkpoint(
    config_factory, tmp_path, long_demo, fake_adapter, context
):
    directory = tmp_path / "checkpoint"
    write_checkpoint(directory, configuration(context))
    complete = saved_run(config_factory, directory, long_demo, context)
    baseline = saved_run(config_factory, directory, long_demo, context, "last_value")
    result, frame = ExperimentStore(tmp_path / "workspace").load_result(
        complete["run_id"]
    )
    record = result["configuration"]["model"]["options"]["checkpoint"]
    assert record["schema_version"] == 2 and record["quantiles"] == []
    assert record["point_forecast"]["profile"] == f"{context}-96-r2"
    assert record["point_forecast"]["training_loss"] == "mse"
    assert set(frame.point_statistic) == {"native_point"}
    shutil.rmtree(directory)
    program = """
import importlib.abc,io,json,sys
blocked={'torch','tsfm_public','tirex2','chronos','timesfm','transformers','huggingface_hub','safetensors'}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Optional runtime import: '+fullname)
sys.meta_path.insert(0,Block())
from rich.console import Console
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.comparison import compare_runs
from tsfm_lab.exports import export_predictions
from tsfm_lab.repl.records import RecordCommands
from tsfm_lab.models import verification
class NoCurrentChecks:
    def __iter__(self):
        raise AssertionError('Saved verification must not read current evidence')
verification.PROJECT_CHECKS=NoCurrentChecks()
workspace,run_id,baseline,destination=sys.argv[1:]
record,frame=ExperimentStore(workspace).load_result(run_id)
assert record['configuration']['model']['adapter_id']=='ttm_r2'
assert record['configuration']['model']['options']['checkpoint']['schema_version']==2
assert set(frame.point_statistic)=={'native_point'}
assert compare_runs([baseline,run_id],workspace)['compatible']
assert export_predictions(run_id,workspace,destination)['rows']==8
out=io.StringIO()
RecordCommands(Console(file=out,width=120),workspace).show(run_id)
assert 'Forecast task: native_point' in out.getvalue()
assert 'Forecast output: native_point' in out.getvalue()
out=io.StringIO()
RecordCommands(Console(file=out,width=120),workspace).compare([baseline,run_id])
assert 'Output' in out.getvalue()
assert 'deterministic' in out.getvalue() and 'native_point' in out.getvalue()
assert 'model-native point accuracy' in out.getvalue()
assert not blocked.intersection(sys.modules)
"""
    output = tmp_path / "export.json"
    subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(tmp_path / "workspace"),
            complete["run_id"],
            baseline["run_id"],
            str(output),
        ],
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
    )
    assert len(json.loads(output.read_text())) == 8


@pytest.mark.parametrize(
    "mutation",
    [
        "schema1",
        "median",
        "loss",
        "other_profile",
        "context",
        "minimum",
        "task",
        "scaling",
        "horizon",
        "inverse",
        "output_label",
    ],
)
def test_saved_point_semantics_and_forecast_options_cannot_be_forged(
    config_factory, tmp_path, long_demo, fake_adapter, mutation
):
    directory = tmp_path / "checkpoint"
    write_checkpoint(directory)
    complete = saved_run(config_factory, directory, long_demo)
    result, frame = ExperimentStore(tmp_path / "workspace").load_result(
        complete["run_id"]
    )
    changed = copy.deepcopy(result)
    model = changed["configuration"]["model"]
    record = model["options"]["checkpoint"]
    options = changed["configuration"]["runtime"]["forecast_options"]
    if mutation == "schema1":
        record["schema_version"] = 1
        record.pop("point_forecast")
    elif mutation == "median":
        record["point_forecast"]["statistic"] = "median"
    elif mutation == "loss":
        record["point_forecast"]["training_loss"] = "mae"
    elif mutation == "other_profile":
        record["point_forecast"]["profile"] = "1024-96-r2"
    elif mutation == "context":
        record["native_context_length"] = 1024
    elif mutation == "minimum":
        model["capabilities"]["min_context"] = 1
    elif mutation == "task":
        changed["configuration"]["evaluation"]["point_statistic"] = "median"
    elif mutation == "scaling":
        options["external_scaling"] = "none"
    elif mutation == "horizon":
        options["horizon_policy"] = "rolling"
    elif mutation == "inverse":
        options["native_inverse"] = False
    else:
        frame["point_statistic"] = "median"
    with pytest.raises(StoreError):
        validate_predictions(frame, changed)
