"""Chronos-2 selection and checked artifacts stay independent of inference runtimes."""

import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from tsfm_lab.application import run_from_config
from tsfm_lab.artifact_validation import validate_predictions
from tsfm_lab.cli import app
from tsfm_lab.config import ConfigError, load_config
from tsfm_lab.contracts import ModelCapabilities
from tsfm_lab.experiments import ExperimentStore, StoreError
from tsfm_lab.models import checkpoints
from tsfm_lab.models.baselines import LastValue
from tsfm_lab.models.catalog import checkpoint_info
from tsfm_lab.models.verification import checkpoint_verification
from tsfm_lab.repl import wizard

SMALL = "autogluon/chronos-2-small"
STANDARD = "amazon/chronos-2"
REFERENCES = {
    SMALL: "ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a",
    STANDARD: "29ec3766d36d6f73f0696f85560a422f50e8498c",
}
FORECAST_OPTIONS = {"cross_learning": False, "limit_prediction_length": True}


@pytest.fixture
def checkpoint_directory(tmp_path):
    directory = tmp_path / "chronos2-checkpoint"
    directory.mkdir()
    configuration = {
        "architectures": ["Chronos2Model"],
        "chronos_pipeline_class": "Chronos2Pipeline",
        "model_type": "t5",
        "d_ff": 2048,
        "d_kv": 64,
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 6,
        "dense_act_fn": "relu",
        "dropout_rate": 0.1,
        "feed_forward_proj": "relu",
        "initializer_factor": 0.05,
        "is_gated_act": False,
        "layer_norm_epsilon": 1e-6,
        "pad_token_id": 0,
        "reg_token_id": 1,
        "rope_theta": 10000.0,
        "torch_dtype": "float32",
        "use_cache": True,
        "vocab_size": 2,
        "chronos_config": {
            "context_length": 8192,
            "input_patch_size": 16,
            "input_patch_stride": 16,
            "max_output_patches": 64,
            "output_patch_size": 16,
            "quantiles": [
                0.01,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.95,
                0.99,
            ],
            "time_encoding_scale": 8192,
            "use_arcsinh": True,
            "use_reg_token": True,
        },
    }
    (directory / "config.json").write_text(json.dumps(configuration))
    (directory / "model.safetensors").write_bytes(b"inert Chronos-2 fixture weights")
    return directory


@pytest.fixture
def fake_adapter(monkeypatch):
    class FakeChronos2(LastValue):
        adapter_id = "chronos_2"

        def __init__(self, checkpoint, **runtime):
            self.checkpoint, self.source = checkpoint, checkpoint.source

        def metadata(self):
            return replace(
                super().metadata(),
                family="chronos",
                capabilities=ModelCapabilities(max_context=8192, max_horizon=1024),
            )

        def configuration(self):
            return {"checkpoint": self.checkpoint.record()}

        def verification_metadata(self):
            return checkpoint_verification(self.checkpoint.record())

        def runtime_configuration(self):
            return {
                "device": "cpu",
                "dtype": "float32",
                "batch_size": 1,
                "forecast_options": dict(FORECAST_OPTIONS),
            }

        def environment_metadata(self):
            return {
                name: "fixture"
                for name in (
                    "torch",
                    "chronos-forecasting",
                    "transformers",
                    "huggingface-hub",
                    "safetensors",
                    "accelerate",
                )
            }

        def predict(self, requests):
            return tuple(
                replace(value, point_statistic="median")
                for value in super().predict(requests)
            )

    monkeypatch.setattr("tsfm_lab.models.chronos2.Chronos2", FakeChronos2)
    return FakeChronos2


@pytest.mark.parametrize("repository", REFERENCES)
def test_both_variants_are_selectable_from_cli_without_runtime_or_cache(repository):
    result = CliRunner().invoke(app, ["models", "info", repository, "--json"])
    assert result.exit_code == 0, result.output
    info = json.loads(result.stdout)
    assert info["adapter_id"] == "chronos_2"
    assert info["selected_checkpoint"]["reference_revision"] == REFERENCES[repository]
    assert info["capabilities"]["max_context"] == 8192
    assert info["capabilities"]["max_horizon"] == 1024
    assert info["point_statistic"] == "median"
    if repository == SMALL:
        assert info["selected_checkpoint"]["project_inference_status"] == "recorded"
    if repository == STANDARD:
        assert info["selected_checkpoint"]["project_inference_status"] == "not_recorded"
        assert (
            "individual inference not run"
            in info["selected_checkpoint"]["verification_label"]
        )


@pytest.mark.parametrize("repository", REFERENCES)
def test_yaml_factory_uses_cached_exact_reference_without_metadata_or_downloads(
    repository,
    config_factory,
    checkpoint_directory,
    fake_adapter,
    tmp_path,
    monkeypatch,
):
    cache = tmp_path / "cache"
    snapshot = (
        cache
        / ("models--" + repository.replace("/", "--"))
        / "snapshots"
        / REFERENCES[repository]
    )
    shutil.copytree(checkpoint_directory, snapshot)
    calls = []

    def cached(repo, **options):
        assert repo == repository
        assert options["local_files_only"] is True
        assert options["revision"] == REFERENCES[repository]
        calls.append(options)
        return str(snapshot)

    def forbidden(*args, **kwargs):
        pytest.fail("Cached experiment must not contact metadata or download weights")

    monkeypatch.setattr(
        checkpoints,
        "_hub",
        lambda: SimpleNamespace(
            snapshot_download=cached, HfApi=forbidden, hf_hub_download=forbidden
        ),
    )
    path = config_factory(
        model={
            "source": "huggingface",
            "adapter_id": "chronos_2",
            "repository": repository,
            "revision": REFERENCES[repository],
            "cache_dir": str(cache),
        }
    )
    config = load_config(path)
    assert config.model["offline"] is True and config.runtime["dtype"] == "float32"
    completed = run_from_config(path)
    result, frame = ExperimentStore(tmp_path / "workspace").load_result(
        completed["run_id"]
    )
    assert len(calls) == 1 and len(frame) == 16
    record = result["configuration"]["model"]["options"]["checkpoint"]
    assert record["resolved_revision"] == REFERENCES[repository]
    assert record["architecture"] == "Chronos2Model"
    assert record["native_prediction_length"] == 1024


def test_saved_results_reload_compare_and_export_without_runtime_or_checkpoint(
    config_factory, checkpoint_directory, fake_adapter, tmp_path
):
    baseline = run_from_config(config_factory())
    result = run_from_config(
        config_factory(
            model={
                "source": "local",
                "adapter_id": "chronos_2",
                "path": str(checkpoint_directory),
            }
        )
    )
    checkpoint_directory.rename(tmp_path / "unavailable-checkpoint")
    program = """
import importlib.abc, json, sys
blocked = {'torch', 'chronos', 'timesfm', 'transformers', 'huggingface_hub', 'safetensors'}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Optional runtime import: ' + fullname)
sys.meta_path.insert(0, Block())
from tsfm_lab.comparison import compare_runs
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.exports import export_predictions
workspace, run_id, baseline, destination = sys.argv[1:]
record, frame = ExperimentStore(workspace).load_result(run_id)
assert record['configuration']['model']['adapter_id'] == 'chronos_2'
assert record['configuration']['runtime']['forecast_options'] == {
    'cross_learning': False, 'limit_prediction_length': True}
assert record['provenance']['project_verification']['selected_checkpoint_checks'] == []
assert compare_runs([baseline, run_id], workspace)['compatible']
assert export_predictions(run_id, workspace, destination)['rows'] == len(frame)
assert not blocked.intersection(sys.modules)
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(tmp_path / "workspace"),
            result["run_id"],
            baseline["run_id"],
            str(tmp_path / "predictions.json"),
        ],
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
    )
    assert len(json.loads((tmp_path / "predictions.json").read_text())) == 16


@pytest.mark.parametrize(
    "options",
    [
        None,
        {},
        {"cross_learning": True, "limit_prediction_length": True},
        {"cross_learning": False, "limit_prediction_length": False},
        {"cross_learning": 0, "limit_prediction_length": True},
        {**FORECAST_OPTIONS, "invented": False},
    ],
)
def test_saved_runtime_requires_explicit_independent_native_forecast_policy(
    options, config_factory, checkpoint_directory, fake_adapter, tmp_path
):
    completed = run_from_config(
        config_factory(
            model={
                "source": "local",
                "adapter_id": "chronos_2",
                "path": str(checkpoint_directory),
            }
        )
    )
    result, frame = ExperimentStore(tmp_path / "workspace").load_result(
        completed["run_id"]
    )
    modified = deepcopy(result)
    modified["configuration"]["runtime"]["forecast_options"] = options
    with pytest.raises(StoreError):
        validate_predictions(frame, modified)


class Questions:
    def __init__(self, answers):
        self.answers, self.calls = defaultdict(deque), []
        for label, values in answers.items():
            self.answers[label].extend(values if isinstance(values, list) else [values])

    def __call__(self, label, default="", choices=()):
        self.calls.append((label, default, choices))
        return self.answers[label].popleft() if self.answers[label] else ""


@pytest.mark.parametrize("second_repository", [SMALL, STANDARD])
def test_terminal_selection_preserves_override_or_defaults_changed_variant(
    second_repository, config_factory, tmp_path, monkeypatch
):
    config_factory()

    class MetadataOnly:
        def metadata(self):
            return SimpleNamespace(
                capabilities=ModelCapabilities(max_context=8192, max_horizon=1024)
            )

        def validate(self, spec):
            pass

        def load(self):
            pytest.fail("Model setup must not load weights")

    monkeypatch.setattr(wizard, "build_model", lambda *_: MetadataOnly())
    monkeypatch.setattr("tsfm_lab.tui.forms.build_model", lambda *_: MetadataOnly())
    override = "b" * 40
    questions = Questions(
        {
            "Model family": "Chronos",
            "Model generation": "2",
            "Model size": [
                checkpoint_info(SMALL)["display_name"],
                checkpoint_info(second_repository)["display_name"],
            ],
            "Advanced checkpoint options?": ["yes", "yes"],
            "Exact checkpoint SHA": [override, ""],
            "Dataset path": ["/back", ""],
            "Context length": ["", "/back", ""],
            "Run experiment?": "yes",
        }
    )
    messages = []
    workspace = tmp_path / "wizard-results"
    config = wizard.build_experiment(
        questions,
        messages.append,
        workspace=workspace,
        demo_path=tmp_path / "seasonal.csv",
    )
    assert config.model["adapter_id"] == "chronos_2"
    assert config.model["repository"] == second_repository
    expected = override if second_repository == SMALL else REFERENCES[STANDARD]
    assert config.model["revision"] == expected and config.model["offline"] is True
    assert [
        call[1] for call in questions.calls if call[0] == "Exact checkpoint SHA"
    ] == [REFERENCES[SMALL], expected]
    choices = next(call[2] for call in questions.calls if call[0] == "Model size")
    assert choices == ("Small", "Standard")
    assert any(second_repository in line for line in messages)
    assert not workspace.exists()


def test_terminal_quick_demo_requires_explicit_chronos2_checkpoint(tmp_path):
    with pytest.raises(ConfigError, match="guided setup"):
        wizard.build_experiment(
            lambda *_: "",
            lambda _: None,
            workspace=tmp_path / "results",
            demo_path=tmp_path / "demo.csv",
            adapter="chronos_2",
            demo=True,
        )


@pytest.mark.parametrize("repository", REFERENCES)
def test_reference_information_is_detached_and_smaller_check_is_not_copied(repository):
    info = checkpoint_info(repository)
    info["reference_revision"] = "0" * 40
    assert checkpoint_info(repository)["reference_revision"] == REFERENCES[repository]
    if repository == STANDARD:
        assert checkpoint_info(repository)["project_inference_status"] == "not_recorded"
