"""TiRex-2 selection and saved artifacts without installed model libraries."""

import copy
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_tirex2 import configuration, write_checkpoint
from typer.testing import CliRunner

from tsfm_lab.application import run_from_config
from tsfm_lab.artifact_validation import validate_predictions
from tsfm_lab.cli import app
from tsfm_lab.config import load_config
from tsfm_lab.contracts import ModelCapabilities
from tsfm_lab.experiments import ExperimentStore, StoreError
from tsfm_lab.models import checkpoints
from tsfm_lab.models.baselines import LastValue
from tsfm_lab.models.verification import checkpoint_verification

REFERENCES = {
    "NX-AI/TiRex-2": "30788a311699acba7e6393ad9700ea7b7cb9bb2d",
    "NX-AI/TiRex-2-fevbench": "acd6153b8ee2179f38cfd3cc4fd64a3c045a424f",
    "NX-AI/TiRex-2-gifteval-zs": "c94872d1c8caacbe7755cbfa01bc5c4c5c5ddb7f",
    "NX-AI/TiRex-2-gifteval-pretrain": "54f88ac06ccfba9fe3e727aa8856b39775481f0f",
}


def flags(gifteval=False):
    return {"tta_sign_flip": gifteval, "tta_diff": not gifteval, "pad_context": True}


@pytest.fixture
def fake_adapter(monkeypatch):
    class FakeTiRex2(LastValue):
        adapter_id = "tirex_2"

        def __init__(self, checkpoint, **runtime):
            self.checkpoint, self.source = checkpoint, checkpoint.source
            self.record = checkpoint.record()

        def metadata(self):
            return replace(
                super().metadata(),
                family="tirex",
                capabilities=ModelCapabilities(
                    max_context=self.record["native_context_length"],
                    max_horizon=self.record["native_prediction_length"],
                ),
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
                "torch_compile": False,
                "forecast_options": flags(self.record["native_context_length"] == 8192),
            }

        def environment_metadata(self):
            return {
                name: "fixture"
                for name in (
                    "torch",
                    "tirex-2",
                    "huggingface-hub",
                    "safetensors",
                    "numpy",
                    "flashrnn",
                    "mlstm-kernels",
                    "ninja",
                    "narwhals",
                )
            }

        def predict(self, requests):
            return tuple(
                replace(value, point_statistic="median")
                for value in super().predict(requests)
            )

    monkeypatch.setattr("tsfm_lab.models.tirex2.TiRex2", FakeTiRex2)
    return FakeTiRex2


@pytest.mark.parametrize("repository", REFERENCES)
def test_all_four_profiles_have_pinned_metadata_and_actual_limits(repository):
    result = CliRunner().invoke(app, ["models", "info", repository, "--json"])
    assert result.exit_code == 0, result.output
    info = json.loads(result.stdout)
    assert info["adapter_id"] == "tirex_2"
    assert info["selected_checkpoint"]["reference_revision"] == REFERENCES[repository]
    expected = (8192, 928) if "gifteval" in repository else (2048, 320)
    assert (
        info["capabilities"]["max_context"],
        info["capabilities"]["max_horizon"],
    ) == expected
    assert info["point_statistic"] == "median"
    if repository != "NX-AI/TiRex-2":
        assert info["selected_checkpoint"]["project_inference_status"] == "not_recorded"


@pytest.mark.parametrize("repository", REFERENCES)
def test_exact_cached_yaml_factory_requires_no_metadata_or_download(
    repository, config_factory, tmp_path, monkeypatch, fake_adapter
):
    cache = tmp_path / "cache"
    snapshot = (
        cache
        / ("models--" + repository.replace("/", "--"))
        / "snapshots"
        / REFERENCES[repository]
    )
    write_checkpoint(snapshot, configuration("gifteval" in repository))
    calls = []

    def cached(repo, **options):
        assert repo == repository and options["revision"] == REFERENCES[repository]
        assert options["local_files_only"] is True
        calls.append(options)
        return str(snapshot)

    def forbidden(*args, **kwargs):
        pytest.fail("Cached execution must not resolve metadata or download weights")

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
            "adapter_id": "tirex_2",
            "repository": repository,
            "revision": REFERENCES[repository],
            "cache_dir": str(cache),
        }
    )
    config = load_config(path)
    assert config.model["offline"] is True and config.runtime["dtype"] == "float32"
    complete = run_from_config(path)
    result, frame = ExperimentStore(tmp_path / "workspace").load_result(
        complete["run_id"]
    )
    assert len(frame) == 16 and len(calls) == 1
    record = result["configuration"]["model"]["options"]["checkpoint"]
    assert (
        record["repository"] == repository
        and record["resolved_revision"] == REFERENCES[repository]
    )
    assert record["architecture"] == "TiRex2"
    assert result["configuration"]["runtime"]["forecast_options"] == flags(
        "gifteval" in repository
    )
    assert (
        result["provenance"]["project_verification"]["selected_checkpoint_checks"] == []
    )


@pytest.mark.parametrize("gifteval", [False, True])
def test_real_saved_schema_reload_compare_export_without_runtime_or_checkpoint(
    config_factory, tmp_path, fake_adapter, gifteval
):
    directory = tmp_path / "checkpoint"
    write_checkpoint(directory, configuration(gifteval))
    baseline = run_from_config(config_factory())
    complete = run_from_config(
        config_factory(
            model={"source": "local", "adapter_id": "tirex_2", "path": str(directory)}
        )
    )
    shutil.rmtree(directory)
    program = """
import importlib.abc,json,sys
blocked={'torch','tirex2','flashrnn','mlstm_kernels','chronos','timesfm','transformers','huggingface_hub','safetensors'}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('Optional runtime import: '+fullname)
sys.meta_path.insert(0,Block())
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.comparison import compare_runs
from tsfm_lab.exports import export_predictions
from tsfm_lab.models import verification
class NoCurrentChecks:
    def __iter__(self):
        raise AssertionError('Saved verification must not read current evidence')
verification.PROJECT_CHECKS=NoCurrentChecks()
workspace,run_id,baseline,destination=sys.argv[1:]
record,frame=ExperimentStore(workspace).load_result(run_id)
assert record['configuration']['model']['adapter_id']=='tirex_2'
assert record['provenance']['project_verification']['selected_checkpoint_checks']==[]
assert compare_runs([baseline,run_id],workspace)['compatible']
assert export_predictions(run_id,workspace,destination)['rows']==16
assert not blocked.intersection(sys.modules)
"""
    output = tmp_path / "predictions.json"
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
    assert len(json.loads(output.read_text())) == 16


@pytest.mark.parametrize("gifteval", [False, True])
@pytest.mark.parametrize(
    "mutation",
    ["missing", "other_profile", "bad_boolean", "padding_off", "extra", "compile"],
)
def test_saved_forecast_options_are_bound_to_actual_checkpoint_profile(
    config_factory, tmp_path, fake_adapter, gifteval, mutation
):
    directory = tmp_path / "checkpoint"
    write_checkpoint(directory, configuration(gifteval))
    complete = run_from_config(
        config_factory(
            model={"source": "local", "adapter_id": "tirex_2", "path": str(directory)}
        )
    )
    result, frame = ExperimentStore(tmp_path / "workspace").load_result(
        complete["run_id"]
    )
    changed = copy.deepcopy(result)
    runtime = changed["configuration"]["runtime"]
    if mutation == "missing":
        runtime.pop("forecast_options")
    elif mutation == "other_profile":
        runtime["forecast_options"] = flags(not gifteval)
    elif mutation == "bad_boolean":
        runtime["forecast_options"]["tta_diff"] = int(not gifteval)
    elif mutation == "padding_off":
        runtime["forecast_options"]["pad_context"] = False
    elif mutation == "extra":
        runtime["forecast_options"]["invented"] = False
    else:
        runtime["torch_compile"] = True
    with pytest.raises(StoreError):
        validate_predictions(frame, changed)
