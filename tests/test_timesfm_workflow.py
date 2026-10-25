"""TimesFM saved runs remain comparable and usable without optional runtimes."""

import copy
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tsfm_lab.application import run_from_config
from tsfm_lab.artifact_validation import validate_predictions
from tsfm_lab.comparison import compare_runs
from tsfm_lab.experiments import ExperimentStore, StoreError
from tsfm_lab.models import checkpoints


def local_config(factory, path):
    return factory(
        model={"source": "local", "adapter_id": "timesfm_2p5", "path": str(path)}
    )


def test_source_selection_records_family_runtime_and_independent_capabilities(
    config_factory, timesfm_checkpoint, timesfm_runtime, tmp_path
):
    value = run_from_config(local_config(config_factory, timesfm_checkpoint))
    result, frame = ExperimentStore(tmp_path / "workspace").load_result(value["run_id"])
    model = result["configuration"]["model"]
    assert model["family"] == "timesfm" and model["adapter_id"] == "timesfm_2p5"
    assert model["capabilities"]["max_context"] == 16256
    assert model["options"]["checkpoint"]["native_context_length"] == 16384
    assert result["provenance"]["environment"]["timesfm"] == "3.0.2"
    assert len(frame) == 16 and set(frame.point_statistic) == {"median"}
    assert len(timesfm_runtime.load_calls) == 1


def test_offline_hub_selection_never_resolves_or_downloads(
    config_factory, timesfm_checkpoint, timesfm_runtime, tmp_path, monkeypatch
):
    revision = "a" * 40
    snapshot = (
        tmp_path / "models--google--timesfm-2.5-200m-pytorch" / "snapshots" / revision
    )
    snapshot.parent.mkdir(parents=True)
    timesfm_checkpoint.rename(snapshot)
    calls = []

    def cached(repository, **options):
        assert repository == "google/timesfm-2.5-200m-pytorch"
        assert options["revision"] == revision and options["local_files_only"] is True
        calls.append(options)
        return str(snapshot)

    def no_network(**options):
        raise AssertionError("Offline execution cannot resolve online metadata")

    monkeypatch.setattr(
        checkpoints,
        "_hub",
        lambda: SimpleNamespace(snapshot_download=cached, HfApi=no_network),
    )
    path = config_factory(
        model={
            "source": "huggingface",
            "adapter_id": "timesfm_2p5",
            "repository": "google/timesfm-2.5-200m-pytorch",
            "revision": revision,
        }
    )
    value = run_from_config(path)
    result, _ = ExperimentStore(tmp_path / "workspace").load_result(value["run_id"])
    assert len(calls) == 1
    assert (
        result["configuration"]["model"]["options"]["checkpoint"]["resolved_revision"]
        == revision
    )


def test_reload_comparison_and_export_need_no_weights_or_optional_libraries(
    config_factory, timesfm_checkpoint, timesfm_runtime, tmp_path
):
    baseline = run_from_config(config_factory())
    value = run_from_config(local_config(config_factory, timesfm_checkpoint))
    workspace = tmp_path / "workspace"
    assert compare_runs([baseline["run_id"], value["run_id"]], workspace)["compatible"]
    timesfm_checkpoint.rename(tmp_path / "unavailable-checkpoint")
    program = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'timesfm', 'chronos', 'transformers', 'huggingface_hub', 'safetensors'}:
            raise AssertionError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, Block())
from tsfm_lab.experiments import ExperimentStore
from tsfm_lab.comparison import compare_runs
from tsfm_lab.exports import export_predictions
workspace, baseline, selected, target = sys.argv[1:]
result, predictions = ExperimentStore(workspace).load_result(selected)
assert result['status'] == 'completed' and len(predictions) == 16
assert compare_runs([baseline, selected], workspace)['compatible']
assert export_predictions(selected, workspace, target)['rows'] == 16
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(workspace),
            baseline["run_id"],
            value["run_id"],
            str(tmp_path / "export.csv"),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "fault",
    [
        "adapter",
        "family",
        "context",
        "dtype",
        "timesfm_version",
        "forecast_option",
        "torch_compile",
    ],
)
def test_corrupt_family_and_runtime_records_are_rejected(
    config_factory, timesfm_checkpoint, timesfm_runtime, tmp_path, fault
):
    value = run_from_config(local_config(config_factory, timesfm_checkpoint))
    result, frame = ExperimentStore(tmp_path / "workspace").load_result(value["run_id"])
    changed = copy.deepcopy(result)
    model = changed["configuration"]["model"]
    runtime = changed["configuration"]["runtime"]
    if fault == "adapter":
        model["adapter_id"] = "chronos_bolt"
    elif fault == "family":
        model["family"] = "chronos"
    elif fault == "context":
        model["capabilities"]["max_context"] = 16384
    elif fault == "dtype":
        runtime["dtype"] = "float64"
    elif fault == "timesfm_version":
        del changed["provenance"]["environment"]["timesfm"]
    elif fault == "forecast_option":
        runtime["forecast_options"]["infer_is_positive"] = True
    else:
        runtime["torch_compile"] = True
    with pytest.raises(StoreError):
        validate_predictions(frame, changed)
