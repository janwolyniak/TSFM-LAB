"""TimesFM selections keep family, offline, and runtime boundaries explicit."""

import builtins
from pathlib import Path

import pytest

from tsfm_lab.config import ConfigError, config_from_mapping

_REPOSITORY = "google/timesfm-2.5-200m-pytorch"
_REVISION = "1d952420fba87f3c6dee4f240de0f1a0fbc790e3"


def mapping(source="huggingface"):
    model = {"source": source, "adapter_id": "timesfm_2p5"}
    if source == "huggingface":
        model.update(repository=_REPOSITORY, revision=_REVISION)
    else:
        model["path"] = "models/timesfm"
    return {
        "schema_version": 1,
        "model": model,
        "dataset": {
            "path": "series.csv",
            "timestamp_column": "timestamp",
            "target_column": "target",
            "frequency": "1h",
        },
        "evaluation": {
            "context_length": 8,
            "horizon": 4,
            "evaluation_start": "2026-01-01T08:00:00Z",
            "evaluation_end": "2026-01-02T08:00:00Z",
        },
    }


@pytest.mark.parametrize("source", ["huggingface", "local"])
def test_timesfm_defaults_are_offline_cpu_float32_without_file_or_runtime_access(
    tmp_path, monkeypatch, source
):
    original_import = builtins.__import__

    def no_runtime(name, *args, **kwargs):
        if name.split(".")[0] in {"timesfm", "torch", "huggingface_hub"}:
            raise AssertionError("Selection must not import model runtimes")
        return original_import(name, *args, **kwargs)

    def no_open(*args, **kwargs):
        raise AssertionError("Selection must not open model or dataset files")

    monkeypatch.setattr(builtins, "__import__", no_runtime)
    monkeypatch.setattr(Path, "open", no_open)
    config = config_from_mapping(mapping(source), base_directory=tmp_path)
    assert config.model["adapter_id"] == "timesfm_2p5"
    assert config.runtime == {
        "device": "cpu",
        "dtype": "float32",
        "batch_size": 1,
        "seed": 0,
    }
    if source == "huggingface":
        assert config.model["offline"] is True
        assert config.model["revision"] == _REVISION
    else:
        assert config.model["path"] == str(tmp_path / "models/timesfm")
    assert not config.output_directory.exists()


@pytest.mark.parametrize(
    "adapter,repository",
    [
        ("timesfm_2p5", "amazon/chronos-bolt-tiny"),
        ("chronos_bolt", _REPOSITORY),
    ],
)
def test_checkpoint_adapter_cannot_select_another_family(tmp_path, adapter, repository):
    value = mapping()
    value["model"].update(adapter_id=adapter, repository=repository)
    with pytest.raises(ConfigError, match="curated"):
        config_from_mapping(value, base_directory=tmp_path)


@pytest.mark.parametrize("context,horizon", [(1, 1), (16256, 128), (129, 127)])
def test_timesfm_context_and_native_horizon_boundaries_are_accepted(
    tmp_path, context, horizon
):
    value = mapping()
    value["evaluation"].update(context_length=context, horizon=horizon)
    config = config_from_mapping(value, base_directory=tmp_path)
    assert config.evaluation.context_length == context
    assert config.evaluation.horizon == horizon


@pytest.mark.parametrize("changes", [{"context_length": 16257}, {"horizon": 129}])
def test_timesfm_refuses_context_truncation_and_horizon_extension(tmp_path, changes):
    value = mapping()
    value["evaluation"].update(changes)
    with pytest.raises(ConfigError, match="context_length <= 16256 and horizon <= 128"):
        config_from_mapping(value, base_directory=tmp_path)


@pytest.mark.parametrize(
    "runtime",
    [{"device": "cuda"}, {"device": "mps"}, {"dtype": "float64"}, {"batch_size": 2}],
)
def test_timesfm_rejects_unverified_runtime_choices(tmp_path, runtime):
    value = mapping() | {"runtime": runtime}
    with pytest.raises(ConfigError):
        config_from_mapping(value, base_directory=tmp_path)


@pytest.mark.parametrize("revision", ["main", "A" * 40, "a" * 39])
def test_timesfm_offline_revision_must_be_exact(tmp_path, revision):
    value = mapping()
    value["model"]["revision"] = revision
    with pytest.raises(ConfigError, match="40-character SHA"):
        config_from_mapping(value, base_directory=tmp_path)
