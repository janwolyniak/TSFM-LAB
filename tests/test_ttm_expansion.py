"""S6 TTM profiles exercise real orchestration with inert weights/fake tensors."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_future_setup import Questions
from test_ttm import FakeTensor, write_checkpoint
from test_ttm import fake_runtime as shared_runtime

from tsfm_lab.application import run_config
from tsfm_lab.artifact_validation import validate_predictions
from tsfm_lab.config import ConfigError, config_from_mapping
from tsfm_lab.contracts import ForecastRequest
from tsfm_lab.experiments import ExperimentStore, StoreError
from tsfm_lab.exports import export_predictions
from tsfm_lab.future_application import run_future_config
from tsfm_lab.future_config import future_config_from_mapping
from tsfm_lab.future_store import FutureStore
from tsfm_lab.guided import _native_bounds
from tsfm_lab.models import ttm
from tsfm_lab.models.catalog import catalog_info, checkpoint_info
from tsfm_lab.models.checkpoint_profiles import config_metadata
from tsfm_lab.models.checkpoints import CheckpointError, validate_checkpoint_record
from tsfm_lab.models.settings import model_settings, validate_settings
from tsfm_lab.models.ttm_profile import point_metadata
from tsfm_lab.repl.wizard import build_experiment

REPOSITORY = "ibm-granite/granite-timeseries-ttm-r2"
FIXTURES = Path(__file__).parent / "fixtures" / "ttm_s6"
fake_runtime = shared_runtime
PROFILES = {
    "52-16-ft-r2.1": (
        52,
        16,
        "7D",
        9,
        "2276979b72771bf1c460356b9211fa24479a0546d5c03ddf52cccdb0f86c2b44",
    ),
    "90-30-ft-r2.1": (
        90,
        30,
        "1D",
        8,
        "abf1b5a26903f1d6b8eb62e66a72bf85bb0ced591b5ffaa655ec031b4d1ec326",
    ),
    "512-192-r2": (
        512,
        192,
        "1h",
        None,
        "012025d3a082d44c593f5f23a0b623be427e628d19094e58e22121941d42f1c8",
    ),
    "512-336-r2": (
        512,
        336,
        "1h",
        None,
        "1c25e5da9d2151047511fd6062ab93ee323a05774a879bbdda856bda406058f6",
    ),
    "512-720-r2": (
        512,
        720,
        "1h",
        None,
        "2b3f031ae0c7a33fb32621feea6381ae63c9c26d170621c3a765fc9d483835c2",
    ),
}


def configuration(name):
    return json.loads((FIXTURES / f"{name}.json").read_bytes())


def runtime(state, monkeypatch, horizon):
    torch, _ = ttm._runtime()
    torch.long = "int64"
    original = torch.as_tensor
    state.tokens = []

    def tensor(values, *, dtype, device):
        if dtype == "int64":
            assert device == "cpu"
            state.tokens.append(np.asarray(values, dtype=np.int64))
            return FakeTensor(state.tokens[-1])
        return original(values, dtype=dtype, device=device)

    monkeypatch.setattr(torch, "as_tensor", tensor)
    state.response = SimpleNamespace(
        prediction_outputs=FakeTensor(
            np.arange(horizon, dtype=np.float32)[None, :, None]
        )
    )
    return state


def mapping(tmp_path, name, *, local=True, future=False, count=None):
    context, horizon, cadence, _, _ = PROFILES[name]
    count = context + 8 if count is None else count
    stamps = pd.date_range("2020-01-01", periods=count, freq=cadence, tz="UTC")
    path = tmp_path / "data.csv"
    pd.DataFrame({"timestamp": stamps, "demand": 10 + np.arange(count) / 4}).to_csv(
        path, index=False
    )
    model = {
        "source": "local",
        "adapter_id": "ttm_r2",
        "path": str(tmp_path / "weights"),
    }
    if not local:
        model = {
            "source": "huggingface",
            "adapter_id": "ttm_r2",
            "repository": REPOSITORY,
            "revision": checkpoint_info(REPOSITORY, name)["reference_revision"],
        }
    result = {
        "schema_version": 1,
        "model": model,
        "dataset": {
            "path": str(path),
            "timestamp_column": "timestamp",
            "target_column": "demand",
            "frequency": cadence,
            "units": "synthetic_units",
        },
        "evaluation": {
            "context_length": context,
            "horizon": 4,
            "stride": 4,
            "point_statistic": "native_point",
            "evaluation_start": stamps[context].isoformat(),
            "evaluation_end": (stamps[-1] + pd.Timedelta(cadence)).isoformat(),
        },
        "output": {"directory": str(tmp_path / "results")},
    }
    if future:
        result.pop("evaluation")
        result.update(
            schema_version=2,
            mode="future_forecast",
            forecast={
                "context_length": context,
                "horizon": horizon,
                "point_statistic": "native_point",
            },
        )
    return result


@pytest.mark.parametrize("name", PROFILES)
def test_pinned_configs_and_catalog_preserve_truthful_profile_bounds(name, tmp_path):
    context, horizon, _, token, config_hash = PROFILES[name]
    assert (
        hashlib.sha256((FIXTURES / f"{name}.json").read_bytes()).hexdigest()
        == config_hash
    )
    checkpoint = write_checkpoint(tmp_path / "weights", configuration(name))
    record = checkpoint.record()
    assert validate_checkpoint_record(record) == record
    assert record["point_forecast"]["profile"] == name
    assert record["point_forecast"].get("frequency_token") == token
    assert record["native_prediction_length"] == horizon
    info = catalog_info(REPOSITORY, name)
    assert (
        info["capabilities"]["min_context"]
        == info["capabilities"]["max_context"]
        == context
    )
    assert info["capabilities"]["max_horizon"] == horizon
    assert info["frequency_policy"] == record["point_forecast"]["frequency_policy"]
    assert info["project_inference_status"] == "recorded"
    settings = model_settings(
        "ttm_r2", repository=REPOSITORY, revision=info["reference_revision"]
    )
    assert settings.default_context == context
    assert settings.generation_label == ("R2.1" if token else "R2")
    assert settings.frequency_policy == info["frequency_policy"]


@pytest.mark.parametrize("name", PROFILES)
def test_native_tensor_prefix_inverse_and_required_frequency_tokens(
    name, tmp_path, fake_runtime, monkeypatch
):
    context, horizon, cadence, token, _ = PROFILES[name]
    state = runtime(fake_runtime, monkeypatch, horizon)
    checkpoint = write_checkpoint(tmp_path / "weights", configuration(name))
    model = ttm.TTM(checkpoint)
    times = pd.date_range(
        "2020-01-01", periods=context + horizon, freq=cadence, tz="UTC"
    )
    values = 1e12 + np.arange(context) / 8
    request = ForecastRequest(
        "request",
        "series",
        times[context - 1],
        times[:context],
        values,
        times[context:],
    )
    model.load()
    result = model.predict((request,))[0]
    np.testing.assert_array_equal(
        result.point,
        np.arange(horizon, dtype=np.float64) * values.std() + values.mean(),
    )
    assert result.point_statistic == "native_point"
    assert state.load_calls[0][1]["trust_remote_code"] is False
    assert state.load_calls[0][1]["use_safetensors"] is True
    assert state.load_calls[0][1]["local_files_only"] is True
    assert state.tensor_calls[0].shape == (1, context, 1)
    if token:
        assert state.tokens[0].tolist() == [token]
        assert state.tokens[0].dtype == np.int64
        assert state.prediction_calls[0]["freq_token"].values.tolist() == [token]
        assert (
            model.runtime_configuration()["forecast_options"]["frequency_token"]
            == token
        )
    else:
        assert state.tokens == [] and "freq_token" not in state.prediction_calls[0]
    model.unload()


@pytest.mark.parametrize("name", PROFILES)
def test_new_historical_and_future_artifacts_reload_export_without_weights(
    name, tmp_path, fake_runtime, monkeypatch
):
    _, horizon, _, _, _ = PROFILES[name]
    runtime(fake_runtime, monkeypatch, horizon)
    write_checkpoint(tmp_path / "weights", configuration(name))
    historical = config_from_mapping(mapping(tmp_path, name), base_directory=tmp_path)
    completed = run_config(historical, include_outputs=True)
    historical_result, frame = ExperimentStore(tmp_path / "results").load_result(
        completed["run_id"]
    )
    assert len(frame) == 8
    future = future_config_from_mapping(
        mapping(tmp_path, name, future=True), base_directory=tmp_path
    )
    completed_future = run_future_config(future)
    # Reload must use saved records without constructing a model or inspecting its files.
    monkeypatch.setattr(
        "tsfm_lab.models.factory.build_model",
        lambda *a, **kw: pytest.fail("model reload"),
    )
    monkeypatch.setattr(
        ttm, "verify_checkpoint", lambda *a: pytest.fail("checkpoint reload")
    )
    result, predictions, _ = FutureStore(tmp_path / "results").load_future_result(
        completed_future["run_id"]
    )
    assert len(predictions) == horizon
    assert result["evaluation"]["status"] == "unavailable"
    for run in (completed, completed_future):
        target = tmp_path / f"{run['run_id']}.csv"
        assert export_predictions(run["run_id"], tmp_path / "results", target)[
            "rows"
        ] == (8 if run is completed else horizon)
    changed = copy.deepcopy(historical_result)
    changed["configuration"]["model"]["options"]["checkpoint"]["point_forecast"][
        "frequency_policy"
    ] = "any_fixed"
    with pytest.raises(StoreError):
        validate_predictions(frame, changed)


@pytest.mark.parametrize("name", PROFILES)
def test_wrong_cadence_refused_at_config_review_and_direct_inference(
    name, tmp_path, fake_runtime, monkeypatch
):
    context, horizon, cadence, _, _ = PROFILES[name]
    wrong = "1h" if cadence != "1h" else "1D"
    raw = mapping(tmp_path, name, local=False)
    raw["dataset"]["frequency"] = wrong
    with pytest.raises(ConfigError):
        config_from_mapping(raw, base_directory=tmp_path)
    valid = mapping(tmp_path, name, local=False)
    config = config_from_mapping(valid, base_directory=tmp_path)
    config.dataset["frequency"] = wrong
    with pytest.raises(ValueError):
        _native_bounds(config, config_metadata(configuration(name)))
    runtime(fake_runtime, monkeypatch, horizon)
    model = ttm.TTM(write_checkpoint(tmp_path / "weights", configuration(name)))
    model.load()
    stamps = pd.date_range("2020-01-01", periods=context + 1, freq=wrong, tz="UTC")
    request = ForecastRequest(
        "bad",
        "series",
        stamps[context - 1],
        stamps[:context],
        np.arange(context),
        stamps[context:],
    )
    with pytest.raises(ValueError):
        model.predict((request,))
    assert fake_runtime.prediction_calls == []


@pytest.mark.parametrize("name", ["52-16-ft-r2.1", "90-30-ft-r2.1"])
@pytest.mark.parametrize("mutation", ["token", "prefix", "loss", "unknown"])
def test_r21_unreviewed_config_or_record_is_rejected(name, mutation, tmp_path):
    config = configuration(name)
    if mutation == "token":
        record = write_checkpoint(tmp_path / "weights", config).record()
        record["point_forecast"]["frequency_token"] = 0
        with pytest.raises(CheckpointError):
            validate_checkpoint_record(record)
        return
    config[
        {"prefix": "resolution_prefix_tuning", "loss": "loss", "unknown": "unreviewed"}[
            mutation
        ]
    ] = {"prefix": False, "loss": "mae", "unknown": True}[mutation]
    with pytest.raises(CheckpointError):
        write_checkpoint(tmp_path / "weights", config)


@pytest.mark.parametrize("name", PROFILES)
def test_wizard_and_assisted_seed_select_profile_and_future_mode(
    name, tmp_path, monkeypatch
):
    raw = mapping(tmp_path, name, local=False)
    checkpoint = write_checkpoint(tmp_path / "weights", configuration(name))
    monkeypatch.setattr(
        "tsfm_lab.repl.wizard.build_model", lambda *a, **kw: ttm.TTM(checkpoint)
    )
    monkeypatch.setattr(
        "tsfm_lab.tui.forms.build_model", lambda *a, **kw: ttm.TTM(checkpoint)
    )
    context, _, cadence, _, _ = PROFILES[name]
    answers = {
        "Context/horizon profile": checkpoint_info(REPOSITORY, name)["display_name"],
        "Observation frequency": cadence,
        "Forecast mode": "Future forecasting",
        "Run experiment?": "yes",
    }
    questions = Questions(answers)
    emitted = []
    first = build_experiment(
        questions,
        emitted.append,
        workspace=tmp_path / "results",
        demo_path=tmp_path / "unused-demo.csv",
        adapter="ttm_r2",
        seed_fields={
            "dataset_path": raw["dataset"]["path"],
            "frequency": cadence,
            "units": "synthetic_units",
        },
    )
    assert first is not None, "\n".join(emitted)
    assert first.model["revision"] == raw["model"]["revision"]
    assert first.forecast.context_length == context
    assert first.dataset["frequency"] == cadence
    assert first.forecast.point_statistic == "native_point"
    validate_settings(first.model, first.forecast, cadence)
    assert "no automatic resampling" in "\n".join(emitted)


def test_old_ttm_point_identity_and_runtime_options_are_unchanged():
    expected = {
        "statistic": "native_point",
        "training_loss": "mse",
        "release": "r2",
        "profile": "512-96-r2",
        "frequency_policy": "minute_hour_fixed_v1",
        "normalization": "external_context_zscore_population_v1_then_native_std",
        "inverse_transform": "native_then_external_v1",
        "horizon_policy": "native_prefix_v1",
    }
    assert point_metadata(512) == expected
