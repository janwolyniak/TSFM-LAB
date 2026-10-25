"""The guided form builds the same validated task without terminal dependencies."""

import builtins
import json
import threading

import pandas as pd
import pytest
import yaml

from tsfm_lab.config import ConfigError
from tsfm_lab.tui.forms import (
    config_from_fields,
    default_fields,
    demo_fields,
    inspect_data,
    prepare_experiment,
    suggested_bounds,
    validate_model_fields,
)


@pytest.fixture
def demo(tmp_path):
    path = tmp_path / "seasonal.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=32, freq="h", tz="UTC"),
            "demand": [10 + i // 4 + i % 4 for i in range(32)],
        }
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def arguments(tmp_path):
    return {"workspace": tmp_path / "runs", "base_directory": tmp_path}


def ready_fields(demo, arguments):
    fields = demo_fields(demo)
    data = inspect_data(fields, **arguments)
    fields.update(suggested_bounds(data, fields))
    return fields


def test_demo_proposes_four_complete_windows_without_writing_results(demo, arguments):
    fields = ready_fields(demo, arguments)
    prepared = prepare_experiment(fields, **arguments)
    assert prepared.windows == 4
    assert prepared.forecast_cases == 16
    assert prepared.observations == 32
    assert prepared.config.evaluation.evaluation_start == pd.Timestamp(
        "2026-01-01T16:00Z"
    )
    assert prepared.config.evaluation.evaluation_end == pd.Timestamp(
        "2026-01-02T08:00Z"
    )
    assert not arguments["workspace"].exists()
    summary = prepared.summary()
    for expected in (
        "demand",
        "synthetic_units",
        "1h",
        "Context: 8",
        "Horizon: 4",
        "Stride: 4",
        "MASE seasonal lag: 4",
        "Windows: 4",
        "Forecast cases: 16",
        str(arguments["workspace"]),
    ):
        assert expected in summary


def test_seasonal_baseline_and_metric_lags_remain_separate(demo, arguments):
    fields = ready_fields(demo, arguments)
    fields.update(adapter="seasonal_naive", seasonal_period="4", mase_period="2")
    config = prepare_experiment(fields, **arguments).config
    assert config.model["options"]["seasonal_period"] == 4
    assert config.evaluation.seasonal_period == 2
    assert (
        "Baseline seasonal period: 4"
        in prepare_experiment(fields, **arguments).summary()
    )


@pytest.mark.parametrize("source", ["huggingface", "local"])
@pytest.mark.parametrize("adapter", ["chronos_bolt", "timesfm_2p5"])
def test_chronos_form_parsing_is_offline_without_checkpoint_or_runtime_access(
    demo, arguments, monkeypatch, source, adapter
):
    fields = ready_fields(demo, arguments)
    fields.update(
        adapter=adapter,
        repository=default_fields(adapter)["repository"],
        source=source,
        revision="a" * 40,
        checkpoint_path="models/bolt",
    )
    original_import = builtins.__import__

    def no_runtime(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "chronos", "timesfm", "huggingface_hub"}:
            raise AssertionError("Form validation must not initialize model runtimes")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_runtime)
    config = config_from_fields(fields, **arguments)
    assert config.runtime["dtype"] == "float32"
    if source == "huggingface":
        assert config.model["offline"] is True
    else:
        assert config.model["path"] == str(arguments["base_directory"] / "models/bolt")


@pytest.fixture
def checkpoint_directory(arguments):
    directory = arguments["base_directory"] / "checkpoint"
    directory.mkdir()
    config = {
        "architectures": ["ChronosBoltModelForForecasting"],
        "model_type": "t5",
        "chronos_pipeline_class": "ChronosBoltPipeline",
        "chronos_config": {
            "context_length": 64,
            "prediction_length": 8,
            "quantiles": [0.1, 0.5, 0.9],
        },
    }
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "model.safetensors").write_bytes(b"metadata fixture; never loaded")
    return directory


def test_review_validates_local_checkpoint_metadata_without_model_runtime(
    demo, arguments, checkpoint_directory, monkeypatch
):
    fields = ready_fields(demo, arguments)
    fields.update(
        adapter="chronos_bolt",
        source="local",
        checkpoint_path=str(checkpoint_directory),
    )
    original_import = builtins.__import__

    def no_runtime(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "chronos", "huggingface_hub"}:
            raise AssertionError("Review must not initialize model runtimes")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_runtime)
    prepared = prepare_experiment(fields, **arguments)
    assert prepared.config.model["path"] == str(checkpoint_directory)
    assert prepared.forecast_cases == 16
    assert not arguments["workspace"].exists()


def test_missing_local_checkpoint_is_rejected_before_review(demo, arguments):
    fields = ready_fields(demo, arguments)
    fields.update(
        adapter="chronos_bolt", source="local", checkpoint_path="missing-model"
    )
    with pytest.raises(ConfigError, match="Checkpoint directory does not exist"):
        prepare_experiment(fields, **arguments)
    assert not arguments["workspace"].exists()


@pytest.mark.parametrize("key,value", [("context_length", 4), ("prediction_length", 2)])
def test_checkpoint_native_limits_are_checked_before_review(
    demo, arguments, checkpoint_directory, key, value
):
    path = checkpoint_directory / "config.json"
    config = json.loads(path.read_text())
    config["chronos_config"][key] = value
    path.write_text(json.dumps(config))
    fields = ready_fields(demo, arguments)
    fields.update(
        adapter="chronos_bolt",
        source="local",
        checkpoint_path=str(checkpoint_directory),
    )
    with pytest.raises(ConfigError, match="native"):
        prepare_experiment(fields, **arguments)
    assert not arguments["workspace"].exists()


@pytest.mark.parametrize("adapter", ["chronos_bolt", "timesfm_2p5"])
def test_hub_review_selects_only_existing_offline_checkpoint_metadata(
    demo, arguments, monkeypatch, adapter
):
    fields = ready_fields(demo, arguments)
    fields.update(
        adapter=adapter,
        repository=default_fields(adapter)["repository"],
        revision="a" * 40,
    )
    calls = []

    class MetadataOnlyModel:
        def validate(self, spec):
            calls.append(spec)

        def load(self):
            raise AssertionError("Review must not initialize model weights")

    def metadata(selection, runtime):
        assert selection["adapter_id"] == adapter
        assert selection["source"] == "huggingface"
        assert selection["offline"] is True
        assert selection["revision"] == "a" * 40
        assert runtime["dtype"] == "float32"
        return MetadataOnlyModel()

    monkeypatch.setattr("tsfm_lab.tui.forms.build_model", metadata)
    prepared = prepare_experiment(fields, **arguments)
    assert calls == [prepared.config.evaluation]


@pytest.mark.parametrize("value", ["", "0", "-1", "1.5", "true", "1e3", True])
@pytest.mark.parametrize(
    "field", ["context_length", "horizon", "stride", "mase_period"]
)
def test_invalid_integer_entries_are_not_coerced(demo, arguments, field, value):
    fields = ready_fields(demo, arguments)
    fields[field] = value
    with pytest.raises(ConfigError):
        config_from_fields(fields, **arguments)


@pytest.mark.parametrize(
    "adapter", ["last_value", "seasonal_naive", "chronos_bolt", "timesfm_2p5"]
)
def test_model_step_does_not_depend_on_unfilled_data_or_evaluation(arguments, adapter):
    fields = default_fields(adapter)
    fields["revision"] = "a" * 40
    validate_model_fields(fields, **arguments)
    assert not arguments["workspace"].exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"name": ""},
        {"adapter": "unknown"},
        {"adapter": "seasonal_naive", "seasonal_period": "false"},
        {"adapter": "chronos_bolt", "revision": "main"},
        {"adapter": "chronos_bolt", "source": "local", "checkpoint_path": ""},
    ],
)
def test_model_step_rejects_invalid_selection(arguments, changes):
    with pytest.raises(ConfigError):
        validate_model_fields(default_fields() | changes, **arguments)


@pytest.mark.parametrize(
    "changes",
    [
        {"dataset_path": "missing.csv"},
        {"target_column": "unknown"},
        {"frequency": "2h"},
    ],
)
def test_data_inspection_reports_shared_loader_errors(demo, arguments, changes):
    with pytest.raises(ConfigError):
        inspect_data(demo_fields(demo) | changes, **arguments)
    assert not arguments["workspace"].exists()


def test_nonfinite_dataset_fails_before_an_experiment_record(demo, arguments):
    frame = pd.read_csv(demo)
    frame["demand"] = frame["demand"].astype(float)
    frame.loc[3, "demand"] = float("inf")
    frame.to_csv(demo, index=False)
    with pytest.raises(ConfigError, match="finite numeric"):
        inspect_data(demo_fields(demo), **arguments)
    assert not arguments["workspace"].exists()


def test_invalid_user_bounds_are_never_silently_repaired(demo, arguments):
    fields = ready_fields(demo, arguments)
    fields["evaluation_start"] = "2026-01-01T16:30:00Z"
    original = dict(fields)
    with pytest.raises(ConfigError, match="dataset grid"):
        prepare_experiment(fields, **arguments)
    assert fields == original


def test_too_long_context_cannot_be_repaired_by_changing_the_window(demo, arguments):
    fields = ready_fields(demo, arguments)
    fields["context_length"] = "33"
    with pytest.raises(ConfigError, match="Adjust context and horizon"):
        suggested_bounds(inspect_data(fields, **arguments), fields)


def test_relative_form_paths_and_yaml_snapshot_use_a_logical_source(demo, arguments):
    fields = ready_fields(demo, arguments)
    fields["dataset_path"] = demo.name
    config = config_from_fields(fields, **arguments)
    assert config.dataset["path"] == str(demo)
    assert config.source_path == demo.parent / "interactive.yaml"
    assert not config.source_path.exists()
    assert yaml.safe_load(config.source_yaml)["dataset"]["path"] == demo.name


def test_forms_can_run_validation_in_a_background_thread(demo, arguments):
    fields = ready_fields(demo, arguments)
    results = []
    worker = threading.Thread(
        target=lambda: results.append(prepare_experiment(fields, **arguments))
    )
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert results[0].forecast_cases == 16


def test_forms_reject_unknown_or_nontext_fields(arguments):
    with pytest.raises(ConfigError, match="unknown fields"):
        config_from_fields({"extra": "value"}, **arguments)
    with pytest.raises(ConfigError, match="contain text"):
        config_from_fields({"horizon": False}, **arguments)
