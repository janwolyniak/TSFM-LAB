"""Strict future schema and untouched historical configuration semantics."""

import copy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tsfm_lab.config import (
    ConfigError,
    config_from_mapping,
    load_config,
    load_run_config,
)
from tsfm_lab.future_config import (
    FutureForecastConfig,
    FutureForecastSpec,
    future_config_from_mapping,
    load_future_config,
)


@pytest.fixture
def mapping():
    return {
        "schema_version": 2,
        "mode": "future_forecast",
        "name": "future",
        "model": {"source": "builtin", "adapter_id": "last_value"},
        "dataset": {
            "path": "observed.csv",
            "timestamp_column": "timestamp",
            "target_column": "target",
            "frequency": "1h",
            "units": "units",
        },
        "forecast": {"context_length": 8, "horizon": 4},
        "output": {"directory": "runs"},
    }


def test_mapping_file_and_dispatch_agree_without_reading_data_or_weights(
    mapping, tmp_path
):
    config = future_config_from_mapping(mapping, base_directory=tmp_path)
    path = tmp_path / "future.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    loaded = load_future_config(path)
    assert isinstance(load_run_config(path), FutureForecastConfig)
    assert loaded.resolved_dict() == config.resolved_dict()
    assert config.dataset["path"] == str(tmp_path / "observed.csv")
    assert config.output_directory == tmp_path / "runs"
    assert config.forecast.origin == "latest" and config.entered_origin == "latest"
    assert config.forecast.point_statistic == "median"
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "observed.csv").exists()
    with pytest.raises(ConfigError, match="historical-only"):
        load_config(path)
    with pytest.raises(ConfigError, match="historical-only"):
        config_from_mapping(mapping, base_directory=tmp_path)


@pytest.mark.parametrize(
    "context,horizon",
    [
        (True, 1),
        (1, False),
        (0, 1),
        (1, -1),
        (100001, 1),
        (1, 10001),
        (1.5, 1),
        ("8", 1),
    ],
)
def test_future_lengths_are_bounded_positive_integers(
    mapping, tmp_path, context, horizon
):
    mapping["forecast"].update(context_length=context, horizon=horizon)
    with pytest.raises(ConfigError):
        future_config_from_mapping(mapping, base_directory=tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("evaluation", {}),
        ("metrics", []),
        ("actuals", [1]),
        ("mode", "historical_evaluation"),
        ("schema_version", 1),
    ],
)
def test_mixed_modes_and_unknown_top_level_fields_fail(mapping, tmp_path, field, value):
    mapping[field] = value
    with pytest.raises(ConfigError):
        future_config_from_mapping(mapping, base_directory=tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "stride",
        "metrics",
        "seasonal_period",
        "evaluation_start",
        "target_timestamps",
        "actual",
    ],
)
def test_forecast_rejects_historical_and_supplied_output_fields(
    mapping, tmp_path, field
):
    mapping["forecast"][field] = 1
    with pytest.raises(ConfigError, match="Unknown forecast fields"):
        future_config_from_mapping(mapping, base_directory=tmp_path)


@pytest.mark.parametrize("origin", ["2026-01-01", "NaT", "invalid", "", 123])
def test_invalid_or_naive_explicit_origin_is_refused(mapping, tmp_path, origin):
    mapping["forecast"]["origin"] = origin
    with pytest.raises(ConfigError):
        future_config_from_mapping(mapping, base_directory=tmp_path)


def test_explicit_origin_is_normalized_losslessly_and_original_input_preserved(
    mapping, tmp_path
):
    entered = "2026-01-01T01:00:00.123456789+01:00"
    mapping["forecast"]["origin"] = entered
    config = future_config_from_mapping(mapping, base_directory=tmp_path)
    assert config.forecast.origin == pd.Timestamp("2026-01-01T00:00:00.123456789Z")
    assert (
        config.resolved_dict()["forecast"]["origin"]
        == "2026-01-01T00:00:00.123456789+00:00"
    )
    assert config.entered_origin == entered
    snapshot = config.resolved_dict()
    snapshot["model"]["adapter_id"] = "changed"
    assert config.model["adapter_id"] == "last_value"


def test_ttm_uses_supported_task_default_and_fixed_profile_without_evaluation(
    mapping, tmp_path
):
    mapping["model"] = {
        "source": "huggingface",
        "adapter_id": "ttm_r2",
        "repository": "ibm-granite/granite-timeseries-ttm-r2",
        "revision": "d6a79570cac0f33d526601cd3a0fc7c80a8f9a2f",
    }
    mapping["forecast"]["context_length"] = 512
    config = future_config_from_mapping(mapping, base_directory=tmp_path)
    assert config.forecast.point_statistic == "native_point"
    assert not hasattr(config.forecast, "evaluation_start")
    mapping["forecast"]["context_length"] = 513
    with pytest.raises(ConfigError, match="requires exactly"):
        future_config_from_mapping(mapping, base_directory=tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        "schema_version: 2\nschema_version: 2\n",
        "a: &a 1\nb: *a\n",
        "[not, a, mapping]",
        "forecast: !!python/object:builtins.object {}",
    ],
)
def test_future_yaml_preserves_safe_loader_boundaries(tmp_path, body):
    path = tmp_path / "invalid.yaml"
    path.write_text(body)
    with pytest.raises(ConfigError):
        load_future_config(path)


def test_size_limit_applies_before_yaml_parse(tmp_path):
    path = tmp_path / "large.yaml"
    path.write_bytes(b"x" * (1024 * 1024 + 1))
    with pytest.raises(ConfigError, match="1 MiB"):
        load_future_config(path)


def test_historical_dispatch_and_resolved_values_are_unchanged():
    path = Path(__file__).parents[1] / "examples/configurations/last-value.yaml"
    historical = load_config(path)
    assert load_run_config(path).resolved_dict() == historical.resolved_dict()
    assert historical.resolved_dict()["schema_version"] == 1
    assert "mode" not in historical.resolved_dict()
    assert historical.evaluation.context_length == 8


def test_future_spec_accepts_only_operational_limits():
    assert FutureForecastSpec(100000, 10000).horizon == 10000
    with pytest.raises(ValueError):
        FutureForecastSpec(True, 1)
    with pytest.raises(ValueError):
        FutureForecastSpec(1, 10001)


def test_future_loader_refuses_historical_input(mapping, tmp_path):
    historical = copy.deepcopy(mapping)
    historical.update(schema_version=1, mode="historical_evaluation")
    with pytest.raises(ConfigError, match="schema_version 2"):
        future_config_from_mapping(historical, base_directory=tmp_path)
