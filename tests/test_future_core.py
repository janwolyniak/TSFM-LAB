"""Future requests are causal, bounded, and independent of historical evaluation."""

import importlib
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsfm_lab.contracts import Dataset, ForecastOutput
from tsfm_lab.future import (
    context_identity,
    model_identity,
    preflight_context,
    prepare_future_request,
    request_identity,
    resolve_future,
    run_future,
)
from tsfm_lab.future_config import FutureForecastSpec, future_config_from_mapping
from tsfm_lab.models.baselines import LastValue, SeasonalNaive


def data(values=None, *, count=8, start="2026-01-01", frequency="1h"):
    values = (
        np.arange(count, dtype=float)
        if values is None
        else np.asarray(values, dtype=float)
    )
    return Dataset(
        pd.date_range(start, periods=len(values), freq=frequency, tz="UTC"),
        values,
        frequency,
        "target",
        "units",
        "series",
    )


def test_no_holdout_is_needed_and_one_observation_can_forecast():
    dataset = data([7])
    result = run_future(dataset, LastValue(), FutureForecastSpec(1, 3))
    assert result.output.point.tolist() == [7, 7, 7]
    assert result.output.point_statistic == "deterministic"
    assert result.request.origin == dataset.timestamps[-1]
    assert result.request.target_timestamps.equals(
        pd.date_range("2026-01-01T01:00Z", periods=3, freq="h")
    )
    assert re.fullmatch("[0-9a-f]{64}", result.request.request_id)
    assert not hasattr(result, "scores") and not hasattr(result.request, "actual")


def test_exact_final_context_is_detached_and_seasonal_forecast_uses_last_season():
    dataset = data(count=12)
    result = run_future(dataset, SeasonalNaive(4), FutureForecastSpec(8, 6))
    assert result.request.context_values.tolist() == list(range(4, 12))
    assert result.output.point.tolist() == [8, 9, 10, 11, 8, 9]
    assert not np.shares_memory(result.request.context_values, dataset.values)
    assert not result.request.context_values.flags.writeable
    assert not result.output.point.flags.writeable
    assert not np.shares_memory(
        result.request.context_timestamps.asi8, dataset.timestamps.asi8
    )


@pytest.mark.parametrize("origin", ["2026-01-01T06:00Z", "2026-01-01T08:00Z"])
def test_earlier_or_later_origin_refused_without_truncation(origin):
    with pytest.raises(ValueError, match="final observed timestamp"):
        prepare_future_request(data(), FutureForecastSpec(8, 4, origin=origin))


def test_origin_equivalence_and_resolved_configuration_keep_original_policy(tmp_path):
    dataset = data()
    origin = "2026-01-01T08:00:00+01:00"
    request = prepare_future_request(dataset, FutureForecastSpec(8, 4, origin=origin))
    assert request.origin == dataset.timestamps[-1]
    mapping = {
        "schema_version": 2,
        "mode": "future_forecast",
        "model": {"source": "builtin", "adapter_id": "last_value"},
        "dataset": {
            "path": "data.csv",
            "timestamp_column": "timestamp",
            "target_column": "target",
            "frequency": "1h",
        },
        "forecast": {"context_length": 8, "horizon": 4},
    }
    config = future_config_from_mapping(mapping, base_directory=tmp_path)
    resolved = resolve_future(config, dataset)
    assert resolved.forecast.origin == request.origin
    assert resolved.source_yaml == config.source_yaml
    assert resolved.entered_origin == "latest"
    assert config.forecast.origin == "latest"


def test_insufficient_history_and_overflow_fail_before_grid_allocation(monkeypatch):
    with pytest.raises(ValueError, match="Insufficient"):
        prepare_future_request(data([1]), FutureForecastSpec(2, 1))
    end = pd.Timestamp.max.tz_localize("UTC")
    dataset = Dataset(pd.DatetimeIndex([end]), [1.0], "1ns")
    monkeypatch.setattr(
        pd,
        "to_datetime",
        lambda *args, **kwargs: pytest.fail("Allocated overflowing grid"),
    )
    with pytest.raises(ValueError, match="nanosecond bounds"):
        prepare_future_request(dataset, FutureForecastSpec(1, 1))


def test_python_integer_grid_avoids_intermediate_int64_multiplication_overflow():
    dataset = data([1], start="1700-01-01", frequency="100000D")
    request = prepare_future_request(dataset, FutureForecastSpec(1, 2))
    delta = int(pd.Timedelta("100000D").value)
    assert request.target_timestamps.asi8.tolist() == [
        int(dataset.timestamps[-1].value) + delta * lead for lead in (1, 2)
    ]


@pytest.mark.parametrize(
    "frequency,start",
    [("1h", "2026-03-29T00:00:00Z"), ("1ns", "2026-01-01T23:59:59.999999998Z")],
)
def test_grid_retains_declared_fixed_delta_through_dst_midnight_and_subseconds(
    frequency, start
):
    dataset = data([1, 2], start=start, frequency=frequency)
    request = prepare_future_request(dataset, FutureForecastSpec(2, 4))
    delta = int(pd.Timedelta(frequency).value)
    assert request.target_timestamps.asi8.tolist() == [
        int(request.origin.value) + lead * delta for lead in range(1, 5)
    ]


@pytest.mark.parametrize(
    "adapter", ["chronos_bolt", "chronos_2", "timesfm_2p5", "tirex_2"]
)
def test_consumed_float32_overflow_fails_in_dataset_only_preflight(adapter):
    with pytest.raises(ValueError, match="representable as float32"):
        preflight_context(data([1e100]), FutureForecastSpec(1, 1), adapter)
    request = preflight_context(data([1e100, 1, 2]), FutureForecastSpec(2, 1), adapter)
    assert request.context_values.tolist() == [1, 2]


def test_ttm_preflight_accepts_large_normalizable_raw_values_and_rejects_unsafe_scaling():
    request = preflight_context(
        data(np.full(512, 2.0**200)),
        FutureForecastSpec(512, 1, "native_point"),
        "ttm_r2",
    )
    assert request.context_values[-1] == 2.0**200
    with pytest.raises(ValueError, match="standardized safely"):
        preflight_context(
            data(np.full(512, 1e308)),
            FutureForecastSpec(512, 1, "native_point"),
            "ttm_r2",
        )


def test_unused_prefix_does_not_change_request_identity_or_prediction():
    original = data(count=16)
    changed = data([1e100] * 8 + list(range(8, 16)))
    a = run_future(original, LastValue(), FutureForecastSpec(8, 4))
    b = run_future(changed, LastValue(), FutureForecastSpec(8, 4))
    assert a.request.request_id == b.request.request_id
    assert a.context_sha256 == b.context_sha256
    np.testing.assert_array_equal(a.output.point, b.output.point)


def test_identity_recomputes_from_detached_context_and_effective_numeric_settings():
    dataset = data()
    result = run_future(dataset, LastValue(), FutureForecastSpec(8, 4, "native_point"))
    measurement = {
        "target_name": dataset.target_name,
        "units": dataset.units,
        "frequency": dataset.frequency,
    }
    assert context_identity(result.request, **measurement) == result.context_sha256
    assert (
        request_identity(
            result.request,
            **measurement,
            point_statistic="native_point",
            model_identity=result.model_identity,
            runtime=result.runtime,
        )
        == result.request.request_id
    )
    assert (
        request_identity(
            result.request,
            **measurement,
            point_statistic="median",
            model_identity=result.model_identity,
            runtime=result.runtime,
        )
        != result.request.request_id
    )
    assert result.model_identity == model_identity(
        result.model["adapter_id"],
        result.model["adapter_version"],
        result.model["options"],
    )


class Controlled(LastValue):
    def __init__(self, alter=None):
        self.loaded, self.unloaded, self.predicted = False, False, False
        self.alter = alter

    def validate(self, spec):
        assert isinstance(spec, FutureForecastSpec)
        assert not hasattr(spec, "metrics")
        super().validate(spec)

    def load(self):
        self.loaded = True

    def predict(self, requests):
        self.predicted = True
        request = requests[0]
        output = super().predict(requests)[0]
        if self.alter == "empty":
            return ()
        if self.alter == "timestamps":
            return (
                replace(
                    output,
                    target_timestamps=output.target_timestamps + pd.Timedelta("1h"),
                ),
            )
        if self.alter == "origin":
            return (replace(output, origin=output.origin - pd.Timedelta("1h")),)
        if self.alter == "label":
            return (replace(output, point_statistic="mean"),)
        if self.alter == "identity":
            return (replace(output, request_id="wrong"),)
        if self.alter == "nan":
            return (replace(output, point=np.full(len(output.point), np.nan)),)
        if self.alter == "mutate_grid":
            request.target_timestamps.values[:] += np.timedelta64(1, "h")
            return (
                ForecastOutput(
                    **{**vars(output), "target_timestamps": request.target_timestamps}
                ),
            )
        if self.alter == "mutate_values":
            request.context_values.setflags(write=True)
            request.context_values[0] += 1
        return (output,)

    def unload(self):
        self.unloaded = True


@pytest.mark.parametrize(
    "alter",
    [
        "empty",
        "timestamps",
        "origin",
        "label",
        "identity",
        "nan",
        "mutate_grid",
        "mutate_values",
    ],
)
def test_bad_outputs_or_mutating_adapter_fail_and_always_unload(alter):
    model = Controlled(alter)
    with pytest.raises(ValueError):
        run_future(data(), model, FutureForecastSpec(8, 4))
    assert model.unloaded


@pytest.mark.parametrize("stop_stage", ["loading", "running"])
def test_stage_cancellation_blocks_the_next_expensive_call(stop_stage):
    cancelled = False
    model = Controlled()

    def progress(stage):
        nonlocal cancelled
        cancelled = stage == stop_stage

    with pytest.raises(KeyboardInterrupt):
        run_future(
            data(),
            model,
            FutureForecastSpec(8, 4),
            progress=progress,
            should_cancel=lambda: cancelled,
        )
    assert model.loaded is (stop_stage == "running")
    assert not model.predicted and model.unloaded


@pytest.mark.parametrize(
    "runtime",
    [
        {"dtype": "float32"},
        {"batch_size": True},
        {"seed": -1},
        {"forecast_options": {}},
        {"device": "cuda"},
    ],
)
def test_invalid_runtime_is_rejected_before_loading(runtime):
    model = Controlled()
    with pytest.raises(ValueError):
        run_future(data(), model, FutureForecastSpec(8, 4), runtime=runtime)
    assert not model.loaded


def test_historical_windows_and_scores_are_never_called(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Future inference entered historical evaluation")

    monkeypatch.setattr("tsfm_lab.evaluation.windows.evaluation_bounds", forbidden)
    monkeypatch.setattr("tsfm_lab.evaluation.windows.iter_windows", forbidden)
    monkeypatch.setattr("tsfm_lab.evaluation.metrics.score_forecasts", forbidden)
    assert (
        len(run_future(data(), LastValue(), FutureForecastSpec(8, 2)).output.point) == 2
    )


@pytest.mark.parametrize(
    "family,adapter_name,class_name",
    [
        ("chronos_bolt", "chronos_bolt", "ChronosBolt"),
        ("chronos2", "chronos2", "Chronos2"),
        ("tirex2", "tirex2", "TiRex2"),
        ("ttm", "ttm", "TTM"),
    ],
)
def test_actual_pretrained_adapters_accept_future_spec_and_preserve_point_labels(
    tmp_path, monkeypatch, family, adapter_name, class_name
):
    fixtures = importlib.import_module(f"test_{family}")
    checkpoint = fixtures.checkpoint.__wrapped__(tmp_path)
    fixtures.fake_runtime.__wrapped__(monkeypatch)
    module = importlib.import_module(f"tsfm_lab.models.{adapter_name}")
    model = getattr(module, class_name)(checkpoint)
    length = model.metadata().capabilities.min_context
    spec = FutureForecastSpec(max(8, length), 3, "native_point")
    result = run_future(data(count=spec.context_length), model, spec)
    assert len(result.output.point) == 3
    assert result.output.point_statistic == (
        "native_point" if family == "ttm" else "median"
    )
    assert (
        result.model["options"]["checkpoint"]["fingerprint"] == checkpoint.fingerprint
    )
    assert "path" not in str(result.model_identity)
    caps = model.metadata().capabilities
    boundary = FutureForecastSpec(caps.max_context, caps.max_horizon, "native_point")
    edge = run_future(data(count=caps.max_context), model, boundary)
    assert len(edge.output.point) == caps.max_horizon
    for context, horizon in (
        (caps.max_context + 1, 1),
        (caps.min_context, caps.max_horizon + 1),
    ):
        with pytest.raises(ValueError, match="native adapter limits"):
            run_future(
                data(count=context),
                model,
                FutureForecastSpec(context, horizon, "native_point"),
            )


def test_timesfm_future_uses_effective_context_and_runtime_options(
    timesfm_checkpoint, timesfm_runtime
):
    from tsfm_lab.models.checkpoints import inspect_local_checkpoint
    from tsfm_lab.models.timesfm import TimesFM

    result = run_future(
        data(),
        TimesFM(inspect_local_checkpoint(timesfm_checkpoint)),
        FutureForecastSpec(8, 3, "native_point"),
    )
    assert result.output.point_statistic == "median"
    assert result.runtime["forecast_options"]["normalize_inputs"] is True
    model = TimesFM(inspect_local_checkpoint(timesfm_checkpoint))
    caps = model.metadata().capabilities
    assert (
        len(
            run_future(
                data(count=caps.max_context),
                model,
                FutureForecastSpec(caps.max_context, caps.max_horizon),
            ).output.point
        )
        == 128
    )


@pytest.mark.parametrize("context", [512, 1024, 1536])
def test_every_ttm_native_context_accepts_true_future_spec(
    tmp_path, monkeypatch, context
):
    fixtures = importlib.import_module("test_ttm")
    checkpoint = fixtures.write_checkpoint(tmp_path, fixtures.configuration(context))
    fixtures.fake_runtime.__wrapped__(monkeypatch)
    from tsfm_lab.models.ttm import TTM

    result = run_future(
        data(count=context),
        TTM(checkpoint),
        FutureForecastSpec(context, 96, "native_point"),
    )
    assert len(result.output.point) == 96
    assert result.output.point_statistic == "native_point"


def test_checkpoint_identity_ignores_its_location_and_entered_revision_alias(tmp_path):
    fixtures = importlib.import_module("test_chronos_bolt")
    checkpoint = fixtures.checkpoint.__wrapped__(tmp_path)
    record = checkpoint.record()
    identity = model_identity("chronos_bolt", "fixture", {"checkpoint": record})
    moved = {**record, "path": str(tmp_path / "relocated")}
    assert model_identity("chronos_bolt", "fixture", {"checkpoint": moved}) == identity
    changed = {
        **record,
        "files": {
            **record["files"],
            "model.safetensors": {
                "sha256": "0" * 64,
                "size": record["files"]["model.safetensors"]["size"],
            },
        },
    }
    from tsfm_lab.models.checkpoints import _fingerprint

    changed["fingerprint"] = _fingerprint(changed)
    assert (
        model_identity("chronos_bolt", "fixture", {"checkpoint": changed}) != identity
    )


def test_baseline_operational_context_and_horizon_boundaries():
    result = run_future(
        data(count=100000), LastValue(), FutureForecastSpec(100000, 10000)
    )
    assert len(result.output.point) == 10000
    assert result.request.context_values.size == 100000


def test_future_import_and_baseline_inference_need_no_optional_models_or_connections():
    code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'torch','transformers','chronos','timesfm','tirex2','tsfm_public','huggingface_hub','httpx','keyring'}:
            raise AssertionError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, Block())
import pandas as pd
from tsfm_lab.contracts import Dataset
from tsfm_lab.future import run_future
from tsfm_lab.future_config import FutureForecastSpec
from tsfm_lab.models.baselines import LastValue
data=Dataset(pd.DatetimeIndex(['2026-01-01T00:00:00Z']),[1.0],'1h')
assert run_future(data,LastValue(),FutureForecastSpec(1,1)).output.point[0] == 1
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, PYTHONPATH=str(root / "src")),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
