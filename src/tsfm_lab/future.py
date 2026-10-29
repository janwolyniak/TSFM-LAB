"""One detached future point request, with no historical windows or scoring."""

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from time import perf_counter

import numpy as np
import pandas as pd

from tsfm_lab.contracts import (
    Dataset,
    ForecastOutput,
    ForecastRequest,
    fixed_interval,
    positive_int,
)
from tsfm_lab.future_config import FutureForecastConfig, FutureForecastSpec
from tsfm_lab.models.checkpoint_profiles import checkpoint_adapter
from tsfm_lab.models.checkpoints import validate_checkpoint_record
from tsfm_lab.models.settings import model_settings, validate_settings
from tsfm_lab.point_tasks import output_statistics, validate_model_frequency

_RAW_FLOAT32 = {"chronos_bolt", "chronos_2", "timesfm_2p5", "tirex_2"}


def _json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _clone(value):
    return json.loads(_json(value))


def _numeric_history(values, adapter_id):
    if adapter_id in _RAW_FLOAT32 and np.any(np.abs(values) > np.finfo(np.float32).max):
        raise ValueError("Consumed future context must be representable as float32")
    if adapter_id == "ttm_r2":
        # Reuse the adapter's causal numerical guard without importing its runtime.
        from tsfm_lab.models.ttm import _standardize

        _standardize(values)


def prepare_future_request(
    dataset, spec, *, model=None, adapter_id=None, request_id="future-preview"
) -> ForecastRequest:
    """Resolve final-row origin and a bounded grid before any model preparation."""
    if not isinstance(dataset, Dataset) or not isinstance(spec, FutureForecastSpec):
        raise ValueError("Future preparation requires Dataset and FutureForecastSpec")
    if len(dataset.values) < spec.context_length:
        raise ValueError("Insufficient observations for the requested future context")
    if model is not None:
        if adapter_id is not None and adapter_id != model["adapter_id"]:
            raise ValueError("Future adapter selections disagree")
        adapter_id = model["adapter_id"]
        validate_settings(model, spec, dataset.frequency)
    elif adapter_id is not None:
        model_settings(adapter_id)
    try:
        times = dataset.timestamps[-spec.context_length :].as_unit("ns")
        origin = times[-1]
        origin_ns = int(origin.value)
        delta = int(fixed_interval(dataset.frequency).value)
        if spec.origin != "latest" and spec.origin != origin:
            raise ValueError("Future origin must equal the final observed timestamp")
        end = origin_ns + spec.horizon * delta
        if end > int(pd.Timestamp.max.value):
            raise ValueError("Future target timestamps exceed UTC nanosecond bounds")
        # Python integer arithmetic avoids int64 multiplication wrapping for old origins.
        targets = pd.to_datetime(
            [origin_ns + step * delta for step in range(1, spec.horizon + 1)],
            unit="ns",
            utc=True,
        )
        request = ForecastRequest(
            request_id,
            dataset.series_id,
            origin,
            times,
            dataset.values[-spec.context_length :],
            targets,
        )
    except (OverflowError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid future origin or timestamp grid: {exc}") from exc
    _numeric_history(request.context_values, adapter_id)
    return request


def preflight_context(dataset, spec, adapter_id):
    return prepare_future_request(dataset, spec, adapter_id=adapter_id)


def resolve_future(config, dataset) -> FutureForecastConfig:
    """Resolve latest to a concrete UTC origin while preserving original YAML."""
    if not isinstance(config, FutureForecastConfig):
        raise ValueError("Expected a future forecast configuration")
    request = prepare_future_request(dataset, config.forecast, model=config.model)
    return replace(config, forecast=replace(config.forecast, origin=request.origin))


def context_identity(request, *, target_name, units, frequency) -> str:
    """Hash only detached consumed history and its measurement declarations."""
    metadata = {
        "serialization": "future_consumed_context_v1",
        "series_id": request.series_id,
        "target_name": target_name,
        "units": units,
        "frequency_ns": int(fixed_interval(frequency).value),
        "observations": len(request.context_values),
    }
    digest = hashlib.sha256(_json(metadata) + b"\n")
    digest.update(
        request.context_timestamps.as_unit("ns")
        .asi8.astype("<i8", copy=False)
        .tobytes()
    )
    digest.update(np.asarray(request.context_values, dtype="<f8").tobytes())
    return digest.hexdigest()


def model_identity(adapter_id, adapter_version, options) -> dict:
    """Keep verified numerical model identity independent of locations and aliases."""
    model_settings(adapter_id)
    if (
        not isinstance(adapter_version, str)
        or not adapter_version.strip()
        or not isinstance(options, dict)
    ):
        raise ValueError("Invalid future model identity")
    if adapter_id in {"last_value", "seasonal_naive"}:
        allowed = {"seasonal_period"} if adapter_id == "seasonal_naive" else set()
        if set(options) != allowed:
            raise ValueError("Invalid baseline model options")
        numerical = (
            {}
            if not allowed
            else {
                "seasonal_period": positive_int(
                    options["seasonal_period"], "seasonal_period"
                )
            }
        )
    else:
        if set(options) != {"checkpoint"}:
            raise ValueError(
                "Future pretrained identity requires its verified checkpoint"
            )
        record = validate_checkpoint_record(options["checkpoint"])
        if checkpoint_adapter(record["architecture"]) != adapter_id:
            raise ValueError("Future adapter and checkpoint architecture disagree")
        numerical = {"checkpoint_fingerprint": record["fingerprint"]}
    return {
        "adapter_id": adapter_id,
        "adapter_version": adapter_version,
        "options": numerical,
    }


def request_identity(
    request, *, target_name, units, frequency, point_statistic, model_identity, runtime
) -> str:
    """Compute the final SHA after verified checkpoint bytes are available."""
    output_statistics(point_statistic)
    payload = {
        "serialization": "future_point_request_v1",
        "series_id": request.series_id,
        "target_name": target_name,
        "units": units,
        "frequency_ns": int(fixed_interval(frequency).value),
        "point_statistic": point_statistic,
        "context_sha256": context_identity(
            request, target_name=target_name, units=units, frequency=frequency
        ),
        "origin": request.origin.isoformat(),
        "context_length": len(request.context_values),
        "horizon": len(request.target_timestamps),
        "target_timestamps": [value.isoformat() for value in request.target_timestamps],
        "model": model_identity,
        "runtime": runtime,
    }
    return hashlib.sha256(_json(payload)).hexdigest()


def _runtime(model, configured):
    adapter = model.metadata().adapter_id
    configured = {} if configured is None else configured
    if not isinstance(configured, dict) or set(configured) - {
        "device",
        "dtype",
        "batch_size",
        "seed",
    }:
        raise ValueError("Unsupported configured future runtime")
    actual = getattr(model, "runtime_configuration", lambda: {})()
    if any(key in actual and actual[key] != value for key, value in configured.items()):
        raise ValueError("Configured future runtime disagrees with the adapter")
    dtype = "float64" if adapter in {"last_value", "seasonal_naive"} else "float32"
    effective = {
        "device": "cpu",
        "dtype": dtype,
        "batch_size": 1,
        "seed": 0,
        "seed_used": False,
        "deterministic": True,
        **configured,
        **actual,
    }
    allowed = {
        "device",
        "dtype",
        "batch_size",
        "seed",
        "seed_used",
        "deterministic",
        "torch_compile",
        "forecast_options",
    }
    if (
        set(effective) - allowed
        or effective["device"] != "cpu"
        or effective["dtype"] != dtype
        or type(effective["batch_size"]) is not int
        or effective["batch_size"] != 1
        or type(effective["seed"]) is not int
        or effective["seed"] < 0
        or effective["seed_used"] is not False
        or effective["deterministic"] is not True
        or effective.get("torch_compile", False) is not False
    ):
        raise ValueError("Unsupported future numerical runtime")
    return _clone(effective)


def _environment(model):
    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "os": platform.platform(),
        "machine": platform.machine(),
        **getattr(model, "environment_metadata", lambda: {})(),
    }


@dataclass(frozen=True)
class FutureForecastResult:
    request: ForecastRequest
    output: ForecastOutput
    context_sha256: str
    model_identity: dict
    model: dict
    runtime: dict
    environment: dict
    source_observations: int
    timing_seconds: dict


def run_future(
    dataset, model, spec, *, runtime=None, progress=None, should_cancel=None
) -> FutureForecastResult:
    """Run one future request, validating input/output without scores or writes."""

    def cancel():
        if should_cancel and should_cancel():
            raise KeyboardInterrupt("Future forecast cancellation requested")

    cancel()
    metadata = model.metadata()
    request = prepare_future_request(dataset, spec, adapter_id=metadata.adapter_id)
    caps = metadata.capabilities
    if (
        spec.context_length < caps.min_context
        or (caps.max_context is not None and spec.context_length > caps.max_context)
        or (caps.max_horizon is not None and spec.horizon > caps.max_horizon)
    ):
        raise ValueError("Future context or horizon exceeds the native adapter limits")
    if (
        caps.target_count != 1
        or "point" not in caps.output_types
        or "cpu" not in caps.devices
    ):
        raise ValueError(
            "Future forecasting requires a single-series CPU point adapter"
        )
    if spec.point_statistic not in model_settings(metadata.adapter_id).supported_tasks:
        raise ValueError("Unsupported future point task for this adapter")
    model.validate(spec)
    validate_model_frequency(model, dataset.frequency)
    options = model.configuration()
    identity = model_identity(metadata.adapter_id, metadata.adapter_version, options)
    effective = _runtime(model, runtime)
    measurement = {
        "target_name": dataset.target_name,
        "units": dataset.units,
        "frequency": dataset.frequency,
    }
    # Requested task is distinct from the model's returned point statistic.
    identifier = request_identity(
        request,
        **measurement,
        point_statistic=spec.point_statistic,
        model_identity=identity,
        runtime=effective,
    )
    request = ForecastRequest(**{**vars(request), "request_id": identifier})
    expected = ForecastRequest(**vars(request))
    context_sha = context_identity(expected, **measurement)
    environment = _clone(_environment(model))
    descriptor = _clone(
        {
            **asdict(metadata),
            "source": getattr(model, "source", "builtin"),
            "options": options,
        }
    )
    try:
        cancel()
        if progress:
            progress("loading")
        cancel()
        before = perf_counter()
        model.load()
        load_time = perf_counter() - before
        cancel()
        if progress:
            progress("running")
        cancel()
        before = perf_counter()
        outputs = model.predict((request,))
        inference_time = perf_counter() - before
        cancel()
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 1:
            raise ValueError("Future adapter must return exactly one forecast output")
        output = outputs[0]
        if (
            request.request_id != expected.request_id
            or request.series_id != expected.series_id
            or request.origin != expected.origin
            or not request.target_timestamps.equals(expected.target_timestamps)
            or context_identity(request, **measurement) != context_sha
        ):
            raise ValueError("Adapter mutated its detached future request")
        if (
            not isinstance(output, ForecastOutput)
            or output.request_id != expected.request_id
            or output.series_id != expected.series_id
            or output.origin != expected.origin
            or output.adapter_id != metadata.adapter_id
            or not output.target_timestamps.equals(expected.target_timestamps)
        ):
            raise ValueError(
                "Future output identity or timestamps do not match the request"
            )
        if output.point_statistic not in output_statistics(spec.point_statistic):
            raise ValueError(
                "Future output point statistic does not match the requested task"
            )
        output = ForecastOutput(**vars(output))
    finally:
        model.unload()
    cancel()
    return FutureForecastResult(
        expected,
        output,
        context_sha,
        identity,
        descriptor,
        effective,
        environment,
        len(dataset.values),
        {"loading": load_time, "inference": inference_time},
    )
