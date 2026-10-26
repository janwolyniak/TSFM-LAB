"""Causal, native point forecasts from reviewed Granite TTM R2 checkpoints."""

import copy
import importlib
import traceback
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral

import numpy as np

from tsfm_lab import __version__
from tsfm_lab.contracts import (
    EvaluationSpec,
    ForecastOutput,
    ForecastRequest,
    ModelCapabilities,
    ModelMetadata,
)
from tsfm_lab.models.checkpoints import (
    ResolvedCheckpoint,
    validate_checkpoint_record,
    verify_checkpoint,
)
from tsfm_lab.models.ttm_profile import (
    forecast_options_for_record,
    validate_frequency,
)

_REQUIREMENTS = {
    "granite-tsfm": "0.3.4",
    "torch": "2.8.",
    "transformers": "4.56.",
    "huggingface-hub": "0.35.3",
    "safetensors": "0.6.",
    "accelerate": "1.12.",
    "tokenizers": "0.22.0",
    "scikit-learn": "1.7.2",
    "filelock": "3.20.3",
}


def _runtime():
    try:
        installed = {name: version(name) for name in _REQUIREMENTS}
        if any(
            (
                not installed[name].startswith(required)
                if required.endswith(".")
                else installed[name] != required
            )
            for name, required in _REQUIREMENTS.items()
        ):
            raise ImportError(
                "TTM requires the reviewed shared runtime; use tsfm-lab[ttm] "
                "with requirements-ttm.txt"
            )
        return importlib.import_module("torch"), importlib.import_module(
            "tsfm_public.models.tinytimemixer"
        )
    except (PackageNotFoundError, ModuleNotFoundError) as exc:
        raise ImportError(
            "TTM requires the optional tsfm-lab[ttm] dependencies"
        ) from exc


def _strict_model(path, torch, runtime):
    class StrictTTM(runtime.TinyTimeMixerForPrediction):
        _keys_to_ignore_on_load_missing = []
        _keys_to_ignore_on_load_unexpected = []

    model = None
    try:
        with torch.device("cpu"):
            model, information = StrictTTM.from_pretrained(
                str(path),
                device_map="cpu",
                torch_dtype=torch.float32,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                output_loading_info=True,
                ignore_mismatched_sizes=False,
            )
        fields = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if not isinstance(information, dict) or any(
            not isinstance(information.get(field), list) or information[field]
            for field in fields
        ):
            raise ValueError(
                "TTM checkpoint has incomplete or incompatible model weights"
            )
        model.eval()
        return model
    except BaseException as exc:
        model = None
        traceback.clear_frames(exc.__traceback__)
        raise


def _standardize(values):
    history = np.array(values, dtype=np.float64, copy=True)
    if not np.isfinite(history).all():
        raise ValueError("TTM history must contain finite numeric values")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        location = history.mean()
        scale = 1.0 if np.all(history == history[0]) else history.std(ddof=0)
        normalized = (history - location) / scale
    if (
        not np.isfinite(location)
        or not np.isfinite(scale)
        or scale <= 0
        or not np.isfinite(normalized).all()
        or np.any(np.abs(normalized) > np.finfo(np.float32).max)
    ):
        raise ValueError("TTM history cannot be standardized safely in float64")
    return normalized.astype(np.float32), location, scale


class TTM:
    """One exact-context univariate request per eager CPU float32 model call."""

    adapter_id = "ttm_r2"

    def __init__(
        self, checkpoint: ResolvedCheckpoint, *, device="cpu", dtype="float32", seed=0
    ):
        if not isinstance(checkpoint, ResolvedCheckpoint):
            raise ValueError("TTM requires a resolved checkpoint")
        if device != "cpu" or dtype != "float32":
            raise ValueError("TTM currently supports only cpu and float32")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        record = validate_checkpoint_record(checkpoint.record())
        if record["architecture"] != "TinyTimeMixerForPrediction":
            raise ValueError("TTM requires its own checkpoint architecture")
        self.checkpoint, self.seed = checkpoint, int(seed)
        self._record = copy.deepcopy(record)
        self._context = record["native_context_length"]
        self._forecast_options = forecast_options_for_record(record)
        self._model = self._torch = None

    @property
    def source(self):
        return self.checkpoint.source

    def metadata(self):
        return ModelMetadata(
            self.adapter_id,
            __version__,
            "ttm",
            ModelCapabilities(
                min_context=self._context, max_context=self._context, max_horizon=96
            ),
        )

    def configuration(self):
        return {"checkpoint": copy.deepcopy(self._record)}

    def verification_metadata(self):
        from tsfm_lab.models.verification import checkpoint_verification

        return checkpoint_verification(self._record)

    def runtime_configuration(self):
        return {
            "device": "cpu",
            "dtype": "float32",
            "batch_size": 1,
            "seed": self.seed,
            "seed_used": False,
            "deterministic": True,
            "torch_compile": False,
            "forecast_options": copy.deepcopy(self._forecast_options),
        }

    def environment_metadata(self):
        return {name: version(name) for name in _REQUIREMENTS}

    def validate(self, spec: EvaluationSpec):
        if spec.context_length != self._context:
            raise ValueError(
                f"TTM requires exactly {self._context} history observations"
            )
        if not 0 < spec.horizon <= 96:
            raise ValueError("TTM horizon must be between 1 and 96")
        if spec.point_statistic != "native_point":
            raise ValueError("TTM requires a native_point evaluation task")

    def validate_frequency(self, frequency):
        validate_frequency(frequency)

    def load(self):
        if self._model is not None:
            return
        checked = verify_checkpoint(self.checkpoint)
        if checked.record() != self._record:
            raise ValueError("Checkpoint identity changed after adapter configuration")
        torch, runtime = _runtime()
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("TTM requires the default Torch dtype to be float32")
        try:
            self._model = _strict_model(checked.path, torch, runtime)
            self._torch = torch
        except BaseException as exc:
            self.unload()
            traceback.clear_frames(exc.__traceback__)
            raise

    def _inputs(self, requests):
        if not requests or any(
            not isinstance(item, ForecastRequest) for item in requests
        ):
            raise ValueError("TTM needs nonempty ForecastRequest objects")
        if len({item.request_id for item in requests}) != len(requests):
            raise ValueError("TTM batch request IDs must be unique")
        horizons = {len(item.target_timestamps) for item in requests}
        if len(horizons) != 1 or not 0 < next(iter(horizons)) <= 96:
            raise ValueError("TTM requests need the same horizon between 1 and 96")
        inputs = []
        for request in requests:
            if (
                request.context_values.ndim != 1
                or len(request.context_values) != self._context
            ):
                raise ValueError(
                    f"TTM requires exactly {self._context} history observations"
                )
            normalized, location, scale = _standardize(request.context_values)
            history = self._torch.as_tensor(
                normalized[None, :, None], dtype=self._torch.float32, device="cpu"
            )
            inputs.append((history, location, scale))
        return inputs, horizons.pop()

    def predict(
        self, requests: tuple[ForecastRequest, ...]
    ) -> tuple[ForecastOutput, ...]:
        if self._model is None:
            raise RuntimeError("Load TTM before requesting forecasts")
        if self._torch.get_default_dtype() != self._torch.float32:
            raise ValueError("TTM requires the default Torch dtype to be float32")
        inputs, horizon = self._inputs(requests)
        outputs = []
        with self._torch.device("cpu"), self._torch.inference_mode():
            for request, (history, location, scale) in zip(
                requests, inputs, strict=True
            ):
                result = self._model(past_values=history, return_dict=True)
                try:
                    native = result.prediction_outputs.detach().cpu().numpy()
                except (AttributeError, TypeError) as exc:
                    raise ValueError("TTM prediction_outputs must be a tensor") from exc
                if (
                    native.shape != (1, 96, 1)
                    or native.dtype.kind not in "iuf"
                    or not np.isfinite(native).all()
                ):
                    raise ValueError(
                        "TTM native predictions must be finite with shape [1,96,1]"
                    )
                with np.errstate(over="ignore", invalid="ignore"):
                    point = native[0, :horizon, 0].astype(np.float64) * scale + location
                if not np.isfinite(point).all():
                    raise ValueError(
                        "TTM external inverse scaling produced nonfinite values"
                    )
                outputs.append(
                    ForecastOutput(
                        request_id=request.request_id,
                        series_id=request.series_id,
                        origin=request.origin,
                        target_timestamps=request.target_timestamps,
                        point=point,
                        adapter_id=self.adapter_id,
                        point_statistic="native_point",
                    )
                )
        return tuple(outputs)

    def unload(self):
        self._model = self._torch = None
