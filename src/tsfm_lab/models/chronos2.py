"""Independent univariate Chronos-2 medians from verified local Safetensors."""

import copy
import importlib
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

FORECAST_OPTIONS = {"cross_learning": False, "limit_prediction_length": True}
_DISTRIBUTIONS = (
    "chronos-forecasting",
    "torch",
    "transformers",
    "huggingface-hub",
    "safetensors",
    "accelerate",
)


def _runtime():
    try:
        installed = {name: version(name) for name in _DISTRIBUTIONS}
        if (
            installed["chronos-forecasting"] != "2.2.0"
            or not installed["torch"].startswith("2.8.")
            or not installed["transformers"].startswith("4.56.")
            or installed["huggingface-hub"] != "0.35.3"
            or not installed["safetensors"].startswith("0.6.")
            or not installed["accelerate"].startswith("1.12.")
        ):
            raise ImportError(
                "Chronos-2 requires chronos-forecasting 2.2.0, torch 2.8.x, "
                "transformers 4.56.x, huggingface-hub 0.35.3, safetensors 0.6.x, "
                "and accelerate 1.12.x; use tsfm-lab[chronos]"
            )
        return importlib.import_module("torch"), importlib.import_module("chronos")
    except (PackageNotFoundError, ModuleNotFoundError) as exc:
        raise ImportError(
            "Chronos-2 requires the optional tsfm-lab[chronos] dependencies"
        ) from exc


def _strict_pipeline(path, torch, chronos):
    base = importlib.import_module("chronos.chronos2").Chronos2Model

    class StrictChronos2(base):
        _keys_to_ignore_on_load_missing = []
        _keys_to_ignore_on_load_unexpected = []

    model = None
    try:
        with torch.device("cpu"):
            model, information = StrictChronos2.from_pretrained(
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
                "Chronos-2 checkpoint has incomplete or incompatible model weights"
            )
        model.eval()
        return chronos.Chronos2Pipeline(model=model)
    except BaseException:
        model = None
        raise


class Chronos2:
    """CPU float32 inference with context-only normalization and native q=0.5."""

    adapter_id = "chronos_2"

    def __init__(
        self,
        checkpoint: ResolvedCheckpoint,
        *,
        device: str = "cpu",
        dtype: str = "float32",
        seed: int = 0,
    ) -> None:
        if not isinstance(checkpoint, ResolvedCheckpoint):
            raise ValueError("Chronos-2 requires a resolved checkpoint")
        if device != "cpu" or dtype != "float32":
            raise ValueError("Chronos-2 currently supports only cpu and float32")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        record = validate_checkpoint_record(checkpoint.record())
        if record["architecture"] != "Chronos2Model":
            raise ValueError("Chronos-2 requires its own checkpoint architecture")
        self.checkpoint = checkpoint
        self.seed = int(seed)
        self._record = copy.deepcopy(record)
        self._max_context = record["native_context_length"]
        self._max_horizon = record["native_prediction_length"]
        self._quantiles = tuple(record["quantiles"])
        self._pipeline = None
        self._torch = None

    @property
    def source(self) -> str:
        return self.checkpoint.source

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            self.adapter_id,
            __version__,
            "chronos",
            ModelCapabilities(
                max_context=self._max_context, max_horizon=self._max_horizon
            ),
        )

    def configuration(self) -> dict:
        return {"checkpoint": copy.deepcopy(self._record)}

    def verification_metadata(self) -> dict:
        from tsfm_lab.models.verification import checkpoint_verification

        return checkpoint_verification(self._record)

    def runtime_configuration(self) -> dict:
        return {
            "device": "cpu",
            "dtype": "float32",
            "batch_size": 1,
            "seed": self.seed,
            "seed_used": False,
            "deterministic": True,
            "forecast_options": copy.deepcopy(FORECAST_OPTIONS),
        }

    def environment_metadata(self) -> dict[str, str]:
        return {name: version(name) for name in _DISTRIBUTIONS}

    def validate(self, spec: EvaluationSpec) -> None:
        if not 0 < spec.context_length <= self._max_context:
            raise ValueError(
                "Chronos-2 context_length exceeds its native context limit"
            )
        if not 0 < spec.horizon <= self._max_horizon:
            raise ValueError("Chronos-2 horizon exceeds its native prediction limit")
        if spec.point_statistic not in {"median", "native_point"}:
            raise ValueError(
                "Chronos-2 point forecasts require a median or native_point task"
            )

    def load(self) -> None:
        if self._pipeline is not None:
            return
        checked = verify_checkpoint(self.checkpoint)
        if checked.record() != self._record:
            raise ValueError("Checkpoint identity changed after adapter configuration")
        torch, chronos = _runtime()
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("Chronos-2 requires the default Torch dtype to be float32")
        pipeline = _strict_pipeline(checked.path, torch, chronos)
        self._torch, self._pipeline = torch, pipeline

    def _inputs(self, requests):
        if not requests or any(
            not isinstance(item, ForecastRequest) for item in requests
        ):
            raise ValueError("Chronos-2 needs nonempty ForecastRequest objects")
        if len({request.request_id for request in requests}) != len(requests):
            raise ValueError("Chronos-2 batch request IDs must be unique")
        horizons = {len(request.target_timestamps) for request in requests}
        if len(horizons) != 1:
            raise ValueError("Chronos-2 requests in one batch need the same horizon")
        horizon = horizons.pop()
        if not 0 < horizon <= self._max_horizon:
            raise ValueError("Chronos-2 request exceeds its native prediction limit")
        inputs = []
        for request in requests:
            values = request.context_values
            if values.ndim != 1 or not 0 < len(values) <= self._max_context:
                raise ValueError("Chronos-2 request exceeds its native context limit")
            if not np.isfinite(values).all() or np.any(
                np.abs(values) > np.finfo(np.float32).max
            ):
                raise ValueError(
                    "Chronos-2 history must be finite and representable as float32"
                )
            inputs.append(
                self._torch.as_tensor(
                    np.array(values, copy=True), dtype=self._torch.float32, device="cpu"
                )
            )
        return inputs, horizon

    def predict(
        self, requests: tuple[ForecastRequest, ...]
    ) -> tuple[ForecastOutput, ...]:
        if self._pipeline is None:
            raise RuntimeError("Load Chronos-2 before requesting forecasts")
        if self._torch.get_default_dtype() != self._torch.float32:
            raise ValueError("Chronos-2 requires the default Torch dtype to be float32")
        inputs, horizon = self._inputs(requests)
        with self._torch.device("cpu"), self._torch.inference_mode():
            result = self._pipeline.predict(
                inputs=inputs,
                prediction_length=horizon,
                context_length=self._max_context,
                batch_size=1,
                **FORECAST_OPTIONS,
            )
        if not isinstance(result, list) or len(result) != len(requests):
            raise ValueError("Chronos-2 must return one quantile tensor per request")
        outputs = []
        median_index = self._quantiles.index(0.5)
        for request, tensor in zip(requests, result, strict=True):
            try:
                quantiles = tensor.detach().cpu().numpy()
            except (AttributeError, TypeError) as exc:
                raise ValueError("Chronos-2 quantile output must be a tensor") from exc
            if (
                quantiles.shape != (1, len(self._quantiles), horizon)
                or quantiles.dtype.kind not in "iuf"
                or not np.isfinite(quantiles).all()
            ):
                raise ValueError(
                    "Chronos-2 quantiles must be finite with shape [1,Q,H]"
                )
            outputs.append(
                ForecastOutput(
                    request_id=request.request_id,
                    series_id=request.series_id,
                    origin=request.origin,
                    target_timestamps=request.target_timestamps,
                    point=np.array(
                        quantiles[0, median_index], dtype=np.float64, copy=True
                    ),
                    adapter_id=self.adapter_id,
                    point_statistic="median",
                )
            )
        return tuple(outputs)

    def unload(self) -> None:
        self._pipeline = None
        self._torch = None
