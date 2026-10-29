"""Optional TimesFM 2.5 median forecasts from verified local Safetensors."""

import copy
import importlib
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral

import numpy as np

from tsfm_lab import __version__
from tsfm_lab.contracts import (
    ForecastOutput,
    ForecastRequest,
    ForecastSettings,
    ModelCapabilities,
    ModelMetadata,
)
from tsfm_lab.models.checkpoint_profiles import ADAPTER_LIMITS
from tsfm_lab.models.checkpoints import (
    ResolvedCheckpoint,
    validate_checkpoint_record,
    verify_checkpoint,
)

FORECAST_OPTIONS = {
    "normalize_inputs": True,
    "force_flip_invariance": True,
    "infer_is_positive": False,
    "use_continuous_quantile_head": False,
    "fix_quantile_crossing": False,
    "return_backcast": False,
}
_DISTRIBUTIONS = ("timesfm", "torch", "huggingface-hub", "safetensors")


def _runtime():
    try:
        installed = {name: version(name) for name in _DISTRIBUTIONS}
        if (
            installed["timesfm"] != "3.0.2"
            or not installed["torch"].startswith("2.8.")
            or installed["huggingface-hub"] != "0.35.3"
            or not installed["safetensors"].startswith("0.6.")
        ):
            raise ImportError(
                "TimesFM 2.5 requires timesfm 3.0.2, torch 2.8.x, "
                "huggingface-hub 0.35.3, and safetensors 0.6.x; "
                "use tsfm-lab[timesfm]"
            )
        torch = importlib.import_module("torch")
        timesfm = importlib.import_module("timesfm")
        if not hasattr(timesfm, "TimesFM_2p5_200M_torch"):
            raise ImportError("The installed TimesFM runtime lacks its 2.5 Torch API")
        return torch, timesfm
    except (PackageNotFoundError, ModuleNotFoundError) as exc:
        raise ImportError(
            "TimesFM 2.5 requires the optional tsfm-lab[timesfm] dependencies"
        ) from exc


class TimesFM:
    """CPU float32 TimesFM 2.5 with one native output patch and median scoring."""

    adapter_id = "timesfm_2p5"

    def __init__(
        self,
        checkpoint: ResolvedCheckpoint,
        *,
        device: str = "cpu",
        dtype: str = "float32",
        seed: int = 0,
    ) -> None:
        if not isinstance(checkpoint, ResolvedCheckpoint):
            raise ValueError("TimesFM requires a resolved checkpoint")
        if device != "cpu" or dtype != "float32":
            raise ValueError("TimesFM currently supports only cpu and float32")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        record = validate_checkpoint_record(checkpoint.record())
        if record["architecture"] != "TimesFmModelForPrediction":
            raise ValueError("TimesFM 2.5 requires its own checkpoint architecture")
        self.checkpoint = checkpoint
        self.seed = int(seed)
        self._record = copy.deepcopy(record)
        # Upstream rounds context to 32 and horizon to 128 before checking their sum.
        self._max_context, self._max_horizon = ADAPTER_LIMITS[self.adapter_id]
        self._pipeline = None
        self._torch = None
        self._timesfm = None
        self._compiled_context = None

    @property
    def source(self) -> str:
        return self.checkpoint.source

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            self.adapter_id,
            __version__,
            "timesfm",
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
            "torch_compile": False,
            "forecast_options": copy.deepcopy(FORECAST_OPTIONS),
        }

    def environment_metadata(self) -> dict[str, str]:
        return {name: version(name) for name in _DISTRIBUTIONS}

    def validate(self, spec: ForecastSettings) -> None:
        if not 0 < spec.context_length <= self._max_context:
            raise ValueError("TimesFM context_length exceeds its supported limit")
        if not 0 < spec.horizon <= self._max_horizon:
            raise ValueError("TimesFM horizon exceeds one native prediction patch")
        if spec.point_statistic not in {"median", "native_point"}:
            raise ValueError(
                "TimesFM point forecasts require a median or native_point task"
            )

    def load(self) -> None:
        if self._pipeline is not None:
            return
        checked = verify_checkpoint(self.checkpoint)
        if checked.record() != self._record:
            raise ValueError("Checkpoint identity changed after adapter configuration")
        torch, timesfm = _runtime()
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("TimesFM requires the default Torch dtype to be float32")
        # Avoid upstream's automatic accelerator choice and any ambient default device.
        pipeline = None
        try:
            with torch.device("cpu"):
                pipeline = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
            pipeline.model.device = torch.device("cpu")
            pipeline.model.device_count = 1
            pipeline.model.to(device="cpu", dtype=torch.float32)
            # The pinned runtime uses Safetensors and strict=True, without its Hub loader.
            pipeline.load_checkpoint(
                str(checked.path / "model.safetensors"), torch_compile=False
            )
            pipeline.model.to(device="cpu", dtype=torch.float32)
            pipeline.model.eval()
        except BaseException:
            pipeline = None
            self.unload()
            raise
        self._torch, self._timesfm, self._pipeline = torch, timesfm, pipeline

    def _inputs(self, requests):
        if not requests or any(
            not isinstance(item, ForecastRequest) for item in requests
        ):
            raise ValueError("TimesFM needs nonempty ForecastRequest objects")
        if len({request.request_id for request in requests}) != len(requests):
            raise ValueError("TimesFM batch request IDs must be unique")
        horizons = {len(request.target_timestamps) for request in requests}
        lengths = {len(request.context_values) for request in requests}
        if len(horizons) != 1 or len(lengths) != 1:
            raise ValueError("TimesFM batch requests need the same context and horizon")
        horizon, length = horizons.pop(), lengths.pop()
        if not 0 < horizon <= self._max_horizon:
            raise ValueError("TimesFM request exceeds one native prediction patch")
        if not 0 < length <= self._max_context:
            raise ValueError("TimesFM request exceeds its supported context limit")
        inputs = []
        for request in requests:
            values = request.context_values
            if (
                values.ndim != 1
                or not np.isfinite(values).all()
                or np.any(np.abs(values) > np.finfo(np.float32).max)
            ):
                raise ValueError(
                    "TimesFM history must be finite and representable as float32"
                )
            inputs.append(np.array(values, dtype=np.float32, copy=True))
        return inputs, length, horizon

    def predict(
        self, requests: tuple[ForecastRequest, ...]
    ) -> tuple[ForecastOutput, ...]:
        if self._pipeline is None:
            raise RuntimeError("Load TimesFM before requesting forecasts")
        if self._torch.get_default_dtype() != self._torch.float32:
            raise ValueError("TimesFM requires the default Torch dtype to be float32")
        inputs, length, horizon = self._inputs(requests)
        context = ((length + 31) // 32) * 32
        if context != self._compiled_context:
            self._pipeline.compile(
                self._timesfm.ForecastConfig(
                    max_context=context,
                    max_horizon=128,
                    per_core_batch_size=1,
                    **FORECAST_OPTIONS,
                )
            )
            self._compiled_context = context
        with self._torch.inference_mode():
            result = self._pipeline.forecast(horizon=horizon, inputs=inputs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("TimesFM must return point and quantile arrays")
        points, quantiles = result
        if (
            not isinstance(points, np.ndarray)
            or not isinstance(quantiles, np.ndarray)
            or points.shape != (len(requests), horizon)
            or quantiles.shape != (len(requests), horizon, 10)
            or points.dtype.kind not in "iuf"
            or quantiles.dtype.kind not in "iuf"
            or not np.isfinite(points).all()
            or not np.isfinite(quantiles).all()
        ):
            raise ValueError("TimesFM needs finite outputs with shapes [B,H], [B,H,10]")
        # Channel 0 is the mean; 5 is q=0.5 in the pinned 2.5 runtime.
        if not np.array_equal(points, quantiles[:, :, 5]):
            raise ValueError("TimesFM point output disagrees with its median channel")
        return tuple(
            ForecastOutput(
                request_id=request.request_id,
                series_id=request.series_id,
                origin=request.origin,
                target_timestamps=request.target_timestamps,
                point=np.array(quantiles[index, :, 5], dtype=np.float64, copy=True),
                adapter_id=self.adapter_id,
                point_statistic="median",
            )
            for index, request in enumerate(requests)
        )

    def unload(self) -> None:
        self._pipeline = None
        self._torch = None
        self._timesfm = None
        self._compiled_context = None
