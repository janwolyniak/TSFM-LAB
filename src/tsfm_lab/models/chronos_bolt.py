"""Optional Chronos-Bolt median forecasts from a verified local checkpoint."""

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
from tsfm_lab.models.checkpoints import ResolvedCheckpoint, verify_checkpoint


def _runtime():
    try:
        versions = {
            name: version(name)
            for name in (
                "chronos-forecasting",
                "torch",
                "transformers",
                "huggingface-hub",
            )
        }
        if (
            versions["chronos-forecasting"] != "2.2.0"
            or not versions["torch"].startswith("2.8.")
            or not versions["transformers"].startswith("4.55.")
            or versions["huggingface-hub"] != "0.34.4"
        ):
            raise ImportError(
                "Chronos-Bolt requires chronos-forecasting 2.2.0, torch 2.8.x, "
                "transformers 4.55.x, and huggingface-hub 0.34.4; use tsfm-lab[chronos]"
            )
        return importlib.import_module("torch"), importlib.import_module("chronos")
    except (PackageNotFoundError, ModuleNotFoundError) as exc:
        raise ImportError(
            "Chronos-Bolt requires the optional tsfm-lab[chronos] dependencies"
        ) from exc


def _limits(checkpoint: ResolvedCheckpoint) -> tuple[int, int]:
    try:
        options = checkpoint.config["chronos_config"]
        context, horizon = options["context_length"], options["prediction_length"]
        if type(context) is not int or not 0 < context <= 2048:
            raise ValueError("context limit must be an integer in 1..2048")
        if type(horizon) is not int or not 0 < horizon <= 64:
            raise ValueError("horizon limit must be an integer in 1..64")
        if 0.5 not in options["quantiles"]:
            raise ValueError("checkpoint must include the median quantile")
        return context, horizon
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid Chronos-Bolt checkpoint capabilities") from exc


def _strict_pipeline(path, torch, chronos):
    base = importlib.import_module(
        "chronos.chronos_bolt"
    ).ChronosBoltModelForForecasting

    class StrictBolt(base):
        # Upstream suppresses missing forecasting heads; an evaluator cannot.
        _keys_to_ignore_on_load_missing = []
        _keys_to_ignore_on_load_unexpected = []

    model, information = StrictBolt.from_pretrained(
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
            "Chronos-Bolt checkpoint has incomplete or incompatible model weights"
        )
    model.eval()
    return chronos.ChronosBoltPipeline(model=model)


class ChronosBolt:
    """CPU float32 model inference, normalized to float64 median point outputs."""

    adapter_id = "chronos_bolt"

    def __init__(
        self,
        checkpoint: ResolvedCheckpoint,
        *,
        device: str = "cpu",
        dtype: str = "float32",
        seed: int = 0,
    ) -> None:
        if not isinstance(checkpoint, ResolvedCheckpoint):
            raise ValueError("Chronos-Bolt requires a resolved checkpoint")
        if device != "cpu" or dtype != "float32":
            raise ValueError("Chronos-Bolt currently supports only cpu and float32")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.checkpoint = checkpoint
        self.seed = int(seed)
        self._max_context, self._max_horizon = _limits(checkpoint)
        self._record = copy.deepcopy(checkpoint.record())
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
        }

    def environment_metadata(self) -> dict[str, str]:
        """Read installed distribution versions without importing model runtimes."""
        return {
            name: version(name)
            for name in (
                "torch",
                "chronos-forecasting",
                "transformers",
                "huggingface-hub",
                "safetensors",
                "accelerate",
            )
        }

    def validate(self, spec: EvaluationSpec) -> None:
        if not 0 < spec.context_length <= self._max_context:
            raise ValueError(
                "Chronos-Bolt context_length exceeds its native context limit"
            )
        if not 0 < spec.horizon <= self._max_horizon:
            raise ValueError("Chronos-Bolt horizon exceeds its native prediction limit")
        if spec.point_statistic != "median":
            raise ValueError("Chronos-Bolt point forecasts require the median task")

    def load(self) -> None:
        if self._pipeline is not None:
            return
        checked = verify_checkpoint(self.checkpoint)
        if checked.record() != self._record:
            raise ValueError("Checkpoint identity changed after adapter configuration")
        torch, chronos = _runtime()
        pipeline = _strict_pipeline(checked.path, torch, chronos)
        self._torch, self._pipeline = torch, pipeline

    def _inputs(self, requests: tuple[ForecastRequest, ...]):
        if not requests or any(
            not isinstance(item, ForecastRequest) for item in requests
        ):
            raise ValueError("Chronos-Bolt needs nonempty ForecastRequest objects")
        if len({request.request_id for request in requests}) != len(requests):
            raise ValueError("Chronos-Bolt batch request IDs must be unique")
        horizons = {len(request.target_timestamps) for request in requests}
        if len(horizons) != 1:
            raise ValueError("Chronos-Bolt requests in one batch need the same horizon")
        horizon = horizons.pop()
        if not 0 < horizon <= self._max_horizon:
            raise ValueError("Chronos-Bolt request exceeds its native prediction limit")
        inputs = []
        for request in requests:
            values = request.context_values
            if values.ndim != 1 or not 0 < len(values) <= self._max_context:
                raise ValueError(
                    "Chronos-Bolt request exceeds its native context limit"
                )
            if not np.isfinite(values).all() or np.any(
                np.abs(values) > np.finfo(np.float32).max
            ):
                raise ValueError(
                    "Chronos-Bolt history must be finite and representable as float32"
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
            raise RuntimeError("Load Chronos-Bolt before requesting forecasts")
        inputs, horizon = self._inputs(requests)
        with self._torch.inference_mode():
            result = self._pipeline.predict_quantiles(
                inputs=inputs, prediction_length=horizon, quantile_levels=[0.5]
            )
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("Chronos-Bolt must return quantile and point tensors")
        # The second upstream output is misleadingly named mean; use q=0.5 itself.
        try:
            quantiles = result[0].detach().cpu().numpy()
        except (AttributeError, TypeError) as exc:
            raise ValueError("Chronos-Bolt quantile output must be a tensor") from exc
        if (
            quantiles.shape != (len(requests), horizon, 1)
            or quantiles.dtype.kind not in "iuf"
            or not np.isfinite(quantiles).all()
        ):
            raise ValueError("Chronos-Bolt quantiles must be finite with shape [B,H,1]")
        return tuple(
            ForecastOutput(
                request_id=request.request_id,
                series_id=request.series_id,
                origin=request.origin,
                target_timestamps=request.target_timestamps,
                point=np.array(quantiles[index, :, 0], dtype=np.float64, copy=True),
                adapter_id=self.adapter_id,
                point_statistic="median",
            )
            for index, request in enumerate(requests)
        )

    def unload(self) -> None:
        self._pipeline = None
        self._torch = None
