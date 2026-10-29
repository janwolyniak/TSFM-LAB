"""Independent native TiRex-2 median forecasts from verified Safetensors."""

import copy
import importlib
import traceback
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral
from types import SimpleNamespace

import numpy as np

from tsfm_lab import __version__
from tsfm_lab.contracts import (
    ForecastOutput,
    ForecastRequest,
    ForecastSettings,
    ModelCapabilities,
    ModelMetadata,
)
from tsfm_lab.models.checkpoints import (
    ResolvedCheckpoint,
    validate_checkpoint_record,
    verify_checkpoint,
)
from tsfm_lab.models.tirex2_profile import forecast_options_for_record

_DISTRIBUTIONS = (
    "tirex-2",
    "torch",
    "huggingface-hub",
    "safetensors",
    "numpy",
    "flashrnn",
    "mlstm-kernels",
    "ninja",
    "narwhals",
)


def _runtime():
    try:
        installed = {name: version(name) for name in _DISTRIBUTIONS}
        from packaging.version import Version

        exact = {
            "tirex-2": "0.4.0",
            "huggingface-hub": "0.35.3",
            "flashrnn": "1.0.8",
            "mlstm-kernels": "2.0.6",
            "ninja": "1.13.2",
            "narwhals": "2.13.0",
        }
        if (
            any(installed[name] != wanted for name, wanted in exact.items())
            or not installed["torch"].startswith("2.8.")
            or not installed["safetensors"].startswith("0.6.")
            or not Version("2.1.3") <= Version(installed["numpy"]) < Version("3")
        ):
            raise ImportError(
                "TiRex-2 requires the pinned tsfm-lab[tirex2] runtime; use requirements-tirex2.txt"
            )
        return importlib.import_module("torch"), SimpleNamespace(
            TiRex2=importlib.import_module("tirex2.model.tirex2").TiRex2,
            TimeseriesType=importlib.import_module("tirex2.model.types").TimeseriesType,
            ResidualBlock=importlib.import_module(
                "tirex2.model.component.residual_block"
            ).ResidualBlock,
            load_file=importlib.import_module("safetensors.torch").load_file,
            expand_shared_duplicates=importlib.import_module(
                "tirex2._state_dict"
            ).expand_shared_duplicates,
        )
    except (PackageNotFoundError, ModuleNotFoundError) as exc:
        raise ImportError(
            "TiRex-2 requires the optional tsfm-lab[tirex2] dependencies"
        ) from exc


def _eager_blocks(model, torch, runtime):
    class EagerResidualBlock(runtime.ResidualBlock):
        # Upstream decorates this class even when loader compilation is disabled.
        # Restore ordinary calls per instance, preserving forward, hooks and weights.
        __call__ = torch.nn.Module.__call__
        _call_impl = torch.nn.Module._call_impl

    count = 0
    for module in model.modules():
        if isinstance(module, runtime.ResidualBlock):
            module.__class__ = EagerResidualBlock
            count += 1
    if count != 2:
        raise ValueError("TiRex-2 requires two reviewed residual embedding blocks")


def _strict_model(checkpoint, torch, runtime):
    model = state = expanded = None
    try:
        with torch.device("cpu"):
            model = runtime.TiRex2(**copy.deepcopy(checkpoint.config), device="cpu")
            model.to(device="cpu", dtype=torch.float32)
            state = runtime.load_file(
                str(checkpoint.path / "model.safetensors"), device="cpu"
            )
            expanded = runtime.expand_shared_duplicates(model, state)
            information = model.load_state_dict(expanded, strict=True)
            if any(
                not isinstance(getattr(information, field, None), list)
                or getattr(information, field)
                for field in ("missing_keys", "unexpected_keys")
            ):
                raise ValueError("TiRex-2 checkpoint has incompatible model weights")
            _eager_blocks(model, torch, runtime)
            model.eval()
        return model
    except BaseException as exc:
        model = state = expanded = None
        traceback.clear_frames(exc.__traceback__)
        raise


class TiRex2:
    """CPU float32 eager inference, one independent univariate request per call."""

    adapter_id = "tirex_2"

    def __init__(
        self, checkpoint: ResolvedCheckpoint, *, device="cpu", dtype="float32", seed=0
    ):
        if not isinstance(checkpoint, ResolvedCheckpoint):
            raise ValueError("TiRex-2 requires a resolved checkpoint")
        if device != "cpu" or dtype != "float32":
            raise ValueError("TiRex-2 currently supports only cpu and float32")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        record = validate_checkpoint_record(checkpoint.record())
        if record["architecture"] != "TiRex2":
            raise ValueError("TiRex-2 requires its own checkpoint architecture")
        self.checkpoint, self.seed = checkpoint, int(seed)
        self._record = copy.deepcopy(record)
        self._max_context = record["native_context_length"]
        self._max_horizon = record["native_prediction_length"]
        self._forecast_options = forecast_options_for_record(record)
        self._model = self._torch = self._series_type = self._predict = None

    @property
    def source(self):
        return self.checkpoint.source

    def metadata(self):
        return ModelMetadata(
            self.adapter_id,
            __version__,
            "tirex",
            ModelCapabilities(
                max_context=self._max_context, max_horizon=self._max_horizon
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
        return {name: version(name) for name in _DISTRIBUTIONS}

    def validate(self, spec: ForecastSettings):
        if not 0 < spec.context_length <= self._max_context:
            raise ValueError("TiRex-2 context_length exceeds its native context limit")
        if not 0 < spec.horizon <= self._max_horizon:
            raise ValueError("TiRex-2 horizon exceeds its native prediction limit")
        if spec.point_statistic not in {"median", "native_point"}:
            raise ValueError(
                "TiRex-2 point forecasts require a median or native_point task"
            )

    def load(self):
        if self._model is not None:
            return
        checked = verify_checkpoint(self.checkpoint)
        if checked.record() != self._record:
            raise ValueError("Checkpoint identity changed after adapter configuration")
        torch, runtime = _runtime()
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("TiRex-2 requires the default Torch dtype to be float32")
        model = None
        try:
            model = _strict_model(checked, torch, runtime)
            predict = torch.compiler.disable(model.predict, recursive=True)
        except BaseException as exc:
            model = None
            traceback.clear_frames(exc.__traceback__)
            self.unload()
            raise
        self._model, self._torch = model, torch
        self._series_type, self._predict = runtime.TimeseriesType, predict

    def _inputs(self, requests):
        if not requests or any(
            not isinstance(item, ForecastRequest) for item in requests
        ):
            raise ValueError("TiRex-2 needs nonempty ForecastRequest objects")
        if len({item.request_id for item in requests}) != len(requests):
            raise ValueError("TiRex-2 batch request IDs must be unique")
        horizons = {len(item.target_timestamps) for item in requests}
        if len(horizons) != 1:
            raise ValueError("TiRex-2 requests in one batch need the same horizon")
        horizon = horizons.pop()
        if not 0 < horizon <= self._max_horizon:
            raise ValueError("TiRex-2 request exceeds its native prediction limit")
        inputs = []
        for request in requests:
            values = request.context_values
            if values.ndim != 1 or not 0 < len(values) <= self._max_context:
                raise ValueError("TiRex-2 request exceeds its native context limit")
            if not np.isfinite(values).all() or np.any(
                np.abs(values) > np.finfo(np.float32).max
            ):
                raise ValueError(
                    "TiRex-2 history must be finite and representable as float32"
                )
            inputs.append(
                self._torch.as_tensor(
                    np.array(values[None, :], copy=True),
                    dtype=self._torch.float32,
                    device="cpu",
                )
            )
        return inputs, horizon

    def predict(
        self, requests: tuple[ForecastRequest, ...]
    ) -> tuple[ForecastOutput, ...]:
        if self._model is None:
            raise RuntimeError("Load TiRex-2 before requesting forecasts")
        if self._torch.get_default_dtype() != self._torch.float32:
            raise ValueError("TiRex-2 requires the default Torch dtype to be float32")
        inputs, horizon = self._inputs(requests)
        outputs = []
        with self._torch.device("cpu"), self._torch.inference_mode():
            for request, history in zip(requests, inputs, strict=True):
                series = self._series_type(
                    target=history, past_covariates=None, future_covariates=None
                )
                result = self._predict(
                    [series], prediction_length=horizon, **self._forecast_options
                )
                if not isinstance(result, list) or len(result) != 1:
                    raise ValueError(
                        "TiRex-2 must return one quantile tensor per request"
                    )
                try:
                    quantiles = result[0].detach().cpu().numpy()
                except (AttributeError, TypeError) as exc:
                    raise ValueError(
                        "TiRex-2 quantile output must be a tensor"
                    ) from exc
                if (
                    quantiles.shape != (1, 9, horizon)
                    or quantiles.dtype.kind not in "iuf"
                    or not np.isfinite(quantiles).all()
                ):
                    raise ValueError(
                        "TiRex-2 quantiles must be finite with shape [1,9,H]"
                    )
                outputs.append(
                    ForecastOutput(
                        request_id=request.request_id,
                        series_id=request.series_id,
                        origin=request.origin,
                        target_timestamps=request.target_timestamps,
                        point=np.array(quantiles[0, 4], dtype=np.float64, copy=True),
                        adapter_id=self.adapter_id,
                        point_statistic="median",
                    )
                )
        return tuple(outputs)

    def unload(self):
        self._predict = self._model = self._torch = self._series_type = None
