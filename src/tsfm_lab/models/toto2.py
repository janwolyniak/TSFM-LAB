"""Bounded TOTO 2 4M medians from local strictly loaded safe weights."""

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
    ModelCapabilities,
    ModelMetadata,
)
from tsfm_lab.models.checkpoints import (
    ResolvedCheckpoint,
    validate_checkpoint_record,
    verify_checkpoint,
)
from tsfm_lab.models.registry import adapter_profile
from tsfm_lab.models.toto2_profile import (
    FORECAST_OPTIONS,
    MAX_CONTEXT,
    MAX_HORIZON,
    MIN_CONTEXT,
    QUANTILES,
    ensure_supported_python,
    validate_bounds,
    validate_values,
)


def _runtime():
    ensure_supported_python()
    from tsfm_lab.runtime_diagnostics import _matches

    try:
        requirements = adapter_profile("toto_2").runtime_requirements
        if any(
            not _matches(version(name), constraint) for name, constraint in requirements
        ):
            raise ImportError(
                "TOTO 2 requires the reviewed tsfm-lab[toto2] runtime; use requirements-toto2.txt"
            )
        return importlib.import_module("torch"), SimpleNamespace(
            Model=importlib.import_module("toto2").Toto2Model,
            Config=importlib.import_module("toto2.configuration").Toto2ModelConfig,
            load_file=importlib.import_module("safetensors.torch").load_file,
        )
    except (PackageNotFoundError, ModuleNotFoundError) as exc:
        raise ImportError(
            "TOTO 2 requires optional tsfm-lab[toto2] dependencies on Python 3.12–3.13"
        ) from exc


def _strict_model(checkpoint, torch, runtime):
    class StrictToto(runtime.Model):
        @staticmethod
        def _clamp_nonfinite(values):
            # Upstream replaces infinities; refuse them before feedback or sorting.
            if not torch.isfinite(values).all().item():
                raise ValueError("TOTO 2 native inverse produced nonfinite quantiles")
            return values

    model = state = None
    try:
        with torch.device("cpu"):
            model = StrictToto(runtime.Config(**copy.deepcopy(checkpoint.config)))
            state = runtime.load_file(
                str(checkpoint.path / "model.safetensors"), device="cpu"
            )
            information = model.load_state_dict(state, strict=True)
            if information.missing_keys or information.unexpected_keys:
                raise ValueError(
                    "TOTO 2 weights contain missing or unexpected parameters"
                )
            state = None
            model.to(device="cpu", dtype=torch.float32)
            model.eval()
        if list(model.output_head.knots) != QUANTILES:
            raise ValueError(
                "TOTO 2 runtime quantile knots disagree with the reviewed profile"
            )
        return model
    except BaseException as exc:
        model = state = None
        traceback.clear_frames(exc.__traceback__)
        raise


class Toto2:
    """One regular univariate CPU float32 request per native forecast call."""

    adapter_id = "toto_2"

    def __init__(
        self, checkpoint: ResolvedCheckpoint, *, device="cpu", dtype="float32", seed=0
    ):
        if not isinstance(checkpoint, ResolvedCheckpoint):
            raise ValueError("TOTO 2 requires a resolved checkpoint")
        if device != "cpu" or dtype != "float32":
            raise ValueError("TOTO 2 supports only cpu and float32")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        record = validate_checkpoint_record(checkpoint.record())
        if record["architecture"] != "Toto2Model":
            raise ValueError("TOTO 2 requires its own reviewed checkpoint architecture")
        self.checkpoint, self.seed = checkpoint, int(seed)
        self._record = copy.deepcopy(record)
        self._torch = self._model = None

    @property
    def source(self):
        return self.checkpoint.source

    def metadata(self):
        return ModelMetadata(
            self.adapter_id,
            __version__,
            "toto",
            ModelCapabilities(
                min_context=MIN_CONTEXT,
                max_context=MAX_CONTEXT,
                max_horizon=MAX_HORIZON,
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
            "forecast_options": copy.deepcopy(FORECAST_OPTIONS),
        }

    def environment_metadata(self):
        return {
            name: version(name)
            for name, _ in adapter_profile(self.adapter_id).runtime_requirements
        }

    def validate(self, spec):
        validate_bounds(spec.context_length, spec.horizon)
        if spec.point_statistic not in {"median", "native_point"}:
            raise ValueError("TOTO 2 requires a median or native_point task")

    def load(self):
        if self._model is not None:
            return
        ensure_supported_python()
        checked = verify_checkpoint(self.checkpoint)
        if checked.record() != self._record:
            raise ValueError("Checkpoint identity changed after adapter configuration")
        torch, runtime = _runtime()
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("TOTO 2 requires the default Torch dtype to be float32")
        try:
            self._model = _strict_model(checked, torch, runtime)
            self._torch = torch
        except BaseException as exc:
            self.unload()
            traceback.clear_frames(exc.__traceback__)
            raise

    def _inputs(self, requests):
        if not requests or any(
            not isinstance(item, ForecastRequest) for item in requests
        ):
            raise ValueError("TOTO 2 needs nonempty ForecastRequest objects")
        if len({item.request_id for item in requests}) != len(requests):
            raise ValueError("TOTO 2 batch request IDs must be unique")
        horizons = {len(item.target_timestamps) for item in requests}
        if len(horizons) != 1:
            raise ValueError("TOTO 2 requests in one batch need the same horizon")
        horizon = horizons.pop()
        inputs = []
        for request in requests:
            values = request.context_values
            validate_bounds(len(values), horizon)
            validate_values(values)
            inputs.append(np.array(values, dtype=np.float32, copy=True)[None, None, :])
        return inputs, horizon

    def predict(self, requests):
        if self._model is None:
            raise RuntimeError("Load TOTO 2 before requesting forecasts")
        if self._torch.get_default_dtype() != self._torch.float32:
            raise ValueError("TOTO 2 requires the default Torch dtype to be float32")
        inputs, horizon = self._inputs(requests)
        options = {
            key: FORECAST_OPTIONS[key]
            for key in (
                "decode_block_size",
                "has_missing_values",
                "scaler_fallback_min_obs",
                "quantile_real_cap_k",
            )
        }
        outputs = []
        with self._torch.device("cpu"), self._torch.inference_mode():
            for request, values in zip(requests, inputs, strict=True):
                target = self._torch.as_tensor(
                    values, dtype=self._torch.float32, device="cpu"
                )
                result = self._model.forecast(
                    {
                        "target": target,
                        "target_mask": self._torch.ones_like(
                            target, dtype=self._torch.bool
                        ),
                        "series_ids": self._torch.zeros(
                            (1, 1), dtype=self._torch.long, device="cpu"
                        ),
                    },
                    horizon=horizon,
                    **options,
                )
                try:
                    quantiles = result.detach().cpu().numpy()
                except (AttributeError, TypeError) as exc:
                    raise ValueError("TOTO 2 quantile output must be a tensor") from exc
                if (
                    quantiles.shape != (9, 1, 1, horizon)
                    or quantiles.dtype.kind not in "iuf"
                    or not np.isfinite(quantiles).all()
                    or np.any(np.diff(quantiles, axis=0) < 0)
                ):
                    raise ValueError(
                        "TOTO 2 quantiles must be ordered and finite with shape [9,1,1,H]"
                    )
                outputs.append(
                    ForecastOutput(
                        request.request_id,
                        request.series_id,
                        request.origin,
                        request.target_timestamps,
                        np.array(quantiles[4, 0, 0], dtype=np.float64, copy=True),
                        self.adapter_id,
                        "median",
                    )
                )
        return tuple(outputs)

    def unload(self):
        self._model = self._torch = None
