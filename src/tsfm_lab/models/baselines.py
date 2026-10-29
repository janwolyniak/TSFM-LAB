"""Deterministic reference models using only request history."""

import numpy as np

from tsfm_lab import __version__
from tsfm_lab.contracts import (
    ForecastOutput,
    ForecastRequest,
    ForecastSettings,
    ModelCapabilities,
    ModelMetadata,
    positive_int,
)


class LastValue:
    """Repeat the last observed value over the requested horizon."""

    adapter_id = "last_value"

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            self.adapter_id, __version__, "baseline", ModelCapabilities()
        )

    def validate(self, spec: ForecastSettings) -> None:
        if spec.context_length < self.metadata().capabilities.min_context:
            raise ValueError("context_length is shorter than the baseline requires")

    def load(self) -> None:
        """Built-ins have no resources to initialize."""

    def _forecast(self, request: ForecastRequest) -> np.ndarray:
        return np.full(len(request.target_timestamps), request.context_values[-1])

    def predict(
        self, requests: tuple[ForecastRequest, ...]
    ) -> tuple[ForecastOutput, ...]:
        return tuple(
            ForecastOutput(
                request_id=request.request_id,
                series_id=request.series_id,
                origin=request.origin,
                target_timestamps=request.target_timestamps,
                point=self._forecast(request),
                adapter_id=self.adapter_id,
            )
            for request in requests
        )

    def unload(self) -> None:
        """Built-ins have no resources to release."""

    def configuration(self) -> dict[str, int]:
        return {}


class SeasonalNaive(LastValue):
    """Repeat the last complete observed seasonal cycle."""

    adapter_id = "seasonal_naive"

    def __init__(self, seasonal_period: int) -> None:
        self.seasonal_period = positive_int(seasonal_period, "seasonal_period")

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            self.adapter_id,
            __version__,
            "baseline",
            ModelCapabilities(min_context=self.seasonal_period),
        )

    def _forecast(self, request: ForecastRequest) -> np.ndarray:
        if len(request.context_values) < self.seasonal_period:
            raise ValueError("seasonal_naive requires at least one complete season")
        indices = np.arange(len(request.target_timestamps)) % self.seasonal_period
        return request.context_values[-self.seasonal_period :][indices]

    def configuration(self) -> dict[str, int]:
        return {"seasonal_period": self.seasonal_period}
