"""Adapter contract; no model runtime or terminal dependencies."""

from typing import Protocol

from tsfm_lab.contracts import (
    ForecastOutput,
    ForecastRequest,
    ForecastSettings,
    ModelMetadata,
)


class ForecastModel(Protocol):
    def metadata(self) -> ModelMetadata: ...

    def configuration(self) -> dict: ...

    def validate(self, spec: ForecastSettings) -> None: ...

    def load(self) -> None: ...

    def predict(
        self, requests: tuple[ForecastRequest, ...]
    ) -> tuple[ForecastOutput, ...]: ...

    def unload(self) -> None: ...
