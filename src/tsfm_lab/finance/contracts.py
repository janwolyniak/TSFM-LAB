"""The agreed first finance contract, independent of terminal/model runtimes."""

import math
from dataclasses import dataclass
from numbers import Real

import pandas as pd

POLICY = "ohlc_next_open_to_target_close_v1"


class FinanceError(ValueError):
    """Financial inputs, timing, or capital accounting violate the contract."""


class FinanceReportError(FinanceError):
    """A saved derived report is malformed or internally inconsistent."""


@dataclass(frozen=True)
class BacktestRules:
    initial_capital: float
    threshold_bps: float
    fee_bps: float
    slippage_bps: float
    short_financing_bps: float
    signal_latency_seconds: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise FinanceError(f"{name} must be a finite real number")
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise FinanceError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.initial_capital <= 0:
            raise FinanceError("initial_capital must be positive")
        if self.fee_bps + self.slippage_bps >= 10000:
            raise FinanceError("Combined one-way fee/slippage must be below 10000 bps")
        try:
            _ = self.latency
        except (OverflowError, ValueError) as exc:
            raise FinanceError("signal latency must fit a nanosecond duration") from exc

    @property
    def latency(self) -> pd.Timedelta:
        return pd.Timedelta(seconds=self.signal_latency_seconds)
