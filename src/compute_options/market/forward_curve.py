"""Forward curves for a GPU rental price index.

GPU-hours cannot be stored, so there is no cash-and-carry link between spot and forward: the
forward is an expectation of the future rental rate, not ``spot * exp(rT)``. Until exchange
futures (Ornn/ICE, Architect) publish settlement prices, forwards come from a model:

- ``FlatForward``: the forward equals today's index (martingale assumption).
- ``SchwartzForward``: log price mean-reverts to a long-run level (Schwartz 1997 one-factor).
  Captures rental rates drifting towards a cost-of-hardware anchor, and a falling long-run level
  captures depreciation as newer GPU generations arrive.
- ``InterpolatedForward``: built from quoted futures prices once available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np


class ForwardCurve(ABC):
    spot: float

    @abstractmethod
    def forward(self, t: float) -> float:
        """Forward price for delivery ``t`` years from the valuation date."""

    @abstractmethod
    def scaled(self, factor: float) -> ForwardCurve:
        """The whole curve multiplied by ``factor`` (used for delta and scenario bumps)."""


@dataclass(frozen=True)
class FlatForward(ForwardCurve):
    spot: float

    def forward(self, t: float) -> float:
        return self.spot

    def scaled(self, factor: float) -> FlatForward:
        return replace(self, spot=self.spot * factor)


@dataclass(frozen=True)
class SchwartzForward(ForwardCurve):
    """Forward under ``d ln S = kappa (theta - ln S) dt + sigma dW`` (risk-neutral parameters).

    ``F(t) = exp(e^{-kt} ln S + (1 - e^{-kt}) theta + sigma^2 (1 - e^{-2kt}) / (4k))``.
    """

    spot: float
    kappa: float
    theta: float  # long-run mean of ln S
    sigma: float
    scale: float = 1.0

    def forward(self, t: float) -> float:
        if t <= 0:
            return self.spot * self.scale
        k = max(self.kappa, 1e-8)
        decay = np.exp(-k * t)
        var_term = self.sigma**2 * (1 - np.exp(-2 * k * t)) / (4 * k)
        log_f = decay * np.log(self.spot) + (1 - decay) * self.theta + var_term
        return float(np.exp(log_f)) * self.scale

    def scaled(self, factor: float) -> SchwartzForward:
        return replace(self, scale=self.scale * factor)


@dataclass(frozen=True)
class InterpolatedForward(ForwardCurve):
    """Forwards quoted at ``tenors`` (years), log-linear interpolation, flat extrapolation."""

    spot: float
    tenors: tuple[float, ...]
    forwards: tuple[float, ...]

    def forward(self, t: float) -> float:
        ts = np.concatenate([[0.0], self.tenors])
        fs = np.log(np.concatenate([[self.spot], self.forwards]))
        return float(np.exp(np.interp(t, ts, fs)))

    def scaled(self, factor: float) -> InterpolatedForward:
        return replace(
            self, spot=self.spot * factor, forwards=tuple(f * factor for f in self.forwards)
        )
