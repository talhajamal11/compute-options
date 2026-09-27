"""Volatility: historical estimators and the vol models used by the pricer.

No compute options trade with public quotes yet (Architect's options are pending regulatory
approval), so volatility is estimated from index history. ``GridVolSurface`` takes implied
vols directly once option quotes exist.

Compute trades every day, so annualisation uses 365 days.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.0


def log_returns(prices: pd.Series) -> pd.Series:
    """Log returns scaled to a one-day step, so gaps between observations don't inflate vol."""
    p = prices.dropna().sort_index()
    r = np.log(p).diff()
    gap_days = p.index.to_series().diff().dt.days
    return (r / np.sqrt(gap_days)).dropna()


def realized_vol(prices: pd.Series, window: int | None = None, horizon_days: int = 1) -> float:
    """Annualised vol of log returns over the last ``window`` days.

    ``horizon_days > 1`` uses overlapping multi-day returns. Daily compute indices carry
    transient noise that reverts within days (Ornn H100's fitted half-life is ~5 days), so
    daily returns overstate the vol relevant to a 3-12 month option; weekly returns filter it.
    Overlapping returns use every start day, so the estimate doesn't jump as the window rolls.
    """
    p = prices.dropna().sort_index()
    if window:
        p = p.loc[p.index[-1] - pd.Timedelta(days=window) :]
    if horizon_days > 1:
        daily = np.log(p.resample("D").last().ffill())
        r = (daily - daily.shift(horizon_days)).dropna()
        return float(r.std(ddof=1) * np.sqrt(DAYS_PER_YEAR / horizon_days))
    return float(log_returns(p).std(ddof=1) * np.sqrt(DAYS_PER_YEAR))


def ewma_vol(prices: pd.Series, lam: float = 0.94) -> float:
    """RiskMetrics EWMA volatility (annualised) as of the last observation."""
    r = log_returns(prices).to_numpy()
    weights = (1 - lam) * lam ** np.arange(len(r))[::-1]
    return float(np.sqrt(np.sum(weights * r**2) / weights.sum() * DAYS_PER_YEAR))


class VolModel(ABC):
    @abstractmethod
    def implied_vol(self, t: float, moneyness: float) -> float:
        """Black-76 vol for expiry ``t`` years and strike/forward ``moneyness``."""

    @abstractmethod
    def shifted(self, dv: float) -> VolModel:
        """Parallel shift by ``dv`` (absolute, e.g. 0.01 = 1 vol point)."""


@dataclass(frozen=True)
class FlatVol(VolModel):
    sigma: float

    def implied_vol(self, t: float, moneyness: float) -> float:
        return self.sigma

    def shifted(self, dv: float) -> FlatVol:
        return replace(self, sigma=self.sigma + dv)


@dataclass(frozen=True)
class MeanRevertingVol(VolModel):
    """Term vol when log price is Ornstein-Uhlenbeck with speed ``kappa``.

    ``sigma_eff(T)^2 = sigma^2 (1 - e^{-2 kappa T}) / (2 kappa T)``: long-dated options are
    cheaper than a flat-vol model implies, because shocks decay.
    """

    sigma: float
    kappa: float

    def implied_vol(self, t: float, moneyness: float) -> float:
        if t <= 0 or self.kappa < 1e-8:
            return self.sigma
        x = 2 * self.kappa * t
        return self.sigma * float(np.sqrt(-np.expm1(-x) / x))

    def shifted(self, dv: float) -> MeanRevertingVol:
        return replace(self, sigma=self.sigma + dv)


@dataclass(frozen=True)
class GridVolSurface(VolModel):
    """Implied vols on a (tenor x moneyness) grid, bilinear interpolation, flat extrapolation."""

    tenors: tuple[float, ...]
    moneyness: tuple[float, ...]
    vols: tuple[tuple[float, ...], ...]  # vols[i][j] for tenors[i], moneyness[j]
    shift: float = 0.0

    def implied_vol(self, t: float, moneyness: float) -> float:
        grid = np.asarray(self.vols)
        by_tenor = np.array([np.interp(moneyness, self.moneyness, row) for row in grid])
        return float(np.interp(t, self.tenors, by_tenor)) + self.shift

    def shifted(self, dv: float) -> GridVolSurface:
        return replace(self, shift=self.shift + dv)
