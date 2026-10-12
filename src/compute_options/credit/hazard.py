"""Hazard-rate (reduced-form) credit curves, bootstrapped from CDS spreads.

Default is the first jump of a Poisson process with piecewise-constant intensity. Survival
``Q(t) = exp(-integral of h)``. CDS spreads at several tenors pin the intensity on each
interval: the bootstrap solves interval by interval so every quoted CDS reprices at par.

The CDS legs use a quarterly premium grid and ignore accrued premium on default, the usual
simplification when the point is a hazard curve, not a trade-level mark.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class HazardCurve:
    tenors: tuple[float, ...]  # right ends of intervals, years, increasing
    hazards: tuple[float, ...]  # intensity on each interval (flat beyond the last)

    @classmethod
    def flat(cls, h: float) -> HazardCurve:
        return cls((1.0,), (h,))

    def cum_hazard(self, t: float | np.ndarray) -> np.ndarray:
        t = np.asarray(t, float)
        edges = np.concatenate([[0.0], self.tenors])
        h = np.concatenate([self.hazards, [self.hazards[-1]]])
        edges = np.concatenate([edges, [np.inf]])
        out = np.zeros_like(t)
        for lo, hi, hz in zip(edges[:-1], edges[1:], h, strict=True):
            out = out + hz * np.clip(np.minimum(t, hi) - lo, 0.0, None)
        return out

    def survival(self, t: float | np.ndarray) -> np.ndarray:
        return np.exp(-self.cum_hazard(t))

    def default_prob(self, t0: float, t1: float) -> float:
        return float(self.survival(t0) - self.survival(t1))

    def credit_spread(self, T: float, recovery: float) -> float:
        """Average spread over [0, T] implied by the curve: (1 - R) * average hazard."""
        return float((1.0 - recovery) * self.cum_hazard(T) / T)

    @classmethod
    def from_cds(
        cls,
        spreads: dict[float, float],
        recovery: float,
        df: Callable[[float], float],
        freq: int = 4,
    ) -> HazardCurve:
        tenors = sorted(spreads)
        hazards: list[float] = []

        def legs(h_new: float, k: int) -> tuple[float, float]:
            curve = cls(tuple(tenors[: k + 1]), tuple([*hazards, h_new]))
            grid = np.arange(1, int(round(tenors[k] * freq)) + 1) / freq
            q = curve.survival(np.concatenate([[0.0], grid]))
            dfs = np.array([df(t) for t in grid])
            rpv01 = float(np.sum(dfs * (1.0 / freq) * q[1:]))
            prot = (1.0 - recovery) * float(np.sum(dfs * (q[:-1] - q[1:])))
            return rpv01, prot

        for k, T in enumerate(tenors):

            def err(h: float, k: int = k, T: float = T) -> float:
                rpv01, prot = legs(h, k)
                return spreads[T] * rpv01 - prot

            hazards.append(brentq(err, 1e-8, 5.0))
        return cls(tuple(tenors), tuple(hazards))


def cds_par_spread(
    curve: HazardCurve, T: float, recovery: float, df: Callable[[float], float], freq: int = 4
) -> float:
    grid = np.arange(1, int(round(T * freq)) + 1) / freq
    q = curve.survival(np.concatenate([[0.0], grid]))
    dfs = np.array([df(t) for t in grid])
    rpv01 = float(np.sum(dfs * (1.0 / freq) * q[1:]))
    prot = (1.0 - recovery) * float(np.sum(dfs * (q[:-1] - q[1:])))
    return prot / rpv01
