"""Structural forward curve: log price reverts to a cost-of-supply anchor that drifts down.

    d ln S = kappa (ln A_t - ln S) dt + sigma dW,    ln A_t = ln A_0 - delta * t

``A_0`` is the levelised cost plus margin (``cost_model.anchor_price``). ``delta`` is the
annual decline of the long-run level as newer GPU generations arrive. ``kappa`` is how fast a
scarcity premium (or discount) to the anchor decays.

Solving the SDE gives the conditional mean in closed form:

    E[ln S_t] = e^{-kt} ln S_0 + (1 - e^{-kt}) ln A_0 - delta (t - (1 - e^{-kt}) / k)

The forward is ``E[S_t]`` times ``exp(-risk_premium * t)``. There is no cash-and-carry link in a
market with a non-storable underlying, so the risk premium is an explicit assumption.

The class holds *arrays* of parameters with weights, i.e. an ensemble. Its ``forward`` is the
weighted mean over members, which is also what a Bayesian posterior mean is. A one-member
ensemble is an ordinary curve. Because it subclasses ``ForwardCurve``, ``MarketState`` and every
structured product price off it with no change.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from compute_options.market.forward_curve import ForwardCurve
from compute_options.structural.assumptions import Params
from compute_options.structural.cost_model import anchor_price


def cond_log_mean(
    x0: np.ndarray | float,
    dt: float,
    log_anchor_start: np.ndarray | float,
    kappa: np.ndarray | float,
    delta: np.ndarray | float,
) -> np.ndarray:
    """Mean of ln S after ``dt`` years given ``ln S = x0`` and the anchor at the interval start."""
    k = np.maximum(kappa, 1e-8)
    d = np.exp(-k * dt)
    return d * x0 + (1.0 - d) * log_anchor_start - delta * (dt - (1.0 - d) / k)


def cond_log_var(dt: float, sigma: np.ndarray | float, kappa: np.ndarray | float) -> np.ndarray:
    k = np.maximum(kappa, 1e-8)
    return sigma**2 * (1.0 - np.exp(-2.0 * k * dt)) / (2.0 * k)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, qs: list[float]) -> np.ndarray:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(qs, cum, v)


@dataclass(frozen=True, eq=False)
class StructuralForward(ForwardCurve):
    spot: float
    anchor: np.ndarray  # A_0 per member
    kappa: np.ndarray
    delta: np.ndarray
    sigma: np.ndarray
    risk_premium: np.ndarray
    weights: np.ndarray
    scale: float = 1.0

    @classmethod
    def from_params(
        cls, spot: float, p: Params, sigma: float | np.ndarray, weights: np.ndarray | None = None
    ) -> StructuralForward:
        anchor = np.atleast_1d(np.asarray(anchor_price(p), float))
        n = len(anchor)

        def col(x) -> np.ndarray:
            return np.broadcast_to(np.atleast_1d(np.asarray(x, float)), (n,)).copy()

        w = np.ones(n) / n if weights is None else np.asarray(weights, float) / np.sum(weights)
        return cls(
            spot=float(spot),
            anchor=anchor,
            kappa=col(p["kappa"]),
            delta=col(p["delta"]),
            sigma=col(sigma),
            risk_premium=col(p["risk_premium"]),
            weights=w,
        )

    # --- per-member quantities -------------------------------------------------------------
    def expected_spot_members(self, t: float) -> np.ndarray:
        """``E[S_t]`` per member, physical measure, before the scale factor."""
        if t <= 0:
            return np.full(len(self.anchor), self.spot)
        mean = cond_log_mean(np.log(self.spot), t, np.log(self.anchor), self.kappa, self.delta)
        return np.exp(mean + 0.5 * cond_log_var(t, self.sigma, self.kappa))

    def forward_members(self, t: float) -> np.ndarray:
        return self.expected_spot_members(t) * np.exp(-self.risk_premium * max(t, 0.0))

    # --- ForwardCurve interface ------------------------------------------------------------
    def forward(self, t: float) -> float:
        return float(self.weights @ self.forward_members(t)) * self.scale

    def scaled(self, factor: float) -> StructuralForward:
        return replace(self, spot=self.spot * factor, anchor=self.anchor * factor)

    # --- uncertainty -----------------------------------------------------------------------
    def forward_quantiles(self, t: float, qs: list[float]) -> np.ndarray:
        """Spread of the forward across members: parameter uncertainty only."""
        return weighted_quantile(self.forward_members(t), self.weights, qs) * self.scale

    def spot_quantiles(
        self, t: float, qs: list[float], n_draws: int = 20_000, seed: int = 0
    ) -> np.ndarray:
        """Quantiles of the spot at ``t`` including diffusion: parameter plus price uncertainty."""
        if t <= 0:
            return np.full(len(qs), self.spot * self.scale)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.anchor), size=n_draws, p=self.weights)
        mean = cond_log_mean(
            np.log(self.spot), t, np.log(self.anchor[idx]), self.kappa[idx], self.delta[idx]
        )
        sd = np.sqrt(cond_log_var(t, self.sigma[idx], self.kappa[idx]))
        samples = np.exp(mean + sd * rng.standard_normal(n_draws))
        return np.quantile(samples, qs) * self.scale

    def point(self) -> StructuralForward:
        """Collapse to the weighted-mean parameters (a single member)."""
        w = self.weights

        def m(x: np.ndarray) -> np.ndarray:
            return np.array([w @ x])

        return replace(
            self,
            anchor=np.array([np.exp(w @ np.log(self.anchor))]),
            kappa=m(self.kappa),
            delta=m(self.delta),
            sigma=m(self.sigma),
            risk_premium=m(self.risk_premium),
            weights=np.ones(1),
        )

    def effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self.weights**2))
