"""Path simulation for the index under the one-factor mean-reverting model.

``S_t = F(0, t) * exp(X_t - Var[X_t] / 2)`` with ``dX = -kappa X dt + sigma dW``, ``X_0 = 0``.
This reproduces the forward curve exactly (``E[S_t] = F(0, t)``) and the terminal log-variance
used by ``MeanRevertingVol``, so Monte Carlo and Black-76 prices agree for European payoffs.
``kappa = 0`` gives geometric Brownian motion around the forward curve.
"""

from __future__ import annotations

import numpy as np

from compute_options.market.forward_curve import ForwardCurve


def ou_variance(t: np.ndarray, sigma: float, kappa: float) -> np.ndarray:
    t = np.asarray(t, float)
    if kappa < 1e-8:
        return sigma**2 * t
    return sigma**2 * -np.expm1(-2 * kappa * t) / (2 * kappa)


def simulate_paths(
    forward_curve: ForwardCurve,
    sigma: float,
    kappa: float,
    times: np.ndarray,
    n_paths: int,
    seed: int | None = None,
    antithetic: bool = True,
) -> np.ndarray:
    """Simulated index levels, shape ``(n_paths, len(times))``; ``times`` in years, increasing."""
    rng = np.random.default_rng(seed)
    times = np.asarray(times, float)
    dts = np.diff(np.concatenate([[0.0], times]))
    half = (n_paths + 1) // 2 if antithetic else n_paths
    z = rng.standard_normal((half, len(times)))
    if antithetic:
        z = np.concatenate([z, -z])[:n_paths]

    x = np.zeros((n_paths, len(times)))
    prev = np.zeros(n_paths)
    for i, dt in enumerate(dts):
        decay = np.exp(-kappa * dt)
        step_sd = np.sqrt(ou_variance(np.array(dt), sigma, kappa))
        prev = prev * decay + step_sd * z[:, i]
        x[:, i] = prev

    fwd = np.array([forward_curve.forward(t) for t in times])
    return fwd * np.exp(x - 0.5 * ou_variance(times, sigma, kappa))
