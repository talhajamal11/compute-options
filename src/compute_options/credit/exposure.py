"""Credit losses along simulated compute-price paths, with wrong-way risk.

Counterparty default is tied to the compute price through a one-factor Gaussian copula. In
interval ``i`` the default latent variable is ``A = -rho z + sqrt(1 - rho^2) eps`` where ``z`` is
the standardised exposure driver (high ``z`` = high exposure) and ``eps`` is independent. A
positive ``rho`` makes default more likely when exposure is high (wrong-way risk); negative is
right-way; zero reduces to the textbook independent CVA.

Conditioning on the price path and integrating out ``eps`` gives the probability of default in
each interval in closed form, so no default-time sampling is needed and the estimate has only
price-path noise.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from compute_options.credit.hazard import HazardCurve
from compute_options.market.forward_curve import ForwardCurve
from compute_options.models.processes import ou_variance, simulate_paths


def monthly_grid(months: int) -> np.ndarray:
    return np.arange(1, months + 1) / 12.0


def simulate_monthly(
    curve: ForwardCurve, sigma: float, kappa: float, months: int, n_paths: int, seed: int = 0
) -> np.ndarray:
    return simulate_paths(curve, sigma, kappa, monthly_grid(months), n_paths, seed)


def standardise(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0)
    return (x - x.mean(axis=0)) / np.where(sd < 1e-12, 1.0, sd)


def conditional_default_prob(
    z: np.ndarray, hazard: HazardCurve, times: np.ndarray, rho: float
) -> np.ndarray:
    """P(default in (t_{i-1}, t_i] | z_i), shape ``z.shape``."""
    rho = float(np.clip(rho, -0.99, 0.99))
    edges = np.concatenate([[0.0], times])
    pd_cum = 1.0 - hazard.survival(edges)
    c = norm.ppf(np.clip(pd_cum, 1e-12, 1 - 1e-12))
    c[0] = -np.inf
    s = np.sqrt(1.0 - rho**2)
    hi = norm.cdf((c[None, 1:] + rho * z) / s)
    lo = norm.cdf((c[None, :-1] + rho * z) / s)
    return hi - lo


def expected_credit_loss(
    loss: np.ndarray,
    z: np.ndarray,
    hazard: HazardCurve,
    times: np.ndarray,
    df: np.ndarray,
    rho: float,
) -> float:
    """PV of ``loss[p, i] * P(default in interval i | path p)``, averaged over paths."""
    cond = conditional_default_prob(z, hazard, times, rho)
    return float(np.mean(np.sum(loss * cond * df[None, :], axis=1)))


def conditional_expected_spot(
    paths: np.ndarray, curve: ForwardCurve, times: np.ndarray, sigma: float, kappa: float
) -> np.ndarray:
    """``E[S_j | S_i]`` for all i, j >= i; shape ``(n_paths, m, m)`` (zero where j < i).

    Exact under the repo's path model ``S_t = F(t) exp(x_t - v_t / 2)`` with OU ``x``.
    """
    m = len(times)
    fwd = np.array([curve.forward(t) for t in times])
    v = ou_variance(times, sigma, kappa)
    x = np.log(paths / fwd) + 0.5 * v
    out = np.zeros((paths.shape[0], m, m))
    for i in range(m):
        dt = times[i:] - times[i]
        phi = np.exp(-kappa * dt)
        out[:, i, i:] = fwd[i:] * np.exp(phi[None, :] * x[:, [i]] - 0.5 * (phi**2)[None, :] * v[i])
    return out
