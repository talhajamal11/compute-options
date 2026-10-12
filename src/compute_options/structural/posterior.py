"""Update the prior ensemble with the observed index: importance-weighted Bayesian posterior.

The prior is the assumptions register (triangular draws). The likelihood is the exact Gaussian
transition density of the structural model on weekly observations (weekly steps avoid the
day-to-day noise and stickiness of posted prices). Each prior draw gets a weight proportional to
its likelihood.

With a few months of data the exact likelihood is far more confident than the data warrant (the
model is misspecified: posted prices move in steps), and the weights collapse onto a handful of
draws. The likelihood is therefore tempered (raised to a power ``beta <= 1``), with ``beta``
chosen so the effective sample size stays above a floor. The result is a posterior that moves
towards the data but stays cautious about how little they pin down. ``beta`` and the effective
sample size are reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from compute_options.market.vol import realized_vol
from compute_options.structural.assumptions import Params, sample_params
from compute_options.structural.curve import StructuralForward, cond_log_mean, cond_log_var


@dataclass(frozen=True)
class Posterior:
    gpu: str
    prior: StructuralForward
    curve: StructuralForward  # same members, posterior weights
    params: Params
    sigma_hat: float  # realised weekly vol of the history
    beta: float  # likelihood temperature
    ess: float
    n_steps: int

    def summary(self) -> pd.DataFrame:
        rows = []
        for key in ("kappa", "delta", "margin", "risk_premium"):
            v = np.asarray(self.params[key], float)
            for label, w in (("prior", self.prior.weights), ("posterior", self.curve.weights)):
                mean = float(w @ v)
                sd = float(np.sqrt(w @ (v - mean) ** 2))
                rows.append({"parameter": key, "distribution": label, "mean": mean, "sd": sd})
        return pd.DataFrame(rows)


def _weekly_log_series(history: pd.Series, step_days: int) -> tuple[np.ndarray, np.ndarray]:
    daily = history.dropna().sort_index().resample("D").last().ffill()
    pick = daily.iloc[::-1].iloc[::step_days].iloc[::-1]
    t = (pick.index - pick.index[-1]).days.to_numpy() / 365.0  # years relative to now (<= 0)
    return np.log(pick.to_numpy()), t


def _log_likelihood(
    x: np.ndarray, t: np.ndarray, anchor0: np.ndarray, kappa, delta, sigma
) -> np.ndarray:
    ll = np.zeros(len(anchor0))
    for k in range(len(x) - 1):
        dt = t[k + 1] - t[k]
        log_anchor_start = np.log(anchor0) - delta * t[k]
        mean = cond_log_mean(x[k], dt, log_anchor_start, kappa, delta)
        var = cond_log_var(dt, sigma, kappa)
        ll += -0.5 * (np.log(2 * np.pi * var) + (x[k + 1] - mean) ** 2 / var)
    return ll


def _temper(ll: np.ndarray, min_ess: float) -> tuple[np.ndarray, float]:
    ll = ll - ll.max()

    def ess_gap(beta: float) -> float:
        w = np.exp(beta * ll)
        w /= w.sum()
        return 1.0 / np.sum(w**2) - min_ess

    beta = 1.0 if ess_gap(1.0) >= 0 else brentq(ess_gap, 0.0, 1.0)
    w = np.exp(beta * ll)
    return w / w.sum(), float(beta)


def fit_posterior(
    history: pd.Series,
    gpu: str,
    n: int = 4000,
    seed: int = 0,
    min_ess_frac: float = 0.15,
    step_days: int = 7,
    vol_window: int = 120,
) -> Posterior:
    hist = history.dropna().sort_index()
    x, t = _weekly_log_series(hist, step_days)
    if len(x) < 5:
        raise ValueError(f"need at least {4 * step_days + 1} days of history, got {len(hist)}")

    sigma_hat = realized_vol(hist, vol_window, horizon_days=7)
    params = sample_params(gpu, n, seed)
    sigma = np.asarray(params["sigma_mult"], float) * sigma_hat
    spot = float(hist.iloc[-1])

    prior = StructuralForward.from_params(spot, params, sigma)
    ll = _log_likelihood(x, t, prior.anchor, prior.kappa, prior.delta, prior.sigma)
    w, beta = _temper(ll, min_ess_frac * n)
    post = StructuralForward.from_params(spot, params, sigma, weights=w)
    return Posterior(
        gpu, prior, post, params, sigma_hat, beta, post.effective_sample_size(), len(x) - 1
    )
