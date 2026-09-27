"""Fit the mean-reverting (Ornstein-Uhlenbeck on log price) model to index history.

Exact discretisation on daily steps: ``x_{t+1} = a + b x_t + e``, ``b = e^{-kappa dt}``,
``theta = a / (1 - b)``, ``sigma = sd(e) sqrt(2 kappa / (1 - b^2))``.

With only months of history, kappa is poorly identified (its standard error is often as large as
the estimate), so it is clipped and can be overridden from config. If no mean reversion is
detected (``b >= 1``) the fit falls back to a random walk (``kappa = 0``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from compute_options.market.vol import DAYS_PER_YEAR


@dataclass(frozen=True)
class OUParams:
    kappa: float  # per year
    theta: float  # long-run mean of ln price
    sigma: float  # annualised instantaneous vol
    n_obs: int

    @property
    def half_life_days(self) -> float:
        return float(np.log(2) / self.kappa * DAYS_PER_YEAR) if self.kappa > 0 else float("inf")


def fit_ou(prices: pd.Series, kappa_max: float = 50.0) -> OUParams:
    """Fit on consecutive-day pairs only; gaps are skipped rather than interpolated."""
    p = prices.dropna().sort_index()
    x = np.log(p)
    prev = x.shift(1)
    step = p.index.to_series().diff().dt.days
    mask = step.eq(1)
    xt, xt1 = prev[mask].to_numpy(), x[mask].to_numpy()
    if len(xt) < 10:
        raise ValueError(f"need at least 10 consecutive-day observations, got {len(xt)}")

    dt = 1.0 / DAYS_PER_YEAR
    b, a = np.polyfit(xt, xt1, 1)
    resid = xt1 - (a + b * xt)
    sd = float(np.std(resid, ddof=2))
    if b >= 1.0 or b <= 0.0:
        return OUParams(0.0, float(x.iloc[-1]), sd / np.sqrt(dt), len(xt))
    theta = float(a / (1 - b))  # from the fitted b, before any clipping of kappa
    kappa = min(-np.log(b) / dt, kappa_max)
    b = np.exp(-kappa * dt)
    sigma = sd * np.sqrt(2 * kappa / (1 - b**2))
    return OUParams(float(kappa), theta, float(sigma), len(xt))


@dataclass(frozen=True)
class TwoFactorParams:
    """``ln S_t = L_t + N_t``: ``L`` a random walk (vol ``sigma_long``), the persistent level;
    ``N`` an OU process (speed ``kappa``, vol ``sigma_short``), transient noise such as
    day-to-day variation in which trades print. The one-factor fit on a noisy index locks onto
    ``N`` and implies prices never drift; this keeps both.
    """

    sigma_long: float
    kappa: float
    sigma_short: float
    n_obs: int

    @property
    def noise_sd(self) -> float:
        """Stationary standard deviation of the transient component (log units)."""
        return self.sigma_short / np.sqrt(2 * self.kappa) if self.kappa > 0 else 0.0


def fit_two_factor(prices: pd.Series, max_horizon_days: int = 30) -> TwoFactorParams:
    """Fit the variance of overlapping k-day log returns, k = 1..``max_horizon_days``.

    With ``b = e^{-kappa dt}`` and stationary noise variance ``s^2`` the model implies

        ``Var(k-day return) = sigma_L^2 k dt + 2 s^2 (1 - b^k)``

    i.e. variance grows linearly (the persistent level) plus a hump that saturates within a few
    half-lives (the noise). Fitting the whole curve in log space is far more stable on short
    samples than lag-1 autocorrelation or an AR(1) regression on levels, which overstates mean
    reversion badly with a few months of data.
    """
    from scipy.optimize import least_squares

    p = prices.dropna().sort_index()
    daily = np.log(p.resample("D").last().ffill())
    dt = 1.0 / DAYS_PER_YEAR
    ks = np.arange(1, max_horizon_days + 1)
    emp = np.array([(daily - daily.shift(k)).dropna().var(ddof=1) for k in ks])
    n = len(daily) - 1

    def model(x: np.ndarray) -> np.ndarray:
        var_long, s2, kappa = np.exp(x)
        return var_long * ks * dt + 2 * s2 * (1 - np.exp(-kappa * dt * ks))

    v1 = emp[0] / dt
    x0 = np.log([0.5 * v1, 0.25 * emp[0], 50.0])
    # Longer horizons have fewer independent observations: weight by sqrt(n / k).
    w = np.sqrt(n / ks)
    fit = least_squares(
        lambda x: w * (np.log(model(x)) - np.log(emp)),
        x0,
        bounds=(np.log([1e-6, 1e-10, 1.0]), np.log([25.0, 1.0, 5000.0])),
    )
    var_long, s2, kappa = np.exp(fit.x)
    return TwoFactorParams(
        float(np.sqrt(var_long)), float(kappa), float(np.sqrt(2 * kappa * s2)), n
    )
