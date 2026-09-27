"""Synthetic index histories from the fitted mean-reverting model.

Real compute price history is months long, too short to judge a 6-12 month note. Simulating
many histories from fitted dynamics gives a distribution of outcomes; ``run_backtest`` runs
unchanged on each. The two-factor model (persistent level + transient noise) is the default;
the one-factor OU model is kept for comparison. Simulations have zero drift in the level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from compute_options.backtest.engine import BacktestConfig, run_backtest
from compute_options.market.calibration import OUParams, TwoFactorParams
from compute_options.market.vol import DAYS_PER_YEAR


def simulate_index(
    start_level: float,
    params: OUParams,
    n_days: int,
    start_date: str = "2026-01-01",
    seed: int | None = None,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    dt = 1.0 / DAYS_PER_YEAR
    k, sd = params.kappa, params.sigma * np.sqrt(dt)
    if k > 0:
        b = np.exp(-k * dt)
        sd = params.sigma * np.sqrt((1 - b**2) / (2 * k))
    x = np.empty(n_days)
    x[0] = np.log(start_level)
    for i in range(1, n_days):
        mean = x[i - 1] if k == 0 else params.theta + (x[i - 1] - params.theta) * b
        x[i] = mean + sd * rng.standard_normal()
    return pd.Series(np.exp(x), index=pd.date_range(start_date, periods=n_days, freq="D"))


def simulate_two_factor(
    start_level: float,
    params: TwoFactorParams,
    n_days: int,
    start_date: str = "2026-01-01",
    seed: int | None = None,
) -> pd.Series:
    """Level random walk plus stationary OU noise, started with the noise at zero."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / DAYS_PER_YEAR
    level = np.log(start_level) + np.concatenate(
        [[0.0], np.cumsum(params.sigma_long * np.sqrt(dt) * rng.standard_normal(n_days - 1))]
    )
    noise = np.zeros(n_days)
    if params.kappa > 0:
        b = np.exp(-params.kappa * dt)
        sd = params.noise_sd * np.sqrt(1 - b**2)
        for i in range(1, n_days):
            noise[i] = noise[i - 1] * b + sd * rng.standard_normal()
    return pd.Series(
        np.exp(level + noise), index=pd.date_range(start_date, periods=n_days, freq="D")
    )


def run_synthetic(
    cfg: BacktestConfig,
    params: OUParams | TwoFactorParams,
    start_level: float,
    n_days: int = 730,
    n_runs: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Pooled note records across ``n_runs`` simulated histories (column ``run``)."""
    frames = []
    for run in range(n_runs):
        if isinstance(params, TwoFactorParams):
            series = simulate_two_factor(start_level, params, n_days, seed=seed + run)
        else:
            series = simulate_index(start_level, params, n_days, seed=seed + run)
        res = run_backtest(series, cfg)
        if len(res.notes):
            frames.append(res.notes.assign(run=run))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
