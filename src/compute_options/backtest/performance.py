"""Annualised performance statistics for a daily NAV series (calendar days: compute trades
every day, so a year is 365 observations)."""

from __future__ import annotations

import numpy as np
import pandas as pd

DAYS = 365.0


def drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def performance_stats(nav: pd.Series, rf: float | pd.Series = 0.0) -> dict[str, float]:
    """``rf``: annual risk-free rate (continuous) for Sharpe and Sortino, either a constant or a
    daily series (each day's return is compared with the previous day's rate)."""
    nav = nav.dropna()
    years = (nav.index[-1] - nav.index[0]).days / DAYS
    total = nav.iloc[-1] / nav.iloc[0] - 1.0
    r = np.log(nav).diff().dropna()
    if isinstance(rf, pd.Series):
        rf = rf.reindex(nav.index).ffill().bfill().shift(1).reindex(r.index)
    excess = r - rf / DAYS
    vol = r.std(ddof=1) * np.sqrt(DAYS)
    risky = vol > 1e-3  # Sharpe of a near-riskless series is noise
    downside = np.sqrt((np.minimum(excess, 0.0) ** 2).mean()) * np.sqrt(DAYS)
    dd = drawdown(nav)
    underwater = dd < 0
    # Longest run of consecutive days below the previous peak.
    runs = underwater.groupby((~underwater).cumsum()).sum()
    apr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else np.nan
    return {
        "total return": total,
        "APR (annualised)": apr,
        "annualised vol": vol,
        "Sharpe": excess.mean() * DAYS / vol if risky else np.nan,
        "Sortino": excess.mean() * DAYS / downside if risky and downside > 0 else np.nan,
        "max drawdown": dd.min(),
        "max drawdown (days)": float(runs.max()) if len(runs) else 0.0,
        "Calmar": apr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "best day": r.max(),
        "worst day": r.min(),
        "years": years,
    }
