"""Expected return of holding a note to maturity, under a stated view of the index.

Pricing uses the forward, so the price alone says nothing about what an investor expects to
earn. Two views:

- ``expected_return``: the index is lognormal around its forward, drifting at ``drift`` a year
  (0 = the forward is the best guess), with the pricing vol.
- ``historical_scenarios``: the payoff applied to every realised move of the same length in
  the index history.

Returns are per unit paid (``issue_price``); APR = ``(payoff / issue_price) ** (1 / T) - 1``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from compute_options.market.state import MarketState
from compute_options.products.base import StructuredProduct


def _summary(payoffs: np.ndarray, issue_price: float, T: float) -> dict[str, float]:
    gross = np.asarray(payoffs, float) / issue_price
    apr = np.where(gross > 0, np.power(np.maximum(gross, 1e-12), 1 / T) - 1, -1.0)
    return {
        "expected return (to maturity)": float(gross.mean() - 1),
        "expected APR": float(gross.mean() ** (1 / T) - 1),
        "median APR": float(np.median(apr)),
        "5th percentile APR": float(np.percentile(apr, 5)),
        "95th percentile APR": float(np.percentile(apr, 95)),
        "probability of loss": float((gross < 1).mean()),
        "probability of full protection": float((gross >= 1 - 1e-9).mean()),
    }


def expected_return(
    product: StructuredProduct,
    market: MarketState,
    initial_fixing: float,
    issue_price: float = 1.0,
    drift: float = 0.0,
    n_paths: int = 200_000,
    seed: int = 0,
) -> dict[str, float]:
    T = product.tenor
    fwd_r = market.forward(T) / initial_fixing * np.exp(drift * T)
    sd = market.implied_vol(T, 1.0) * np.sqrt(T)
    z = np.random.default_rng(seed).standard_normal(n_paths // 2)
    z = np.concatenate([z, -z])
    R = fwd_r * np.exp(sd * z - 0.5 * sd**2)
    return _summary(product.payoff(R), issue_price, T)


def historical_scenarios(
    product: StructuredProduct, series: pd.Series, issue_price: float = 1.0
) -> tuple[dict[str, float], pd.Series]:
    """Summary plus the realised ``R`` for each historical start date."""
    daily = series.resample("D").last().ffill()
    days = round(product.tenor * 365)
    R = (daily.shift(-days) / daily).dropna()
    if R.empty:
        return {}, R
    return _summary(product.payoff(R.to_numpy()), issue_price, product.tenor) | {
        "scenarios": float(len(R))
    }, R
