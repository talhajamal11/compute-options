"""Black-76 on forwards: vanilla and cash-or-nothing digital options, with analytic greeks.

All functions are vectorised over numpy arrays. Greeks are with respect to the forward ``F``.
At ``T <= 0`` or ``sigma <= 0`` prices collapse to discounted intrinsic value.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm


def _d1d2(
    F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    F, K, T, sigma = map(np.asarray, (F, K, T, sigma))
    # A tiny floor turns T=0 or sigma=0 into +/-inf d's, i.e. intrinsic value.
    sd = np.maximum(sigma * np.sqrt(np.maximum(T, 0.0)), 1e-12)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    return d1, d1 - sd


def call(
    F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike, df: ArrayLike = 1.0
) -> np.ndarray:
    F, K = np.asarray(F, float), np.asarray(K, float)
    d1, d2 = _d1d2(F, K, T, sigma)
    return np.asarray(df) * (F * norm.cdf(d1) - K * norm.cdf(d2))


def put(
    F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike, df: ArrayLike = 1.0
) -> np.ndarray:
    F, K = np.asarray(F, float), np.asarray(K, float)
    d1, d2 = _d1d2(F, K, T, sigma)
    return np.asarray(df) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def digital_call(
    F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike, df: ArrayLike = 1.0
) -> np.ndarray:
    """Pays 1 if ``F_T > K``."""
    _, d2 = _d1d2(F, K, T, sigma)
    return np.asarray(df) * norm.cdf(d2)


def digital_put(
    F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike, df: ArrayLike = 1.0
) -> np.ndarray:
    """Pays 1 if ``F_T < K``."""
    _, d2 = _d1d2(F, K, T, sigma)
    return np.asarray(df) * norm.cdf(-d2)


def delta(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    sigma: ArrayLike,
    df: ArrayLike = 1.0,
    is_call: bool = True,
) -> np.ndarray:
    d1, _ = _d1d2(F, K, T, sigma)
    return np.asarray(df) * (norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0)


def gamma(
    F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike, df: ArrayLike = 1.0
) -> np.ndarray:
    d1, _ = _d1d2(F, K, T, sigma)
    sd = np.asarray(sigma) * np.sqrt(np.asarray(T))
    return np.asarray(df) * norm.pdf(d1) / (np.asarray(F) * sd)


def vega(
    F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike, df: ArrayLike = 1.0
) -> np.ndarray:
    """Per unit of vol (multiply by 0.01 for one vol point)."""
    d1, _ = _d1d2(F, K, T, sigma)
    return np.asarray(df) * np.asarray(F) * norm.pdf(d1) * np.sqrt(np.asarray(T))


def implied_vol(
    price: float, F: float, K: float, T: float, df: float = 1.0, is_call: bool = True
) -> float:
    """Invert Black-76 by Brent's method (for calibrating to option quotes when they exist)."""
    from scipy.optimize import brentq

    fn = call if is_call else put
    return float(brentq(lambda s: float(fn(F, K, T, s, df)) - price, 1e-6, 10.0))
