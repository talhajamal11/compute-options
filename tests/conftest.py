import numpy as np
import pandas as pd
import pytest

from compute_options.market import DiscountCurve, FlatForward, FlatVol, MarketState


@pytest.fixture
def market() -> MarketState:
    return MarketState(
        valuation_date=pd.Timestamp("2026-09-26"),
        forward_curve=FlatForward(2.5),
        vol=FlatVol(0.45),
        discount=DiscountCurve.flat(0.04, spread=0.01),
    )


@pytest.fixture
def R_grid() -> np.ndarray:
    # Include the kinks (strikes, barriers, caps) and points either side.
    return np.unique(np.concatenate([np.linspace(0.0, 3.0, 601), [0.8, 0.9, 1.0, 1.2, 1.5]]))


@pytest.fixture(scope="module")
def history() -> pd.Series:
    """Seeded daily index with mild noise, long enough for the posterior fit."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2026-03-01", periods=170, freq="D")
    level = 3.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.009, len(idx))))
    return pd.Series(level, index=idx, name="H100_SXM")


@pytest.fixture(scope="module")
def discount() -> DiscountCurve:
    return DiscountCurve.flat(0.04)


@pytest.fixture(scope="module")
def posterior(history):
    from compute_options.structural import fit_posterior

    return fit_posterior(history, "H100_SXM", n=1500, seed=0)
