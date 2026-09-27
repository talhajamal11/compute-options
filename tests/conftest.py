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
