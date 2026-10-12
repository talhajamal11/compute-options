"""Build a ``MarketState`` from the structural forward curve."""

from __future__ import annotations

import numpy as np
import pandas as pd

from compute_options.market.rates import DiscountCurve
from compute_options.market.state import MarketState
from compute_options.market.vol import MeanRevertingVol
from compute_options.structural.posterior import Posterior


def posterior_vol(post: Posterior, floor: float = 0.10, cap: float = 2.0) -> MeanRevertingVol:
    c = post.curve
    sigma = float(np.clip(c.weights @ c.sigma, floor, cap))
    return MeanRevertingVol(sigma, float(c.weights @ c.kappa))


def structural_state(
    post: Posterior,
    date: pd.Timestamp,
    discount: DiscountCurve,
    member: int | None = None,
) -> MarketState:
    """Market on ``date``. ``member`` picks one posterior draw (for parameter-uncertainty runs);
    default uses the posterior-weighted ensemble forward and posterior-mean volatility."""
    if member is None:
        return MarketState(pd.Timestamp(date), post.curve, posterior_vol(post), discount)
    c = post.curve
    single = StructuralForward_one(c, member)
    vol = MeanRevertingVol(float(np.clip(c.sigma[member], 0.10, 2.0)), float(c.kappa[member]))
    return MarketState(pd.Timestamp(date), single, vol, discount)


def StructuralForward_one(c, i: int):  # noqa: N802 - small constructor-style helper
    from dataclasses import replace

    return replace(
        c,
        anchor=c.anchor[i : i + 1],
        kappa=c.kappa[i : i + 1],
        delta=c.delta[i : i + 1],
        sigma=c.sigma[i : i + 1],
        risk_premium=c.risk_premium[i : i + 1],
        weights=np.ones(1),
    )
