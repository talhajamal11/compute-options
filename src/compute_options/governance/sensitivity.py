"""Which assumptions move the price? One-at-a-time ranges and rank correlation across the prior."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from compute_options.market.rates import DiscountCurve
from compute_options.quoting.engine import strip_price_members
from compute_options.structural.assumptions import central_params, register
from compute_options.structural.curve import StructuralForward
from compute_options.structural.posterior import Posterior


def _strip(spot: float, p, sigma: float, discount: DiscountCurve, months: int) -> float:
    curve = StructuralForward.from_params(spot, p, sigma)
    members, w = strip_price_members(curve, discount, months, 0)
    return float(w @ members)


def tornado(
    gpu: str, spot: float, sigma: float, discount: DiscountCurve, months: int = 12
) -> pd.DataFrame:
    """12-month strip price with each assumption moved to its low and high, others central.

    ``sigma_mult`` scales volatility; everything else is a register parameter.
    """
    base_p = central_params(gpu)
    base = _strip(spot, base_p, sigma, discount, months)
    rows = []
    for a in register(gpu):
        vals = []
        for v in (a.low, a.high):
            p = {**base_p, a.key: v}
            vals.append(_strip(spot, p, sigma * p["sigma_mult"], discount, months))
        rows.append(
            {
                "assumption": a.label,
                "key": a.key,
                "at low": vals[0],
                "at high": vals[1],
                "swing": abs(vals[1] - vals[0]),
                "basis": a.basis,
                "confidence": a.confidence,
            }
        )
    out = pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)
    out.attrs["base"] = base
    return out


def rank_importance(post: Posterior, discount: DiscountCurve, months: int = 12) -> pd.DataFrame:
    """Spearman correlation of each prior parameter with the 12-month strip price across draws."""
    members, _ = strip_price_members(post.prior, discount, months, 0)
    rows = []
    for a in register(post.gpu):
        x = np.asarray(post.params[a.key], float)
        rows.append({"key": a.key, "assumption": a.label, "rank corr": spearmanr(x, members)[0]})
    return pd.DataFrame(rows).sort_values("rank corr", key=np.abs, ascending=False)
