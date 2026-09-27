"""Daily mark-to-market of an issued note, from issue to maturity (or the last data date)."""

from __future__ import annotations

import pandas as pd

from compute_options.backtest.issuance import IssuedNote
from compute_options.backtest.market_builder import MarketBuilder
from compute_options.market.state import MarketState
from compute_options.products import StructuredProduct

BUMP = 0.01


def value_and_delta(
    product: StructuredProduct, market: MarketState, s0: float, T: float, fwd_r: float
) -> tuple[float, float]:
    """Value and futures hedge ratio (``dV/dF_R``); cheaper than a full ``risk.greeks``."""
    v = product.price(market, s0, T)
    up = product.price(market.spot_scaled(1 + BUMP), s0, T)
    dn = product.price(market.spot_scaled(1 - BUMP), s0, T)
    return v, (up - dn) / (2 * BUMP * fwd_r)


def mark_to_market(note: IssuedNote, builder: MarketBuilder) -> pd.DataFrame:
    """One row per calendar day. Columns: T (years left), spot, R, forward_R (the maturity-matched
    futures price in R units), value, delta_notional, rate (issuer zero rate), matured."""
    last = builder.index.index[-1]
    end = min(note.maturity_date, last)
    rows = []
    for date in pd.date_range(note.issue_date, end, freq="D"):
        T = max((note.maturity_date - date).days / 365.0, 0.0)
        spot = builder.spot(date)
        R = spot / note.initial_fixing
        if T == 0.0:
            payoff = float(note.product.payoff(R))
            rows.append((date, T, spot, R, R, payoff, 0.0, 0.0, True))
            continue
        market = builder.state(date)
        fwd_r = market.forward(T) / note.initial_fixing
        value, delta = value_and_delta(note.product, market, note.initial_fixing, T, fwd_r)
        rows.append((date, T, spot, R, fwd_r, value, delta, market.discount.zero_rate(T), False))
    cols = ["date", "T", "spot", "R", "forward_R", "value", "delta_notional", "rate", "matured"]
    return pd.DataFrame(rows, columns=cols).set_index("date")
