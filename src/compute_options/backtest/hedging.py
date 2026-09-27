"""Issuer delta hedge with a futures contract on the index expiring at the note's maturity.

No such futures trade publicly yet, so the hedge is marked at the model forward, which is also
where the note is valued: the hedge error below measures gamma, vol-estimation and gap risk,
not basis to a real contract.

The hedge portfolio starts at the note's fair value, holds ``delta_notional`` futures (rebalanced
daily, costless to enter) and accrues cash at the issuer's rate. ``hedge_error = portfolio -
note value``: at maturity it is the issuer's P&L beyond the day-one margin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def delta_hedge(mtm: pd.DataFrame) -> pd.DataFrame:
    """Add ``hedge_portfolio`` and ``hedge_error`` columns to a mark-to-market frame."""
    out = mtm.copy()
    dt = 1.0 / 365.0
    port = np.empty(len(out))
    port[0] = out["value"].iloc[0]
    for i in range(1, len(out)):
        prev = out.iloc[i - 1]
        futures_pnl = prev["delta_notional"] * (out["forward_R"].iloc[i] - prev["forward_R"])
        port[i] = port[i - 1] * np.exp(prev["rate"] * dt) + futures_pnl
    out["hedge_portfolio"] = port
    out["hedge_error"] = out["hedge_portfolio"] - out["value"]
    return out
