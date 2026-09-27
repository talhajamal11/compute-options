"""Revaluation grids for stress testing."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from compute_options.market.state import MarketState
from compute_options.products.base import StructuredProduct


def scenario_grid(
    product: StructuredProduct,
    market: MarketState,
    initial_fixing: float,
    spot_moves: Sequence[float] = (-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4),
    vol_shifts: Sequence[float] = (-0.1, 0.0, 0.1),
    T: float | None = None,
) -> pd.DataFrame:
    """Value for each (spot move, vol shift): rows are vol shifts, columns spot moves."""
    rows = {}
    for dv in vol_shifts:
        m_v = market.vol_shifted(dv)
        rows[dv] = {
            ds: product.price(m_v.spot_scaled(1 + ds), initial_fixing, T) for ds in spot_moves
        }
    out = pd.DataFrame(rows).T
    out.index.name, out.columns.name = "vol_shift", "spot_move"
    return out
