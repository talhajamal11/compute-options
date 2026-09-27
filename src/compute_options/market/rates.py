"""Discount curves: risk-free zero rates plus an issuer credit spread.

The issuer's funding spread matters for structured notes: a wider spread makes the embedded
zero-coupon bond cheaper, leaving more budget for options (higher participation).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from compute_options.data.rates import SERIES


@dataclass(frozen=True)
class DiscountCurve:
    """Continuously compounded zero rates at ``tenors`` (years), linear interpolation."""

    tenors: tuple[float, ...]
    rates: tuple[float, ...]
    spread: float = 0.0

    @classmethod
    def flat(cls, rate: float, spread: float = 0.0) -> DiscountCurve:
        return cls((1.0,), (rate,), spread)

    @classmethod
    def from_fred_row(cls, row: pd.Series, spread: float = 0.0) -> DiscountCurve:
        """Build from one row of ``data.rates.fetch_rates`` (bond-equivalent yields)."""
        pts = sorted((SERIES[k], float(v)) for k, v in row.items() if k in SERIES and pd.notna(v))
        if not pts:
            raise ValueError("no rates available in row")
        tenors, yields = zip(*pts, strict=True)
        # Semi-annual bond-equivalent yield -> continuous compounding.
        cont = tuple(float(2 * np.log1p(y / 2)) for y in yields)
        return cls(tuple(tenors), cont, spread)

    def zero_rate(self, t: float) -> float:
        return float(np.interp(t, self.tenors, self.rates)) + self.spread

    def df(self, t: float) -> float:
        return float(np.exp(-self.zero_rate(t) * max(t, 0.0)))

    def shifted(self, dr: float) -> DiscountCurve:
        return replace(self, rates=tuple(r + dr for r in self.rates))

    def with_spread(self, spread: float) -> DiscountCurve:
        return replace(self, spread=spread)
