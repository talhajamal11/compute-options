"""``MarketState``: everything needed to price on one valuation date."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from compute_options.market.forward_curve import ForwardCurve
from compute_options.market.rates import DiscountCurve
from compute_options.market.vol import VolModel


@dataclass(frozen=True)
class MarketState:
    valuation_date: pd.Timestamp
    forward_curve: ForwardCurve
    vol: VolModel
    discount: DiscountCurve

    @property
    def spot(self) -> float:
        return self.forward_curve.forward(0.0)

    def forward(self, t: float) -> float:
        return self.forward_curve.forward(t)

    def df(self, t: float) -> float:
        return self.discount.df(t)

    def implied_vol(self, t: float, moneyness: float) -> float:
        return self.vol.implied_vol(t, moneyness)

    # Bumped copies for risk.
    def spot_scaled(self, factor: float) -> MarketState:
        return replace(self, forward_curve=self.forward_curve.scaled(factor))

    def vol_shifted(self, dv: float) -> MarketState:
        return replace(self, vol=self.vol.shifted(dv))

    def rates_shifted(self, dr: float) -> MarketState:
        return replace(self, discount=self.discount.shifted(dr))
