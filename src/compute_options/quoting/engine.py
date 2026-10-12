"""Reservation bid and offer for a bilateral compute request.

A request arrives with no market to reference: no listed futures, a spot index that is a noisy
composite, a counterparty with its own credit risk. The quote is built up from the risks being
carried, and each piece is reported separately:

    offer = mid + model uncertainty + risk capital + credit + minimum edge     (client buys)
    bid   = mid - (the same four charges, computed for the opposite position)  (client sells)

- **mid**: delivery-weighted average of the structural forward curve (posterior mean).
- **model uncertainty**: a multiple of the standard deviation of the strip price across the
  posterior ensemble. It pays for not knowing the right parameters.
- **risk capital**: a hurdle rate on the 99% value-at-risk of holding the position unhedged to
  delivery. Without listed futures there is nothing to hedge with, so the whole position is held.
- **credit**: CVA on the position with the counterparty's hazard curve and wrong-way risk.
- **minimum edge**: a floor, as a fraction of mid, for operational and funding cost.

``HURDLE``, ``MODEL_RISK_MULT`` and ``MIN_EDGE`` are policy settings, not estimated; they are inputs
to the governance pack, not outputs of the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from compute_options.credit import (
    Counterparty,
    CreditMarket,
    ForwardStrip,
    make_counterparty,
    strip_cva,
)
from compute_options.credit.exposure import monthly_grid, simulate_monthly
from compute_options.structural.curve import StructuralForward, weighted_quantile

HURDLE = 0.12  # required annual return on risk capital
VAR_LEVEL = 0.99
MODEL_RISK_MULT = 0.5  # std devs of ensemble strip price charged
MIN_EDGE = 0.010  # fraction of mid

Structure = Literal["forward", "take_or_pay", "prepaid"]


@dataclass(frozen=True)
class BilateralRequest:
    gpu: str
    gpu_hours_per_month: float
    months: int  # last delivery month
    counterparty: str
    start_month: int = 0
    wrong_way: float = 0.0
    note: str = ""

    @property
    def total_hours(self) -> float:
        return self.gpu_hours_per_month * (self.months - self.start_month)


@dataclass(frozen=True)
class SideQuote:
    side: str  # "bid" (we buy compute) or "offer" (we sell compute)
    price: float
    mid: float
    model_uncertainty: float
    risk_capital: float
    credit: float
    edge: float

    @property
    def components(self) -> dict[str, float]:
        return {
            "mid": self.mid,
            "model uncertainty": self.model_uncertainty,
            "risk capital": self.risk_capital,
            "credit (CVA)": self.credit,
            "minimum edge": self.edge,
        }


@dataclass(frozen=True)
class Quote:
    request: BilateralRequest
    bid: SideQuote
    offer: SideQuote
    mid: float
    mid_range: tuple[float, float]  # 5-95% across posterior members

    @property
    def spread(self) -> float:
        return self.offer.price - self.bid.price

    def table(self) -> pd.DataFrame:
        rows = {k: [self.bid.components[k], self.offer.components[k]] for k in self.bid.components}
        df = pd.DataFrame(rows, index=["bid", "offer"]).T
        df.loc["quote"] = [self.bid.price, self.offer.price]
        return df

    def memo(self) -> str:
        r = self.request
        b, o = self.bid, self.offer
        return (
            f"**{r.gpu}, {r.gpu_hours_per_month:,.0f} GPU-hrs/month, months "
            f"{r.start_month + 1}-{r.months}, counterparty: {r.counterparty}.** "
            f"Two-way ${b.price:.3f} / ${o.price:.3f} per GPU-hr around a structural mid of "
            f"${self.mid:.3f}. The posterior 5-95% range for the mid is "
            f"${self.mid_range[0]:.3f}-${self.mid_range[1]:.3f}, so we charge "
            f"${o.model_uncertainty:.3f} for model uncertainty. There is no listed contract to "
            f"hedge with, so we carry the position unhedged and charge ${o.risk_capital:.3f} "
            f"(offer) / ${b.risk_capital:.3f} (bid) for the capital it ties up. Credit costs "
            f"${o.credit:.3f} if we sell to them and ${b.credit:.3f} if we buy from them. "
            f"Total width ${self.spread:.3f} ({self.spread / self.mid:.1%} of mid)."
        )


def strip_price_members(
    curve: StructuralForward, discount, months: int, start: int
) -> tuple[np.ndarray, np.ndarray]:
    """Delivery-weighted average forward for each ensemble member, and the weights."""
    times = monthly_grid(months)[start:]
    dfs = np.array([discount.df(t) for t in times])
    f = np.stack([curve.forward_members(t) for t in times], axis=1) * curve.scale
    return (f * dfs).sum(axis=1) / dfs.sum(), curve.weights


def _risk_capital(strip: ForwardStrip, paths: np.ndarray, mkt: CreditMarket) -> float:
    """Annual hurdle on 99% VaR of the unhedged position's PV, per GPU-hour traded."""
    times = monthly_grid(strip.months)
    dfs = np.array([mkt.discount.df(t) for t in times])
    pv = strip.sign * ((paths - strip.strike) * (strip.weights() * dfs)[None, :]).sum(axis=1)
    var = float(np.mean(pv) - np.quantile(pv, 1.0 - VAR_LEVEL))
    avg_life = float(np.sum(times * strip.weights()) / strip.weights().sum())
    return HURDLE * var * avg_life / strip.total_hours


def quote(
    req: BilateralRequest,
    structure: Structure,
    curve: StructuralForward,
    sigma: float,
    kappa: float,
    discount,
    cpty: Counterparty | None = None,
    n_paths: int = 20_000,
    seed: int = 0,
) -> Quote:
    """Two-way price for a fixed-price strip. ``structure`` selects the credit treatment."""
    cpty = cpty or make_counterparty(req.counterparty, discount)
    mkt = CreditMarket(curve, sigma, kappa, discount.with_spread(0.0), n_paths, seed)
    members, w = strip_price_members(curve, discount, req.months, req.start_month)
    mid = float(w @ members)
    sd = float(np.sqrt(w @ (members - mid) ** 2))
    lo, hi = weighted_quantile(members, w, [0.05, 0.95])

    paths = simulate_monthly(curve, sigma, kappa, req.months, n_paths, seed)

    def side(name: str, sign: int) -> SideQuote:
        strip = ForwardStrip(req.gpu_hours_per_month, req.months, mid, sign, start=req.start_month)
        capital = _risk_capital(strip, paths, mkt)
        if structure == "prepaid":
            # Whoever pays upfront bears the credit risk. On the bid we buy compute and pay in
            # advance, so the counterparty's default strands undelivered hours; on the offer
            # the client pays us first and we carry none.
            credit = (
                0.0
                if sign == -1
                else strip_cva(
                    ForwardStrip(req.gpu_hours_per_month, req.months, 0.0, 1, req.start_month),
                    cpty,
                    mkt,
                    req.wrong_way,
                    paths,
                ).cva_per_hour
            )
        else:
            credit = strip_cva(strip, cpty, mkt, req.wrong_way, paths).cva_per_hour
        return SideQuote(
            side=name,
            price=mid
            + (1 if name == "offer" else -1)
            * (MODEL_RISK_MULT * sd + capital + credit + MIN_EDGE * mid),
            mid=mid,
            model_uncertainty=MODEL_RISK_MULT * sd,
            risk_capital=capital,
            credit=credit,
            edge=MIN_EDGE * mid,
        )

    return Quote(req, side("bid", +1), side("offer", -1), mid, (float(lo), float(hi)))
