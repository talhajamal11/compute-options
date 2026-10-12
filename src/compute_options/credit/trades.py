"""Credit-linked compute transactions.

Three trades, one engine (``exposure.expected_credit_loss``):

- ``ForwardStrip``: a strip of monthly fixed-price deliveries (a bilateral forward, take-or-pay
  offtake or prepaid capacity), with CVA from the counterparty's hazard curve and wrong-way risk.
- ``prepaid_discount``: pay upfront for future GPU-hours; the supplier's default strands the
  undelivered hours, so the fair prepay price carries a credit discount.
- ``GPULoan``: an amortising loan secured on GPUs. The collateral's value moves with the compute
  price, so default and recovery are linked and the fair spread depends on the correlation.
- ``issuer_spread``: ties the structured notes' funding spread to the issuer's hazard curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compute_options.credit.exposure import (
    conditional_expected_spot,
    expected_credit_loss,
    monthly_grid,
    simulate_monthly,
    standardise,
)
from compute_options.credit.hazard import HazardCurve
from compute_options.market.forward_curve import ForwardCurve
from compute_options.market.rates import DiscountCurve


@dataclass(frozen=True)
class Counterparty:
    name: str
    hazard: HazardCurve
    recovery: float = 0.40  # standard senior unsecured assumption

    def spread(self, T: float) -> float:
        return self.hazard.credit_spread(T, self.recovery)


@dataclass(frozen=True)
class CreditMarket:
    """Everything a credit-linked valuation needs besides the counterparty."""

    forward_curve: ForwardCurve
    sigma: float
    kappa: float
    discount: DiscountCurve  # risk-free (spread = 0)
    n_paths: int = 20_000
    seed: int = 0


# Illustrative CDS term structures (bp). Only the neocloud 5y level is sourced (CoreWeave about
# 855 bp, Aug 2026); the shape and the other two names are judgment, labelled as such in the README.
CDS_ARCHETYPES: dict[str, dict[float, float]] = {
    "Hyperscaler (IG)": {1: 0.0030, 3: 0.0050, 5: 0.0065},
    "Neocloud (HY)": {1: 0.0600, 3: 0.0780, 5: 0.0855},
    "AI lab (unrated)": {1: 0.1100, 3: 0.1400, 5: 0.1500},
}


def make_counterparty(name: str, discount: DiscountCurve, recovery: float = 0.40) -> Counterparty:
    rf = discount.with_spread(0.0)
    return Counterparty(name, HazardCurve.from_cds(CDS_ARCHETYPES[name], recovery, rf.df), recovery)


@dataclass(frozen=True)
class CVAResult:
    cva: float  # $
    cva_per_hour: float  # $ per GPU-hour traded
    expected_exposure: np.ndarray
    pfe95: np.ndarray
    times: np.ndarray
    risk_free_value: float  # PV of the strip, no credit adjustment, $


@dataclass(frozen=True)
class ForwardStrip:
    """``qty`` GPU-hours per month for ``months`` months at fixed price ``strike`` ($/GPU-hr).

    ``sign = +1``: we receive the GPUs' market value and pay ``strike`` (we are long compute).
    ``sign = -1``: we deliver compute and receive ``strike`` (take-or-pay offtaker pays us).
    """

    qty: float
    months: int  # last delivery month
    strike: float
    sign: int = 1
    start: int = 0  # first delivery is month ``start + 1``

    @property
    def total_hours(self) -> float:
        return self.qty * (self.months - self.start)

    def weights(self) -> np.ndarray:
        """GPU-hours delivered in each monthly slot (zero before ``start``)."""
        return np.where(np.arange(self.months) >= self.start, self.qty, 0.0)


def strip_cva(
    strip: ForwardStrip,
    cpty: Counterparty,
    mkt: CreditMarket,
    wrong_way: float = 0.0,
    paths: np.ndarray | None = None,
) -> CVAResult:
    """CVA on one netting set. ``wrong_way`` > 0: default more likely when our exposure is high."""
    times = monthly_grid(strip.months)
    if paths is None:
        paths = simulate_monthly(
            mkt.forward_curve, mkt.sigma, mkt.kappa, strip.months, mkt.n_paths, mkt.seed
        )
    dfs = np.array([mkt.discount.df(t) for t in times])
    cond = conditional_expected_spot(paths, mkt.forward_curve, times, mkt.sigma, mkt.kappa)
    # Value at t_i of payments j >= i, discounted back to t_i.
    ratio = dfs[None, :] / dfs[:, None]
    # cond is zero below the diagonal, but (0 - K) is not: mask explicitly.
    mask = np.triu(np.ones((strip.months, strip.months)))
    value = strip.sign * np.einsum(
        "pij,ij,j->pi", (cond - strip.strike) * mask, ratio, strip.weights()
    )
    exposure = np.maximum(value, 0.0)

    z = standardise(strip.sign * np.log(paths))  # high z = high price if long, low if short
    loss = (1.0 - cpty.recovery) * exposure
    cva = expected_credit_loss(loss, z, cpty.hazard, times, dfs, wrong_way)
    fwd = np.array([mkt.forward_curve.forward(t) for t in times])
    pv0 = float(strip.sign * np.sum(strip.weights() * (fwd - strip.strike) * dfs))
    return CVAResult(
        cva,
        cva / strip.total_hours,
        exposure.mean(axis=0),
        np.quantile(exposure, 0.95, axis=0),
        times,
        pv0,
    )


def prepaid_discount(
    qty: float,
    months: int,
    cpty: Counterparty,
    mkt: CreditMarket,
    wrong_way: float = 0.0,
) -> dict[str, float]:
    """Fair price to prepay for ``qty`` GPU-hours a month, versus the risk-free price.

    On default the buyer loses (1 - R) of the market value of undelivered hours (it must buy
    them again). That is exactly a strip with strike 0 and ``sign = +1``.
    """
    strip = ForwardStrip(qty, months, strike=0.0, sign=1)
    res = strip_cva(strip, cpty, mkt, wrong_way)
    rf = res.risk_free_value
    return {
        "risk_free_price": rf / strip.total_hours,
        "credit_discount": res.cva_per_hour,
        "fair_prepay_price": (rf - res.cva) / strip.total_hours,
        "discount_pct": res.cva / rf,
    }


@dataclass(frozen=True)
class GPULoan:
    """Amortising loan secured on GPUs, valued per $1 of GPU purchase price.

    ``ltv``: loan / initial collateral value. Collateral value follows straight-line
    depreciation to ``resid`` over ``life`` years, times the compute price relative to today
    raised to ``beta`` (1 = collateral value moves one-for-one with the price). On default the
    lender sells the collateral at a ``haircut``.
    """

    ltv: float = 0.70
    months: int = 36
    life: float = 5.0
    resid: float = 0.10
    haircut: float = 0.20
    beta: float = 1.0


def loan_fair_spread(
    loan: GPULoan,
    cpty: Counterparty,
    mkt: CreditMarket,
    wrong_way: float = 0.0,
    paths: np.ndarray | None = None,
) -> dict[str, float]:
    """Spread over risk-free that makes the loan's expected credit loss zero (per annum)."""
    m = loan.months
    times = monthly_grid(m)
    if paths is None:
        paths = simulate_monthly(mkt.forward_curve, mkt.sigma, mkt.kappa, m, mkt.n_paths, mkt.seed)
    dfs = np.array([mkt.discount.df(t) for t in times])
    spot = mkt.forward_curve.forward(0.0)

    balance_start = loan.ltv * (1.0 - np.arange(m) / m)  # outstanding during interval i
    dep = np.maximum(1.0 - (1.0 - loan.resid) * times / loan.life, loan.resid)
    collateral = dep[None, :] * (paths / spot) ** loan.beta
    loss = np.maximum(balance_start[None, :] - (1.0 - loan.haircut) * collateral, 0.0)

    # Collateral value low -> loss high: driver is minus the price.
    z = standardise(-np.log(paths))
    el = expected_credit_loss(loss, z, cpty.hazard, times, dfs, wrong_way)
    surv = cpty.hazard.survival(np.concatenate([[0.0], times[:-1]]))
    annuity = float(np.sum(balance_start * (1.0 / 12.0) * dfs * surv))
    return {
        "expected_loss_pv": el,
        "spread": el / annuity,
        "indep_spread_flat": cpty.spread(m / 12.0),
    }


def issuer_spread(cpty: Counterparty, tenor: float) -> float:
    """Funding spread over risk-free for a note of ``tenor`` years issued by ``cpty``."""
    return cpty.spread(tenor)
