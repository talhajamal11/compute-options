import numpy as np
import pytest

from compute_options.market import MeanRevertingVol, SchwartzForward
from compute_options.models.processes import simulate_paths
from compute_options.products import (
    PRODUCTS,
    BufferedNote,
    CapitalProtectedNote,
    DualDirectionalNote,
    RiskReversalNote,
    TrackerNote,
    ZeroCouponNote,
    build_product,
    solve,
)

CASES = [
    ZeroCouponNote(1.0, redemption=1.05),
    CapitalProtectedNote(0.5, protection=0.95, strike=1.0, participation=0.8),
    CapitalProtectedNote(0.5, participation=1.2, cap=1.5),
    BufferedNote(0.25, buffer=0.1, participation=1.5, cap=1.2),
    BufferedNote(0.25, buffer=0.2, participation=1.0, downside_gearing=1 / 0.8),
    DualDirectionalNote(0.25, barrier=0.8, participation=1.0),
    DualDirectionalNote(0.25, barrier=0.7, participation=1.3, cap=1.25),
    RiskReversalNote(0.25, put_strike=0.9, call_strike=1.1),
    RiskReversalNote(0.25, put_strike=0.85, call_strike=1.2, participation=2.0, funded=False),
    TrackerNote(0.25, participation=0.95),
]
IDS = [f"{p.kind}-{i}" for i, p in enumerate(CASES)]


def test_every_product_is_covered():
    assert {p.kind for p in CASES} == set(PRODUCTS)


@pytest.mark.parametrize("product", CASES, ids=IDS)
def test_replication_matches_term_sheet(product, R_grid):
    """The legs must reproduce the term-sheet payoff everywhere (digitals flip at the strike
    itself, so exact kink points are excluded)."""
    kinks = {getattr(product, a) for a in ("barrier",) if hasattr(product, a)}
    R = R_grid[~np.isin(R_grid, list(kinks))]
    assert np.allclose(product.replicated_payoff(R), product.payoff(R), atol=1e-12)


@pytest.mark.parametrize("product", CASES, ids=IDS)
def test_monte_carlo_agrees_with_closed_form(product, market):
    fwd = SchwartzForward(2.5, kappa=2.0, theta=np.log(2.8), sigma=0.5)
    m = market.__class__(market.valuation_date, fwd, MeanRevertingVol(0.5, 2.0), market.discount)
    paths = simulate_paths(fwd, 0.5, 2.0, np.array([product.tenor]), 400_000, seed=1)
    mc = product.price_mc(paths, m.df(product.tenor), initial_fixing=2.5)
    assert product.price(m, 2.5) == pytest.approx(mc.price, abs=4 * mc.stderr + 1e-4)


@pytest.mark.parametrize("kind", sorted(PRODUCTS))
def test_solver_prices_to_target(kind, market):
    base = {"kind": kind, "tenor": 0.5}
    if kind == "dual_directional":
        base["cap"] = 1.3
    product = build_product(base)
    target = 0.99 if kind != "zero_coupon" else 1.0
    solved = solve(product, market, 2.5, target=target)
    assert solved.price(market, 2.5) == pytest.approx(target, abs=1e-8)


def test_tracker_participation_is_inverse_discounted_forward(market):
    solved = solve(TrackerNote(0.5), market, initial_fixing=2.0, target=1.0)
    fwd_r = market.forward(0.5) / 2.0
    assert solved.participation == pytest.approx(1 / (market.df(0.5) * fwd_r))


def test_zero_coupon_redemption_gives_issuer_yield(market):
    solved = solve(ZeroCouponNote(1.0), market, 2.5, target=1.0)
    assert np.log(solved.redemption) == pytest.approx(0.05)  # 4% rate + 1% spread


def test_capital_protection_raises_participation_when_protection_lowered(market):
    full = solve(CapitalProtectedNote(1.0, protection=1.0), market, 2.5, target=0.99)
    partial = solve(CapitalProtectedNote(1.0, protection=0.9), market, 2.5, target=0.99)
    assert partial.participation > full.participation
