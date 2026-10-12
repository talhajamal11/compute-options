import numpy as np
import pytest

from compute_options.credit import (
    CDS_ARCHETYPES,
    CreditMarket,
    ForwardStrip,
    GPULoan,
    HazardCurve,
    cds_par_spread,
    loan_fair_spread,
    make_counterparty,
    prepaid_discount,
    strip_cva,
)
from compute_options.credit.exposure import monthly_grid
from compute_options.market.forward_curve import SchwartzForward


@pytest.fixture(scope="module")
def mkt(discount):
    fwd = SchwartzForward(3.0, kappa=1.5, theta=np.log(2.6), sigma=0.2)
    return CreditMarket(fwd, 0.2, 1.5, discount, n_paths=8000, seed=1)


def test_flat_hazard_spread_is_credit_triangle():
    h = HazardCurve.flat(0.05)
    assert h.credit_spread(5.0, 0.4) == pytest.approx(0.6 * 0.05)
    assert h.survival(2.0) == pytest.approx(np.exp(-0.1))


@pytest.mark.parametrize("name", list(CDS_ARCHETYPES))
def test_bootstrap_reprices_every_cds(name, discount):
    hz = HazardCurve.from_cds(CDS_ARCHETYPES[name], 0.4, discount.df)
    for T, s in CDS_ARCHETYPES[name].items():
        assert cds_par_spread(hz, T, 0.4, discount.df) == pytest.approx(s, abs=1e-9)
    t = np.linspace(0, 6, 50)
    assert np.all(np.diff(hz.survival(t)) <= 0)


def test_cva_without_wrong_way_equals_textbook_sum(mkt, discount):
    cp = make_counterparty("Neocloud (HY)", discount)
    strip = ForwardStrip(1000, 12, 3.0, 1)
    res = strip_cva(strip, cp, mkt, 0.0)
    t = monthly_grid(12)
    dpd = -np.diff(cp.hazard.survival(np.concatenate([[0.0], t])))
    dfs = np.array([discount.df(x) for x in t])
    expected = (1 - cp.recovery) * np.sum(res.expected_exposure * dpd * dfs)
    assert res.cva == pytest.approx(expected, rel=1e-9)


def test_cva_rises_with_credit_risk_and_wrong_way(mkt, discount):
    strip = ForwardStrip(1000, 12, 3.0, 1)
    by_name = [
        strip_cva(strip, make_counterparty(n, discount), mkt, 0.0).cva for n in CDS_ARCHETYPES
    ]
    assert by_name == sorted(by_name)
    cp = make_counterparty("Neocloud (HY)", discount)
    assert strip_cva(strip, cp, mkt, 0.6).cva > strip_cva(strip, cp, mkt, 0.0).cva
    assert strip_cva(strip, cp, mkt, -0.6).cva < strip_cva(strip, cp, mkt, 0.0).cva


def test_prepaid_discount_is_close_to_loss_given_default_times_pd(mkt, discount):
    cp = make_counterparty("Neocloud (HY)", discount)
    d = prepaid_discount(1000, 12, cp, mkt, 0.0)
    # Deliveries are spread over a year, so average exposure is about half the price.
    approx = (1 - cp.recovery) * cp.hazard.default_prob(0.0, 1.0) * 0.5
    assert d["discount_pct"] == pytest.approx(approx, rel=0.35)
    assert d["fair_prepay_price"] < d["risk_free_price"]


def test_loan_spread_zero_at_low_ltv_and_monotone(mkt, discount):
    cp = make_counterparty("Neocloud (HY)", discount)
    spreads = [
        loan_fair_spread(GPULoan(ltv=lv), cp, mkt, 0.3)["spread"] for lv in (0.3, 0.6, 0.8, 0.95)
    ]
    assert spreads[0] == pytest.approx(0.0, abs=1e-6)
    assert spreads == sorted(spreads)
    hi = loan_fair_spread(GPULoan(ltv=0.8), cp, mkt, 0.7)["spread"]
    lo = loan_fair_spread(GPULoan(ltv=0.8), cp, mkt, 0.0)["spread"]
    assert hi > lo
