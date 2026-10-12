import pytest

from compute_options.credit import CDS_ARCHETYPES
from compute_options.governance import (
    ipv_table,
    model_risk_reserve,
    run_checks,
    tornado,
    walk_forward,
    walk_forward_summary,
)
from compute_options.quoting import BilateralRequest, quote
from compute_options.structural import posterior_vol


def _quote(post, discount, cp, structure="forward", months=12):
    v = posterior_vol(post)
    req = BilateralRequest("H100_SXM", 100_000, months, cp, wrong_way=0.3)
    return quote(req, structure, post.curve, v.sigma, v.kappa, discount, n_paths=8000)


def test_quote_brackets_the_mid(posterior, discount):
    q = _quote(posterior, discount, "Neocloud (HY)")
    assert q.bid.price < q.mid < q.offer.price
    assert q.table().loc["quote", "bid"] == pytest.approx(q.bid.price)


def test_spread_widens_with_credit_risk_and_tenor(posterior, discount):
    widths = [_quote(posterior, discount, n, "prepaid").spread for n in CDS_ARCHETYPES]
    assert widths == sorted(widths)
    assert (
        _quote(posterior, discount, "Neocloud (HY)", "prepaid", 24).spread
        > _quote(posterior, discount, "Neocloud (HY)", "prepaid", 6).spread
    )


def test_prepaid_credit_charge_falls_on_the_side_that_pays_first(posterior, discount):
    q = _quote(posterior, discount, "AI lab (unrated)", "prepaid")
    assert q.offer.credit == 0.0  # client pays us upfront
    assert q.bid.credit > 0.0  # we pay upfront


def test_memo_mentions_every_charge(posterior, discount):
    memo = _quote(posterior, discount, "Neocloud (HY)").memo()
    for word in ("model uncertainty", "capital", "Credit", "mid"):
        assert word in memo


def test_numerical_checks_all_pass(posterior, discount):
    failed = [c for c in run_checks(posterior, discount) if not c.passed]
    assert not failed, [(c.name, c.detail) for c in failed]


def test_ipv_and_reserve(history, posterior, discount):
    ipv = ipv_table(history, posterior, discount)
    assert len(ipv) == 4
    assert model_risk_reserve(ipv) > 0


def test_tornado_is_sorted_and_centred(history, posterior, discount):
    tor = tornado("H100_SXM", float(history.iloc[-1]), posterior.sigma_hat, discount)
    assert tor["swing"].is_monotonic_decreasing
    base = tor.attrs["base"]
    assert (tor[["at low", "at high"]].min(axis=1) <= base + 1e-9).all()


def test_walk_forward_uses_only_past_data(history):
    wf = walk_forward(history, "H100_SXM", horizons=(30,), step=30, n=400)
    assert len(wf) >= 2
    assert (wf["origin"] + __import__("pandas").Timedelta(days=30) <= history.index[-1]).all()
    summary = walk_forward_summary(wf)
    assert {"RMSE structural", "RMSE flat", "90% band coverage"} <= set(summary.columns)
