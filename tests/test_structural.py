import numpy as np
import pandas as pd
import pytest

from compute_options.backtest.market_builder import MarketBuilder, MarketModelConfig
from compute_options.products import build_product, solve
from compute_options.structural import (
    central_params,
    fit_posterior,
    levelised_cost,
    register,
    sample_params,
)
from compute_options.structural.assumptions import with_value
from compute_options.structural.curve import StructuralForward


def test_levelised_cost_matches_hand_calculation():
    p = {**central_params("H100_SXM"), "rate": 0.10, "life": 5.0, "resid": 0.0, "util": 1.0}
    crf = 0.10 / (1 - 1.1**-5)
    capital_per_hr = p["capex"] * crf / 8760
    assert float(levelised_cost(p).capital) == pytest.approx(capital_per_hr)


@pytest.mark.parametrize(
    ("key", "direction"),
    [("capex", 1), ("tariff", 1), ("rate", 1), ("util", -1), ("life", -1), ("resid", -1)],
)
def test_cost_moves_the_right_way(key, direction):
    base = central_params("B200")
    a = next(x for x in register("B200") if x.key == key)
    lo = float(levelised_cost(with_value(base, **{key: a.low})).total)
    hi = float(levelised_cost(with_value(base, **{key: a.high})).total)
    assert (hi - lo) * direction > 0


def test_register_ranges_are_ordered_and_documented():
    for gpu in ("H100_SXM", "B200"):
        for a in register(gpu):
            assert a.low <= a.central <= a.high
            assert a.source
            if a.basis == "sourced":
                assert a.url


def test_sampling_stays_inside_ranges():
    draws = sample_params("H100_SXM", 2000, seed=1)
    for a in register("H100_SXM"):
        assert draws[a.key].min() >= a.low and draws[a.key].max() <= a.high


def test_curve_scaling_is_linear():
    c = StructuralForward.from_params(3.0, central_params("H100_SXM"), 0.2)
    s = c.scaled(1.3)
    for t in (0.0, 0.5, 2.0):
        assert s.forward(t) == pytest.approx(1.3 * c.forward(t), rel=1e-9)


def test_curve_converges_to_a_static_anchor_without_drift():
    p = {**central_params("H100_SXM"), "delta": 0.0, "risk_premium": 0.0, "kappa": 3.0}
    c = StructuralForward.from_params(6.0, p, 1e-6)
    assert c.forward(10.0) == pytest.approx(float(c.anchor[0]), rel=1e-4)


def test_posterior_weights_and_ess(posterior):
    w = posterior.curve.weights
    assert w.sum() == pytest.approx(1.0)
    assert posterior.ess >= 0.15 * len(w) - 1
    assert 0 < posterior.beta <= 1


def test_posterior_needs_history():
    short = pd.Series([3.0] * 10, index=pd.date_range("2026-01-01", periods=10))
    with pytest.raises(ValueError, match="history"):
        fit_posterior(short, "H100_SXM")


def test_posterior_moves_towards_a_higher_price_level(history):
    """If the index sits far above cost, the posterior margin should rise above the prior."""
    high = history * 2.5
    post = fit_posterior(high, "H100_SXM", n=1500, seed=0)
    margin = np.asarray(post.params["margin"])
    assert post.curve.weights @ margin > post.prior.weights @ margin


def test_notes_price_off_the_structural_curve(history):
    cfg = MarketModelConfig(
        forward_model="structural", structural_gpu="H100_SXM", structural_draws=800, rate=0.04
    )
    b = MarketBuilder(history, cfg)
    date = history.index[-1]
    state, s0 = b.state(date), b.spot(date)
    note = build_product({"kind": "tracker", "tenor": "6M", "participation": 1.0})
    solved = solve(note, state, s0, target=0.99)
    assert solved.price(state, s0) == pytest.approx(0.99, abs=1e-8)
    # Tracker participation at par is 1 / (DF * F/S): backwardation pushes it above flat's.
    assert state.forward(0.5) != pytest.approx(s0)
