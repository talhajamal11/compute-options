import numpy as np
import pandas as pd
import pytest

from compute_options.backtest import (
    HoldingConfig,
    MarketModelConfig,
    performance_stats,
    run_holding,
)
from compute_options.backtest.synthetic import simulate_index
from compute_options.market.calibration import OUParams
from compute_options.products import BufferedNote, parse_tenor, solve
from compute_options.risk import expected_return

MARKET = MarketModelConfig(vol_window=90, rate=0.04, issuer_spread=0.01, min_history=30)


@pytest.fixture(scope="module")
def series():
    return simulate_index(2.5, OUParams(0.0, 0.0, 0.4, 0), 400, start_date="2026-01-01", seed=11)


@pytest.mark.parametrize(
    "tenor,years",
    [("3M", 0.25), ("6m", 0.5), ("12M", 1.0), ("1Y", 1.0), ("2W", 14 / 365), (0.75, 0.75)],
)
def test_parse_tenor(tenor, years):
    assert parse_tenor(tenor) == pytest.approx(years)


def test_performance_stats_on_known_series():
    idx = pd.date_range("2026-01-01", periods=366, freq="D")
    nav = pd.Series(np.exp(0.05 * np.arange(366) / 365), idx)
    nav.iloc[100:110] *= 0.9  # a 10% dip that recovers
    s = performance_stats(nav, rf=0.0)
    assert s["total return"] == pytest.approx(np.exp(0.05) - 1)
    assert s["APR (annualised)"] == pytest.approx(np.exp(0.05) - 1, rel=1e-3)
    assert s["max drawdown"] == pytest.approx(-0.1, abs=1e-3)
    assert s["max drawdown (days)"] == 10


def test_holding_rolls_and_reinvests(series):
    cfg = HoldingConfig(
        product={"kind": "buffered", "tenor": "3M", "buffer": 0.1},
        solve_for="participation",
        market=MARKET,
    )
    res = run_holding(series, cfg)
    notes = res.notes
    assert len(notes) >= 3 and notes["matured"].iloc[:-1].all()
    # Each roll reinvests exactly the previous note's payoff.
    for prev, nxt in zip(notes.itertuples(), notes.iloc[1:].itertuples(), strict=False):
        assert nxt.invested == pytest.approx(prev.units * prev.value)
    # NAV starts at fair value (issue price less margin) and ends at units * last value.
    # (tolerance: the note is solved on a 0.25y tenor, marked on 91 calendar days)
    assert res.nav["structure"].iloc[0] == pytest.approx(0.99, abs=1e-3)
    assert res.nav["structure"].iloc[-1] == pytest.approx(
        notes["units"].iloc[-1] * notes["value"].iloc[-1]
    )
    assert set(res.stats.columns) == {"structure", "index", "cash"}


def test_holding_without_roll_sits_in_cash(series):
    cfg = HoldingConfig(
        product={"kind": "tracker", "tenor": "1M"},
        solve_for="participation",
        roll=False,
        market=MARKET,
    )
    res = run_holding(series, cfg)
    assert len(res.notes) == 1
    assert (res.nav["note_id"] == "cash").any()


def test_expected_return_with_zero_drift_is_funding_less_margin(market):
    note = solve(BufferedNote(1.0, buffer=0.1), market, 2.5, target=0.99)
    out = expected_return(note, market, 2.5, issue_price=1.0, drift=0.0, n_paths=400_000)
    # E[payoff] under the pricing distribution = fair value / DF.
    assert out["expected return (to maturity)"] == pytest.approx(
        0.99 / market.df(1.0) - 1, abs=2e-3
    )
