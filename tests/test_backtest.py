import numpy as np
import pytest

from compute_options.backtest import BacktestConfig, MarketModelConfig, run_backtest
from compute_options.backtest.synthetic import simulate_index
from compute_options.market.calibration import OUParams

PARAMS = OUParams(kappa=0.0, theta=0.0, sigma=0.4, n_obs=0)
MARKET = MarketModelConfig(vol_window=120, rate=0.04, issuer_spread=0.01, min_history=60)


@pytest.fixture(scope="module")
def series():
    return simulate_index(2.5, PARAMS, 500, seed=7)


def cfg(product, **kw):
    return BacktestConfig(product=product, market=MARKET, issue_every_days=14, **kw)


def test_notes_issue_at_par_minus_margin_and_mature(series):
    res = run_backtest(series, cfg({"kind": "buffered", "tenor": 0.25, "buffer": 0.1}))
    notes = res.notes
    assert len(notes) > 10 and notes["matured"].sum() > 5
    assert np.allclose(notes["fair_value"], 0.99)
    assert np.allclose(notes["issuer_margin"], 0.01)
    done = notes[notes["matured"]]
    expected = (
        1
        + done["term_participation"] * np.maximum(done["final_R"] - 1, 0)
        - np.maximum(0.9 - done["final_R"], 0)
    )
    assert np.allclose(done["value"], expected)


def test_daily_delta_hedge_reduces_issuer_risk(series):
    notes = run_backtest(
        series, cfg({"kind": "capital_protected", "tenor": 0.25, "cap": 1.5})
    ).notes
    done = notes[notes["matured"]]
    assert done["issuer_hedged_pnl"].std() < 0.5 * done["issuer_unhedged_pnl"].std()


def test_tracker_hedge_is_near_perfect(series):
    notes = run_backtest(series, cfg({"kind": "tracker", "tenor": 0.25})).notes
    assert notes["issuer_hedge_error"].abs().max() < 5e-3
