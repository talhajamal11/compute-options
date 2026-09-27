import numpy as np
import pandas as pd
import pytest

from compute_options.backtest.synthetic import simulate_index
from compute_options.market import DiscountCurve, MeanRevertingVol, SchwartzForward
from compute_options.market.calibration import OUParams, fit_ou
from compute_options.market.vol import realized_vol
from compute_options.models.processes import simulate_paths


def test_simulated_paths_are_centred_on_forward_curve():
    fwd = SchwartzForward(2.5, kappa=3.0, theta=np.log(2.0), sigma=0.6)
    times = np.array([0.1, 0.5, 1.0])
    paths = simulate_paths(fwd, 0.6, 3.0, times, 200_000, seed=0)
    assert paths.mean(axis=0) == pytest.approx([fwd.forward(t) for t in times], rel=5e-3)


def test_mean_reverting_vol_matches_simulated_terminal_dispersion():
    fwd = SchwartzForward(2.5, kappa=4.0, theta=np.log(2.5), sigma=0.8)
    paths = simulate_paths(fwd, 0.8, 4.0, np.linspace(0.01, 1.0, 50), 100_000, seed=2)
    realised = np.log(paths[:, -1]).std() / np.sqrt(1.0)
    assert realised == pytest.approx(MeanRevertingVol(0.8, 4.0).implied_vol(1.0, 1.0), rel=1e-2)


def test_ou_fit_recovers_parameters():
    true = OUParams(kappa=12.0, theta=np.log(3.0), sigma=0.7, n_obs=0)
    s = simulate_index(3.5, true, 4000, seed=3)
    fit = fit_ou(s)
    assert fit.kappa == pytest.approx(12.0, rel=0.35)
    assert fit.sigma == pytest.approx(0.7, rel=0.05)
    assert np.exp(fit.theta) == pytest.approx(3.0, rel=0.05)


def test_realized_vol_ignores_gaps_and_supports_weekly_horizon():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=2000, freq="D")
    s = pd.Series(np.exp(np.cumsum(rng.normal(0, 0.5 / np.sqrt(365), len(idx)))), idx)
    gappy = s.drop(s.index[::3])
    assert realized_vol(gappy) == pytest.approx(0.5, rel=0.06)
    assert realized_vol(s, horizon_days=7) == pytest.approx(0.5, rel=0.1)


def test_discount_curve_from_fred_row_converts_to_continuous():
    row = pd.Series({"DGS3MO": 0.04, "DGS1": 0.04, "DGS2": 0.04})
    curve = DiscountCurve.from_fred_row(row, spread=0.01)
    assert curve.zero_rate(1.0) == pytest.approx(2 * np.log1p(0.02) + 0.01)
    assert curve.df(0.0) == 1.0


def test_two_factor_fit_separates_level_and_noise():
    from compute_options.backtest.synthetic import simulate_two_factor
    from compute_options.market.calibration import TwoFactorParams, fit_two_factor

    true = TwoFactorParams(sigma_long=0.4, kappa=60.0, sigma_short=1.2, n_obs=0)
    s = simulate_two_factor(2.5, true, 6000, seed=2)
    fit = fit_two_factor(s)
    assert fit.sigma_long == pytest.approx(0.4, rel=0.2)
    assert fit.noise_sd == pytest.approx(true.noise_sd, rel=0.2)
    assert fit.kappa == pytest.approx(60.0, rel=0.35)
