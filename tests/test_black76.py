import numpy as np
import pytest

from compute_options.models import black76 as b76

F, K, T, S, DF = 2.5, np.array([1.5, 2.5, 3.5]), 0.5, 0.45, 0.98


def test_put_call_parity():
    assert np.allclose(b76.call(F, K, T, S, DF) - b76.put(F, K, T, S, DF), DF * (F - K))


def test_digitals_sum_to_bond():
    assert np.allclose(b76.digital_call(F, K, T, S, DF) + b76.digital_put(F, K, T, S, DF), DF)


def test_digital_is_minus_call_strike_derivative():
    h = 1e-5
    fd = -(b76.call(F, K + h, T, S, DF) - b76.call(F, K - h, T, S, DF)) / (2 * h)
    assert np.allclose(b76.digital_call(F, K, T, S, DF), fd, atol=1e-6)


@pytest.mark.parametrize("greek,bumped", [("delta", "F"), ("vega", "sigma")])
def test_greeks_match_finite_differences(greek, bumped):
    h = 1e-5
    args = {"F": F, "K": K, "T": T, "sigma": S, "df": DF}
    up, dn = dict(args), dict(args)
    up[bumped] += h
    dn[bumped] -= h
    fd = (b76.call(**up) - b76.call(**dn)) / (2 * h)
    assert np.allclose(getattr(b76, greek)(**args), fd, atol=1e-6)


def test_zero_vol_and_expiry_give_intrinsic():
    assert np.allclose(b76.call(F, K, 0.0, S, DF), DF * np.maximum(F - K, 0))
    assert np.allclose(b76.put(F, K, T, 0.0, DF), DF * np.maximum(K - F, 0))


def test_implied_vol_round_trip():
    price = float(b76.call(F, 2.7, T, 0.6, DF))
    assert b76.implied_vol(price, F, 2.7, T, DF) == pytest.approx(0.6, abs=1e-8)
