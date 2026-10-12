"""Model validation: numerical checks, independent price verification and a walk-forward test.

The checks are the standard first ones: does the code reproduce a case with
a known answer, does the simulation agree with the closed form, do limits behave. The walk-forward
test is the stricter one: with the history available at each past date, how did the forward curve's
prediction compare with what happened, against the naive "price stays flat" forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from compute_options.credit import ForwardStrip, HazardCurve, cds_par_spread, strip_cva
from compute_options.credit.exposure import (
    conditional_expected_spot,
    monthly_grid,
    simulate_monthly,
)
from compute_options.credit.trades import CDS_ARCHETYPES, CreditMarket, make_counterparty
from compute_options.market.calibration import fit_ou
from compute_options.market.forward_curve import FlatForward, SchwartzForward
from compute_options.market.rates import DiscountCurve
from compute_options.models import black76
from compute_options.models.processes import simulate_paths
from compute_options.structural.assumptions import central_params
from compute_options.structural.curve import StructuralForward
from compute_options.structural.posterior import Posterior, fit_posterior


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def run_checks(post: Posterior, discount: DiscountCurve) -> list[Check]:
    out: list[Check] = []
    p0 = central_params(post.gpu)

    # 1. Fast reversion to a static anchor: forward converges to the anchor.
    c = StructuralForward.from_params(
        5.0, {**p0, "kappa": 200.0, "delta": 0.0, "risk_premium": 0.0, "margin": 0.0}, 1e-6
    )
    f = c.forward(1.0)
    anchor = float(c.anchor[0])
    out.append(
        Check(
            "Fast reversion lands on the anchor",
            abs(f / anchor - 1) < 1e-3,
            f"F(1y)={f:.4f} vs anchor {anchor:.4f}",
        )
    )

    # 2. Spot at the anchor, no drift, no vol: the curve is flat.
    a0 = float(
        StructuralForward.from_params(1.0, {**p0, "delta": 0.0, "risk_premium": 0.0}, 1e-9).anchor[
            0
        ]
    )
    flat = StructuralForward.from_params(a0, {**p0, "delta": 0.0, "risk_premium": 0.0}, 1e-9)
    dev = max(abs(flat.forward(t) / a0 - 1) for t in (0.25, 1.0, 3.0))
    out.append(
        Check(
            "Spot at anchor with no drift or vol gives a flat curve",
            dev < 1e-6,
            f"max deviation {dev:.2e}",
        )
    )

    # 3. Monte Carlo reproduces the forward curve it is built on.
    curve = post.curve.point()
    sigma = float(curve.sigma[0])
    kappa = float(curve.kappa[0])
    times = np.array([0.25, 0.5, 1.0, 2.0])
    paths = simulate_paths(curve, sigma, kappa, times, 200_000, seed=1)
    z = [
        (paths[:, i].mean() - curve.forward(t)) / (paths[:, i].std() / np.sqrt(len(paths)))
        for i, t in enumerate(times)
    ]
    out.append(
        Check(
            "Simulated paths average to the forward",
            max(abs(x) for x in z) < 3.5,
            "z-scores " + ", ".join(f"{x:+.2f}" for x in z),
        )
    )

    # 4. Black-76 call equals Monte Carlo on the same paths.
    T, K = 0.5, curve.forward(0.5)
    sig_eff = np.sqrt(sigma**2 * (1 - np.exp(-2 * kappa * T)) / (2 * kappa * T))
    bs = float(black76.call(curve.forward(T), K, T, sig_eff, 1.0))
    sim = simulate_paths(curve, sigma, kappa, np.array([T]), 400_000, seed=2)[:, 0]
    mc = np.maximum(sim - K, 0).mean()
    se = np.maximum(sim - K, 0).std() / np.sqrt(len(sim))
    out.append(
        Check(
            "At-the-money call: Black-76 equals Monte Carlo",
            abs(mc - bs) < 3.5 * se,
            f"Black-76 {bs:.4f}, MC {mc:.4f} +/- {se:.4f}",
        )
    )

    # 5. CDS bootstrap reprices every input spread.
    rf = discount.with_spread(0.0)
    errs = []
    for spreads in CDS_ARCHETYPES.values():
        hz = HazardCurve.from_cds(spreads, 0.4, rf.df)
        errs += [abs(cds_par_spread(hz, T, 0.4, rf.df) - s) for T, s in spreads.items()]
    out.append(
        Check(
            "Hazard bootstrap reprices every quoted CDS",
            max(errs) < 1e-8,
            f"max error {max(errs) * 1e4:.2e} bp",
        )
    )

    # 6. With zero wrong-way correlation, CVA is the textbook EE x PD sum.
    cp = make_counterparty("Neocloud (HY)", discount)
    mkt = CreditMarket(post.curve, sigma, kappa, rf, n_paths=20_000, seed=3)
    strip = ForwardStrip(1000, 12, curve.forward(0.5), 1)
    res = strip_cva(strip, cp, mkt, 0.0)
    t = monthly_grid(12)
    dpd = -np.diff(cp.hazard.survival(np.concatenate([[0.0], t])))
    dfs = np.array([rf.df(x) for x in t])
    textbook = (1 - cp.recovery) * float(np.sum(res.expected_exposure * dpd * dfs))
    out.append(
        Check(
            "CVA with rho = 0 equals sum of EE x default probability x discount",
            abs(res.cva / textbook - 1) < 1e-9,
            f"{res.cva:.2f} vs {textbook:.2f}",
        )
    )

    # 7. Wrong-way risk raises CVA.
    hi = strip_cva(strip, cp, mkt, 0.6).cva
    out.append(
        Check(
            "Wrong-way correlation raises CVA",
            hi > res.cva,
            f"rho 0: {res.cva:.1f}, rho 0.6: {hi:.1f}",
        )
    )

    # 8. Conditional expected spot is a martingale step: E[E_i[S_j]] = F_j.
    pth = simulate_monthly(post.curve, sigma, kappa, 12, 100_000, 4)
    cond = conditional_expected_spot(pth, post.curve, t, sigma, kappa)
    j = 11
    err = max(abs(cond[:, i, j].mean() / post.curve.forward(t[j]) - 1) for i in range(0, 12, 3))
    out.append(
        Check(
            "Tower property: mean of conditional expectations equals the forward",
            err < 5e-3,
            f"max relative error {err:.1e}",
        )
    )

    # 9. Posterior is not degenerate.
    out.append(
        Check(
            "Posterior effective sample size above 10% of draws",
            post.ess > 0.1 * len(post.curve.weights),
            f"ESS {post.ess:.0f} of {len(post.curve.weights)}, likelihood power {post.beta:.2f}",
        )
    )
    return out


def ipv_table(
    history: pd.Series, post: Posterior, discount: DiscountCurve, months: int = 12
) -> pd.DataFrame:
    """Independent price verification: the 12-month strip under four unrelated curve models."""
    hist = history.dropna()
    spot = float(hist.iloc[-1])
    ou = fit_ou(hist.loc[hist.index[-1] - pd.Timedelta(days=120) :])
    models = {
        "Structural, posterior": post.curve,
        "Structural, prior only": post.prior,
        "Schwartz, fitted to index": SchwartzForward(spot, ou.kappa, ou.theta, ou.sigma),
        "Flat (martingale)": FlatForward(spot),
    }
    times = monthly_grid(months)
    dfs = np.array([discount.df(t) for t in times])
    rows = []
    for name, c in models.items():
        f = np.array([c.forward(t) for t in times])
        rows.append(
            {
                "model": name,
                "strip price": float((f * dfs).sum() / dfs.sum()),
                "1y forward": c.forward(1.0),
            }
        )
    df = pd.DataFrame(rows)
    df["vs structural"] = df["strip price"] / df["strip price"].iloc[0] - 1
    return df


def model_risk_reserve(ipv: pd.DataFrame) -> float:
    """Std. dev. of the strip price across the independent models (excl. the prior-only variant)."""
    use = ipv[~ipv["model"].str.contains("prior")]
    return float(use["strip price"].std(ddof=0))


def walk_forward(
    history: pd.Series,
    gpu: str,
    horizons: tuple[int, ...] = (30, 60),
    step: int = 14,
    min_days: int = 56,
    n: int = 1500,
) -> pd.DataFrame:
    """At each origin fit on data to date, predict the index ``h`` days ahead, compare."""
    hist = history.dropna().sort_index()
    daily = hist.resample("D").last().ffill()
    rows = []
    for h in horizons:
        origins = daily.index[min_days : len(daily) - h : step]
        for o in origins:
            post = fit_posterior(hist.loc[:o], gpu, n=n, seed=0)
            realised = float(daily.loc[o + pd.Timedelta(days=h)])
            spot = float(daily.loc[o])
            t = h / 365.0
            q = post.curve.spot_quantiles(t, [0.05, 0.5, 0.95])
            rows.append(
                {
                    "origin": o,
                    "horizon": h,
                    "spot": spot,
                    "realised": realised,
                    "structural": post.curve.forward(t),
                    "q05": q[0],
                    "q95": q[2],
                }
            )
    df = pd.DataFrame(rows)
    df["err_struct"] = np.log(df["realised"] / df["structural"])
    df["err_flat"] = np.log(df["realised"] / df["spot"])
    df["in_band"] = (df["realised"] >= df["q05"]) & (df["realised"] <= df["q95"])
    return df


def walk_forward_summary(wf: pd.DataFrame) -> pd.DataFrame:
    g = wf.groupby("horizon")
    return pd.DataFrame(
        {
            "origins": g.size(),
            "bias structural": g["err_struct"].mean(),
            "bias flat": g["err_flat"].mean(),
            "RMSE structural": g["err_struct"].apply(lambda x: np.sqrt((x**2).mean())),
            "RMSE flat": g["err_flat"].apply(lambda x: np.sqrt((x**2).mean())),
            "90% band coverage": g["in_band"].mean(),
        }
    )
