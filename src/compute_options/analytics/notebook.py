"""Helpers shared by the per-structure notebooks: choose an underlying, price a structure, show
its payoff and expected return, and backtest holding it.

Each notebook sets its parameters in one cell and calls these, so every structure is analysed
the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compute_options.backtest import HoldingConfig, HoldingResult, MarketModelConfig, run_holding
from compute_options.backtest.market_builder import MarketBuilder
from compute_options.backtest.performance import drawdown
from compute_options.data.store import ObservationStore
from compute_options.market.state import MarketState
from compute_options.market.vol import realized_vol
from compute_options.products import StructuredProduct, build_product, solve
from compute_options.risk import expected_return, greeks, historical_scenarios

SOLVE = "solve"  # put this as a term's value to have the solver set it

# Terms drawn as vertical markers on payoff charts.
KEY_LEVELS = ("strike", "cap", "buffer", "barrier", "put_strike", "call_strike")

PERCENT_ROWS = {
    "total return",
    "APR (annualised)",
    "annualised vol",
    "max drawdown",
    "best day",
    "worst day",
}


def _pct(x: float) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.2%}"


# ---------------------------------------------------------------------------------------------
# Underlying
# ---------------------------------------------------------------------------------------------


def available_underlyings(series: str = "composite_index", store: ObservationStore | None = None):
    """Every instrument in ``series`` with its history, latest price and weekly-return vol."""
    frame = (store or ObservationStore()).read_frame(series)
    rows = []
    for inst in frame.columns:
        s = frame[inst].dropna()
        if len(s) < 10:
            continue
        rows.append(
            {
                "instrument": inst,
                "from": s.index[0].date(),
                "to": s.index[-1].date(),
                "days": len(s),
                "last ($/GPU-hr)": round(float(s.iloc[-1]), 3),
                "vol (weekly returns)": _pct(realized_vol(s, horizon_days=7)),
            }
        )
    return pd.DataFrame(rows).sort_values("days", ascending=False).set_index("instrument")


def load_underlying(
    instrument: str, series: str = "composite_index", store: ObservationStore | None = None
) -> pd.Series:
    frame = (store or ObservationStore()).read_frame(series)
    if instrument not in frame:
        raise KeyError(f"{instrument} not in {series}; available: {list(frame.columns)}")
    return frame[instrument].dropna().rename(instrument)


def load_rates(
    market: dict[str, Any], store: ObservationStore | None = None
) -> pd.DataFrame | None:
    if market.get("rate") is not None:
        return None
    return (store or ObservationStore()).read_frame("rates")


# ---------------------------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------------------------


def split_terms(kind: str, tenor: str | float, terms: dict[str, Any]) -> tuple[dict, str | None]:
    """Product spec plus the term marked ``"solve"`` (at most one)."""
    to_solve = [k for k, v in terms.items() if v == SOLVE]
    if len(to_solve) > 1:
        raise ValueError(f"only one term can be solved, got {to_solve}")
    spec = {"kind": kind, "tenor": tenor, **{k: v for k, v in terms.items() if v != SOLVE}}
    return spec, (to_solve[0] if to_solve else None)


@dataclass
class PricedStructure:
    product: StructuredProduct
    market: MarketState
    initial_fixing: float
    issue_price: float
    solve_for: str | None
    spec: dict[str, Any]

    @property
    def date(self) -> pd.Timestamp:
        return self.market.valuation_date

    @property
    def fair_value(self) -> float:
        return self.product.price(self.market, self.initial_fixing)


def price_structure(
    series: pd.Series,
    kind: str,
    tenor: str | float,
    terms: dict[str, Any],
    market: dict[str, Any],
    issue_price: float = 1.0,
    issuer_margin: float = 0.01,
    date: str | None = None,
) -> PricedStructure:
    """Price on the last date of ``series`` (or ``date``). If a term is ``"solve"``, it is set
    so the fair value equals ``issue_price - issuer_margin``; otherwise the terms are priced as
    given and the issuer's margin is whatever is left over."""
    spec, solve_for = split_terms(kind, tenor, terms)
    builder = MarketBuilder(series, MarketModelConfig(**market), load_rates(market))
    when = pd.Timestamp(date) if date else series.index[-1]
    state, s0 = builder.state(when), builder.spot(when)
    product = build_product(spec)
    if solve_for:
        product = solve(product, state, s0, target=issue_price - issuer_margin, field=solve_for)
    return PricedStructure(product, state, s0, issue_price, solve_for, spec)


def terms_table(p: PricedStructure) -> pd.DataFrame:
    rows = []
    for k, v in p.product.terms().items():
        label = f"{k} (solved)" if k == p.solve_for else k
        if k == "tenor":
            v = f"{v:g}y ({round(v * 12)}M)"
        elif isinstance(v, float):
            v = f"{v:.2%}" if k != "downside_gearing" else f"{v:.3f}"
        rows.append({"term": label, "value": v})
    return pd.DataFrame(rows).set_index("term")


def pricing_summary(p: PricedStructure) -> pd.DataFrame:
    T = p.product.tenor
    fv = p.fair_value
    rows = {
        "valuation date": p.date.date(),
        "initial fixing ($/GPU-hr)": f"{p.initial_fixing:.4f}",
        "forward at maturity ($/GPU-hr)": f"{p.market.forward(T):.4f}",
        "vol used (ATM)": _pct(p.market.implied_vol(T, 1.0)),
        "issuer zero rate (incl. spread)": _pct(p.market.discount.zero_rate(T)),
        "issue price": _pct(p.issue_price),
        "fair value": _pct(fv),
        "issuer margin (issue price - fair value)": _pct(p.issue_price - fv),
    }
    return pd.DataFrame({"value": rows})


def legs_table(p: PricedStructure) -> pd.DataFrame:
    df = pd.DataFrame(p.product.leg_breakdown(p.market, p.initial_fixing))
    return df.set_index("leg").round(4)


def greeks_table(p: PricedStructure) -> pd.DataFrame:
    g = greeks(p.product, p.market, p.initial_fixing).as_dict()
    return pd.DataFrame({"per unit notional": g}).round(5)


def expected_table(p: PricedStructure, series: pd.Series, drift: float = 0.0) -> pd.DataFrame:
    """Expected return of buying at the issue price and holding to maturity, under the model
    (index lognormal around its forward, plus ``drift`` a year) and under every realised move
    of the same length in the history."""
    model = expected_return(p.product, p.market, p.initial_fixing, p.issue_price, drift)
    hist, _ = historical_scenarios(p.product, series, p.issue_price)
    n = int(hist.pop("scenarios", 0)) if hist else 0
    cols = {f"model (drift {drift:+.0%}/yr)": model}
    if hist:
        cols[f"historical ({n} start dates)"] = hist
    out = pd.DataFrame(cols)
    return out.map(_pct)


def historical_R(series: pd.Series, tenor: float) -> pd.Series:
    """Realised ``S_{t+tenor} / S_t`` for every start date in the history."""
    daily = series.resample("D").last().ffill()
    days = round(tenor * 365)
    return (daily.shift(-days) / daily).dropna()


def plot_payoff(
    p: PricedStructure, series: pd.Series | None = None, lo: float = 0.4, hi: float = 1.8
):
    """Payoff at maturity per unit notional; behind it, a histogram of realised index moves
    over the same tenor, showing which part of the payoff history actually visited."""
    R = np.linspace(lo, hi, 561)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if series is not None:
        hist = historical_R(series, p.product.tenor)
        if len(hist):
            ax2 = ax.twinx()
            ax2.hist(hist.clip(lo, hi), bins=40, range=(lo, hi), color="tab:blue", alpha=0.12)
            ax2.set_ylabel(
                f"historical {round(p.product.tenor * 12)}M moves (count)", color="tab:blue"
            )
            ax2.tick_params(axis="y", colors="tab:blue")
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)
    ax.plot(R, R, color="grey", ls="--", lw=1, label="index (R)")
    ax.plot(R, p.product.payoff(R), color="black", lw=2.5, label=p.product.kind.replace("_", " "))
    ax.axhline(p.issue_price, color="black", lw=0.5)
    ax.axvline(1.0, color="black", lw=0.5)
    for key in KEY_LEVELS:
        v = getattr(p.product, key, None)
        if v is None:
            continue
        x = 1.0 - v if key == "buffer" else v
        if lo < x < hi:
            ax.axvline(x, color="tab:red", ls=":", lw=1)
            ax.annotate(
                f"{key} {x:.0%}",
                (x, ax.get_ylim()[1]),
                rotation=90,
                va="top",
                ha="right",
                fontsize=8,
                color="tab:red",
            )
    ax.set_xlabel("index at maturity / initial fixing (R)")
    ax.set_ylabel("redemption (fraction of notional)")
    ax.set_title(
        f"{p.product.kind.replace('_', ' ').title()} "
        f"({round(p.product.tenor * 12)}M): payoff at maturity, priced {p.date.date()}"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
# Holding backtest
# ---------------------------------------------------------------------------------------------


def run_hold(
    series: pd.Series,
    p: PricedStructure,
    market: dict[str, Any],
    issuer_margin: float,
    start: str | None = None,
    end: str | None = None,
    roll: bool = True,
) -> HoldingResult:
    """Hold ``p``'s structure (same kind, tenor and fixed terms; the solved term is re-solved
    on each purchase date) from ``start`` to ``end``, rolling at maturity."""
    cfg = HoldingConfig(
        product=p.spec,
        solve_for=p.solve_for,
        issue_price=p.issue_price,
        issuer_margin=issuer_margin,
        start=start,
        end=end,
        roll=roll,
        market=MarketModelConfig(**market),
    )
    return run_holding(series, cfg, load_rates(market))


def holding_stats_table(res: HoldingResult) -> pd.DataFrame:
    out = res.stats.astype(object)
    for row in out.index:
        if row in PERCENT_ROWS:
            out.loc[row] = out.loc[row].map(_pct)
        elif row == "max drawdown (days)":
            out.loc[row] = out.loc[row].map(lambda v: f"{v:.0f}")
        else:
            out.loc[row] = out.loc[row].map(lambda v: "n/a" if pd.isna(v) else f"{v:.2f}")
    return out.rename(columns={"structure": "structure (held)", "index": "index (not investable)"})


def notes_held_table(res: HoldingResult) -> pd.DataFrame:
    n = res.notes
    solved = [
        c
        for c in n.columns
        if c.startswith("term_")
        and c != "term_tenor"
        and pd.api.types.is_numeric_dtype(n[c])
        and n[c].nunique() > 1
    ]
    df = pd.DataFrame(
        {
            "bought": pd.to_datetime(n["issue_date"]).dt.date,
            "matures": pd.to_datetime(n["maturity_date"]).dt.date,
            "status": np.where(n["matured"], "matured", "held (marked to model)"),
            "fixing ($/GPU-hr)": n["initial_fixing"].round(4),
            **{c.removeprefix("term_"): n[c].map(_pct) for c in solved},
            "index move": n["underlying_return"].map(_pct),
            "note return": n["investor_return"].map(_pct),
            "note APR": n["investor_return_ann"].map(_pct),
            "issuer margin": n["issuer_margin"].map(_pct),
            "issuer P&L unhedged": n["issuer_unhedged_pnl"].map(_pct),
            "issuer P&L hedged": n.get("issuer_hedged_pnl", pd.Series(np.nan, n.index)).map(_pct),
        }
    )
    return df.set_index("bought")


def plot_holding(res: HoldingResult):
    nav = res.nav
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]}
    )
    ax1.plot(nav.index, nav["structure"], color="black", lw=2, label="structure (held, rolled)")
    ax1.plot(
        nav.index, nav["index"], color="tab:blue", lw=1, alpha=0.8, label="index (not investable)"
    )
    ax1.plot(nav.index, nav["cash"], color="grey", lw=1, ls="--", label="cash (T-bill)")
    for _, n in res.notes.iterrows():
        ax1.axvline(n["issue_date"], color="tab:green", lw=1, alpha=0.6)
    ax1.axhline(1.0, color="black", lw=0.5)
    ax1.set_ylabel("value of $1 invested")
    ax1.set_title(
        f"Holding from {res.start.date()} to {res.end.date()} (green lines = note purchases)"
    )
    ax1.legend()

    for col, color in (("structure", "black"), ("index", "tab:blue")):
        ax2.fill_between(
            nav.index,
            drawdown(nav[col]),
            0,
            color=color,
            alpha=0.25 if col == "structure" else 0.12,
            label=col,
        )
    ax2.set_ylabel("drawdown")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig
