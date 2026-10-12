"""Run the full analysis and write the figures and tables that the README embeds.

    uv run compute-options showcase            # writes docs/figures/*.png, docs/results.md
                                               # and refreshes the README results block

Every number in the README results block comes from this module, so the README cannot drift from
the code. Charts follow the conventions in ``market_viz`` (validated palette, one axis per chart,
thin marks, direct labels, text in ink colours).
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from compute_options.analytics.market_viz import (  # noqa: E402
    GRID,
    INK,
    INK_2,
    MUTED,
    PALETTE,
    SEQ_BLUE,
    SURFACE,
    _dollars,
    apply_style,
    gpu_color,
    with_gaps,
)
from compute_options.backtest.market_builder import MarketBuilder, MarketModelConfig  # noqa: E402
from compute_options.config import PROJECT_ROOT, load_yaml  # noqa: E402
from compute_options.credit import (  # noqa: E402
    CDS_ARCHETYPES,
    CreditMarket,
    ForwardStrip,
    GPULoan,
    issuer_spread,
    loan_fair_spread,
    make_counterparty,
    prepaid_discount,
    strip_cva,
)
from compute_options.data.store import ObservationStore  # noqa: E402
from compute_options.governance import (  # noqa: E402
    ipv_table,
    model_risk_reserve,
    rank_importance,
    run_checks,
    tornado,
    walk_forward,
    walk_forward_summary,
)
from compute_options.market.rates import DiscountCurve  # noqa: E402
from compute_options.products import SolveError, build_product, solve  # noqa: E402
from compute_options.quoting import BilateralRequest, quote  # noqa: E402
from compute_options.structural import (  # noqa: E402
    central_params,
    fit_posterior,
    levelised_cost,
    posterior_vol,
    register,
    structural_state,
)
from compute_options.structural.assumptions import (  # noqa: E402
    CREDIT_SOURCES,
    PERF_RATIO_B200_VS_H100,
    REVISIONS,
)

GPUS = ("H100_SXM", "B200")
COUNTERPARTIES = tuple(CDS_ARCHETYPES)
CP_COLORS = dict(zip(COUNTERPARTIES, PALETTE[:3], strict=True))
START, END = "<!-- showcase:start -->", "<!-- showcase:end -->"


# ---------------------------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------------------------
@dataclass
class Context:
    index: pd.DataFrame
    ornn: pd.DataFrame
    rates: pd.DataFrame
    discount: DiscountCurve
    date: pd.Timestamp
    posts: dict
    figdir: Path

    def hist(self, gpu: str) -> pd.Series:
        return self.index[gpu].dropna()


def build_context(figdir: Path) -> Context:
    store = ObservationStore()
    index = store.read_frame("composite_index")
    rates = store.read_frame("rates")
    date = index[list(GPUS)].dropna(how="all").index[-1]
    discount = DiscountCurve.from_fred_row(rates.loc[:date].ffill().iloc[-1])
    posts = {g: fit_posterior(index[g].dropna().loc[:date], g, n=4000, seed=0) for g in GPUS}
    try:
        ornn = store.read_frame("benchmarks")
    except FileNotFoundError:
        ornn = pd.DataFrame()
    figdir.mkdir(parents=True, exist_ok=True)
    return Context(index, ornn, rates, discount, date, posts, figdir)


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------
def md_table(df: pd.DataFrame, fmt: dict[str, str] | None = None, index: bool = True) -> str:
    fmt = fmt or {}
    cols = ([df.index.name or ""] if index else []) + list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)] if index else []
        for c in df.columns:
            v = row[c]
            f = fmt.get(c)
            cells.append(f.format(v) if f and pd.notna(v) and not isinstance(v, str) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _save(fig, ctx: Context, name: str) -> str:
    path = ctx.figdir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"docs/figures/{name}.png"


def _label_end(ax, x, y, text, color=INK_2, dx=4):
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, 0),
        textcoords="offset points",
        va="center",
        fontsize=8.5,
        color=color,
    )


# ---------------------------------------------------------------------------------------------
# 1. Structural cost stack and curve
# ---------------------------------------------------------------------------------------------
def cost_table(ctx: Context) -> pd.DataFrame:
    rows = {}
    for g in GPUS:
        lc = levelised_cost(central_params(g))
        spot = float(ctx.hist(g).iloc[-1])
        rows[g] = {
            "capital": float(lc.capital),
            "facility": float(lc.facility),
            "energy": float(lc.energy),
            "other opex": float(lc.other),
            "levelised cost": float(lc.total),
            "anchor (cost + margin)": float(ctx.posts[g].prior.anchor.mean()),
            "index today": spot,
            "index / cost": spot / float(lc.total),
        }
    return pd.DataFrame(rows).T.rename_axis("GPU")


def fig_cost(ctx: Context) -> str:
    apply_style()
    tab = cost_table(ctx)
    comps = ["capital", "facility", "energy", "other opex"]
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    for y, g in enumerate(GPUS[::-1]):
        left = 0.0
        for k, c in enumerate(comps):
            v = tab.loc[g, c]
            ax.barh(
                y,
                v,
                left=left,
                color=PALETTE[k],
                height=0.5,
                edgecolor=SURFACE,
                linewidth=1.5,
                label=c if y == 0 else None,
            )
            left += v
        spot = tab.loc[g, "index today"]
        ax.plot([spot, spot], [y - 0.34, y + 0.34], color=INK, lw=2)
        ax.annotate(
            f"index ${spot:.2f}",
            (spot, y + 0.34),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=INK_2,
        )
        ax.annotate(
            f"cost ${left:.2f}",
            (left, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK_2,
        )
    ax.set_yticks(range(len(GPUS)), GPUS[::-1])
    ax.set_ylim(-0.5, len(GPUS) - 0.1)
    ax.set_xlabel("$ per sold GPU-hour (central assumptions)")
    _dollars(ax, "x")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(tab["index today"].max(), tab["levelised cost"].max()) * 1.18)
    ax.set_title("Market price vs the cost of supplying a GPU-hour")
    ax.legend(ncol=4, loc="lower right", bbox_to_anchor=(1.0, -0.42))
    return _save(fig, ctx, "cost_stack")


def fig_fan(ctx: Context) -> str:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ts = np.linspace(0.0, 2.0, 41)
    for ax, g in zip(axes, GPUS, strict=True):
        post = ctx.posts[g]
        h = ctx.hist(g)
        col = gpu_color(g)
        dates = [ctx.date + pd.Timedelta(days=365.25 * t) for t in ts]
        q = np.array([post.curve.spot_quantiles(t, [0.05, 0.25, 0.5, 0.75, 0.95]) for t in ts])
        ax.fill_between(dates, q[:, 0], q[:, 4], color=col, alpha=0.14, linewidth=0)
        ax.fill_between(dates, q[:, 1], q[:, 3], color=col, alpha=0.26, linewidth=0)
        hg = with_gaps(h)
        ax.plot(hg.index, hg.values, color=INK, lw=1.4)
        fwd = [post.curve.forward(t) for t in ts]
        prior = [post.prior.forward(t) for t in ts]
        ax.plot(dates, fwd, color=col, lw=2.0)
        ax.plot(dates, prior, color=MUTED, lw=1.2, ls=(0, (4, 3)))
        ax.plot(dates, [h.iloc[-1]] * len(ts), color=MUTED, lw=1.0, ls=(0, (1, 2)))
        ax.axvline(ctx.date, color=GRID, lw=1)
        ax.set_title(f"{g}: index, forward curve and 90% range")
        _dollars(ax)
        _label_end(ax, dates[-1], fwd[-1], "posterior forward", INK_2)
        _label_end(ax, dates[-1], prior[-1], "prior only", MUTED)
        _label_end(ax, dates[-1], h.iloc[-1], "flat", MUTED)
        ax.margins(x=0.02)
        ax.set_xlim(h.index[0], dates[-1] + pd.Timedelta(days=170))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.tick_params(axis="x", labelsize=8.5)
    fig.text(
        0.01,
        -0.02,
        "Black: composite price index. Shaded: 50% and 90% range of the spot price "
        "(parameter plus price uncertainty). Dashed: prior-only forward; dotted: flat.",
        fontsize=8.5,
        color=INK_2,
    )
    fig.tight_layout()
    return _save(fig, ctx, "forward_curve_fan")


def posterior_table(ctx: Context) -> pd.DataFrame:
    rows = []
    for g in GPUS:
        s = ctx.posts[g].summary()
        for key in ("kappa", "delta", "margin"):
            pr = s[(s.parameter == key) & (s.distribution == "prior")].iloc[0]
            po = s[(s.parameter == key) & (s.distribution == "posterior")].iloc[0]
            rows.append(
                {
                    "GPU": g,
                    "parameter": key,
                    "prior mean": pr["mean"],
                    "posterior mean": po["mean"],
                    "prior sd": pr["sd"],
                    "posterior sd": po["sd"],
                }
            )
    return pd.DataFrame(rows).set_index("GPU")


def curve_table(ctx: Context) -> pd.DataFrame:
    rows = []
    for g in GPUS:
        c = ctx.posts[g].curve
        for t, lab in ((0.25, "3M"), (0.5, "6M"), (1.0, "1Y"), (2.0, "2Y")):
            lo, hi = c.forward_quantiles(t, [0.05, 0.95])
            rows.append(
                {
                    "GPU": g,
                    "tenor": lab,
                    "forward": c.forward(t),
                    "vs spot": c.forward(t) / c.spot - 1,
                    "param. 5%": lo,
                    "param. 95%": hi,
                    "prior-only": ctx.posts[g].prior.forward(t),
                }
            )
    return pd.DataFrame(rows).set_index("GPU")


def fig_spread(ctx: Context) -> str:
    """Substitution bound: H100 price vs the B200 price per unit of H100-equivalent work."""
    apply_style()
    h100, b200 = ctx.hist("H100_SXM"), ctx.hist("B200")
    both = pd.concat([h100, b200], axis=1, keys=["h", "b"]).dropna()
    pr = PERF_RATIO_B200_VS_H100
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    both = both.resample("D").last().pipe(lambda d: d.where(d.ffill(limit=3).notna()))
    ax.fill_between(
        both.index,
        both.b / pr.high,
        both.b / pr.low,
        color=gpu_color("B200"),
        alpha=0.2,
        linewidth=0,
    )
    ax.plot(both.index, both.b / pr.central, color=gpu_color("B200"), lw=1.6)
    ax.plot(both.index, both.h, color=gpu_color("H100_SXM"), lw=1.8)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    _dollars(ax)
    ax.set_title("H100 price vs the price its B200 substitute implies")
    _label_end(ax, both.index[-1], both.h.dropna().iloc[-1], "H100 index")
    _label_end(
        ax,
        both.index[-1],
        both.b.dropna().iloc[-1] / pr.central,
        f"B200 / {pr.central:g} (range {pr.low:g}-{pr.high:g})",
    )
    ax.set_xlim(both.index[0], both.index[-1] + pd.Timedelta(days=45))
    ax.set_ylabel("$ per H100-equivalent GPU-hour")
    return _save(fig, ctx, "generational_spread")


# ---------------------------------------------------------------------------------------------
# 2. Credit-linked trades
# ---------------------------------------------------------------------------------------------
def credit_market(ctx: Context, g: str = "H100_SXM") -> CreditMarket:
    post = ctx.posts[g]
    v = posterior_vol(post)
    return CreditMarket(
        post.curve, v.sigma, v.kappa, ctx.discount.with_spread(0.0), n_paths=20_000, seed=0
    )


def counterparties(ctx: Context):
    return {n: make_counterparty(n, ctx.discount) for n in COUNTERPARTIES}


def hazard_table(ctx: Context) -> pd.DataFrame:
    rows = []
    for n, cp in counterparties(ctx).items():
        rows.append(
            {
                "counterparty": n,
                "1y CDS (bp)": CDS_ARCHETYPES[n][1] * 1e4,
                "3y CDS (bp)": CDS_ARCHETYPES[n][3] * 1e4,
                "5y CDS (bp)": CDS_ARCHETYPES[n][5] * 1e4,
                "5y default prob.": 1 - float(cp.hazard.survival(5.0)),
                "hazard 0-1y": cp.hazard.hazards[0],
                "hazard 3-5y": cp.hazard.hazards[2],
            }
        )
    return pd.DataFrame(rows).set_index("counterparty")


def fig_cva(ctx: Context) -> tuple[str, pd.DataFrame]:
    apply_style()
    mkt = credit_market(ctx)
    cps = counterparties(ctx)
    rhos = np.linspace(-0.4, 0.8, 7)
    fwd_mid = mkt.forward_curve.forward(0.5)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), sharey=True)
    records = []
    for ax, (title, kind) in zip(
        axes,
        (
            ("Prepaid capacity: we pay upfront", "prepaid"),
            ("Take-or-pay offtake: they owe us", "top"),
        ),
        strict=True,
    ):
        for n, cp in cps.items():
            vals = []
            for r in rhos:
                if kind == "prepaid":
                    d = prepaid_discount(100_000, 12, cp, mkt, r)
                    vals.append(d["discount_pct"])
                else:
                    res = strip_cva(ForwardStrip(100_000, 12, fwd_mid, -1), cp, mkt, r)
                    vals.append(res.cva_per_hour / fwd_mid)
                records.append(
                    {
                        "structure": title.split(":")[0],
                        "counterparty": n,
                        "rho": r,
                        "cost % of mid": vals[-1],
                    }
                )
            ax.plot(rhos, np.array(vals) * 100, color=CP_COLORS[n], lw=1.8)
            nudge = {"Hyperscaler (IG)": -7, "Neocloud (HY)": 0, "AI lab (unrated)": 7}[n]
            ax.annotate(
                n,
                (rhos[-1], vals[-1] * 100),
                xytext=(4, nudge),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
                color=INK_2,
            )
        ax.set_title(title)
        ax.set_xlabel("wrong-way correlation  (default likelier when our exposure is high)")
        ax.set_xlim(rhos[0], rhos[-1] + 0.55)
        ax.axvline(0, color=GRID, lw=1)
    axes[0].set_ylabel("credit cost, % of price (12-month strip)")
    fig.tight_layout()
    df = pd.DataFrame(records)
    return _save(fig, ctx, "cva_wrong_way"), df


def credit_summary(ctx: Context) -> pd.DataFrame:
    mkt = credit_market(ctx)
    rows = []
    for n, cp in counterparties(ctx).items():
        pp = prepaid_discount(100_000, 12, cp, mkt, 0.3)
        top = strip_cva(ForwardStrip(100_000, 12, mkt.forward_curve.forward(0.5), -1), cp, mkt, 0.3)
        ln = loan_fair_spread(GPULoan(ltv=0.7), cp, mkt, 0.3)
        rows.append(
            {
                "counterparty": n,
                "prepaid 12M: fair price": pp["fair_prepay_price"],
                "prepaid discount": pp["discount_pct"],
                "take-or-pay CVA ($/hr)": top.cva_per_hour,
                "GPU loan 70% LTV: fair spread (bp)": ln["spread"] * 1e4,
                "unsecured CDS-implied (bp)": cp.spread(3.0) * 1e4,
            }
        )
    return pd.DataFrame(rows).set_index("counterparty")


def fig_loan(ctx: Context) -> tuple[str, pd.DataFrame]:
    apply_style()
    mkt = credit_market(ctx)
    cp = make_counterparty("Neocloud (HY)", ctx.discount)
    ltvs = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rhos = [0.0, 0.2, 0.4, 0.6, 0.8]
    grid = np.array(
        [
            [loan_fair_spread(GPULoan(ltv=lv), cp, mkt, r)["spread"] * 1e4 for r in rhos]
            for lv in ltvs
        ]
    )
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.imshow(np.log10(grid + 1), cmap=cmap, aspect="auto", origin="lower")
    ax.grid(False)
    for i in range(len(ltvs)):
        for j in range(len(rhos)):
            dark = np.log10(grid[i, j] + 1) > np.log10(grid + 1).mean()
            ax.text(
                j,
                i,
                "<1" if grid[i, j] < 0.5 else f"{grid[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=9,
                color="#ffffff" if dark else INK,
            )
    ax.set_xticks(range(len(rhos)), [f"{r:.1f}" for r in rhos])
    ax.set_yticks(range(len(ltvs)), [f"{lv:.0%}" for lv in ltvs])
    ax.set_xlabel("wrong-way correlation between GPU price and default")
    ax.set_ylabel("loan-to-value at origination")
    ax.set_title("Fair credit spread (bp) on a 3-year GPU-backed loan")
    for s in ax.spines.values():
        s.set_visible(False)
    df = pd.DataFrame(
        grid, index=[f"{lv:.0%}" for lv in ltvs], columns=[f"rho {r:.1f}" for r in rhos]
    )
    df.index.name = "LTV"
    return _save(fig, ctx, "gpu_loan_spread"), df


# ---------------------------------------------------------------------------------------------
# 3. Bid / offer
# ---------------------------------------------------------------------------------------------
def make_quote(
    ctx: Context,
    cp: str,
    structure: str,
    months: int = 12,
    wrong_way: float = 0.3,
    gpu: str = "H100_SXM",
    hours: float = 1_000_000,
):
    post = ctx.posts[gpu]
    v = posterior_vol(post)
    req = BilateralRequest(gpu, hours, months, cp, wrong_way=wrong_way)
    return quote(req, structure, post.curve, v.sigma, v.kappa, ctx.discount)


def fig_ladder(ctx: Context) -> tuple[str, pd.DataFrame]:
    apply_style()
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    names = ["model uncertainty", "risk capital", "credit (CVA)", "minimum edge"]
    colors = PALETTE[:4]
    rows = []
    for y, cp in enumerate(COUNTERPARTIES[::-1]):
        q = make_quote(ctx, cp, "prepaid")
        for side, sgn in ((q.bid, -1), (q.offer, 1)):
            left = 0.0
            for k, n in enumerate(names):
                v = side.components[n] * sgn
                ax.barh(
                    y,
                    v,
                    left=left,
                    color=colors[k],
                    height=0.46,
                    edgecolor=SURFACE,
                    linewidth=1.5,
                    label=n if (y == 0 and sgn == 1) else None,
                )
                left += v
        ax.annotate(
            f"${q.bid.price:.2f}",
            (-(q.mid - q.bid.price), y),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8.5,
            color=INK_2,
        )
        ax.annotate(
            f"${q.offer.price:.2f}",
            (q.offer.price - q.mid, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK_2,
        )
        rows.append(
            {
                "counterparty": cp,
                "bid": q.bid.price,
                "mid": q.mid,
                "offer": q.offer.price,
                "width % of mid": q.spread / q.mid,
            }
        )
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(range(len(COUNTERPARTIES)), COUNTERPARTIES[::-1])
    ax.set_xlabel("$ per GPU-hour away from the structural mid")
    ax.grid(axis="y", visible=False)
    ax.set_title("Bid/offer on a 12-month prepaid H100 request (1M GPU-hrs a month)")
    ax.set_xlim(-0.55, 0.42)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.5))
    df = pd.DataFrame(rows).set_index("counterparty")
    return _save(fig, ctx, "bid_offer_ladder"), df


def quote_surface(ctx: Context) -> tuple[str, pd.DataFrame]:
    apply_style()
    tenors = [3, 6, 12, 24]
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    recs = []
    for cp in COUNTERPARTIES:
        w = []
        for m in tenors:
            q = make_quote(ctx, cp, "prepaid", months=m)
            w.append(q.spread / q.mid)
            recs.append({"counterparty": cp, "months": m, "width % of mid": w[-1]})
        ax.plot(
            tenors,
            np.array(w) * 100,
            color=CP_COLORS[cp],
            lw=1.8,
            marker="o",
            ms=5,
            markeredgecolor=SURFACE,
        )
        _label_end(ax, tenors[-1], w[-1] * 100, cp)
    ax.set_xticks(tenors)
    ax.set_xlim(2, 33)
    ax.set_xlabel("delivery horizon, months")
    ax.set_ylabel("bid-offer width, % of mid")
    ax.set_title("Bid-offer width widens with tenor and with counterparty credit")
    return _save(fig, ctx, "bid_offer_tenor"), pd.DataFrame(recs)


# ---------------------------------------------------------------------------------------------
# 4. Structured notes on the new curve
# ---------------------------------------------------------------------------------------------
def _builder(ctx: Context, series: pd.Series, cfg: dict, model: str, gpu: str):
    mc = dict(cfg.get("market", {}))
    mc.update(
        forward_model=model,
        structural_gpu=gpu if model == "structural" else None,
        structural_draws=3000,
        rate=None,
    )
    return MarketBuilder(series, MarketModelConfig(**mc), ctx.rates)


def notes_table(ctx: Context, gpu: str = "H100_SXM") -> pd.DataFrame:
    series = ctx.hist(gpu)
    rows = []
    for path in sorted((PROJECT_ROOT / "configs" / "products").glob("*.yaml")):
        cfg = load_yaml(path)
        prod = build_product(cfg["product"])
        solve_for = cfg.get("solve_for", prod.default_solve_for)
        target = cfg.get("issue_price", 1.0) - cfg.get("issuer_margin", 0.0)
        out = {"note": prod.kind, "solved term": solve_for}
        for model in ("flat", "schwartz", "structural"):
            b = _builder(ctx, series, cfg, model, gpu)
            st, s0 = b.state(ctx.date), b.spot(ctx.date)
            try:
                out[model] = getattr(solve(prod, st, s0, target=target, field=solve_for), solve_for)
            except SolveError:
                out[model] = float("nan")  # no value of the term reaches the target price
            if model == "structural":
                out["forward/spot at maturity"] = st.forward(prod.tenor) / s0
        rows.append(out)
    return pd.DataFrame(rows).set_index("note")


def participation_distribution(ctx: Context, gpu: str = "H100_SXM", n: int = 300):
    """Solve the capital-protected note's participation under each sampled posterior member."""
    cfg = load_yaml("products/capital_protected.yaml")
    prod = build_product(cfg["product"])
    post = ctx.posts[gpu]
    s0 = float(ctx.hist(gpu).iloc[-1])
    target = 1.0 - cfg.get("issuer_margin", 0.0)
    disc = ctx.discount.with_spread(cfg["market"].get("issuer_spread", 0.01))
    rng = np.random.default_rng(0)
    members = rng.choice(len(post.curve.weights), size=n, p=post.curve.weights)
    vals, unfundable = [], 0
    for i in members:
        st = structural_state(post, ctx.date, disc, member=int(i))
        try:
            vals.append(solve(prod, st, s0, target=target).participation)
        except SolveError:
            unfundable += 1
    base = solve(prod, structural_state(post, ctx.date, disc), s0, target=target).participation
    return np.array(vals), base, prod, unfundable / n


def issuer_table(ctx: Context, gpu: str = "H100_SXM") -> pd.DataFrame:
    cfg = load_yaml("products/capital_protected.yaml")
    prod = build_product(cfg["product"])
    post = ctx.posts[gpu]
    s0 = float(ctx.hist(gpu).iloc[-1])
    rows = []
    for name, cp in [("Risk-free", None), *counterparties(ctx).items()]:
        spread = 0.0 if cp is None else issuer_spread(cp, prod.tenor)
        disc = ctx.discount.with_spread(spread)
        st = structural_state(post, ctx.date, disc)
        p = solve(prod, st, s0, target=0.99).participation
        rows.append({"issuer": name, "funding spread (bp)": spread * 1e4, "participation": p})
    return pd.DataFrame(rows).set_index("issuer")


def fig_participation(ctx: Context) -> tuple[str, dict]:
    apply_style()
    vals, base, prod, unfund = participation_distribution(ctx)
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    clip = 5.0
    ax.hist(
        np.minimum(vals, clip),
        bins=np.linspace(0, clip, 26),
        color=PALETTE[0],
        edgecolor=SURFACE,
        linewidth=1.2,
    )
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0%}+" if v >= clip else f"{v:.0%}")
    )
    lo, hi = np.quantile(vals, [0.05, 0.95])
    ax.axvline(base, color=INK, lw=1.8)
    ax.annotate(
        f"posterior curve: {base:.0%}",
        (base, ax.get_ylim()[1]),
        xytext=(6, -10),
        textcoords="offset points",
        fontsize=8.5,
        color=INK_2,
    )
    ax.set_xlabel("solved participation (100%-protected 6M note, issued at 99%)")
    ax.set_ylabel("posterior draws")
    ax.set_title("Participation rate under parameter uncertainty")
    ax.grid(axis="x", visible=False)
    return _save(fig, ctx, "note_participation"), {"base": base, "lo": lo, "hi": hi, "unf": unfund}


# ---------------------------------------------------------------------------------------------
# 5. Governance
# ---------------------------------------------------------------------------------------------
def fig_tornado(ctx: Context, gpu: str = "H100_SXM") -> tuple[str, pd.DataFrame]:
    apply_style()
    tor = tornado(gpu, float(ctx.hist(gpu).iloc[-1]), ctx.posts[gpu].sigma_hat, ctx.discount)
    base = tor.attrs["base"]
    top = tor.head(8).iloc[::-1]
    basis_color = {"sourced": PALETTE[0], "derived": PALETTE[1], "judgment": PALETTE[2]}
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for y, (_, r) in enumerate(top.iterrows()):
        lo, hi = sorted([r["at low"], r["at high"]])
        ax.barh(
            y,
            hi - lo,
            left=lo,
            color=basis_color[r["basis"]],
            height=0.55,
            edgecolor=SURFACE,
            linewidth=1.5,
        )
    ax.axvline(base, color=INK, lw=1.4)
    ax.set_yticks(
        range(len(top)),
        ["\n".join(textwrap.wrap(a.split(" (")[0], 30)) for a in top["assumption"]],
        fontsize=8.5,
    )
    ax.set_xlabel("12-month strip price, $ per GPU-hour (each assumption at its low and high)")
    _dollars(ax, "x")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in basis_color.values()]
    ax.legend(handles, [f"{b} input" for b in basis_color], loc="lower right")
    ax.set_title(f"What moves the {gpu} 12-month price")
    return _save(fig, ctx, "tornado"), tor


def fig_walkforward(ctx: Context) -> tuple[str, pd.DataFrame]:
    apply_style()
    frames = []
    for g in GPUS:
        s = walk_forward_summary(walk_forward(ctx.hist(g).loc[: ctx.date], g))
        s.insert(0, "GPU", g)
        frames.append(s.reset_index())
    df = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(df))
    w = 0.36
    ax.bar(
        x - w / 2,
        df["RMSE structural"] * 100,
        w,
        color=PALETTE[0],
        edgecolor=SURFACE,
        linewidth=1.5,
        label="structural curve",
    )
    ax.bar(
        x + w / 2,
        df["RMSE flat"] * 100,
        w,
        color=PALETTE[1],
        edgecolor=SURFACE,
        linewidth=1.5,
        label="flat (no change)",
    )
    ax.set_xticks(x, [f"{r.GPU}\n{r.horizon} days" for r in df.itertuples()], fontsize=8.5)
    ax.set_ylabel("forecast RMSE, % (log error)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Walk-forward forecast error: structural curve vs 'no change'")
    ax.legend()
    return _save(fig, ctx, "walk_forward"), df.set_index("GPU")


# ---------------------------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------------------------
def register_md(gpu: str) -> pd.DataFrame:
    rows = []
    for a in register(gpu):
        src = f"[{a.source}]({a.url})" if a.url else a.source
        rows.append(
            {
                "assumption": a.label,
                "unit": a.unit,
                "low / central / high": f"{a.low:g} / {a.central:g} / {a.high:g}",
                "basis": a.basis,
                "confidence": a.confidence,
                "source": src,
                "revise if": a.revise_if,
            }
        )
    return pd.DataFrame(rows).set_index("assumption")


def build(out_root: Path | None = None, update_readme: bool = True) -> str:
    root = out_root or PROJECT_ROOT
    ctx = build_context(root / "docs" / "figures")
    g = "H100_SXM"
    post = ctx.posts[g]

    p_cost, p_fan, p_spread = fig_cost(ctx), fig_fan(ctx), fig_spread(ctx)
    p_cva, cva_df = fig_cva(ctx)
    p_loan, loan_df = fig_loan(ctx)
    p_lad, lad_df = fig_ladder(ctx)
    p_surf, _ = quote_surface(ctx)
    p_part, part = fig_participation(ctx)
    p_tor, tor = fig_tornado(ctx, g)
    p_wf, wf = fig_walkforward(ctx)

    checks = run_checks(post, ctx.discount)
    ipv = ipv_table(ctx.hist(g), post, ctx.discount)
    reserve = model_risk_reserve(ipv)
    rank = rank_importance(post, ctx.discount).head(5)
    q = make_quote(ctx, "Neocloud (HY)", "prepaid")
    notes = notes_table(ctx)
    iss = issuer_table(ctx)

    d = ctx.date.date()
    parts: list[str] = []
    a = parts.append
    a(
        f"_Generated by `compute-options showcase` on data to {d}. Every number below comes from "
        "the code in this repo._\n"
    )

    a("### 1. Forward curve with no forward market\n")
    a(
        "No compute forward trades, and a GPU-hour can't be stored, so spot and forward are not "
        "linked by arbitrage. The curve is built from the economics of supply: the price reverts "
        "to "
        "the cost of running a GPU (plus a margin), that cost level drifts down as newer "
        "generations arrive, and the rate at which a scarcity premium decays is learned from the "
        "index. Every input is a range in an [assumptions register](#assumptions-register), the "
        "ensemble of curves is reweighted by the observed index (Bayesian update), and the pricing "
        "stack uses the posterior mean.\n"
    )
    a(f"![Cost stack]({p_cost})\n")
    a(
        md_table(
            cost_table(ctx),
            {
                c: "{:.2f}"
                for c in (
                    "capital",
                    "facility",
                    "energy",
                    "other opex",
                    "levelised cost",
                    "anchor (cost + margin)",
                    "index today",
                )
            }
            | {"index / cost": "{:.1f}x"},
        )
        + "\n"
    )
    a(f"![Forward curves]({p_fan})\n")
    a(
        md_table(
            curve_table(ctx),
            {
                "forward": "${:.2f}",
                "vs spot": "{:+.0%}",
                "param. 5%": "${:.2f}",
                "param. 95%": "${:.2f}",
                "prior-only": "${:.2f}",
            },
        )
        + "\n"
    )
    a(
        f"What the data changed (prior to posterior; likelihood power {post.beta:.2f}, effective "
        f"sample {post.ess:,.0f} of {len(post.curve.weights):,} draws):\n"
    )
    a(
        md_table(
            posterior_table(ctx),
            {c: "{:.2f}" for c in ("prior mean", "posterior mean", "prior sd", "posterior sd")},
        )
        + "\n"
    )
    a(f"![H100 vs B200]({p_spread})\n")

    a("### 2. Credit-linked transactions\n")
    a(
        "Counterparty default is modelled with hazard-rate curves bootstrapped from CDS spreads. "
        "Exposure comes from Monte Carlo paths of the compute price, and default is linked to the "
        "price through a Gaussian copula, so the same code prices **wrong-way risk** (default "
        "likelier when we are most exposed). Three trades use it:\n"
    )
    a(
        "- **Prepaid capacity**: pay upfront for GPU-hours; the supplier's default strands the "
        "undelivered hours.\n"
        "- **Take-or-pay offtake**: a counterparty commits to buy a volume at a fixed price; we "
        "are "
        "exposed when the market is below their price.\n"
        "- **GPU-backed loan**: collateral value moves with the compute price, so default and "
        "recovery are linked.\n"
    )
    a(
        md_table(
            hazard_table(ctx),
            {
                "1y CDS (bp)": "{:.0f}",
                "3y CDS (bp)": "{:.0f}",
                "5y CDS (bp)": "{:.0f}",
                "5y default prob.": "{:.1%}",
                "hazard 0-1y": "{:.3f}",
                "hazard 3-5y": "{:.3f}",
            },
        )
        + "\n"
    )
    a(
        "Only the neocloud 5-year level (about 855 bp, August 2026) is from a public source; the "
        "other two names and all curve shapes are illustrative. Recovery is a standard 40%.\n"
    )
    a(f"![CVA]({p_cva})\n")
    a(
        md_table(
            credit_summary(ctx),
            {
                "prepaid 12M: fair price": "${:.3f}",
                "prepaid discount": "{:.1%}",
                "take-or-pay CVA ($/hr)": "${:.4f}",
                "GPU loan 70% LTV: fair spread (bp)": "{:.0f}",
                "unsecured CDS-implied (bp)": "{:.0f}",
            },
        )
        + "\n"
    )
    a("Wrong-way correlation 0.3, 12-month strips of 100,000 GPU-hours a month, 3-year loan.\n")
    a(f"![GPU loan]({p_loan})\n")
    a(
        "The secured loan is cheap against unsecured CDS at low loan-to-value, then reprices "
        "sharply as the collateral cushion thins, and the correlation matters more the higher the "
        "LTV.\n"
    )

    a("### 3. Reservation bid and offer\n")
    a(
        "A bilateral request has no market price to refer to. The quote is built from the risks "
        "being carried and each piece is reported: structural mid, a charge for model "
        "uncertainty (width of the posterior), a capital charge on the 99% VaR of holding the "
        "position unhedged (no listed contract to hedge with), credit (CVA with wrong-way risk), "
        "and a minimum edge. Charges are policy parameters "
        "(`quoting.engine.HURDLE`, `MODEL_RISK_MULT`, `MIN_EDGE`), not estimates.\n"
    )
    a(f"![Ladder]({p_lad})\n")
    a(
        md_table(
            lad_df,
            {"bid": "${:.3f}", "mid": "${:.3f}", "offer": "${:.3f}", "width % of mid": "{:.1%}"},
        )
        + "\n"
    )
    a("A quote memo, generated from the same numbers:\n")
    a("> " + q.memo() + "\n")
    a(f"![Tenor]({p_surf})\n")

    a("### 4. Structured notes on the same stack\n")
    a(
        "The six structured notes price off the same "
        "`MarketState`, so they pick up the structural curve, the posterior volatility and the "
        "issuer's hazard curve with no change to the products. Solved terms on "
        f"{g} ({d}):\n"
    )
    a(
        md_table(
            notes,
            {
                "flat": "{:.3f}",
                "schwartz": "{:.3f}",
                "structural": "{:.3f}",
                "forward/spot at maturity": "{:.2f}",
            },
        )
        + "\n"
    )
    a(
        (
            "Cells show nan where no value of the term reaches the issue price. "
            if notes.isna().any().any()
            else ""
        )
        + "Backwardation lowers the "
        "forward the options are priced on, which moves every solved term (a call struck at spot "
        "is out of the money against a lower forward, so it takes more participation to fund the "
        "same budget). Issuer credit matters too. Capital-protected 6M note, funding spread taken "
        "from each issuer's hazard curve:\n"
    )
    a(md_table(iss, {"funding spread (bp)": "{:.0f}", "participation": "{:.1%}"}) + "\n")
    a(f"![Participation]({p_part})\n")
    a(
        f"Across posterior draws the solved participation runs from {part['lo']:.0%} to "
        f"{part['hi']:.0%} (5th to 95th percentile) around {part['base']:.0%} on the "
        "posterior-mean curve, and in "
        f"{part['unf']:.0%} of draws no participation reaches the "
        "issue price (the draw's forward is too backwardated to fund the protection). That spread "
        "is parameter risk the issuer carries and a quote would price in.\n"
    )

    a("### 5. Model governance\n")
    a("**Numerical checks** (run in the test suite as well):\n")
    a(
        md_table(
            pd.DataFrame(
                [
                    {"check": c.name, "result": "pass" if c.passed else "FAIL", "detail": c.detail}
                    for c in checks
                ]
            ).set_index("check")
        )
        + "\n"
    )
    a(
        "**Independent price verification**: the 12-month strip under four unrelated curve "
        "models. The spread across the three that use the index is the model-risk reserve "
        f"(${reserve:.2f} per GPU-hour, one standard deviation).\n"
    )
    a(
        md_table(
            ipv.set_index("model"),
            {"strip price": "${:.3f}", "1y forward": "${:.3f}", "vs structural": "{:+.1%}"},
        )
        + "\n"
    )
    a(f"![Tornado]({p_tor})\n")
    a(
        "The biggest movers are judgment inputs with low confidence (rent decay, margin, "
        "utilisation), and not the sourced ones. Rank correlation across the prior says the same "
        "(top five: " + ", ".join(f"{r.key} {r._3:+.2f}" for r in rank.itertuples()) + ").\n"
    )
    a(
        "**Walk-forward test.** At each past origin the model is refit on the data available at "
        "that date and its forecast is compared with the index 30 and 60 days later.\n"
    )
    a(f"![Walk-forward]({p_wf})\n")
    a(
        md_table(
            wf,
            {
                "horizon": "{:.0f} days",
                "origins": "{:.0f}",
                "bias structural": "{:+.3f}",
                "bias flat": "{:+.3f}",
                "RMSE structural": "{:.3f}",
                "RMSE flat": "{:.3f}",
                "90% band coverage": "{:.0%}",
            },
        )
        + "\n"
    )
    loses = int((wf["RMSE structural"] > wf["RMSE flat"]).sum())
    a(
        f"**Interpretation.** Over this sample the index rose, the structural curve forecasts "
        f"decay, and so it is biased low and loses to 'no change' in {loses} of {len(wf)} "
        f"GPU-horizon cases. The 90% band covers {wf['90% band coverage'].min():.0%}-"
        f"{wf['90% band coverage'].max():.0%} of outcomes (a few origins, overlapping windows, so "
        "the coverage figures are rough). Seven months of one regime can't validate a curve meant "
        "to describe 1-3 "
        "years. The test points to the cap on the long-run margin as the assumption doing the "
        "damage (see the revision log). Until a listed contract produces settlement prices, the "
        "model-uncertainty charge in the quote covers this risk.\n"
    )
    a("**Revision log**\n")
    for when, what, why in REVISIONS:
        a(f"- **{when}**: {what}. {why}\n")

    a("#### Assumptions register\n")
    a(
        "`basis`: *sourced* = from a public source (mostly press and vendor posts, not filings; "
        "figures gathered October 2026), *derived* = computed from sourced numbers, *judgment* = "
        "no evidence found, range left wide. `confidence` is how far to trust the central value. "
        "Ranges are the model's priors (triangular).\n"
    )
    for gpu in GPUS:
        a(f"<details><summary>{gpu}</summary>\n")
        a(md_table(register_md(gpu)) + "\n")
        a("</details>\n")
    a("Credit inputs: " + "; ".join(f"[{t}]({u})" for t, u in CREDIT_SOURCES.values()) + ".\n")

    block = "\n".join(parts)
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "results.md").write_text(block)
    if update_readme:
        readme = root / "README.md"
        text = readme.read_text()
        if START in text and END in text:
            head, rest = text.split(START, 1)
            _, tail = rest.split(END, 1)
            readme.write_text(f"{head}{START}\n\n{block}\n{END}{tail}")
    return block


def main() -> None:
    build()


if __name__ == "__main__":
    main()
