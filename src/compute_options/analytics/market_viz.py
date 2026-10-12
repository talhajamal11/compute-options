"""Charts of GPU rental market prices for the data-viz notebooks.

Conventions (applied to every chart):

- Each GPU keeps one colour everywhere (``GPU_COLORS``), taken in fixed order from a
  colour-blind-validated categorical palette. Colour follows the GPU, never its rank.
- One y-axis per chart. GPUs at different price levels are compared on a common base (indexed
  to 100), not on two scales.
- Thin lines, recessive grid and axes, direct labels at line ends, text in ink colours (never
  in the series colour).
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from compute_options.data.store import ObservationStore
from compute_options.market.vol import DAYS_PER_YEAR

# Validated categorical palette (light mode), in its fixed order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Fixed GPU -> colour. The flagship data-centre GPUs take the first slots.
GPU_ORDER = ["H100_SXM", "H200", "B200", "A100_SXM_80GB", "B300", "L40S", "RTX_5090", "MI300X"]
GPU_COLORS = dict(zip(GPU_ORDER, PALETTE, strict=True))
OTHER = "#9a9994"

TIER_ORDER = ["on_demand", "secure", "community", "spot", "serverless"]


def gpu_color(gpu: str) -> str:
    return GPU_COLORS.get(gpu, OTHER)


def with_gaps(s: pd.Series, max_gap_days: int = 3) -> pd.Series:
    """Daily series with short gaps carried forward and gaps longer than ``max_gap_days`` left
    as NaN, so lines break instead of drawing a straight segment through missing data."""
    s = s.dropna()
    daily = s.resample("D").last()
    run = daily.isna().groupby(daily.notna().cumsum()).transform("sum")
    return daily.ffill().where(~(daily.isna() & (run > max_gap_days)))


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.6,
            "lines.solid_capstyle": "round",
            "font.size": 10,
            "figure.dpi": 110,
        }
    )


def _dollars(ax, axis: str = "y", decimals: int = 2) -> None:
    fmt = FuncFormatter(lambda v, _: f"${v:,.{decimals}f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _percent(ax, axis: str = "y") -> None:
    fmt = FuncFormatter(lambda v, _: f"{v:.0%}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------


class MarketData:
    """The aggregated frames the charts read, loaded once."""

    def __init__(self, store: ObservationStore | None = None):
        store = store or ObservationStore()
        self.index = store.read_frame("composite_index")
        self.level = store.read_frame("composite_level")
        self.n_providers = store.read_frame("composite_n_providers")
        self.providers = store.read_frame("provider_prices")
        self.benchmarks = store.read_frame("benchmarks")
        self.observations = store.read()

    def gpus(self, min_days: int = 60, charted_only: bool = True) -> list[str]:
        """GPUs with at least ``min_days`` of index history, flagship order first.

        ``charted_only`` keeps the GPUs that have a palette colour (at most 8), so every line
        on a chart stays identifiable; set False for tables.
        """
        have = [g for g in self.index.columns if self.index[g].notna().sum() >= min_days]
        ordered = [g for g in GPU_ORDER if g in have]
        return ordered if charted_only else ordered + sorted(g for g in have if g not in GPU_ORDER)

    def benchmark(self, gpu: str) -> pd.Series | None:
        col = f"ornn:{gpu}"
        return self.benchmarks[col].dropna() if col in self.benchmarks else None


def summary_table(md: MarketData, gpus: Sequence[str]) -> pd.DataFrame:
    """Latest price and changes per GPU, from the price index."""
    rows = []
    for g in gpus:
        s = md.index[g].dropna()
        daily = s.resample("D").last().ffill()
        last = daily.iloc[-1]

        def chg(days: int, daily: pd.Series = daily, last: float = last) -> str:
            past = daily.asof(daily.index[-1] - pd.Timedelta(days=days))
            return f"{last / past - 1:+.1%}" if pd.notna(past) else "n/a"

        r = np.log(daily).diff(7).dropna()
        bench = md.benchmark(g)
        rows.append(
            {
                "GPU": g,
                "index ($/GPU-hr)": f"${last:.2f}",
                "avg posted ($/GPU-hr)": f"${md.level[g].dropna().iloc[-1]:.2f}",
                "Ornn transacted": f"${bench.iloc[-1]:.2f}" if bench is not None else "-",
                "1M": chg(30),
                "3M": chg(91),
                "since start": f"{last / daily.iloc[0] - 1:+.1%}",
                "vol (weekly, ann.)": f"{r.std() * np.sqrt(DAYS_PER_YEAR / 7):.0%}",
                "providers today": int(md.n_providers[g].dropna().iloc[-1]),
                "history from": s.index[0].date(),
            }
        )
    return pd.DataFrame(rows).set_index("GPU")


# ---------------------------------------------------------------------------------------------
# Cross-GPU charts
# ---------------------------------------------------------------------------------------------


def plot_indexed(md: MarketData, gpus: Sequence[str], start: str | None = None):
    """All GPUs on one axis, each rebased to 100 at ``start`` (default: the earliest date).
    GPUs whose data begins after ``start`` are rebased at their own first day and dashed."""
    fig, ax = plt.subplots(figsize=(11, 5.2))
    series = {}
    base_date = pd.Timestamp(start) if start else min(md.index[x].first_valid_index() for x in gpus)
    late = []
    for g in gpus:
        s = with_gaps(md.index[g]).loc[base_date:]
        if s.dropna().empty:
            continue
        first = s.first_valid_index()
        s = 100 * s / s[first]
        series[g] = s
        is_late = first > base_date + pd.Timedelta(days=14)
        if is_late:
            late.append(f"{g} from {first:%d %b}")
        ax.plot(s.index, s.values, color=gpu_color(g), label=g, ls="--" if is_late else "-")
    ax.axhline(100, color=MUTED, lw=0.8)
    ax.set_ylabel("price index (start = 100)")
    title = f"GPU rental prices, rebased to 100 on {base_date:%d %b %Y}"
    ax.set_title(
        title + (f"  (dashed: data starts later - {', '.join(late)})" if late else ""), fontsize=11
    )
    ax.margins(x=0.01)
    ax.legend(loc="upper left", ncol=min(4, len(series)))
    fig.tight_layout()
    fig.subplots_adjust(right=0.86)
    for g, s in series.items():
        ax.annotate(
            f"{g}  {s.dropna().iloc[-1]:.0f}",
            xy=(s.dropna().index[-1], s.dropna().iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=INK_2,
            annotation_clip=False,
        )
    _declutter_end_labels(ax)
    return fig


def _declutter_end_labels(ax) -> None:
    """Spread overlapping end-of-line annotations vertically (in display space)."""
    texts = [t for t in ax.texts if hasattr(t, "xy")]
    if not texts:
        return
    ax.figure.canvas.draw()
    import matplotlib.dates as mdates

    def y_pixels(t) -> float:
        x, y = t.xy
        x = mdates.date2num(x) if isinstance(x, pd.Timestamp) else x
        return float(ax.transData.transform((x, y))[1])

    ys = sorted(((y_pixels(t), t) for t in texts), key=lambda p: p[0])
    min_gap = 13.0  # pixels
    placed: list[float] = []
    for y, t in ys:
        target = max(y, placed[-1] + min_gap) if placed else y
        placed.append(target)
        t.set_position((6, (target - y) * 72 / ax.figure.dpi))


def plot_price_panels(md: MarketData, gpus: Sequence[str], ncols: int = 3):
    """Small multiples: each GPU in $/GPU-hr on its own axis: price index, average posted
    price, and Ornn's transacted index where available."""
    n = len(gpus)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), sharex=True, squeeze=False
    )
    for ax, g in zip(axes.flat, gpus, strict=False):
        lv = md.level[g].dropna()
        ix = with_gaps(md.index[g])
        ax.plot(lv.index, lv.values, color=MUTED, lw=0.9, label="average posted")
        ax.plot(ix.index, ix.values, color=gpu_color(g), lw=1.8, label="price index")
        bench = md.benchmark(g)
        if bench is not None and len(bench) > 1:
            ax.plot(bench.index, bench.values, color=INK, lw=1.2, ls="--", label="Ornn transacted")
        ax.set_title(g, fontsize=10.5)
        _dollars(ax)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    handles = [
        plt.Line2D([], [], color=INK_2, lw=1.8, label="price index (GPU colour)"),
        plt.Line2D([], [], color=MUTED, lw=0.9, label="average posted price"),
        plt.Line2D([], [], color=INK, lw=1.2, ls="--", label="Ornn transacted index"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(
        "$ per GPU-hour", x=0.01, ha="left", y=1.0, fontsize=12, fontweight="bold", color=INK
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ax in axes.flat[:n]:
        ax.tick_params(axis="x", labelrotation=30)
    return fig


def plot_latest_prices(md: MarketData, gpus: Sequence[str]):
    """Latest price per GPU: index as a bar, the spread of provider prices as a range."""
    prov = md.providers
    last_date = prov["date"].max()
    recent = prov[prov["date"] > last_date - pd.Timedelta(days=7)]
    rows = []
    for g in gpus:
        p = recent[recent["instrument"] == g].groupby("provider")["price"].median()
        rows.append((g, md.index[g].dropna().iloc[-1], p.quantile(0.1), p.quantile(0.9), len(p)))
    df = pd.DataFrame(rows, columns=["gpu", "index", "p10", "p90", "n"]).sort_values("index")
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(df) + 1.2))
    y = np.arange(len(df))
    ax.barh(y, df["index"], height=0.55, color=[gpu_color(g) for g in df["gpu"]])
    ax.hlines(y, df["p10"], df["p90"], color=INK_2, lw=1.2)
    ax.scatter(df["p10"], y, marker="|", s=90, color=INK_2, zorder=3)
    ax.scatter(df["p90"], y, marker="|", s=90, color=INK_2, zorder=3)
    for yi, (_, r) in zip(y, df.iterrows(), strict=True):
        ax.text(
            max(r["p90"], r["index"]) * 1.02,
            yi,
            f"${r['index']:.2f}   ({r['n']} providers)",
            va="center",
            fontsize=8.5,
            color=INK_2,
        )
    ax.set_yticks(y, df["gpu"])
    ax.grid(axis="y", visible=False)
    _dollars(ax, "x", 0)
    ax.set_xlim(0, df["p90"].max() * 1.35)
    ax.set_xlabel("$ per GPU-hour")
    ax.set_title(
        f"Latest price index; whiskers = 10th-90th percentile of provider prices "
        f"(week to {last_date.date()})"
    )
    fig.tight_layout()
    return fig


def plot_coverage(md: MarketData, gpus: Sequence[str]):
    """Providers quoted per day for each GPU (sequential blue: darker = more providers)."""
    cov = md.providers.groupby(["date", "instrument"])["provider"].nunique().unstack()
    cov = cov.reindex(columns=list(gpus)).T
    cov = cov.reindex(columns=pd.date_range(cov.columns.min(), cov.columns.max()))
    from matplotlib.colors import BoundaryNorm, ListedColormap

    bounds = [0.5, 3, 5, 10, 15, 20, 30, 60]
    cmap = ListedColormap(SEQ_BLUE)
    cmap.set_bad(SURFACE)
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(gpus) + 1.3))
    x_edges = pd.date_range(cov.columns[0], periods=len(cov.columns) + 1, freq="D")
    im = ax.pcolormesh(
        x_edges,
        np.arange(len(gpus) + 1),
        np.ma.masked_invalid(cov.values),
        cmap=cmap,
        norm=BoundaryNorm(bounds, cmap.N),
        shading="flat",
    )
    ax.set_yticks(np.arange(len(gpus)) + 0.5, gpus)
    ax.invert_yaxis()
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.01, ticks=[1, 3, 5, 10, 15, 20, 30])
    cb.set_label("providers quoted", color=INK_2)
    cb.outline.set_visible(False)
    ax.set_title("Data coverage: providers quoted per day (blank = no data)")
    fig.tight_layout()
    return fig


def plot_rolling_vol(md: MarketData, gpus: Sequence[str], window_days: int = 60):
    """Annualised volatility of weekly returns over a trailing window."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for g in gpus:
        s = np.log(with_gaps(md.index[g]))
        r7 = s.diff(7)
        roll = r7.rolling(window_days, min_periods=window_days // 2)
        vol = roll.std() * np.sqrt(DAYS_PER_YEAR / 7)
        ax.plot(vol.index, vol.values, color=gpu_color(g), label=g)
    _percent(ax)
    ax.set_ylabel("annualised vol")
    ax.set_title(f"Volatility: weekly returns, trailing {window_days} days")
    ax.legend(loc="upper left", ncol=min(4, len(gpus)))
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------------
# Single-GPU charts
# ---------------------------------------------------------------------------------------------


def plot_gpu_prices(md: MarketData, gpu: str, highlight: Sequence[str] = ()):
    """Every provider's daily price (grey dots), the price index, average posted price and Ornn.
    ``highlight`` providers are drawn as labelled lines."""
    p = md.providers[md.providers["instrument"] == gpu]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.scatter(
        p["date"],
        p["price"],
        s=5,
        color="#c9c8c3",
        alpha=0.7,
        lw=0,
        label="individual provider prices",
    )
    for name in highlight:
        h = p[p["provider"] == name].set_index("date")["price"]
        if len(h):
            h = with_gaps(h)
            ax.plot(h.index, h.values, color=INK_2, lw=0.9, ls=":")
            h = h.dropna()
            ax.annotate(
                name,
                xy=(h.index[-1], h.iloc[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=8,
                color=INK_2,
                va="center",
            )
    lv = md.level[gpu].dropna()
    ix = with_gaps(md.index[gpu])
    ax.plot(lv.index, lv.values, color=MUTED, lw=1.0, label="average posted price")
    ax.plot(ix.index, ix.values, color=gpu_color(gpu), lw=2.2, label="price index")
    bench = md.benchmark(gpu)
    if bench is not None and len(bench) > 1:
        ax.plot(
            bench.index, bench.values, color=INK, lw=1.4, ls="--", label="Ornn transacted index"
        )
    hi = p["price"].quantile(0.97)
    ax.set_ylim(0, hi * 1.1)
    _dollars(ax)
    ax.set_ylabel("$ per GPU-hour")
    ax.set_title(
        f"{gpu}: provider prices and indices (dots above the "
        f"97th percentile, ${hi:.2f}, are off-scale)"
    )
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    return fig


def plot_provider_ladder(md: MarketData, gpu: str, days: int = 7, top: int = 25):
    """Latest price by provider, cheapest to dearest, coloured by tier group."""
    obs = md.observations
    last = obs["date"].max()
    o = obs[
        (obs["instrument"] == gpu)
        & (obs["date"] > last - pd.Timedelta(days=days))
        & (obs["rental_type"].isin(["on_demand", "secure", "community"]))
    ]
    p = o.groupby("provider")["price"].median().sort_values()
    if len(p) > top:
        p = pd.concat([p.iloc[: top // 2], p.iloc[-(top - top // 2) :]])
    idx_last = md.index[gpu].dropna().iloc[-1]
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(p) + 1.4))
    y = np.arange(len(p))
    ax.barh(y, p.values, height=0.6, color=gpu_color(gpu))
    for yi, v in zip(y, p.values, strict=True):
        ax.text(v * 1.01, yi, f"${v:.2f}", va="center", fontsize=8, color=INK_2)
    ax.axvline(idx_last, color=INK, lw=1, ls="--")
    ax.text(idx_last, len(p) - 0.3, f" index ${idx_last:.2f}", fontsize=8.5, color=INK, va="bottom")
    ax.set_yticks(y, p.index)
    ax.grid(axis="y", visible=False)
    _dollars(ax, "x", 0)
    ax.set_xlabel("$ per GPU-hour (on-demand, median of the last week)")
    ax.set_title(f"{gpu}: price by provider, week to {last.date()}")
    ax.set_xlim(0, p.max() * 1.15)
    fig.tight_layout()
    return fig


def plot_tiers(md: MarketData, gpu: str, days: int = 30):
    """Distribution of provider prices by rental tier over the last ``days``."""
    obs = md.observations
    last = obs["date"].max()
    o = obs[
        (obs["instrument"] == gpu)
        & (obs["date"] > last - pd.Timedelta(days=days))
        & (obs["rental_type"] != "index")
    ]
    per = o.groupby(["rental_type", "provider"])["price"].median().reset_index()
    tiers = [t for t in TIER_ORDER if t in set(per["rental_type"])]
    data = [per.loc[per["rental_type"] == t, "price"].to_numpy() for t in tiers]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bp = ax.boxplot(
        data,
        vert=False,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "lw": 1.4},
        whiskerprops={"color": INK_2},
        capprops={"color": INK_2},
        boxprops={"edgecolor": INK_2},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(gpu_color(gpu))
        patch.set_alpha(0.35)
    for i, d in enumerate(data, start=1):
        ax.scatter(
            d,
            np.full_like(d, i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(d)),
            s=12,
            color=gpu_color(gpu),
            zorder=3,
            lw=0,
        )
        if len(d):
            ax.text(
                np.max(d) * 1.03,
                i,
                f"median ${np.median(d):.2f}  (n={len(d)})",
                va="center",
                fontsize=8.5,
                color=INK_2,
            )
    ax.set_yticks(range(1, len(tiers) + 1), [t.replace("_", "-") for t in tiers])
    ax.grid(axis="y", visible=False)
    _dollars(ax, "x", 0)
    ax.set_xlabel("$ per GPU-hour (one dot per provider, median over the period)")
    ax.set_title(f"{gpu}: price by rental tier, last {days} days")
    ax.set_xlim(0, max((np.max(d) for d in data if len(d)), default=1) * 1.35)
    fig.tight_layout()
    return fig


def plot_return_distribution(md: MarketData, gpu: str, horizon_days: int = 7):
    """Histogram of overlapping ``horizon_days`` price changes of the index."""
    s = md.index[gpu].dropna().resample("D").last().ffill()
    r = (s / s.shift(horizon_days) - 1).dropna()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(r, bins=40, color=gpu_color(gpu), edgecolor=SURFACE, lw=1)
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.axvline(r.mean(), color=INK, lw=1.2, ls="--")
    ax.text(r.mean(), ax.get_ylim()[1] * 0.95, f"  mean {r.mean():+.1%}", fontsize=8.5, color=INK)
    _percent(ax, "x")
    ax.set_xlabel(f"{horizon_days}-day price change")
    ax.set_ylabel("days")
    q5, q95 = r.quantile([0.05, 0.95])
    ax.set_title(
        f"{gpu}: {horizon_days}-day price changes "
        f"(5th pct {q5:+.1%}, 95th pct {q95:+.1%}, n={len(r)} overlapping)"
    )
    fig.tight_layout()
    return fig


def plot_premium_to_ornn(md: MarketData, gpus: Sequence[str]):
    """Average posted price relative to Ornn's transacted index, where Ornn has history."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    shown = []
    for g in gpus:
        b = md.benchmark(g)
        if b is None or len(b) < 10:
            continue
        lv = md.level[g].dropna()
        prem = (lv.reindex(b.index) / b - 1).dropna()
        ax.plot(prem.index, prem.values, color=gpu_color(g), label=g)
        shown.append(g)
    ax.axhline(0, color=MUTED, lw=0.8)
    _percent(ax)
    ax.set_ylabel("posted premium over transacted")
    ax.set_title("How far posted list prices sit above Ornn's transaction index")
    if shown:
        ax.legend(loc="upper left")
    fig.tight_layout()
    return fig
