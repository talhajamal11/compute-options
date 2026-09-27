"""Combine observations from all sources into one daily price series per instrument.

Steps:

1. **Provider level.** The same provider is often scraped by several sources (Vast.ai and RunPod
   appear in four). Take the median across sources for each (date, provider, instrument, tier),
   then the median across the provider's tiers, so no provider counts more than once.
2. **Composite level.** Trimmed mean across providers of that day's provider prices.
3. **Price index (time-product dummy).** The set of quoted providers changes from day to day, so
   the level composite jumps when a cheap or expensive provider appears or drops out, although
   no price moved. The index instead fits all prices at once,

       ``log p[date, provider] = alpha[date] + beta[provider] + e``,

   and ``exp(alpha)`` is the index: the price level after removing each provider's persistent
   premium or discount (``beta``). This is the time-product-dummy method used for price indices
   with sellers entering and leaving. Because it compares every price with the provider's own
   typical level rather than with yesterday's price, errors don't accumulate the way they do in
   a day-to-day chained index, and a provider that disappears for months and returns at a new
   price moves the index correctly.

   Robustness: residuals are clipped at ``clip`` (30%) in the fit, so a single bad print (a unit
   error, a mislabelled tier) can move a day by at most ``clip / n``; days with fewer than
   ``min_providers_index`` providers are left out; and each provider's price is first replaced
   by its trailing ``smoothing_days`` median on days it quoted, removing one-day flips in
   marketplace medians without using future data. The fit uses the whole sample, so past
   values are revised slightly as new data arrives. The index is scaled to match the level
   composite on average.

   A median-based fit or a trimmed-mean chain is *wrong* for posted prices: they are sticky,
   so on a typical day one or two providers reprice and the rest don't, and a median discards
   exactly the providers that moved (H100 would show about 0% from March to September while
   the providers quoted on both dates rose a median 15%).

Use the index for volatility and backtests, and the level series for "what does an H100 cost
today".

Published benchmarks (``Source.is_index``) are not averaged in; each becomes its own series named
``"{source}:{instrument}"``, e.g. ``ornn:H100_SXM``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from compute_options.data.schemas import RentalType
from compute_options.data.sources import REGISTRY


@dataclass(frozen=True)
class AggregationConfig:
    rental_types: tuple[str, ...] = (
        RentalType.ON_DEMAND.value,
        RentalType.SECURE.value,
        RentalType.COMMUNITY.value,
    )
    trim: float = 0.1  # fraction trimmed from each tail of the cross-provider *level* mean
    min_providers: int = 3  # composite days with fewer providers are dropped
    min_providers_index: int = 5  # index days with fewer providers quoting are left out
    clip: float = 0.3  # residual clip (log) in the index fit; limits one bad print's influence
    smoothing_days: int = 3  # trailing median per provider before fitting; 1 = off
    max_iter: int = 500


def _index_sources() -> set[str]:
    return {name for name, cls in REGISTRY.items() if cls.is_index}


def provider_prices(
    obs: pd.DataFrame, cfg: AggregationConfig = AggregationConfig()
) -> pd.DataFrame:
    """One price per (date, instrument, provider)."""
    quotes = obs[~obs["source"].isin(_index_sources()) & obs["rental_type"].isin(cfg.rental_types)]
    by_tier = quotes.groupby(["date", "instrument", "provider", "rental_type"], as_index=False).agg(
        price=("price", "median"), n_sources=("source", "nunique")
    )
    return by_tier.groupby(["date", "instrument", "provider"], as_index=False).agg(
        price=("price", "median"), n_sources=("n_sources", "max")
    )


def composite_levels(
    prov: pd.DataFrame, cfg: AggregationConfig = AggregationConfig()
) -> pd.DataFrame:
    """Trimmed-mean level per (date, instrument). Columns: date, instrument, price, n_providers."""
    out = prov.groupby(["date", "instrument"]).agg(
        price=("price", lambda p: float(trim_mean(p, cfg.trim))),
        n_providers=("provider", "nunique"),
    )
    out = out[out["n_providers"] >= cfg.min_providers]
    return out.reset_index()


def price_index(
    prov: pd.DataFrame, levels: pd.DataFrame, cfg: AggregationConfig = AggregationConfig()
) -> pd.DataFrame:
    """Time-product-dummy index per instrument. Columns: date, instrument, price, n_providers."""
    frames = []
    for inst, g in prov.groupby("instrument"):
        wide = g.pivot_table(index="date", columns="provider", values="price").sort_index()
        if cfg.smoothing_days > 1:
            # Only on days the provider actually quoted: no carrying stale prices forward.
            wide = wide.rolling(cfg.smoothing_days, min_periods=1).median().where(wide.notna())
        n = wide.notna().sum(axis=1)
        wide = wide[n >= cfg.min_providers_index]
        wide = wide.loc[:, wide.notna().sum() >= 2]  # a provider needs 2+ days to inform moves
        if wide.shape[0] < 2 or wide.shape[1] < cfg.min_providers_index:
            continue
        alpha, _ = _fit_tpd(np.log(wide), cfg)
        lv = levels[levels["instrument"] == inst].set_index("date")["price"]
        common = alpha.index.intersection(lv.index)
        shift = float((np.log(lv[common]) - alpha[common]).mean()) if len(common) else 0.0
        frames.append(
            pd.DataFrame(
                {
                    "date": alpha.index,
                    "instrument": inst,
                    "price": np.exp(alpha + shift).to_numpy(),
                    "n_providers": wide.notna().sum(axis=1).to_numpy(),
                }
            )
        )
    cols = ["date", "instrument", "price", "n_providers"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def _fit_tpd(y: pd.DataFrame, cfg: AggregationConfig) -> tuple[pd.Series, pd.Series]:
    """Alternating least squares for ``y[t, i] = alpha[t] + beta[i]`` with clipped residuals
    (a Huber-type fit). ``y`` is dates x providers with NaN where a provider didn't quote."""
    alpha = y.mean(axis=1)
    beta = pd.Series(0.0, index=y.columns)
    for _ in range(cfg.max_iter):
        res = y.sub(alpha, axis=0).sub(beta, axis=1).clip(-cfg.clip, cfg.clip)
        alpha = alpha + res.mean(axis=1)
        res = y.sub(alpha, axis=0).sub(beta, axis=1).clip(-cfg.clip, cfg.clip)
        step = res.mean(axis=0)
        beta = beta + step
        if step.abs().max() < 1e-10:
            break
    return alpha, beta


def index_series(obs: pd.DataFrame) -> pd.DataFrame:
    """Published benchmark series. Columns: date, instrument (``source:INSTR``), price."""
    idx = obs[obs["source"].isin(_index_sources())]
    return pd.DataFrame(
        {
            "date": idx["date"],
            "instrument": idx["source"] + ":" + idx["instrument"],
            "price": idx["price"],
        }
    ).reset_index(drop=True)


def build(
    obs: pd.DataFrame, cfg: AggregationConfig = AggregationConfig()
) -> dict[str, pd.DataFrame]:
    """All aggregated outputs, as wide date x instrument panels plus the provider-level table."""
    prov = provider_prices(obs, cfg)
    levels = composite_levels(prov, cfg)
    index = price_index(prov, levels, cfg)
    idx = index_series(obs)

    def wide(df: pd.DataFrame, col: str = "price") -> pd.DataFrame:
        return df.pivot_table(index="date", columns="instrument", values=col).sort_index()

    return {
        "provider_prices": prov,
        "composite_level": wide(levels),
        "composite_n_providers": wide(levels, "n_providers"),
        "composite_index": wide(index),
        "benchmarks": wide(idx) if not idx.empty else pd.DataFrame(),
    }
