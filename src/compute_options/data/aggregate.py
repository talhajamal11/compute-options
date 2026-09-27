"""Combine observations from all sources into one daily price series per instrument.

Three steps:

1. **Provider level.** The same provider is often scraped by several sources (Vast.ai and RunPod
   appear in four). Take the median across sources for each (date, provider, instrument, tier),
   then the median across the provider's tiers, so no provider counts more than once.
2. **Composite level.** Trimmed mean across providers of that day's provider prices.
3. **Chain-linking.** The set of providers changes from day to day, and a cheap provider dropping
   out makes the level jump even though no price moved. The chained index moves with the log
   changes of providers quoted on *both* days, anchored to the composite level on the first day.
   Use the chained series for volatility estimation and the level series for "what does an H100
   cost today".

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
    trim: float = 0.1  # fraction trimmed from each tail in the cross-provider mean
    min_providers: int = 3  # composite days with fewer providers are dropped
    min_matched: int = 2  # chained steps need at least this many providers on both days
    max_step: float = float(np.log(2.0))  # |daily log move| above this is treated as bad data


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


def chained_index(
    prov: pd.DataFrame, levels: pd.DataFrame, cfg: AggregationConfig = AggregationConfig()
) -> pd.DataFrame:
    """Chain-linked series per instrument. Columns: date, instrument, price, n_matched."""
    rows = []
    for inst, g in prov.groupby("instrument"):
        lv = levels[levels["instrument"] == inst].set_index("date")["price"]
        if lv.empty:
            continue
        wide = g.pivot_table(index="date", columns="provider", values="price").sort_index()
        wide = wide.loc[wide.index >= lv.index[0]]
        logp = np.log(wide)
        level = float(lv.iloc[0])
        rows.append((lv.index[0], inst, level, int(wide.iloc[0].notna().sum())))
        for i in range(1, len(logp)):
            step = (logp.iloc[i] - logp.iloc[i - 1]).dropna()
            step = step[step.abs() <= cfg.max_step]
            if len(step) < cfg.min_matched:
                continue
            level *= float(np.exp(trim_mean(step, cfg.trim)))
            rows.append((logp.index[i], inst, level, len(step)))
    return pd.DataFrame(rows, columns=["date", "instrument", "price", "n_matched"])


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
    chained = chained_index(prov, levels, cfg)
    idx = index_series(obs)

    def wide(df: pd.DataFrame, col: str = "price") -> pd.DataFrame:
        return df.pivot_table(index="date", columns="instrument", values=col).sort_index()

    return {
        "provider_prices": prov,
        "composite_level": wide(levels),
        "composite_n_providers": wide(levels, "n_providers"),
        "composite_chained": wide(chained),
        "benchmarks": wide(idx) if not idx.empty else pd.DataFrame(),
    }
