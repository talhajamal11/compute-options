"""Canonical schema for price observations.

Every source adapter returns a DataFrame with ``OBSERVATION_COLUMNS``. One row is one price
for one (date, source, provider, instrument, rental tier), in USD per GPU-hour.
"""

from __future__ import annotations

from enum import StrEnum

import pandas as pd


class RentalType(StrEnum):
    ON_DEMAND = "on_demand"
    SECURE = "secure"  # e.g. RunPod secure cloud: datacentre on-demand
    COMMUNITY = "community"  # peer-hosted on-demand (RunPod community, Vast marketplace)
    SPOT = "spot"  # interruptible
    SERVERLESS = "serverless"  # per-second billing, carries a premium
    RESERVED = "reserved"
    INDEX = "index"  # published benchmark (e.g. Ornn OCPI), not a posted rate


OBSERVATION_COLUMNS: dict[str, str] = {
    "date": "datetime64[ns]",
    "source": "string",
    "provider": "string",
    "instrument": "string",
    "raw_instrument": "string",
    "rental_type": "string",
    "price": "float64",
    "n_obs": "Int64",
}

KEY = ["date", "source", "provider", "instrument", "rental_type"]


def empty_observations() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in OBSERVATION_COLUMNS.items()})


def validate_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, drop unusable rows and collapse duplicates on ``KEY`` (median price)."""
    missing = set(OBSERVATION_COLUMNS) - set(df.columns)
    if "n_obs" in missing:
        df = df.assign(n_obs=pd.NA)
        missing.discard("n_obs")
    if missing:
        raise ValueError(f"observations missing columns: {sorted(missing)}")
    if df.empty:
        return empty_observations()

    out = df[list(OBSERVATION_COLUMNS)].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["n_obs"] = pd.to_numeric(out["n_obs"], errors="coerce").astype("Int64")
    for c in ("source", "provider", "instrument", "raw_instrument", "rental_type"):
        out[c] = out[c].astype("string")
    valid_types = {t.value for t in RentalType}
    bad = ~out["rental_type"].isin(valid_types)
    if bad.any():
        raise ValueError(f"unknown rental types: {sorted(out.loc[bad, 'rental_type'].unique())}")
    out = out[out["price"].gt(0) & out["price"].notna() & out["instrument"].notna()]

    out = (
        out.groupby(KEY, as_index=False, dropna=False)
        .agg(
            raw_instrument=("raw_instrument", "first"),
            price=("price", "median"),
            n_obs=("n_obs", lambda s: s.sum(min_count=1)),
        )
        .astype({"n_obs": "Int64"})
    )
    return out[list(OBSERVATION_COLUMNS)].sort_values(KEY, ignore_index=True)
