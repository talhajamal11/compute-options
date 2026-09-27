"""US Treasury constant-maturity yields and SOFR from FRED's public CSV endpoint (no key)."""

from __future__ import annotations

import io

import pandas as pd
import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={ids}"

# Series id -> tenor in years (SOFR treated as overnight).
SERIES: dict[str, float] = {
    "SOFR": 1 / 365,
    "DGS1MO": 1 / 12,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
}


def fetch_rates(start: str = "2025-01-01", timeout: float = 60.0) -> pd.DataFrame:
    """Daily yields as decimals, indexed by date, one column per FRED series id."""
    url = FRED_CSV.format(ids=",".join(SERIES)) + f"&cosd={start}"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), na_values=["."])
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.Timestamp(start)]
    return (df.set_index("date").astype(float) / 100.0).sort_index()
