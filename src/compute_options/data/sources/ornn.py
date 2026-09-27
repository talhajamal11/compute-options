"""Ornn Compute Price Index (OCPI): transaction-based GPU rental benchmark.

Free, keyless endpoints (https://data.ornn.com): the latest settled value for every tracked GPU,
and three months of daily history for H100 SXM. Values settle once per day at 4pm ET. Longer
history and the forward curve need an API key; add them here when one is available.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from compute_options.data.schemas import RentalType
from compute_options.data.sources.base import Source, register
from compute_options.data.symbology import canonical_instrument

BASE = "https://api.ornnai.com/api"
SETTLE_TZ = "America/New_York"


def _settle_date(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True).dt.tz_convert(SETTLE_TZ).dt.tz_localize(None).dt.normalize()


@register
class OrnnSource(Source):
    name = "ornn"
    url = "https://data.ornn.com"
    license = "Ornn free tier"
    is_index = True

    def _fetch(self, since: date | None) -> pd.DataFrame:
        hist = pd.DataFrame(self._get_json(f"{BASE}/h100-history")["data"])
        hist = pd.DataFrame(
            {
                "date": _settle_date(hist["timestamp"]),
                "raw_instrument": "H100 SXM",
                "price": hist["index_value"],
            }
        )
        latest = pd.DataFrame(self._get_json(f"{BASE}/daily-index/all")["data"])
        latest = pd.DataFrame(
            {
                "date": _settle_date(latest["date"]),
                "raw_instrument": latest["gpu_type"],
                "price": latest["index_value"],
            }
        )
        df = pd.concat([hist, latest], ignore_index=True)
        return df.assign(
            provider="ornn",
            instrument=df["raw_instrument"].map(canonical_instrument),
            rental_type=RentalType.INDEX.value,
            n_obs=pd.NA,
        )
