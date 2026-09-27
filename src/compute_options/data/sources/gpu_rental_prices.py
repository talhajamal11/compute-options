"""github.com/adriannutiu/gpu-rental-prices (gpurentalprices.com): daily snapshots since Jul 2026.

~30 providers including hyperscalers (AWS, GCP, Azure), CoreWeave, Lambda, Nebius, Crusoe and
inference platforms. Licensed CC BY 4.0. One JSON file per day; only new days are downloaded.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from compute_options.data.schemas import RentalType
from compute_options.data.sources.base import Source, register
from compute_options.data.symbology import canonical_instrument

REPO = "adriannutiu/gpu-rental-prices"
TREE_URL = f"https://api.github.com/repos/{REPO}/git/trees/main?recursive=1"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/main/"

KINDS = {
    "on-demand": RentalType.ON_DEMAND,
    "secure": RentalType.SECURE,
    "community": RentalType.COMMUNITY,
    "spot": RentalType.SPOT,
    "serverless": RentalType.SERVERLESS,
    "reserved": RentalType.RESERVED,
}


@register
class GpuRentalPricesSource(Source):
    name = "gpu_rental_prices"
    url = "https://gpurentalprices.com"
    license = "CC BY 4.0"

    def _fetch(self, since: date | None) -> pd.DataFrame:
        tree = self._get_json(TREE_URL)["tree"]
        paths = sorted(
            t["path"]
            for t in tree
            if t["path"].startswith("data/snapshots/") and t["path"].endswith(".json")
        )
        if since is not None:
            paths = [p for p in paths if p.rsplit("/", 1)[-1][:10] >= since.isoformat()]
        frames = [self.parse(self._get_json(RAW_URL + p)) for p in paths]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date"])

    @staticmethod
    def parse(snapshot: dict[str, Any]) -> pd.DataFrame:
        offers = pd.DataFrame(snapshot["offers"])
        return pd.DataFrame(
            {
                "date": snapshot["date"],
                "provider": offers["provider"].str.lower(),
                "raw_instrument": offers["gpu"],
                "instrument": offers["gpu"].map(canonical_instrument),
                "rental_type": offers["kind"].map(
                    lambda k: KINDS.get(k, RentalType.ON_DEMAND).value
                ),
                "price": offers["usd_hr"],
                "n_obs": 1,
            }
        )
