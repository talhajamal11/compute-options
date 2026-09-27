"""github.com/Zeno00-00/gpu-price-tracker: daily Vast.ai and RunPod medians since Jun 2026.

Covers H100/H200/B200/A100 on Vast.ai (median of verified rentable offers) and RunPod secure and
community cloud. No licence file: fine for research, check before redistributing.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from compute_options.data.schemas import RentalType
from compute_options.data.sources.base import Source, register
from compute_options.data.symbology import canonical_instrument

CSV_URL = "https://raw.githubusercontent.com/Zeno00-00/gpu-price-tracker/main/data/gpu_prices.csv"

TIERS = {
    "marketplace_on_demand": RentalType.COMMUNITY,
    "secure_cloud": RentalType.SECURE,
    "community_cloud": RentalType.COMMUNITY,
    "on_demand": RentalType.ON_DEMAND,
    "spot": RentalType.SPOT,
}


@register
class GpuPriceTrackerSource(Source):
    name = "gpu_price_tracker"
    url = "https://github.com/Zeno00-00/gpu-price-tracker"

    def _fetch(self, since: date | None) -> pd.DataFrame:
        return self.parse(self._get_csv(CSV_URL))

    @staticmethod
    def parse(raw: pd.DataFrame) -> pd.DataFrame:
        tiers = raw["tier"].map(lambda t: TIERS.get(t, RentalType.ON_DEMAND).value)
        return pd.DataFrame(
            {
                "date": raw["date"],
                "provider": raw["provider"].str.lower(),
                "raw_instrument": raw["gpu_model"],
                "instrument": raw["gpu_model"].map(canonical_instrument),
                "rental_type": tiers,
                "price": raw["price_usd_per_gpu_hr"],
                "n_obs": raw["n_offers"],
            }
        )
