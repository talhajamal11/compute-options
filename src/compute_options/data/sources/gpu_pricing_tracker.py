"""github.com/cherielilili/gpu-pricing-tracker: daily multi-provider scrape since Feb 2026.

Covers Vast.ai, RunPod, Lambda, Crusoe, Nebius and ~15 others (partly via getdeploying.com).
Prices are per GPU-hour, except before 2026-02-19 when multi-GPU rows were per *instance*
(8x H100 on AWS at $55.04 rather than $6.88); those are divided by ``gpu_count``, and
2026-02-18, which mixes both units, is dropped for multi-GPU rows. Rows are split by instance
size and region, which the schema validator collapses to a median.

The repo has no licence file: fine for research, check before redistributing.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from compute_options.data.sources.base import Source, register
from compute_options.data.symbology import canonical_instrument

PER_GPU_FROM = "2026-02-19"
MIXED_UNITS_DATE = "2026-02-18"

CSV_URL = (
    "https://raw.githubusercontent.com/cherielilili/gpu-pricing-tracker/main/data/observations.csv"
)


@register
class GpuPricingTrackerSource(Source):
    name = "gpu_pricing_tracker"
    url = "https://github.com/cherielilili/gpu-pricing-tracker"

    def _fetch(self, since: date | None) -> pd.DataFrame:
        return self.parse(self._get_csv(CSV_URL))

    @staticmethod
    def parse(raw: pd.DataFrame) -> pd.DataFrame:
        multi = raw["gpu_count"] > 1
        raw = raw[~(multi & (raw["date"] == MIXED_UNITS_DATE))].copy()
        per_instance = (raw["gpu_count"] > 1) & (raw["date"] < PER_GPU_FROM)
        raw.loc[per_instance, "price_median_usd"] /= raw.loc[per_instance, "gpu_count"]
        return pd.DataFrame(
            {
                "date": raw["date"],
                "provider": raw["provider"].str.lower(),
                "raw_instrument": raw["gpu_model"],
                "instrument": raw["gpu_model"].map(canonical_instrument),
                "rental_type": raw["rental_type"],
                "price": raw["price_median_usd"],
                "n_obs": raw["n_offers"],
            }
        )
