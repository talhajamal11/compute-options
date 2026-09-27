"""gpu.watchworks.dev: free current-price endpoint for RunPod and Vast.ai.

Snapshot only (the history API is paid), so run ``download`` daily to accumulate history.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from compute_options.data.schemas import RentalType
from compute_options.data.sources.base import Source, register
from compute_options.data.symbology import canonical_instrument

URL = "https://gpu.watchworks.dev/api/prices"

KINDS = {
    "community": RentalType.COMMUNITY,
    "secure": RentalType.SECURE,
    "verified": RentalType.COMMUNITY,
    "any": RentalType.COMMUNITY,
}


@register
class GpuWatchSource(Source):
    name = "gpuwatch"
    url = "https://gpu.watchworks.dev"
    has_history = False

    def _fetch(self, since: date | None) -> pd.DataFrame:
        payload = self._get_json(URL)
        prices = pd.DataFrame(payload["prices"])
        provider = prices["provider"].str.lower().replace({"vast": "vast.ai"})
        return pd.DataFrame(
            {
                "date": pd.Timestamp(payload["updated_at"]).tz_convert("UTC").normalize(),
                "provider": provider,
                "raw_instrument": prices["gpu"],
                "instrument": prices["gpu"].map(canonical_instrument),
                "rental_type": prices["kind"].map(
                    lambda k: KINDS.get(k, RentalType.ON_DEMAND).value
                ),
                "price": prices["usd_hr"],
                "n_obs": 1,
            }
        )
