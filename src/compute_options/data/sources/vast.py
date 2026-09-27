"""Vast.ai marketplace: median live on-demand price of verified, rentable offers.

Snapshot only, so run ``download`` daily to accumulate history.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from compute_options.data.schemas import RentalType
from compute_options.data.sources.base import Source, register
from compute_options.data.symbology import canonical_instrument

URL = "https://console.vast.ai/api/v0/bundles/"
GPUS = [
    "H100 SXM",
    "H100 NVL",
    "H100 PCIE",
    "H200",
    "H200 NVL",
    "B200",
    "A100 SXM4",
    "A100 PCIE",
    "L40S",
    "RTX 4090",
    "RTX 5090",
    "RTX PRO 6000 WS",
    "RTX PRO 6000 S",
]


@register
class VastSource(Source):
    name = "vast"
    url = "https://vast.ai"
    has_history = False

    def _fetch(self, since: date | None) -> pd.DataFrame:
        rows = []
        today = pd.Timestamp.now(tz="UTC").normalize()
        for gpu in GPUS:
            query = {
                "gpu_name": {"eq": gpu},
                "rentable": {"eq": True},
                "verified": {"eq": True},
                "type": "on-demand",
                "limit": 500,
            }
            resp = self.session.post(URL, json=query, timeout=self.timeout)
            resp.raise_for_status()
            offers = resp.json().get("offers", [])
            per_gpu = [o["dph_total"] / o["num_gpus"] for o in offers if o.get("num_gpus")]
            if per_gpu:
                rows.append(
                    {
                        "date": today,
                        "provider": "vast.ai",
                        "raw_instrument": gpu,
                        "instrument": canonical_instrument(gpu),
                        "rental_type": RentalType.COMMUNITY.value,
                        "price": float(np.median(per_gpu)),
                        "n_obs": len(per_gpu),
                    }
                )
        return pd.DataFrame(
            rows,
            columns=[
                "date",
                "provider",
                "raw_instrument",
                "instrument",
                "rental_type",
                "price",
                "n_obs",
            ],
        )
