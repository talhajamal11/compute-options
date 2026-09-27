"""Map each source's GPU naming onto canonical instrument ids (e.g. ``H100_SXM``).

Generic names without a form factor ("H100", "A100") map to the SXM variant, which is what the
published benchmarks (Ornn OCPI, Silicon Data SDH100RT) track and what most cloud listings are.
The original name is kept in ``raw_instrument`` so this choice can be audited.
"""

from __future__ import annotations

import re

ALIASES: dict[str, str] = {
    "H100": "H100_SXM",
    "H100_SXM5": "H100_SXM",
    "H100_80GB": "H100_SXM",
    "H100_SXM_80GB": "H100_SXM",
    "H100_PCIE_80GB": "H100_PCIE",
    "H200_SXM": "H200",
    "H200_141GB": "H200",
    "A100": "A100_SXM_80GB",
    "A100_80GB": "A100_SXM_80GB",
    "A100_SXM4": "A100_SXM_80GB",
    "A100_SXM4_80GB": "A100_SXM_80GB",
    "A100_SXM_80": "A100_SXM_80GB",
    "A100_SXM_40": "A100_SXM_40GB",
    "A100_SXM4_40GB": "A100_SXM_40GB",
    "A100_PCIE": "A100_PCIE_80GB",
    "A100_PCIE_80": "A100_PCIE_80GB",
    "A100_PCIE_40": "A100_PCIE_40GB",
    "RTX_PRO_6000_WS": "RTX_PRO_6000",
    "RTX_PRO_6000_WK": "RTX_PRO_6000",
    "RTX_PRO_6000_S": "RTX_PRO_6000_SERVER",
    "RTX_6000ADA": "RTX_6000_ADA",
}


def normalize(raw: str) -> str:
    """Upper-case, collapse separators to ``_``: ``"h100-sxm"`` -> ``"H100_SXM"``."""
    s = re.sub(r"[\s\-/]+", "_", raw.strip().upper())
    return re.sub(r"_+", "_", s).strip("_")


def canonical_instrument(raw: str) -> str:
    key = normalize(raw)
    return ALIASES.get(key, key)
