"""Structured products and a registry for building them from config."""

import re
from typing import Any

from compute_options.products.base import StructuredProduct
from compute_options.products.buffered import BufferedNote
from compute_options.products.capital_protected import CapitalProtectedNote
from compute_options.products.dual_directional import DualDirectionalNote
from compute_options.products.risk_reversal import RiskReversalNote
from compute_options.products.solver import SolveError, solve
from compute_options.products.tracker import TrackerNote
from compute_options.products.zero_coupon import ZeroCouponNote

PRODUCTS: dict[str, type[StructuredProduct]] = {
    cls.kind: cls
    for cls in (
        ZeroCouponNote,
        CapitalProtectedNote,
        BufferedNote,
        DualDirectionalNote,
        RiskReversalNote,
        TrackerNote,
    )
}


def parse_tenor(tenor: str | float) -> float:
    """``"3M"`` -> 0.25, ``"1Y"`` / ``"12M"`` -> 1.0, ``"2W"``, ``"90D"``; numbers are years."""
    if isinstance(tenor, int | float):
        return float(tenor)
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([DWMY])\s*", tenor.upper())
    if not m:
        raise ValueError(f"bad tenor {tenor!r}; use e.g. '3M', '6M', '1Y', '90D'")
    n, unit = float(m[1]), m[2]
    return {"D": n / 365, "W": n * 7 / 365, "M": n / 12, "Y": n}[unit]


def build_product(spec: dict[str, Any]) -> StructuredProduct:
    """``{"kind": "buffered", "tenor": "6M", "buffer": 0.1, ...}`` -> ``BufferedNote``."""
    spec = dict(spec)
    kind = spec.pop("kind")
    if "tenor" in spec:
        spec["tenor"] = parse_tenor(spec["tenor"])
    try:
        return PRODUCTS[kind](**spec)
    except KeyError:
        raise KeyError(f"unknown product {kind!r}; available: {sorted(PRODUCTS)}") from None


__all__ = [
    "PRODUCTS",
    "BufferedNote",
    "CapitalProtectedNote",
    "DualDirectionalNote",
    "RiskReversalNote",
    "SolveError",
    "StructuredProduct",
    "TrackerNote",
    "ZeroCouponNote",
    "build_product",
    "parse_tenor",
    "solve",
]
