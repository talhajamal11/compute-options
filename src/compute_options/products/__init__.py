"""Structured products and a registry for building them from config."""

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


def build_product(spec: dict[str, Any]) -> StructuredProduct:
    """``{"kind": "buffered", "tenor": 0.5, "buffer": 0.1, ...}`` -> ``BufferedNote``."""
    spec = dict(spec)
    kind = spec.pop("kind")
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
    "solve",
]
