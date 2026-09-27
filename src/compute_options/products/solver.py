"""Solve a product's free term so it prices at a target (par less the issuer's margin)."""

from __future__ import annotations

from dataclasses import replace

from scipy.optimize import brentq

from compute_options.market.state import MarketState
from compute_options.products.base import StructuredProduct


class SolveError(ValueError):
    pass


def solve(
    product: StructuredProduct,
    market: MarketState,
    initial_fixing: float,
    target: float = 1.0,
    field: str | None = None,
    bounds: tuple[float, float] | None = None,
) -> StructuredProduct:
    """Copy of ``product`` with ``field`` set so ``price == target``.

    ``target`` is the fair value the issuer is willing to embed: an issue price of 100% with a
    1.5% margin means ``target = 0.985``.
    """
    field = field or product.default_solve_for
    lo, hi = bounds or product.bounds.get(field, (1e-6, 10.0))

    def err(x: float) -> float:
        return replace(product, **{field: x}).price(market, initial_fixing) - target

    f_lo, f_hi = err(lo), err(hi)
    if f_lo * f_hi > 0:
        raise SolveError(
            f"{product.kind}: cannot reach target {target:.4f} by varying {field} in "
            f"[{lo}, {hi}] (price range {f_lo + target:.4f} to {f_hi + target:.4f})"
        )
    return replace(product, **{field: brentq(err, lo, hi, xtol=1e-10)})
