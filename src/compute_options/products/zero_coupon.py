from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from compute_options.instruments import Leg, ZeroCouponBond
from compute_options.products.base import StructuredProduct


@dataclass(frozen=True)
class ZeroCouponNote(StructuredProduct):
    """Pays ``redemption`` at maturity; issued at a discount. No exposure to compute prices.

    It is the funding leg inside every other note: its price is the issuer's discount factor, so
    ``1 - price`` is the budget left over for options in a capital-protected structure. Solving
    for ``redemption`` at par gives the note's yield.
    """

    redemption: float = 1.0

    kind: ClassVar[str] = "zero_coupon"
    default_solve_for: ClassVar[str] = "redemption"
    bounds: ClassVar[dict[str, tuple[float, float]]] = {"redemption": (0.5, 2.0)}

    def payoff(self, R: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(R, float), self.redemption)

    def legs(self) -> list[tuple[float, Leg]]:
        return [(1.0, ZeroCouponBond(self.redemption))]
