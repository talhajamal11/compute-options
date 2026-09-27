from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from compute_options.instruments import Call, Leg, ZeroCouponBond
from compute_options.products.base import StructuredProduct


@dataclass(frozen=True)
class CapitalProtectedNote(StructuredProduct):
    """Returns at least ``protection`` of notional, plus ``participation`` in any rise above
    ``strike``, optionally capped at ``cap``.

    ``payoff = protection + p * (min(R, cap) - strike)^+``

    Replication: zero-coupon bond for the protected amount, plus ``p`` calls at ``strike``,
    minus ``p`` calls at ``cap``.
    """

    protection: float = 1.0
    strike: float = 1.0
    participation: float = 1.0
    cap: float | None = None

    kind: ClassVar[str] = "capital_protected"
    default_solve_for: ClassVar[str] = "participation"
    bounds: ClassVar[dict[str, tuple[float, float]]] = {
        "participation": (0.0, 20.0),
        "cap": (1.0001, 10.0),
        "protection": (0.0, 1.5),
        "strike": (0.2, 5.0),
    }

    def payoff(self, R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, float)
        upside = R if self.cap is None else np.minimum(R, self.cap)
        return self.protection + self.participation * np.maximum(upside - self.strike, 0.0)

    def legs(self) -> list[tuple[float, Leg]]:
        legs: list[tuple[float, Leg]] = [
            (1.0, ZeroCouponBond(self.protection)),
            (self.participation, Call(self.strike)),
        ]
        if self.cap is not None:
            legs.append((-self.participation, Call(self.cap)))
        return legs
