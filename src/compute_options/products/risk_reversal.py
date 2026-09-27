from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from compute_options.instruments import Call, Leg, Put, ZeroCouponBond
from compute_options.products.base import StructuredProduct


@dataclass(frozen=True)
class RiskReversalNote(StructuredProduct):
    """Long upside above ``call_strike``, short downside below ``put_strike``; flat in between.

    ``payoff = N + p * (R - call_strike)^+ - (put_strike - R)^+``, with ``N = 1`` when
    ``funded`` (a note bought for cash) and ``N = 0`` for the unfunded option overlay.

    The classic trade is zero-cost: the solver moves ``call_strike`` until the note prices at
    par (funded) or zero (unfunded). A GPU buyer hedging rising rental costs holds the
    unfunded version; a data-centre operator hedging falling rates holds its mirror image.
    """

    put_strike: float = 0.90
    call_strike: float = 1.10
    participation: float = 1.0
    funded: bool = True

    kind: ClassVar[str] = "risk_reversal"
    default_solve_for: ClassVar[str] = "call_strike"
    bounds: ClassVar[dict[str, tuple[float, float]]] = {
        "call_strike": (0.2, 10.0),
        "put_strike": (0.05, 3.0),
        "participation": (0.0, 20.0),
    }

    def payoff(self, R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, float)
        base = 1.0 if self.funded else 0.0
        return (
            base
            + self.participation * np.maximum(R - self.call_strike, 0.0)
            - np.maximum(self.put_strike - R, 0.0)
        )

    def legs(self) -> list[tuple[float, Leg]]:
        legs: list[tuple[float, Leg]] = [
            (self.participation, Call(self.call_strike)),
            (-1.0, Put(self.put_strike)),
        ]
        if self.funded:
            legs.insert(0, (1.0, ZeroCouponBond(1.0)))
        return legs
