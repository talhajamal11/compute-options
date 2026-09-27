from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from compute_options.instruments import Call, Leg, Put, ZeroCouponBond
from compute_options.products.base import StructuredProduct


@dataclass(frozen=True)
class BufferedNote(StructuredProduct):
    """Upside participation (optionally capped); the first ``buffer`` of losses is absorbed.

    ``payoff = 1 + p * (min(R, cap) - 1)^+ - g * (1 - buffer - R)^+``

    ``downside_gearing`` ``g`` = 1 is a standard buffer. With ``g = 1 / (1 - buffer)`` the
    investor loses everything at ``R = 0`` (a "geared buffer").

    Replication: zero-coupon bond, ``p`` calls at 100% minus ``p`` calls at ``cap``, and short
    ``g`` puts at ``1 - buffer``.
    """

    buffer: float = 0.10
    participation: float = 1.0
    cap: float | None = None
    downside_gearing: float = 1.0

    kind: ClassVar[str] = "buffered"
    default_solve_for: ClassVar[str] = "participation"
    bounds: ClassVar[dict[str, tuple[float, float]]] = {
        "participation": (0.0, 20.0),
        "cap": (1.0001, 10.0),
        "buffer": (0.0, 0.99),
    }

    def payoff(self, R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, float)
        upside = R if self.cap is None else np.minimum(R, self.cap)
        return (
            1.0
            + self.participation * np.maximum(upside - 1.0, 0.0)
            - self.downside_gearing * np.maximum(1.0 - self.buffer - R, 0.0)
        )

    def legs(self) -> list[tuple[float, Leg]]:
        legs: list[tuple[float, Leg]] = [
            (1.0, ZeroCouponBond(1.0)),
            (self.participation, Call(1.0)),
            (-self.downside_gearing, Put(1.0 - self.buffer)),
        ]
        if self.cap is not None:
            legs.append((-self.participation, Call(self.cap)))
        return legs
