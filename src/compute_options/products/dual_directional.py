from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from compute_options.instruments import Call, DigitalPut, Leg, Put, ZeroCouponBond
from compute_options.products.base import StructuredProduct


@dataclass(frozen=True)
class DualDirectionalNote(StructuredProduct):
    """Gains whether compute prices rise or fall moderately.

    - ``R >= 1``: ``1 + p * (min(R, cap) - 1)``  (upside participation)
    - ``barrier <= R < 1``: ``1 + (1 - R)``      (absolute return on the fall)
    - ``R < barrier``: ``R``                     (full downside, 1:1)

    The barrier is observed at maturity only (European). Replication: zero-coupon bond,
    ``p`` call spread, a put at 100%, minus two puts at the barrier, minus ``2 (1 - barrier)``
    digital puts at the barrier (the last two flip the absolute return back to a loss).
    """

    barrier: float = 0.80
    participation: float = 1.0
    cap: float | None = None

    kind: ClassVar[str] = "dual_directional"
    default_solve_for: ClassVar[str] = "participation"
    bounds: ClassVar[dict[str, tuple[float, float]]] = {
        "participation": (0.0, 20.0),
        "cap": (1.0001, 10.0),
        "barrier": (0.01, 0.9999),
    }

    def payoff(self, R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, float)
        upside = R if self.cap is None else np.minimum(R, self.cap)
        up = 1.0 + self.participation * (upside - 1.0)
        return np.where(R >= 1.0, up, np.where(self.barrier <= R, 2.0 - R, R))

    def legs(self) -> list[tuple[float, Leg]]:
        legs: list[tuple[float, Leg]] = [
            (1.0, ZeroCouponBond(1.0)),
            (self.participation, Call(1.0)),
            (1.0, Put(1.0)),
            (-2.0, Put(self.barrier)),
            (-2.0 * (1.0 - self.barrier), DigitalPut(self.barrier)),
        ]
        if self.cap is not None:
            legs.append((-self.participation, Call(self.cap)))
        return legs
