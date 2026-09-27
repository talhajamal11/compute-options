from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from compute_options.instruments import Leg, Participation
from compute_options.products.base import StructuredProduct


@dataclass(frozen=True)
class TrackerNote(StructuredProduct):
    """Delta-one exposure to the compute index: ``payoff = participation * R``.

    Because GPU-hours can't be stored, the note is hedged with futures, not by holding the
    underlying. Its fair value is ``p * DF * F(T) / S_0``, so at par the participation is
    ``1 / (DF * F / S_0)``. A futures curve in contango (``F > S``) means participation below
    100%; backwardation means above. This roll cost is the main economic difference from an
    equity tracker.
    """

    participation: float = 1.0

    kind: ClassVar[str] = "tracker"
    default_solve_for: ClassVar[str] = "participation"
    bounds: ClassVar[dict[str, tuple[float, float]]] = {"participation": (0.01, 20.0)}

    def payoff(self, R: np.ndarray) -> np.ndarray:
        return self.participation * np.asarray(R, float)

    def legs(self) -> list[tuple[float, Leg]]:
        return [(self.participation, Participation())]
