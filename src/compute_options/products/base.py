"""Base class for structured products.

A product states its payoff twice:

- ``payoff(R)``: the term-sheet formula, as an investor would read it.
- ``legs()``: a static replication by zero-coupon bonds, calls, puts and digitals.

Pricing uses the legs; tests assert the two agree on a grid of ``R``, so a wrong decomposition
cannot slip through. All amounts are fractions of notional (1.0 = 100%).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar

import numpy as np

from compute_options.instruments import Leg, PricingContext
from compute_options.market.state import MarketState
from compute_options.models.monte_carlo import MCResult, price_paths


@dataclass(frozen=True)
class StructuredProduct(ABC):
    tenor: float  # years from issue to maturity

    kind: ClassVar[str]
    # Term the solver adjusts by default to price the note at the target issue price.
    default_solve_for: ClassVar[str]
    # Search interval for each solvable term.
    bounds: ClassVar[dict[str, tuple[float, float]]] = {}

    @abstractmethod
    def payoff(self, R: np.ndarray) -> np.ndarray:
        """Redemption per unit notional given ``R = S_T / S_0``."""

    @abstractmethod
    def legs(self) -> list[tuple[float, Leg]]:
        """``(quantity, leg)`` pairs whose combined payoff equals ``payoff``."""

    def replicated_payoff(self, R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, float)
        return sum((q * leg.payoff(R) for q, leg in self.legs()), np.zeros_like(R))

    def price(self, market: MarketState, initial_fixing: float, T: float | None = None) -> float:
        """Fair value per unit notional; ``T`` is time remaining (defaults to the full tenor)."""
        T = self.tenor if T is None else T
        ctx = PricingContext.from_market(market, initial_fixing, T)
        return float(sum(q * leg.price(ctx) for q, leg in self.legs()))

    def leg_breakdown(
        self, market: MarketState, initial_fixing: float, T: float | None = None
    ) -> list[dict[str, Any]]:
        T = self.tenor if T is None else T
        ctx = PricingContext.from_market(market, initial_fixing, T)
        return [
            {"leg": type(leg).__name__, **asdict(leg), "quantity": q, "value": q * leg.price(ctx)}
            for q, leg in self.legs()
        ]

    def price_mc(self, paths: np.ndarray, df: float, initial_fixing: float) -> MCResult:
        """Monte Carlo value from simulated index paths (last column = maturity)."""
        return price_paths(lambda p: self.payoff_path(p / initial_fixing), paths, df)

    def payoff_path(self, R_paths: np.ndarray) -> np.ndarray:
        """Override for path-dependent features (e.g. daily-monitored barriers)."""
        return self.payoff(R_paths[:, -1])

    def terms(self) -> dict[str, Any]:
        return {"kind": self.kind, **{f.name: getattr(self, f.name) for f in fields(self)}}
