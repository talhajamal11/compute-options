"""Legs, written in *performance space*: the underlying is ``R = S_T / S_0``, the index at
maturity relative to its level at the initial fixing, and payoffs are fractions of notional.
A strike of 0.9 is 90% of the initial fixing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from compute_options.market.state import MarketState
from compute_options.models import black76


@dataclass(frozen=True)
class PricingContext:
    """Inputs for pricing legs on one note at one valuation date."""

    forward: float  # forward of R: F(T) / S_0
    T: float  # years to maturity
    df: float  # issuer discount factor to maturity
    vol: Callable[[float], float]  # strike (in R) -> Black vol

    @classmethod
    def from_market(cls, market: MarketState, initial_fixing: float, T: float) -> PricingContext:
        fwd = market.forward(T) / initial_fixing
        return cls(
            forward=fwd,
            T=T,
            df=market.df(T),
            vol=lambda k: market.implied_vol(T, k / fwd),
        )


class Leg(ABC):
    @abstractmethod
    def payoff(self, R: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def price(self, ctx: PricingContext) -> float: ...


@dataclass(frozen=True)
class ZeroCouponBond(Leg):
    amount: float = 1.0

    def payoff(self, R: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(R, float), self.amount)

    def price(self, ctx: PricingContext) -> float:
        return self.amount * ctx.df


@dataclass(frozen=True)
class Participation(Leg):
    """Pays ``R`` at maturity: a zero-strike call, i.e. a prepaid forward on the index."""

    def payoff(self, R: np.ndarray) -> np.ndarray:
        return np.asarray(R, float)

    def price(self, ctx: PricingContext) -> float:
        return ctx.df * ctx.forward


@dataclass(frozen=True)
class Call(Leg):
    strike: float

    def payoff(self, R: np.ndarray) -> np.ndarray:
        return np.maximum(np.asarray(R, float) - self.strike, 0.0)

    def price(self, ctx: PricingContext) -> float:
        return float(black76.call(ctx.forward, self.strike, ctx.T, ctx.vol(self.strike), ctx.df))


@dataclass(frozen=True)
class Put(Leg):
    strike: float

    def payoff(self, R: np.ndarray) -> np.ndarray:
        return np.maximum(self.strike - np.asarray(R, float), 0.0)

    def price(self, ctx: PricingContext) -> float:
        return float(black76.put(ctx.forward, self.strike, ctx.T, ctx.vol(self.strike), ctx.df))


@dataclass(frozen=True)
class DigitalCall(Leg):
    """Pays 1 if ``R > strike`` (priced at the strike's vol, i.e. ignoring smile)."""

    strike: float

    def payoff(self, R: np.ndarray) -> np.ndarray:
        return (np.asarray(R, float) > self.strike).astype(float)

    def price(self, ctx: PricingContext) -> float:
        return float(
            black76.digital_call(ctx.forward, self.strike, ctx.T, ctx.vol(self.strike), ctx.df)
        )


@dataclass(frozen=True)
class DigitalPut(Leg):
    """Pays 1 if ``R < strike``."""

    strike: float

    def payoff(self, R: np.ndarray) -> np.ndarray:
        return (np.asarray(R, float) < self.strike).astype(float)

    def price(self, ctx: PricingContext) -> float:
        return float(
            black76.digital_put(ctx.forward, self.strike, ctx.T, ctx.vol(self.strike), ctx.df)
        )
