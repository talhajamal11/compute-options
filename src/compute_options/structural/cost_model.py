"""Levelised cost of a GPU-hour: the floor that new supply keeps pulling the price towards.

A GPU-hour can't be stored, so nothing links its price to spot by arbitrage. What anchors the
long-run level is entry: if the market pays well above the all-in cost of running a GPU, more
capacity gets built. The cost of supply therefore sets the level the forward curve reverts to.

    cost per sold hour = (annualised capex + facility + energy + other opex) / hours sold

Everything works on floats or numpy arrays, so a whole set of prior draws is costed at once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compute_options.structural.assumptions import Params

HOURS_PER_YEAR = 8760.0
# Fraction of full power an idle GPU server draws. Fixed, flagged as an assumption in the README.
IDLE_POWER_FRACTION = 0.40


@dataclass(frozen=True)
class CostBreakdown:
    """All fields in $ per sold GPU-hour."""

    capital: float | np.ndarray
    facility: float | np.ndarray
    energy: float | np.ndarray
    other: float | np.ndarray

    @property
    def total(self) -> float | np.ndarray:
        return self.capital + self.facility + self.energy + self.other


def capital_recovery_factor(rate: float | np.ndarray, life: float | np.ndarray):
    return rate / (1.0 - (1.0 + rate) ** -life)


def levelised_cost(p: Params) -> CostBreakdown:
    """Cost per sold GPU-hour from a parameter set (see ``assumptions.register``)."""
    capex = np.asarray(p["capex"], float)
    crf = capital_recovery_factor(np.asarray(p["rate"], float), np.asarray(p["life"], float))
    pv_resid = capex * p["resid"] * (1.0 + p["rate"]) ** -p["life"]
    capital_per_year = (capex - pv_resid) * crf

    hours_sold = HOURS_PER_YEAR * p["util"]
    facility_per_year = p["it_kw"] * p["colo"] * 12.0
    draw = IDLE_POWER_FRACTION + (1.0 - IDLE_POWER_FRACTION) * p["util"]
    energy_per_year = p["it_kw"] * p["pue"] * HOURS_PER_YEAR * p["tariff"] * draw
    other_per_year = capex * p["opex_pct"]

    return CostBreakdown(
        capital=capital_per_year / hours_sold,
        facility=facility_per_year / hours_sold,
        energy=energy_per_year / hours_sold,
        other=other_per_year / hours_sold,
    )


def anchor_price(p: Params) -> float | np.ndarray:
    """Long-run competitive price today: levelised cost plus the required margin."""
    return levelised_cost(p).total * (1.0 + np.asarray(p["margin"], float))


def substitution_ceiling(new_gen_price: float, perf_ratio: float) -> float:
    """Highest price an older GPU can sustain: the newer generation's price per unit of work.

    If a B200-hour does ``perf_ratio`` times the work of an H100-hour, nobody pays more than
    ``B200 price / perf_ratio`` for an H100-hour once both are available. In practice other
    constraints (software, supply, contracts) let the older GPU trade above or below it; the
    gap is a read on how tight supply is.
    """
    return new_gen_price / perf_ratio
