"""Greeks and scenario analysis by bump-and-reprice (works for any product)."""

from compute_options.risk.expected import expected_return, historical_scenarios
from compute_options.risk.greeks import Greeks, greeks
from compute_options.risk.scenarios import scenario_grid

__all__ = ["Greeks", "expected_return", "greeks", "historical_scenarios", "scenario_grid"]
