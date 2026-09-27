"""Monte Carlo pricing of path-dependent payoffs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MCResult:
    price: float
    stderr: float
    n_paths: int


def price_paths(
    payoff: Callable[[np.ndarray], np.ndarray], paths: np.ndarray, df: float
) -> MCResult:
    """Discounted mean of ``payoff(paths)``; ``paths`` has shape ``(n_paths, n_times)``."""
    values = df * np.asarray(payoff(paths), float)
    return MCResult(
        float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values))), len(values)
    )
