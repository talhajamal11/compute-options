"""Payoff diagrams (requires the ``plot`` extra: matplotlib)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from compute_options.products import StructuredProduct


def payoff_table(
    products: Sequence[StructuredProduct], lo: float = 0.4, hi: float = 1.8, n: int = 141
):
    import pandas as pd

    R = np.linspace(lo, hi, n)
    return pd.DataFrame({p.kind: p.payoff(R) for p in products}, index=pd.Index(R, name="R"))


def plot_payoffs(
    products: Sequence[StructuredProduct],
    path: str | Path | None = None,
    lo: float = 0.4,
    hi: float = 1.8,
):
    import matplotlib.pyplot as plt

    table = payoff_table(products, lo, hi)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table.index, table.index, color="grey", ls="--", lw=1, label="index (R)")
    for col in table:
        ax.plot(table.index, table[col], lw=2, label=col)
    ax.axhline(1.0, color="black", lw=0.5)
    ax.set_xlabel("Index at maturity / initial fixing (R)")
    ax.set_ylabel("Redemption (fraction of notional)")
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
