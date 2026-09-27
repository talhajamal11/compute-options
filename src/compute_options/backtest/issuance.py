"""Issuing a note: fix the initial level, solve its terms against the day's market."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from compute_options.backtest.market_builder import MarketBuilder
from compute_options.products import StructuredProduct, solve


@dataclass(frozen=True)
class IssuedNote:
    note_id: str
    issue_date: pd.Timestamp
    maturity_date: pd.Timestamp
    initial_fixing: float
    product: StructuredProduct
    issue_price: float  # what the investor pays, per unit notional
    fair_value: float  # model value at issue; issue_price - fair_value = issuer margin
    growth: float  # 1 / issuer DF to maturity: value at maturity of 1 received at issue


def issue_note(
    template: StructuredProduct,
    builder: MarketBuilder,
    date: pd.Timestamp,
    issue_price: float = 1.0,
    margin: float = 0.01,
    solve_for: str | None = None,
) -> IssuedNote:
    """``solve_for=None`` issues ``template`` as-is; otherwise that term is solved so the fair
    value equals ``issue_price - margin``."""
    market = builder.state(date)
    s0 = builder.spot(date)
    product = template
    if solve_for:
        product = solve(template, market, s0, target=issue_price - margin, field=solve_for)
    maturity = date + pd.Timedelta(days=round(template.tenor * 365))
    return IssuedNote(
        note_id=f"{template.kind}-{date.date()}",
        issue_date=date,
        maturity_date=maturity,
        initial_fixing=s0,
        product=product,
        issue_price=issue_price,
        fair_value=product.price(market, s0),
        growth=1.0 / market.df(template.tenor),
    )
