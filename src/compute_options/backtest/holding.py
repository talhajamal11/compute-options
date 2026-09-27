"""Buy-and-hold backtest: an investor buys one note and holds it.

At maturity the proceeds are reinvested in a new note on the same terms (re-solved against that
day's market if a term is being solved), until ``end``. A note still alive at ``end`` is marked
to model value. The result is a daily NAV for the investor, the index and cash, plus the
issuer's P&L on each note the investor held.

The investor pays the issue price but the note is marked at fair value, so NAV drops by the
issuer's margin on each purchase, as it would on a dealer's bid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from compute_options.backtest.hedging import delta_hedge
from compute_options.backtest.issuance import issue_note
from compute_options.backtest.lifecycle import mark_to_market
from compute_options.backtest.market_builder import MarketBuilder, MarketModelConfig
from compute_options.backtest.metrics import note_record
from compute_options.backtest.performance import performance_stats
from compute_options.products import build_product


@dataclass(frozen=True)
class HoldingConfig:
    product: dict[str, Any]
    solve_for: str | None = None
    issue_price: float = 1.0
    issuer_margin: float = 0.01  # only used when a term is solved
    start: str | None = None  # first date a note can be priced on or after this
    end: str | None = None
    roll: bool = True  # reinvest at maturity; False = hold cash after the first note
    hedge: bool = True  # issuer delta-hedges (for the issuer P&L view)
    market: MarketModelConfig = field(default_factory=MarketModelConfig)


@dataclass
class HoldingResult:
    nav: pd.DataFrame  # columns: structure, index, cash, note_id
    notes: pd.DataFrame  # one row per note held
    stats: pd.DataFrame  # annualised stats: rows = metric, cols = structure / index / cash
    start: pd.Timestamp
    end: pd.Timestamp
    rf: float


def _risk_free(builder: MarketBuilder, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Daily short rate: flat config rate, or FRED 3M (SOFR fallback) forward-filled."""
    days = pd.date_range(start, end, freq="D")
    if builder.cfg.rate is not None or builder.rates is None:
        return pd.Series(builder.cfg.rate or 0.0, index=days)
    col = "DGS3MO" if "DGS3MO" in builder.rates else "SOFR"
    rates = builder.rates[col].reindex(builder.rates.index.union(days)).ffill().bfill()
    return rates.reindex(days)


def run_holding(
    series: pd.Series, cfg: HoldingConfig, rates: pd.DataFrame | None = None
) -> HoldingResult:
    builder = MarketBuilder(series, cfg.market, rates)
    template = build_product(cfg.product)
    idx = builder.index.index
    end = min(pd.Timestamp(cfg.end), idx[-1]) if cfg.end else idx[-1]
    first = max(pd.Timestamp(cfg.start), idx[0]) if cfg.start else idx[0]
    candidates = pd.date_range(first, end, freq="D")
    start = next((d for d in candidates if builder.can_price(d)), None)
    if start is None:
        raise ValueError(f"not enough history to price a note before {end.date()}")

    units, date = None, start
    pieces, records = [], []
    while date < end:
        note = issue_note(
            template, builder, date, cfg.issue_price, cfg.issuer_margin, cfg.solve_for
        )
        # First note: invest 1.0. Rolls: reinvest the previous note's payoff.
        wealth = 1.0 if units is None else float(pieces[-1]["structure"].iloc[-1])
        units = wealth / note.issue_price
        path = mark_to_market(note, builder, until=end)
        if cfg.hedge:
            path = delta_hedge(path)
        records.append(note_record(note, path) | {"units": units, "invested": wealth})
        piece = pd.DataFrame({"structure": units * path["value"], "note_id": note.note_id})
        if pieces:
            piece = piece.iloc[1:]  # the maturity day is already in the previous piece
        pieces.append(piece)
        if not path["matured"].iloc[-1] or not cfg.roll:
            break
        date = note.maturity_date

    nav = pd.concat(pieces)
    nav = nav[~nav.index.duplicated(keep="last")]
    rf = _risk_free(builder, start, end)
    if nav.index[-1] < end:  # rolled off without reinvesting: sit in cash
        tail = pd.date_range(nav.index[-1], end, freq="D")[1:]
        growth = np.exp(rf.reindex(tail).cumsum() / 365.0)
        cash_tail = pd.DataFrame(
            {"structure": nav["structure"].iloc[-1] * growth, "note_id": "cash"}, index=tail
        )
        nav = pd.concat([nav, cash_tail])

    daily = builder.index.resample("D").last().ffill().reindex(nav.index).ffill()
    nav["index"] = daily / daily.iloc[0]
    nav["cash"] = np.exp(rf.reindex(nav.index).fillna(0).cumsum().shift(fill_value=0) / 365.0)

    rf_avg = float(rf.mean())
    stats = pd.DataFrame({c: performance_stats(nav[c], rf) for c in ("structure", "index", "cash")})
    return HoldingResult(nav, pd.DataFrame(records), stats, start, end, rf_avg)
