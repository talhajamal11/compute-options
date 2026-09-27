"""Rolling issuance backtest: issue a note every ``issue_every_days``, mark every note daily,
settle at maturity, and report investor returns and issuer P&L."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from compute_options.backtest.hedging import delta_hedge
from compute_options.backtest.issuance import IssuedNote, issue_note
from compute_options.backtest.lifecycle import mark_to_market
from compute_options.backtest.market_builder import MarketBuilder, MarketModelConfig
from compute_options.backtest.metrics import note_record, summarize
from compute_options.products import SolveError, build_product


@dataclass(frozen=True)
class BacktestConfig:
    product: dict[str, Any]  # product spec, e.g. {"kind": "buffered", "tenor": 0.25, ...}
    solve_for: str | None = "default"  # term solved at each issue; "default" = product's own
    issue_price: float = 1.0
    issuer_margin: float = 0.01
    issue_every_days: int = 7
    start: str | None = None
    end: str | None = None
    hedge: bool = True
    market: MarketModelConfig = field(default_factory=MarketModelConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BacktestConfig:
        d = dict(d)
        d["market"] = MarketModelConfig(**d.get("market", {}))
        return cls(**d)


@dataclass
class BacktestResult:
    notes: pd.DataFrame
    paths: dict[str, pd.DataFrame]
    summary: dict[str, Any]
    skipped: list[tuple[pd.Timestamp, str]]


def run_backtest(
    index: pd.Series, cfg: BacktestConfig, rates: pd.DataFrame | None = None
) -> BacktestResult:
    builder = MarketBuilder(index, cfg.market, rates)
    template = build_product(cfg.product)
    solve_for = template.default_solve_for if cfg.solve_for == "default" else cfg.solve_for

    idx = builder.index.index
    start = pd.Timestamp(cfg.start) if cfg.start else idx[0]
    end = pd.Timestamp(cfg.end) if cfg.end else idx[-1]

    records, paths, skipped = [], {}, []
    for date in pd.date_range(start, end, freq=f"{cfg.issue_every_days}D"):
        if not builder.can_price(date):
            skipped.append((date, "insufficient history"))
            continue
        try:
            note: IssuedNote = issue_note(
                template, builder, date, cfg.issue_price, cfg.issuer_margin, solve_for
            )
        except SolveError as e:
            skipped.append((date, str(e)))
            continue
        path = mark_to_market(note, builder)
        if cfg.hedge:
            path = delta_hedge(path)
        paths[note.note_id] = path
        records.append(note_record(note, path))

    notes = pd.DataFrame(records)
    return BacktestResult(notes, paths, summarize(notes) if len(notes) else {}, skipped)
