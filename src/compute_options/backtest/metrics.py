"""Summary statistics per note and across a backtest, from the investor's and issuer's side."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from compute_options.backtest.issuance import IssuedNote


def note_record(note: IssuedNote, path: pd.DataFrame) -> dict[str, Any]:
    last = path.iloc[-1]
    matured = bool(last["matured"])
    years = max((path.index[-1] - note.issue_date).days / 365.0, 1 / 365)
    value = float(last["value"])  # payoff if matured, else mark-to-market
    inv_ret = value / note.issue_price - 1.0
    margin = note.issue_price - note.fair_value
    rec: dict[str, Any] = {
        "note_id": note.note_id,
        "issue_date": note.issue_date,
        "maturity_date": note.maturity_date,
        "matured": matured,
        "initial_fixing": note.initial_fixing,
        **{f"term_{k}": v for k, v in note.product.terms().items() if k != "kind"},
        "issue_price": note.issue_price,
        "fair_value": note.fair_value,
        # Investor
        "final_R": float(last["R"]),
        "value": value,
        "investor_return": inv_ret,
        "investor_return_ann": (1 + inv_ret) ** (1 / years) - 1 if inv_ret > -1 else -1.0,
        "underlying_return": float(last["R"]) - 1.0,
        "excess_vs_underlying": inv_ret - (float(last["R"]) - 1.0),
        # Issuer (per unit notional). Unhedged: keep the cash, pay the note.
        "issuer_margin": margin,
        "issuer_unhedged_pnl": note.issue_price * note.growth ** (years / note.product.tenor)
        - value,
    }
    if "hedge_error" in path:
        rec["issuer_hedge_error"] = float(last["hedge_error"])
        rec["issuer_hedged_pnl"] = (
            margin * note.growth ** (years / note.product.tenor) + rec["issuer_hedge_error"]
        )
    return rec


def summarize(notes: pd.DataFrame) -> dict[str, Any]:
    done = notes[notes["matured"]]
    if done.empty:
        return {"n_notes": len(notes), "n_matured": 0}
    r = done["investor_return"]
    out: dict[str, Any] = {
        "n_notes": len(notes),
        "n_matured": len(done),
        "investor_mean_return": r.mean(),
        "investor_median_return": r.median(),
        "investor_worst_return": r.min(),
        "investor_best_return": r.max(),
        "investor_prob_loss": (r < 0).mean(),
        "investor_mean_excess_vs_underlying": done["excess_vs_underlying"].mean(),
        "issuer_mean_margin": done["issuer_margin"].mean(),
        "issuer_mean_unhedged_pnl": done["issuer_unhedged_pnl"].mean(),
        "issuer_worst_unhedged_pnl": done["issuer_unhedged_pnl"].min(),
    }
    if "issuer_hedged_pnl" in done:
        h = done["issuer_hedged_pnl"]
        out |= {
            "issuer_mean_hedged_pnl": h.mean(),
            "issuer_hedged_pnl_std": h.std(ddof=1) if len(h) > 1 else np.nan,
            "issuer_worst_hedged_pnl": h.min(),
        }
    return out
