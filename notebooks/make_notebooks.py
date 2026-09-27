"""Generate one notebook per structure from a shared template.

    uv run python notebooks/make_notebooks.py            # write notebooks
    uv run python notebooks/make_notebooks.py --execute  # write and run them

Edit the parameter cell inside a notebook to explore; edit this file to change the defaults
for everyone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent

MARKET = {
    "vol_method": "realized",
    "vol_window": 90,
    "vol_horizon_days": 7,
    "vol_floor": 0.10,
    "forward_model": "flat",
    "rate": None,
    "issuer_spread": 0.01,
    "min_history": 21,
}
MARKET_COMMENTS = {
    "vol_method": '"realized" | "ewma" | "ou"',
    "vol_window": "calendar days of history used for vol",
    "vol_horizon_days": "7 = weekly returns (filters day-to-day noise)",
    "vol_floor": "minimum vol used for pricing",
    "forward_model": '"flat" (forward = spot) | "schwartz" (mean-reverting)',
    "rate": "None = US Treasury curve from FRED; or a flat rate, e.g. 0.04",
    "issuer_spread": "issuer credit spread over Treasuries",
    "min_history": "days of prices needed before the first note can be priced",
}

# (term, default, comment) per structure; "solve" marks the term the solver sets.
STRUCTURES = [
    {
        "file": "01_zero_coupon_note.ipynb",
        "title": "Zero Coupon Note",
        "kind": "zero_coupon",
        "tenor": "12M",
        "terms": [("redemption", "solve", "paid at maturity; solved at par = the note's yield")],
        "margin": 0.0,
        "about": (
            "Pays a fixed redemption at maturity and is issued at a discount. It has **no "
            "exposure to compute prices**: it is the funding leg inside every other note. "
            "Solving the redemption at par gives the note's yield (Treasury rate plus the "
            "issuer's credit spread); `1 - price` is the budget a capital-protected note can "
            "spend on options.\n\n`payoff = redemption`"
        ),
    },
    {
        "file": "02_capital_protected_note.ipynb",
        "title": "Capital Protected Note",
        "kind": "capital_protected",
        "tenor": "12M",
        "terms": [
            ("protection", 1.00, "capital returned at maturity (1.00 = 100%)"),
            ("strike", 1.00, "upside starts above this level of the initial fixing"),
            ("participation", "solve", '"solve", or a number, e.g. 0.60 = 60% of the upside'),
            ("cap", 1.50, "upside capped at this level; None = uncapped"),
        ],
        "margin": 0.01,
        "about": (
            "Returns at least `protection` of notional at maturity, plus `participation` in "
            "any rise of the compute index above `strike`, capped at `cap`. The issuer buys a "
            "zero-coupon bond for the protected amount and spends the rest on calls.\n\n"
            "`payoff = protection + participation * (min(R, cap) - strike)+`"
        ),
    },
    {
        "file": "03_buffered_note.ipynb",
        "title": "Buffered Note",
        "kind": "buffered",
        "tenor": "12M",
        "terms": [
            ("buffer", 0.10, "first 10% of losses absorbed"),
            ("participation", "solve", '"solve", or a number, e.g. 1.50 = 150% of the upside'),
            ("cap", None, "upside capped at this level; None = uncapped"),
            ("downside_gearing", 1.0, "loss per 1% fall beyond the buffer (1/(1-buffer) = geared)"),
        ],
        "margin": 0.01,
        "about": (
            "Participation in the upside (optionally capped); the first `buffer` of losses is "
            "absorbed, and losses beyond it are taken one-for-one (or geared). The investor "
            "effectively sells a put at `1 - buffer` to pay for the upside.\n\n"
            "`payoff = 1 + participation * (min(R, cap) - 1)+ - gearing * (1 - buffer - R)+`"
        ),
    },
    {
        "file": "04_dual_directional_note.ipynb",
        "title": "Dual Directional Note",
        "kind": "dual_directional",
        "tenor": "6M",
        "terms": [
            ("barrier", "solve", "absolute return on falls down to this level (at maturity)"),
            ("participation", 1.00, "share of the upside"),
            ("cap", 1.20, "upside capped at this level; None = uncapped"),
        ],
        "margin": 0.01,
        "about": (
            "Gains if compute prices rise *or* fall moderately: upside participation up to "
            "`cap`, and an absolute return on falls as long as the index ends above `barrier` "
            "(observed at maturity only). Below the barrier the investor takes the full loss.\n\n"
            "- `R >= 1`: `1 + participation * (min(R, cap) - 1)`\n"
            "- `barrier <= R < 1`: `2 - R`\n"
            "- `R < barrier`: `R`"
        ),
    },
    {
        "file": "05_risk_reversal_note.ipynb",
        "title": "Risk Reversal Note",
        "kind": "risk_reversal",
        "tenor": "6M",
        "terms": [
            ("put_strike", 0.90, "investor loses 1:1 below this level"),
            ("call_strike", "solve", "investor gains above this level"),
            ("participation", 1.00, "share of the upside above the call strike"),
            ("funded", True, "True = a note bought for cash; False = unfunded option overlay"),
        ],
        "margin": 0.01,
        "about": (
            "Long upside above `call_strike`, short downside below `put_strike`, flat in "
            "between. The put sold pays for the call. A GPU buyer hedging rising rental costs "
            "holds the unfunded version.\n\n"
            "`payoff = 1 + participation * (R - call_strike)+ - (put_strike - R)+`"
        ),
    },
    {
        "file": "06_tracker_note.ipynb",
        "title": "Tracker Note",
        "kind": "tracker",
        "tenor": "3M",
        "terms": [("participation", "solve", '"solve", or a number, e.g. 1.00')],
        "margin": 0.005,
        "about": (
            "Delta-one exposure to the compute index. GPU-hours can't be stored, so the issuer "
            "hedges with futures, and the fair participation is `1 / (DF * F / S)`: contango "
            'costs the investor, backwardation helps. Set `forward_model` to `"schwartz"` '
            "to price a mean-reverting forward.\n\n`payoff = participation * R`"
        ),
    },
]


def lit(v: object) -> str:
    return repr(v) if not isinstance(v, float) else f"{v:.2f}"


def dict_block(name: str, items: list[tuple[str, object, str]]) -> str:
    width = max(len(f'    "{k}": {lit(v)},') for k, v, _ in items)
    lines = [f'    "{k}": {lit(v)},'.ljust(width) + f"  # {c}" for k, v, c in items]
    return f"{name} = {{\n" + "\n".join(lines) + "\n}"


def params_cell(s: dict) -> str:
    market = [(k, v, MARKET_COMMENTS[k]) for k, v in MARKET.items()]
    return f'''# --- Underlying -----------------------------------------------------------------
UNDERLYING = "H100_SXM"           # any instrument listed in the table below
SERIES = "composite_index"      # "composite_index" | "composite_level" | "benchmarks" (Ornn)

# --- Structure ------------------------------------------------------------------
TENOR = "{s["tenor"]}"                    # "1M", "3M", "6M", "12M", "18M", "2Y", ...
{dict_block("TERMS", s["terms"])}
ISSUE_PRICE = 1.00                # what the investor pays per unit notional
ISSUER_MARGIN = {s["margin"]:.3f}             # issuer's fee, used when a term is "solve"

# --- Expected return ------------------------------------------------------------
EXPECTED_DRIFT = 0.00             # your view of the index's annual drift (0 = forward is fair)

# --- Holding backtest -----------------------------------------------------------
HOLD_START = "2026-01-01"         # first note bought on the first priceable date on/after this
HOLD_END = None                   # None = latest data
ROLL = True                       # at maturity, reinvest in a new note with the same terms

# --- Market model ---------------------------------------------------------------
{dict_block("MARKET", market)}'''


def build(s: dict) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [
        md(
            f"# {s['title']}\n\n{s['about']}\n\n"
            "`R` = index at maturity / index at issue; amounts are fractions of notional.\n\n"
            "Refresh data first with `uv run compute-options download && uv run "
            "compute-options aggregate`."
        ),
        md(
            "## 1. Parameters\n\n"
            'Set any term to a number to fix it, or to `"solve"` to have the solver set it '
            "so the note is worth `ISSUE_PRICE - ISSUER_MARGIN` (at most one). With every term "
            "fixed, the note is priced as given and the table below shows the margin the "
            "issuer would earn (negative = the investor gets a bargain)."
        ),
        code(params_cell(s)),
        code(
            "%matplotlib inline\n"
            "import pandas as pd\n\n"
            "from compute_options.analytics import notebook as nb\n\n"
            'pd.set_option("display.width", 160)\n'
            'print(f"Underlyings in {SERIES}:")\n'
            "nb.available_underlyings(SERIES)"
        ),
        md("## 2. Pricing today"),
        code(
            "series = nb.load_underlying(UNDERLYING, SERIES)\n"
            f'priced = nb.price_structure(series, "{s["kind"]}", TENOR, TERMS, MARKET, '
            "ISSUE_PRICE, ISSUER_MARGIN)\n"
            "display(nb.terms_table(priced))\n"
            "display(nb.pricing_summary(priced))"
        ),
        md(
            "### Expected return of holding to maturity\n\n"
            "Buying at the issue price and holding to maturity. **APR** annualises the return "
            "to maturity: `(payoff / issue price) ^ (1 / tenor) - 1`.\n\n"
            "- **Model:** the index ends lognormally distributed around its forward (plus "
            "`EXPECTED_DRIFT`), with the pricing vol. With zero drift the expected APR is "
            "roughly the issuer's funding rate less the margin: a fairly priced note "
            "doesn't beat cash on average unless you have a view on the index.\n"
            "- **Historical:** the payoff applied to every realised move of the same length "
            "in the history (few, overlapping scenarios: illustrative only)."
        ),
        code("nb.expected_table(priced, series, EXPECTED_DRIFT)"),
        md(
            "### Replication and Greeks\n\n"
            "How the price is built from zero-coupon bonds, calls, puts and digitals, and its "
            "sensitivities per unit notional (delta per 1% index move, vega per vol point, "
            "theta per day, rho per bp)."
        ),
        code("display(nb.legs_table(priced))\ndisplay(nb.greeks_table(priced))"),
        md(
            "## 3. Payoff at maturity\n\n"
            "Red dotted lines mark strikes, caps and barriers. The blue histogram shows the "
            "index moves that actually happened over the same tenor in the history."
        ),
        code("nb.plot_payoff(priced, series);"),
        md(
            "## 4. Backtest: holding the structure\n\n"
            "Invest $1 in the note on the first date it can be priced on or after "
            "`HOLD_START` and hold it. At maturity the proceeds are reinvested in a new note "
            "on the same terms (a solved term is re-solved against that day's market). A note "
            "still alive at the end is marked to model value. The note is bought at the issue "
            "price but marked at fair value, so the value dips by the issuer's margin on each "
            "purchase.\n\n"
            "Statistics are annualised over the holding period, with Sharpe and Sortino "
            "measured against the 3-month T-bill. The index is shown for comparison but is "
            "not investable. **The history is months long, so annualised figures are "
            "extrapolated from a short sample.**"
        ),
        code(
            "res = nb.run_hold(series, priced, MARKET, ISSUER_MARGIN, HOLD_START, HOLD_END, "
            "ROLL)\n"
            'print(f"Held {UNDERLYING} {TENOR} {priced.product.kind} from {res.start.date()} '
            'to {res.end.date()} "\n'
            '      f"({len(res.notes)} note(s), average T-bill {res.rf:.2%})")\n'
            "nb.holding_stats_table(res)"
        ),
        code("nb.plot_holding(res);"),
        md(
            "### Notes held\n\n"
            "Each note bought, its terms, the investor's return, and the P&L of the issuer "
            "who sold it: unhedged (keeps the cash, pays the note), or delta-hedged with a "
            "futures contract on the index marked at the model forward (no public futures "
            "settlements exist yet)."
        ),
        code("nb.notes_held_table(res)"),
    ]
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    return nb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    for s in STRUCTURES:
        path = HERE / s["file"]
        nb = build(s)
        if args.execute:
            from nbconvert.preprocessors import ExecutePreprocessor

            ExecutePreprocessor(timeout=900).preprocess(nb, {"metadata": {"path": str(HERE)}})
        nbf.write(nb, path)
        print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
