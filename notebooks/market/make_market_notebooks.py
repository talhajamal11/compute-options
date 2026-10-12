"""Generate the market-data notebooks.

    uv run python notebooks/market/make_market_notebooks.py --execute

Refresh the data first: ``uv run compute-options download && uv run compute-options aggregate``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = """%matplotlib inline
import pandas as pd

from compute_options.analytics import market_viz as mv

mv.apply_style()
pd.set_option("display.width", 160)
data = mv.MarketData()"""


def overview() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# GPU rental market: overview\n\n"
            "Daily rental prices for data-centre GPUs, in $ per GPU-hour, from the aggregated "
            "data in `data/processed/`:\n\n"
            "- **Price index** (`composite_index`): the market price level after removing each "
            "provider's persistent premium or discount; unaffected by providers entering or "
            "leaving the data. Used for pricing and backtests.\n"
            "- **Average posted price** (`composite_level`): a trimmed average of the providers "
            "quoted that day. It jumps when the mix of providers changes.\n"
            "- **Ornn transacted index**: prices from actual trades (free tier: H100 history "
            "from late June; today's value for other GPUs).\n\n"
            "All are posted *list* prices except Ornn. See `src/compute_options/data/aggregate.py` "
            "for the method."
        ),
        md("## Parameters"),
        code(
            "# None = the data-centre GPUs with 60+ days of history (up to 8, one colour each),\n"
            '# or a list, e.g. ["H100_SXM", "H200", "B200"]\n'
            "GPUS = None\n\n"
            "# Rebase date for the relative-performance chart. None = earliest date.\n"
            "# GPUs whose data starts later are rebased at their own first day (dashed).\n"
            'REBASE_FROM = "2026-03-01"\n\n'
            "VOL_WINDOW_DAYS = 60  # trailing window for rolling volatility"
        ),
        code(
            SETUP + "\n\ngpus = GPUS or data.gpus(min_days=60)\n"
            'print(f"{len(gpus)} GPUs, data to {data.index.index[-1].date()}")'
        ),
        md(
            "## Latest prices\n\n"
            "Price index today, the average posted price, Ornn's transacted price, changes over "
            "1 and 3 months and since the start of each GPU's data (note the different start "
            "dates), and annualised volatility of weekly changes. The table lists every GPU "
            "with 60+ days of history; the charts show the data-centre GPUs."
        ),
        code("mv.summary_table(data, data.gpus(min_days=60, charted_only=False))"),
        code("mv.plot_latest_prices(data, gpus);"),
        md(
            "## Price history\n\n"
            "Each GPU in its own panel, in $ per GPU-hour. The coloured line is the price index; "
            "grey is the average posted price (note its jumps when providers enter or leave); "
            "dashed black is Ornn's transacted index where it has history."
        ),
        code("mv.plot_price_panels(data, gpus);"),
        md(
            "## Relative performance\n\n"
            "All GPUs on one axis, rebased to 100, so their moves can be compared despite very "
            "different price levels."
        ),
        code("mv.plot_indexed(data, gpus, REBASE_FROM);"),
        md(
            "## Volatility\n\n"
            "Annualised volatility of weekly price changes over a trailing window. Posted prices "
            "move in steps, so volatility comes in bursts when providers reprice."
        ),
        code("mv.plot_rolling_vol(data, gpus, VOL_WINDOW_DAYS);"),
        md(
            "## Posted vs transacted\n\n"
            "How far the average posted list price sits above Ornn's transaction-based index. "
            "Ornn's free tier only has history for H100."
        ),
        code("mv.plot_premium_to_ornn(data, gpus);"),
        md(
            "## Data coverage\n\n"
            "How many providers were quoted each day. Thin coverage (light cells) makes the "
            "index noisier; days with fewer than 5 providers are left out of the index. Most "
            "history before July comes from a single scraper."
        ),
        code("mv.plot_coverage(data, gpus);"),
    ]
    return nb


def deep_dive() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            "# GPU rental market: single-GPU deep dive\n\n"
            "Every provider's price for one GPU, how the indices summarise them, the spread "
            "between providers and rental tiers, and the distribution of price changes."
        ),
        md("## Parameters"),
        code(
            'GPU = "H100_SXM"                 # e.g. "H200", "B200", "A100_SXM_80GB"\n'
            'HIGHLIGHT = ["runpod", "lambda", "nebius"]  # providers drawn as labelled lines\n'
            "LADDER_DAYS = 7                  # window for the price-by-provider chart\n"
            "TIER_DAYS = 30                   # window for the price-by-tier chart\n"
            "RETURN_HORIZON_DAYS = 7          # horizon for the price-change histogram"
        ),
        code(SETUP + "\n\nmv.summary_table(data, [GPU])"),
        md(
            "## Provider prices and indices\n\n"
            "Grey dots are individual providers' daily prices. The coloured line is the price "
            "index, the thin grey line the average posted price, and the dashed black line "
            "Ornn's transacted index."
        ),
        code("mv.plot_gpu_prices(data, GPU, HIGHLIGHT);"),
        md(
            "## Price by provider\n\n"
            "On-demand price per provider over the last week, against today's price index. "
            "Hyperscalers (AWS, Azure, GCP) typically sit well above specialist GPU clouds."
        ),
        code("mv.plot_provider_ladder(data, GPU, LADDER_DAYS);"),
        md(
            "## Price by rental tier\n\n"
            "Spot capacity can be interrupted and is cheaper; serverless is billed per second "
            "and carries a premium. Only on-demand, secure and community prices feed the "
            "index."
        ),
        code("mv.plot_tiers(data, GPU, TIER_DAYS);"),
        md("## Distribution of price changes"),
        code("mv.plot_return_distribution(data, GPU, RETURN_HORIZON_DAYS);"),
    ]
    return nb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    for name, build in (
        ("01_market_overview.ipynb", overview),
        ("02_gpu_deep_dive.ipynb", deep_dive),
    ):
        nb = build()
        nb.metadata["kernelspec"] = {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        }
        if args.execute:
            from nbconvert.preprocessors import ExecutePreprocessor

            ExecutePreprocessor(timeout=600).preprocess(nb, {"metadata": {"path": str(HERE)}})
        nbf.write(nb, HERE / name)
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
