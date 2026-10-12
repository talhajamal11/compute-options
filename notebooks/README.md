# Notebooks

One notebook per structure: `01_zero_coupon_note` … `06_tracker_note`.

Each notebook:

1. **Parameters** (one cell): `UNDERLYING` (e.g. `"H100_SXM"`, `"B200"`), `TENOR` (`"3M"`,
   `"6M"`, `"12M"`, …), `TERMS` (participation, protection, buffer, cap, barrier, strikes; set
   one to `"solve"` to have it solved, or fix them all to price as given), issuer margin,
   expected-drift view, holding period and market model.
2. **Pricing today**: solved terms, fair value and issuer margin, **expected APR** of holding to
   maturity (model and historical), replication by options, Greeks.
3. **Payoff** at maturity, with the historical distribution of index moves over the tenor.
4. **Backtest: holding the structure** from `HOLD_START` to `HOLD_END`, rolling at maturity:
   annualised APR, vol, Sharpe, Sortino, max drawdown, Calmar, compared with the index and
   T-bills; value and drawdown chart; every note held, with the issuer's P&L on it.

```bash
uv run compute-options download && uv run compute-options aggregate   # refresh data
uv run jupyter lab                                                    # edit parameters and run
uv run python notebooks/make_notebooks.py --execute                   # regenerate with fresh outputs
```

Notebooks are generated from `make_notebooks.py` (defaults live there); shared logic is in
`compute_options.analytics.notebook`.

**Data caveat.** Public H100 history starts 2026-02-11 and comes from posted list prices, which
are sparse and sticky. The first note can be priced once 21 days of prices exist (early March).
Annualised statistics come from about 7 months of data, so read them as illustrative.

## Market data notebooks (`notebooks/market/`)

| Notebook | Shows |
|---|---|
| `01_market_overview.ipynb` | Latest prices and changes for every GPU, price history per GPU (index, average posted, Ornn), all GPUs rebased to 100, rolling volatility, posted-vs-transacted premium, data coverage |
| `02_gpu_deep_dive.ipynb` | One GPU (`GPU = "H100_SXM"`): every provider's daily price against the indices, price by provider, price by rental tier, distribution of price changes |

```bash
uv run python notebooks/market/make_market_notebooks.py --execute   # regenerate
```

Chart helpers live in `compute_options.analytics.market_viz`. Each GPU keeps one colour on
every chart (from a colour-blind-validated palette), so charts show up to 8 GPUs; the summary
table lists all of them.
