# compute-options

Market data, option pricing and structured-product backtesting on GPU compute rental prices.

The pipeline downloads daily GPU rental prices from every public source it can find, combines
them into one price series per GPU, prices six structured notes on those series, and backtests
rolling issuance from both the **investor's** and the **issuer's** side.

## Quick start

```bash
uv sync
uv run compute-options download          # all sources + FRED rates -> data/raw/*.parquet
uv run compute-options aggregate         # composite series per GPU -> data/processed/
uv run compute-options price configs/products/capital_protected.yaml
uv run compute-options backtest configs/products/buffered.yaml --series composite_chained --instrument B200
uv run compute-options backtest configs/products/capital_protected.yaml --synthetic 20
uv run pytest                            # unit tests; `-m network` hits live endpoints
```

Three sources only publish today's snapshot and the Ornn free tier serves a rolling three
months, so **run `download` daily** (cron or a scheduled job) to build up history. Storage upserts,
so reruns are safe.

## Layout

```
src/compute_options/
├── data/                 1. INGESTION
│   ├── sources/          one adapter per source, shared base class and registry
│   ├── schemas.py        standard observation schema: date, source, provider, instrument, tier, $/GPU-hr
│   ├── symbology.py      source GPU names -> standard ids (H100_SXM, B200, ...)
│   ├── store.py          Parquet storage: raw per source, processed frames
│   ├── aggregate.py      provider de-duplication, composite level, chain-linked index
│   └── rates.py          FRED Treasury yields and SOFR
├── market/               2. MARKET OBJECTS
│   ├── forward_curve.py  flat, Schwartz mean-reverting, interpolated from futures quotes
│   ├── vol.py            realized, EWMA, multi-day vol; flat, mean-reverting term vol; grid surface
│   ├── rates.py          discount curve plus issuer credit spread
│   ├── calibration.py    one-factor OU fit; two-factor (persistent level + transient noise) fit
│   └── state.py          MarketState: everything needed to price on one date
├── models/               3. PRICING MODELS: Black-76 (vanilla, digital, Greeks, implied vol), path simulation, Monte Carlo
├── instruments/          4. BUILDING BLOCKS: zero-coupon bond, prepaid forward, call, put, digitals
├── products/             5. STRUCTURED PRODUCTS: term-sheet payoff + replication by legs, solver
├── risk/                 6. GREEKS AND SCENARIOS: bump-and-reprice Greeks, stress grids
├── backtest/             7. BACKTESTING
│   ├── market_builder.py market on any past date from data available up to that date only
│   ├── issuance.py       solve terms at issue; lifecycle.py daily mark-to-market
│   ├── hedging.py        issuer delta hedge with maturity-matched futures
│   ├── metrics.py        investor returns, issuer margin, unhedged and hedged P&L
│   └── synthetic.py      simulated histories for when real history is shorter than the tenor
├── analytics/            8. REPORTS: payoff diagrams, text reports
└── cli.py
configs/products/         one YAML per product: underlying, term template, solved term, market model, backtest
```

## Data sources

| Source | Kind | Coverage | Licence |
|---|---|---|---|
| [Ornn OCPI](https://data.ornn.com) | transaction-based benchmark | latest value for 5 GPUs, 3 months of H100 history (free tier) | Ornn free tier |
| [cherielilili/gpu-pricing-tracker](https://github.com/cherielilili/gpu-pricing-tracker) | posted prices, ~30 providers | daily since 2026-02-11 | none stated |
| [adriannutiu/gpu-rental-prices](https://github.com/adriannutiu/gpu-rental-prices) | posted prices, ~30 providers incl. hyperscalers | daily since 2026-07-05 | CC BY 4.0 |
| [Zeno00-00/gpu-price-tracker](https://github.com/Zeno00-00/gpu-price-tracker) | Vast.ai / RunPod medians | daily since 2026-06-13 | none stated |
| [Vast.ai](https://vast.ai) API | live marketplace median | snapshot | — |
| [gpuwatch](https://gpu.watchworks.dev) | RunPod / Vast.ai | snapshot | — |
| [FRED](https://fred.stlouisfed.org) | Treasury 1M–5Y, SOFR | daily | public domain |

Paid or gated, adapter not yet written: Silicon Data SDH100RT (portal and Bloomberg), Ornn API
key tier (full history and forward curve), Compute Desk indices, and Architect's American
Innovation Exchange (compute futures and options, pending CFTC approval). Adding a source means
writing one `Source` subclass.

## Modelling notes

- **Price on futures, not spot.** GPU-hours can't be stored, so there's no cash-and-carry link
  between spot and forward. Black-76 prices off the forward curve. A tracker's participation at
  par is `1 / (DF · F/S)`, so contango costs the investor. The difference between the forward and
  spot plays the role that dividends and carry play for equities.
- **Posted prices vs transactions.** Posted list prices are sticky. Most providers go months
  without changing, so posted-price composites have low volatility (about 8 to 16% on H100) and
  move in steps. Ornn's transaction index has about 74% daily volatility, much of it noise that
  reverts within days (about 39% on weekly returns). Price options on the benchmark that futures
  will settle on (Ornn/ICE, Silicon Data/CME, Compute Desk/Architect) and use composites for
  longer history and cross-checks.
- **Chain-linking.** The level composite jumps when a cheap provider starts or stops being
  quoted. The chained index only uses price changes of providers quoted on consecutive days, and
  ignores single-provider moves larger than 2× in a day, which are unit errors such as
  per-instance vs per-GPU prices.
- **Volatility.** Use `vol_horizon_days: 7` to estimate from overlapping weekly returns. The
  two-factor model (persistent level plus transient noise) separates the two on long samples, but
  a few months of data barely identify the long-run volatility, so `--sigma-long` lets you set it
  for synthetic runs.
- **Replication is tested.** Each product states its term-sheet payoff and its replication by
  legs. Tests check that the two agree, that Monte Carlo matches the closed form, and that the
  solver reaches par.
- **Issuer hedge.** The issuer delta-hedges with a futures contract expiring at the note's
  maturity, marked at the model forward. Hedged P&L = day-one margin + hedge error (gamma, vega
  from volatility re-estimation, gap risk). There is no basis risk yet, because no real futures
  settlement prices are available.

## Products

All amounts are fractions of notional. `R` = index at maturity / index at issue.

| Product | Payoff | Replication | Default solved term |
|---|---|---|---|
| Zero Coupon Note | `redemption` | zero-coupon bond | redemption (the issuer's yield) |
| Capital Protected | `protection + p·(min(R,cap) − K)⁺` | zero-coupon bond + call spread | participation |
| Buffered | `1 + p·(min(R,cap) − 1)⁺ − g·(1 − b − R)⁺` | zero-coupon bond + call spread − put | participation |
| Dual Directional | up: `1 + p·(R−1)`; `B ≤ R < 1`: `2 − R`; `R < B`: `R` | zero-coupon bond + calls + put(1) − 2·put(B) − 2(1−B)·digital put(B) | participation / barrier |
| Risk Reversal | `1 + p·(R − Kc)⁺ − (Kp − R)⁺` (funded or unfunded) | (zero-coupon bond) + call − put | call strike |
| Tracker | `p·R` | prepaid forward | participation |

## Roadmap

- Adapters for paid sources (Ornn API key, Silicon Data) and exchange settlements once
  ICE, CME and Architect list contracts. Fit vols to option quotes with `GridVolSurface` +
  `black76.implied_vol`.
- Daily-monitored barriers (`payoff_path` + Monte Carlo) and autocallables.
- Spread products across GPUs (H100 vs B200 as a generational-depreciation trade).
