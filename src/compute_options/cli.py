"""Command line interface.

compute-options sources                      list data sources
compute-options download [-s ornn ...]       fetch raw data (+ FRED rates)
compute-options aggregate                    build composite series per instrument
compute-options series                       show the latest aggregated prices
compute-options price configs/products/x.yaml
compute-options backtest configs/products/x.yaml [--synthetic 50]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from compute_options.config import PROJECT_ROOT, load_yaml
from compute_options.data.store import ObservationStore

log = logging.getLogger("compute_options")


def cmd_sources(args: argparse.Namespace) -> None:
    from compute_options.data.sources import REGISTRY

    store = ObservationStore(args.data_dir)
    for name, cls in sorted(REGISTRY.items()):
        last = store.last_date(name)
        kind = "benchmark" if cls.is_index else ("history" if cls.has_history else "snapshot")
        print(
            f"{name:<22} {kind:<10} last={last.date() if last is not None else '-':<11} "
            f"licence={cls.license:<16} {cls.url}"
        )


def cmd_download(args: argparse.Namespace) -> None:
    from compute_options.data.rates import fetch_rates
    from compute_options.data.sources import REGISTRY, get_source

    store = ObservationStore(args.data_dir)
    failures = []
    for name in args.source or sorted(REGISTRY):
        src = get_source(name)
        last = store.last_date(name)
        # Re-fetch a few days back so late revisions to recent data are picked up.
        since = (last - pd.Timedelta(days=3)).date() if last is not None and not args.full else None
        try:
            df = src.fetch(since=since)
        except Exception as e:  # keep going: one broken source shouldn't block the rest
            failures.append(name)
            print(f"{name:<22} FAILED: {e}")
            continue
        total = store.upsert(name, df)
        span = f"{df['date'].min().date()}..{df['date'].max().date()}" if len(df) else "no new rows"
        print(f"{name:<22} {len(df):>6} rows fetched ({span}), {total} stored")
    if not args.no_rates:
        try:
            rates = fetch_rates()
            store.write_frame("rates", rates)
            last_rate = rates.dropna(how="all").index[-1].date()
            print(f"{'fred_rates':<22} {len(rates):>6} rows, last {last_rate}")
        except Exception as e:
            failures.append("fred_rates")
            print(f"{'fred_rates':<22} FAILED: {e}")
    if failures:
        sys.exit(1)


def cmd_aggregate(args: argparse.Namespace) -> None:
    from compute_options.data.aggregate import AggregationConfig, build

    store = ObservationStore(args.data_dir)
    obs = store.read()
    if obs.empty:
        sys.exit("no raw data; run `compute-options download` first")
    cfg = AggregationConfig(min_providers=args.min_providers)
    for name, frame in build(obs, cfg).items():
        if not frame.empty:
            store.write_frame(name, frame)
    print(f"aggregated {len(obs)} observations from {obs['source'].nunique()} sources")
    cmd_series(args)


def cmd_series(args: argparse.Namespace) -> None:
    store = ObservationStore(args.data_dir)
    level = store.read_frame("composite_level")
    n_prov = store.read_frame("composite_n_providers")
    index = store.read_frame("composite_index")
    rows = []
    for inst in level.columns:
        s = level[inst].dropna()
        rows.append(
            {
                "instrument": inst,
                "first": s.index[0].date(),
                "last": s.index[-1].date(),
                "days": len(s),
                "providers(last)": int(n_prov[inst].dropna().iloc[-1]),
                "level": s.iloc[-1],
                "index": index[inst].dropna().iloc[-1] if inst in index else float("nan"),
            }
        )
    table = pd.DataFrame(rows).sort_values("days", ascending=False).set_index("instrument")
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))
    try:
        bench = store.read_frame("benchmarks")
        print("\nbenchmarks (latest):")
        print(bench.ffill().iloc[-1].to_string(float_format=lambda x: f"{x:.3f}"))
    except FileNotFoundError:
        pass


def load_underlying(store: ObservationStore, spec: dict, args: argparse.Namespace) -> pd.Series:
    """``{"instrument": "H100_SXM", "series": "composite_index"}`` -> price series.
    ``--series`` / ``--instrument`` on the command line override the config."""
    spec = {
        **spec,
        **{k: getattr(args, k) for k in ("series", "instrument") if getattr(args, k, None)},
    }
    frame = store.read_frame(spec.get("series", "composite_index"))
    inst = spec["instrument"]
    if inst not in frame:
        sys.exit(f"{inst} not in {spec.get('series')}; available: {list(frame.columns)}")
    return frame[inst].dropna()


def load_rates(store: ObservationStore, market_cfg: dict) -> pd.DataFrame | None:
    if market_cfg.get("rate") is not None:
        return None
    return store.read_frame("rates")


def cmd_price(args: argparse.Namespace) -> None:
    from compute_options.analytics.reports import fmt_frame, fmt_kv
    from compute_options.backtest.market_builder import MarketBuilder, MarketModelConfig
    from compute_options.products import build_product, solve
    from compute_options.risk import greeks, scenario_grid

    cfg = load_yaml(args.config)
    store = ObservationStore(args.data_dir)
    series = load_underlying(store, cfg["underlying"], args)
    builder = MarketBuilder(
        series, MarketModelConfig(**cfg.get("market", {})), load_rates(store, cfg.get("market", {}))
    )
    date = pd.Timestamp(args.date) if args.date else series.index[-1]
    market, s0 = builder.state(date), builder.spot(date)

    product = build_product(cfg["product"])
    solve_for = cfg.get("solve_for", product.default_solve_for)
    target = cfg.get("issue_price", 1.0) - cfg.get("issuer_margin", 0.0)
    if solve_for:
        product = solve(product, market, s0, target=target, field=solve_for)

    T = product.tenor
    print(f"{product.kind} on {series.name} as of {date.date()}")
    print(
        fmt_kv(
            {
                "initial fixing ($/GPU-hr)": s0,
                f"forward at {T:g}y": market.forward(T),
                "implied vol (ATM)": market.implied_vol(T, 1.0),
                "issuer discount factor": market.df(T),
            }
        )
    )
    print("\nterms" + (f" (solved {solve_for} for fair value {target:.2%})" if solve_for else ""))
    print(fmt_kv(product.terms()))
    print("\nreplication")
    print(fmt_frame(pd.DataFrame(product.leg_breakdown(market, s0)).set_index("leg")))
    print("\ngreeks (per unit notional)")
    print(fmt_kv(greeks(product, market, s0).as_dict()))
    print("\nscenario values (rows: vol shift, cols: index move)")
    print(fmt_frame(scenario_grid(product, market, s0)))


def cmd_backtest(args: argparse.Namespace) -> None:
    from compute_options.analytics.reports import fmt_kv
    from compute_options.backtest import BacktestConfig, run_backtest
    from compute_options.backtest.metrics import summarize
    from compute_options.backtest.synthetic import run_synthetic
    from compute_options.market.calibration import fit_ou, fit_two_factor

    cfg_raw = load_yaml(args.config)
    store = ObservationStore(args.data_dir)
    series = load_underlying(store, cfg_raw["underlying"], args)
    bt_cfg = BacktestConfig.from_dict(
        {k: v for k, v in cfg_raw.items() if k in BacktestConfig.__dataclass_fields__}
        | cfg_raw.get("backtest", {})
    )
    out_dir = PROJECT_ROOT / "outputs" / Path(args.config).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        params = fit_two_factor(series) if args.model == "two_factor" else fit_ou(series)
        if bt_cfg.market.rate is None:
            # Simulated dates have no FRED history: hold today's 1y yield flat.
            one_year = float(store.read_frame("rates")["DGS1"].dropna().iloc[-1])
            bt_cfg = replace(bt_cfg, market=replace(bt_cfg.market, rate=one_year))
        if args.sigma_long is not None and args.model == "two_factor":
            params = replace(params, sigma_long=args.sigma_long)
        print(f"{args.model} model on {series.name}: {params}")
        print("  (a few months of history barely identify long-run vol; set --sigma-long to test)")
        notes = run_synthetic(bt_cfg, params, float(series.iloc[-1]), args.days, args.synthetic)
        notes.to_csv(out_dir / "synthetic_notes.csv", index=False)
        print(f"\nsynthetic backtest: {args.synthetic} histories x {args.days} days")
        print(fmt_kv(summarize(notes), pct_keys=("return", "prob", "pnl", "margin")))
        print(f"\nwrote {out_dir / 'synthetic_notes.csv'}")
        return

    res = run_backtest(series, bt_cfg, load_rates(store, asdict(bt_cfg.market)))
    if res.notes.empty:
        sys.exit(f"no notes issued; skipped: {res.skipped[:3]}")
    res.notes.to_csv(out_dir / "notes.csv", index=False)
    pd.concat(res.paths, names=["note_id"]).to_csv(out_dir / "paths.csv")
    print(
        f"historical backtest on {series.name} "
        f"({series.index[0].date()}..{series.index[-1].date()})"
    )
    print(fmt_kv(res.summary, pct_keys=("return", "prob", "pnl", "margin")))
    if res.summary.get("n_matured", 0) == 0:
        print(
            "\nno note has matured yet: history is shorter than the tenor. Returns above are "
            "mark-to-market; try a shorter tenor or --synthetic."
        )
    cols = [
        c
        for c in (
            "note_id",
            "matured",
            "final_R",
            "investor_return",
            "issuer_margin",
            "issuer_hedged_pnl",
        )
        if c in res.notes
    ]
    print("\n" + res.notes[cols].tail(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {out_dir}/notes.csv and paths.csv")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="compute-options",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir", default=None, help="override data directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("sources").set_defaults(fn=cmd_sources)

    d = sub.add_parser("download")
    d.add_argument("-s", "--source", action="append", help="source name (repeatable); default all")
    d.add_argument("--full", action="store_true", help="refetch full history")
    d.add_argument("--no-rates", action="store_true")
    d.set_defaults(fn=cmd_download)

    a = sub.add_parser("aggregate")
    a.add_argument("--min-providers", type=int, default=3)
    a.set_defaults(fn=cmd_aggregate)

    sub.add_parser("series").set_defaults(fn=cmd_series)

    pr = sub.add_parser("price")
    pr.add_argument("config")
    pr.add_argument("--date", help="valuation date (default: last available)")
    for sp in (pr,):
        sp.add_argument("--series", help="override underlying series, e.g. composite_index")
        sp.add_argument("--instrument", help="override underlying instrument, e.g. B200")
    pr.set_defaults(fn=cmd_price)

    b = sub.add_parser("backtest")
    b.add_argument("config")
    b.add_argument("--synthetic", type=int, default=0, help="number of simulated histories")
    b.add_argument("--days", type=int, default=730, help="length of each simulated history")
    b.add_argument(
        "--model",
        choices=["two_factor", "ou"],
        default="two_factor",
        help="dynamics for --synthetic",
    )
    b.add_argument("--sigma-long", type=float, help="override the fitted persistent vol")
    b.add_argument("--series", help="override underlying series, e.g. composite_index")
    b.add_argument("--instrument", help="override underlying instrument, e.g. B200")
    b.set_defaults(fn=cmd_backtest)

    args = p.parse_args(argv)
    if args.data_dir is None:
        from compute_options.config import DATA_DIR

        args.data_dir = DATA_DIR
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.fn(args)


if __name__ == "__main__":
    main()
