import numpy as np
import pandas as pd
import pytest

from compute_options.data.aggregate import (
    AggregationConfig,
    build,
    composite_levels,
    price_index,
    provider_prices,
)
from compute_options.data.schemas import validate_observations
from compute_options.data.sources.gpu_pricing_tracker import GpuPricingTrackerSource
from compute_options.data.store import ObservationStore
from compute_options.data.symbology import canonical_instrument


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("H100 SXM", "H100_SXM"),
        ("h100-sxm", "H100_SXM"),
        ("H100", "H100_SXM"),
        ("A100 SXM4", "A100_SXM_80GB"),
        ("a100-sxm-80", "A100_SXM_80GB"),
        ("RTX PRO 6000 WS", "RTX_PRO_6000"),
        ("B200", "B200"),
        ("mi300x", "MI300X"),
    ],
)
def test_symbology(raw, expected):
    assert canonical_instrument(raw) == expected


def obs(rows):
    cols = ["date", "source", "provider", "instrument", "rental_type", "price"]
    df = pd.DataFrame(rows, columns=cols)
    return validate_observations(df.assign(raw_instrument=df["instrument"], n_obs=1))


def test_validation_collapses_duplicates_and_drops_bad_prices():
    df = obs(
        [
            ("2026-09-01", "s", "p", "H100_SXM", "on_demand", 2.0),
            ("2026-09-01", "s", "p", "H100_SXM", "on_demand", 4.0),
            ("2026-09-01", "s", "q", "H100_SXM", "on_demand", -1.0),
        ]
    )
    assert len(df) == 1 and df["price"].iloc[0] == 3.0


def test_validation_rejects_unknown_rental_type():
    with pytest.raises(ValueError, match="unknown rental types"):
        obs([("2026-09-01", "s", "p", "H100_SXM", "weird", 2.0)])


def test_gpu_pricing_tracker_converts_early_per_instance_prices():
    raw = pd.DataFrame(
        {
            "date": ["2026-02-15", "2026-02-18", "2026-02-18", "2026-02-20"],
            "provider": ["AWS"] * 4,
            "gpu_model": ["H100_SXM"] * 4,
            "gpu_count": [8, 8, 1, 8],
            "rental_type": ["on_demand"] * 4,
            "price_median_usd": [55.04, 55.04, 6.88, 6.88],
            "n_offers": [2] * 4,
        }
    )
    out = GpuPricingTrackerSource.parse(raw)
    assert out["price"].tolist() == pytest.approx([6.88, 6.88, 6.88])
    assert out["provider"].eq("aws").all()


def test_same_provider_from_two_sources_counts_once():
    df = obs(
        [
            ("2026-09-01", "a", "runpod", "H100_SXM", "secure", 3.0),
            ("2026-09-01", "b", "runpod", "H100_SXM", "secure", 3.2),
            ("2026-09-01", "a", "lambda", "H100_SXM", "on_demand", 4.0),
        ]
    )
    prov = provider_prices(df)
    assert len(prov) == 2
    assert prov.set_index("provider").loc["runpod", "price"] == pytest.approx(3.1)


def test_index_sources_are_not_averaged_into_composite():
    df = obs(
        [
            ("2026-09-01", "ornn", "ornn", "H100_SXM", "index", 2.5),
            *[("2026-09-01", "x", p, "H100_SXM", "on_demand", 3.0) for p in "abc"],
        ]
    )
    out = build(df, AggregationConfig(min_providers=1))
    assert out["composite_level"].loc["2026-09-01", "H100_SXM"] == 3.0
    assert out["benchmarks"].loc["2026-09-01", "ornn:H100_SXM"] == 2.5


def _index_rows(days, prices):
    rows = []
    for i, d in enumerate(pd.date_range("2026-09-01", periods=days)):
        for p, px in prices(i).items():
            if px is not None:
                rows.append((d, "x", p, "H100_SXM", "on_demand", px))
    return rows


def _fit(rows, **kw):
    cfg = AggregationConfig(min_providers=1, smoothing_days=1, **kw)
    prov = provider_prices(obs(rows), cfg)
    return price_index(prov, composite_levels(prov, cfg), cfg).set_index("date")["price"]


BASE = {"a": 2.0, "b": 2.2, "c": 2.4, "d": 2.6, "e": 2.8}


def test_index_ignores_providers_entering_and_leaving():
    # A cheap provider quotes only on days 3-5: the level composite drops, true prices don't.
    rows = _index_rows(8, lambda i: BASE | {"cheap": 1.0 if 3 <= i <= 5 else None})
    idx = _fit(rows)
    assert np.allclose(idx, idx.iloc[0], rtol=1e-6)


def test_index_follows_sticky_repricings():
    # One provider of five reprices +50% on day 4 and holds: a median would stay flat.
    rows = _index_rows(8, lambda i: BASE | {"a": 3.0 if i >= 4 else 2.0})
    idx = _fit(rows)
    assert idx.iloc[-1] / idx.iloc[0] == pytest.approx(np.exp(np.log(1.5) / 5), rel=1e-3)


def test_index_counts_moves_made_while_a_provider_was_missing():
    # Provider "a" disappears for days 3-5 and returns 50% higher; a day-to-day chain misses it.
    rows = _index_rows(9, lambda i: BASE | {"a": None if 3 <= i <= 5 else (3.0 if i > 5 else 2.0)})
    idx = _fit(rows)
    assert idx.iloc[-1] / idx.iloc[0] == pytest.approx(np.exp(np.log(1.5) / 5), rel=1e-3)


def test_index_limits_a_single_bad_print():
    # Provider "a" reports a per-instance price (x8) on one day.
    rows = _index_rows(8, lambda i: BASE | {"a": 16.0 if i == 4 else 2.0})
    clipped = np.log(_fit(rows, clip=0.3)).diff().abs().max()
    unclipped = np.log(_fit(rows, clip=10.0)).diff().abs().max()
    assert unclipped == pytest.approx(np.log(8) / 5, rel=0.05)
    assert clipped < 0.1 < unclipped


def test_store_upsert_is_idempotent(tmp_path):
    store = ObservationStore(tmp_path)
    df = obs([("2026-09-01", "s", "p", "H100_SXM", "on_demand", 2.0)])
    store.upsert("s", df)
    newer = obs(
        [
            ("2026-09-01", "s", "p", "H100_SXM", "on_demand", 2.5),
            ("2026-09-02", "s", "p", "H100_SXM", "on_demand", 2.6),
        ]
    )
    assert store.upsert("s", newer) == 2
    got = store.read()
    assert got["price"].tolist() == [2.5, 2.6]
    assert store.last_date("s") == pd.Timestamp("2026-09-02")
