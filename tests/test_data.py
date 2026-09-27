import numpy as np
import pandas as pd
import pytest

from compute_options.data.aggregate import (
    AggregationConfig,
    build,
    chained_index,
    composite_levels,
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


def test_chained_index_ignores_composition_changes_and_bad_steps():
    rows = []
    for d in pd.date_range("2026-09-01", periods=4):
        for p, px in {"a": 2.0, "b": 2.2, "c": 2.4}.items():
            rows.append((d, "x", p, "H100_SXM", "on_demand", px))
    # A cheap provider appears on day 3 (level drops, true prices unchanged) and provider "a"
    # reports a per-instance price on day 4 (x8, a data error).
    rows.append((pd.Timestamp("2026-09-03"), "x", "cheap", "H100_SXM", "on_demand", 1.0))
    rows.append((pd.Timestamp("2026-09-04"), "x", "cheap", "H100_SXM", "on_demand", 1.0))
    rows = [r for r in rows if not (r[0] == pd.Timestamp("2026-09-04") and r[2] == "a")]
    rows.append((pd.Timestamp("2026-09-04"), "x", "a", "H100_SXM", "on_demand", 16.0))
    cfg = AggregationConfig(min_providers=1, trim=0.0)
    prov = provider_prices(obs(rows), cfg)
    levels = composite_levels(prov, cfg)
    chained = chained_index(prov, levels, cfg).set_index("date")["price"]
    assert levels.set_index("date")["price"].nunique() > 1
    assert np.allclose(chained, chained.iloc[0])


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
