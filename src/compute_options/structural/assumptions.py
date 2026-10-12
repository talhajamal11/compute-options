"""Assumptions register for the structural cost-of-supply model.

Every number that drives the forward curve lives here with a range, a source, how well it is
evidenced and what would make us revise it. ``low / central / high`` define a triangular prior
that the Monte Carlo and the Bayesian update sample from, so the register is both the
documentation and the model input.

``basis`` says where a number came from:

- ``sourced``: read from a public source (URL given). Most are secondary sources (vendor blogs,
  press), not filings or primary data; ``confidence`` reflects that.
- ``derived``: computed from sourced numbers (e.g. server power from GPU TDP plus overhead).
- ``judgment``: no public evidence found; an analyst's range, kept wide on purpose.

Figures were gathered in October 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

Params = dict[str, "float | np.ndarray"]


@dataclass(frozen=True)
class Assumption:
    key: str
    label: str
    unit: str
    low: float
    central: float
    high: float
    basis: Literal["sourced", "derived", "judgment"]
    confidence: Literal["high", "medium", "low"]
    source: str
    url: str | None = None
    revise_if: str = ""

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.triangular(self.low, self.central, self.high, n)


_EIA = "https://www.eia.gov/TOTALENERGY/data/monthly/pdf/sec9_10.pdf"
_CLOUDZERO = "https://cloudzero.com/blog/h100-gpu-cost/"
_INTUITION = "https://intuitionlabs.ai/articles/data-center-gpu-pricing-2026"
_NV_RA = "https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html"
_COLO = "https://www.datacenterknowledge.com/colocation/power-not-space-the-colocation-battleground-in-2026"
_DDTL4 = "https://www.globaldatacenterhub.com/p/coreweaves-85b-quarter-when-gpu-debt"
_DDTL5 = "https://www.boardroomalpha.com/sec/crwv-8-k-2026-05-18-0001769628-26-000236"
_CRWV_BOND = (
    "https://realinvestmentadvice.com/resources/blog/is-coreweave-at-the-mercy-of-the-bond-market/"
)
_DEPR = "https://electroneconomics.substack.com/p/the-depreciation-clock-coreweaves"
_GPU_COLLATERAL = "https://www.wing.vc/content/how-the-gpu-became-collateral"

_COMMON: tuple[Assumption, ...] = (
    Assumption(
        "tariff",
        "Industrial electricity price",
        "$/kWh",
        0.065,
        0.087,
        0.120,
        "sourced",
        "high",
        "EIA Monthly Energy Review: US industrial average 8.71 c/kWh, May 2026 (8.13 in 2024)",
        _EIA,
        "Site-specific PPA or a move to a different market",
    ),
    Assumption(
        "pue",
        "Power usage effectiveness (facility kW per IT kW)",
        "x",
        1.15,
        1.30,
        1.50,
        "judgment",
        "low",
        "Not verified in this research pass. Modern liquid-cooled AI halls are typically quoted "
        "at 1.2-1.4; industry-wide surveys report higher",
        None,
        "Replace with operator disclosures (e.g. hyperscaler sustainability reports)",
    ),
    Assumption(
        "colo",
        "Colocation rent (space and cooling, power billed separately)",
        "$/kW-month",
        100.0,
        150.0,
        200.0,
        "sourced",
        "medium",
        "CBRE via press: ~$196/kW-month asking for 250-500 kW in primary markets (H2 2025); "
        "wholesale >1 MW reported $80-157",
        _COLO,
        "New CBRE North America Data Center Trends release",
    ),
    Assumption(
        "rate",
        "Required return on GPU capital (financing cost)",
        "per year",
        0.06,
        0.095,
        0.13,
        "sourced",
        "medium",
        "CoreWeave: DDTL 4.0 fixed tranche ~5.9% (A3, Mar 2026); unsecured notes 9.25-9.75%; "
        "5y bonds quoted near 13% in Aug 2026",
        _DDTL4,
        "Each new neocloud financing print; Fed path",
    ),
    Assumption(
        "life",
        "Economic life of a GPU server",
        "years",
        4.0,
        5.0,
        6.0,
        "sourced",
        "medium",
        "Reported accounting lives: Nebius 4y, Lambda 5y, Amazon 5y (cut from 6y in 2025), "
        "CoreWeave 6y",
        _DEPR,
        "Impairments or accelerated depreciation disclosed by a large operator",
    ),
    Assumption(
        "resid",
        "Residual value at end of life (fraction of capex)",
        "fraction",
        0.03,
        0.10,
        0.20,
        "judgment",
        "low",
        "No transaction data found for end-of-life GPU resale",
        None,
        "A liquid secondary market for used accelerators (the GPU-collateral lenders see)",
    ),
    Assumption(
        "util",
        "Utilisation (fraction of hours sold)",
        "fraction",
        0.65,
        0.80,
        0.92,
        "judgment",
        "low",
        "Operators rarely disclose fleet utilisation; set wide",
        None,
        "Operator disclosures; listed-futures volumes",
    ),
    Assumption(
        "opex_pct",
        "Other operating cost (staff, network ops, software, support)",
        "fraction of capex / year",
        0.04,
        0.06,
        0.10,
        "judgment",
        "low",
        "Analyst range; not sourced",
        None,
        "Segment margins from a listed neocloud",
    ),
    Assumption(
        "margin",
        "Long-run competitive margin over levelised cost",
        "fraction",
        0.0,
        0.30,
        1.20,
        "judgment",
        "low",
        "Entry is open to anyone with capital, so the long-run price should sit near cost plus a "
        "required margin; the width reflects how little is known about it",
        None,
        "Observed price of contracted long-term capacity",
    ),
    Assumption(
        "kappa",
        "Speed at which scarcity rent decays",
        "per year",
        0.1,
        1.5,
        4.0,
        "judgment",
        "low",
        "The data are too short to identify this; the Bayesian update narrows it modestly",
        None,
        "More than ~2 years of index history; first listed-futures curve",
    ),
    Assumption(
        "sigma_mult",
        "Volatility multiplier on realised weekly volatility",
        "x",
        0.75,
        1.00,
        1.40,
        "judgment",
        "medium",
        "Realised weekly vol is the base. Multiplier allows for noise in posted prices and "
        "regime change",
        None,
        "Option quotes (implied vol)",
    ),
    Assumption(
        "risk_premium",
        "Forward minus expected spot (annualised, negative = backwardation)",
        "per year",
        -0.04,
        0.0,
        0.08,
        "judgment",
        "low",
        "GPU-hours can't be stored, so no arbitrage ties the forward to expected spot. Suppliers "
        "selling forward to lock in revenue push forwards down; buyers locking in capacity push "
        "them up",
        None,
        "First traded forward price",
    ),
)

_PER_GPU: dict[str, tuple[Assumption, ...]] = {
    "H100_SXM": (
        Assumption(
            "capex",
            "All-in capex per GPU (server share, networking, storage)",
            "$",
            28_000.0,
            36_000.0,
            46_000.0,
            "derived",
            "medium",
            "8-GPU HGX H100 servers quoted $216k-$320k (about $27-40k per GPU) plus 15-20% "
            "for networking and storage",
            _CLOUDZERO,
            "NVIDIA price changes; used-market prices",
        ),
        Assumption(
            "it_kw",
            "IT power per GPU incl. CPU, NIC, fans",
            "kW",
            0.90,
            1.00,
            1.28,
            "derived",
            "medium",
            "H100 SXM TDP 700 W (vendor spec, not re-verified here); HGX 8-GPU system "
            "maximum around 10.2 kW",
            _NV_RA,
            "Measured fleet power draw",
        ),
        Assumption(
            "delta",
            "Annual decline of the long-run anchor from newer generations",
            "per year",
            0.05,
            0.15,
            0.30,
            "judgment",
            "low",
            "Blackwell supply lowers the equilibrium price of Hopper-class hours; "
            "tied to the substitution bound in the cost model",
            None,
            "H100/B200 price-per-performance spread",
        ),
    ),
    "B200": (
        Assumption(
            "capex",
            "All-in capex per GPU (server share, networking, storage)",
            "$",
            40_000.0,
            52_000.0,
            68_000.0,
            "derived",
            "low",
            "B200 estimated $30-50k per GPU (single reseller estimate) plus server, "
            "networking and storage",
            _INTUITION,
            "A published HGX B200 system price",
        ),
        Assumption(
            "it_kw",
            "IT power per GPU incl. CPU, NIC, fans",
            "kW",
            1.20,
            1.50,
            1.80,
            "derived",
            "medium",
            "B200 up to 1 kW per GPU per NVIDIA reference architecture; "
            "8-GPU baseboard listed at 10.2 kW",
            _NV_RA,
            "Measured fleet power draw",
        ),
        Assumption(
            "delta",
            "Annual decline of the long-run anchor from newer generations",
            "per year",
            0.03,
            0.10,
            0.20,
            "judgment",
            "low",
            "Newest generation in the sample, so it depreciates more slowly until Rubin ships",
            None,
            "Rubin timing and pricing",
        ),
    ),
}

# Used only for the generational-spread chart and substitution bound.
PERF_RATIO_B200_VS_H100 = Assumption(
    "perf_ratio",
    "B200 useful work per GPU-hour relative to H100",
    "x",
    1.8,
    2.5,
    3.2,
    "judgment",
    "low",
    "Workload-dependent (training vs inference, precision). Wide on purpose",
    None,
    "Benchmark results for the workloads customers actually run",
)

# Credit inputs (used by the credit package).
CREDIT_SOURCES = {
    "neocloud_cds_5y": (
        "CoreWeave 5y CDS about 855 bp, August 2026",
        "https://tradersunion.com/news/market-voices/show/2886667-coreweave-debt-cds-yield/",
    ),
    "gpu_loan_pricing": (
        "CoreWeave DDTL 4.0: SOFR+225 (A3); DDTL 5.0: SOFR+450; later $2.6B loan SOFR+550",
        _DDTL5,
    ),
    "gpu_collateral": ("How GPU-backed lending works", _GPU_COLLATERAL),
    "bond_yield": ("CoreWeave 5y bond yields 9.3% to ~13%", _CRWV_BOND),
}


# Dated log of changes to ranges, so it is clear what changed and why.
REVISIONS: tuple[tuple[str, str, str], ...] = (
    (
        "v1 -> v2",
        "margin high 0.60 -> 1.20, central 0.25 -> 0.30; kappa low 0.5 -> 0.1",
        "Walk-forward test: v1 forecasts were biased low by 4-10% at 30-60 days because the "
        "market paid 1.9x (H100) to 2.5x (B200) levelised cost, above the cost-plus-60% cap, "
        "and the data showed no fast reversion. Widening cut H100 60-day RMSE from 0.100 to 0.078. "
        "Caveat: the same 7 months of data motivated and scored the change, so it is in-sample.",
    ),
)


def register(gpu: str) -> list[Assumption]:
    if gpu not in _PER_GPU:
        raise KeyError(f"no structural assumptions for {gpu!r}; have {sorted(_PER_GPU)}")
    return [*_PER_GPU[gpu], *_COMMON]


def central_params(gpu: str) -> Params:
    return {a.key: a.central for a in register(gpu)}


def sample_params(gpu: str, n: int, seed: int | None = None) -> Params:
    rng = np.random.default_rng(seed)
    return {a.key: a.sample(rng, n) for a in register(gpu)}


def with_value(params: Params, **kw: float) -> Params:
    return {**params, **kw}


def register_frame(gpu: str) -> pd.DataFrame:
    rows = [
        {
            "assumption": a.label,
            "unit": a.unit,
            "low": a.low,
            "central": a.central,
            "high": a.high,
            "basis": a.basis,
            "confidence": a.confidence,
            "source": a.source,
            "url": a.url or "",
            "revise if": a.revise_if,
        }
        for a in register(gpu)
    ]
    return pd.DataFrame(rows)


def widen(a: Assumption, factor: float) -> Assumption:
    """Same central value, range scaled around it (for stress-testing the priors)."""
    return replace(
        a,
        low=a.central - (a.central - a.low) * factor,
        high=a.central + (a.high - a.central) * factor,
    )
