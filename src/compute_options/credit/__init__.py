"""Credit-linked compute transactions: hazard curves, CVA with wrong-way risk, GPU-backed loans."""

from compute_options.credit.hazard import HazardCurve, cds_par_spread
from compute_options.credit.trades import (
    CDS_ARCHETYPES,
    Counterparty,
    CreditMarket,
    CVAResult,
    ForwardStrip,
    GPULoan,
    issuer_spread,
    loan_fair_spread,
    make_counterparty,
    prepaid_discount,
    strip_cva,
)

__all__ = [
    "CDS_ARCHETYPES",
    "CVAResult",
    "Counterparty",
    "CreditMarket",
    "ForwardStrip",
    "GPULoan",
    "HazardCurve",
    "cds_par_spread",
    "issuer_spread",
    "loan_fair_spread",
    "make_counterparty",
    "prepaid_discount",
    "strip_cva",
]
