"""Greeks per unit notional, by central finite differences on the ``MarketState``.

- ``delta``: change in value per +1% move in the whole forward curve (value units).
- ``delta_notional``: futures position (in notional of R) that hedges the note:
  ``dV / dF_R`` where ``F_R = F(T) / S_0``.
- ``gamma``: change in ``delta_notional`` per +1% move.
- ``vega``: per +1 vol point. ``theta``: one calendar day. ``rho``: per +1bp in rates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from compute_options.market.state import MarketState
from compute_options.products.base import StructuredProduct

ONE_DAY = 1.0 / 365.0


@dataclass(frozen=True)
class Greeks:
    value: float
    delta: float
    delta_notional: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def greeks(
    product: StructuredProduct,
    market: MarketState,
    initial_fixing: float,
    T: float | None = None,
    bump: float = 0.01,
) -> Greeks:
    T = product.tenor if T is None else T

    def pv(m: MarketState, t: float = T) -> float:
        return product.price(m, initial_fixing, t)

    v0 = pv(market)
    up, dn = pv(market.spot_scaled(1 + bump)), pv(market.spot_scaled(1 - bump))
    fwd_r = market.forward(T) / initial_fixing
    delta_notional = (up - dn) / (2 * bump * fwd_r)
    gamma = (up - 2 * v0 + dn) / (bump**2 * fwd_r) * bump
    return Greeks(
        value=v0,
        delta=(up - dn) / 2,
        delta_notional=delta_notional,
        gamma=gamma,
        vega=(pv(market.vol_shifted(0.01)) - pv(market.vol_shifted(-0.01))) / 2,
        theta=pv(market, max(T - ONE_DAY, 0.0)) - v0,
        rho=(pv(market.rates_shifted(1e-4)) - pv(market.rates_shifted(-1e-4))) / 2,
    )
