"""Build a ``MarketState`` on any historical date using only data available up to that date."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from compute_options.market.calibration import fit_ou
from compute_options.market.forward_curve import FlatForward, ForwardCurve, SchwartzForward
from compute_options.market.rates import DiscountCurve
from compute_options.market.state import MarketState
from compute_options.market.vol import FlatVol, MeanRevertingVol, VolModel, ewma_vol, realized_vol


@dataclass(frozen=True)
class MarketModelConfig:
    vol_method: Literal["realized", "ewma", "ou"] = "realized"
    vol_window: int = 90  # trailing calendar days used for vol estimation
    vol_horizon_days: int = 1  # return horizon for realized vol; 7 filters daily noise
    vol_floor: float = 0.10
    vol_cap: float = 2.0
    vol_markup: float = 0.0  # added to estimated vol: issuers price options above realized
    forward_model: Literal["flat", "schwartz", "structural"] = "flat"
    structural_gpu: str | None = None  # assumptions to use when forward_model == "structural"
    structural_draws: int = 1500
    kappa: float | None = None  # overrides the fitted mean-reversion speed
    min_history: int = 20  # observations needed before a date can be priced
    rate: float | None = None  # flat risk-free rate; None -> use FRED curve
    issuer_spread: float = 0.01  # issuer credit spread over risk-free


class MarketBuilder:
    def __init__(
        self,
        index: pd.Series,
        cfg: MarketModelConfig = MarketModelConfig(),
        rates: pd.DataFrame | None = None,
    ):
        self.index = index.dropna().sort_index()
        self.cfg = cfg
        self.rates = rates.sort_index().ffill() if rates is not None else None
        if cfg.rate is None and self.rates is None:
            raise ValueError("provide a flat `rate` in config or a FRED rates frame")
        # Overlapping notes value on the same dates; each date's market is built once.
        self._states: dict[pd.Timestamp, MarketState] = {}

    def history(self, date: pd.Timestamp) -> pd.Series:
        return self.index.loc[:date]

    def can_price(self, date: pd.Timestamp) -> bool:
        return len(self.history(date)) >= self.cfg.min_history

    def spot(self, date: pd.Timestamp) -> float:
        return float(self.history(date).iloc[-1])

    def _vol_and_forward(self, hist: pd.Series) -> tuple[VolModel, ForwardCurve]:
        cfg, spot = self.cfg, float(hist.iloc[-1])
        if cfg.forward_model == "structural":
            return self._structural(hist)
        kappa, theta = 0.0, float(np.log(spot))
        if cfg.vol_method == "ou" or cfg.forward_model == "schwartz":
            ou = fit_ou(hist.loc[hist.index[-1] - pd.Timedelta(days=max(cfg.vol_window, 30)) :])
            kappa = ou.kappa if cfg.kappa is None else cfg.kappa
            theta = ou.theta
        if cfg.vol_method == "ewma":
            sigma = ewma_vol(hist)
        elif cfg.vol_method == "ou":
            sigma = ou.sigma
        else:
            sigma = realized_vol(hist, cfg.vol_window, cfg.vol_horizon_days)
        sigma = float(np.clip(sigma + cfg.vol_markup, cfg.vol_floor, cfg.vol_cap))

        vol: VolModel = MeanRevertingVol(sigma, kappa) if kappa > 0 else FlatVol(sigma)
        fwd: ForwardCurve = (
            SchwartzForward(spot, kappa, theta, sigma)
            if cfg.forward_model == "schwartz"
            else FlatForward(spot)
        )
        return vol, fwd

    def _structural(self, hist: pd.Series) -> tuple[VolModel, ForwardCurve]:
        # Imported here: the structural package depends on market/, not the other way round.
        from compute_options.structural import fit_posterior, posterior_vol

        if not self.cfg.structural_gpu:
            raise ValueError("forward_model='structural' needs structural_gpu in the config")
        post = fit_posterior(hist, self.cfg.structural_gpu, n=self.cfg.structural_draws)
        vol = posterior_vol(post, self.cfg.vol_floor, self.cfg.vol_cap)
        return vol, post.curve

    def discount(self, date: pd.Timestamp) -> DiscountCurve:
        if self.cfg.rate is not None:
            return DiscountCurve.flat(self.cfg.rate, self.cfg.issuer_spread)
        assert self.rates is not None
        row = self.rates.loc[:date].iloc[-1]
        return DiscountCurve.from_fred_row(row, self.cfg.issuer_spread)

    def state(self, date: pd.Timestamp) -> MarketState:
        date = pd.Timestamp(date)
        if date not in self._states:
            self._states[date] = self._build_state(date)
        return self._states[date]

    def _build_state(self, date: pd.Timestamp) -> MarketState:
        hist = self.history(date)
        if len(hist) < self.cfg.min_history:
            raise ValueError(f"{date.date()}: {len(hist)} observations < min_history")
        vol, fwd = self._vol_and_forward(hist)
        return MarketState(date, fwd, vol, self.discount(date))

    def risk_free_rate(self, date: pd.Timestamp, T: float) -> float:
        return self.discount(date).with_spread(0.0).zero_rate(T)
