"""Rolling-issuance backtests from both the investor's and the issuer's side."""

from compute_options.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from compute_options.backtest.holding import HoldingConfig, HoldingResult, run_holding
from compute_options.backtest.market_builder import MarketBuilder, MarketModelConfig
from compute_options.backtest.performance import performance_stats

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "HoldingConfig",
    "HoldingResult",
    "MarketBuilder",
    "MarketModelConfig",
    "performance_stats",
    "run_backtest",
    "run_holding",
]
