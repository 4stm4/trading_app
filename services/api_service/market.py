"""Facade exports for market-related API use-cases."""

from .market_dashboard import (
    build_dashboard_backtest_response,
    build_dashboard_market_response,
    build_dashboard_robustness_response,
)
from .market_instruments import build_candles_response, build_moex_instruments_response
from .market_strategy import build_backtest_response, build_optimize_response, build_signal_response

__all__ = [
    "build_moex_instruments_response",
    "build_candles_response",
    "build_dashboard_market_response",
    "build_dashboard_backtest_response",
    "build_dashboard_robustness_response",
    "build_signal_response",
    "build_backtest_response",
    "build_optimize_response",
]
