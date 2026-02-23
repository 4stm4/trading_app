"""Facade exports for dashboard-specific market use-cases."""

from .dashboard_backtest import build_dashboard_backtest_response
from .dashboard_market import build_dashboard_market_response
from .dashboard_robustness import build_dashboard_robustness_response

__all__ = [
    "build_dashboard_market_response",
    "build_dashboard_backtest_response",
    "build_dashboard_robustness_response",
]
