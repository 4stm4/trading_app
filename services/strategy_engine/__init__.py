"""
Strategy Engine - универсальный движок торговых стратегий

Может использоваться со всеми адаптерами: MOEX, Binance, Alfa Invest и др.
"""

from .models import TradingModel, get_model, MODELS, list_models, compare_models
from .signals import TradingSignal, generate_signal
from .risk import (
    RiskParameters,
    calculate_position_risk,
    calculate_stop_by_atr,
    calculate_target_by_rr,
    calculate_kelly_criterion,
    get_risk_summary
)
from .backtest import (
    Trade,
    BacktestResults,
    run_backtest,
    compare_models_results
)
from .core import (
    calculate_atr,
    calculate_structure,
    calculate_distance_to_ma,
    calculate_volume_stats
)

__all__ = [
    # Models
    "TradingModel",
    "get_model",
    "MODELS",
    "list_models",
    "compare_models",
    # Signals
    "TradingSignal",
    "generate_signal",
    # Risk
    "RiskParameters",
    "calculate_position_risk",
    "calculate_stop_by_atr",
    "calculate_target_by_rr",
    "calculate_kelly_criterion",
    "get_risk_summary",
    # Backtest
    "Trade",
    "BacktestResults",
    "run_backtest",
    "compare_models_results",
    # Core
    "calculate_atr",
    "calculate_structure",
    "calculate_distance_to_ma",
    "calculate_volume_stats",
]
