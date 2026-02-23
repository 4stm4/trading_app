"""Public stable API for backend HTTP services.

This module intentionally exposes only the subset required by `services.api_service`.
"""

from ..backtest import run_backtest
from ..backtest_engine import evaluate_model
from ..core import calculate_atr, calculate_structure
from ..models import MODELS, get_model
from ..regime_detection import classify_market_regime
from ..risk import calculate_kelly_criterion
from ..signals import generate_signal

__all__ = [
    "MODELS",
    "get_model",
    "generate_signal",
    "run_backtest",
    "evaluate_model",
    "calculate_atr",
    "calculate_structure",
    "classify_market_regime",
    "calculate_kelly_criterion",
]
