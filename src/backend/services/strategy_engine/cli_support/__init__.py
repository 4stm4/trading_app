"""CLI-focused API for trading command-line tools."""

from ..backtest import compare_models_results, run_backtest
from ..models import MODELS, compare_models, get_model
from ..signals import generate_signal

__all__ = [
    "MODELS",
    "get_model",
    "compare_models",
    "generate_signal",
    "run_backtest",
    "compare_models_results",
]
