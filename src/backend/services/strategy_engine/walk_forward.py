"""
Walk-forward тестирование стратегии
"""

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from .backtest import BacktestResults, run_backtest
from .models import TradingModel
from .monte_carlo import MonteCarloResult, run_monte_carlo
from .robustness import RobustnessMetrics, evaluate_robustness


@dataclass
class WalkForwardResult:
    """Результат walk-forward проверки"""
    model_name: str
    train_results: BacktestResults
    test_results: BacktestResults
    robustness: RobustnessMetrics
    monte_carlo: Optional[MonteCarloResult] = None
    enabled_regimes: Optional[list[str]] = None
    disabled_regimes: Optional[dict[str, str]] = None
    regime_train_performance: Optional[dict[str, dict[str, float]]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'model_name': self.model_name,
            'train': self.train_results.to_dict(),
            'test': self.test_results.to_dict(),
            'robustness': self.robustness.to_dict(),
            'monte_carlo': self.monte_carlo.to_dict() if self.monte_carlo else None,
            'enabled_regimes': self.enabled_regimes or [],
            'disabled_regimes': self.disabled_regimes or {},
            'regime_train_performance': self.regime_train_performance or {},
        }


def split_train_test(df: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df

    split_idx = int(len(df) * train_ratio)
    split_idx = max(1, min(split_idx, len(df) - 1))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def run_walk_forward(
    df: pd.DataFrame,
    signal_generator,
    deposit: float,
    model: TradingModel,
    signal_kwargs: Optional[dict] = None,
    lookback_window: int = 300,
    max_holding_candles: int = 50,
    train_ratio: float = 0.7,
    monte_carlo_simulations: int = 0,
    risk_constraints: Optional[dict[str, float]] = None,
    execution_config: Optional[dict[str, Any]] = None,
    cost_config: Optional[dict[str, Any]] = None,
) -> WalkForwardResult:
    """
    1) Train (70%): оптимизация/оценка модели
    2) Test (30%): out-of-sample проверка
    """
    train_df, test_df = split_train_test(df, train_ratio=train_ratio)
    train_lookback = _adaptive_lookback(len(train_df), lookback_window, max_holding_candles)
    test_lookback = _adaptive_lookback(len(test_df), lookback_window, max_holding_candles)

    train_results = run_backtest(
        df=train_df,
        signal_generator=signal_generator,
        deposit=deposit,
        model=model,
        lookback_window=train_lookback,
        max_holding_candles=max_holding_candles,
        signal_kwargs=signal_kwargs,
        debug_filters=False,
        risk_constraints=risk_constraints,
        execution_config=execution_config,
        cost_config=cost_config,
    )

    test_results = run_backtest(
        df=test_df,
        signal_generator=signal_generator,
        deposit=deposit,
        model=model,
        lookback_window=test_lookback,
        max_holding_candles=max_holding_candles,
        signal_kwargs=signal_kwargs,
        debug_filters=False,
        risk_constraints=risk_constraints,
        execution_config=execution_config,
        cost_config=cost_config,
    )

    monte_carlo_result = None
    if monte_carlo_simulations > 0:
        monte_carlo_result = run_monte_carlo(
            trades=test_results.trades,
            initial_balance=deposit,
            simulations=monte_carlo_simulations
        )

    robustness = evaluate_robustness(
        pf_train=train_results.profit_factor,
        pf_test=test_results.profit_factor,
        maxdd_test=test_results.max_drawdown_percent,
        monte_carlo=monte_carlo_result
    )

    return WalkForwardResult(
        model_name=model.name,
        train_results=train_results,
        test_results=test_results,
        robustness=robustness,
        monte_carlo=monte_carlo_result
    )


def _adaptive_lookback(df_len: int, default_lookback: int, max_holding_candles: int) -> int:
    """
    Подбирает lookback для коротких выборок, чтобы test не был пустым.
    """
    if df_len <= max_holding_candles + 20:
        return 20

    max_allowed = max(20, df_len - max_holding_candles - 1)
    return min(default_lookback, max_allowed)
