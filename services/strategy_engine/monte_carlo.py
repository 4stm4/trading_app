"""
Monte Carlo симуляция для оценки устойчивости стратегии
"""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class MonteCarloResult:
    """Результаты Monte Carlo симуляции"""
    simulations: int
    worst_drawdown_percent: float
    quantile_5_equity: float
    stability_score: float
    median_final_equity: float
    probability_drawdown_over_30: float

    def to_dict(self) -> dict[str, Any]:
        return {
            'simulations': self.simulations,
            'worst_drawdown_percent': round(self.worst_drawdown_percent, 2),
            'quantile_5_equity': round(self.quantile_5_equity, 2),
            'stability_score': round(self.stability_score, 4),
            'median_final_equity': round(self.median_final_equity, 2),
            'probability_drawdown_over_30': round(self.probability_drawdown_over_30, 2),
        }


def run_monte_carlo(
    trades: Sequence[Any],
    initial_balance: float,
    simulations: int = 1000,
    random_seed: int | None = None
) -> MonteCarloResult:
    """
    Monte Carlo: перемешивание последовательности сделок.
    """
    if simulations <= 0:
        return MonteCarloResult(
            simulations=0,
            worst_drawdown_percent=0,
            quantile_5_equity=initial_balance,
            stability_score=0,
            median_final_equity=initial_balance,
            probability_drawdown_over_30=0
        )

    pnls = _extract_pnls(trades)
    if pnls.size == 0:
        return MonteCarloResult(
            simulations=simulations,
            worst_drawdown_percent=0,
            quantile_5_equity=initial_balance,
            stability_score=0,
            median_final_equity=initial_balance,
            probability_drawdown_over_30=0
        )

    rng = np.random.default_rng(random_seed)
    final_equities = []
    max_drawdowns = []

    for _ in range(simulations):
        shuffled = rng.permutation(pnls)
        equity_curve = initial_balance + np.cumsum(shuffled)
        max_dd = _max_drawdown_percent(equity_curve, initial_balance)
        max_drawdowns.append(max_dd)
        final_equities.append(float(equity_curve[-1]))

    final_equities_arr = np.array(final_equities)
    max_drawdowns_arr = np.array(max_drawdowns)

    worst_dd = float(np.max(max_drawdowns_arr))
    quantile_5 = float(np.quantile(final_equities_arr, 0.05))
    median_final = float(np.median(final_equities_arr))

    positive_runs = float(np.mean(final_equities_arr > initial_balance))
    safe_dd_runs = float(np.mean(max_drawdowns_arr <= 20.0))
    stability_score = (positive_runs * 0.6) + (safe_dd_runs * 0.4)
    probability_drawdown_over_30 = float(np.mean(max_drawdowns_arr > 30.0) * 100)

    return MonteCarloResult(
        simulations=simulations,
        worst_drawdown_percent=worst_dd,
        quantile_5_equity=quantile_5,
        stability_score=stability_score,
        median_final_equity=median_final,
        probability_drawdown_over_30=probability_drawdown_over_30
    )


def _extract_pnls(trades: Sequence[Any]) -> np.ndarray:
    values = []
    for trade in trades:
        if hasattr(trade, 'pnl'):
            values.append(float(trade.pnl))
        elif isinstance(trade, dict) and 'pnl' in trade:
            values.append(float(trade['pnl']))
    return np.array(values, dtype=float)


def _max_drawdown_percent(equity_curve: np.ndarray, initial_balance: float) -> float:
    if equity_curve.size == 0:
        return 0.0

    peaks = np.maximum.accumulate(np.concatenate(([initial_balance], equity_curve)))
    curve_with_start = np.concatenate(([initial_balance], equity_curve))
    drawdowns = (peaks - curve_with_start) / np.where(peaks == 0, 1, peaks) * 100
    return float(np.max(drawdowns))
