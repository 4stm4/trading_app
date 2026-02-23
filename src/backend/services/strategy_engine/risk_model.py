"""
Риск-модель: оценка вероятности разорения (Risk of Ruin).
"""

from typing import Optional


def estimate_risk_of_ruin(
    winrate_pct: float,
    rr: float,
    risk_percent: float
) -> float:
    """
    Оценка вероятности разорения в процентах.

    Args:
        winrate_pct: winrate в процентах (0..100)
        rr: отношение среднего выигрыша к среднему проигрышу
        risk_percent: риск на сделку в процентах от депозита
    """
    if risk_percent <= 0:
        return 100.0

    p = max(0.0, min(1.0, winrate_pct / 100.0))
    q = 1.0 - p
    b = max(rr, 0.0)
    risk_fraction = risk_percent / 100.0

    if p <= 0 or b <= 0:
        return 100.0

    edge = (p * b) - q
    if edge <= 0:
        return 100.0

    denominator = p * b
    if denominator <= 0:
        return 100.0

    ratio = q / denominator
    if ratio <= 0:
        return 0.0

    capital_units = max(1, int(1.0 / risk_fraction))
    ruin_prob = min(1.0, ratio ** capital_units)
    return ruin_prob * 100.0


def estimate_risk_of_ruin_from_backtest(
    winrate_pct: float,
    avg_win: float,
    avg_loss: float,
    risk_percent: float
) -> float:
    """
    Оценка risk of ruin на основе метрик бэктеста.
    """
    if avg_loss <= 0:
        return 0.0

    rr = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    return estimate_risk_of_ruin(
        winrate_pct=winrate_pct,
        rr=rr,
        risk_percent=risk_percent
    )
