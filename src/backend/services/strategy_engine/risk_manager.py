"""
Режимное управление риском.
"""

from dataclasses import replace

from .models import TradingModel


RISK_ALLOCATION_MULTIPLIERS = {
    "trend": 1.0,
    "range": 0.7,
    "high_volatility": 0.5,
}


def apply_regime_risk(model: TradingModel, regime: str) -> TradingModel:
    """
    Режимное распределение капитала:
    trend -> base risk
    range -> base risk * 0.7
    high_volatility -> base risk * 0.5
    """
    multiplier = RISK_ALLOCATION_MULTIPLIERS.get(regime, 1.0)
    risk = model.max_risk_percent * multiplier
    atr_multiplier = model.atr_multiplier_stop

    if regime == "high_volatility":
        atr_multiplier = model.atr_multiplier_stop * 1.2

    return replace(
        model,
        max_risk_percent=risk,
        atr_multiplier_stop=atr_multiplier
    )
