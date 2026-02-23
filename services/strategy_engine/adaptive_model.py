"""
Выбор и параметризация модели под текущий режим рынка.
"""

from dataclasses import replace

from .models import TradingModel, get_model


def build_regime_model(regime: str) -> TradingModel:
    """
    Базовая модель по режиму:
    - trend -> conservative
    - range -> high_rr
    - high_volatility -> conservative (с дополнительной risk/ATR адаптацией)
    """
    if regime == "trend":
        base = get_model("conservative")
        return replace(
            base,
            name="regime_trend_conservative",
            description="Adaptive trend regime",
            min_rr=2.2,
            max_risk_percent=0.3,
            min_volume_ratio=1.2,
        )

    if regime == "range":
        base = get_model("high_rr")
        return replace(
            base,
            name="regime_range_high_rr",
            description="Adaptive range regime",
            min_rr=2.0,
            # Base risk before regime allocation (0.3 final in trend, 0.21 in range after *0.7)
            max_risk_percent=0.3,
            min_volume_ratio=1.0,
            trend_required=False,
            allow_range=True,
        )

    # high_volatility
    base = get_model("conservative")
    return replace(
        base,
        name="regime_high_volatility",
        description="Adaptive high volatility regime",
        min_rr=2.0,
        # Base risk before regime allocation (0.15 final after *0.5)
        max_risk_percent=0.3,
        min_volume_ratio=1.2,
        trend_required=False,
        atr_multiplier_stop=base.atr_multiplier_stop * 1.3,
    )
