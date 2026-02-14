"""
Торговые модели с различными параметрами риска и фильтрации
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class TradingModel:
    """Конфигурация торговой модели"""
    name: str
    description: str
    min_rr: float
    max_risk_percent: float
    min_volume_ratio: float
    atr_multiplier_stop: float
    trend_required: bool
    allow_range: bool
    min_trend_strength: float
    rsi_overbought: float
    rsi_oversold: float
    max_distance_ma50: float  # Максимальное отклонение от MA50 в %
    require_impulse: bool  # Требовать импульсный объем
    min_confidence: str  # 'low', 'medium', 'high'


# Предустановленные модели
MODELS: Dict[str, TradingModel] = {
    "conservative": TradingModel(
        name="conservative",
        description="Консервативная модель с высоким RR и строгими фильтрами",
        min_rr=2.5,
        max_risk_percent=1.0,
        min_volume_ratio=1.5,
        atr_multiplier_stop=1.5,
        trend_required=True,
        allow_range=False,
        min_trend_strength=2.0,
        rsi_overbought=70,
        rsi_oversold=30,
        max_distance_ma50=3.0,
        require_impulse=True,
        min_confidence='high'
    ),

    "high_rr": TradingModel(
        name="high_rr",
        description="Высокий RR с умеренными фильтрами",
        min_rr=2.0,
        max_risk_percent=1.5,
        min_volume_ratio=1.2,
        atr_multiplier_stop=1.2,
        trend_required=True,
        allow_range=False,
        min_trend_strength=1.5,
        rsi_overbought=70,
        rsi_oversold=30,
        max_distance_ma50=4.0,
        require_impulse=False,
        min_confidence='medium'
    ),

    "aggressive": TradingModel(
        name="aggressive",
        description="Агрессивная модель с низким RR и мягкими фильтрами",
        min_rr=1.5,
        max_risk_percent=2.0,
        min_volume_ratio=1.0,
        atr_multiplier_stop=1.0,
        trend_required=False,
        allow_range=False,
        min_trend_strength=0.5,
        rsi_overbought=75,
        rsi_oversold=25,
        max_distance_ma50=6.0,
        require_impulse=False,
        min_confidence='low'
    ),

    "scalp": TradingModel(
        name="scalp",
        description="Скальпинг с минимальным RR и быстрыми входами",
        min_rr=1.2,
        max_risk_percent=1.5,
        min_volume_ratio=0.8,
        atr_multiplier_stop=0.6,
        trend_required=False,
        allow_range=True,
        min_trend_strength=0.0,
        rsi_overbought=80,
        rsi_oversold=20,
        max_distance_ma50=8.0,
        require_impulse=False,
        min_confidence='low'
    ),

    "balanced": TradingModel(
        name="balanced",
        description="Сбалансированная модель (по умолчанию)",
        min_rr=2.0,
        max_risk_percent=1.5,
        min_volume_ratio=1.2,
        atr_multiplier_stop=1.2,
        trend_required=True,
        allow_range=False,
        min_trend_strength=1.0,
        rsi_overbought=70,
        rsi_oversold=30,
        max_distance_ma50=5.0,
        require_impulse=False,
        min_confidence='medium'
    )
}


def get_model(model_name: str) -> TradingModel:
    """
    Получить торговую модель по имени

    Args:
        model_name: Название модели

    Returns:
        TradingModel

    Raises:
        ValueError: Если модель не найдена
    """
    if model_name not in MODELS:
        available = ', '.join(MODELS.keys())
        raise ValueError(f"Модель '{model_name}' не найдена. Доступные: {available}")

    return MODELS[model_name]


def list_models() -> Dict[str, str]:
    """
    Список доступных моделей с описаниями

    Returns:
        Словарь {название: описание}
    """
    return {name: model.description for name, model in MODELS.items()}


def compare_models() -> str:
    """
    Сравнительная таблица параметров моделей

    Returns:
        Форматированная таблица
    """
    lines = []
    lines.append("=" * 100)
    lines.append("СРАВНЕНИЕ ТОРГОВЫХ МОДЕЛЕЙ")
    lines.append("=" * 100)

    # Заголовок
    header = f"{'Model':<15} {'RR':>6} {'Risk%':>6} {'Vol':>6} {'ATR':>6} {'Trend':>6} {'Range':>6} {'Conf':>8}"
    lines.append(header)
    lines.append("-" * 100)

    # Данные
    for name, model in MODELS.items():
        row = (
            f"{name:<15} "
            f"{model.min_rr:>6.1f} "
            f"{model.max_risk_percent:>6.1f} "
            f"{model.min_volume_ratio:>6.1f} "
            f"{model.atr_multiplier_stop:>6.1f} "
            f"{'Yes' if model.trend_required else 'No':>6} "
            f"{'Yes' if model.allow_range else 'No':>6} "
            f"{model.min_confidence:>8}"
        )
        lines.append(row)

    lines.append("=" * 100)

    return "\n".join(lines)
