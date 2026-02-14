"""
Ядро торговой системы - общие функции анализа рынка
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Расчет Average True Range (ATR)

    Args:
        df: DataFrame с OHLC данными
        period: Период ATR

    Returns:
        Series с значениями ATR
    """
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def calculate_structure(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Определение структуры рынка: HH/HL (восходящий тренд) или LH/LL (нисходящий тренд)

    Args:
        df: DataFrame с данными
        lookback: Количество свечей для анализа структуры

    Returns:
        Словарь с информацией о структуре
    """
    if len(df) < lookback * 2:
        return {
            'structure': 'unknown',
            'phase': 'unknown',
            'trend_strength': 0,
            'breakout': False,
            'last_swing_high': None,
            'last_swing_low': None
        }

    # Находим свинг-максимумы и минимумы
    df_tail = df.tail(lookback * 3).copy()

    # Простой алгоритм поиска свингов
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(df_tail) - lookback):
        # Свинг-максимум
        if df_tail['high'].iloc[i] == df_tail['high'].iloc[i-lookback:i+lookback+1].max():
            swing_highs.append((df_tail.index[i], df_tail['high'].iloc[i]))

        # Свинг-минимум
        if df_tail['low'].iloc[i] == df_tail['low'].iloc[i-lookback:i+lookback+1].min():
            swing_lows.append((df_tail.index[i], df_tail['low'].iloc[i]))

    # Определяем структуру
    structure = 'range'
    trend_strength = 0

    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        # Проверяем последние 2 максимума и 2 минимума
        if swing_highs[-1][1] > swing_highs[-2][1] and swing_lows[-1][1] > swing_lows[-2][1]:
            structure = 'uptrend'  # HH и HL
            trend_strength = (swing_highs[-1][1] - swing_highs[-2][1]) / swing_highs[-2][1] * 100
        elif swing_highs[-1][1] < swing_highs[-2][1] and swing_lows[-1][1] < swing_lows[-2][1]:
            structure = 'downtrend'  # LH и LL
            trend_strength = (swing_highs[-2][1] - swing_highs[-1][1]) / swing_highs[-2][1] * 100

    # Определяем фазу
    current_price = df['close'].iloc[-1]
    ma50 = df['ma50'].iloc[-1] if 'ma50' in df.columns and not pd.isna(df['ma50'].iloc[-1]) else None
    ma200 = df['ma200'].iloc[-1] if 'ma200' in df.columns and not pd.isna(df['ma200'].iloc[-1]) else None

    phase = 'unknown'
    if ma50 is not None and ma200 is not None:
        if structure == 'uptrend':
            if current_price < ma50:
                phase = 'pullback'
            else:
                phase = 'trend'
        elif structure == 'downtrend':
            if current_price > ma50:
                phase = 'pullback'
            else:
                phase = 'trend'
        else:
            phase = 'range'

    # Проверка пробоя структуры
    breakout = False
    if swing_highs and swing_lows:
        last_swing_high = swing_highs[-1][1] if swing_highs else None
        last_swing_low = swing_lows[-1][1] if swing_lows else None

        if structure == 'downtrend' and last_swing_high and current_price > last_swing_high:
            breakout = True
        elif structure == 'uptrend' and last_swing_low and current_price < last_swing_low:
            breakout = True
    else:
        last_swing_high = None
        last_swing_low = None

    return {
        'structure': structure,
        'phase': phase,
        'trend_strength': abs(trend_strength),
        'breakout': breakout,
        'last_swing_high': swing_highs[-1][1] if swing_highs else None,
        'last_swing_low': swing_lows[-1][1] if swing_lows else None,
        'swing_highs_count': len(swing_highs),
        'swing_lows_count': len(swing_lows)
    }


def calculate_distance_to_ma(price: float, ma_value: Optional[float]) -> float:
    """
    Расчет процентного расстояния от цены до MA

    Args:
        price: Текущая цена
        ma_value: Значение MA

    Returns:
        Процентное отклонение (положительное - выше MA, отрицательное - ниже)
    """
    if ma_value is None or pd.isna(ma_value) or ma_value == 0:
        return 0.0

    return ((price - ma_value) / ma_value) * 100


def calculate_volume_stats(df: pd.DataFrame, period: int = 20) -> Dict:
    """
    Анализ объемов

    Args:
        df: DataFrame с данными
        period: Период для расчета среднего объема

    Returns:
        Словарь со статистикой объемов
    """
    if 'volume' not in df.columns or len(df) < period:
        return {
            'avg_volume': 0,
            'current_volume': 0,
            'volume_ratio': 0,
            'is_impulse': False
        }

    avg_volume = df['volume'].tail(period).mean()
    current_volume = df['volume'].iloc[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

    # Импульсная свеча - объем > 1.5x среднего
    is_impulse = volume_ratio > 1.5

    return {
        'avg_volume': avg_volume,
        'current_volume': current_volume,
        'volume_ratio': volume_ratio,
        'is_impulse': is_impulse
    }
