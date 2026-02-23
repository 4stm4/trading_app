"""
Общие технические индикаторы и агрегаты по OHLCV-данным.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd


def calculate_ma(df: pd.DataFrame, column: str = "close", period: int = 50) -> pd.Series:
    """
    Рассчитывает простую скользящую среднюю (SMA).
    """
    return df[column].rolling(window=period, min_periods=period).mean()


def calculate_rsi(df: pd.DataFrame, column: str = "close", period: int = 14) -> pd.Series:
    """
    Рассчитывает индекс относительной силы (RSI) по формуле Wilder (RMA),
    что соответствует дефолтному RSI в TradingView.
    """
    delta = df[column].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing (RMA): alpha = 1 / period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    rsi = 100 - (100 / (1 + rs))

    # Спецслучаи нулевых движений.
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    rsi = rsi.where(~((avg_gain > 0) & (avg_loss == 0)), 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss > 0)), 0.0)
    return rsi


def calculate_volume_stats(df: pd.DataFrame) -> Dict[str, float]:
    """
    Простейшая статистика по объему для датасета.
    """
    if df.empty or "volume" not in df.columns:
        return {
            "total_volume": 0,
            "avg_volume": 0,
            "max_volume": 0,
            "min_volume": 0,
        }

    return {
        "total_volume": float(df["volume"].sum()),
        "avg_volume": float(df["volume"].mean()),
        "max_volume": float(df["volume"].max()),
        "min_volume": float(df["volume"].min()),
        "current_volume": float(df["volume"].iloc[-1]) if len(df) > 0 else 0,
    }


def add_indicators(
    df: pd.DataFrame,
    ma_periods: Optional[list] = None,
    rsi_period: int = 14,
) -> pd.DataFrame:
    """
    Добавляет MA и RSI в DataFrame.
    """
    if ma_periods is None:
        ma_periods = [50, 200]

    df_with_indicators = df.copy()
    for period in ma_periods:
        col_name = f"ma{period}"
        df_with_indicators[col_name] = calculate_ma(df_with_indicators, period=period)

    df_with_indicators["rsi"] = calculate_rsi(df_with_indicators, period=rsi_period)
    return df_with_indicators


def get_latest_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Возвращает сводку последних значений индикаторов.
    """
    if df.empty:
        return {}

    latest = df.iloc[-1]
    signals: Dict[str, Any] = {
        "timestamp": latest.name if hasattr(latest, "name") else None,
        "close": float(latest["close"]),
        "volume": float(latest["volume"]),
    }

    if "ma50" in df.columns and not pd.isna(latest["ma50"]):
        signals["ma50"] = float(latest["ma50"])
        signals["price_vs_ma50"] = "above" if latest["close"] > latest["ma50"] else "below"

    if "ma200" in df.columns and not pd.isna(latest["ma200"]):
        signals["ma200"] = float(latest["ma200"])
        signals["price_vs_ma200"] = "above" if latest["close"] > latest["ma200"] else "below"

    if "ma50" in df.columns and "ma200" in df.columns:
        if not pd.isna(latest["ma50"]) and not pd.isna(latest["ma200"]):
            signals["ma_cross"] = "golden_cross" if latest["ma50"] > latest["ma200"] else "death_cross"

    if "rsi" in df.columns and not pd.isna(latest["rsi"]):
        rsi_value = float(latest["rsi"])
        signals["rsi"] = rsi_value
        if rsi_value > 70:
            signals["rsi_signal"] = "overbought"
        elif rsi_value < 30:
            signals["rsi_signal"] = "oversold"
        else:
            signals["rsi_signal"] = "neutral"

    return signals
