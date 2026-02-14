import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from .moex import MOEXAdapter

logger = logging.getLogger(__name__)


def calculate_ma(df: pd.DataFrame, column: str = "close", period: int = 50) -> pd.Series:
    """
    Рассчитывает простую скользящую среднюю (Moving Average)

    Args:
        df: DataFrame с данными
        column: Название колонки для расчета (по умолчанию 'close')
        period: Период MA (например, 50 или 200)

    Returns:
        Series с значениями MA
    """
    return df[column].rolling(window=period, min_periods=period).mean()


def calculate_rsi(df: pd.DataFrame, column: str = "close", period: int = 14) -> pd.Series:
    """
    Рассчитывает индекс относительной силы (RSI - Relative Strength Index)

    Args:
        df: DataFrame с данными
        column: Название колонки для расчета (по умолчанию 'close')
        period: Период RSI (по умолчанию 14)

    Returns:
        Series с значениями RSI (от 0 до 100)
    """
    delta = df[column].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_volume_stats(df: pd.DataFrame) -> Dict[str, float]:
    """
    Рассчитывает статистику по объему за период

    Args:
        df: DataFrame с данными

    Returns:
        Словарь со статистикой объема
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
    include_volume_stats: bool = True
) -> pd.DataFrame:
    """
    Добавляет технические индикаторы к DataFrame

    Args:
        df: DataFrame с OHLCV данными
        ma_periods: Список периодов для MA (по умолчанию [50, 200])
        rsi_period: Период для RSI (по умолчанию 14)
        include_volume_stats: Добавлять ли статистику по объему

    Returns:
        DataFrame с добавленными индикаторами
    """
    if ma_periods is None:
        ma_periods = [50, 200]

    df_with_indicators = df.copy()

    # Добавляем MA
    for period in ma_periods:
        col_name = f"ma{period}"
        df_with_indicators[col_name] = calculate_ma(df_with_indicators, period=period)
        logger.debug(f"Рассчитан индикатор {col_name}")

    # Добавляем RSI
    df_with_indicators["rsi"] = calculate_rsi(df_with_indicators, period=rsi_period)
    logger.debug(f"Рассчитан индикатор RSI с периодом {rsi_period}")

    return df_with_indicators


def load_data_with_indicators(
    ticker: str,
    timeframe: str = "1h",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    board: str = "TQBR",
    ma_periods: Optional[list] = None,
    rsi_period: int = 14,
    adapter: Optional[MOEXAdapter] = None
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Загружает данные из MOEX и добавляет технические индикаторы

    Args:
        ticker: Тикер инструмента (например, 'SBER', 'GAZP')
        timeframe: Таймфрейм ('1m', '5m', '10m', '15m', '30m', '1h', '4h', '1d', '1w', '1M')
        start_date: Дата начала в формате 'YYYY-MM-DD'
        end_date: Дата окончания в формате 'YYYY-MM-DD'
        board: Режим торгов (по умолчанию 'TQBR')
        ma_periods: Список периодов для MA (по умолчанию [50, 200])
        rsi_period: Период для RSI (по умолчанию 14)
        adapter: MOEXAdapter объект (если None, создастся новый)

    Returns:
        Кортеж (DataFrame с данными и индикаторами, словарь со статистикой объема)

    Example:
        >>> df, vol_stats = load_data_with_indicators('SBER', timeframe='1h')
        >>> print(df[['close', 'ma50', 'ma200', 'rsi']].tail())
        >>> print(f"Общий объем: {vol_stats['total_volume']}")
    """
    if adapter is None:
        adapter = MOEXAdapter()

    if ma_periods is None:
        ma_periods = [50, 200]

    logger.info(f"Загрузка данных для {ticker} на таймфрейме {timeframe}")

    try:
        # Загружаем данные
        df = adapter.get_candles(
            ticker=ticker,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            board=board
        )

        if df.empty:
            logger.warning(f"Нет данных для {ticker}")
            return pd.DataFrame(), {}

        logger.info(f"Загружено {len(df)} свечей для {ticker}")

        # Добавляем индикаторы
        df_with_indicators = add_indicators(
            df,
            ma_periods=ma_periods,
            rsi_period=rsi_period
        )

        # Рассчитываем статистику по объему
        volume_stats = calculate_volume_stats(df_with_indicators)

        logger.info(f"Индикаторы рассчитаны. Средний объем: {volume_stats['avg_volume']:.0f}")

        return df_with_indicators, volume_stats

    except Exception as e:
        logger.error(f"Ошибка загрузки данных с индикаторами для {ticker}: {e}")
        raise


def get_latest_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Получает последние сигналы по индикаторам

    Args:
        df: DataFrame с данными и индикаторами

    Returns:
        Словарь с последними значениями индикаторов и сигналами
    """
    if df.empty:
        return {}

    latest = df.iloc[-1]

    signals = {
        "timestamp": latest.name if hasattr(latest, "name") else None,
        "close": float(latest["close"]),
        "volume": float(latest["volume"]),
    }

    # MA сигналы
    if "ma50" in df.columns and not pd.isna(latest["ma50"]):
        signals["ma50"] = float(latest["ma50"])
        signals["price_vs_ma50"] = "above" if latest["close"] > latest["ma50"] else "below"

    if "ma200" in df.columns and not pd.isna(latest["ma200"]):
        signals["ma200"] = float(latest["ma200"])
        signals["price_vs_ma200"] = "above" if latest["close"] > latest["ma200"] else "below"

    # Золотой крест / мертвый крест
    if "ma50" in df.columns and "ma200" in df.columns:
        if not pd.isna(latest["ma50"]) and not pd.isna(latest["ma200"]):
            if latest["ma50"] > latest["ma200"]:
                signals["ma_cross"] = "golden_cross"  # Бычий сигнал
            else:
                signals["ma_cross"] = "death_cross"  # Медвежий сигнал

    # RSI сигналы
    if "rsi" in df.columns and not pd.isna(latest["rsi"]):
        rsi_value = float(latest["rsi"])
        signals["rsi"] = rsi_value

        if rsi_value > 70:
            signals["rsi_signal"] = "overbought"  # Перекупленность
        elif rsi_value < 30:
            signals["rsi_signal"] = "oversold"  # Перепроданность
        else:
            signals["rsi_signal"] = "neutral"

    return signals


if __name__ == "__main__":
    # Пример использования
    logging.basicConfig(level=logging.INFO)

    # Загружаем данные с индикаторами для Сбербанка
    ticker = "SBER"
    timeframe = "1h"

    print(f"\n{'='*60}")
    print(f"Загрузка данных для {ticker} на таймфрейме {timeframe}")
    print(f"{'='*60}\n")

    df, volume_stats = load_data_with_indicators(
        ticker=ticker,
        timeframe=timeframe,
        ma_periods=[50, 200],
        rsi_period=14
    )

    if not df.empty:
        print("Последние 5 записей с индикаторами:")
        print(df[["close", "volume", "ma50", "ma200", "rsi"]].tail())

        print(f"\n{'='*60}")
        print("Статистика объема:")
        print(f"{'='*60}")
        for key, value in volume_stats.items():
            print(f"{key:20s}: {value:,.0f}")

        print(f"\n{'='*60}")
        print("Последние сигналы:")
        print(f"{'='*60}")
        signals = get_latest_signals(df)
        for key, value in signals.items():
            print(f"{key:20s}: {value}")

        print(f"\n{'='*60}")
        print(f"Всего загружено свечей: {len(df)}")
        print(f"Период: {df.index[0]} - {df.index[-1]}")
        print(f"{'='*60}\n")
    else:
        print("Не удалось загрузить данные")
