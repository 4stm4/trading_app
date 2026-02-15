"""
Загрузка данных для CLI с проверкой качества выборки.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .indicator_dataset import load_data_with_indicators
from .iss_client import MOEXAdapter


_MIN_SIGNIFICANT_CANDLES_BY_TF = {
    "1m": 20000,
    "5m": 12000,
    "10m": 6000,
    "15m": 5000,
    "30m": 3500,
    "1h": 2000,
    "4h": 900,
    "1d": 400,
    "1w": 200,
    "1M": 80,
}


@dataclass
class DataLoadResult:
    df: pd.DataFrame
    volume_stats: dict
    warnings: list[str]


def load_cli_dataset(
    ticker: str,
    timeframe: str,
    start_date: Optional[str],
    end_date: Optional[str],
    board: str,
    adapter: MOEXAdapter,
    limit: Optional[int] = None,
    min_significant_candles: Optional[int] = None,
) -> DataLoadResult:
    """
    Загружает данные и возвращает предупреждения о качестве выборки.
    """
    required_candles = _resolve_min_significant_candles(
        timeframe=timeframe,
        min_significant_candles=min_significant_candles,
    )

    # В CLI без явных дат стараемся получить статистически достаточный объем данных.
    # Добавляем буфер к порогу значимости, чтобы после очистки/дубликатов оставался запас.
    effective_limit = limit
    if effective_limit is None and start_date is None and end_date is None:
        effective_limit = max(int(required_candles) + 500, int(required_candles * 1.1))

    df, volume_stats = load_data_with_indicators(
        ticker=ticker,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        board=board,
        ma_periods=[50, 200],
        rsi_period=14,
        adapter=adapter,
        limit=effective_limit
    )

    warnings = []
    if not df.empty and len(df) < required_candles:
        warnings.append(
            f"⚠ Недостаточно данных для статистически значимого теста (нужно >= {required_candles} свечей)."
        )

    return DataLoadResult(df=df, volume_stats=volume_stats, warnings=warnings)


def _resolve_min_significant_candles(
    *,
    timeframe: str,
    min_significant_candles: Optional[int],
) -> int:
    if min_significant_candles is not None and int(min_significant_candles) > 0:
        return int(min_significant_candles)

    return int(_MIN_SIGNIFICANT_CANDLES_BY_TF.get(timeframe, 3000))
