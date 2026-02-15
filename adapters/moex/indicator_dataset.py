"""
MOEX market data loader с обогащением индикаторами.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

from services.strategy_engine.indicators import add_indicators, calculate_volume_stats

from .iss_client import MOEXAdapter


logger = logging.getLogger(__name__)


def load_data_with_indicators(
    ticker: str,
    timeframe: str = "1h",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    board: str = "TQBR",
    ma_periods: Optional[list] = None,
    rsi_period: int = 14,
    adapter: Optional[MOEXAdapter] = None,
    limit: Optional[int] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Загружает данные из MOEX и добавляет технические индикаторы.
    """
    if adapter is None:
        adapter = MOEXAdapter()

    if ma_periods is None:
        ma_periods = [50, 200]

    logger.info("Загрузка данных для %s на таймфрейме %s", ticker, timeframe)

    try:
        df = adapter.get_candles(
            ticker=ticker,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            board=board,
            limit=limit,
        )

        if df.empty:
            logger.warning("Нет данных для %s", ticker)
            return pd.DataFrame(), {}

        logger.info("Загружено %d свечей для %s", len(df), ticker)

        df_with_indicators = add_indicators(
            df=df,
            ma_periods=ma_periods,
            rsi_period=rsi_period,
        )
        volume_stats = calculate_volume_stats(df_with_indicators)

        logger.info(
            "Индикаторы рассчитаны. Средний объем: %.0f",
            volume_stats.get("avg_volume", 0.0),
        )
        return df_with_indicators, volume_stats

    except Exception:
        logger.exception("Ошибка загрузки данных с индикаторами для %s", ticker)
        raise


__all__ = ["load_data_with_indicators"]

