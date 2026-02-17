"""
Binance market data loader with indicators enrichment.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

from services.strategy_engine.indicators import add_indicators, calculate_volume_stats

from .client import BinanceAdapter


logger = logging.getLogger(__name__)


def load_data_with_indicators(
    ticker: str,
    timeframe: str = "1h",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    board: str = "",
    ma_periods: Optional[list] = None,
    rsi_period: int = 14,
    adapter: Optional[BinanceAdapter] = None,
    limit: Optional[int] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if adapter is None:
        adapter = BinanceAdapter()

    if ma_periods is None:
        ma_periods = [50, 200]

    logger.info("Loading Binance data for %s on timeframe %s", ticker, timeframe)

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
            logger.warning("No Binance data for %s", ticker)
            return pd.DataFrame(), {}

        df_with_indicators = add_indicators(
            df=df,
            ma_periods=ma_periods,
            rsi_period=rsi_period,
        )
        volume_stats = calculate_volume_stats(df_with_indicators)
        return df_with_indicators, volume_stats

    except Exception:
        logger.exception("Failed to load Binance data with indicators for %s", ticker)
        raise


__all__ = ["load_data_with_indicators"]

