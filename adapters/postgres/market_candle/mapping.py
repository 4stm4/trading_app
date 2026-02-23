from __future__ import annotations

try:
    from entities.market_candle import MarketCandle
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.market_candle import MarketCandle

from .tables import MarketCandleTable


def to_entity(table_row: MarketCandleTable) -> MarketCandle:
    return MarketCandle(
        id=table_row.id,
        exchange=table_row.exchange,
        engine=table_row.engine,
        market=table_row.market,
        board=table_row.board or None,
        symbol=table_row.symbol,
        timeframe=table_row.timeframe,
        timestamp=table_row.timestamp,
        open=table_row.open,
        high=table_row.high,
        low=table_row.low,
        close=table_row.close,
        volume=table_row.volume,
        created_at=table_row.created_at,
        updated_at=table_row.updated_at,
    )


def to_table(candle: MarketCandle, target: MarketCandleTable | None = None) -> MarketCandleTable:
    table_row = target or MarketCandleTable()
    table_row.exchange = str(candle.exchange or "").strip().lower()
    table_row.engine = str(candle.engine or "stock").strip().lower() or "stock"
    table_row.market = str(candle.market or "shares").strip().lower() or "shares"
    table_row.board = str(candle.board or "").strip().upper()
    table_row.symbol = str(candle.symbol or "").strip().upper()
    table_row.timeframe = str(candle.timeframe or "").strip().lower()
    table_row.timestamp = candle.timestamp

    table_row.open = float(candle.open)
    table_row.high = float(candle.high)
    table_row.low = float(candle.low)
    table_row.close = float(candle.close)
    table_row.volume = float(candle.volume)
    return table_row
