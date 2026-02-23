from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

import pandas as pd
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

try:
    from entities.market_candle import MarketCandle
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.market_candle import MarketCandle

from .mapping import to_entity, to_table
from .tables import MarketCandleTable


class MarketCandlePostgresRepository:
    def __init__(self, session: Session):
        self._session = session

    def get_last_timestamp(
        self,
        *,
        exchange: str,
        engine: str,
        market: str,
        board: str | None,
        symbol: str,
        timeframe: str,
    ) -> datetime | None:
        stmt = (
            select(func.max(MarketCandleTable.timestamp))
            .where(
                MarketCandleTable.exchange == _normalize_exchange(exchange),
                MarketCandleTable.engine == _normalize_engine(engine),
                MarketCandleTable.market == _normalize_market(market),
                MarketCandleTable.board == _normalize_board(board),
                MarketCandleTable.symbol == _normalize_symbol(symbol),
                MarketCandleTable.timeframe == _normalize_timeframe(timeframe),
            )
        )
        return self._session.scalar(stmt)

    def list(
        self,
        *,
        exchange: str,
        engine: str,
        market: str,
        board: str | None,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> list[MarketCandle]:
        stmt = (
            select(MarketCandleTable)
            .where(
                MarketCandleTable.exchange == _normalize_exchange(exchange),
                MarketCandleTable.engine == _normalize_engine(engine),
                MarketCandleTable.market == _normalize_market(market),
                MarketCandleTable.board == _normalize_board(board),
                MarketCandleTable.symbol == _normalize_symbol(symbol),
                MarketCandleTable.timeframe == _normalize_timeframe(timeframe),
            )
            .order_by(MarketCandleTable.timestamp.desc())
            .limit(max(1, int(limit)))
        )
        rows: Sequence[MarketCandleTable] = self._session.scalars(stmt).all()
        return [to_entity(row) for row in rows]

    def get_frame(
        self,
        *,
        exchange: str,
        engine: str,
        market: str,
        board: str | None,
        symbol: str,
        timeframe: str,
        limit: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        stmt = select(MarketCandleTable).where(
            MarketCandleTable.exchange == _normalize_exchange(exchange),
            MarketCandleTable.engine == _normalize_engine(engine),
            MarketCandleTable.market == _normalize_market(market),
            MarketCandleTable.board == _normalize_board(board),
            MarketCandleTable.symbol == _normalize_symbol(symbol),
            MarketCandleTable.timeframe == _normalize_timeframe(timeframe),
        )

        start_ts = _parse_start_datetime(start_date)
        if start_ts is not None:
            stmt = stmt.where(MarketCandleTable.timestamp >= start_ts)

        end_ts = _parse_end_datetime(end_date)
        if end_ts is not None:
            stmt = stmt.where(MarketCandleTable.timestamp <= end_ts)

        # Keep API semantics: when limit is set, return the latest candles.
        if limit is not None and int(limit) > 0:
            stmt = stmt.order_by(MarketCandleTable.timestamp.desc()).limit(int(limit))
            rows: Sequence[MarketCandleTable] = self._session.scalars(stmt).all()
            rows = sorted(rows, key=lambda item: item.timestamp)
        else:
            stmt = stmt.order_by(MarketCandleTable.timestamp.asc())
            rows = self._session.scalars(stmt).all()

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        frame = pd.DataFrame(
            [
                {
                    "timestamp": row.timestamp,
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": float(row.volume),
                }
                for row in rows
            ]
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["timestamp"])
        frame = frame.set_index("timestamp")
        frame = frame.sort_index()
        frame = frame[~frame.index.duplicated(keep="last")]
        return frame[["open", "high", "low", "close", "volume"]]

    def upsert_many(self, candles: list[MarketCandle]) -> tuple[int, int]:
        if not candles:
            return 0, 0

        sample = candles[0]
        timestamps = [c.timestamp for c in candles]
        min_ts = min(timestamps)
        max_ts = max(timestamps)

        existing_stmt = select(MarketCandleTable).where(
            MarketCandleTable.exchange == _normalize_exchange(sample.exchange),
            MarketCandleTable.engine == _normalize_engine(sample.engine),
            MarketCandleTable.market == _normalize_market(sample.market),
            MarketCandleTable.board == _normalize_board(sample.board),
            MarketCandleTable.symbol == _normalize_symbol(sample.symbol),
            MarketCandleTable.timeframe == _normalize_timeframe(sample.timeframe),
            and_(
                MarketCandleTable.timestamp >= min_ts,
                MarketCandleTable.timestamp <= max_ts,
            ),
        )
        existing_rows = self._session.scalars(existing_stmt).all()
        existing_by_ts = {row.timestamp: row for row in existing_rows}

        inserted = 0
        updated = 0

        for candle in candles:
            existing = existing_by_ts.get(candle.timestamp)
            if existing is None:
                self._session.add(to_table(candle))
                inserted += 1
            else:
                to_table(candle, target=existing)
                updated += 1

        self._session.flush()
        return inserted, updated

    def upsert_from_dataframe(
        self,
        *,
        exchange: str,
        engine: str,
        market: str,
        board: str | None,
        symbol: str,
        timeframe: str,
        frame: pd.DataFrame,
    ) -> tuple[int, int]:
        candles = _to_candles(
            exchange=exchange,
            engine=engine,
            market=market,
            board=board,
            symbol=symbol,
            timeframe=timeframe,
            frame=frame,
        )
        return self.upsert_many(candles)


def _to_candles(
    *,
    exchange: str,
    engine: str,
    market: str,
    board: str | None,
    symbol: str,
    timeframe: str,
    frame: pd.DataFrame,
) -> list[MarketCandle]:
    if frame.empty:
        return []

    normalized = frame.copy()
    normalized = normalized.sort_index()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    normalized = normalized.dropna(subset=required)

    candles: list[MarketCandle] = []
    for timestamp, row in normalized.iterrows():
        ts = _normalize_timestamp(timestamp)
        if ts is None:
            continue
        candles.append(
            MarketCandle(
                exchange=exchange,
                engine=engine,
                market=market,
                board=board,
                symbol=symbol,
                timeframe=timeframe,
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
        )
    return candles


def _normalize_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().replace(tzinfo=None)
    try:
        parsed = pd.to_datetime(value)
    except Exception:  # noqa: BLE001
        return None
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime().replace(tzinfo=None)


def _normalize_exchange(exchange: str) -> str:
    return str(exchange or "").strip().lower()


def _normalize_engine(engine: str) -> str:
    return str(engine or "stock").strip().lower() or "stock"


def _normalize_market(market: str) -> str:
    return str(market or "shares").strip().lower() or "shares"


def _normalize_board(board: str | None) -> str:
    return str(board or "").strip().upper()


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def _normalize_timeframe(timeframe: str) -> str:
    return str(timeframe or "").strip().lower()


def _parse_start_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    parsed_ts = pd.Timestamp(parsed)
    return parsed_ts.to_pydatetime().replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)


def _parse_end_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    parsed_ts = pd.Timestamp(parsed)
    # inclusive end of day for date-only inputs
    end_dt = parsed_ts.to_pydatetime().replace(tzinfo=None, hour=23, minute=59, second=59, microsecond=999999)
    return end_dt
