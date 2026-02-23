from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session, sessionmaker

from adapters import build_exchange_adapter, resolve_default_board
from adapters.postgres.instrument.repo import InstrumentPostgresRepository
from adapters.postgres.market_candle.repo import MarketCandlePostgresRepository
from entities.instrument import Instrument


logger = logging.getLogger(__name__)


@dataclass
class EtlConfig:
    exchange: str = "moex"
    engine: str = "stock"
    market: str = "shares"
    board: str | None = None
    timeframe: str = "1h"
    symbols: list[str] | None = None
    max_symbols: int | None = None
    initial_lookback_days: int = 365
    limit: int | None = None
    start_date: str | None = None
    end_date: str | None = None
    sync_instruments: bool = True


@dataclass
class SymbolEtlResult:
    symbol: str
    rows_loaded: int
    inserted: int
    updated: int
    start_date_used: str | None
    status: str
    error: str | None = None


@dataclass
class EtlRunResult:
    exchange: str
    engine: str
    market: str
    board: str
    timeframe: str
    started_at: datetime
    finished_at: datetime
    symbols_total: int
    symbols_ok: int
    symbols_failed: int
    inserted_total: int
    updated_total: int
    instruments_synced: int
    details: list[SymbolEtlResult]


class MarketDataEtlService:
    def __init__(self, session_factory: sessionmaker[Session]):
        self._session_factory = session_factory

    def run(self, config: EtlConfig) -> EtlRunResult:
        started_at = datetime.utcnow()
        exchange = str(config.exchange or "moex").strip().lower()
        engine = str(config.engine or "stock").strip().lower()
        market = str(config.market or "shares").strip().lower()
        board = (config.board or resolve_default_board(exchange, engine)).strip().upper()
        timeframe = str(config.timeframe or "1h").strip().lower()
        initial_lookback_days = max(1, int(config.initial_lookback_days))

        adapter = build_exchange_adapter(exchange, engine, market)
        symbols = self._resolve_symbols(
            adapter=adapter,
            board=board,
            explicit_symbols=config.symbols,
            max_symbols=config.max_symbols,
        )
        logger.info("ETL start: exchange=%s board=%s timeframe=%s symbols=%d", exchange, board, timeframe, len(symbols))

        details: list[SymbolEtlResult] = []
        inserted_total = 0
        updated_total = 0
        symbols_ok = 0
        instruments_synced = 0

        with self._session_factory() as session:
            instrument_repo = InstrumentPostgresRepository(session)
            candle_repo = MarketCandlePostgresRepository(session)

            if config.sync_instruments:
                instruments_synced = self._sync_instruments(
                    adapter=adapter,
                    board=board,
                    exchange=exchange,
                    instrument_repo=instrument_repo,
                )
                session.commit()

            for symbol in symbols:
                try:
                    result = self._sync_symbol(
                        adapter=adapter,
                        candle_repo=candle_repo,
                        exchange=exchange,
                        engine=engine,
                        market=market,
                        board=board,
                        timeframe=timeframe,
                        symbol=symbol,
                        start_date=config.start_date,
                        end_date=config.end_date,
                        limit=config.limit,
                        initial_lookback_days=initial_lookback_days,
                    )
                    session.commit()
                except Exception as exc:  # noqa: BLE001
                    session.rollback()
                    logger.exception("ETL failed for %s", symbol)
                    result = SymbolEtlResult(
                        symbol=symbol,
                        rows_loaded=0,
                        inserted=0,
                        updated=0,
                        start_date_used=config.start_date,
                        status="error",
                        error=str(exc),
                    )
                details.append(result)
                if result.status == "ok":
                    symbols_ok += 1
                    inserted_total += result.inserted
                    updated_total += result.updated

        finished_at = datetime.utcnow()
        return EtlRunResult(
            exchange=exchange,
            engine=engine,
            market=market,
            board=board,
            timeframe=timeframe,
            started_at=started_at,
            finished_at=finished_at,
            symbols_total=len(symbols),
            symbols_ok=symbols_ok,
            symbols_failed=len(symbols) - symbols_ok,
            inserted_total=inserted_total,
            updated_total=updated_total,
            instruments_synced=instruments_synced,
            details=details,
        )

    def _resolve_symbols(
        self,
        *,
        adapter: Any,
        board: str,
        explicit_symbols: list[str] | None,
        max_symbols: int | None,
    ) -> list[str]:
        if explicit_symbols:
            symbols = [str(item).strip().upper() for item in explicit_symbols if str(item).strip()]
            unique = list(dict.fromkeys(symbols))
        else:
            getter = getattr(adapter, "get_securities", None)
            if not callable(getter):
                raise ValueError("Symbols are required for this exchange: pass --symbols")
            frame = getter(board=board)
            unique = self._extract_symbols(frame)

        if max_symbols is not None:
            unique = unique[: max(1, int(max_symbols))]
        return unique

    def _sync_instruments(
        self,
        *,
        adapter: Any,
        board: str,
        exchange: str,
        instrument_repo: InstrumentPostgresRepository,
    ) -> int:
        getter = getattr(adapter, "get_securities", None)
        if not callable(getter):
            return 0

        frame = getter(board=board)
        if frame is None or frame.empty:
            return 0

        synced = 0
        for _, row in frame.iterrows():
            symbol = str(row.get("SECID", "")).strip().upper()
            if not symbol:
                continue
            name = str(row.get("NAME", "")).strip() or str(row.get("SHORTNAME", "")).strip() or symbol
            lot_size = _safe_int(row.get("LOTSIZE"))
            currency = str(row.get("CURRENCY", "RUB")).strip().upper() or "RUB"

            instrument_repo.upsert(
                Instrument(
                    exchange=exchange,
                    symbol=symbol,
                    name=name,
                    board=board,
                    lot_size=lot_size,
                    currency=currency,
                    is_active=True,
                )
            )
            synced += 1

        logger.info("Instruments synced: %d", synced)
        return synced

    def _sync_symbol(
        self,
        *,
        adapter: Any,
        candle_repo: MarketCandlePostgresRepository,
        exchange: str,
        engine: str,
        market: str,
        board: str,
        timeframe: str,
        symbol: str,
        start_date: str | None,
        end_date: str | None,
        limit: int | None,
        initial_lookback_days: int,
    ) -> SymbolEtlResult:
        start_date_used = start_date
        if not start_date_used:
            last_ts = candle_repo.get_last_timestamp(
                exchange=exchange,
                engine=engine,
                market=market,
                board=board,
                symbol=symbol,
                timeframe=timeframe,
            )
            if last_ts is None:
                base_dt = datetime.utcnow() - timedelta(days=initial_lookback_days)
                start_date_used = base_dt.strftime("%Y-%m-%d")
            else:
                # Small overlap window prevents gaps when exchange updates the latest candle.
                overlap = last_ts - timedelta(days=1)
                start_date_used = overlap.strftime("%Y-%m-%d")

        frame = adapter.get_candles(
            ticker=symbol,
            timeframe=timeframe,
            start_date=start_date_used,
            end_date=end_date,
            board=board,
            limit=limit,
        )
        if frame is None or frame.empty:
            return SymbolEtlResult(
                symbol=symbol,
                rows_loaded=0,
                inserted=0,
                updated=0,
                start_date_used=start_date_used,
                status="ok",
            )

        inserted, updated = candle_repo.upsert_from_dataframe(
            exchange=exchange,
            engine=engine,
            market=market,
            board=board,
            symbol=symbol,
            timeframe=timeframe,
            frame=frame,
        )

        return SymbolEtlResult(
            symbol=symbol,
            rows_loaded=int(len(frame)),
            inserted=inserted,
            updated=updated,
            start_date_used=start_date_used,
            status="ok",
        )

    @staticmethod
    def _extract_symbols(frame: pd.DataFrame) -> list[str]:
        if frame is None or frame.empty:
            return []
        symbols = [str(value).strip().upper() for value in frame.get("SECID", []) if str(value).strip()]
        return list(dict.fromkeys(symbols))


def _safe_int(value: Any) -> int | None:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None
