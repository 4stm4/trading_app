from __future__ import annotations

from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

try:
    from entities.trading_system_signal import TradingSystemSignal
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.trading_system_signal import TradingSystemSignal

from .mapping import to_entity, to_table
from .tables import TradingSystemSignalTable


class TradingSystemSignalPostgresRepository:
    def __init__(self, session: Session):
        self._session = session

    def get_by_id(self, signal_id: int) -> TradingSystemSignal | None:
        row = self._session.get(TradingSystemSignalTable, int(signal_id))
        if row is None:
            return None
        return to_entity(row)

    def get_latest_for_symbol(
        self,
        *,
        system_id: int,
        exchange: str,
        symbol: str,
        timeframe: str | None = None,
    ) -> TradingSystemSignal | None:
        stmt = select(TradingSystemSignalTable).where(
            TradingSystemSignalTable.system_id == int(system_id),
            TradingSystemSignalTable.exchange == _normalize_exchange(exchange),
            TradingSystemSignalTable.symbol == _normalize_symbol(symbol),
        )
        if timeframe:
            stmt = stmt.where(TradingSystemSignalTable.timeframe == _normalize_timeframe(timeframe))
        stmt = stmt.order_by(
            TradingSystemSignalTable.generated_at.desc(),
            TradingSystemSignalTable.id.desc(),
        )
        row = self._session.scalars(stmt).first()
        if row is None:
            return None
        return to_entity(row)

    def list_by_system(
        self,
        *,
        system_id: int,
        signal: str | None = None,
        limit: int = 200,
    ) -> list[TradingSystemSignal]:
        stmt = select(TradingSystemSignalTable).where(TradingSystemSignalTable.system_id == int(system_id))
        if signal:
            stmt = stmt.where(TradingSystemSignalTable.signal == _normalize_signal(signal))
        stmt = stmt.order_by(
            TradingSystemSignalTable.generated_at.desc(),
            TradingSystemSignalTable.id.desc(),
        ).limit(max(1, int(limit)))
        rows: Sequence[TradingSystemSignalTable] = self._session.scalars(stmt).all()
        return [to_entity(row) for row in rows]

    def add(self, system_signal: TradingSystemSignal) -> TradingSystemSignal:
        row = to_table(system_signal)
        self._session.add(row)
        self._session.flush()
        return to_entity(row)

    def delete(self, signal_id: int) -> bool:
        row = self._session.get(TradingSystemSignalTable, int(signal_id))
        if row is None:
            return False
        self._session.delete(row)
        self._session.flush()
        return True


def _normalize_exchange(exchange: str) -> str:
    return str(exchange or "moex").strip().lower() or "moex"


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def _normalize_timeframe(timeframe: str) -> str:
    return str(timeframe or "1h").strip().lower() or "1h"


def _normalize_signal(signal: str) -> str:
    return str(signal or "none").strip().lower() or "none"
