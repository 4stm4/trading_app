from __future__ import annotations

from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

try:
    from entities.instrument import Instrument
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.instrument import Instrument

from .mapping import to_entity, to_table
from .tables import InstrumentTable


class InstrumentPostgresRepository:
    def __init__(self, session: Session):
        self._session = session

    def get_by_id(self, instrument_id: int) -> Instrument | None:
        row = self._session.get(InstrumentTable, instrument_id)
        if row is None:
            return None
        return to_entity(row)

    def get_by_symbol(self, *, exchange: str, symbol: str) -> Instrument | None:
        normalized_exchange = _normalize_exchange(exchange)
        normalized_symbol = _normalize_symbol(symbol)

        stmt = select(InstrumentTable).where(
            InstrumentTable.exchange == normalized_exchange,
            InstrumentTable.symbol == normalized_symbol,
        )
        row = self._session.scalars(stmt).first()
        if row is None:
            return None
        return to_entity(row)

    def list(
        self,
        *,
        exchange: str | None = None,
        board: str | None = None,
        only_active: bool = True,
        limit: int | None = None,
    ) -> list[Instrument]:
        stmt = select(InstrumentTable).order_by(InstrumentTable.exchange, InstrumentTable.symbol)

        if exchange:
            stmt = stmt.where(InstrumentTable.exchange == _normalize_exchange(exchange))
        if board:
            stmt = stmt.where(InstrumentTable.board == str(board).strip().upper())
        if only_active:
            stmt = stmt.where(InstrumentTable.is_active.is_(True))
        if limit is not None:
            stmt = stmt.limit(max(1, int(limit)))

        rows: Sequence[InstrumentTable] = self._session.scalars(stmt).all()
        return [to_entity(row) for row in rows]

    def add(self, instrument: Instrument) -> Instrument:
        row = to_table(instrument)
        self._session.add(row)
        self._session.flush()
        return to_entity(row)

    def upsert(self, instrument: Instrument) -> Instrument:
        normalized_exchange = _normalize_exchange(instrument.exchange)
        normalized_symbol = _normalize_symbol(instrument.symbol)

        stmt = select(InstrumentTable).where(
            InstrumentTable.exchange == normalized_exchange,
            InstrumentTable.symbol == normalized_symbol,
        )
        row = self._session.scalars(stmt).first()

        if row is None:
            row = to_table(instrument)
            row.exchange = normalized_exchange
            row.symbol = normalized_symbol
            self._session.add(row)
        else:
            to_table(instrument, target=row)
            row.exchange = normalized_exchange
            row.symbol = normalized_symbol

        self._session.flush()
        return to_entity(row)

    def delete(self, *, exchange: str, symbol: str) -> bool:
        normalized_exchange = _normalize_exchange(exchange)
        normalized_symbol = _normalize_symbol(symbol)

        stmt = select(InstrumentTable).where(
            InstrumentTable.exchange == normalized_exchange,
            InstrumentTable.symbol == normalized_symbol,
        )
        row = self._session.scalars(stmt).first()
        if row is None:
            return False

        self._session.delete(row)
        self._session.flush()
        return True


def _normalize_exchange(exchange: str) -> str:
    return str(exchange or "").strip().lower()


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()
