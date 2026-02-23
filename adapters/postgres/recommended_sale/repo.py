from __future__ import annotations

from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

try:
    from entities.recommended_sale import RecommendedSale
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.recommended_sale import RecommendedSale

from .mapping import to_entity, to_table
from .tables import RecommendedSaleTable


class RecommendedSalePostgresRepository:
    def __init__(self, session: Session):
        self._session = session

    def get_by_id(self, sale_id: int) -> RecommendedSale | None:
        row = self._session.get(RecommendedSaleTable, sale_id)
        if row is None:
            return None
        return to_entity(row)

    def get_latest(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str | None = None,
    ) -> RecommendedSale | None:
        stmt = select(RecommendedSaleTable).where(
            RecommendedSaleTable.exchange == _normalize_exchange(exchange),
            RecommendedSaleTable.symbol == _normalize_symbol(symbol),
        )
        if timeframe:
            stmt = stmt.where(RecommendedSaleTable.timeframe == _normalize_timeframe(timeframe))
        stmt = stmt.order_by(
            RecommendedSaleTable.recommended_at.desc(),
            RecommendedSaleTable.id.desc(),
        )
        row = self._session.scalars(stmt).first()
        if row is None:
            return None
        return to_entity(row)

    def list(
        self,
        *,
        exchange: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[RecommendedSale]:
        stmt = select(RecommendedSaleTable)
        if exchange:
            stmt = stmt.where(RecommendedSaleTable.exchange == _normalize_exchange(exchange))
        if symbol:
            stmt = stmt.where(RecommendedSaleTable.symbol == _normalize_symbol(symbol))
        if timeframe:
            stmt = stmt.where(RecommendedSaleTable.timeframe == _normalize_timeframe(timeframe))
        if status:
            stmt = stmt.where(RecommendedSaleTable.status == _normalize_status(status))

        stmt = stmt.order_by(
            RecommendedSaleTable.recommended_at.desc(),
            RecommendedSaleTable.id.desc(),
        ).limit(max(1, int(limit)))
        rows: Sequence[RecommendedSaleTable] = self._session.scalars(stmt).all()
        return [to_entity(row) for row in rows]

    def add(self, sale: RecommendedSale) -> RecommendedSale:
        row = to_table(sale)
        self._session.add(row)
        self._session.flush()
        return to_entity(row)

    def update_status(self, sale_id: int, *, status: str, note: str | None = None) -> RecommendedSale | None:
        row = self._session.get(RecommendedSaleTable, sale_id)
        if row is None:
            return None

        row.status = _normalize_status(status)
        if note is not None:
            normalized_note = str(note).strip()
            row.note = normalized_note or None
        self._session.flush()
        return to_entity(row)

    def delete(self, sale_id: int) -> bool:
        row = self._session.get(RecommendedSaleTable, sale_id)
        if row is None:
            return False
        self._session.delete(row)
        self._session.flush()
        return True


def _normalize_exchange(exchange: str) -> str:
    return str(exchange or "").strip().lower()


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def _normalize_timeframe(timeframe: str) -> str:
    return str(timeframe or "1h").strip().lower() or "1h"


def _normalize_status(status: str) -> str:
    return str(status or "new").strip().lower() or "new"
