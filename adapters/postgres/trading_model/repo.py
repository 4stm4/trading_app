from __future__ import annotations

from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

try:
    from entities.trading_model import TradingModel
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.trading_model import TradingModel

from .mapping import to_entity, to_table
from .tables import TradingModelTable


class TradingModelPostgresRepository:
    def __init__(self, session: Session):
        self._session = session

    def get_by_id(self, model_id: int) -> TradingModel | None:
        row = self._session.get(TradingModelTable, int(model_id))
        if row is None:
            return None
        return to_entity(row)

    def get_by_key(self, key: str) -> TradingModel | None:
        stmt = select(TradingModelTable).where(TradingModelTable.key == _normalize_key(key))
        row = self._session.scalars(stmt).first()
        if row is None:
            return None
        return to_entity(row)

    def list(self, *, only_active: bool = True, limit: int | None = None) -> list[TradingModel]:
        stmt = select(TradingModelTable).order_by(TradingModelTable.key.asc())
        if only_active:
            stmt = stmt.where(TradingModelTable.is_active.is_(True))
        if limit is not None:
            stmt = stmt.limit(max(1, int(limit)))
        rows: Sequence[TradingModelTable] = self._session.scalars(stmt).all()
        return [to_entity(row) for row in rows]

    def add(self, model: TradingModel) -> TradingModel:
        row = to_table(model)
        self._session.add(row)
        self._session.flush()
        return to_entity(row)

    def upsert_by_key(self, model: TradingModel) -> TradingModel:
        normalized_key = _normalize_key(model.key)
        stmt = select(TradingModelTable).where(TradingModelTable.key == normalized_key)
        row = self._session.scalars(stmt).first()
        if row is None:
            row = to_table(model)
            row.key = normalized_key
            self._session.add(row)
        else:
            to_table(model, target=row)
            row.key = normalized_key
        self._session.flush()
        return to_entity(row)

    def delete(self, model_id: int) -> bool:
        row = self._session.get(TradingModelTable, int(model_id))
        if row is None:
            return False
        self._session.delete(row)
        self._session.flush()
        return True


def _normalize_key(value: str) -> str:
    return str(value or "").strip().lower()
