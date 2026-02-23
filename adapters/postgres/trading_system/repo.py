from __future__ import annotations

from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

try:
    from entities.trading_system import TradingSystem
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.trading_system import TradingSystem

from .mapping import to_entity, to_table
from .tables import TradingSystemTable


class TradingSystemPostgresRepository:
    def __init__(self, session: Session):
        self._session = session

    def get_by_id(self, system_id: int) -> TradingSystem | None:
        row = self._session.get(TradingSystemTable, system_id)
        if row is None:
            return None
        return to_entity(row)

    def get_by_owner_and_name(self, *, owner_user_id: int, name: str) -> TradingSystem | None:
        stmt = select(TradingSystemTable).where(
            TradingSystemTable.owner_user_id == int(owner_user_id),
            TradingSystemTable.name == _normalize_name(name),
        )
        row = self._session.scalars(stmt).first()
        if row is None:
            return None
        return to_entity(row)

    def list_by_owner(
        self,
        *,
        owner_user_id: int,
        only_active: bool = True,
        limit: int | None = None,
    ) -> list[TradingSystem]:
        stmt = select(TradingSystemTable).where(TradingSystemTable.owner_user_id == int(owner_user_id))
        if only_active:
            stmt = stmt.where(TradingSystemTable.is_active.is_(True))
        stmt = stmt.order_by(TradingSystemTable.updated_at.desc(), TradingSystemTable.id.desc())
        if limit is not None:
            stmt = stmt.limit(max(1, int(limit)))

        rows: Sequence[TradingSystemTable] = self._session.scalars(stmt).all()
        return [to_entity(row) for row in rows]

    def add(self, system: TradingSystem) -> TradingSystem:
        row = to_table(system)
        self._session.add(row)
        self._session.flush()
        return to_entity(row)

    def upsert_by_owner_name(self, system: TradingSystem) -> TradingSystem:
        stmt = select(TradingSystemTable).where(
            TradingSystemTable.owner_user_id == int(system.owner_user_id),
            TradingSystemTable.name == _normalize_name(system.name),
        )
        row = self._session.scalars(stmt).first()
        if row is None:
            row = to_table(system)
            self._session.add(row)
        else:
            to_table(system, target=row)
        self._session.flush()
        return to_entity(row)

    def delete(self, system_id: int) -> bool:
        row = self._session.get(TradingSystemTable, int(system_id))
        if row is None:
            return False
        self._session.delete(row)
        self._session.flush()
        return True


def _normalize_name(name: str) -> str:
    normalized = str(name or "").strip()
    return normalized or "default"
