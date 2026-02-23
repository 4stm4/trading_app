from __future__ import annotations

from typing import Any, Sequence

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

try:
    from entities.trading_system_version import TradingSystemVersion
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.trading_system_version import TradingSystemVersion

from .mapping import to_entity, to_table
from .tables import TradingSystemVersionTable


class TradingSystemVersionPostgresRepository:
    def __init__(self, session: Session):
        self._session = session

    def get_by_id(self, version_id: int) -> TradingSystemVersion | None:
        row = self._session.get(TradingSystemVersionTable, int(version_id))
        if row is None:
            return None
        return to_entity(row)

    def get_current(self, *, system_id: int) -> TradingSystemVersion | None:
        stmt = (
            select(TradingSystemVersionTable)
            .where(
                TradingSystemVersionTable.system_id == int(system_id),
                TradingSystemVersionTable.is_current.is_(True),
            )
            .order_by(TradingSystemVersionTable.version.desc())
        )
        row = self._session.scalars(stmt).first()
        if row is None:
            return None
        return to_entity(row)

    def list_by_system(self, *, system_id: int, limit: int = 30) -> list[TradingSystemVersion]:
        stmt = (
            select(TradingSystemVersionTable)
            .where(TradingSystemVersionTable.system_id == int(system_id))
            .order_by(TradingSystemVersionTable.version.desc())
            .limit(max(1, int(limit)))
        )
        rows: Sequence[TradingSystemVersionTable] = self._session.scalars(stmt).all()
        return [to_entity(row) for row in rows]

    def add(self, system_version: TradingSystemVersion) -> TradingSystemVersion:
        if system_version.is_current:
            self._unset_current(system_version.system_id)
        row = to_table(system_version)
        self._session.add(row)
        self._session.flush()
        return to_entity(row)

    def create_next(
        self,
        *,
        system_id: int,
        config_json: dict[str, Any] | None,
        created_by_user_id: int | None = None,
        make_current: bool = True,
    ) -> TradingSystemVersion:
        next_version = self._next_version(system_id)
        entity = TradingSystemVersion(
            system_id=int(system_id),
            version=next_version,
            config_json=dict(config_json or {}),
            is_current=bool(make_current),
            created_by_user_id=created_by_user_id,
        )
        return self.add(entity)

    def mark_current(self, version_id: int) -> TradingSystemVersion | None:
        row = self._session.get(TradingSystemVersionTable, int(version_id))
        if row is None:
            return None
        self._unset_current(row.system_id)
        row.is_current = True
        self._session.flush()
        return to_entity(row)

    def delete(self, version_id: int) -> bool:
        row = self._session.get(TradingSystemVersionTable, int(version_id))
        if row is None:
            return False
        self._session.delete(row)
        self._session.flush()
        return True

    def _next_version(self, system_id: int) -> int:
        stmt = select(func.max(TradingSystemVersionTable.version)).where(
            TradingSystemVersionTable.system_id == int(system_id)
        )
        max_version = self._session.scalar(stmt)
        return int(max_version or 0) + 1

    def _unset_current(self, system_id: int) -> None:
        stmt = (
            update(TradingSystemVersionTable)
            .where(
                TradingSystemVersionTable.system_id == int(system_id),
                TradingSystemVersionTable.is_current.is_(True),
            )
            .values(is_current=False)
        )
        self._session.execute(stmt)
