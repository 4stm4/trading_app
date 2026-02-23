from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

try:
    from entities.trading_system_run import TradingSystemRun
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.trading_system_run import TradingSystemRun

from .mapping import to_entity, to_table
from .tables import TradingSystemRunTable


class TradingSystemRunPostgresRepository:
    def __init__(self, session: Session):
        self._session = session

    def get_by_id(self, run_id: int) -> TradingSystemRun | None:
        row = self._session.get(TradingSystemRunTable, int(run_id))
        if row is None:
            return None
        return to_entity(row)

    def list_by_system(
        self,
        *,
        system_id: int,
        run_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[TradingSystemRun]:
        stmt = select(TradingSystemRunTable).where(TradingSystemRunTable.system_id == int(system_id))
        if run_type:
            stmt = stmt.where(TradingSystemRunTable.run_type == _normalize_run_type(run_type))
        if status:
            stmt = stmt.where(TradingSystemRunTable.status == _normalize_status(status))
        stmt = stmt.order_by(TradingSystemRunTable.created_at.desc(), TradingSystemRunTable.id.desc()).limit(
            max(1, int(limit))
        )
        rows: Sequence[TradingSystemRunTable] = self._session.scalars(stmt).all()
        return [to_entity(row) for row in rows]

    def add(self, run: TradingSystemRun) -> TradingSystemRun:
        row = to_table(run)
        self._session.add(row)
        self._session.flush()
        return to_entity(row)

    def start(
        self,
        *,
        system_id: int,
        run_type: str,
        system_version_id: int | None = None,
        request_json: dict[str, Any] | None = None,
    ) -> TradingSystemRun:
        now = datetime.now(timezone.utc)
        return self.add(
            TradingSystemRun(
                system_id=int(system_id),
                system_version_id=system_version_id,
                run_type=_normalize_run_type(run_type),
                status="running",
                request_json=dict(request_json or {}) if request_json is not None else None,
                started_at=now,
            )
        )

    def update_status(
        self,
        run_id: int,
        *,
        status: str,
        result_summary_json: dict[str, Any] | None = None,
        error_text: str | None = None,
        finished_at: datetime | None = None,
    ) -> TradingSystemRun | None:
        row = self._session.get(TradingSystemRunTable, int(run_id))
        if row is None:
            return None
        row.status = _normalize_status(status)
        if result_summary_json is not None:
            row.result_summary_json = dict(result_summary_json)
        if error_text is not None:
            normalized_error = str(error_text).strip()
            row.error_text = normalized_error or None
        if finished_at is not None:
            row.finished_at = finished_at
        elif row.status in {"done", "failed", "cancelled"} and row.finished_at is None:
            row.finished_at = datetime.now(timezone.utc)
        self._session.flush()
        return to_entity(row)

    def delete(self, run_id: int) -> bool:
        row = self._session.get(TradingSystemRunTable, int(run_id))
        if row is None:
            return False
        self._session.delete(row)
        self._session.flush()
        return True


def _normalize_run_type(run_type: str) -> str:
    return str(run_type or "unknown").strip().lower() or "unknown"


def _normalize_status(status: str) -> str:
    return str(status or "pending").strip().lower() or "pending"
