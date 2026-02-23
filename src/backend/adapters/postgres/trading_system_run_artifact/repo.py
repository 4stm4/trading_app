from __future__ import annotations

from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

try:
    from entities.trading_system_run_artifact import TradingSystemRunArtifact
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.trading_system_run_artifact import TradingSystemRunArtifact

from .mapping import to_entity, to_table
from .tables import TradingSystemRunArtifactTable


class TradingSystemRunArtifactPostgresRepository:
    def __init__(self, session: Session):
        self._session = session

    def get_by_id(self, artifact_id: int) -> TradingSystemRunArtifact | None:
        row = self._session.get(TradingSystemRunArtifactTable, int(artifact_id))
        if row is None:
            return None
        return to_entity(row)

    def list_by_run(
        self,
        *,
        owner_user_id: int | None = None,
        run_id: int,
        artifact_type: str | None = None,
        limit: int = 50,
    ) -> list[TradingSystemRunArtifact]:
        stmt = select(TradingSystemRunArtifactTable).where(TradingSystemRunArtifactTable.run_id == int(run_id))
        if owner_user_id is not None:
            stmt = stmt.where(TradingSystemRunArtifactTable.owner_user_id == int(owner_user_id))
        if artifact_type:
            stmt = stmt.where(TradingSystemRunArtifactTable.artifact_type == _normalize_artifact_type(artifact_type))
        stmt = stmt.order_by(TradingSystemRunArtifactTable.created_at.desc(), TradingSystemRunArtifactTable.id.desc())
        stmt = stmt.limit(max(1, int(limit)))
        rows: Sequence[TradingSystemRunArtifactTable] = self._session.scalars(stmt).all()
        return [to_entity(row) for row in rows]

    def add(self, artifact: TradingSystemRunArtifact) -> TradingSystemRunArtifact:
        row = to_table(artifact)
        self._session.add(row)
        self._session.flush()
        return to_entity(row)

    def upsert_by_run_type(self, artifact: TradingSystemRunArtifact) -> TradingSystemRunArtifact:
        normalized_type = _normalize_artifact_type(artifact.artifact_type)
        stmt = select(TradingSystemRunArtifactTable).where(
            TradingSystemRunArtifactTable.run_id == int(artifact.run_id),
            TradingSystemRunArtifactTable.artifact_type == normalized_type,
        )
        row = self._session.scalars(stmt).first()
        if row is None:
            row = to_table(artifact)
            row.artifact_type = normalized_type
            self._session.add(row)
        else:
            to_table(artifact, target=row)
            row.artifact_type = normalized_type
        self._session.flush()
        return to_entity(row)

    def delete(self, artifact_id: int) -> bool:
        row = self._session.get(TradingSystemRunArtifactTable, int(artifact_id))
        if row is None:
            return False
        self._session.delete(row)
        self._session.flush()
        return True


def _normalize_artifact_type(value: str) -> str:
    return str(value or "unknown").strip().lower() or "unknown"
