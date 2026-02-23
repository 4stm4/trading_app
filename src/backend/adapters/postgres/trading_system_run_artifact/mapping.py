from __future__ import annotations

from typing import Any

try:
    from entities.trading_system_run_artifact import TradingSystemRunArtifact
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.trading_system_run_artifact import TradingSystemRunArtifact

from .tables import TradingSystemRunArtifactTable


def to_entity(table_row: TradingSystemRunArtifactTable) -> TradingSystemRunArtifact:
    return TradingSystemRunArtifact(
        id=table_row.id,
        owner_user_id=table_row.owner_user_id,
        run_id=table_row.run_id,
        system_id=table_row.system_id,
        system_version_id=table_row.system_version_id,
        artifact_type=table_row.artifact_type,
        payload_json=_normalize_payload(table_row.payload_json),
        created_at=table_row.created_at,
    )


def to_table(
    artifact: TradingSystemRunArtifact,
    target: TradingSystemRunArtifactTable | None = None,
) -> TradingSystemRunArtifactTable:
    table_row = target or TradingSystemRunArtifactTable()
    table_row.owner_user_id = int(artifact.owner_user_id)
    table_row.run_id = int(artifact.run_id)
    table_row.system_id = int(artifact.system_id)
    table_row.system_version_id = int(artifact.system_version_id) if artifact.system_version_id is not None else None
    table_row.artifact_type = _normalize_artifact_type(artifact.artifact_type)
    table_row.payload_json = _normalize_payload(artifact.payload_json)
    return table_row


def _normalize_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    return dict(payload or {})


def _normalize_artifact_type(value: str) -> str:
    return str(value or "unknown").strip().lower() or "unknown"
