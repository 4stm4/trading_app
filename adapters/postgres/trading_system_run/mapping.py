from __future__ import annotations

from typing import Any

try:
    from entities.trading_system_run import TradingSystemRun
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.trading_system_run import TradingSystemRun

from .tables import TradingSystemRunTable


def to_entity(table_row: TradingSystemRunTable) -> TradingSystemRun:
    return TradingSystemRun(
        id=table_row.id,
        system_id=table_row.system_id,
        system_version_id=table_row.system_version_id,
        run_type=table_row.run_type,
        status=table_row.status,
        request_json=_normalize_json_or_none(table_row.request_json),
        result_summary_json=_normalize_json_or_none(table_row.result_summary_json),
        error_text=table_row.error_text,
        started_at=table_row.started_at,
        finished_at=table_row.finished_at,
        created_at=table_row.created_at,
    )


def to_table(system_run: TradingSystemRun, target: TradingSystemRunTable | None = None) -> TradingSystemRunTable:
    table_row = target or TradingSystemRunTable()
    table_row.system_id = int(system_run.system_id)
    table_row.system_version_id = int(system_run.system_version_id) if system_run.system_version_id is not None else None
    table_row.run_type = _normalize_run_type(system_run.run_type)
    table_row.status = _normalize_status(system_run.status)
    table_row.request_json = _normalize_json_or_none(system_run.request_json)
    table_row.result_summary_json = _normalize_json_or_none(system_run.result_summary_json)
    error_text = str(system_run.error_text or "").strip()
    table_row.error_text = error_text or None
    table_row.started_at = system_run.started_at
    table_row.finished_at = system_run.finished_at
    return table_row


def _normalize_json_or_none(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return dict(payload)


def _normalize_run_type(run_type: str) -> str:
    return str(run_type or "unknown").strip().lower() or "unknown"


def _normalize_status(status: str) -> str:
    return str(status or "pending").strip().lower() or "pending"
