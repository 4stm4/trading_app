from __future__ import annotations

from typing import Any

try:
    from entities.trading_system_version import TradingSystemVersion
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.trading_system_version import TradingSystemVersion

from .tables import TradingSystemVersionTable


def to_entity(table_row: TradingSystemVersionTable) -> TradingSystemVersion:
    return TradingSystemVersion(
        id=table_row.id,
        system_id=table_row.system_id,
        version=table_row.version,
        config_json=_normalize_json(table_row.config_json),
        is_current=table_row.is_current,
        created_by_user_id=table_row.created_by_user_id,
        created_at=table_row.created_at,
    )


def to_table(
    system_version: TradingSystemVersion,
    target: TradingSystemVersionTable | None = None,
) -> TradingSystemVersionTable:
    table_row = target or TradingSystemVersionTable()
    table_row.system_id = int(system_version.system_id)
    table_row.version = int(system_version.version)
    table_row.config_json = _normalize_json(system_version.config_json)
    table_row.is_current = bool(system_version.is_current)
    table_row.created_by_user_id = (
        int(system_version.created_by_user_id) if system_version.created_by_user_id is not None else None
    )
    return table_row


def _normalize_json(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    return dict(payload)
