from __future__ import annotations

try:
    from entities.trading_model import TradingModel
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.trading_model import TradingModel

from .tables import TradingModelTable


def to_entity(table_row: TradingModelTable) -> TradingModel:
    return TradingModel(
        id=table_row.id,
        key=table_row.key,
        name=table_row.name,
        description=table_row.description,
        is_active=table_row.is_active,
        created_at=table_row.created_at,
        updated_at=table_row.updated_at,
    )


def to_table(model: TradingModel, target: TradingModelTable | None = None) -> TradingModelTable:
    table_row = target or TradingModelTable()
    table_row.key = _normalize_key(model.key)
    table_row.name = _normalize_name(model.name)
    description = str(model.description or "").strip()
    table_row.description = description or None
    table_row.is_active = bool(model.is_active)
    return table_row


def _normalize_key(value: str) -> str:
    return str(value or "").strip().lower()


def _normalize_name(value: str) -> str:
    normalized = str(value or "").strip()
    return normalized or "Unnamed model"
