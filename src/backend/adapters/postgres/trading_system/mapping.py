from __future__ import annotations

try:
    from entities.trading_system import TradingSystem
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.trading_system import TradingSystem

from .tables import TradingSystemTable


def to_entity(table_row: TradingSystemTable) -> TradingSystem:
    return TradingSystem(
        id=table_row.id,
        owner_user_id=table_row.owner_user_id,
        model_id=table_row.model_id,
        name=table_row.name,
        description=table_row.description,
        exchange=table_row.exchange,
        engine=table_row.engine,
        market=table_row.market,
        board=table_row.board,
        timeframe=table_row.timeframe,
        model_name=table_row.model_name,
        is_active=table_row.is_active,
        is_current=table_row.is_current,
        created_at=table_row.created_at,
        updated_at=table_row.updated_at,
    )


def to_table(system: TradingSystem, target: TradingSystemTable | None = None) -> TradingSystemTable:
    table_row = target or TradingSystemTable()
    table_row.owner_user_id = int(system.owner_user_id)
    table_row.model_id = int(system.model_id) if system.model_id is not None else None
    table_row.name = _normalize_name(system.name)
    description = str(system.description or "").strip()
    table_row.description = description or None
    table_row.exchange = _normalize_exchange(system.exchange)
    table_row.engine = _normalize_engine(system.engine)
    table_row.market = _normalize_market(system.market)
    table_row.board = _normalize_board(system.board)
    table_row.timeframe = _normalize_timeframe(system.timeframe)
    table_row.model_name = _normalize_model_name(system.model_name)
    table_row.is_active = bool(system.is_active)
    table_row.is_current = bool(system.is_current)
    return table_row


def _normalize_name(name: str) -> str:
    normalized = str(name or "").strip()
    return normalized or "default"


def _normalize_exchange(exchange: str) -> str:
    return str(exchange or "moex").strip().lower() or "moex"


def _normalize_engine(engine: str) -> str:
    return str(engine or "stock").strip().lower() or "stock"


def _normalize_market(market: str) -> str:
    return str(market or "shares").strip().lower() or "shares"


def _normalize_board(board: str | None) -> str:
    return str(board or "").strip().upper()


def _normalize_timeframe(timeframe: str) -> str:
    return str(timeframe or "1h").strip().lower() or "1h"


def _normalize_model_name(model_name: str) -> str:
    return str(model_name or "balanced").strip().lower() or "balanced"
