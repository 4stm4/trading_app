from __future__ import annotations

try:
    from entities.portfolio import Portfolio
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.portfolio import Portfolio

from .tables import PortfolioTable


def to_entity(table_row: PortfolioTable) -> Portfolio:
    return Portfolio(
        id=table_row.id,
        owner_user_id=table_row.owner_user_id,
        balance=table_row.balance,
        currency=table_row.currency,
        is_active=table_row.is_active,
        created_at=table_row.created_at,
        updated_at=table_row.updated_at,
    )


def to_table(portfolio: Portfolio, target: PortfolioTable | None = None) -> PortfolioTable:
    table_row = target or PortfolioTable()
    table_row.owner_user_id = int(portfolio.owner_user_id)
    table_row.balance = max(float(portfolio.balance), 0.0)
    table_row.currency = _normalize_currency(portfolio.currency)
    table_row.is_active = bool(portfolio.is_active)
    return table_row


def _normalize_currency(value: str) -> str:
    normalized = str(value or "RUB").strip().upper()
    return normalized or "RUB"
