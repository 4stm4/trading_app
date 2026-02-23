"""Portfolio use-cases linked to authenticated users."""

from __future__ import annotations

from typing import Any

from adapters.postgres import PortfolioPostgresRepository, session_scope
from entities.portfolio import Portfolio

from .auth import ensure_system_user, extract_user_email
from .errors import ApiServiceError, ApiValidationError
from .helpers import (
    DEFAULT_GUEST_PORTFOLIO_BALANCE,
    DEFAULT_GUEST_PORTFOLIO_CURRENCY,
    coerce_float,
    get_db_session_factory,
    json_safe,
    to_bool,
)
from .serializers import serialize_portfolio


def build_portfolio_response(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    session_factory = get_db_session_factory()
    if session_factory is None:
        raise ApiServiceError("Database is not configured")

    body = payload or {}
    user_email = extract_user_email(body, required=False)
    if user_email is None:
        return {
            "owner_user_id": None,
            "portfolio": {
                "id": None,
                "owner_user_id": None,
                "balance": float(DEFAULT_GUEST_PORTFOLIO_BALANCE),
                "currency": DEFAULT_GUEST_PORTFOLIO_CURRENCY,
                "is_active": False,
                "created_at": None,
                "updated_at": None,
            },
        }
    with session_scope(session_factory) as session:
        user = ensure_system_user(session, user_email=user_email)
        portfolio_repo = PortfolioPostgresRepository(session)
        portfolio = portfolio_repo.ensure_for_owner(owner_user_id=int(user.id))
        return json_safe({"owner_user_id": int(user.id), "portfolio": serialize_portfolio(portfolio)})


def build_portfolio_update_response(payload: dict[str, Any]) -> dict[str, Any]:
    session_factory = get_db_session_factory()
    if session_factory is None:
        raise ApiServiceError("Database is not configured")

    user_email = extract_user_email(payload, required=True)
    raw_balance = payload.get("balance", payload.get("deposit"))
    if raw_balance is None:
        raise ApiValidationError("balance is required")
    balance = coerce_float(raw_balance, default=100000.0, min_value=0.0, max_value=10_000_000_000.0)
    currency = str(payload.get("currency") or "RUB").strip().upper() or "RUB"
    is_active = to_bool(payload.get("is_active"), default=True)

    with session_scope(session_factory) as session:
        user = ensure_system_user(session, user_email=user_email)
        portfolio_repo = PortfolioPostgresRepository(session)
        existing = portfolio_repo.ensure_for_owner(owner_user_id=int(user.id))
        updated = portfolio_repo.upsert_by_owner(
            Portfolio(
                owner_user_id=int(user.id),
                balance=balance,
                currency=currency,
                is_active=bool(is_active),
                id=existing.id,
                created_at=existing.created_at,
            )
        )
        return json_safe({"owner_user_id": int(user.id), "portfolio": serialize_portfolio(updated)})
