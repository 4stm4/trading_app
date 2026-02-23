"""System, portfolio and run-history use-cases."""

from __future__ import annotations

from typing import Any

from adapters.postgres import (
    PortfolioPostgresRepository,
    TradingModelPostgresRepository,
    TradingSystemPostgresRepository,
    TradingSystemRunArtifactPostgresRepository,
    TradingSystemRunPostgresRepository,
    TradingSystemVersionPostgresRepository,
    session_scope,
)
from entities.portfolio import Portfolio
from entities.trading_system import TradingSystem

from .auth import ensure_system_user, extract_user_email
from .errors import ApiNotFoundError, ApiServiceError, ApiValidationError
from .helpers import (
    DEFAULT_GUEST_PORTFOLIO_BALANCE,
    DEFAULT_GUEST_PORTFOLIO_CURRENCY,
    coerce_float,
    coerce_int,
    get_db_session_factory,
    json_safe,
    normalize_system_config,
    to_bool,
)
from .serializers import (
    serialize_portfolio,
    serialize_run_artifact,
    serialize_system_run,
    serialize_trading_system,
)


def build_systems_response(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    session_factory = get_db_session_factory()
    if session_factory is None:
        raise ApiServiceError("Database is not configured")

    body = payload or {}
    user_email = extract_user_email(body, required=False)
    if user_email is None:
        return {
            "owner_user_id": None,
            "current_system_id": None,
            "systems": [],
        }
    with session_scope(session_factory) as session:
        user = ensure_system_user(session, user_email=user_email)
        system_repo = TradingSystemPostgresRepository(session)
        version_repo = TradingSystemVersionPostgresRepository(session)

        systems = system_repo.list_by_owner(owner_user_id=int(user.id), only_active=False, limit=500)
        current = system_repo.get_current(owner_user_id=int(user.id))
        if current is None and systems:
            current = system_repo.set_current(owner_user_id=int(user.id), system_id=int(systems[0].id))
            systems = system_repo.list_by_owner(owner_user_id=int(user.id), only_active=False, limit=500)

        serialized = [serialize_trading_system(item, version_repo=version_repo) for item in systems]
        return json_safe(
            {
                "owner_user_id": int(user.id),
                "current_system_id": int(current.id) if current is not None and current.id is not None else None,
                "systems": serialized,
            }
        )


def build_system_create_response(payload: dict[str, Any]) -> dict[str, Any]:
    session_factory = get_db_session_factory()
    if session_factory is None:
        raise ApiServiceError("Database is not configured")

    name = str(payload.get("name") or "").strip()
    if not name:
        raise ApiValidationError("name is required")

    user_email = extract_user_email(payload, required=True)
    model_key = str(payload.get("model") or "balanced").strip().lower() or "balanced"
    make_current = to_bool(payload.get("make_current"), default=False)
    config = normalize_system_config(payload.get("config") or {})
    timeframe = str(payload.get("timeframe") or "1h").strip().lower() or "1h"
    exchange = str(payload.get("exchange") or "moex").strip().lower() or "moex"
    engine = str(payload.get("engine") or "stock").strip().lower() or "stock"
    market = str(payload.get("market") or "shares").strip().lower() or "shares"
    board = str(payload.get("board") or "").strip().upper()

    with session_scope(session_factory) as session:
        user = ensure_system_user(session, user_email=user_email)
        model_repo = TradingModelPostgresRepository(session)
        system_repo = TradingSystemPostgresRepository(session)
        version_repo = TradingSystemVersionPostgresRepository(session)

        model = model_repo.get_by_key(model_key)
        if model is None or model.id is None:
            raise ApiValidationError(f"Unknown model: {model_key}")

        saved = system_repo.upsert_by_owner_name(
            TradingSystem(
                owner_user_id=int(user.id),
                name=name,
                model_id=int(model.id),
                model_name=model.key,
                timeframe=timeframe,
                exchange=exchange,
                engine=engine,
                market=market,
                board=board,
                is_active=True,
                is_current=make_current,
            )
        )
        if saved.id is None:
            raise ApiServiceError("Failed to persist system")

        version_repo.create_next(
            system_id=int(saved.id),
            config_json=config,
            created_by_user_id=int(user.id),
            make_current=True,
        )

        current = system_repo.get_current(owner_user_id=int(user.id))
        if make_current or current is None:
            current = system_repo.set_current(owner_user_id=int(user.id), system_id=int(saved.id))

        refreshed = system_repo.get_by_id(int(saved.id)) or saved
        return json_safe(
            {
                "system": serialize_trading_system(refreshed, version_repo=version_repo),
                "current_system_id": int(current.id) if current is not None and current.id is not None else None,
            }
        )


def build_system_update_config_response(system_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    session_factory = get_db_session_factory()
    if session_factory is None:
        raise ApiServiceError("Database is not configured")

    user_email = extract_user_email(payload, required=True)
    with session_scope(session_factory) as session:
        user = ensure_system_user(session, user_email=user_email)
        model_repo = TradingModelPostgresRepository(session)
        system_repo = TradingSystemPostgresRepository(session)
        version_repo = TradingSystemVersionPostgresRepository(session)

        system = system_repo.get_by_id(int(system_id))
        if system is None or int(system.owner_user_id) != int(user.id):
            raise ApiNotFoundError("System not found")

        if "model" in payload:
            model_key = str(payload.get("model") or "").strip().lower()
            if not model_key:
                raise ApiValidationError("model cannot be empty")
            model = model_repo.get_by_key(model_key)
            if model is None or model.id is None:
                raise ApiValidationError(f"Unknown model: {model_key}")
            system.model_id = int(model.id)
            system.model_name = model.key

        if "timeframe" in payload:
            system.timeframe = str(payload.get("timeframe") or system.timeframe).strip().lower() or system.timeframe
        if "exchange" in payload:
            system.exchange = str(payload.get("exchange") or system.exchange).strip().lower() or system.exchange
        if "engine" in payload:
            system.engine = str(payload.get("engine") or system.engine).strip().lower() or system.engine
        if "market" in payload:
            system.market = str(payload.get("market") or system.market).strip().lower() or system.market
        if "board" in payload:
            system.board = str(payload.get("board") or "").strip().upper()

        updated = system_repo.upsert_by_owner_name(system)
        if updated.id is None:
            raise ApiServiceError("Failed to update system")

        config = normalize_system_config(payload.get("config") or {})
        version_repo.create_next(
            system_id=int(updated.id),
            config_json=config,
            created_by_user_id=int(user.id),
            make_current=True,
        )

        refreshed = system_repo.get_by_id(int(updated.id)) or updated
        return json_safe({"system": serialize_trading_system(refreshed, version_repo=version_repo)})


def build_system_set_current_response(payload: dict[str, Any]) -> dict[str, Any]:
    session_factory = get_db_session_factory()
    if session_factory is None:
        raise ApiServiceError("Database is not configured")

    user_email = extract_user_email(payload, required=True)
    system_id = payload.get("system_id")
    system_name = str(payload.get("name") or "").strip()
    if system_id is None and not system_name:
        raise ApiValidationError("system_id or name is required")

    with session_scope(session_factory) as session:
        user = ensure_system_user(session, user_email=user_email)
        system_repo = TradingSystemPostgresRepository(session)
        version_repo = TradingSystemVersionPostgresRepository(session)

        target = None
        if system_id is not None:
            target = system_repo.get_by_id(int(system_id))
        elif system_name:
            target = system_repo.get_by_owner_and_name(owner_user_id=int(user.id), name=system_name)

        if target is None or target.id is None or int(target.owner_user_id) != int(user.id):
            raise ApiNotFoundError("System not found")

        current = system_repo.set_current(owner_user_id=int(user.id), system_id=int(target.id))
        if current is None:
            raise ApiServiceError("Failed to set current system")

        refreshed = system_repo.get_by_id(int(current.id)) or current
        return json_safe(
            {
                "current_system_id": int(refreshed.id),
                "system": serialize_trading_system(refreshed, version_repo=version_repo),
            }
        )


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


def build_system_runs_response(system_id: int, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    session_factory = get_db_session_factory()
    if session_factory is None:
        raise ApiServiceError("Database is not configured")

    body = payload or {}
    user_email = extract_user_email(body, required=True)
    run_type = str(body.get("run_type") or "").strip().lower() or None
    status = str(body.get("status") or "").strip().lower() or None
    limit = coerce_int(body.get("limit", 100), default=100, min_value=1, max_value=500)

    with session_scope(session_factory) as session:
        user = ensure_system_user(session, user_email=user_email)
        system_repo = TradingSystemPostgresRepository(session)
        run_repo = TradingSystemRunPostgresRepository(session)

        system = system_repo.get_by_id(int(system_id))
        if system is None or system.id is None or int(system.owner_user_id) != int(user.id):
            raise ApiNotFoundError("System not found")

        runs = run_repo.list_by_system(
            owner_user_id=int(user.id),
            system_id=int(system.id),
            run_type=run_type,
            status=status,
            limit=limit,
        )
        return json_safe(
            {
                "owner_user_id": int(user.id),
                "system_id": int(system.id),
                "count": len(runs),
                "runs": [serialize_system_run(item) for item in runs],
            }
        )


def build_system_run_artifacts_response(run_id: int, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    session_factory = get_db_session_factory()
    if session_factory is None:
        raise ApiServiceError("Database is not configured")

    body = payload or {}
    user_email = extract_user_email(body, required=True)
    artifact_type = str(body.get("artifact_type") or "").strip().lower() or None
    limit = coerce_int(body.get("limit", 50), default=50, min_value=1, max_value=500)

    with session_scope(session_factory) as session:
        user = ensure_system_user(session, user_email=user_email)
        run_repo = TradingSystemRunPostgresRepository(session)
        artifact_repo = TradingSystemRunArtifactPostgresRepository(session)

        run = run_repo.get_by_id(int(run_id))
        if run is None or run.id is None or int(run.owner_user_id) != int(user.id):
            raise ApiNotFoundError("Run not found")

        artifacts = artifact_repo.list_by_run(
            owner_user_id=int(user.id),
            run_id=int(run.id),
            artifact_type=artifact_type,
            limit=limit,
        )
        return json_safe(
            {
                "owner_user_id": int(user.id),
                "run": serialize_system_run(run),
                "count": len(artifacts),
                "artifacts": [serialize_run_artifact(item) for item in artifacts],
            }
        )
