"""System run tracking helpers (persist run metadata/artifacts)."""

from __future__ import annotations

from typing import Any

from adapters.postgres import (
    PortfolioPostgresRepository,
    TradingSystemPostgresRepository,
    TradingSystemRunArtifactPostgresRepository,
    TradingSystemRunPostgresRepository,
    TradingSystemVersionPostgresRepository,
    session_scope,
)
from entities.trading_system_run_artifact import TradingSystemRunArtifact

from .auth import ensure_system_user, extract_user_email
from .errors import ApiNotFoundError, ApiValidationError
from .helpers import get_db_session_factory, to_float_or_none, to_int_or_none, to_json_object


def resolve_system_run_context(payload: dict[str, Any], *, run_type: str) -> dict[str, Any] | None:
    session_factory = get_db_session_factory()
    if session_factory is None:
        return None

    body = payload or {}
    if "system_id" not in body and "user_email" not in body and "_auth_user_email" not in body:
        return None

    user_email = extract_user_email(body, required=False)
    if user_email is None:
        return None
    raw_system_id = body.get("system_id")
    with session_scope(session_factory) as session:
        user = ensure_system_user(session, user_email=user_email)
        portfolio_repo = PortfolioPostgresRepository(session)
        system_repo = TradingSystemPostgresRepository(session)
        version_repo = TradingSystemVersionPostgresRepository(session)

        system = None
        if raw_system_id is not None:
            try:
                system_id = int(raw_system_id)
            except (TypeError, ValueError) as error:
                raise ApiValidationError("system_id must be integer") from error
            system = system_repo.get_by_id(system_id)
            if system is None or system.id is None or int(system.owner_user_id) != int(user.id):
                raise ApiNotFoundError("System not found")
        else:
            system = system_repo.get_current(owner_user_id=int(user.id))

        portfolio = portfolio_repo.ensure_for_owner(owner_user_id=int(user.id))
        system_id: int | None = None
        system_version_id: int | None = None
        if system is not None and system.id is not None:
            system_id = int(system.id)
            version = version_repo.get_current(system_id=system_id)
            if version is not None and version.id is not None:
                system_version_id = int(version.id)

        return {
            "run_type": str(run_type).strip().lower() or "unknown",
            "owner_user_id": int(user.id),
            "system_id": system_id,
            "system_version_id": system_version_id,
            "portfolio_id": int(portfolio.id) if portfolio.id is not None else None,
            "portfolio_balance_snapshot": float(portfolio.balance),
        }


def start_system_run(
    run_context: dict[str, Any] | None,
    *,
    run_type: str,
    request_payload: dict[str, Any],
) -> int | None:
    if not run_context:
        return None
    system_id = run_context.get("system_id")
    owner_user_id = run_context.get("owner_user_id")
    if system_id is None or owner_user_id is None:
        return None

    session_factory = get_db_session_factory()
    if session_factory is None:
        return None
    with session_scope(session_factory) as session:
        run_repo = TradingSystemRunPostgresRepository(session)
        run = run_repo.start(
            owner_user_id=int(owner_user_id),
            system_id=int(system_id),
            portfolio_id=to_int_or_none(run_context.get("portfolio_id")),
            portfolio_balance_snapshot=to_float_or_none(run_context.get("portfolio_balance_snapshot")),
            run_type=str(run_type),
            system_version_id=to_int_or_none(run_context.get("system_version_id")),
            request_json=to_json_object(request_payload),
        )
        return int(run.id) if run.id is not None else None


def finish_system_run_success(
    *,
    run_context: dict[str, Any] | None,
    run_id: int | None,
    result_summary: dict[str, Any] | None,
    artifacts: dict[str, Any] | None = None,
) -> None:
    if not run_context or run_id is None:
        return

    session_factory = get_db_session_factory()
    if session_factory is None:
        return

    with session_scope(session_factory) as session:
        run_repo = TradingSystemRunPostgresRepository(session)
        artifact_repo = TradingSystemRunArtifactPostgresRepository(session)
        run_repo.update_status(
            int(run_id),
            status="done",
            result_summary_json=to_json_object(result_summary),
        )

        owner_user_id = to_int_or_none(run_context.get("owner_user_id"))
        system_id = to_int_or_none(run_context.get("system_id"))
        if owner_user_id is None or system_id is None:
            return

        for artifact_type, artifact_payload in (artifacts or {}).items():
            artifact_repo.upsert_by_run_type(
                TradingSystemRunArtifact(
                    owner_user_id=int(owner_user_id),
                    run_id=int(run_id),
                    system_id=int(system_id),
                    system_version_id=to_int_or_none(run_context.get("system_version_id")),
                    artifact_type=str(artifact_type),
                    payload_json=to_json_object(artifact_payload),
                )
            )


def finish_system_run_failure(
    *,
    run_context: dict[str, Any] | None,
    run_id: int | None,
    error_text: str,
) -> None:
    if not run_context or run_id is None:
        return
    session_factory = get_db_session_factory()
    if session_factory is None:
        return
    with session_scope(session_factory) as session:
        run_repo = TradingSystemRunPostgresRepository(session)
        run_repo.update_status(
            int(run_id),
            status="failed",
            error_text=str(error_text or "").strip() or "unknown error",
        )
