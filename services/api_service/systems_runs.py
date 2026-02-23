"""System-run history and artifacts use-cases."""

from __future__ import annotations

from typing import Any

from adapters.postgres import (
    TradingSystemPostgresRepository,
    TradingSystemRunArtifactPostgresRepository,
    TradingSystemRunPostgresRepository,
    session_scope,
)

from .auth import ensure_system_user, extract_user_email
from .errors import ApiNotFoundError, ApiServiceError
from .helpers import coerce_int, get_db_session_factory, json_safe
from .serializers import serialize_run_artifact, serialize_system_run


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
