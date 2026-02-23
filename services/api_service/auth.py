"""Authentication and user identity helpers."""

from __future__ import annotations

from typing import Any

from adapters.postgres import PortfolioPostgresRepository, UserPostgresRepository, session_scope
from entities.user import User
from services.auth_security import (
    PASSWORD_SCHEME,
    TOKEN_TTL_SECONDS,
    hash_password,
    issue_access_token,
    parse_bearer_token,
    resolve_user_email_from_token,
    verify_password,
)

from .errors import ApiConflictError, ApiServiceError, ApiUnauthorizedError, ApiValidationError
from .helpers import get_db_session_factory
from .serializers import serialize_user


def build_auth_register_response(payload: dict[str, Any]) -> dict[str, Any]:
    session_factory = get_db_session_factory()
    if session_factory is None:
        raise ApiServiceError("Database is not configured")

    email = normalize_email(payload.get("email"))
    if not email:
        raise ApiValidationError("email is required")
    validate_email(email)
    password = str(payload.get("password") or "")
    validate_password(password)

    with session_scope(session_factory) as session:
        user_repo = UserPostgresRepository(session)
        portfolio_repo = PortfolioPostgresRepository(session)
        existing = user_repo.get_by_email(email)
        if existing is not None:
            if is_supported_password_hash(existing.password_hash):
                raise ApiConflictError("User already exists")
            upgraded = user_repo.upsert(
                User(
                    id=existing.id,
                    email=email,
                    password_hash=hash_password(password),
                    is_active=True,
                    created_at=existing.created_at,
                )
            )
            if upgraded.id is not None:
                portfolio_repo.ensure_for_owner(owner_user_id=int(upgraded.id))
            return build_auth_response(upgraded)

        user = user_repo.add(
            User(
                email=email,
                password_hash=hash_password(password),
                is_active=True,
            )
        )
        if user.id is not None:
            portfolio_repo.ensure_for_owner(owner_user_id=int(user.id))

        return build_auth_response(user)


def build_auth_login_response(payload: dict[str, Any]) -> dict[str, Any]:
    session_factory = get_db_session_factory()
    if session_factory is None:
        raise ApiServiceError("Database is not configured")

    email = normalize_email(payload.get("email"))
    if not email:
        raise ApiValidationError("email is required")
    validate_email(email)
    password = str(payload.get("password") or "")
    if not password:
        raise ApiValidationError("password is required")

    with session_scope(session_factory) as session:
        user_repo = UserPostgresRepository(session)
        user = user_repo.get_by_email(email)
        if user is None or not verify_password(password, user.password_hash):
            raise ApiUnauthorizedError("Invalid credentials")
        if not bool(user.is_active):
            raise ApiUnauthorizedError("User is inactive")
        return build_auth_response(user)


def build_auth_me_response(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    session_factory = get_db_session_factory()
    if session_factory is None:
        raise ApiServiceError("Database is not configured")

    user_email = extract_user_email(payload or {}, required=True)
    with session_scope(session_factory) as session:
        user = ensure_system_user(session, user_email=user_email)
        return {"user": serialize_user(user)}


def resolve_authenticated_user_email(authorization: str | None) -> str | None:
    token = parse_bearer_token(authorization)
    if token is None:
        return None
    user_email = resolve_user_email_from_token(token)
    if user_email is None:
        raise ApiUnauthorizedError("Invalid or expired token")
    return user_email


def extract_user_email(payload: dict[str, Any] | None, *, required: bool) -> str | None:
    body = payload or {}
    auth_user_email = normalize_email(body.get("_auth_user_email"))
    request_user_email = normalize_email(body.get("user_email"))
    if auth_user_email and request_user_email and auth_user_email != request_user_email:
        raise ApiUnauthorizedError("Authenticated user mismatch")
    normalized_email = auth_user_email or request_user_email
    if normalized_email:
        return normalized_email
    if required:
        raise ApiUnauthorizedError("Authentication required")
    return None


def ensure_system_user(session: Any, *, user_email: str) -> User:
    user_repo = UserPostgresRepository(session)
    portfolio_repo = PortfolioPostgresRepository(session)
    normalized_email = normalize_email(user_email)
    if not normalized_email:
        raise ApiUnauthorizedError("Authentication required")
    existing = user_repo.get_by_email(normalized_email)
    if existing is None:
        raise ApiUnauthorizedError("User not found")
    if existing.id is not None:
        portfolio_repo.ensure_for_owner(owner_user_id=int(existing.id))
    return existing


def normalize_email(value: Any) -> str:
    return str(value or "").strip().lower()


def validate_password(password: str) -> None:
    raw_password = str(password or "")
    if len(raw_password) < 8:
        raise ApiValidationError("password must contain at least 8 characters")


def validate_email(email: str) -> None:
    normalized = normalize_email(email)
    if not normalized or "@" not in normalized or normalized.startswith("@") or normalized.endswith("@"):
        raise ApiValidationError("invalid email format")


def is_supported_password_hash(password_hash: str) -> bool:
    normalized = str(password_hash or "")
    return normalized.startswith(f"{PASSWORD_SCHEME}$")


def build_auth_response(user: User) -> dict[str, Any]:
    if user.id is None:
        raise ApiServiceError("Failed to load user")
    token = issue_access_token(user_email=str(user.email))
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": int(TOKEN_TTL_SECONDS),
        "user": serialize_user(user),
    }
