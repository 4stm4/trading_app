"""Database URL resolution helpers."""

from __future__ import annotations

import os
from urllib.parse import quote


def _normalize_database_url(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    if normalized.startswith("postgres://"):
        return normalized.replace("postgres://", "postgresql://", 1)
    return normalized


def resolve_database_url_from_env() -> str | None:
    for env_key in ("DATABASE_URL", "ALEMBIC_DATABASE_URL", "AUTH_DB_URL"):
        normalized = _normalize_database_url(os.getenv(env_key))
        if normalized:
            return normalized

    host = str(os.getenv("DB_HOST") or os.getenv("APP_DB_HOST") or "").strip()
    port = str(os.getenv("DB_PORT") or os.getenv("APP_DB_PORT") or "5432").strip()
    name = str(os.getenv("DB_NAME") or os.getenv("APP_DB_NAME") or "").strip()
    user = str(os.getenv("DB_USER") or os.getenv("APP_DB_USER") or "").strip()
    password = str(os.getenv("DB_PASSWORD") or os.getenv("APP_DB_PASS") or "").strip()
    if not host or not name or not user or not password:
        return None

    user_q = quote(user, safe="")
    password_q = quote(password, safe="")
    name_q = quote(name, safe="")
    return f"postgresql+psycopg://{user_q}:{password_q}@{host}:{port}/{name_q}"
