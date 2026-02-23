from __future__ import annotations

try:
    from entities.user import User
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.user import User

from .tables import UserTable


def to_entity(table_row: UserTable) -> User:
    return User(
        id=table_row.id,
        email=table_row.email,
        password_hash=table_row.password_hash,
        is_active=table_row.is_active,
        created_at=table_row.created_at,
        updated_at=table_row.updated_at,
    )


def to_table(user: User, target: UserTable | None = None) -> UserTable:
    table_row = target or UserTable()
    table_row.email = str(user.email or "").strip().lower()
    table_row.password_hash = str(user.password_hash or "")
    table_row.is_active = bool(user.is_active)
    return table_row
