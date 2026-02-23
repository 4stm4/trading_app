from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

try:
    from entities.user import User
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.user import User

from .mapping import to_entity, to_table
from .tables import UserTable


class UserPostgresRepository:
    def __init__(self, session: Session):
        self._session = session

    def get_by_id(self, user_id: int) -> User | None:
        row = self._session.get(UserTable, user_id)
        if row is None:
            return None
        return to_entity(row)

    def get_by_email(self, email: str) -> User | None:
        normalized_email = _normalize_email(email)
        stmt = select(UserTable).where(UserTable.email == normalized_email)
        row = self._session.scalars(stmt).first()
        if row is None:
            return None
        return to_entity(row)

    def add(self, user: User) -> User:
        row = to_table(user)
        self._session.add(row)
        self._session.flush()
        return to_entity(row)

    def upsert(self, user: User) -> User:
        normalized_email = _normalize_email(user.email)
        stmt = select(UserTable).where(UserTable.email == normalized_email)
        row = self._session.scalars(stmt).first()
        if row is None:
            row = to_table(user)
            row.email = normalized_email
            self._session.add(row)
        else:
            to_table(user, target=row)
            row.email = normalized_email

        self._session.flush()
        return to_entity(row)

    def delete(self, email: str) -> bool:
        normalized_email = _normalize_email(email)
        stmt = select(UserTable).where(UserTable.email == normalized_email)
        row = self._session.scalars(stmt).first()
        if row is None:
            return False

        self._session.delete(row)
        self._session.flush()
        return True


def _normalize_email(email: str) -> str:
    return str(email or "").strip().lower()
