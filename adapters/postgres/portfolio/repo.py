from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

try:
    from entities.portfolio import Portfolio
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.portfolio import Portfolio

from .mapping import to_entity, to_table
from .tables import PortfolioTable


class PortfolioPostgresRepository:
    def __init__(self, session: Session):
        self._session = session

    def get_by_id(self, portfolio_id: int) -> Portfolio | None:
        row = self._session.get(PortfolioTable, int(portfolio_id))
        if row is None:
            return None
        return to_entity(row)

    def get_by_owner(self, *, owner_user_id: int) -> Portfolio | None:
        stmt = select(PortfolioTable).where(PortfolioTable.owner_user_id == int(owner_user_id))
        row = self._session.scalars(stmt).first()
        if row is None:
            return None
        return to_entity(row)

    def add(self, portfolio: Portfolio) -> Portfolio:
        row = to_table(portfolio)
        self._session.add(row)
        self._session.flush()
        return to_entity(row)

    def upsert_by_owner(self, portfolio: Portfolio) -> Portfolio:
        stmt = select(PortfolioTable).where(PortfolioTable.owner_user_id == int(portfolio.owner_user_id))
        row = self._session.scalars(stmt).first()
        if row is None:
            row = to_table(portfolio)
            self._session.add(row)
        else:
            to_table(portfolio, target=row)
        self._session.flush()
        return to_entity(row)

    def ensure_for_owner(
        self,
        *,
        owner_user_id: int,
        default_balance: float = 100000.0,
        currency: str = "RUB",
    ) -> Portfolio:
        existing = self.get_by_owner(owner_user_id=int(owner_user_id))
        if existing is not None:
            return existing
        return self.add(
            Portfolio(
                owner_user_id=int(owner_user_id),
                balance=max(float(default_balance), 0.0),
                currency=currency,
                is_active=True,
            )
        )

    def set_balance(self, *, owner_user_id: int, balance: float) -> Portfolio | None:
        row = self._session.scalars(
            select(PortfolioTable).where(PortfolioTable.owner_user_id == int(owner_user_id))
        ).first()
        if row is None:
            return None
        row.balance = max(float(balance), 0.0)
        self._session.flush()
        return to_entity(row)
