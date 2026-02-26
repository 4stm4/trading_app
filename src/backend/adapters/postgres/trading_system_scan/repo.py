from __future__ import annotations

from typing import Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

try:
    from entities.trading_system_scan import TradingSystemScan
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.trading_system_scan import TradingSystemScan

from .mapping import to_entity, to_table
from .tables import TradingSystemScanTable


class TradingSystemScanPostgresRepository:
    def __init__(self, session: Session):
        self._session = session

    def get_by_id(self, scan_id: int) -> TradingSystemScan | None:
        row = self._session.get(TradingSystemScanTable, int(scan_id))
        if row is None:
            return None
        return to_entity(row)

    def list_by_owner(
        self,
        *,
        owner_user_id: int,
        system_id: int | None = None,
        tradable_only: bool = False,
        signal: str | None = None,
        limit: int = 200,
    ) -> list[TradingSystemScan]:
        stmt = select(TradingSystemScanTable).where(TradingSystemScanTable.owner_user_id == int(owner_user_id))
        if system_id is not None:
            stmt = stmt.where(TradingSystemScanTable.system_id == int(system_id))
        if tradable_only:
            stmt = stmt.where(TradingSystemScanTable.tradable.is_(True))
        if signal:
            stmt = stmt.where(TradingSystemScanTable.signal == _normalize_signal(signal))
        stmt = stmt.order_by(TradingSystemScanTable.created_at.desc(), TradingSystemScanTable.id.desc()).limit(
            max(1, int(limit))
        )
        rows: Sequence[TradingSystemScanTable] = self._session.scalars(stmt).all()
        return [to_entity(row) for row in rows]

    def list_by_scan_key(
        self,
        *,
        owner_user_id: int,
        scan_key: str,
        tradable_only: bool = False,
        limit: int = 1000,
    ) -> list[TradingSystemScan]:
        stmt = select(TradingSystemScanTable).where(
            TradingSystemScanTable.owner_user_id == int(owner_user_id),
            TradingSystemScanTable.scan_key == _normalize_scan_key(scan_key),
        )
        if tradable_only:
            stmt = stmt.where(TradingSystemScanTable.tradable.is_(True))
        stmt = stmt.order_by(TradingSystemScanTable.created_at.desc(), TradingSystemScanTable.id.desc()).limit(
            max(1, int(limit))
        )
        rows: Sequence[TradingSystemScanTable] = self._session.scalars(stmt).all()
        return [to_entity(row) for row in rows]

    def add(self, scan: TradingSystemScan) -> TradingSystemScan:
        row = to_table(scan)
        self._session.add(row)
        self._session.flush()
        return to_entity(row)

    def add_many(self, scans: list[TradingSystemScan]) -> list[TradingSystemScan]:
        if not scans:
            return []
        rows = [to_table(scan) for scan in scans]
        self._session.add_all(rows)
        self._session.flush()
        return [to_entity(row) for row in rows]

    def upsert_by_scan_key(self, scan: TradingSystemScan) -> TradingSystemScan:
        stmt = select(TradingSystemScanTable).where(
            TradingSystemScanTable.owner_user_id == int(scan.owner_user_id),
            TradingSystemScanTable.scan_key == _normalize_scan_key(scan.scan_key),
            TradingSystemScanTable.exchange == _normalize_exchange(scan.exchange),
            TradingSystemScanTable.symbol == _normalize_symbol(scan.symbol),
            TradingSystemScanTable.timeframe == _normalize_timeframe(scan.timeframe),
            TradingSystemScanTable.model_name == _normalize_model_name(scan.model_name),
        )
        row = self._session.scalars(stmt).first()
        if row is None:
            row = to_table(scan)
            self._session.add(row)
        else:
            to_table(scan, target=row)
        self._session.flush()
        return to_entity(row)

    def delete(self, scan_id: int) -> bool:
        row = self._session.get(TradingSystemScanTable, int(scan_id))
        if row is None:
            return False
        self._session.delete(row)
        self._session.flush()
        return True

    def delete_by_scan_key(self, *, owner_user_id: int, scan_key: str) -> int:
        stmt = delete(TradingSystemScanTable).where(
            TradingSystemScanTable.owner_user_id == int(owner_user_id),
            TradingSystemScanTable.scan_key == _normalize_scan_key(scan_key),
        )
        result = self._session.execute(stmt)
        self._session.flush()
        return int(result.rowcount or 0)


def _normalize_scan_key(scan_key: str) -> str:
    value = str(scan_key or "").strip()
    return value or "default"


def _normalize_exchange(exchange: str) -> str:
    return str(exchange or "moex").strip().lower() or "moex"


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def _normalize_timeframe(timeframe: str) -> str:
    return str(timeframe or "1h").strip().lower() or "1h"


def _normalize_model_name(model_name: str) -> str:
    return str(model_name or "balanced").strip().lower() or "balanced"


def _normalize_signal(signal: str) -> str:
    return str(signal or "none").strip().lower() or "none"
