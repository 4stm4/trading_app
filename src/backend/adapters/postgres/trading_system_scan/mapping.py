from __future__ import annotations

from typing import Any

try:
    from entities.trading_system_scan import TradingSystemScan
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.trading_system_scan import TradingSystemScan

from .tables import TradingSystemScanTable


def to_entity(table_row: TradingSystemScanTable) -> TradingSystemScan:
    return TradingSystemScan(
        id=table_row.id,
        owner_user_id=table_row.owner_user_id,
        system_id=table_row.system_id,
        system_version_id=table_row.system_version_id,
        scan_key=table_row.scan_key,
        exchange=table_row.exchange,
        engine=table_row.engine,
        market=table_row.market,
        board=table_row.board,
        symbol=table_row.symbol,
        timeframe=table_row.timeframe,
        model_name=table_row.model_name,
        signal=table_row.signal,
        confidence=table_row.confidence,
        tradable=bool(table_row.tradable),
        entry=table_row.entry,
        stop=table_row.stop,
        target=table_row.target,
        rr=table_row.rr,
        market_regime=table_row.market_regime,
        phase=table_row.phase,
        issues_json=_normalize_json_or_none(table_row.issues_json),
        generated_at=table_row.generated_at,
        created_at=table_row.created_at,
    )


def to_table(scan: TradingSystemScan, target: TradingSystemScanTable | None = None) -> TradingSystemScanTable:
    table_row = target or TradingSystemScanTable()
    table_row.owner_user_id = int(scan.owner_user_id)
    table_row.system_id = int(scan.system_id)
    table_row.system_version_id = int(scan.system_version_id) if scan.system_version_id is not None else None
    table_row.scan_key = _normalize_scan_key(scan.scan_key)
    table_row.exchange = _normalize_exchange(scan.exchange)
    table_row.engine = _normalize_engine(scan.engine)
    table_row.market = _normalize_market(scan.market)
    table_row.board = _normalize_board(scan.board)
    table_row.symbol = _normalize_symbol(scan.symbol)
    table_row.timeframe = _normalize_timeframe(scan.timeframe)
    table_row.model_name = _normalize_model_name(scan.model_name)
    table_row.signal = _normalize_signal(scan.signal)
    table_row.confidence = _normalize_confidence(scan.confidence)
    table_row.tradable = bool(scan.tradable)
    table_row.entry = float(scan.entry) if scan.entry is not None else None
    table_row.stop = float(scan.stop) if scan.stop is not None else None
    table_row.target = float(scan.target) if scan.target is not None else None
    table_row.rr = float(scan.rr) if scan.rr is not None else None
    market_regime = str(scan.market_regime or "").strip().lower()
    table_row.market_regime = market_regime or None
    phase = str(scan.phase or "").strip().lower()
    table_row.phase = phase or None
    table_row.issues_json = _normalize_json_or_none(scan.issues_json)
    if scan.generated_at is not None:
        table_row.generated_at = scan.generated_at
    return table_row


def _normalize_json_or_none(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return dict(payload)


def _normalize_scan_key(scan_key: str) -> str:
    value = str(scan_key or "").strip()
    return value or "default"


def _normalize_exchange(exchange: str) -> str:
    return str(exchange or "moex").strip().lower() or "moex"


def _normalize_engine(engine: str) -> str:
    return str(engine or "stock").strip().lower() or "stock"


def _normalize_market(market: str) -> str:
    return str(market or "shares").strip().lower() or "shares"


def _normalize_board(board: str | None) -> str:
    return str(board or "").strip().upper()


def _normalize_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def _normalize_timeframe(timeframe: str) -> str:
    return str(timeframe or "1h").strip().lower() or "1h"


def _normalize_model_name(model_name: str) -> str:
    return str(model_name or "balanced").strip().lower() or "balanced"


def _normalize_signal(signal: str) -> str:
    return str(signal or "none").strip().lower() or "none"


def _normalize_confidence(confidence: str) -> str:
    return str(confidence or "none").strip().lower() or "none"
