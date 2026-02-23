from __future__ import annotations

try:
    from entities.trading_system_signal import TradingSystemSignal
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.trading_system_signal import TradingSystemSignal

from .tables import TradingSystemSignalTable


def to_entity(table_row: TradingSystemSignalTable) -> TradingSystemSignal:
    return TradingSystemSignal(
        id=table_row.id,
        system_id=table_row.system_id,
        system_version_id=table_row.system_version_id,
        exchange=table_row.exchange,
        symbol=table_row.symbol,
        timeframe=table_row.timeframe,
        model_name=table_row.model_name,
        signal=table_row.signal,
        confidence=table_row.confidence,
        entry=table_row.entry,
        stop=table_row.stop,
        target=table_row.target,
        rr=table_row.rr,
        market_regime=table_row.market_regime,
        phase=table_row.phase,
        generated_at=table_row.generated_at,
        created_at=table_row.created_at,
    )


def to_table(
    system_signal: TradingSystemSignal,
    target: TradingSystemSignalTable | None = None,
) -> TradingSystemSignalTable:
    table_row = target or TradingSystemSignalTable()
    table_row.system_id = int(system_signal.system_id)
    table_row.system_version_id = (
        int(system_signal.system_version_id) if system_signal.system_version_id is not None else None
    )
    table_row.exchange = _normalize_exchange(system_signal.exchange)
    table_row.symbol = _normalize_symbol(system_signal.symbol)
    table_row.timeframe = _normalize_timeframe(system_signal.timeframe)
    table_row.model_name = _normalize_model_name(system_signal.model_name)
    table_row.signal = _normalize_signal(system_signal.signal)
    table_row.confidence = _normalize_confidence(system_signal.confidence)
    table_row.entry = float(system_signal.entry) if system_signal.entry is not None else None
    table_row.stop = float(system_signal.stop) if system_signal.stop is not None else None
    table_row.target = float(system_signal.target) if system_signal.target is not None else None
    table_row.rr = float(system_signal.rr) if system_signal.rr is not None else None
    market_regime = str(system_signal.market_regime or "").strip().lower()
    table_row.market_regime = market_regime or None
    phase = str(system_signal.phase or "").strip().lower()
    table_row.phase = phase or None
    if system_signal.generated_at is not None:
        table_row.generated_at = system_signal.generated_at
    return table_row


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


def _normalize_confidence(confidence: str) -> str:
    return str(confidence or "none").strip().lower() or "none"
