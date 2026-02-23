from __future__ import annotations

try:
    from entities.instrument import Instrument
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.instrument import Instrument

from .tables import InstrumentTable


def to_entity(table_row: InstrumentTable) -> Instrument:
    return Instrument(
        id=table_row.id,
        exchange=table_row.exchange,
        symbol=table_row.symbol,
        name=table_row.name,
        board=table_row.board,
        lot_size=table_row.lot_size,
        currency=table_row.currency,
        is_active=table_row.is_active,
        created_at=table_row.created_at,
        updated_at=table_row.updated_at,
    )


def to_table(instrument: Instrument, target: InstrumentTable | None = None) -> InstrumentTable:
    table_row = target or InstrumentTable()
    table_row.exchange = str(instrument.exchange or "").strip().lower()
    table_row.symbol = str(instrument.symbol or "").strip().upper()
    table_row.name = str(instrument.name or instrument.symbol or "").strip()
    board = str(instrument.board or "").strip().upper()
    table_row.board = board or None
    table_row.lot_size = int(instrument.lot_size) if instrument.lot_size is not None else None
    currency = str(instrument.currency or "RUB").strip().upper()
    table_row.currency = currency or "RUB"
    table_row.is_active = bool(instrument.is_active)
    return table_row
