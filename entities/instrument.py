from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Instrument:
    exchange: str
    symbol: str
    name: str
    board: str | None = None
    lot_size: int | None = None
    currency: str = "RUB"
    is_active: bool = True
    id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
