from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class MarketCandle:
    exchange: str
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    engine: str = "stock"
    market: str = "shares"
    board: str | None = None
    id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
