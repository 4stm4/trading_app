from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class RecommendedSale:
    exchange: str
    symbol: str
    timeframe: str = "1h"
    model_name: str = "balanced"
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    rr: float | None = None
    confidence: str = "none"
    market_regime: str | None = None
    status: str = "new"
    note: str | None = None
    id: int | None = None
    recommended_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
