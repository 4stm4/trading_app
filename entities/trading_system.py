from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class TradingSystem:
    owner_user_id: int
    name: str
    model_id: int | None = None
    description: str | None = None
    exchange: str = "moex"
    engine: str = "stock"
    market: str = "shares"
    board: str = ""
    timeframe: str = "1h"
    model_name: str = "balanced"
    is_active: bool = True
    id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
