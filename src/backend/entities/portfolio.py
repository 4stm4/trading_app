from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Portfolio:
    owner_user_id: int
    balance: float = 100000.0
    currency: str = "RUB"
    is_active: bool = True
    id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
