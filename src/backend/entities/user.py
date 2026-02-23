from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class User:
    email: str
    password_hash: str
    is_active: bool = True
    id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
