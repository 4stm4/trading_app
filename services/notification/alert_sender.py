"""
Отправка уведомлений для мониторинга сетапов.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests
from loguru import logger


@dataclass
class AlertSender:
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    enabled_console: bool = True

    def __post_init__(self) -> None:
        if self.telegram_token is None:
            self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if self.telegram_chat_id is None:
            self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

    def send(self, message: str, level: str = "info") -> None:
        prefix = {
            "info": "ℹ",
            "warning": "⚠",
            "error": "❌",
            "success": "✅",
        }.get(level, "ℹ")

        formatted = f"{prefix} [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"

        if self.enabled_console:
            log_method = getattr(logger, level, logger.info)
            log_method(formatted)

        self._send_telegram(formatted)

    def _send_telegram(self, message: str) -> None:
        if not self.telegram_token or not self.telegram_chat_id:
            return

        try:
            requests.post(
                f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                json={"chat_id": self.telegram_chat_id, "text": message},
                timeout=10,
            )
        except Exception:
            # Telegram отправка опциональна и не должна ломать монитор.
            return
