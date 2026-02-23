"""
Порты (абстракции) для integration boundary.
"""

from __future__ import annotations

from typing import Any, Protocol


class MonitorPriceFeedPort(Protocol):
    """Порт получения актуальных цен/ошибок для набора сетапов."""

    def collect_prices(
        self, setups: list[dict[str, Any]]
    ) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
        """
        Returns:
            prices_by_symbol: {symbol: {"price": ..., "price_step": ..., "contract_multiplier": ...}}
            errors_by_symbol: {symbol: "error text"}
        """


class MonitorNotifierPort(Protocol):
    """Порт уведомлений (console/telegram/email/etc)."""

    def send(self, message: str, level: str = "info") -> None:
        ...


class SetupPersistencePort(Protocol):
    """Порт сохранения состояния активных сетапов."""

    def save_active_setups(self, setups: list[dict[str, Any]]) -> None:
        ...
