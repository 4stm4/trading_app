"""
Application-service оркестрация мониторинга сетапов через порты.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from .monitor_core import (
    MonitorContext,
    MonitorRiskConfig,
    prepare_active_setups,
    process_monitor_cycle,
)
from .ports import MonitorNotifierPort, MonitorPriceFeedPort, SetupPersistencePort


@dataclass(frozen=True)
class MonitorRunConfig:
    interval_seconds: int = 15
    max_cycles: int = 0
    skip_entry_on_first_cycle: bool = True


def run_monitoring(
    *,
    setups: list[dict[str, Any]],
    price_feed: MonitorPriceFeedPort,
    notifier: MonitorNotifierPort,
    risk_config: MonitorRiskConfig,
    run_config: MonitorRunConfig | None = None,
    persistence: SetupPersistencePort | None = None,
    now_fn: Callable[[], datetime] = datetime.now,
    sleep_fn: Callable[[int], None] = time.sleep,
) -> list[dict[str, Any]]:
    config = run_config or MonitorRunConfig()
    active = prepare_active_setups(setups)
    context = MonitorContext()
    notifier.send(f"Мониторинг запущен. Активных сетапов: {len(active)}", level="info")

    while active:
        context.cycle += 1
        now = now_fn()
        prices_by_symbol, errors_by_symbol = price_feed.collect_prices(active)
        result = process_monitor_cycle(
            active_setups=active,
            prices_by_symbol=prices_by_symbol,
            errors_by_symbol=errors_by_symbol,
            now=now,
            context=context,
            risk=risk_config,
            skip_entry_for_waiting=(config.skip_entry_on_first_cycle and context.cycle == 1),
        )

        for event in result.events:
            notifier.send(event.message, level=event.level)

        active = result.active_setups
        if persistence is not None:
            persistence.save_active_setups(active)

        if not active:
            notifier.send("Мониторинг завершен: активных сетапов не осталось", level="info")
            break

        if result.should_stop:
            break

        if config.max_cycles > 0 and context.cycle >= config.max_cycles:
            notifier.send(
                f"Мониторинг остановлен по лимиту циклов ({config.max_cycles}). Активных: {len(active)}",
                level="info",
            )
            break

        sleep_fn(config.interval_seconds)

    return active
