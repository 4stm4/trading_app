from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from services.strategy_engine.monitor_core import (
    ACTIVE,
    MonitorContext,
    MonitorRiskConfig,
    WAITING_FOR_ENTRY,
    process_monitor_cycle,
)


class MonitorCoreStateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 2, 15, 12, 0, 0)
        self.risk = MonitorRiskConfig(risk_per_trade_pct=0.3, max_daily_loss_pct=100.0, equity_dd_stop_pct=100.0)

    def test_normal_cross_does_not_close_tp_in_same_tick(self) -> None:
        setup = {
            "symbol": "SiH6",
            "direction": "long",
            "entry_price": 100.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
            "rr": 2.0,
            "status": WAITING_FOR_ENTRY,
            "expires_at": (self.now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        }

        cycle1 = process_monitor_cycle(
            [setup],
            prices_by_symbol={"SiH6": {"price": 99.0, "price_step": 1.0, "contract_multiplier": 1.0}},
            errors_by_symbol={},
            now=self.now,
            context=MonitorContext(),
            risk=self.risk,
            skip_entry_for_waiting=False,
        )
        self.assertEqual(len(cycle1.active_setups), 1)
        self.assertEqual(cycle1.active_setups[0]["status"], WAITING_FOR_ENTRY)

        cycle2 = process_monitor_cycle(
            cycle1.active_setups,
            prices_by_symbol={"SiH6": {"price": 101.0, "price_step": 1.0, "contract_multiplier": 1.0}},
            errors_by_symbol={},
            now=self.now + timedelta(minutes=10),
            context=cycle1.context,
            risk=self.risk,
            skip_entry_for_waiting=False,
        )
        self.assertEqual(len(cycle2.active_setups), 1)
        self.assertEqual(cycle2.active_setups[0]["status"], ACTIVE)
        self.assertTrue(any("ENTRY HIT" in e.message for e in cycle2.events))
        self.assertFalse(any("TAKE PROFIT" in e.message for e in cycle2.events))

        cycle3 = process_monitor_cycle(
            cycle2.active_setups,
            prices_by_symbol={"SiH6": {"price": 112.0, "price_step": 1.0, "contract_multiplier": 1.0}},
            errors_by_symbol={},
            now=self.now + timedelta(minutes=20),
            context=cycle2.context,
            risk=self.risk,
            skip_entry_for_waiting=False,
        )
        self.assertEqual(len(cycle3.active_setups), 0)
        self.assertTrue(any("TAKE PROFIT" in e.message for e in cycle3.events))

    def test_missed_entry_when_first_observation_already_beyond_entry(self) -> None:
        setup = {
            "symbol": "RIH6",
            "direction": "short",
            "entry_price": 100.0,
            "stop_loss": 105.0,
            "take_profit": 90.0,
            "rr": 2.0,
            "status": WAITING_FOR_ENTRY,
            "expires_at": (self.now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        }

        cycle = process_monitor_cycle(
            [setup],
            prices_by_symbol={"RIH6": {"price": 99.0, "price_step": 1.0, "contract_multiplier": 1.0}},
            errors_by_symbol={},
            now=self.now,
            context=MonitorContext(),
            risk=self.risk,
            skip_entry_for_waiting=False,
        )
        self.assertEqual(len(cycle.active_setups), 0)
        self.assertTrue(any("MISSED ENTRY" in e.message for e in cycle.events))

    def test_invalid_scale_marks_setup_stale_and_deactivates(self) -> None:
        setup = {
            "symbol": "MXH6",
            "direction": "long",
            "entry_price": 3350.0,
            "stop_loss": 3300.0,
            "take_profit": 3450.0,
            "rr": 2.0,
            "status": WAITING_FOR_ENTRY,
            "expires_at": (self.now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
        }

        cycle = process_monitor_cycle(
            [setup],
            prices_by_symbol={"MXH6": {"price": 282200.0, "price_step": 1.0, "contract_multiplier": 1.0}},
            errors_by_symbol={},
            now=self.now,
            context=MonitorContext(),
            risk=self.risk,
            skip_entry_for_waiting=False,
        )
        self.assertEqual(len(cycle.active_setups), 0)
        self.assertTrue(any("status=STALE" in e.message for e in cycle.events))


if __name__ == "__main__":
    unittest.main()
