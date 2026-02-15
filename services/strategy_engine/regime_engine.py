"""
Движок режима рынка и защитных ограничений по режимам.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import pandas as pd

from .regime_detection import REGIMES, classify_current_regime


@dataclass
class RegimeEngine:
    """
    Хранит производительность по режимам и умеет временно отключать режим.
    """
    pf_window: int = 20
    min_trades_for_block: int = 5
    disable_threshold_pf: float = 1.0
    thresholds_by_regime: Optional[dict[str, float]] = None
    enabled_regimes: Optional[set[str]] = None
    _recent_pnls: dict[str, Deque[float]] = field(default_factory=dict)
    _blocked: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        for regime in REGIMES:
            self._recent_pnls[regime] = deque(maxlen=self.pf_window)
            self._blocked[regime] = False

    def detect_regime(self, df_window: pd.DataFrame) -> str:
        return classify_current_regime(df_window)

    def is_regime_enabled(self, regime: str) -> bool:
        if self.enabled_regimes is not None and regime not in self.enabled_regimes:
            return False
        return not self._blocked.get(regime, False)

    def register_trade_result(self, regime: str, pnl: float):
        if regime not in self._recent_pnls:
            return

        self._recent_pnls[regime].append(float(pnl))
        pf = self.pf_for_regime(regime)
        trades_count = len(self._recent_pnls[regime])
        threshold = self._threshold_for_regime(regime)

        if trades_count >= max(self.min_trades_for_block, self.pf_window) and pf < threshold:
            self._blocked[regime] = True
        elif pf >= threshold:
            self._blocked[regime] = False

    def pf_for_regime(self, regime: str) -> float:
        pnls = list(self._recent_pnls.get(regime, []))
        if not pnls:
            return 0.0

        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x <= 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))

        if gross_loss == 0:
            return float(gross_profit) if gross_profit > 0 else 0.0
        return float(gross_profit / gross_loss)

    def blocked_regimes(self) -> dict[str, bool]:
        return dict(self._blocked)

    def _threshold_for_regime(self, regime: str) -> float:
        if self.thresholds_by_regime and regime in self.thresholds_by_regime:
            return float(self.thresholds_by_regime[regime])
        return float(self.disable_threshold_pf)
