"""
Определение рыночного режима и метрики перформанса по режимам.
"""

from typing import Any

import numpy as np
import pandas as pd

from .core import calculate_atr


REGIMES = ("trend", "range", "high_volatility")


def classify_market_regime(df: pd.DataFrame) -> pd.Series:
    """
    Классификация режима на каждой свече:

    if abs(MA50 - MA200) / MA200 > 1.5%:
        regime = "trend"
    elif ATR_percentile > 70%:
        regime = "high_volatility"
    else:
        regime = "range"
    """
    if df.empty:
        return pd.Series(dtype="object")

    local = df.copy()
    if "atr" not in local.columns:
        local["atr"] = calculate_atr(local)

    ma_diff_pct = np.abs((local["ma50"] - local["ma200"]) / local["ma200"]) * 100
    atr_pct = (local["atr"] / local["close"]) * 100
    atr_percentile = atr_pct.rank(pct=True) * 100

    regime = pd.Series(index=local.index, dtype="object")
    regime.loc[ma_diff_pct > 1.5] = "trend"
    regime.loc[regime.isna() & (atr_percentile > 70)] = "high_volatility"
    regime.loc[regime.isna()] = "range"

    return regime


def classify_current_regime(df_window: pd.DataFrame) -> str:
    """
    Режим на текущей свече (последняя строка df_window).
    """
    if df_window.empty:
        return "range"
    regime_series = classify_market_regime(df_window)
    if regime_series.empty:
        return "range"
    return str(regime_series.iloc[-1])


def performance_by_regime(
    trades: list[Any],
    regime_series: pd.Series,
    initial_balance: float = 100.0
) -> dict[str, dict[str, float]]:
    """
    Статистика по режимам:
    - trades
    - pf
    - winrate
    - maxdd
    - return_pct
    """
    result = {
        regime: {"trades": 0, "pf": 0.0, "winrate": 0.0, "maxdd": 0.0, "return_pct": 0.0}
        for regime in REGIMES
    }
    if not trades or regime_series.empty:
        return result

    grouped: dict[str, list[float]] = {regime: [] for regime in REGIMES}
    regime_series = regime_series.sort_index()

    for trade in trades:
        pnl = getattr(trade, "pnl", None)
        trade_regime = getattr(trade, "regime", None)
        entry_time = getattr(trade, "entry_time", None)

        if pnl is None:
            continue

        if isinstance(trade_regime, str):
            trade_regime = trade_regime.strip()

        # unknown/None/invalid -> восстанавливаем режим по времени входа
        if not trade_regime or trade_regime not in grouped:
            trade_regime = _resolve_regime_by_time(regime_series, entry_time)
            if trade_regime is None:
                trade_regime = "range"

        if trade_regime not in grouped:
            continue
        grouped[trade_regime].append(float(pnl))

    for regime, pnls in grouped.items():
        if not pnls:
            continue

        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x <= 0]

        gross_profit = float(sum(wins))
        gross_loss = abs(float(sum(losses)))
        pf = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        winrate = (len(wins) / len(pnls) * 100) if pnls else 0.0

        equity_curve = initial_balance + np.cumsum(np.array(pnls, dtype=float))
        maxdd = _max_drawdown_percent(equity_curve, initial_balance)
        return_pct = (sum(pnls) / initial_balance) * 100 if initial_balance > 0 else 0.0

        result[regime] = {
            "trades": len(pnls),
            "pf": round(pf, 2),
            "winrate": round(winrate, 2),
            "maxdd": round(maxdd, 2),
            "return_pct": round(return_pct, 2),
        }

    return result


def _max_drawdown_percent(equity_curve: np.ndarray, initial_balance: float) -> float:
    if equity_curve.size == 0:
        return 0.0

    curve = np.concatenate(([initial_balance], equity_curve))
    peaks = np.maximum.accumulate(curve)
    drawdowns = (peaks - curve) / np.where(peaks == 0, 1, peaks) * 100
    return float(np.max(drawdowns))


def _resolve_regime_by_time(regime_series: pd.Series, entry_time: Any) -> str | None:
    if entry_time is None or regime_series.empty:
        return None

    try:
        ts = pd.Timestamp(entry_time)
    except Exception:
        return None

    # 1) Прямое/срез по индексу
    try:
        subset = regime_series.loc[:ts]
        if len(subset) > 0:
            value = subset.iloc[-1]
            if isinstance(value, str) and value in REGIMES:
                return value
    except Exception:
        pass

    # 2) asof для несоответствия точного timestamp
    try:
        value = regime_series.asof(ts)
        if isinstance(value, str) and value in REGIMES:
            return value
    except Exception:
        pass

    # 3) fallback на ближайшую левую позицию
    try:
        idx = regime_series.index.searchsorted(ts, side="right") - 1
        if idx >= 0:
            value = regime_series.iloc[int(idx)]
            if isinstance(value, str) and value in REGIMES:
                return value
    except Exception:
        pass

    return None
