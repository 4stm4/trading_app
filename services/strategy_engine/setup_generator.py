"""
Генератор торговых сетапов для последующего мониторинга/алертов.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .core import calculate_atr, calculate_structure, calculate_volume_stats
from .models import TradingModel

from .regime_detection import classify_current_regime


@dataclass
class TradeSetup:
    symbol: str
    direction: str
    regime: str
    entry_price: float
    stop_loss: float
    take_profit: float
    rr: float
    risk_percent: float
    volume_condition: str
    trend_condition: str
    trigger_condition: str
    confidence: float
    expires_at: str
    created_at: str
    expires_in_candles: int
    model: str
    timeframe: str
    board: str
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("entry_price", "stop_loss", "take_profit", "rr", "risk_percent", "confidence"):
            data[key] = round(float(data[key]), 4)
        return data


@dataclass
class SetupGenerationResult:
    setup: TradeSetup | None
    reasons: list[str]


def generate_trade_setup(
    symbol: str,
    df: pd.DataFrame,
    model: TradingModel,
    timeframe: str,
    board: str,
    volume_mode: str = "fixed",
    structure_mode: str = "strict",
    expiry_candles: int = 5,
    disable_rr: bool = False,
    disable_volume: bool = False,
    disable_trend: bool = False,
    rsi_enabled: bool = True,
    rsi_trend_confirmation_only: bool = False,
    atr_enabled: bool = True,
    atr_min_percentile: int = 0,
) -> SetupGenerationResult:
    reasons: list[str] = []

    if df.empty or len(df) < 220:
        return SetupGenerationResult(None, ["Недостаточно данных для генерации сетапа"])

    local = df.copy()
    if "atr" not in local.columns:
        local["atr"] = calculate_atr(local)

    current_price = _safe_float(local["close"].iloc[-1])
    current_atr = _safe_float(local["atr"].iloc[-1])
    current_rsi = _safe_float(local["rsi"].iloc[-1], default=50.0)
    ma50 = _safe_float(local["ma50"].iloc[-1], default=0.0)
    ma200 = _safe_float(local["ma200"].iloc[-1], default=0.0)

    if current_atr <= 0:
        return SetupGenerationResult(None, ["Нет валидного ATR для постановки стопа"])

    regime = classify_current_regime(local)
    if not disable_trend:
        regime_ok, regime_reason = _is_regime_compatible(model, regime)
        if not regime_ok:
            return SetupGenerationResult(None, [regime_reason])

    volume_info = calculate_volume_stats(local, mode=volume_mode)
    required_volume_ratio = _resolve_required_volume_ratio(model, volume_info, volume_mode)
    volume_ratio = float(volume_info.get("volume_ratio", 0.0))
    volume_ok = volume_ratio >= required_volume_ratio
    if model.require_impulse:
        volume_ok = volume_ok and bool(volume_info.get("is_impulse", False))

    if not disable_volume and not volume_ok:
        return SetupGenerationResult(
            None,
            [
                (
                    "Volume filter не выполнен: "
                    f"{volume_ratio:.2f} < {required_volume_ratio:.2f}"
                )
            ],
        )

    atr_percentile = _atr_percentile(local)
    if atr_enabled and atr_percentile > 95:
        return SetupGenerationResult(
            None,
            [
                (
                    "ATR conflict: волатильность вне рабочего диапазона "
                    f"({atr_percentile:.1f} percentile)"
                )
            ],
        )
    if atr_enabled and atr_percentile < atr_min_percentile:
        return SetupGenerationResult(
            None,
            [f"ATR percentile {atr_percentile:.1f} < min {atr_min_percentile}"],
        )

    # Расчет структуры выполняем для консистентности с остальным движком (mode может быть strict/simple).
    calculate_structure(local, mode=structure_mode)

    if regime == "range":
        setup_data = _build_range_setup(
            df=local,
            atr=current_atr,
            model=model,
            price=current_price,
            ma50=ma50,
            ma200=ma200,
        )
    elif regime == "high_volatility":
        setup_data = _build_trend_setup(
            df=local,
            atr=current_atr * 1.2,
            model=model,
            ma50=ma50,
            ma200=ma200,
            trigger_suffix=" with volatility confirmation",
            trend_condition="high volatility with MA bias",
        )
    else:
        setup_data = _build_trend_setup(
            df=local,
            atr=current_atr,
            model=model,
            ma50=ma50,
            ma200=ma200,
        )

    if setup_data is None:
        return SetupGenerationResult(None, ["Нет валидного trigger для текущего режима"])

    direction, entry, stop, take, trend_condition, trigger_condition = setup_data

    rr = _calculate_rr(direction, entry, stop, take)
    if not disable_rr and rr < model.min_rr:
        return SetupGenerationResult(None, [f"RR filter не выполнен: {rr:.2f} < {model.min_rr:.2f}"])

    if not disable_rr and regime == "range" and rr < 1.6:
        return SetupGenerationResult(None, [f"Для range требуется RR >= 1.6, получено {rr:.2f}"])

    apply_rsi = rsi_enabled and (not rsi_trend_confirmation_only or regime == "trend")
    rsi_ok = True
    if apply_rsi:
        rsi_ok = (direction == "long" and current_rsi < model.rsi_overbought) or (
            direction == "short" and current_rsi > model.rsi_oversold
        )
    if not rsi_ok:
        return SetupGenerationResult(None, ["RSI находится в конфликтной зоне"])

    trend_confirmed = _trend_confirmed(direction, regime, ma50, ma200)
    atr_normal = 20 <= atr_percentile <= 80
    confidence = _build_confidence(
        volume_ok=volume_ok,
        trend_confirmed=trend_confirmed,
        rsi_ok=rsi_ok,
        atr_normal=atr_normal,
    )

    generated_dt = datetime.now()
    expires_dt = generated_dt + timeframe_to_timedelta(timeframe) * expiry_candles

    setup = TradeSetup(
        symbol=symbol,
        direction=direction,
        regime=regime,
        entry_price=entry,
        stop_loss=stop,
        take_profit=take,
        rr=rr,
        risk_percent=model.max_risk_percent,
        volume_condition=f"volume_ratio >= {required_volume_ratio:.2f}",
        trend_condition=trend_condition,
        trigger_condition=trigger_condition,
        confidence=confidence,
        expires_at=expires_dt.strftime("%Y-%m-%d %H:%M"),
        created_at=generated_dt.strftime("%Y-%m-%d %H:%M"),
        expires_in_candles=expiry_candles,
        model=model.name,
        timeframe=timeframe,
        board=board,
    )
    return SetupGenerationResult(setup, reasons)


def export_setups_json(path: str, setups: Sequence[TradeSetup | dict[str, Any]]) -> None:
    payload: list[dict[str, Any]] = []
    for setup in setups:
        if isinstance(setup, TradeSetup):
            payload.append(setup.to_dict())
        else:
            payload.append(dict(setup))

    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def load_setups_json(path: str) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []

    with target.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    mapping = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "10m": timedelta(minutes=10),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
        "1w": timedelta(weeks=1),
        "1M": timedelta(days=30),
    }
    return mapping.get(timeframe, timedelta(minutes=10))


def _build_trend_setup(
    df: pd.DataFrame,
    atr: float,
    model: TradingModel,
    ma50: float,
    ma200: float,
    trigger_suffix: str = "",
    trend_condition: str | None = None,
) -> tuple[str, float, float, float, str, str] | None:
    prev_high = _safe_float(df["high"].iloc[-2])
    prev_low = _safe_float(df["low"].iloc[-2])

    if ma50 > ma200:
        direction = "long"
        entry = prev_high
        stop = entry - atr * model.atr_multiplier_stop
        take = entry + (entry - stop) * model.min_rr
        trend_condition = trend_condition or "MA50 > MA200"
        trigger_condition = f"close > previous_high{trigger_suffix}"
    elif ma50 < ma200:
        direction = "short"
        entry = prev_low
        stop = entry + atr * model.atr_multiplier_stop
        take = entry - (stop - entry) * model.min_rr
        trend_condition = trend_condition or "MA50 < MA200"
        trigger_condition = f"close < previous_low{trigger_suffix}"
    else:
        return None

    if direction == "long" and not (stop < entry < take):
        return None
    if direction == "short" and not (take < entry < stop):
        return None

    return direction, entry, stop, take, trend_condition, trigger_condition


def _build_range_setup(
    df: pd.DataFrame,
    atr: float,
    model: TradingModel,
    price: float,
    ma50: float,
    ma200: float,
    window: int = 40,
) -> tuple[str, float, float, float, str, str] | None:
    tail = df.tail(window)
    range_high = _safe_float(tail["high"].max())
    range_low = _safe_float(tail["low"].min())

    if range_high <= range_low:
        return None

    mid = (range_high + range_low) / 2.0
    edge_threshold = atr * 0.7

    min_rr = max(model.min_rr, 1.6)
    trend_condition = "MA50 ~ MA200"

    if price <= range_low + edge_threshold:
        direction = "long"
        entry = max(price, range_low + atr * 0.2)
        stop = range_low - atr * 0.5
        take = mid
        risk_per_unit = max(entry - stop, 1e-9)
        if _calculate_rr(direction, entry, stop, take) < min_rr:
            take = entry + risk_per_unit * min_rr
        trigger = "close rebounds from range_low"
    elif price >= range_high - edge_threshold:
        direction = "short"
        entry = min(price, range_high - atr * 0.2)
        stop = range_high + atr * 0.5
        take = mid
        risk_per_unit = max(stop - entry, 1e-9)
        if _calculate_rr(direction, entry, stop, take) < min_rr:
            take = entry - risk_per_unit * min_rr
        trigger = "close rebounds from range_high"
    else:
        return None

    if direction == "long" and not (stop < entry < take):
        return None
    if direction == "short" and not (take < entry < stop):
        return None

    return direction, entry, stop, take, trend_condition, trigger


def _calculate_rr(direction: str, entry: float, stop: float, take: float) -> float:
    if direction == "long":
        risk = entry - stop
        reward = take - entry
    else:
        risk = stop - entry
        reward = entry - take

    if risk <= 0:
        return 0.0
    return reward / risk


def _resolve_required_volume_ratio(model: TradingModel, volume_info: dict[str, Any], volume_mode: str) -> float:
    if volume_mode == "adaptive":
        threshold = float(volume_info.get("threshold_ratio", 0.0))
        if threshold <= 0:
            threshold = model.min_volume_ratio
    else:
        threshold = model.min_volume_ratio

    if model.require_impulse:
        threshold = max(threshold, 1.5)

    return float(threshold)


def _is_regime_compatible(model: TradingModel, regime: str) -> tuple[bool, str]:
    # Для adaptive-regime моделей режим задан самим именем модели.
    if model.name.startswith("regime_trend") and regime != "trend":
        return False, f"Regime mismatch: модель {model.name} требует trend, текущий режим {regime}"

    if model.name.startswith("regime_range") and regime != "range":
        return False, f"Regime mismatch: модель {model.name} требует range, текущий режим {regime}"

    if model.name.startswith("regime_high_volatility") and regime != "high_volatility":
        return (
            False,
            (
                f"Regime mismatch: модель {model.name} требует high_volatility, "
                f"текущий режим {regime}"
            ),
        )

    if model.trend_required and regime != "trend":
        return False, f"Regime mismatch: модель {model.name} требует trend, текущий режим {regime}"

    if regime == "range" and not model.allow_range:
        return False, f"Regime mismatch: модель {model.name} не торгует range"

    return True, ""


def _trend_confirmed(direction: str, regime: str, ma50: float, ma200: float) -> bool:
    if regime == "range":
        if ma200 == 0:
            return False
        return abs((ma50 - ma200) / ma200) * 100 < 0.5

    if direction == "long":
        return ma50 > ma200
    return ma50 < ma200


def _build_confidence(volume_ok: bool, trend_confirmed: bool, rsi_ok: bool, atr_normal: bool) -> float:
    score = 0.0
    if volume_ok:
        score += 0.3
    if trend_confirmed:
        score += 0.3
    if rsi_ok:
        score += 0.2
    if atr_normal:
        score += 0.2
    return min(1.0, score)


def _atr_percentile(df: pd.DataFrame) -> float:
    atr_pct = (df["atr"] / df["close"] * 100).replace([pd.NA, pd.NaT], 0).fillna(0)
    if atr_pct.empty:
        return 0.0
    ranked = atr_pct.rank(pct=True) * 100
    return float(ranked.iloc[-1])


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(val):
        return default
    return val
