"""
Бирженезависимое ядро мониторинга торговых сетапов.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

WAITING_FOR_ENTRY = "WAITING_FOR_ENTRY"
ACTIVE = "ACTIVE"
CLOSED_SL = "CLOSED_SL"
CLOSED_TP = "CLOSED_TP"
EXPIRED = "EXPIRED"
MISSED_ENTRY = "MISSED_ENTRY"
INVALID = "INVALID"
STALE = "STALE"


@dataclass(frozen=True)
class MonitorRiskConfig:
    risk_per_trade_pct: float = 0.7
    max_daily_loss_pct: float = 3.0
    equity_dd_stop_pct: float = 15.0


@dataclass
class MonitorContext:
    cycle: int = 0
    equity: float = 100.0
    peak_equity: float = 100.0
    daily_pnl_pct: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MonitorEvent:
    level: str
    message: str


@dataclass(frozen=True)
class MonitorCycleResult:
    active_setups: list[dict[str, Any]]
    events: list[MonitorEvent]
    context: MonitorContext
    should_stop: bool = False


def prepare_active_setups(setups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    closed_statuses = {CLOSED_SL, CLOSED_TP, EXPIRED, MISSED_ENTRY, INVALID, STALE}
    for setup in setups:
        item = dict(setup)
        raw_status = str(item.get("status", "")).strip().upper()
        if raw_status in {"PENDING", "", "WAITING"}:
            item["status"] = WAITING_FOR_ENTRY
        elif raw_status in {ACTIVE, WAITING_FOR_ENTRY, *closed_statuses}:
            item["status"] = raw_status
        else:
            item["status"] = WAITING_FOR_ENTRY
        if item["status"] in closed_statuses:
            continue
        active.append(item)
    return active


def process_monitor_cycle(
    active_setups: list[dict[str, Any]],
    *,
    prices_by_symbol: dict[str, dict[str, float]],
    errors_by_symbol: dict[str, str],
    now: datetime,
    context: MonitorContext,
    risk: MonitorRiskConfig,
    skip_entry_for_waiting: bool = False,
) -> MonitorCycleResult:
    next_active: list[dict[str, Any]] = []
    events: list[MonitorEvent] = []
    day_key = now.strftime("%Y-%m-%d")

    for setup in active_setups:
        item = dict(setup)
        symbol = str(item.get("symbol", ""))
        direction = str(item.get("direction", "")).lower().strip()
        status = str(item.get("status", WAITING_FOR_ENTRY)).strip().upper()

        entry = _to_float(item.get("entry_price"))
        stop = _to_float(item.get("stop_loss"))
        take = _to_float(item.get("take_profit"))
        expires_at = _parse_dt(item.get("expires_at"))

        if expires_at and now >= expires_at and status == WAITING_FOR_ENTRY:
            item["status"] = EXPIRED
            events.append(
                MonitorEvent(
                    level="warning",
                    message=(
                        f"Сетап истек: {symbol} {direction.upper()} | "
                        f"status={item['status']} | reason=expiry before activation"
                    ),
                )
            )
            continue

        if symbol in errors_by_symbol:
            events.append(
                MonitorEvent(
                    level="warning",
                    message=f"Ошибка получения цены {symbol}: {errors_by_symbol[symbol]}",
                )
            )
            next_active.append(item)
            continue

        price_info = prices_by_symbol.get(symbol)
        if not price_info:
            next_active.append(item)
            continue
        current_price_raw = _to_float(price_info.get("price"))
        if current_price_raw <= 0:
            next_active.append(item)
            continue

        price_step = _to_float_or_none(price_info.get("price_step")) or _infer_price_step(
            entry, stop, take, current_price_raw
        )
        contract_multiplier = _to_float_or_none(price_info.get("contract_multiplier")) or 1.0
        atr_value = _to_float_or_none(item.get("atr"))
        normalized = normalize_price_geometry(
            direction=direction,
            entry=entry,
            stop=stop,
            take=take,
            current_price=current_price_raw,
            price_step=price_step,
            contract_multiplier=contract_multiplier,
        )
        if not normalized.valid:
            item["status"] = INVALID
            item["deactivation_reason"] = normalized.reason
            events.append(
                MonitorEvent(
                    level="warning",
                    message=(
                        f"Некорректная геометрия сетапа {symbol}: {normalized.reason} | "
                        f"status={item['status']} | prev={_fmt_price(item.get('last_price'))} "
                        f"cur={normalized.current_price:.4f}"
                    ),
                )
            )
            continue

        entry = normalized.entry
        stop = normalized.stop
        take = normalized.take
        price = normalized.current_price
        item["entry_price"] = entry
        item["stop_loss"] = stop
        item["take_profit"] = take

        prev_price = _to_float_or_none(item.get("last_price"))

        if status == WAITING_FOR_ENTRY:
            stale_reason = _validate_scale_sanity(
                entry=entry,
                current=price,
                atr=atr_value,
            )
            if stale_reason:
                item["status"] = STALE
                item["deactivation_reason"] = stale_reason
                events.append(
                    MonitorEvent(
                        level="warning",
                        message=(
                            f"Сетап деактивирован: {symbol} {direction.upper()} | "
                            f"status={item['status']} | reason={stale_reason} | "
                            f"prev={_fmt_price(item.get('last_price'))} cur={price:.4f}"
                        ),
                    )
                )
                continue

            if prev_price is None:
                if _is_trigger_side(direction, price, entry):
                    item["status"] = MISSED_ENTRY
                    item["deactivation_reason"] = "price already beyond entry at first observation"
                    events.append(
                        MonitorEvent(
                            level="warning",
                            message=(
                                f"MISSED ENTRY: {symbol} {direction.upper()} | "
                                f"status={item['status']} | reason={item['deactivation_reason']} | "
                                f"prev=None cur={price:.4f} entry={entry:.4f}"
                            ),
                        )
                    )
                    continue
                item["last_price"] = price
                next_active.append(item)
                continue

            if _entry_crossed(direction, prev_price, price, entry):
                item["status"] = ACTIVE
                item["activated_at"] = now.strftime("%Y-%m-%d %H:%M:%S")
                item["last_price"] = price
                events.append(
                    MonitorEvent(
                        level="success",
                        message=(
                            f"ENTRY HIT: {symbol} {direction.upper()} | status={item['status']} | "
                            f"reason=entry crossed | prev={prev_price:.4f} cur={price:.4f} entry={entry:.4f}"
                        ),
                    )
                )
                # Не допускаем ENTRY + TP/SL в одном тике.
                next_active.append(item)
                continue

            if _is_trigger_side(direction, price, entry):
                item["status"] = MISSED_ENTRY
                item["deactivation_reason"] = "entry already passed without crossing event"
                events.append(
                    MonitorEvent(
                        level="warning",
                        message=(
                            f"MISSED ENTRY: {symbol} {direction.upper()} | status={item['status']} | "
                            f"reason={item['deactivation_reason']} | prev={prev_price:.4f} "
                            f"cur={price:.4f} entry={entry:.4f}"
                        ),
                    )
                )
                continue

            item["last_price"] = price
            next_active.append(item)
            continue

        # Пока setup не активирован, стоп/тейк не проверяем.
        if status == ACTIVE:
            item["last_price"] = price
            if _take_hit(direction, price, take):
                item["status"] = CLOSED_TP
                item["closed_at"] = now.strftime("%Y-%m-%d %H:%M:%S")
                item["close_price"] = price
                rr = max(_to_float(item.get("rr")), 0.0)
                pnl_pct = risk.risk_per_trade_pct * rr
                _apply_pnl(context, day_key, pnl_pct)
                events.append(
                    MonitorEvent(
                        level="success",
                        message=(
                            f"TAKE PROFIT: {symbol} {direction.upper()} | status={item['status']} | "
                            f"reason=take reached | prev={_fmt_price(prev_price)} cur={price:.4f} tp={take:.4f}"
                        ),
                    )
                )
                continue
            if _stop_hit(direction, price, stop):
                item["status"] = CLOSED_SL
                item["closed_at"] = now.strftime("%Y-%m-%d %H:%M:%S")
                item["close_price"] = price
                pnl_pct = -risk.risk_per_trade_pct
                _apply_pnl(context, day_key, pnl_pct)
                events.append(
                    MonitorEvent(
                        level="error",
                        message=(
                            f"STOP LOSS: {symbol} {direction.upper()} | status={item['status']} | "
                            f"reason=stop reached | prev={_fmt_price(prev_price)} cur={price:.4f} stop={stop:.4f}"
                        ),
                    )
                )
                continue

        next_active.append(item)

    should_stop = False
    day_pnl = context.daily_pnl_pct.get(day_key, 0.0)
    if risk.max_daily_loss_pct > 0 and day_pnl <= -risk.max_daily_loss_pct:
        events.append(
            MonitorEvent(
                level="warning",
                message=(
                    "Мониторинг остановлен: достигнут лимит дневного убытка "
                    f"{risk.max_daily_loss_pct:.2f}%"
                ),
            )
        )
        should_stop = True

    dd_pct = drawdown_pct(context.equity, context.peak_equity)
    if (not should_stop) and risk.equity_dd_stop_pct > 0 and dd_pct >= risk.equity_dd_stop_pct:
        events.append(
            MonitorEvent(
                level="error",
                message=(
                    "Мониторинг остановлен: достигнут equity DD stop "
                    f"{dd_pct:.2f}% >= {risk.equity_dd_stop_pct:.2f}%"
                ),
            )
        )
        should_stop = True

    return MonitorCycleResult(
        active_setups=next_active,
        events=events,
        context=context,
        should_stop=should_stop,
    )


def drawdown_pct(equity: float, peak: float) -> float:
    if peak <= 0:
        return 0.0
    return (peak - equity) / peak * 100.0


def _apply_pnl(context: MonitorContext, day_key: str, pnl_pct: float) -> None:
    context.equity += pnl_pct
    context.peak_equity = max(context.peak_equity, context.equity)
    context.daily_pnl_pct[day_key] = context.daily_pnl_pct.get(day_key, 0.0) + pnl_pct


def _entry_crossed(direction: str, prev_price: float, cur_price: float, entry: float) -> bool:
    if direction == "long":
        return prev_price < entry <= cur_price
    return prev_price > entry >= cur_price


def _is_trigger_side(direction: str, price: float, entry: float) -> bool:
    if direction == "long":
        return price >= entry
    return price <= entry


def _take_hit(direction: str, price: float, take: float) -> bool:
    if direction == "long":
        return price >= take
    return price <= take


def _stop_hit(direction: str, price: float, stop: float) -> bool:
    if direction == "long":
        return price <= stop
    return price >= stop


def _validate_scale_sanity(
    *,
    entry: float,
    current: float,
    atr: Optional[float],
    max_dist_pct: float = 0.05,
    atr_multiplier: float = 4.0,
) -> Optional[str]:
    if current <= 0:
        return "current price is non-positive"
    diff = abs(entry - current)
    dist_pct = diff / current
    atr_violation = atr is not None and atr > 0 and diff > atr_multiplier * atr
    if dist_pct > max_dist_pct or atr_violation:
        reason = f"scale sanity failed (dist={dist_pct:.2%}, diff={diff:.4f})"
        if atr_violation:
            reason += f", atr={atr:.4f}, atr_limit={atr_multiplier * atr:.4f}"
        return reason
    return None


@dataclass(frozen=True)
class NormalizedGeometry:
    valid: bool
    reason: str
    entry: float
    stop: float
    take: float
    current_price: float
    scale_factor: float


def normalize_price_geometry(
    *,
    direction: str,
    entry: float,
    stop: float,
    take: float,
    current_price: float,
    price_step: float,
    contract_multiplier: float,
) -> NormalizedGeometry:
    direction = direction.lower().strip()
    if direction not in {"long", "short"}:
        return NormalizedGeometry(False, f"unknown direction: {direction}", entry, stop, take, current_price, 1.0)

    if min(entry, stop, take, current_price) <= 0:
        return NormalizedGeometry(False, "non-positive price levels", entry, stop, take, current_price, 1.0)

    step = price_step if price_step and price_step > 0 else 0.0
    mult = contract_multiplier if contract_multiplier and contract_multiplier > 0 else 1.0
    candidates = [1.0]
    if mult != 1.0:
        candidates.extend([mult, 1.0 / mult])

    # Выбираем шкалу setup-уровней, максимально близкую к текущей цене.
    best_scale = 1.0
    best_dist = float("inf")
    for scale in candidates:
        e = _quantize(entry * scale, step)
        s = _quantize(stop * scale, step)
        t = _quantize(take * scale, step)
        median_level = sorted([e, s, t])[1]
        dist = abs((median_level / current_price) - 1.0)
        if dist < best_dist:
            best_dist = dist
            best_scale = scale

    e = _quantize(entry * best_scale, step)
    s = _quantize(stop * best_scale, step)
    t = _quantize(take * best_scale, step)
    p = _quantize(current_price, step)

    if direction == "long":
        valid = s < e < t
        reason = "LONG geometry invalid: require stop < entry < take_profit"
    else:
        valid = t < e < s
        reason = "SHORT geometry invalid: require take_profit < entry < stop"

    return NormalizedGeometry(valid, "OK" if valid else reason, e, s, t, p, best_scale)


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(str(value), pattern)
        except ValueError:
            continue
    return None


def _to_float_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _quantize(value: float, step: float) -> float:
    if step <= 0:
        return float(value)
    return round(round(value / step) * step, 10)


def _infer_price_step(entry: float, stop: float, take: float, current: float) -> float:
    for v in (entry, stop, take, current):
        s = f"{v:.10f}".rstrip("0")
        if "." in s:
            decimals = len(s.split(".")[1])
            if decimals > 0:
                return 10 ** (-decimals)
    return 0.0


def _fmt_price(value: Any) -> str:
    number = _to_float_or_none(value)
    if number is None:
        return "None"
    return f"{number:.4f}"
