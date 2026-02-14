"""
Модуль управления рисками
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class RiskParameters:
    """Результат расчета риска"""
    valid: bool
    rr: float
    risk_rub: float
    risk_percent: float
    position_size: float
    potential_profit: float
    potential_loss: float
    reason: str


def calculate_position_risk(
    entry: float,
    stop: float,
    target: float,
    deposit: float,
    max_risk_percent: float = 1.5,
    min_rr: float = 2.0
) -> RiskParameters:
    """
    Расчет риск-менеджмента для позиции

    Args:
        entry: Цена входа
        stop: Стоп-лосс
        target: Тейк-профит
        deposit: Размер депозита
        max_risk_percent: Максимальный риск в % от депозита
        min_rr: Минимальный RR

    Returns:
        RiskParameters с результатами расчета
    """
    risk_per_unit = abs(entry - stop)
    profit_per_unit = abs(target - entry)

    # Проверка на нулевое расстояние
    if risk_per_unit == 0:
        return RiskParameters(
            valid=False,
            rr=0,
            risk_rub=0,
            risk_percent=0,
            position_size=0,
            potential_profit=0,
            potential_loss=0,
            reason='Нулевое расстояние до стопа'
        )

    # Расчет RR
    rr = profit_per_unit / risk_per_unit

    # Проверка минимального RR
    if rr < min_rr:
        return RiskParameters(
            valid=False,
            rr=rr,
            risk_rub=0,
            risk_percent=0,
            position_size=0,
            potential_profit=0,
            potential_loss=0,
            reason=f'RR {rr:.2f} < минимального {min_rr}'
        )

    # Расчет максимального риска
    max_risk_rub = deposit * (max_risk_percent / 100)

    # Расчет размера позиции
    position_size = max_risk_rub / risk_per_unit

    # Фактический риск и профит
    actual_risk_rub = position_size * risk_per_unit
    actual_risk_percent = (actual_risk_rub / deposit) * 100
    potential_profit = position_size * profit_per_unit
    potential_loss = position_size * risk_per_unit

    return RiskParameters(
        valid=True,
        rr=rr,
        risk_rub=actual_risk_rub,
        risk_percent=actual_risk_percent,
        position_size=position_size,
        potential_profit=potential_profit,
        potential_loss=potential_loss,
        reason='OK'
    )


def validate_risk_limits(
    risk_params: RiskParameters,
    max_risk_percent: float,
    min_position_size: float = 1.0
) -> tuple[bool, list[str]]:
    """
    Валидация лимитов риска

    Args:
        risk_params: Параметры риска
        max_risk_percent: Максимальный допустимый риск в %
        min_position_size: Минимальный размер позиции

    Returns:
        (валидность, список предупреждений)
    """
    warnings = []

    if not risk_params.valid:
        return False, [risk_params.reason]

    # Проверка превышения риска
    if risk_params.risk_percent > max_risk_percent:
        warnings.append(
            f'Риск {risk_params.risk_percent:.2f}% превышает максимум {max_risk_percent}%'
        )
        return False, warnings

    # Проверка минимального размера позиции
    if risk_params.position_size < min_position_size:
        warnings.append(
            f'Размер позиции {risk_params.position_size:.1f} меньше минимума {min_position_size}'
        )
        return False, warnings

    return True, []


def calculate_stop_by_atr(
    entry: float,
    atr: float,
    multiplier: float,
    direction: str
) -> float:
    """
    Расчет стоп-лосса на основе ATR

    Args:
        entry: Цена входа
        atr: Значение ATR
        multiplier: Множитель ATR
        direction: 'long' или 'short'

    Returns:
        Цена стоп-лосса
    """
    stop_distance = atr * multiplier

    if direction == 'long':
        return entry - stop_distance
    elif direction == 'short':
        return entry + stop_distance
    else:
        raise ValueError(f"Неверное направление: {direction}")


def calculate_target_by_rr(
    entry: float,
    stop: float,
    rr: float,
    direction: str
) -> float:
    """
    Расчет тейк-профита на основе RR

    Args:
        entry: Цена входа
        stop: Стоп-лосс
        rr: Соотношение риск/прибыль
        direction: 'long' или 'short'

    Returns:
        Цена тейк-профита
    """
    risk_distance = abs(entry - stop)
    profit_distance = risk_distance * rr

    if direction == 'long':
        return entry + profit_distance
    elif direction == 'short':
        return entry - profit_distance
    else:
        raise ValueError(f"Неверное направление: {direction}")


def calculate_kelly_criterion(
    winrate: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Расчет оптимального размера позиции по критерию Келли

    Args:
        winrate: Процент выигрышных сделок (0-100)
        avg_win: Средний выигрыш
        avg_loss: Средний проигрыш (положительное число)

    Returns:
        Оптимальный процент капитала для риска (0-100)
    """
    if avg_loss == 0 or avg_win == 0:
        return 0

    w = winrate / 100  # Вероятность выигрыша
    r = avg_win / avg_loss  # Соотношение выигрыш/проигрыш

    kelly = w - ((1 - w) / r)

    # Ограничиваем Kelly, обычно используют половину или четверть
    kelly_fraction = max(0, min(kelly * 0.5, 0.25))  # Максимум 25%

    return kelly_fraction * 100


def get_risk_summary(
    risk_params: RiskParameters,
    deposit: float
) -> Dict[str, any]:
    """
    Получить сводку по риску

    Args:
        risk_params: Параметры риска
        deposit: Размер депозита

    Returns:
        Словарь с информацией о риске
    """
    return {
        'valid': risk_params.valid,
        'deposit': deposit,
        'position_size': risk_params.position_size,
        'risk_rub': risk_params.risk_rub,
        'risk_percent': risk_params.risk_percent,
        'potential_profit': risk_params.potential_profit,
        'potential_loss': risk_params.potential_loss,
        'rr': risk_params.rr,
        'expected_balance_if_win': deposit + risk_params.potential_profit,
        'expected_balance_if_loss': deposit - risk_params.potential_loss,
        'reason': risk_params.reason
    }
