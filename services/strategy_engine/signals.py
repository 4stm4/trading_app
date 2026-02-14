"""
Генерация торговых сигналов на основе структуры рынка и индикаторов
"""

import pandas as pd
from typing import List
from dataclasses import dataclass

from .models import TradingModel
from .risk import (
    calculate_position_risk,
    calculate_stop_by_atr,
    calculate_target_by_rr
)
from .core import (
    calculate_atr,
    calculate_structure,
    calculate_distance_to_ma,
    calculate_volume_stats
)


@dataclass
class TradingSignal:
    """Торговый сигнал с полной информацией"""
    signal: str  # 'long', 'short', 'none'
    entry: float
    stop: float
    target: float
    rr: float
    risk_rub: float
    risk_percent: float
    position_size: float
    structure: str
    phase: str
    volume_ratio: float
    atr: float
    distance_ma50_pct: float
    distance_ma200_pct: float
    rsi: float
    confidence: str  # 'high', 'medium', 'low', 'none'
    warnings: List[str]
    model_name: str

    def to_dict(self) -> dict:
        """Конвертация в словарь"""
        return {
            "signal": self.signal,
            "entry": round(self.entry, 2),
            "stop": round(self.stop, 2),
            "target": round(self.target, 2),
            "rr": round(self.rr, 2),
            "risk_rub": round(self.risk_rub, 2),
            "risk_percent": round(self.risk_percent, 2),
            "position_size": round(self.position_size, 2),
            "structure": self.structure,
            "phase": self.phase,
            "volume_ratio": round(self.volume_ratio, 2),
            "atr": round(self.atr, 2),
            "distance_ma50_pct": round(self.distance_ma50_pct, 2),
            "distance_ma200_pct": round(self.distance_ma200_pct, 2),
            "rsi": round(self.rsi, 2),
            "confidence": self.confidence,
            "warnings": self.warnings,
            "model_name": self.model_name
        }


def generate_signal(
    df: pd.DataFrame,
    deposit: float,
    model: TradingModel
) -> TradingSignal:
    """
    Генерация торгового сигнала с использованием торговой модели

    Args:
        df: DataFrame с данными и индикаторами (должен содержать OHLCV, ma50, ma200, rsi)
        deposit: Размер депозита
        model: Торговая модель

    Returns:
        TradingSignal с полной информацией о сигнале
    """
    warnings = []

    # Проверка данных
    if df.empty or len(df) < 200:
        return _create_no_signal(deposit, "Недостаточно данных", model.name)

    # Добавляем ATR если нет
    if 'atr' not in df.columns:
        df['atr'] = calculate_atr(df)

    # Текущие значения
    current_price = df['close'].iloc[-1]
    current_atr = df['atr'].iloc[-1] if not pd.isna(df['atr'].iloc[-1]) else current_price * 0.02
    current_rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[-1]) else 50
    ma50 = df['ma50'].iloc[-1] if 'ma50' in df.columns and not pd.isna(df['ma50'].iloc[-1]) else None
    ma200 = df['ma200'].iloc[-1] if 'ma200' in df.columns and not pd.isna(df['ma200'].iloc[-1]) else None

    # Структура рынка
    structure_info = calculate_structure(df)

    # Объемы
    volume_info = calculate_volume_stats(df)

    # Расстояние до MA
    dist_ma50 = calculate_distance_to_ma(current_price, ma50)
    dist_ma200 = calculate_distance_to_ma(current_price, ma200)

    # === ФИЛЬТРЫ МОДЕЛИ ===

    # Фильтр по тренду
    if model.trend_required and structure_info['structure'] == 'range':
        return _create_no_signal_with_data(
            deposit, "Требуется тренд, а рынок в рейндже",
            current_price, current_atr, current_rsi, dist_ma50, dist_ma200,
            volume_info['volume_ratio'], structure_info, model.name
        )

    # Фильтр по рейнджу
    if not model.allow_range and structure_info['structure'] == 'range':
        return _create_no_signal_with_data(
            deposit, "Модель не торгует в рейндже",
            current_price, current_atr, current_rsi, dist_ma50, dist_ma200,
            volume_info['volume_ratio'], structure_info, model.name
        )

    # Фильтр по силе тренда
    if structure_info['trend_strength'] < model.min_trend_strength:
        warnings.append(f"Слабый тренд: {structure_info['trend_strength']:.1f}%")

    # Фильтр по объему
    if volume_info['volume_ratio'] < model.min_volume_ratio:
        if model.require_impulse:
            return _create_no_signal_with_data(
                deposit, f"Недостаточный объем: {volume_info['volume_ratio']:.2f}x < {model.min_volume_ratio}x",
                current_price, current_atr, current_rsi, dist_ma50, dist_ma200,
                volume_info['volume_ratio'], structure_info, model.name
            )
        warnings.append(f"Низкий объем: {volume_info['volume_ratio']:.2f}x")

    # === ЛОГИКА СИГНАЛА ===
    signal = 'none'
    direction = None
    confidence = 'low'

    # ШОРТ
    if (structure_info['structure'] == 'downtrend' and
        ma50 is not None and ma200 is not None):

        conditions_met = []

        # 1. Цена ниже MA200
        if current_price < ma200:
            conditions_met.append('below_ma200')

        # 2. Структура downtrend
        if structure_info['structure'] == 'downtrend':
            conditions_met.append('downtrend')

        # 3. Откат к MA50
        if structure_info['phase'] == 'pullback' and abs(dist_ma50) < model.max_distance_ma50:
            conditions_met.append('pullback')
            warnings.append('Цена на откате к MA50')

        # 4. RSI не в перепроданности
        if current_rsi > model.rsi_oversold:
            conditions_met.append('rsi_ok')
        else:
            warnings.append(f'RSI перепродан: {current_rsi:.1f}')

        # 5. Объем
        if volume_info['is_impulse']:
            conditions_met.append('impulse_volume')
            confidence = 'high'
        elif volume_info['volume_ratio'] >= model.min_volume_ratio:
            conditions_met.append('volume_ok')
            confidence = 'medium'

        # 6. Фильтр перегрева
        if abs(dist_ma50) <= model.max_distance_ma50:
            conditions_met.append('not_overheated')
        else:
            warnings.append(f'Перегрев: {dist_ma50:.1f}%')

        # Формируем сигнал
        if len(conditions_met) >= 4:
            signal = 'short'
            direction = 'short'

    # ЛОНГ
    elif (structure_info['structure'] == 'uptrend' and
          ma50 is not None and ma200 is not None):

        conditions_met = []

        # 1. Цена выше MA200
        if current_price > ma200:
            conditions_met.append('above_ma200')

        # 2. Структура uptrend
        if structure_info['structure'] == 'uptrend':
            conditions_met.append('uptrend')

        # 3. Откат к MA50
        if structure_info['phase'] == 'pullback' and abs(dist_ma50) < model.max_distance_ma50:
            conditions_met.append('pullback')
            warnings.append('Цена на откате к MA50')

        # 4. RSI не в перекупленности
        if current_rsi < model.rsi_overbought:
            conditions_met.append('rsi_ok')
        else:
            warnings.append(f'RSI перекуплен: {current_rsi:.1f}')

        # 5. Объем
        if volume_info['is_impulse']:
            conditions_met.append('impulse_volume')
            confidence = 'high'
        elif volume_info['volume_ratio'] >= model.min_volume_ratio:
            conditions_met.append('volume_ok')
            confidence = 'medium'

        # 6. Фильтр перегрева
        if abs(dist_ma50) <= model.max_distance_ma50:
            conditions_met.append('not_overheated')
        else:
            warnings.append(f'Перегрев: {dist_ma50:.1f}%')

        # Формируем сигнал
        if len(conditions_met) >= 4:
            signal = 'long'
            direction = 'long'

    # Если сигнал не сформирован
    if signal == 'none':
        return TradingSignal(
            signal='none',
            entry=current_price,
            stop=0,
            target=0,
            rr=0,
            risk_rub=0,
            risk_percent=0,
            position_size=0,
            structure=structure_info['structure'],
            phase=structure_info['phase'],
            volume_ratio=volume_info['volume_ratio'],
            atr=current_atr,
            distance_ma50_pct=dist_ma50,
            distance_ma200_pct=dist_ma200,
            rsi=current_rsi,
            confidence='none',
            warnings=['Условия для входа не выполнены'],
            model_name=model.name
        )

    # Расчет стопа и тейка
    entry = current_price
    stop = calculate_stop_by_atr(entry, current_atr, model.atr_multiplier_stop, direction)
    target = calculate_target_by_rr(entry, stop, model.min_rr, direction)

    # Расчет риска
    risk_params = calculate_position_risk(
        entry, stop, target, deposit,
        model.max_risk_percent, model.min_rr
    )

    if not risk_params.valid:
        warnings.append(risk_params.reason)
        return TradingSignal(
            signal='none',
            entry=entry,
            stop=stop,
            target=target,
            rr=risk_params.rr,
            risk_rub=0,
            risk_percent=0,
            position_size=0,
            structure=structure_info['structure'],
            phase=structure_info['phase'],
            volume_ratio=volume_info['volume_ratio'],
            atr=current_atr,
            distance_ma50_pct=dist_ma50,
            distance_ma200_pct=dist_ma200,
            rsi=current_rsi,
            confidence='none',
            warnings=warnings,
            model_name=model.name
        )

    # Проверка минимальной уверенности
    confidence_levels = {'low': 0, 'medium': 1, 'high': 2}
    if confidence_levels.get(confidence, 0) < confidence_levels.get(model.min_confidence, 0):
        warnings.append(f"Уверенность {confidence} < требуемой {model.min_confidence}")
        return TradingSignal(
            signal='none',
            entry=entry,
            stop=stop,
            target=target,
            rr=risk_params.rr,
            risk_rub=risk_params.risk_rub,
            risk_percent=risk_params.risk_percent,
            position_size=risk_params.position_size,
            structure=structure_info['structure'],
            phase=structure_info['phase'],
            volume_ratio=volume_info['volume_ratio'],
            atr=current_atr,
            distance_ma50_pct=dist_ma50,
            distance_ma200_pct=dist_ma200,
            rsi=current_rsi,
            confidence=confidence,
            warnings=warnings,
            model_name=model.name
        )

    # Формируем валидный сигнал
    return TradingSignal(
        signal=signal,
        entry=entry,
        stop=stop,
        target=target,
        rr=risk_params.rr,
        risk_rub=risk_params.risk_rub,
        risk_percent=risk_params.risk_percent,
        position_size=risk_params.position_size,
        structure=structure_info['structure'],
        phase=structure_info['phase'],
        volume_ratio=volume_info['volume_ratio'],
        atr=current_atr,
        distance_ma50_pct=dist_ma50,
        distance_ma200_pct=dist_ma200,
        rsi=current_rsi,
        confidence=confidence,
        warnings=warnings if warnings else ['Сигнал валиден'],
        model_name=model.name
    )


def _create_no_signal(deposit: float, reason: str, model_name: str) -> TradingSignal:
    """Создание пустого сигнала"""
    return TradingSignal(
        signal='none',
        entry=0,
        stop=0,
        target=0,
        rr=0,
        risk_rub=0,
        risk_percent=0,
        position_size=0,
        structure='unknown',
        phase='unknown',
        volume_ratio=0,
        atr=0,
        distance_ma50_pct=0,
        distance_ma200_pct=0,
        rsi=0,
        confidence='none',
        warnings=[reason],
        model_name=model_name
    )


def _create_no_signal_with_data(
    deposit: float,
    reason: str,
    price: float,
    atr: float,
    rsi: float,
    dist_ma50: float,
    dist_ma200: float,
    volume_ratio: float,
    structure_info: dict,
    model_name: str
) -> TradingSignal:
    """Создание пустого сигнала с данными рынка"""
    return TradingSignal(
        signal='none',
        entry=price,
        stop=0,
        target=0,
        rr=0,
        risk_rub=0,
        risk_percent=0,
        position_size=0,
        structure=structure_info['structure'],
        phase=structure_info['phase'],
        volume_ratio=volume_ratio,
        atr=atr,
        distance_ma50_pct=dist_ma50,
        distance_ma200_pct=dist_ma200,
        rsi=rsi,
        confidence='none',
        warnings=[reason],
        model_name=model_name
    )
