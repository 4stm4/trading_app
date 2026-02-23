"""
Генерация торговых сигналов на основе структуры рынка и индикаторов
"""

import pandas as pd
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .conservative_v2 import (
    ConservativeModelV2,
    load_conservative_v2_config,
    should_use_conservative_v2,
)
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
    market_regime: str = 'unknown'
    debug_potential_setup: bool = False
    debug_filter_stage: str = 'none'

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
            "model_name": self.model_name,
            "market_regime": self.market_regime
        }


def generate_signal(
    df: pd.DataFrame,
    deposit: float,
    model: TradingModel,
    volume_mode: str = 'fixed',
    structure_mode: str = 'strict',
    disable_rr: bool = False,
    disable_volume: bool = False,
    disable_trend: bool = False,
    debug_filters: bool = False,
    rsi_enabled: bool = True,
    rsi_trend_confirmation_only: bool = False,
    atr_enabled: bool = True,
    atr_min_percentile: int = 0,
    filter_config: Optional[Dict[str, Any]] = None,
    contract_margin_rub: Optional[float] = None,
    contract_multiplier: float = 1.0,
    soft_threshold_relaxation: float = 0.0,
    min_expected_trades_per_month: float = 0.0,
) -> TradingSignal:
    """
    Генерация торгового сигнала с использованием торговой модели

    Args:
        df: DataFrame с данными и индикаторами (должен содержать OHLCV, ma50, ma200, rsi)
        deposit: Размер депозита
        model: Торговая модель
        volume_mode: Режим volume фильтра ('fixed' | 'adaptive')
        structure_mode: Режим структуры ('strict' | 'simple')
        disable_rr: Отключить RR фильтр
        disable_volume: Отключить volume фильтр
        disable_trend: Отключить trend filter
        debug_filters: Сохранить этап фильтрации в TradingSignal для бэктеста

    Returns:
        TradingSignal с полной информацией о сигнале
    """
    warnings = []

    # Проверка данных
    if df.empty or len(df) < 200:
        return _create_no_signal(
            deposit,
            "Недостаточно данных",
            model.name,
            debug_filter_stage='no_setup'
        )

    # Добавляем ATR если нет
    if 'atr' not in df.columns:
        df = df.copy()
        df['atr'] = calculate_atr(df)

    # Текущие значения
    current_price = df['close'].iloc[-1]
    atr_raw = df['atr'].iloc[-1] if 'atr' in df.columns else None
    current_atr = float(atr_raw) if atr_raw is not None and not pd.isna(atr_raw) else current_price * 0.02
    current_rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[-1]) else 50
    ma50 = df['ma50'].iloc[-1] if 'ma50' in df.columns and not pd.isna(df['ma50'].iloc[-1]) else None
    ma200 = df['ma200'].iloc[-1] if 'ma200' in df.columns and not pd.isna(df['ma200'].iloc[-1]) else None

    # Структура рынка
    structure_info = calculate_structure(df, mode=structure_mode)

    # Объемы
    volume_info = calculate_volume_stats(df, mode=volume_mode)
    required_volume_ratio = _resolve_required_volume_ratio(model, volume_info, volume_mode)

    # Расстояние до MA
    dist_ma50 = calculate_distance_to_ma(current_price, ma50)
    dist_ma200 = calculate_distance_to_ma(current_price, ma200)

    potential_setup = ma50 is not None and ma200 is not None
    range_context = _build_range_context(df, current_price, current_atr, current_rsi)

    if should_use_conservative_v2(model.name, filter_config):
        return _generate_signal_conservative_v2(
            df=df,
            deposit=deposit,
            model=model,
            current_price=current_price,
            current_atr=current_atr,
            current_rsi=float(current_rsi),
            ma50=ma50,
            ma200=ma200,
            structure_info=structure_info,
            volume_info=volume_info,
            required_volume_ratio=required_volume_ratio,
            dist_ma50=dist_ma50,
            dist_ma200=dist_ma200,
            range_context=range_context,
            disable_rr=disable_rr,
            disable_volume=disable_volume,
            disable_trend=disable_trend,
            debug_filters=debug_filters,
            rsi_enabled=rsi_enabled,
            rsi_trend_confirmation_only=rsi_trend_confirmation_only,
            atr_enabled=atr_enabled,
            atr_min_percentile=atr_min_percentile,
            filter_config=filter_config,
            contract_margin_rub=contract_margin_rub,
            contract_multiplier=contract_multiplier,
            potential_setup=potential_setup,
            soft_threshold_relaxation=soft_threshold_relaxation,
            min_expected_trades_per_month=min_expected_trades_per_month,
        )

    # === ФИЛЬТРЫ МОДЕЛИ ===

    # Фильтр по тренду
    if not disable_trend:
        if structure_info['structure'] == 'range' and not model.allow_range:
            reason = (
                "Требуется тренд, а рынок в рейндже"
                if model.trend_required
                else "Модель не торгует в рейндже"
            )
            return _create_no_signal_with_data(
                deposit, reason,
                current_price, current_atr, current_rsi, dist_ma50, dist_ma200,
                volume_info['volume_ratio'], structure_info, model.name,
                debug_potential_setup=potential_setup if debug_filters else False,
                debug_filter_stage='filtered_trend'
            )

    # Фильтр по силе тренда (hard)
    trend_strength = float(structure_info.get('trend_strength', 0.0))
    if (
        not disable_trend
        and structure_info.get('structure') in {'uptrend', 'downtrend'}
        and trend_strength + 1e-9 < float(model.min_trend_strength)
    ):
        return _create_no_signal_with_data(
            deposit,
            f"Слабый тренд: {trend_strength:.3f}% < {float(model.min_trend_strength):.3f}%",
            current_price,
            current_atr,
            current_rsi,
            dist_ma50,
            dist_ma200,
            volume_info['volume_ratio'],
            structure_info,
            model.name,
            debug_potential_setup=potential_setup if debug_filters else False,
            debug_filter_stage='filtered_trend'
        )

    # Определяем базовое направление
    direction = None
    entry_context = 'trend'
    if structure_info['structure'] == 'downtrend' and ma50 is not None and ma200 is not None:
        direction = 'short'
    elif structure_info['structure'] == 'uptrend' and ma50 is not None and ma200 is not None:
        direction = 'long'
    elif model.allow_range and structure_info['structure'] == 'range' and range_context['is_valid']:
        direction = range_context['direction']
        entry_context = 'range'
        warnings.append(range_context['reason'])

    # Если направление не определено
    if direction is None:
        no_setup_reason = 'Условия для входа не выполнены'
        if model.allow_range and structure_info['structure'] == 'range' and not range_context['is_valid']:
            no_setup_reason = range_context['reason']
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
            warnings=[no_setup_reason],
            model_name=model.name,
            debug_potential_setup=potential_setup if debug_filters else False,
            debug_filter_stage='no_setup'
        )

    # Жесткая проверка качества тренда по MA-выравниванию.
    if entry_context == 'trend' and ma50 is not None and ma200 is not None:
        if direction == 'long' and not (ma50 > ma200):
            return _create_no_signal_with_data(
                deposit, "MA alignment failed: требуется MA50 > MA200 для LONG",
                current_price, current_atr, current_rsi, dist_ma50, dist_ma200,
                volume_info['volume_ratio'], structure_info, model.name,
                debug_potential_setup=potential_setup if debug_filters else False,
                debug_filter_stage='filtered_trend'
            )
        if direction == 'short' and not (ma50 < ma200):
            return _create_no_signal_with_data(
                deposit, "MA alignment failed: требуется MA50 < MA200 для SHORT",
                current_price, current_atr, current_rsi, dist_ma50, dist_ma200,
                volume_info['volume_ratio'], structure_info, model.name,
                debug_potential_setup=potential_setup if debug_filters else False,
                debug_filter_stage='filtered_trend'
            )

    pullback_enabled, require_pullback, max_pullback_distance_pct = _resolve_pullback_constraints(
        filter_config=filter_config,
        model=model,
    )
    pullback_reject = _check_pullback_entry(
        direction=direction,
        entry_context=entry_context,
        current_price=float(current_price),
        ma50=ma50,
        pullback_enabled=pullback_enabled,
        require_pullback=require_pullback,
        max_pullback_distance_pct=max_pullback_distance_pct,
    )
    if pullback_reject:
        return _create_no_signal_with_data(
            deposit, pullback_reject,
            current_price, current_atr, current_rsi, dist_ma50, dist_ma200,
            volume_info['volume_ratio'], structure_info, model.name,
            debug_potential_setup=potential_setup if debug_filters else False,
            debug_filter_stage='filtered_trend'
        )

    # Фильтр по объему
    if not disable_volume and volume_info['volume_ratio'] < required_volume_ratio:
        return _create_no_signal_with_data(
            deposit,
            f"Недостаточный объем: {volume_info['volume_ratio']:.2f}x < {required_volume_ratio:.2f}x",
            current_price,
            current_atr,
            current_rsi,
            dist_ma50,
            dist_ma200,
            volume_info['volume_ratio'],
            structure_info,
            model.name,
            debug_potential_setup=potential_setup if debug_filters else False,
            debug_filter_stage='filtered_volume'
        )

    # === ЛОГИКА СИГНАЛА ===
    signal = 'none'
    confidence = 'low'

    reject_reasons: List[str] = []

    if direction == 'short':
        conditions_met = []
        if entry_context == 'range':
            if range_context['direction'] == 'short':
                conditions_met.append('range_boundary_bounce')
            else:
                reject_reasons.append('range_direction_not_short')
            if 30 <= current_rsi <= 70:
                conditions_met.append('rsi_range_ok')
            else:
                warnings.append(f'RSI вне range-зоны: {current_rsi:.1f}')
                reject_reasons.append('rsi_out_of_range_band')
            if disable_volume or volume_info['volume_ratio'] >= required_volume_ratio:
                conditions_met.append('volume_ok')
                confidence = 'medium'
            else:
                reject_reasons.append('volume_below_threshold')
            if len(conditions_met) >= 3:
                signal = 'short'
            else:
                reject_reasons.append(f'conditions_short_range={len(conditions_met)}/3')
        else:
            # 1. Цена ниже MA200
            if current_price < ma200:
                conditions_met.append('below_ma200')
            else:
                reject_reasons.append('price_not_below_ma200')

            # 2. Структура downtrend
            conditions_met.append('downtrend')

            # 3. Откат к MA50 (hard-валидация уже выполнена pullback фильтром выше).
            conditions_met.append('pullback')
            if pullback_enabled:
                warnings.append('Цена в допустимой зоне pullback к MA50')

            # 4. RSI не в перепроданности
            apply_rsi = rsi_enabled and (not rsi_trend_confirmation_only or structure_info['phase'] == 'trend')
            if not apply_rsi:
                conditions_met.append('rsi_ok')
            elif current_rsi > model.rsi_oversold:
                conditions_met.append('rsi_ok')
            else:
                warnings.append(
                    f"RSI перепродан: {current_rsi:.2f} <= {float(model.rsi_oversold):.2f}"
                )
                reject_reasons.append('rsi_oversold')

            # 5. Объем
            if disable_volume or volume_info['volume_ratio'] >= required_volume_ratio:
                conditions_met.append('volume_ok')
                confidence = 'high' if (not disable_volume and volume_info['is_impulse']) else 'medium'
            else:
                reject_reasons.append('volume_below_threshold')

            # 6. Фильтр перегрева
            if abs(dist_ma50) <= model.max_distance_ma50:
                conditions_met.append('not_overheated')
            else:
                warnings.append(f'Перегрев: {dist_ma50:.1f}%')
                reject_reasons.append('distance_to_ma50_too_large')

            if len(conditions_met) >= 4:
                signal = 'short'
            else:
                reject_reasons.append(f'conditions_short_trend={len(conditions_met)}/4')

    elif direction == 'long':
        conditions_met = []
        if entry_context == 'range':
            if range_context['direction'] == 'long':
                conditions_met.append('range_boundary_bounce')
            else:
                reject_reasons.append('range_direction_not_long')
            if 30 <= current_rsi <= 70:
                conditions_met.append('rsi_range_ok')
            else:
                warnings.append(f'RSI вне range-зоны: {current_rsi:.1f}')
                reject_reasons.append('rsi_out_of_range_band')
            if disable_volume or volume_info['volume_ratio'] >= required_volume_ratio:
                conditions_met.append('volume_ok')
                confidence = 'medium'
            else:
                reject_reasons.append('volume_below_threshold')
            if len(conditions_met) >= 3:
                signal = 'long'
            else:
                reject_reasons.append(f'conditions_long_range={len(conditions_met)}/3')
        else:
            # 1. Цена выше MA200
            if current_price > ma200:
                conditions_met.append('above_ma200')
            else:
                reject_reasons.append('price_not_above_ma200')

            # 2. Структура uptrend
            conditions_met.append('uptrend')

            # 3. Откат к MA50 (hard-валидация уже выполнена pullback фильтром выше).
            conditions_met.append('pullback')
            if pullback_enabled:
                warnings.append('Цена в допустимой зоне pullback к MA50')

            # 4. RSI не в перекупленности
            apply_rsi = rsi_enabled and (not rsi_trend_confirmation_only or structure_info['phase'] == 'trend')
            if not apply_rsi:
                conditions_met.append('rsi_ok')
            elif current_rsi < model.rsi_overbought:
                conditions_met.append('rsi_ok')
            else:
                warnings.append(
                    f"RSI перекуплен: {current_rsi:.2f} >= {float(model.rsi_overbought):.2f}"
                )
                reject_reasons.append('rsi_overbought')

            # 5. Объем
            if disable_volume or volume_info['volume_ratio'] >= required_volume_ratio:
                conditions_met.append('volume_ok')
                confidence = 'high' if (not disable_volume and volume_info['is_impulse']) else 'medium'
            else:
                reject_reasons.append('volume_below_threshold')

            # 6. Фильтр перегрева
            if abs(dist_ma50) <= model.max_distance_ma50:
                conditions_met.append('not_overheated')
            else:
                warnings.append(f'Перегрев: {dist_ma50:.1f}%')
                reject_reasons.append('distance_to_ma50_too_large')

            if len(conditions_met) >= 4:
                signal = 'long'
            else:
                reject_reasons.append(f'conditions_long_trend={len(conditions_met)}/4')

    if signal == 'none':
        no_setup_warnings = ['Условия для входа не выполнены']
        if reject_reasons:
            # Убираем дубликаты, сохраняя порядок.
            unique_reasons = list(dict.fromkeys(reject_reasons))
            no_setup_warnings.append("Причины: " + ", ".join(unique_reasons[:4]))
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
            warnings=no_setup_warnings,
            model_name=model.name,
            debug_potential_setup=potential_setup if debug_filters else False,
            debug_filter_stage='no_setup'
        )

    # ATR фильтр
    if current_atr <= 0:
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
            warnings=['Некорректный ATR для расчета стопа'],
            model_name=model.name,
            debug_potential_setup=potential_setup if debug_filters else False,
            debug_filter_stage='filtered_atr'
        )

    if atr_enabled and atr_min_percentile > 0:
        atr_pct_series = (df['atr'] / df['close']) * 100
        atr_percentile = atr_pct_series.rank(pct=True).iloc[-1] * 100 if len(atr_pct_series) > 0 else 0
        if pd.isna(atr_percentile):
            atr_percentile = 0
        if atr_percentile < atr_min_percentile:
            return _create_no_signal_with_data(
                deposit,
                f"ATR percentile {atr_percentile:.1f} < min {atr_min_percentile}",
                current_price,
                current_atr,
                current_rsi,
                dist_ma50,
                dist_ma200,
                volume_info['volume_ratio'],
                structure_info,
                model.name,
                debug_potential_setup=potential_setup if debug_filters else False,
                debug_filter_stage='filtered_atr'
            )

    # Расчет стопа и тейка
    entry = current_price
    stop = calculate_stop_by_atr(entry, current_atr, model.atr_multiplier_stop, direction)
    effective_min_rr = _resolve_effective_min_rr(filter_config=filter_config, model=model)
    target = calculate_target_by_rr(entry, stop, effective_min_rr, direction)

    if entry_context == 'range' and range_context['is_valid']:
        range_low = range_context['range_low']
        range_high = range_context['range_high']
        range_mid = range_context['range_mid']

        if direction == 'long':
            stop = min(stop, range_low - current_atr * 0.5)
            target_mid = range_mid
            rr_mid = _rr_for_levels(direction, entry, stop, target_mid)
            if rr_mid >= effective_min_rr:
                target = target_mid
        else:
            stop = max(stop, range_high + current_atr * 0.5)
            target_mid = range_mid
            rr_mid = _rr_for_levels(direction, entry, stop, target_mid)
            if rr_mid >= effective_min_rr:
                target = target_mid

    min_stop_distance_pct = _resolve_min_stop_distance_pct(filter_config, model.name)
    stop, target, stop_distance_warning = _enforce_min_stop_distance(
        entry=entry,
        stop=stop,
        target=target,
        direction=direction,
        min_stop_distance_pct=min_stop_distance_pct,
        min_rr=effective_min_rr,
        disable_rr=disable_rr,
    )
    if stop_distance_warning:
        warnings.append(stop_distance_warning)

    # Расчет риска
    max_position_notional_pct, position_step, futures_margin_safety_factor = _resolve_portfolio_sizing(filter_config)
    risk_params = calculate_position_risk(
        entry,
        stop,
        target,
        deposit,
        model.max_risk_percent,
        0 if disable_rr else effective_min_rr,
        max_position_notional_pct=max_position_notional_pct,
        position_step=position_step,
        contract_margin_rub=contract_margin_rub,
        contract_multiplier=contract_multiplier,
        futures_margin_safety_factor=futures_margin_safety_factor,
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
            model_name=model.name,
            debug_potential_setup=potential_setup if debug_filters else False,
            debug_filter_stage=_detect_risk_filter_stage(risk_params.reason, disable_rr)
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
            model_name=model.name,
            debug_potential_setup=potential_setup if debug_filters else False,
            debug_filter_stage='no_setup'
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
        model_name=model.name,
        debug_potential_setup=potential_setup if debug_filters else False,
        debug_filter_stage='final_trade'
    )


def _generate_signal_conservative_v2(
    df: pd.DataFrame,
    deposit: float,
    model: TradingModel,
    current_price: float,
    current_atr: float,
    current_rsi: float,
    ma50: Optional[float],
    ma200: Optional[float],
    structure_info: dict,
    volume_info: dict,
    required_volume_ratio: float,
    dist_ma50: float,
    dist_ma200: float,
    range_context: dict,
    disable_rr: bool,
    disable_volume: bool,
    disable_trend: bool,
    debug_filters: bool,
    rsi_enabled: bool,
    rsi_trend_confirmation_only: bool,
    atr_enabled: bool,
    atr_min_percentile: int,
    filter_config: Optional[Dict[str, Any]],
    contract_margin_rub: Optional[float],
    contract_multiplier: float,
    potential_setup: bool,
    soft_threshold_relaxation: float,
    min_expected_trades_per_month: float,
) -> TradingSignal:
    warnings: List[str] = []
    direction: Optional[str] = None
    entry_context = "trend"

    if structure_info['structure'] == 'downtrend' and ma50 is not None and ma200 is not None:
        direction = 'short'
    elif structure_info['structure'] == 'uptrend' and ma50 is not None and ma200 is not None:
        direction = 'long'
    elif model.allow_range and structure_info['structure'] == 'range' and range_context['is_valid']:
        direction = range_context['direction']
        entry_context = 'range'
        warnings.append(range_context['reason'])

    if direction is None:
        no_setup_reason = "Условия для входа не выполнены"
        if model.allow_range and structure_info['structure'] == 'range' and not range_context['is_valid']:
            no_setup_reason = range_context['reason']
        return _create_no_signal_with_data(
            deposit=deposit,
            reason=no_setup_reason,
            price=current_price,
            atr=current_atr,
            rsi=current_rsi,
            dist_ma50=dist_ma50,
            dist_ma200=dist_ma200,
            volume_ratio=volume_info['volume_ratio'],
            structure_info=structure_info,
            model_name=model.name,
            debug_potential_setup=potential_setup if debug_filters else False,
            debug_filter_stage='no_setup',
        )

    trend_strength = float(structure_info.get('trend_strength', 0.0))
    if (
        not disable_trend
        and entry_context == 'trend'
        and trend_strength + 1e-9 < float(model.min_trend_strength)
    ):
        return _create_no_signal_with_data(
            deposit=deposit,
            reason=f"Слабый тренд: {trend_strength:.3f}% < {float(model.min_trend_strength):.3f}%",
            price=current_price,
            atr=current_atr,
            rsi=current_rsi,
            dist_ma50=dist_ma50,
            dist_ma200=dist_ma200,
            volume_ratio=volume_info['volume_ratio'],
            structure_info=structure_info,
            model_name=model.name,
            debug_potential_setup=potential_setup if debug_filters else False,
            debug_filter_stage='filtered_trend',
        )

    # Жесткая проверка качества тренда по MA-выравниванию.
    if entry_context == 'trend' and ma50 is not None and ma200 is not None:
        if direction == 'long' and not (ma50 > ma200):
            return _create_no_signal_with_data(
                deposit=deposit,
                reason="MA alignment failed: требуется MA50 > MA200 для LONG",
                price=current_price,
                atr=current_atr,
                rsi=current_rsi,
                dist_ma50=dist_ma50,
                dist_ma200=dist_ma200,
                volume_ratio=volume_info['volume_ratio'],
                structure_info=structure_info,
                model_name=model.name,
                debug_potential_setup=potential_setup if debug_filters else False,
                debug_filter_stage='filtered_trend',
            )
        if direction == 'short' and not (ma50 < ma200):
            return _create_no_signal_with_data(
                deposit=deposit,
                reason="MA alignment failed: требуется MA50 < MA200 для SHORT",
                price=current_price,
                atr=current_atr,
                rsi=current_rsi,
                dist_ma50=dist_ma50,
                dist_ma200=dist_ma200,
                volume_ratio=volume_info['volume_ratio'],
                structure_info=structure_info,
                model_name=model.name,
                debug_potential_setup=potential_setup if debug_filters else False,
                debug_filter_stage='filtered_trend',
            )

    pullback_enabled, require_pullback, max_pullback_distance_pct = _resolve_pullback_constraints(
        filter_config=filter_config,
        model=model,
    )
    pullback_reject = _check_pullback_entry(
        direction=direction,
        entry_context=entry_context,
        current_price=float(current_price),
        ma50=ma50,
        pullback_enabled=pullback_enabled,
        require_pullback=require_pullback,
        max_pullback_distance_pct=max_pullback_distance_pct,
    )
    if pullback_reject:
        return _create_no_signal_with_data(
            deposit=deposit,
            reason=pullback_reject,
            price=current_price,
            atr=current_atr,
            rsi=current_rsi,
            dist_ma50=dist_ma50,
            dist_ma200=dist_ma200,
            volume_ratio=volume_info['volume_ratio'],
            structure_info=structure_info,
            model_name=model.name,
            debug_potential_setup=potential_setup if debug_filters else False,
            debug_filter_stage='filtered_trend',
        )

    if current_atr <= 0:
        return _create_no_signal_with_data(
            deposit=deposit,
            reason="Некорректный ATR для расчета стопа",
            price=current_price,
            atr=current_atr,
            rsi=current_rsi,
            dist_ma50=dist_ma50,
            dist_ma200=dist_ma200,
            volume_ratio=volume_info['volume_ratio'],
            structure_info=structure_info,
            model_name=model.name,
            debug_potential_setup=potential_setup if debug_filters else False,
            debug_filter_stage='filtered_atr',
        )

    atr_percentile = _calc_atr_percentile(df)
    if atr_enabled and atr_min_percentile > 0 and atr_percentile < atr_min_percentile:
        return _create_no_signal_with_data(
            deposit=deposit,
            reason=f"ATR percentile {atr_percentile:.1f} < min {atr_min_percentile}",
            price=current_price,
            atr=current_atr,
            rsi=current_rsi,
            dist_ma50=dist_ma50,
            dist_ma200=dist_ma200,
            volume_ratio=volume_info['volume_ratio'],
            structure_info=structure_info,
            model_name=model.name,
            debug_potential_setup=potential_setup if debug_filters else False,
            debug_filter_stage='filtered_atr',
        )

    entry = current_price
    stop = calculate_stop_by_atr(entry, current_atr, model.atr_multiplier_stop, direction)
    effective_min_rr = _resolve_effective_min_rr(filter_config=filter_config, model=model)
    target = calculate_target_by_rr(entry, stop, effective_min_rr, direction)

    if entry_context == 'range' and range_context['is_valid']:
        range_low = range_context['range_low']
        range_high = range_context['range_high']
        range_mid = range_context['range_mid']

        if direction == 'long':
            stop = min(stop, range_low - current_atr * 0.5)
            target_mid = range_mid
            rr_mid = _rr_for_levels(direction, entry, stop, target_mid)
            if rr_mid >= effective_min_rr:
                target = target_mid
        else:
            stop = max(stop, range_high + current_atr * 0.5)
            target_mid = range_mid
            rr_mid = _rr_for_levels(direction, entry, stop, target_mid)
            if rr_mid >= effective_min_rr:
                target = target_mid

    min_stop_distance_pct = _resolve_min_stop_distance_pct(filter_config, model.name)
    stop, target, stop_distance_warning = _enforce_min_stop_distance(
        entry=entry,
        stop=stop,
        target=target,
        direction=direction,
        min_stop_distance_pct=min_stop_distance_pct,
        min_rr=effective_min_rr,
        disable_rr=disable_rr,
    )
    if stop_distance_warning:
        warnings.append(stop_distance_warning)

    max_position_notional_pct, position_step, futures_margin_safety_factor = _resolve_portfolio_sizing(filter_config)
    risk_params = calculate_position_risk(
        entry,
        stop,
        target,
        deposit,
        model.max_risk_percent,
        0 if disable_rr else effective_min_rr,
        max_position_notional_pct=max_position_notional_pct,
        position_step=position_step,
        contract_margin_rub=contract_margin_rub,
        contract_multiplier=contract_multiplier,
        futures_margin_safety_factor=futures_margin_safety_factor,
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
            model_name=model.name,
            debug_potential_setup=potential_setup if debug_filters else False,
            debug_filter_stage=_detect_risk_filter_stage(risk_params.reason, disable_rr),
        )

    score_engine = ConservativeModelV2(model=model, config=load_conservative_v2_config(filter_config))
    score_result = score_engine.score(
        df=df,
        direction=direction,
        structure=structure_info['structure'],
        phase=structure_info['phase'],
        trend_strength_pct=float(structure_info.get('trend_strength', 0.0)),
        current_price=current_price,
        ma50=ma50,
        ma200=ma200,
        atr=current_atr,
        rsi=current_rsi,
        volume_ratio=float(volume_info.get('volume_ratio', 0.0)),
        required_volume_ratio=float(required_volume_ratio),
        volume_is_impulse=bool(volume_info.get('is_impulse', False)),
        dist_ma50_pct=float(dist_ma50),
        max_distance_ma50_pct=float(model.max_distance_ma50),
        atr_percentile=float(atr_percentile),
        disable_volume=disable_volume,
        disable_trend=disable_trend,
        rsi_enabled=rsi_enabled,
        rsi_trend_confirmation_only=rsi_trend_confirmation_only,
        allow_range=bool(model.allow_range),
        soft_threshold_relaxation=max(0.0, float(soft_threshold_relaxation)),
    )
    warnings.extend(score_result.notes)

    if soft_threshold_relaxation > 0:
        warnings.append(f"Soft filters auto-relaxed by {soft_threshold_relaxation:.2f} for trade frequency")
    if min_expected_trades_per_month > 0:
        warnings.append(f"Trade frequency target: {min_expected_trades_per_month:.1f}/month")

    if score_result.total_score < score_result.threshold:
        warnings.append(
            f"Soft score {score_result.total_score:.3f} < threshold {score_result.threshold:.3f}"
        )
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
            confidence=score_result.confidence,
            warnings=warnings,
            model_name=model.name,
            debug_potential_setup=potential_setup if debug_filters else False,
            debug_filter_stage=score_result.rejection_stage,
        )

    return TradingSignal(
        signal=direction,
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
        confidence=score_result.confidence,
        warnings=warnings if warnings else ['ConservativeModelV2: сигнал валиден'],
        model_name=model.name,
        debug_potential_setup=potential_setup if debug_filters else False,
        debug_filter_stage='final_trade',
    )


def _resolve_required_volume_ratio(model: TradingModel, volume_info: dict, volume_mode: str) -> float:
    """Определение минимального порога volume ratio для фильтра."""
    if volume_mode == 'adaptive':
        required = volume_info.get('threshold_ratio', 0)
        if required <= 0:
            required = model.min_volume_ratio
    else:
        required = model.min_volume_ratio

    if model.require_impulse:
        required = max(required, 1.5)

    return required


def _detect_risk_filter_stage(reason: str, disable_rr: bool) -> str:
    """Классификация причины отклонения риск-блока для debug-статистики."""
    reason_lower = reason.lower()

    if 'стоп' in reason_lower:
        return 'filtered_atr'

    if not disable_rr and ('rr ' in reason_lower or reason.startswith('RR')):
        return 'filtered_rr'

    # Любой риск-фейл вне RR считаем RR-группой для упрощенной диагностики
    return 'filtered_rr'


def _resolve_min_stop_distance_pct(filter_config: Optional[Dict[str, Any]], model_name: str) -> float:
    """
    Минимальная дистанция стопа (% от цены входа).
    По умолчанию отключена для scalp и включена для остальных моделей.
    """
    default_value = 0.0 if model_name == "scalp" else 0.7
    if not filter_config:
        return default_value

    raw = (
        filter_config.get("filters", {})
        .get("risk", {})
        .get("min_stop_distance_pct", default_value)
    )
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default_value
    return max(0.0, value)


def _resolve_portfolio_sizing(filter_config: Optional[Dict[str, Any]]) -> tuple[float, float, float]:
    """
    Параметры ограничения позиции по портфелю.
    """
    default_notional_pct = 80.0
    default_step = 1.0
    default_futures_margin_safety = 1.4
    if not filter_config:
        return default_notional_pct, default_step, default_futures_margin_safety

    risk_cfg = filter_config.get("filters", {}).get("risk", {})
    raw_notional_pct = risk_cfg.get("max_position_notional_pct", default_notional_pct)
    raw_step = risk_cfg.get("position_step", default_step)
    raw_futures_margin_safety = risk_cfg.get("futures_margin_safety_factor", default_futures_margin_safety)

    try:
        notional_pct = float(raw_notional_pct)
    except (TypeError, ValueError):
        notional_pct = default_notional_pct
    try:
        step = float(raw_step)
    except (TypeError, ValueError):
        step = default_step
    try:
        futures_margin_safety = float(raw_futures_margin_safety)
    except (TypeError, ValueError):
        futures_margin_safety = default_futures_margin_safety

    return max(0.0, notional_pct), max(0.0, step), max(0.0, futures_margin_safety)


def _enforce_min_stop_distance(
    *,
    entry: float,
    stop: float,
    target: float,
    direction: str,
    min_stop_distance_pct: float,
    min_rr: float,
    disable_rr: bool,
) -> tuple[float, float, str | None]:
    if min_stop_distance_pct <= 0 or entry <= 0:
        return stop, target, None

    min_distance_abs = entry * (min_stop_distance_pct / 100.0)
    current_distance = abs(entry - stop)
    if current_distance >= min_distance_abs:
        return stop, target, None

    adjusted_stop = entry - min_distance_abs if direction == "long" else entry + min_distance_abs
    adjusted_target = target

    if not disable_rr:
        adjusted_target = calculate_target_by_rr(entry, adjusted_stop, min_rr, direction)

    warning = (
        f"Стоп расширен до min_stop_distance_pct={min_stop_distance_pct:.2f}% "
        f"(~{min_distance_abs:.2f} по цене)"
    )
    return adjusted_stop, adjusted_target, warning


def _create_no_signal(
    deposit: float,
    reason: str,
    model_name: str,
    debug_potential_setup: bool = False,
    debug_filter_stage: str = 'none'
) -> TradingSignal:
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
        model_name=model_name,
        debug_potential_setup=debug_potential_setup,
        debug_filter_stage=debug_filter_stage
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
    model_name: str,
    debug_potential_setup: bool = False,
    debug_filter_stage: str = 'none'
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
        model_name=model_name,
        debug_potential_setup=debug_potential_setup,
        debug_filter_stage=debug_filter_stage
    )


def _resolve_effective_min_rr(filter_config: Optional[Dict[str, Any]], model: TradingModel) -> float:
    cfg_rr = (
        (filter_config or {})
        .get("filters", {})
        .get("rr", {})
        .get("min_rr", 2.5)
    )
    try:
        cfg_rr_value = float(cfg_rr)
    except (TypeError, ValueError):
        cfg_rr_value = 2.5
    return max(float(model.min_rr), cfg_rr_value, 2.5)


def _resolve_pullback_constraints(
    *,
    filter_config: Optional[Dict[str, Any]],
    model: TradingModel,
) -> tuple[bool, bool, float]:
    pullback_cfg = (
        (filter_config or {})
        .get("filters", {})
        .get("pullback", {})
    )
    enabled = bool(pullback_cfg.get("enabled", True))
    require_pullback = bool(pullback_cfg.get("require_pullback", True))
    raw_max = pullback_cfg.get("max_ma50_distance_pct", 1.5)
    try:
        max_distance = float(raw_max)
    except (TypeError, ValueError):
        max_distance = 1.5
    max_distance = max(0.0, max_distance)
    if max_distance <= 0:
        max_distance = max(0.1, float(model.max_distance_ma50))
    return enabled, require_pullback, max_distance


def _check_pullback_entry(
    *,
    direction: str,
    entry_context: str,
    current_price: float,
    ma50: Optional[float],
    pullback_enabled: bool,
    require_pullback: bool,
    max_pullback_distance_pct: float,
) -> Optional[str]:
    if not pullback_enabled or entry_context != "trend":
        return None
    if ma50 is None or ma50 <= 0:
        return "Pullback filter: MA50 недоступна"

    distance_pct = abs((current_price - ma50) / ma50) * 100.0
    if distance_pct > max_pullback_distance_pct:
        return (
            f"Цена слишком далеко от MA50: {distance_pct:.2f}% > "
            f"{max_pullback_distance_pct:.2f}%"
        )

    if not require_pullback:
        return None

    upper_bound = ma50 * (1.0 + max_pullback_distance_pct / 100.0)
    lower_bound = ma50 * (1.0 - max_pullback_distance_pct / 100.0)
    if direction == "long":
        if current_price < ma50 or current_price > upper_bound:
            return (
                "Нет валидного pullback LONG: "
                f"ожидается MA50 <= price <= {upper_bound:.4f}"
            )
    elif direction == "short":
        if current_price > ma50 or current_price < lower_bound:
            return (
                "Нет валидного pullback SHORT: "
                f"ожидается {lower_bound:.4f} <= price <= MA50"
            )
    return None


def _build_range_context(
    df: pd.DataFrame,
    current_price: float,
    current_atr: float,
    current_rsi: float,
    window: int = 40
) -> dict:
    if len(df) < max(window, 2):
        return {
            'is_valid': False,
            'direction': None,
            'reason': 'Недостаточно данных для range-контекста',
            'range_low': current_price,
            'range_high': current_price,
            'range_mid': current_price,
        }

    tail = df.tail(window)
    range_high = float(tail['high'].max())
    range_low = float(tail['low'].min())
    range_mid = (range_high + range_low) / 2.0
    if range_high <= range_low:
        return {
            'is_valid': False,
            'direction': None,
            'reason': 'Некорректный диапазон range',
            'range_low': range_low,
            'range_high': range_high,
            'range_mid': range_mid,
        }

    prev_close = float(df['close'].iloc[-2])
    in_channel = (range_low - current_atr) <= current_price <= (range_high + current_atr)
    near_low = current_price <= (range_low + current_atr)
    near_high = current_price >= (range_high - current_atr)
    rsi_ok = 30 <= current_rsi <= 70

    if in_channel and rsi_ok and near_low and current_price > prev_close:
        return {
            'is_valid': True,
            'direction': 'long',
            'reason': 'Range bounce: отскок от нижней границы',
            'range_low': range_low,
            'range_high': range_high,
            'range_mid': range_mid,
        }

    if in_channel and rsi_ok and near_high and current_price < prev_close:
        return {
            'is_valid': True,
            'direction': 'short',
            'reason': 'Range bounce: отскок от верхней границы',
            'range_low': range_low,
            'range_high': range_high,
            'range_mid': range_mid,
        }

    reason = 'Range setup не подтвержден'
    if not in_channel:
        reason = 'Цена вне канала ±1 ATR'
    elif not rsi_ok:
        reason = f'RSI вне диапазона 30-70: {current_rsi:.1f}'
    elif not (near_low or near_high):
        reason = 'Цена не у границы диапазона'
    else:
        reason = 'Нет подтверждения отскока'

    return {
        'is_valid': False,
        'direction': None,
        'reason': reason,
        'range_low': range_low,
        'range_high': range_high,
        'range_mid': range_mid,
    }


def _calc_atr_percentile(df: pd.DataFrame) -> float:
    if 'atr' not in df.columns or 'close' not in df.columns or len(df) == 0:
        return 0.0
    atr_pct_series = (df['atr'] / df['close']) * 100
    atr_pct_series = atr_pct_series.replace([float('inf'), float('-inf')], pd.NA).dropna()
    if len(atr_pct_series) == 0:
        return 0.0
    atr_percentile = atr_pct_series.rank(pct=True).iloc[-1] * 100
    if pd.isna(atr_percentile):
        return 0.0
    return float(atr_percentile)


def _rr_for_levels(direction: str, entry: float, stop: float, target: float) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    if direction == 'long':
        reward = target - entry
    else:
        reward = entry - target
    return reward / risk
