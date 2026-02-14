"""
Профессиональная торговая система с управлением рисками и структурным анализом
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


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
    confidence: str  # 'high', 'medium', 'low'
    warnings: List[str]

    def to_dict(self) -> Dict:
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
            "warnings": self.warnings
        }


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Расчет Average True Range (ATR)

    Args:
        df: DataFrame с OHLC данными
        period: Период ATR

    Returns:
        Series с значениями ATR
    """
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def calculate_structure(df: pd.DataFrame, lookback: int = 20) -> Dict:
    """
    Определение структуры рынка: HH/HL (восходящий тренд) или LH/LL (нисходящий тренд)

    Args:
        df: DataFrame с данными
        lookback: Количество свечей для анализа структуры

    Returns:
        Словарь с информацией о структуре
    """
    if len(df) < lookback * 2:
        return {
            'structure': 'unknown',
            'phase': 'unknown',
            'trend_strength': 0,
            'breakout': False,
            'last_swing_high': None,
            'last_swing_low': None
        }

    # Находим свинг-максимумы и минимумы
    df_tail = df.tail(lookback * 3).copy()

    # Простой алгоритм поиска свингов
    swing_highs = []
    swing_lows = []

    for i in range(lookback, len(df_tail) - lookback):
        # Свинг-максимум
        if df_tail['high'].iloc[i] == df_tail['high'].iloc[i-lookback:i+lookback+1].max():
            swing_highs.append((df_tail.index[i], df_tail['high'].iloc[i]))

        # Свинг-минимум
        if df_tail['low'].iloc[i] == df_tail['low'].iloc[i-lookback:i+lookback+1].min():
            swing_lows.append((df_tail.index[i], df_tail['low'].iloc[i]))

    # Определяем структуру
    structure = 'range'
    trend_strength = 0

    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        # Проверяем последние 2 максимума и 2 минимума
        if swing_highs[-1][1] > swing_highs[-2][1] and swing_lows[-1][1] > swing_lows[-2][1]:
            structure = 'uptrend'  # HH и HL
            trend_strength = (swing_highs[-1][1] - swing_highs[-2][1]) / swing_highs[-2][1] * 100
        elif swing_highs[-1][1] < swing_highs[-2][1] and swing_lows[-1][1] < swing_lows[-2][1]:
            structure = 'downtrend'  # LH и LL
            trend_strength = (swing_highs[-2][1] - swing_highs[-1][1]) / swing_highs[-2][1] * 100

    # Определяем фазу
    current_price = df['close'].iloc[-1]
    ma50 = df['ma50'].iloc[-1] if 'ma50' in df.columns and not pd.isna(df['ma50'].iloc[-1]) else None
    ma200 = df['ma200'].iloc[-1] if 'ma200' in df.columns and not pd.isna(df['ma200'].iloc[-1]) else None

    phase = 'unknown'
    if ma50 is not None and ma200 is not None:
        if structure == 'uptrend':
            if current_price < ma50:
                phase = 'pullback'
            else:
                phase = 'trend'
        elif structure == 'downtrend':
            if current_price > ma50:
                phase = 'pullback'
            else:
                phase = 'trend'
        else:
            phase = 'range'

    # Проверка пробоя структуры
    breakout = False
    if swing_highs and swing_lows:
        last_swing_high = swing_highs[-1][1] if swing_highs else None
        last_swing_low = swing_lows[-1][1] if swing_lows else None

        if structure == 'downtrend' and last_swing_high and current_price > last_swing_high:
            breakout = True
        elif structure == 'uptrend' and last_swing_low and current_price < last_swing_low:
            breakout = True
    else:
        last_swing_high = None
        last_swing_low = None

    return {
        'structure': structure,
        'phase': phase,
        'trend_strength': abs(trend_strength),
        'breakout': breakout,
        'last_swing_high': swing_highs[-1][1] if swing_highs else None,
        'last_swing_low': swing_lows[-1][1] if swing_lows else None,
        'swing_highs_count': len(swing_highs),
        'swing_lows_count': len(swing_lows)
    }


def calculate_distance_to_ma(price: float, ma_value: Optional[float]) -> float:
    """
    Расчет процентного расстояния от цены до MA

    Args:
        price: Текущая цена
        ma_value: Значение MA

    Returns:
        Процентное отклонение (положительное - выше MA, отрицательное - ниже)
    """
    if ma_value is None or pd.isna(ma_value) or ma_value == 0:
        return 0.0

    return ((price - ma_value) / ma_value) * 100


def calculate_volume_stats(df: pd.DataFrame, period: int = 20) -> Dict:
    """
    Анализ объемов

    Args:
        df: DataFrame с данными
        period: Период для расчета среднего объема

    Returns:
        Словарь со статистикой объемов
    """
    if 'volume' not in df.columns or len(df) < period:
        return {
            'avg_volume': 0,
            'current_volume': 0,
            'volume_ratio': 0,
            'is_impulse': False
        }

    avg_volume = df['volume'].tail(period).mean()
    current_volume = df['volume'].iloc[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

    # Импульсная свеча - объем > 1.5x среднего
    is_impulse = volume_ratio > 1.5

    return {
        'avg_volume': avg_volume,
        'current_volume': current_volume,
        'volume_ratio': volume_ratio,
        'is_impulse': is_impulse
    }


def calculate_risk(
    entry: float,
    stop: float,
    target: float,
    deposit: float,
    max_risk_percent: float = 1.5,
    min_rr: float = 2.0
) -> Dict:
    """
    Расчет риск-менеджмента

    Args:
        entry: Цена входа
        stop: Стоп-лосс
        target: Тейк-профит
        deposit: Размер депозита
        max_risk_percent: Максимальный риск в % от депозита
        min_rr: Минимальный RR

    Returns:
        Словарь с параметрами риска
    """
    risk_per_unit = abs(entry - stop)
    profit_per_unit = abs(target - entry)

    if risk_per_unit == 0:
        return {
            'rr': 0,
            'risk_rub': 0,
            'risk_percent': 0,
            'position_size': 0,
            'valid': False,
            'reason': 'Нулевое расстояние до стопа'
        }

    rr = profit_per_unit / risk_per_unit

    # Проверка минимального RR
    if rr < min_rr:
        return {
            'rr': rr,
            'risk_rub': 0,
            'risk_percent': 0,
            'position_size': 0,
            'valid': False,
            'reason': f'RR {rr:.2f} < минимального {min_rr}'
        }

    # Расчет максимального риска
    max_risk_rub = deposit * (max_risk_percent / 100)

    # Расчет размера позиции
    position_size = max_risk_rub / risk_per_unit

    # Фактический риск
    actual_risk_rub = position_size * risk_per_unit
    actual_risk_percent = (actual_risk_rub / deposit) * 100

    return {
        'rr': rr,
        'risk_rub': actual_risk_rub,
        'risk_percent': actual_risk_percent,
        'position_size': position_size,
        'potential_profit': position_size * profit_per_unit,
        'valid': True,
        'reason': 'OK'
    }


def generate_signal(
    df: pd.DataFrame,
    deposit: float,
    max_risk_percent: float = 1.5,
    min_rr: float = 2.0,
    atr_multiplier: float = 2.0,
    impulse_volume_threshold: float = 1.5,
    overheated_threshold: float = 5.0
) -> TradingSignal:
    """
    Генерация торгового сигнала на основе структуры, волатильности и объема

    Args:
        df: DataFrame с данными и индикаторами
        deposit: Размер депозита
        max_risk_percent: Максимальный риск в %
        min_rr: Минимальный RR
        atr_multiplier: Множитель ATR для стопа
        impulse_volume_threshold: Порог для импульсного объема
        overheated_threshold: Порог перегретости в %

    Returns:
        TradingSignal с полной информацией о сигнале
    """
    warnings = []

    # Проверка наличия данных
    if df.empty or len(df) < 200:
        return _create_no_signal(deposit, "Недостаточно данных")

    # Добавляем ATR если его нет
    if 'atr' not in df.columns:
        df['atr'] = calculate_atr(df)

    # Текущие значения
    current_price = df['close'].iloc[-1]
    current_atr = df['atr'].iloc[-1] if not pd.isna(df['atr'].iloc[-1]) else current_price * 0.02
    current_rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[-1]) else 50
    ma50 = df['ma50'].iloc[-1] if 'ma50' in df.columns and not pd.isna(df['ma50'].iloc[-1]) else None
    ma200 = df['ma200'].iloc[-1] if 'ma200' in df.columns and not pd.isna(df['ma200'].iloc[-1]) else None

    # Расчет структуры
    structure_info = calculate_structure(df)

    # Анализ объемов
    volume_info = calculate_volume_stats(df)

    # Расстояние до MA
    dist_ma50 = calculate_distance_to_ma(current_price, ma50)
    dist_ma200 = calculate_distance_to_ma(current_price, ma200)

    # Определение направления сигнала
    signal = 'none'
    entry = current_price
    stop = 0.0
    target = 0.0
    confidence = 'low'

    # === ЛОГИКА СИГНАЛА НА ШОРТ ===
    if (structure_info['structure'] == 'downtrend' and
        structure_info['phase'] == 'pullback' and
        ma50 is not None and ma200 is not None and
        current_price < ma200):

        # Проверка условий для шорта
        conditions_met = []

        # 1. Цена ниже MA50 и MA200
        if current_price < ma50 and current_price < ma200:
            conditions_met.append('price_below_mas')

        # 2. Структура LH/LL
        if structure_info['structure'] == 'downtrend':
            conditions_met.append('downtrend_structure')

        # 3. Откат к MA50
        if abs(dist_ma50) < 3:  # В пределах 3% от MA50
            conditions_met.append('pullback_to_ma50')
            warnings.append('Цена на откате к MA50')

        # 4. RSI не в зоне перепроданности
        if current_rsi > 35:
            conditions_met.append('rsi_valid')
        else:
            warnings.append(f'RSI слишком низкий: {current_rsi:.1f}')

        # 5. Объем
        if volume_info['is_impulse']:
            conditions_met.append('impulse_volume')
            confidence = 'high'
        elif volume_info['volume_ratio'] > 1.0:
            conditions_met.append('good_volume')
            confidence = 'medium'

        # Если выполнено >= 4 условий - формируем сигнал
        if len(conditions_met) >= 4:
            signal = 'short'
            entry = current_price
            stop = entry + (current_atr * atr_multiplier)
            target = entry - (abs(stop - entry) * min_rr)

            # Проверка на перегретость
            if abs(dist_ma50) > overheated_threshold:
                warnings.append(f'Перегрев относительно MA50: {dist_ma50:.1f}%')

    # === ЛОГИКА СИГНАЛА НА ЛОНГ ===
    elif (structure_info['structure'] == 'uptrend' and
          structure_info['phase'] == 'pullback' and
          ma50 is not None and ma200 is not None and
          current_price > ma200):

        conditions_met = []

        # 1. Цена выше MA50 и MA200
        if current_price > ma50 and current_price > ma200:
            conditions_met.append('price_above_mas')

        # 2. Структура HH/HL
        if structure_info['structure'] == 'uptrend':
            conditions_met.append('uptrend_structure')

        # 3. Откат к MA50
        if abs(dist_ma50) < 3:
            conditions_met.append('pullback_to_ma50')
            warnings.append('Цена на откате к MA50')

        # 4. RSI не в зоне перекупленности
        if current_rsi < 65:
            conditions_met.append('rsi_valid')
        else:
            warnings.append(f'RSI слишком высокий: {current_rsi:.1f}')

        # 5. Объем
        if volume_info['is_impulse']:
            conditions_met.append('impulse_volume')
            confidence = 'high'
        elif volume_info['volume_ratio'] > 1.0:
            conditions_met.append('good_volume')
            confidence = 'medium'

        if len(conditions_met) >= 4:
            signal = 'long'
            entry = current_price
            stop = entry - (current_atr * atr_multiplier)
            target = entry + (abs(entry - stop) * min_rr)

            if abs(dist_ma50) > overheated_threshold:
                warnings.append(f'Перегрев относительно MA50: {dist_ma50:.1f}%')

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
            warnings=['Условия для входа не выполнены']
        )

    # Расчет риска
    risk_info = calculate_risk(entry, stop, target, deposit, max_risk_percent, min_rr)

    if not risk_info['valid']:
        warnings.append(risk_info['reason'])
        return TradingSignal(
            signal='none',
            entry=current_price,
            stop=stop,
            target=target,
            rr=risk_info['rr'],
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
            warnings=warnings
        )

    return TradingSignal(
        signal=signal,
        entry=entry,
        stop=stop,
        target=target,
        rr=risk_info['rr'],
        risk_rub=risk_info['risk_rub'],
        risk_percent=risk_info['risk_percent'],
        position_size=risk_info['position_size'],
        structure=structure_info['structure'],
        phase=structure_info['phase'],
        volume_ratio=volume_info['volume_ratio'],
        atr=current_atr,
        distance_ma50_pct=dist_ma50,
        distance_ma200_pct=dist_ma200,
        rsi=current_rsi,
        confidence=confidence,
        warnings=warnings if warnings else ['Сигнал валиден']
    )


def _create_no_signal(deposit: float, reason: str) -> TradingSignal:
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
        warnings=[reason]
    )


def backtest_strategy(
    df: pd.DataFrame,
    deposit: float,
    max_risk_percent: float = 1.5,
    min_rr: float = 2.0,
    lookback_window: int = 300
) -> Dict:
    """
    Бэктест стратегии на исторических данных

    Args:
        df: DataFrame с данными
        deposit: Начальный депозит
        max_risk_percent: Максимальный риск в %
        min_rr: Минимальный RR
        lookback_window: Минимальное окно данных для генерации сигнала

    Returns:
        Словарь с результатами бэктеста
    """
    if len(df) < lookback_window + 50:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'winrate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'expectancy': 0,
            'max_drawdown': 0,
            'total_profit': 0,
            'final_balance': deposit
        }

    # Добавляем ATR
    if 'atr' not in df.columns:
        df['atr'] = calculate_atr(df)

    trades = []
    balance = deposit
    peak_balance = deposit
    max_drawdown = 0

    # Проходим по данным с окном
    for i in range(lookback_window, len(df) - 10):  # -10 чтобы было место для выхода
        df_window = df.iloc[:i+1].copy()

        # Генерируем сигнал
        signal = generate_signal(df_window, balance, max_risk_percent, min_rr)

        if signal.signal == 'none':
            continue

        # Симулируем сделку
        entry_price = signal.entry
        stop_price = signal.stop
        target_price = signal.target
        position_size = signal.position_size

        # Ищем выход в следующих 10 свечах
        exit_price = None
        exit_reason = None

        for j in range(i + 1, min(i + 11, len(df))):
            candle_high = df['high'].iloc[j]
            candle_low = df['low'].iloc[j]

            if signal.signal == 'long':
                # Проверяем стоп
                if candle_low <= stop_price:
                    exit_price = stop_price
                    exit_reason = 'stop'
                    break
                # Проверяем тейк
                if candle_high >= target_price:
                    exit_price = target_price
                    exit_reason = 'target'
                    break

            elif signal.signal == 'short':
                # Проверяем стоп
                if candle_high >= stop_price:
                    exit_price = stop_price
                    exit_reason = 'stop'
                    break
                # Проверяем тейк
                if candle_low <= target_price:
                    exit_price = target_price
                    exit_reason = 'target'
                    break

        # Если не вышли, закрываем по рынку
        if exit_price is None:
            exit_price = df['close'].iloc[min(i + 10, len(df) - 1)]
            exit_reason = 'timeout'

        # Расчет P&L
        if signal.signal == 'long':
            pnl = (exit_price - entry_price) * position_size
        else:
            pnl = (entry_price - exit_price) * position_size

        balance += pnl

        # Обновляем максимальный баланс и просадку
        if balance > peak_balance:
            peak_balance = balance

        current_drawdown = (peak_balance - balance) / peak_balance * 100
        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown

        trades.append({
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl': pnl,
            'signal': signal.signal,
            'rr_planned': signal.rr
        })

    # Статистика
    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'winrate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'expectancy': 0,
            'max_drawdown': 0,
            'total_profit': 0,
            'final_balance': balance,
            'profit_factor': 0
        }

    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] <= 0]

    total_trades = len(trades)
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    winrate = (win_count / total_trades * 100) if total_trades > 0 else 0

    avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 0

    total_profit = sum([t['pnl'] for t in trades])

    # Expectancy
    expectancy = (winrate / 100 * avg_win) - ((100 - winrate) / 100 * avg_loss)

    # Profit factor
    gross_profit = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
    gross_loss = abs(sum([t['pnl'] for t in losing_trades])) if losing_trades else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    return {
        'total_trades': total_trades,
        'winning_trades': win_count,
        'losing_trades': loss_count,
        'winrate': round(winrate, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'expectancy': round(expectancy, 2),
        'max_drawdown': round(max_drawdown, 2),
        'total_profit': round(total_profit, 2),
        'final_balance': round(balance, 2),
        'profit_factor': round(profit_factor, 2),
        'return_pct': round((balance - deposit) / deposit * 100, 2)
    }
