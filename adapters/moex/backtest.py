"""
Модуль бэктестинга торговых стратегий
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from .models import TradingModel


@dataclass
class Trade:
    """Информация о сделке"""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    direction: str
    position_size: float
    pnl: float
    pnl_percent: float
    exit_reason: str
    rr_planned: float
    rr_actual: float
    duration_candles: int


@dataclass
class BacktestResults:
    """Результаты бэктеста"""
    model_name: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    winrate: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    expectancy: float
    max_drawdown: float
    max_drawdown_percent: float
    total_profit: float
    final_balance: float
    profit_factor: float
    return_pct: float
    sharpe_ratio: float
    avg_trade_duration: int
    trades: List[Trade]

    def to_dict(self) -> Dict:
        """Конвертация в словарь"""
        return {
            'model_name': self.model_name,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'winrate': round(self.winrate, 2),
            'avg_win': round(self.avg_win, 2),
            'avg_loss': round(self.avg_loss, 2),
            'best_trade': round(self.best_trade, 2),
            'worst_trade': round(self.worst_trade, 2),
            'expectancy': round(self.expectancy, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'max_drawdown_percent': round(self.max_drawdown_percent, 2),
            'total_profit': round(self.total_profit, 2),
            'final_balance': round(self.final_balance, 2),
            'profit_factor': round(self.profit_factor, 2),
            'return_pct': round(self.return_pct, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'avg_trade_duration': self.avg_trade_duration
        }


def run_backtest(
    df: pd.DataFrame,
    signal_generator,
    deposit: float,
    model: TradingModel,
    lookback_window: int = 300,
    max_holding_candles: int = 50
) -> BacktestResults:
    """
    Запуск бэктеста стратегии

    Args:
        df: DataFrame с данными
        signal_generator: Функция генерации сигналов
        deposit: Начальный депозит
        model: Торговая модель
        lookback_window: Минимальное окно данных для сигнала
        max_holding_candles: Максимальное удержание позиции в свечах

    Returns:
        BacktestResults с результатами
    """
    if len(df) < lookback_window + 50:
        return _create_empty_results(model.name, deposit)

    trades = []
    balance = deposit
    peak_balance = deposit
    max_drawdown = 0
    balances = [deposit]

    # Проходим по данным
    for i in range(lookback_window, len(df) - max_holding_candles):
        df_window = df.iloc[:i+1].copy()

        # Генерируем сигнал
        signal = signal_generator(df_window, balance, model)

        if signal.signal == 'none':
            balances.append(balance)
            continue

        # Симулируем сделку
        entry_time = df.index[i]
        entry_price = signal.entry
        stop_price = signal.stop
        target_price = signal.target
        position_size = signal.position_size
        direction = signal.signal

        # Ищем выход
        exit_price = None
        exit_reason = None
        exit_time = None
        exit_candle_idx = None

        for j in range(i + 1, min(i + max_holding_candles + 1, len(df))):
            candle_high = df['high'].iloc[j]
            candle_low = df['low'].iloc[j]

            if direction == 'long':
                # Проверяем стоп
                if candle_low <= stop_price:
                    exit_price = stop_price
                    exit_reason = 'stop'
                    exit_time = df.index[j]
                    exit_candle_idx = j
                    break
                # Проверяем тейк
                if candle_high >= target_price:
                    exit_price = target_price
                    exit_reason = 'target'
                    exit_time = df.index[j]
                    exit_candle_idx = j
                    break

            elif direction == 'short':
                # Проверяем стоп
                if candle_high >= stop_price:
                    exit_price = stop_price
                    exit_reason = 'stop'
                    exit_time = df.index[j]
                    exit_candle_idx = j
                    break
                # Проверяем тейк
                if candle_low <= target_price:
                    exit_price = target_price
                    exit_reason = 'target'
                    exit_time = df.index[j]
                    exit_candle_idx = j
                    break

        # Если не вышли - закрываем по рынку
        if exit_price is None:
            exit_candle_idx = min(i + max_holding_candles, len(df) - 1)
            exit_price = df['close'].iloc[exit_candle_idx]
            exit_reason = 'timeout'
            exit_time = df.index[exit_candle_idx]

        # Расчет P&L
        if direction == 'long':
            pnl = (exit_price - entry_price) * position_size
            rr_actual = (exit_price - entry_price) / (entry_price - stop_price)
        else:
            pnl = (entry_price - exit_price) * position_size
            rr_actual = (entry_price - exit_price) / (stop_price - entry_price)

        pnl_percent = (pnl / balance) * 100
        balance += pnl

        # Обновляем максимальный баланс и просадку
        if balance > peak_balance:
            peak_balance = balance

        current_drawdown = peak_balance - balance
        current_drawdown_percent = (current_drawdown / peak_balance) * 100

        if current_drawdown_percent > max_drawdown:
            max_drawdown = current_drawdown_percent

        balances.append(balance)

        # Сохраняем сделку
        trade = Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_price=stop_price,
            target_price=target_price,
            direction=direction,
            position_size=position_size,
            pnl=pnl,
            pnl_percent=pnl_percent,
            exit_reason=exit_reason,
            rr_planned=signal.rr,
            rr_actual=rr_actual,
            duration_candles=exit_candle_idx - i
        )
        trades.append(trade)

    # Расчет статистики
    return _calculate_statistics(trades, deposit, balance, max_drawdown, balances, model.name)


def _calculate_statistics(
    trades: List[Trade],
    initial_deposit: float,
    final_balance: float,
    max_drawdown: float,
    balances: List[float],
    model_name: str
) -> BacktestResults:
    """Расчет статистики по сделкам"""

    if not trades:
        return _create_empty_results(model_name, initial_deposit)

    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl <= 0]

    total_trades = len(trades)
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    winrate = (win_count / total_trades * 100) if total_trades > 0 else 0

    avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
    avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0
    best_trade = max([t.pnl for t in trades]) if trades else 0
    worst_trade = min([t.pnl for t in trades]) if trades else 0

    total_profit = sum([t.pnl for t in trades])

    # Expectancy
    expectancy = (winrate / 100 * avg_win) - ((100 - winrate) / 100 * avg_loss)

    # Profit factor
    gross_profit = sum([t.pnl for t in winning_trades]) if winning_trades else 0
    gross_loss = abs(sum([t.pnl for t in losing_trades])) if losing_trades else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Sharpe ratio (упрощенный)
    returns = [t.pnl_percent for t in trades]
    sharpe_ratio = (np.mean(returns) / np.std(returns)) if len(returns) > 1 and np.std(returns) > 0 else 0

    # Средняя длительность сделки
    avg_duration = int(np.mean([t.duration_candles for t in trades])) if trades else 0

    return BacktestResults(
        model_name=model_name,
        total_trades=total_trades,
        winning_trades=win_count,
        losing_trades=loss_count,
        winrate=winrate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        best_trade=best_trade,
        worst_trade=worst_trade,
        expectancy=expectancy,
        max_drawdown=max_drawdown,
        max_drawdown_percent=max_drawdown,
        total_profit=total_profit,
        final_balance=final_balance,
        profit_factor=profit_factor,
        return_pct=(final_balance - initial_deposit) / initial_deposit * 100,
        sharpe_ratio=sharpe_ratio * np.sqrt(252),  # Annualized
        avg_trade_duration=avg_duration,
        trades=trades
    )


def _create_empty_results(model_name: str, deposit: float) -> BacktestResults:
    """Создание пустых результатов"""
    return BacktestResults(
        model_name=model_name,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        winrate=0,
        avg_win=0,
        avg_loss=0,
        best_trade=0,
        worst_trade=0,
        expectancy=0,
        max_drawdown=0,
        max_drawdown_percent=0,
        total_profit=0,
        final_balance=deposit,
        profit_factor=0,
        return_pct=0,
        sharpe_ratio=0,
        avg_trade_duration=0,
        trades=[]
    )


def compare_models_results(results: List[BacktestResults]) -> str:
    """
    Сравнительная таблица результатов моделей

    Args:
        results: Список результатов бэктеста

    Returns:
        Форматированная таблица
    """
    if not results:
        return "Нет результатов для сравнения"

    lines = []
    lines.append("=" * 120)
    lines.append("СРАВНЕНИЕ МОДЕЛЕЙ - РЕЗУЛЬТАТЫ БЭКТЕСТА")
    lines.append("=" * 120)

    # Заголовок
    header = (
        f"{'Model':<15} "
        f"{'Trades':>7} "
        f"{'Win%':>7} "
        f"{'Expect':>8} "
        f"{'PF':>6} "
        f"{'MaxDD%':>8} "
        f"{'Return%':>9} "
        f"{'Sharpe':>7} "
        f"{'AvgDur':>7}"
    )
    lines.append(header)
    lines.append("-" * 120)

    # Данные
    for res in sorted(results, key=lambda x: x.expectancy, reverse=True):
        row = (
            f"{res.model_name:<15} "
            f"{res.total_trades:>7} "
            f"{res.winrate:>7.1f} "
            f"{res.expectancy:>8.2f} "
            f"{res.profit_factor:>6.2f} "
            f"{res.max_drawdown_percent:>8.1f} "
            f"{res.return_pct:>9.1f} "
            f"{res.sharpe_ratio:>7.2f} "
            f"{res.avg_trade_duration:>7}"
        )
        lines.append(row)

    lines.append("=" * 120)

    # Лучшая модель
    best_model = max(results, key=lambda x: x.expectancy)
    lines.append(f"\n🏆 Лучшая модель: {best_model.model_name.upper()}")
    lines.append(f"   Expectancy: {best_model.expectancy:.2f}")
    lines.append(f"   Winrate: {best_model.winrate:.1f}%")
    lines.append(f"   Return: {best_model.return_pct:.1f}%")

    return "\n".join(lines)
