"""
Модуль бэктестинга торговых стратегий
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
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
    regime: Optional[str] = None
    gross_pnl: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    entry_turnover: float = 0.0
    exit_turnover: float = 0.0


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
    filter_stats: Optional['FilterStats'] = None
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    best_trade_r: float = 0.0
    worst_trade_r: float = 0.0
    expectancy_r: float = 0.0
    avg_risk_per_trade: float = 0.0
    gross_total_profit: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_costs: float = 0.0
    gross_winning_trades: int = 0
    gross_losing_trades: int = 0
    gross_winrate: float = 0.0
    gross_profit_factor: float = 0.0

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
            'avg_trade_duration': self.avg_trade_duration,
            'filter_stats': self.filter_stats.to_dict() if self.filter_stats else None,
            'avg_win_r': round(self.avg_win_r, 2),
            'avg_loss_r': round(self.avg_loss_r, 2),
            'best_trade_r': round(self.best_trade_r, 2),
            'worst_trade_r': round(self.worst_trade_r, 2),
            'expectancy_r': round(self.expectancy_r, 2),
            'avg_risk_per_trade': round(self.avg_risk_per_trade, 2),
            'gross_total_profit': round(self.gross_total_profit, 2),
            'total_fees': round(self.total_fees, 2),
            'total_slippage': round(self.total_slippage, 2),
            'total_costs': round(self.total_costs, 2),
            'gross_winning_trades': self.gross_winning_trades,
            'gross_losing_trades': self.gross_losing_trades,
            'gross_winrate': round(self.gross_winrate, 2),
            'gross_profit_factor': round(self.gross_profit_factor, 2),
        }


@dataclass
class FilterStats:
    """Диагностика отсева сигналов по фильтрам"""
    potential_setups: int = 0
    filtered_by_trend: int = 0
    filtered_by_volume: int = 0
    filtered_by_rr: int = 0
    filtered_by_atr: int = 0
    final_trades: int = 0

    def to_dict(self) -> Dict:
        return {
            'potential_setups': self.potential_setups,
            'filtered_by_trend': self.filtered_by_trend,
            'filtered_by_volume': self.filtered_by_volume,
            'filtered_by_rr': self.filtered_by_rr,
            'filtered_by_atr': self.filtered_by_atr,
            'final_trades': self.final_trades
        }


def run_backtest(
    df: pd.DataFrame,
    signal_generator,
    deposit: float,
    model: TradingModel,
    lookback_window: int = 300,
    max_holding_candles: int = 50,
    signal_kwargs: Optional[Dict] = None,
    debug_filters: bool = False,
    on_trade_close=None,
    risk_constraints: Optional[Dict[str, float]] = None,
    execution_config: Optional[Dict[str, Any]] = None,
    cost_config: Optional[Dict[str, Any]] = None,
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
        return _create_empty_results(
            model.name,
            deposit,
            FilterStats() if debug_filters else None
        )

    trades = []
    balance = deposit
    peak_balance = deposit
    max_drawdown = 0
    balances = [deposit]
    filter_stats = FilterStats() if debug_filters else None
    risk_constraints = risk_constraints or {}
    execution_config = execution_config or {}
    cost_config = cost_config or {}
    max_daily_loss_pct = float(risk_constraints.get('max_daily_loss_pct', 0.0))
    equity_dd_stop_pct = float(risk_constraints.get('equity_dd_stop_pct', 0.0))
    day_start_balance: dict[pd.Timestamp, float] = {}
    day_realized_pnl: dict[pd.Timestamp, float] = {}
    blocked_days: set[pd.Timestamp] = set()
    backtest_stopped = False

    # Проходим по данным
    for i in range(lookback_window, len(df) - max_holding_candles):
        df_window = df.iloc[:i+1].copy()
        trade_day = pd.Timestamp(df.index[i]).normalize()
        if trade_day not in day_start_balance:
            day_start_balance[trade_day] = balance
            day_realized_pnl.setdefault(trade_day, 0.0)

        if trade_day in blocked_days:
            balances.append(balance)
            continue

        if max_daily_loss_pct > 0:
            day_base = day_start_balance.get(trade_day, balance)
            if day_base > 0:
                day_pnl_pct = (day_realized_pnl.get(trade_day, 0.0) / day_base) * 100.0
                if day_pnl_pct <= -max_daily_loss_pct:
                    blocked_days.add(trade_day)
                    balances.append(balance)
                    continue

        # Генерируем сигнал
        if signal_kwargs:
            effective_signal_kwargs = dict(signal_kwargs)
            effective_signal_kwargs = _inject_trade_frequency_relaxation(
                signal_kwargs=effective_signal_kwargs,
                trades_count=len(trades),
                window_df=df_window,
            )
            signal = signal_generator(df_window, balance, model, **effective_signal_kwargs)
        else:
            signal = signal_generator(df_window, balance, model)

        if filter_stats is not None:
            if getattr(signal, 'debug_potential_setup', False):
                filter_stats.potential_setups += 1

            filter_stage = getattr(signal, 'debug_filter_stage', 'none')
            if filter_stage == 'filtered_trend':
                filter_stats.filtered_by_trend += 1
            elif filter_stage == 'filtered_volume':
                filter_stats.filtered_by_volume += 1
            elif filter_stage == 'filtered_rr':
                filter_stats.filtered_by_rr += 1
            elif filter_stage == 'filtered_atr':
                filter_stats.filtered_by_atr += 1

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

        exit_state = _simulate_trade_exit(
            df=df,
            start_idx=i,
            max_holding_candles=max_holding_candles,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            position_size=position_size,
            execution_config=execution_config,
        )
        if not bool(exit_state.get('resolved', True)):
            balances.append(balance)
            continue
        exit_price = float(exit_state['exit_price'])
        exit_reason = str(exit_state['exit_reason'])
        exit_time = exit_state['exit_time']
        exit_candle_idx = int(exit_state['exit_candle_idx'])
        gross_pnl = float(exit_state['pnl'])
        entry_turnover = abs(entry_price * position_size)
        exit_turnover = float(exit_state.get('exit_turnover', abs(exit_price * position_size)))
        fees = _calculate_trading_fees(
            position_size=position_size,
            entry_turnover=entry_turnover,
            exit_turnover=exit_turnover,
            cost_config=cost_config,
        )
        slippage = _calculate_slippage_cost(entry_turnover, exit_turnover, cost_config)
        pnl = gross_pnl - fees - slippage

        risk_amount = abs(entry_price - stop_price) * position_size
        rr_actual = (pnl / risk_amount) if risk_amount > 0 else 0.0

        pnl_percent = (pnl / balance) * 100
        balance += pnl
        day_realized_pnl[trade_day] = day_realized_pnl.get(trade_day, 0.0) + pnl

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
            duration_candles=exit_candle_idx - i,
            regime=getattr(signal, 'market_regime', None),
            gross_pnl=gross_pnl,
            fees=fees,
            slippage=slippage,
            entry_turnover=entry_turnover,
            exit_turnover=exit_turnover,
        )
        trades.append(trade)
        if on_trade_close is not None:
            on_trade_close(trade)
        if filter_stats is not None:
            filter_stats.final_trades += 1

        if max_daily_loss_pct > 0:
            day_base = day_start_balance.get(trade_day, balance)
            day_pnl_pct = (day_realized_pnl.get(trade_day, 0.0) / day_base) * 100.0 if day_base > 0 else 0.0
            if day_pnl_pct <= -max_daily_loss_pct:
                blocked_days.add(trade_day)

        if equity_dd_stop_pct > 0 and max_drawdown >= equity_dd_stop_pct:
            backtest_stopped = True
            break

    if backtest_stopped:
        balances.append(balance)

    # Расчет статистики
    return _calculate_statistics(
        trades,
        deposit,
        balance,
        max_drawdown,
        balances,
        model.name,
        filter_stats=filter_stats
    )


def _inject_trade_frequency_relaxation(
    signal_kwargs: Dict[str, Any],
    trades_count: int,
    window_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Автоматическое ослабление soft-фильтров, если фактическая частота сделок
    отстает от целевой min_expected_trades_per_month.
    """
    target = float(signal_kwargs.get("min_expected_trades_per_month", 0.0) or 0.0)
    if target <= 0:
        return signal_kwargs

    if window_df.empty or len(window_df.index) < 2:
        signal_kwargs["soft_threshold_relaxation"] = 0.0
        return signal_kwargs

    start_ts = pd.Timestamp(window_df.index[0])
    end_ts = pd.Timestamp(window_df.index[-1])
    elapsed_days = max((end_ts - start_ts).total_seconds() / 86400.0, 1.0)
    elapsed_months = max(elapsed_days / 30.0, 1.0 / 30.0)
    expected_trades = target * elapsed_months
    if expected_trades <= 0:
        signal_kwargs["soft_threshold_relaxation"] = 0.0
        return signal_kwargs

    deficit_ratio = max(0.0, (expected_trades - float(trades_count)) / expected_trades)
    # До -0.18 к порогу confidence; этого достаточно, чтобы поднять trade-flow,
    # но не разрушить risk-профиль.
    relaxation = min(0.18, deficit_ratio * 0.18)
    signal_kwargs["soft_threshold_relaxation"] = relaxation
    return signal_kwargs


def _simulate_trade_exit(
    df: pd.DataFrame,
    start_idx: int,
    max_holding_candles: int,
    direction: str,
    entry_price: float,
    stop_price: float,
    target_price: float,
    position_size: float,
    execution_config: Dict[str, Any],
) -> Dict[str, Any]:
    sl_tp_only = bool(execution_config.get('sl_tp_only', True))
    partial_enabled = bool(execution_config.get('partial_take_profit', False)) and not sl_tp_only
    partial_tp_rr = float(execution_config.get('partial_tp_rr', 1.0))
    partial_close_fraction = float(execution_config.get('partial_close_fraction', 0.5))
    partial_close_fraction = max(0.0, min(1.0, partial_close_fraction))
    trail_after_partial = bool(execution_config.get('trail_after_partial', False)) and not sl_tp_only
    trailing_atr_multiplier = float(execution_config.get('trailing_atr_multiplier', 1.0))

    risk_per_unit = abs(entry_price - stop_price)
    if risk_per_unit <= 0 or position_size <= 0:
        return {
            'resolved': False,
            'exit_price': entry_price,
            'exit_reason': 'invalid_risk',
            'exit_time': df.index[start_idx],
            'exit_candle_idx': start_idx,
            'pnl': 0.0,
            'exit_turnover': 0.0,
        }

    if direction == 'long':
        partial_price = entry_price + risk_per_unit * partial_tp_rr
    else:
        partial_price = entry_price - risk_per_unit * partial_tp_rr

    remaining_size = position_size
    realized_pnl = 0.0
    exit_turnover = 0.0
    partial_taken = False
    stop_dynamic = stop_price

    end_idx = len(df) - 1 if sl_tp_only else min(start_idx + max_holding_candles, len(df) - 1)
    for j in range(start_idx + 1, end_idx + 1):
        candle_high = float(df['high'].iloc[j])
        candle_low = float(df['low'].iloc[j])
        candle_close = float(df['close'].iloc[j])
        atr_value = float(df['atr'].iloc[j]) if 'atr' in df.columns and not pd.isna(df['atr'].iloc[j]) else 0.0

        if direction == 'long':
            if candle_low <= stop_dynamic:
                realized_pnl += (stop_dynamic - entry_price) * remaining_size
                exit_turnover += abs(stop_dynamic * remaining_size)
                return {
                    'resolved': True,
                    'exit_price': stop_dynamic,
                    'exit_reason': 'stop_after_partial' if partial_taken else 'stop',
                    'exit_time': df.index[j],
                    'exit_candle_idx': j,
                    'pnl': realized_pnl,
                    'exit_turnover': exit_turnover,
                }

            if partial_enabled and not partial_taken and candle_high >= partial_price:
                close_size = remaining_size * partial_close_fraction
                realized_pnl += (partial_price - entry_price) * close_size
                exit_turnover += abs(partial_price * close_size)
                remaining_size -= close_size
                partial_taken = True
                stop_dynamic = max(stop_dynamic, entry_price)

                if remaining_size <= 1e-12:
                    return {
                        'resolved': True,
                        'exit_price': partial_price,
                        'exit_reason': 'partial_full_exit',
                        'exit_time': df.index[j],
                        'exit_candle_idx': j,
                        'pnl': realized_pnl,
                        'exit_turnover': exit_turnover,
                    }

            if partial_taken and trail_after_partial and atr_value > 0:
                stop_dynamic = max(stop_dynamic, candle_close - atr_value * trailing_atr_multiplier)

            if candle_high >= target_price:
                realized_pnl += (target_price - entry_price) * remaining_size
                exit_turnover += abs(target_price * remaining_size)
                return {
                    'resolved': True,
                    'exit_price': target_price,
                    'exit_reason': 'target_after_partial' if partial_taken else 'target',
                    'exit_time': df.index[j],
                    'exit_candle_idx': j,
                    'pnl': realized_pnl,
                    'exit_turnover': exit_turnover,
                }

        else:
            if candle_high >= stop_dynamic:
                realized_pnl += (entry_price - stop_dynamic) * remaining_size
                exit_turnover += abs(stop_dynamic * remaining_size)
                return {
                    'resolved': True,
                    'exit_price': stop_dynamic,
                    'exit_reason': 'stop_after_partial' if partial_taken else 'stop',
                    'exit_time': df.index[j],
                    'exit_candle_idx': j,
                    'pnl': realized_pnl,
                    'exit_turnover': exit_turnover,
                }

            if partial_enabled and not partial_taken and candle_low <= partial_price:
                close_size = remaining_size * partial_close_fraction
                realized_pnl += (entry_price - partial_price) * close_size
                exit_turnover += abs(partial_price * close_size)
                remaining_size -= close_size
                partial_taken = True
                stop_dynamic = min(stop_dynamic, entry_price)

                if remaining_size <= 1e-12:
                    return {
                        'resolved': True,
                        'exit_price': partial_price,
                        'exit_reason': 'partial_full_exit',
                        'exit_time': df.index[j],
                        'exit_candle_idx': j,
                        'pnl': realized_pnl,
                        'exit_turnover': exit_turnover,
                    }

            if partial_taken and trail_after_partial and atr_value > 0:
                stop_dynamic = min(stop_dynamic, candle_close + atr_value * trailing_atr_multiplier)

            if candle_low <= target_price:
                realized_pnl += (entry_price - target_price) * remaining_size
                exit_turnover += abs(target_price * remaining_size)
                return {
                    'resolved': True,
                    'exit_price': target_price,
                    'exit_reason': 'target_after_partial' if partial_taken else 'target',
                    'exit_time': df.index[j],
                    'exit_candle_idx': j,
                    'pnl': realized_pnl,
                    'exit_turnover': exit_turnover,
                }

    if sl_tp_only:
        # Без SL/TP-срабатывания позиция считается незавершенной и не учитывается в статистике.
        return {
            'resolved': False,
            'exit_price': entry_price,
            'exit_reason': 'unresolved',
            'exit_time': df.index[end_idx],
            'exit_candle_idx': end_idx,
            'pnl': 0.0,
            'exit_turnover': 0.0,
        }

    timeout_price = float(df['close'].iloc[end_idx])
    if direction == 'long':
        realized_pnl += (timeout_price - entry_price) * remaining_size
    else:
        realized_pnl += (entry_price - timeout_price) * remaining_size
    exit_turnover += abs(timeout_price * remaining_size)

    return {
        'resolved': True,
        'exit_price': timeout_price,
        'exit_reason': 'timeout_after_partial' if partial_taken else 'timeout',
        'exit_time': df.index[end_idx],
        'exit_candle_idx': end_idx,
        'pnl': realized_pnl,
        'exit_turnover': exit_turnover,
    }


def _calculate_statistics(
    trades: List[Trade],
    initial_deposit: float,
    final_balance: float,
    max_drawdown: float,
    balances: List[float],
    model_name: str,
    filter_stats: Optional[FilterStats] = None
) -> BacktestResults:
    """Расчет статистики по сделкам"""

    if not trades:
        return _create_empty_results(model_name, initial_deposit, filter_stats)

    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl <= 0]
    gross_winning_trades_list = [t for t in trades if t.gross_pnl > 0]
    gross_losing_trades_list = [t for t in trades if t.gross_pnl <= 0]

    total_trades = len(trades)
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    winrate = (win_count / total_trades * 100) if total_trades > 0 else 0
    gross_win_count = len(gross_winning_trades_list)
    gross_loss_count = len(gross_losing_trades_list)
    gross_winrate = (gross_win_count / total_trades * 100) if total_trades > 0 else 0

    avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
    avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0
    best_trade = max([t.pnl for t in trades]) if trades else 0
    worst_trade = min([t.pnl for t in trades]) if trades else 0

    total_profit = sum([t.pnl for t in trades])
    gross_total_profit = sum([t.gross_pnl for t in trades])
    total_fees = sum([t.fees for t in trades])
    total_slippage = sum([t.slippage for t in trades])
    total_costs = total_fees + total_slippage

    # Expectancy
    expectancy = (winrate / 100 * avg_win) - ((100 - winrate) / 100 * avg_loss)

    # R-multiples (P&L normalized by planned risk per trade)
    trade_r_values = []
    risk_amounts = []
    for t in trades:
        risk_amt = abs(t.entry_price - t.stop_price) * t.position_size
        if risk_amt > 0:
            risk_amounts.append(risk_amt)
            trade_r_values.append(t.pnl / risk_amt)

    avg_risk_per_trade = float(np.mean(risk_amounts)) if risk_amounts else 0.0
    winning_r = [r for r in trade_r_values if r > 0]
    losing_r = [r for r in trade_r_values if r <= 0]
    avg_win_r = float(np.mean(winning_r)) if winning_r else 0.0
    avg_loss_r = abs(float(np.mean(losing_r))) if losing_r else 0.0
    best_trade_r = float(max(trade_r_values)) if trade_r_values else 0.0
    worst_trade_r = float(min(trade_r_values)) if trade_r_values else 0.0
    expectancy_r = float(np.mean(trade_r_values)) if trade_r_values else 0.0

    # Profit factor
    gross_profit = sum([t.pnl for t in winning_trades]) if winning_trades else 0
    gross_loss = abs(sum([t.pnl for t in losing_trades])) if losing_trades else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    gross_profit_before_cost = sum([t.gross_pnl for t in gross_winning_trades_list]) if gross_winning_trades_list else 0
    gross_loss_before_cost = abs(sum([t.gross_pnl for t in gross_losing_trades_list])) if gross_losing_trades_list else 1
    gross_profit_factor = (
        gross_profit_before_cost / gross_loss_before_cost if gross_loss_before_cost > 0 else 0
    )

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
        trades=trades,
        filter_stats=filter_stats,
        avg_win_r=avg_win_r,
        avg_loss_r=avg_loss_r,
        best_trade_r=best_trade_r,
        worst_trade_r=worst_trade_r,
        expectancy_r=expectancy_r,
        avg_risk_per_trade=avg_risk_per_trade,
        gross_total_profit=gross_total_profit,
        total_fees=total_fees,
        total_slippage=total_slippage,
        total_costs=total_costs,
        gross_winning_trades=gross_win_count,
        gross_losing_trades=gross_loss_count,
        gross_winrate=gross_winrate,
        gross_profit_factor=gross_profit_factor,
    )


def _create_empty_results(
    model_name: str,
    deposit: float,
    filter_stats: Optional[FilterStats] = None
) -> BacktestResults:
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
        trades=[],
        filter_stats=filter_stats,
        avg_win_r=0,
        avg_loss_r=0,
        best_trade_r=0,
        worst_trade_r=0,
        expectancy_r=0,
        avg_risk_per_trade=0,
        gross_total_profit=0,
        total_fees=0,
        total_slippage=0,
        total_costs=0,
        gross_winning_trades=0,
        gross_losing_trades=0,
        gross_winrate=0,
        gross_profit_factor=0,
    )


def _calculate_trading_fees(
    position_size: float,
    entry_turnover: float,
    exit_turnover: float,
    cost_config: Dict[str, Any],
) -> float:
    if not cost_config.get('enabled', True):
        return 0.0

    engine = str(cost_config.get('engine', 'stock')).lower()
    if engine == 'futures':
        per_contract = float(cost_config.get('futures_per_contract_rub', 2.0))
        fee_mode = str(cost_config.get('futures_fee_mode', 'round_trip')).lower()
        contracts = abs(position_size)
        if fee_mode == 'per_side':
            # Комиссия задана за сторону, считаем вход + выход.
            return per_contract * contracts * 2.0
        # round_trip: комиссия уже задана за полный цикл сделки.
        return per_contract * contracts

    bps = float(cost_config.get('securities_bps', 0.0)) + float(cost_config.get('settlement_bps', 0.0))
    turnover = abs(entry_turnover) + abs(exit_turnover)
    return turnover * bps / 10000.0


def _calculate_slippage_cost(entry_turnover: float, exit_turnover: float, cost_config: Dict[str, Any]) -> float:
    slippage_bps = float(cost_config.get('slippage_bps', 0.0))
    if slippage_bps <= 0:
        return 0.0
    turnover = abs(entry_turnover) + abs(exit_turnover)
    return turnover * slippage_bps / 10000.0


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
        f"{'GrossW%':>8} "
        f"{'Expect':>8} "
        f"{'PF':>6} "
        f"{'GrossPF':>8} "
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
            f"{res.gross_winrate:>8.1f} "
            f"{res.expectancy:>8.2f} "
            f"{res.profit_factor:>6.2f} "
            f"{res.gross_profit_factor:>8.2f} "
            f"{res.max_drawdown_percent:>8.1f} "
            f"{res.return_pct:>9.1f} "
            f"{res.sharpe_ratio:>7.2f} "
            f"{res.avg_trade_duration:>7}"
        )
        lines.append(row)

    lines.append("=" * 120)

    # Лучшая модель: выбираем только среди моделей с реальной торговой активностью
    active_results = [r for r in results if r.total_trades > 0]
    if active_results:
        best_model = max(active_results, key=lambda x: x.expectancy)
        lines.append(f"\n🏆 Лучшая модель: {best_model.model_name.upper()}")
        lines.append(f"   Expectancy: {best_model.expectancy:.2f}")
        lines.append(f"   Winrate: {best_model.winrate:.1f}%")
        lines.append(f"   Gross Winrate: {best_model.gross_winrate:.1f}%")
        lines.append(f"   Return: {best_model.return_pct:.1f}%")
    else:
        lines.append("\n🏆 Лучшая модель: НЕТ (все модели без сделок)")

    # Диагностика влияния издержек на net-метрики
    costs_kill_edge = [
        r for r in results
        if r.total_trades > 0 and r.gross_winrate > r.winrate and r.gross_profit_factor > r.profit_factor
    ]
    if costs_kill_edge:
        lines.append("\nℹ Net Win% и PF учитывают комиссии/слиппедж.")
        lines.append("ℹ GrossW% и GrossPF — метрики до издержек.")

    return "\n".join(lines)
