#!/usr/bin/env python
"""
CLI для профессиональной торговой системы с моделями
"""

import argparse
import json
import sys
from dataclasses import replace
from typing import Any, Optional

try:
    from colorama import Fore, Style, init as colorama_init
    HAS_COLORAMA = True
except ImportError:  # pragma: no cover - optional dependency
    Fore = Style = None
    HAS_COLORAMA = False
from services.strategy_engine.adaptive_model import build_regime_model
from services.strategy_engine.adaptive_signal_engine import AdaptiveSignalEngine
from services.strategy_engine.backtest_engine import evaluate_model
from services.strategy_engine.regime_detection import classify_current_regime
from services.strategy_engine.risk_manager import apply_regime_risk
from services.strategy_engine.setup_generator import (
    TradeSetup,
    export_setups_json,
    generate_trade_setup,
    load_setups_json,
)
from services.notification.alert_sender import AlertSender
from loguru import logger
from services.strategy_engine.filter_config import (
    apply_filters_to_model,
    load_config,
)
from services.strategy_engine import (
    get_model,
    MODELS,
    compare_models,
    generate_signal,
    run_backtest,
    compare_models_results
)


COLOR_ENABLED = False
SUPPORTED_EXCHANGES = ("moex", "binance")


def _default_board(exchange: str, engine: str) -> str:
    if exchange == "moex":
        return "RFUD" if engine == "futures" else "TQBR"
    return ""


def _build_exchange_adapter(exchange: str, engine: str, market: str):
    if exchange == "moex":
        from adapters.moex.iss_client import MOEXAdapter

        return MOEXAdapter(engine=engine, market=market)
    raise NotImplementedError(f"Exchange '{exchange}' is not implemented yet.")


def _load_cli_dataset(
    *,
    exchange: str,
    ticker: str,
    timeframe: str,
    start_date: Optional[str],
    end_date: Optional[str],
    board: str,
    adapter,
    limit: Optional[int] = None,
):
    if exchange == "moex":
        from adapters.moex.cli_dataset_loader import load_cli_dataset

        return load_cli_dataset(
            ticker=ticker,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            board=board,
            adapter=adapter,
            limit=limit,
        )
    raise NotImplementedError(f"Exchange '{exchange}' is not implemented yet.")


def _monitor_setups(
    *,
    exchange: str,
    setups: list[dict[str, Any]],
    adapter,
    board: str,
    interval_seconds: int,
    export_path: str | None,
    sender: AlertSender | None,
    max_cycles: int,
    filter_config: dict[str, Any] | None,
    monitor_workers: int,
):
    if exchange == "moex":
        from adapters.moex.setup_monitor import monitor_setups

        return monitor_setups(
            setups=setups,
            adapter=adapter,
            board=board,
            interval_seconds=interval_seconds,
            export_path=export_path,
            sender=sender,
            max_cycles=max_cycles,
            filter_config=filter_config,
            monitor_workers=monitor_workers,
        )
    raise NotImplementedError(f"Exchange '{exchange}' is not implemented yet.")


def _resolve_contract_spec(
    *,
    exchange: str,
    adapter,
    symbol: str,
    board: str,
) -> dict[str, Any]:
    if exchange != "moex":
        return {}
    getter = getattr(adapter, "get_security_spec", None)
    if not callable(getter):
        return {}
    try:
        return getter(symbol, board=board) or {}
    except Exception:
        return {}


def setup_cli_colors(enabled: bool):
    global COLOR_ENABLED
    COLOR_ENABLED = bool(enabled and HAS_COLORAMA and sys.stdout.isatty())
    if COLOR_ENABLED:
        colorama_init(autoreset=True)


def colorize(text: str, color: Optional[str] = None, bright: bool = False) -> str:
    if not COLOR_ENABLED or not color:
        return text

    palette = {
        'red': Fore.RED,
        'green': Fore.GREEN,
        'yellow': Fore.YELLOW,
        'blue': Fore.BLUE,
        'magenta': Fore.MAGENTA,
        'cyan': Fore.CYAN,
        'white': Fore.WHITE,
    }
    prefix = palette.get(color, '')
    if bright:
        prefix = f"{Style.BRIGHT}{prefix}"
    return f"{prefix}{text}{Style.RESET_ALL}"


def print_separator(char='=', length=80, color: Optional[str] = 'cyan'):
    logger.info(colorize(char * length, color=color, bright=True))


def estimate_signal_fee(signal_dict: dict, cost_config: dict[str, Any]) -> float:
    """Оценка комиссии round-trip для текущего сигнала."""
    if signal_dict.get("signal") == "none":
        return 0.0
    if not bool(cost_config.get("enabled", True)):
        return 0.0

    position_size = abs(float(signal_dict.get("position_size", 0.0)))
    entry_price = abs(float(signal_dict.get("entry", 0.0)))
    if position_size <= 0 or entry_price <= 0:
        return 0.0

    engine = str(cost_config.get("engine", "stock")).lower()
    if engine == "futures":
        per_contract = float(cost_config.get("futures_per_contract_rub", 2.0))
        fee_mode = str(cost_config.get("futures_fee_mode", "round_trip")).lower()
        if fee_mode == "per_side":
            return per_contract * position_size * 2.0
        return per_contract * position_size

    bps = float(cost_config.get("securities_bps", 0.0)) + float(cost_config.get("settlement_bps", 0.0))
    turnover = (entry_price * position_size) * 2.0
    return turnover * bps / 10000.0


def print_model_info(model):
    """Вывод информации о модели"""
    print_separator(color='blue')
    logger.info(colorize(f"📊 АКТИВНАЯ МОДЕЛЬ: {model.name.upper()}", 'blue', bright=True))
    print_separator(color='blue')
    logger.info(f"\n📝 Описание: {model.description}")
    logger.info(f"\n📉 Параметры:")
    logger.info(f"   Min RR:              {model.min_rr}")
    logger.info(f"   Max Risk:            {model.max_risk_percent}%")
    logger.info(f"   Volume filter:       {model.min_volume_ratio}x")
    logger.info(f"   Trend required:      {'Yes' if model.trend_required else 'No'}")
    logger.info(f"   Allow range:         {'Yes' if model.allow_range else 'No'}")
    logger.info(f"   ATR stop multiplier: {model.atr_multiplier_stop}")
    logger.info(f"   Min confidence:      {model.min_confidence}")
    logger.info(f"   Min trend strength:  {model.min_trend_strength}%")
    print_separator()


def print_signal_report(
    signal_dict: dict,
    cost_config: Optional[dict[str, Any]] = None,
    contract_multiplier: float = 1.0,
):
    """Красивый вывод торгового сигнала"""
    print_separator(color='blue')
    logger.info(colorize("📊 ТОРГОВЫЙ СИГНАЛ", 'blue', bright=True))
    print_separator(color='blue')

    signal_emoji = {
        'long': '🟢 LONG',
        'short': '🔴 SHORT',
        'none': '⚪ НЕТ СИГНАЛА'
    }
    direction_color = {'long': 'green', 'short': 'red', 'none': 'yellow'}.get(signal_dict['signal'], 'white')
    logger.info(f"\nНаправление: {colorize(signal_emoji.get(signal_dict['signal'], signal_dict['signal']), direction_color, bright=True)}")
    logger.info(f"Уверенность: {signal_dict['confidence'].upper()}")

    if signal_dict['signal'] != 'none':
        effective_multiplier = contract_multiplier if contract_multiplier > 0 else 1.0

        logger.info(f"\n💰 ПАРАМЕТРЫ СДЕЛКИ:")
        logger.info(f"   Вход:     {signal_dict['entry']:.2f}")
        logger.info(f"   Стоп:     {signal_dict['stop']:.2f}")
        logger.info(f"   Цель:     {signal_dict['target']:.2f}")
        logger.info(f"   RR:       {signal_dict['rr']:.2f}")

        logger.info(f"\n📈 РИСК-МЕНЕДЖМЕНТ:")
        logger.info(f"   Размер позиции:  {signal_dict['position_size']:.0f} контрактов")
        logger.info(f"   Риск в рублях:   {signal_dict['risk_rub']:.2f} ₽")
        logger.info(f"   Риск в %:        {signal_dict['risk_percent']:.2f}%")
        fee_estimate = estimate_signal_fee(signal_dict, cost_config or {})
        logger.info(f"   Fee (оценка):    {fee_estimate:.2f} ₽")
        potential_profit = (
            abs(signal_dict['target'] - signal_dict['entry'])
            * signal_dict['position_size']
            * effective_multiplier
        )
        logger.info(f"   Потенциал:       {potential_profit:.2f} ₽")

    logger.info(f"\n📉 СТРУКТУРА РЫНКА:")
    logger.info(f"   Тренд:           {signal_dict['structure']}")
    logger.info(f"   Фаза:            {signal_dict['phase']}")
    logger.info(f"   ATR:             {signal_dict['atr']:.2f}")
    if signal_dict.get('market_regime'):
        logger.info(f"   Regime:          {signal_dict['market_regime']}")

    logger.info(f"\n📊 ИНДИКАТОРЫ:")
    logger.info(f"   RSI:             {signal_dict['rsi']:.1f}")
    logger.info(f"   Расст. до MA50:  {signal_dict['distance_ma50_pct']:+.2f}%")
    logger.info(f"   Расст. до MA200: {signal_dict['distance_ma200_pct']:+.2f}%")
    logger.info(f"   Volume ratio:    {signal_dict['volume_ratio']:.2f}x")

    if signal_dict['warnings']:
        logger.info(colorize("\n⚠️  ПРЕДУПРЕЖДЕНИЯ:", 'yellow', bright=True))
        for warning in signal_dict['warnings']:
            logger.info(colorize(f"   • {warning}", 'yellow'))

    print_separator(color='blue')


def print_backtest_report(backtest: dict, show_details: bool = True):
    """Красивый вывод результатов бэктеста"""
    print_separator(color='blue')
    logger.info(colorize(f"📈 РЕЗУЛЬТАТЫ БЭКТЕСТА - {backtest['model_name'].upper()}", 'blue', bright=True))
    print_separator(color='blue')

    if backtest['total_trades'] == 0:
        logger.info(colorize("\n❌ Сделок не найдено", 'red', bright=True))
        return

    logger.info(f"\n📊 СТАТИСТИКА:")
    logger.info(f"   Всего сделок:     {backtest['total_trades']}")
    logger.info(f"   Прибыльных:       {backtest['winning_trades']} ({backtest['winrate']:.1f}%)")
    logger.info(f"   Убыточных:        {backtest['losing_trades']}")
    if 'gross_winning_trades' in backtest:
        logger.info(
            "   Gross прибыльных: "
            f"{backtest['gross_winning_trades']} ({backtest.get('gross_winrate', 0):.1f}%)"
        )
    logger.info(f"   Средняя длит.:    {backtest['avg_trade_duration']} свечей")

    if show_details:
        initial_deposit = backtest['final_balance'] - backtest['total_profit']
        logger.info(f"\n💰 ФИНАНСЫ:")
        logger.info(f"   Начальный депо:   {initial_deposit:.2f} ₽")
        logger.info(f"   Конечный депо:    {backtest['final_balance']:.2f} ₽")
        if 'gross_total_profit' in backtest:
            logger.info(f"   Gross PnL:        {backtest['gross_total_profit']:+.2f} ₽")
            logger.info(f"   Fees:             -{backtest.get('total_fees', 0):.2f} ₽")
            logger.info(f"   Slippage:         -{backtest.get('total_slippage', 0):.2f} ₽")
            logger.info(f"   Total costs:      -{backtest.get('total_costs', 0):.2f} ₽")
        logger.info(f"   Чистая прибыль:   {backtest['total_profit']:+.2f} ₽")
        logger.info(f"   Доходность:       {backtest['return_pct']:+.2f}%")

    logger.info(f"\n📈 МЕТРИКИ:")
    if 'avg_risk_per_trade' in backtest:
        logger.info(f"   Ср. риск/сделку:  {backtest['avg_risk_per_trade']:.2f} ₽")
    logger.info(f"   Средний выигрыш:  {backtest['avg_win']:.2f} ₽")
    logger.info(f"   Средний проигрыш: {backtest['avg_loss']:.2f} ₽")
    logger.info(f"   Лучшая сделка:    {backtest['best_trade']:.2f} ₽")
    logger.info(f"   Худшая сделка:    {backtest['worst_trade']:.2f} ₽")
    logger.info(f"   Expectancy:       {backtest['expectancy']:.2f} ₽")
    if 'expectancy_r' in backtest:
        logger.info(f"   Avg Win (R):      {backtest['avg_win_r']:.2f}R")
        logger.info(f"   Avg Loss (R):     {backtest['avg_loss_r']:.2f}R")
        logger.info(f"   Best/Worst (R):   {backtest['best_trade_r']:.2f}R / {backtest['worst_trade_r']:.2f}R")
        logger.info(f"   Expectancy (R):   {backtest['expectancy_r']:.3f}R")
    logger.info(f"   Profit Factor:    {backtest['profit_factor']:.2f}")
    if 'gross_profit_factor' in backtest:
        logger.info(f"   Gross PF:         {backtest['gross_profit_factor']:.2f}")
    logger.info(f"   Sharpe Ratio:     {backtest['sharpe_ratio']:.2f}")
    logger.info(f"   Max Drawdown:     {backtest['max_drawdown_percent']:.2f}%")

    if show_details:
        # Оценка системы
        logger.info(f"\n🎯 ОЦЕНКА СИСТЕМЫ:")
        score = 0
        if backtest['winrate'] >= 40:
            score += 1
            logger.info(colorize("   ✅ Winrate >= 40%", 'green'))
        else:
            logger.info(colorize("   ❌ Winrate < 40%", 'red'))

        if backtest['profit_factor'] >= 1.5:
            score += 1
            logger.info(colorize("   ✅ Profit Factor >= 1.5", 'green'))
        else:
            logger.info(colorize("   ❌ Profit Factor < 1.5", 'red'))

        if backtest['expectancy'] > 0:
            score += 1
            logger.info(colorize("   ✅ Expectancy > 0", 'green'))
        else:
            logger.info(colorize("   ❌ Expectancy <= 0", 'red'))

        if backtest['max_drawdown_percent'] < 20:
            score += 1
            logger.info(colorize("   ✅ Drawdown < 20%", 'green'))
        else:
            logger.info(colorize("   ⚠️  Drawdown >= 20%", 'yellow'))

        logger.info(f"\n   Итоговая оценка: {score}/4")

        if score >= 3:
            logger.info(colorize("   🌟 СИСТЕМА ПЕРСПЕКТИВНА", 'green', bright=True))
        elif score >= 2:
            logger.info(colorize("   ⚠️  СИСТЕМА ТРЕБУЕТ ДОРАБОТКИ", 'yellow', bright=True))
        else:
            logger.info(colorize("   ❌ СИСТЕМА НЕ РЕКОМЕНДУЕТСЯ", 'red', bright=True))

    print_separator(color='blue')


def print_filter_debug_stats(filter_stats: dict):
    """Вывод диагностики фильтров"""
    if not filter_stats:
        return

    print_separator('-', 80, color='yellow')
    logger.info(colorize("🔍 ДИАГНОСТИКА ФИЛЬТРАЦИИ", 'yellow', bright=True))
    print_separator('-', 80, color='yellow')
    logger.info(f"   Potential setups:   {filter_stats['potential_setups']}")
    logger.info(f"   Filtered by trend:  {filter_stats['filtered_by_trend']}")
    logger.info(f"   Filtered by volume: {filter_stats['filtered_by_volume']}")
    logger.info(f"   Filtered by RR:     {filter_stats['filtered_by_rr']}")
    logger.info(f"   Filtered by ATR:    {filter_stats['filtered_by_atr']}")
    logger.info(f"   Final trades:       {filter_stats['final_trades']}")
    print_separator('-', 80, color='yellow')


def print_setup_detected(setup: dict[str, Any]):
    logger.info(colorize("\n📈 NEW SETUP DETECTED", "green", bright=True))
    logger.info(f"Symbol: {setup['symbol']}")
    logger.info(f"Direction: {str(setup['direction']).upper()}")
    logger.info(f"Entry: {setup['entry_price']}")
    logger.info(f"Stop: {setup['stop_loss']}")
    logger.info(f"Take: {setup['take_profit']}")
    logger.info(f"RR: {setup['rr']}")
    logger.info(f"Confidence: {setup['confidence'] * 100:.0f}%")
    logger.info(f"Regime: {setup['regime']}")
    logger.info(f"Expires in: {setup.get('expires_in_candles', '?')} candles")


def calculate_score(model_stats: dict) -> float:
    """Базовый рейтинг (без walk-forward)"""
    pf = model_stats.get('profit_factor', 0)
    ret = model_stats.get('return_pct', 0)
    maxdd = model_stats.get('max_drawdown_percent', 0)
    return (pf * 0.5) + (ret * 0.3) - (maxdd * 0.2)


def get_model_selection_score(model_stats: dict) -> float:
    """Скор для выбора лучшей модели."""
    if 'robustness_score' in model_stats:
        return model_stats.get('robustness_score', 0)
    return model_stats.get('score', calculate_score(model_stats))


def model_has_trade_activity(model_stats: dict) -> bool:
    """Проверка, есть ли реальные сделки в статистике модели."""
    total_trades = model_stats.get('total_trades')
    if isinstance(total_trades, (int, float)):
        return total_trades > 0

    test_block = model_stats.get('test')
    if isinstance(test_block, dict):
        test_trades = test_block.get('total_trades', 0)
        if isinstance(test_trades, (int, float)):
            return test_trades > 0

    trades_by_regime = model_stats.get('trades_by_regime')
    if isinstance(trades_by_regime, dict):
        return sum(int(v or 0) for v in trades_by_regime.values()) > 0

    return False


def print_walk_forward_table(model_stats: dict[str, dict]):
    """Таблица устойчивости по моделям (walk-forward)."""
    if not model_stats:
        logger.info(colorize("\n❌ Нет данных walk-forward", 'red', bright=True))
        return

    logger.info("\n" + "-" * 72)
    logger.info(f"{'Model':<15} {'PF_train':>9} {'PF_test':>8} {'Stability':>10} {'RobustScore':>12}")
    logger.info("-" * 72)

    for model_name, stats in sorted(model_stats.items(), key=lambda x: x[1].get('robustness_score', 0), reverse=True):
        stability = stats.get('stability_ratio', 0)
        robustness = stats.get('robustness_score', 0)
        marker = " unstable" if stability < 0.7 else ""
        if stats.get('unstable_oos'):
            marker += " unstable_oos"
        if stats.get('overfit'):
            marker += " overfit"

        logger.info(
            f"{model_name:<15} "
            f"{stats.get('pf_train', 0):>9.2f} "
            f"{stats.get('pf_test', 0):>8.2f} "
            f"{stability:>10.2f} "
            f"{robustness:>12.2f}{marker}"
        )
    logger.info("-" * 72)


def print_regime_performance_summary(
    regime_stats: dict[str, dict[str, float]],
    overall_pf: Optional[float] = None,
    overall_maxdd: Optional[float] = None
):
    """Итоговая сводка перформанса по рыночным режимам."""
    if not regime_stats:
        return

    logger.info(colorize("\n📊 REGIME PERFORMANCE SUMMARY", 'magenta', bright=True))
    logger.info("-" * 65)
    logger.info(f"{'Regime':<16} {'Trades':>8} {'PF':>8} {'MaxDD':>8} {'Return':>10}")
    logger.info("-" * 65)

    rows = [
        ("trend", "Trend"),
        ("range", "Range"),
        ("high_volatility", "High Vol"),
    ]
    for key, label in rows:
        stats = regime_stats.get(key, {})
        logger.info(
            f"{label:<16} "
            f"{int(stats.get('trades', 0)):>8} "
            f"{float(stats.get('pf', 0)):>8.2f} "
            f"{float(stats.get('maxdd', 0)):>8.2f}% "
            f"{float(stats.get('return_pct', 0)):>+9.2f}%"
        )
    logger.info("-" * 65)

    if overall_pf is not None:
        logger.info(f"Overall PF: {overall_pf:.2f}")
    if overall_maxdd is not None:
        logger.info(f"MaxDD: {overall_maxdd:.2f}%")


def print_regime_final_decision(
    best_regime: Optional[str],
    best_regime_pf: Optional[float],
    enabled_regimes: Optional[list[str]] = None,
    disabled_regimes: Optional[dict[str, str]] = None,
):
    if best_regime:
        logger.info(colorize(f"\n🏆 Финальный режим: {best_regime} (PF {float(best_regime_pf or 0):.2f})", 'green', bright=True))

    if enabled_regimes:
        logger.info(f"Активные режимы: {', '.join(enabled_regimes)}")

    if disabled_regimes:
        logger.info(colorize("Отключенные режимы:", 'yellow', bright=True))
        for regime, reason in disabled_regimes.items():
            logger.info(colorize(f"  - {regime}: {reason}", 'yellow'))


def print_admission_report(admission: Optional[dict[str, Any]]):
    if not admission:
        return
    checks = admission.get('checks', {})
    if not checks:
        return

    logger.info(colorize("\nSystem Admission:", 'cyan', bright=True))
    for name, passed in checks.items():
        mark = colorize("PASS", 'green', bright=True) if passed else colorize("FAIL", 'red', bright=True)
        logger.info(f"  {name:<12} {mark}")

    if admission.get('all_passed'):
        logger.info(colorize("✅ Edge criteria passed", 'green', bright=True))
    else:
        logger.info(colorize(f"❌ Edge criteria failed: {', '.join(admission.get('failed', []))}", 'red', bright=True))


def print_edge_recommendations(admission: Optional[dict[str, Any]]):
    if not admission or admission.get('all_passed'):
        return

    failed = set(admission.get('failed', []))
    logger.info(colorize("\nРекомендации по улучшению edge:", 'yellow', bright=True))
    if 'pf' in failed:
        logger.info("  1. Ослабить volume/trend фильтры на 10-15% и перезапустить walk-forward.")
    if 'expectancy' in failed:
        logger.info("  2. Поднять минимальный RR до 2.2 и ужесточить входы только в направлении режима.")
    if 'sharpe' in failed:
        logger.info("  3. Снизить risk_per_trade_pct до 0.2-0.25% для стабилизации волатильности equity.")
    if 'maxdd' in failed:
        logger.info("  4. Ужесточить equity_dd_stop_pct до 6-7% и включить trailing после partial.")
    if 'stability' in failed:
        logger.info("  5. Увеличить train период и отключать режимы с PF_train < 1.15.")
    if 'risk_of_ruin' in failed:
        logger.info("  6. Снизить риск на сделку и ограничить торговлю только режимами с PF_test >= 1.3.")

def print_cross_instrument_stability(results: dict[str, dict[str, Any]]):
    """Cross-instrument устойчивость: сколько инструментов проходит у каждой модели."""
    if not results:
        return

    symbols = list(results.keys())
    total = len(symbols)
    if total == 0:
        return

    models = list(MODELS.keys())
    logger.info(colorize("\nCross-Instrument Stability:", 'magenta', bright=True))
    for model_name in models:
        stable_count = 0
        for symbol in symbols:
            stats = results.get(symbol, {}).get(model_name)
            if not stats:
                continue

            if 'pf_test' in stats:
                is_stable = stats.get('pf_test', 0) >= 1.2 and not stats.get('overfit', False)
            else:
                is_stable = stats.get('profit_factor', 0) >= 1.2

            if is_stable:
                stable_count += 1

        logger.info(f"   {model_name:<15} Stability across instruments: {stable_count}/{total}")


def parse_symbols(symbols_arg: Optional[str]) -> list[str]:
    """Парсинг списка тикеров из --symbols"""
    if not symbols_arg:
        return []
    return [s.strip() for s in symbols_arg.split(',') if s.strip()]


def run_backtest_for_symbol(
    exchange: str,
    symbol: str,
    deposit: float,
    timeframe: str,
    board: str,
    start_date: Optional[str],
    end_date: Optional[str],
    adapter,
    signal_kwargs: Optional[dict] = None,
    debug_filters: bool = False,
    verbose: bool = True,
    walk_forward: bool = False,
    monte_carlo_simulations: int = 0,
    limit: Optional[int] = None,
    adaptive_regime: bool = False,
    filter_config: Optional[dict] = None,
) -> dict[str, Any]:
    """Загрузка данных и optimize-прогон для одного инструмента"""
    if verbose:
        print_separator(color='cyan')
        logger.info(colorize(f"📥 Загрузка данных для {symbol}", 'cyan', bright=True))
        logger.info(f"   Таймфрейм: {timeframe}")
        logger.info(f"   Движок: {adapter.engine}, Рынок: {adapter.market}, Режим: {board}")
        print_separator(color='cyan')

    try:
        data_result = _load_cli_dataset(
            exchange=exchange,
            ticker=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            board=board,
            adapter=adapter,
            limit=limit
        )
    except Exception as exc:
        error_text = f"Ошибка загрузки данных: {exc}"
        if verbose:
            logger.info(colorize(f"\n❌ {error_text}", 'red', bright=True))
        return {
            'symbol': symbol,
            'models': {},
            'best_model': None,
            'best_stats': None,
            'data_points': 0,
            'period': None,
            'results': [],
            'error': error_text
        }
    df = data_result.df

    if df.empty:
        if verbose:
            logger.info(colorize(f"\n❌ Нет данных для {symbol}", 'red', bright=True))
        return {
            'symbol': symbol,
            'models': {},
            'best_model': None,
            'best_stats': None,
            'data_points': 0,
            'period': None,
            'results': [],
            'error': f'Нет данных для {symbol}'
        }

    if verbose:
        logger.info(colorize(f"✅ Загружено {len(df)} свечей", 'green', bright=True))
        logger.info(f"   Период: {df.index[0]} - {df.index[-1]}\n")
        for warning in data_result.warnings:
            logger.info(warning)
        if data_result.warnings:
            logger.info()
        print_separator(color='cyan')
        logger.info(colorize(f"🔄 ОПТИМИЗАЦИЯ МОДЕЛЕЙ ДЛЯ {symbol}", 'cyan', bright=True))
        print_separator(color='cyan')
        logger.info("\nЗапуск бэктеста для всех моделей...\n")

    results = []
    model_stats: dict[str, dict] = {}
    symbol_signal_kwargs = dict(signal_kwargs or {})
    spec = _resolve_contract_spec(
        exchange=exchange,
        adapter=adapter,
        symbol=symbol,
        board=board,
    )
    try:
        contract_margin_rub = float(spec.get("initial_margin"))
    except (TypeError, ValueError):
        contract_margin_rub = None
    try:
        contract_multiplier = float(spec.get("contract_multiplier"))
    except (TypeError, ValueError):
        contract_multiplier = None
    if contract_margin_rub is not None and contract_margin_rub > 0:
        symbol_signal_kwargs["contract_margin_rub"] = contract_margin_rub
    if contract_multiplier is not None and contract_multiplier > 0:
        symbol_signal_kwargs["contract_multiplier"] = contract_multiplier

    for model_name in MODELS.keys():
        if verbose:
            logger.info(f"  Тестирование модели: {model_name}...")
        model = get_model(model_name)
        if filter_config:
            model = apply_filters_to_model(model, filter_config)

        result_dict, raw_result = evaluate_model(
            df=df,
            deposit=deposit,
            model=model,
            signal_kwargs=symbol_signal_kwargs,
            walk_forward=walk_forward,
            monte_carlo_simulations=monte_carlo_simulations,
            debug_filters=debug_filters,
            adaptive_regime=adaptive_regime
        )

        results.append(raw_result)
        model_stats[model_name] = result_dict

        if verbose:
            if walk_forward:
                logger.info(
                    f"✓ (PF train/test: {result_dict['pf_train']:.2f}/{result_dict['pf_test']:.2f}, "
                    f"DD test: {result_dict['maxdd_test']:.2f}%, "
                    f"Stability: {result_dict['stability_ratio']:.2f}, "
                    f"Robust: {result_dict['robustness_score']:.2f})"
                )
                if result_dict.get('unstable'):
                    logger.info(colorize("   ⚠ unstable (Stability ratio < 0.7)", 'yellow'))
                if result_dict.get('monte_carlo'):
                    mc = result_dict['monte_carlo']
                    prob = mc.get('probability_drawdown_over_30', 0)
                    logger.info(
                        f"   MonteCarlo: worst DD {mc.get('worst_drawdown_percent', 0):.2f}%, "
                        f"q5 equity {mc.get('quantile_5_equity', 0):.2f}"
                    )
                    logger.info(f"   📉 Вероятность просадки > 30%: {prob:.2f}%")
                if result_dict.get('disabled_regimes'):
                    logger.info(colorize(f"   Disabled regimes: {result_dict['disabled_regimes']}", 'yellow'))
                logger.info(colorize(f"   Risk of ruin: {result_dict.get('risk_of_ruin', 0):.2f}%", 'cyan'))
                print_admission_report(result_dict.get('admission'))
            else:
                logger.info(f"✓ ({raw_result.total_trades} сделок)")
                logger.info(colorize(f"   Risk of ruin: {result_dict.get('risk_of_ruin', 0):.2f}%", 'cyan'))
                if debug_filters:
                    print_filter_debug_stats(result_dict.get('filter_stats'))

    best_model = None
    best_stats = None
    if model_stats:
        active_models = [name for name, stats in model_stats.items() if model_has_trade_activity(stats)]
        candidate_models = active_models if active_models else list(model_stats.keys())
        best_model = max(candidate_models, key=lambda name: get_model_selection_score(model_stats[name]))
        best_stats = model_stats[best_model]

    if verbose and best_stats and best_stats.get('market_regime_performance'):
        overall_pf = best_stats.get('pf_test', best_stats.get('profit_factor'))
        overall_maxdd = best_stats.get('maxdd_test', best_stats.get('max_drawdown_percent'))
        print_regime_performance_summary(
            best_stats['market_regime_performance'],
            overall_pf=overall_pf,
            overall_maxdd=overall_maxdd
        )

    return {
        'symbol': symbol,
        'models': model_stats,
        'best_model': best_model,
        'best_stats': best_stats,
        'data_points': len(df),
        'period': {'start': str(df.index[0]), 'end': str(df.index[-1])},
        'results': results,
        'error': None
    }


def aggregate_results(results: dict[str, dict]) -> dict[str, Any]:
    """Агрегация результатов по инструментам и вычисление победителей"""
    summary_rows = []
    best_models_by_symbol = {}

    for symbol, model_stats in results.items():
        if not model_stats:
            continue

        active_models = [name for name, stats in model_stats.items() if model_has_trade_activity(stats)]
        candidate_models = active_models if active_models else list(model_stats.keys())
        best_model = max(
            candidate_models,
            key=lambda name: get_model_selection_score(model_stats[name])
        )
        best_stats = model_stats[best_model]
        best_models_by_symbol[symbol] = best_model

        pf_train = best_stats.get('pf_train', best_stats.get('profit_factor', 0))
        pf_test = best_stats.get('pf_test', best_stats.get('profit_factor', 0))
        maxdd_value = best_stats.get('maxdd_test', best_stats.get('max_drawdown_percent', 0))
        return_value = best_stats.get('return_pct_test', best_stats.get('return_pct', 0))
        stability = best_stats.get('stability_ratio', 1.0)
        score_value = get_model_selection_score(best_stats)

        summary_rows.append({
            'symbol': symbol,
            'best_model': best_model,
            'pf_train': pf_train,
            'pf_test': pf_test,
            'profit_factor': pf_test,
            'max_drawdown_percent': maxdd_value,
            'return_pct': return_value,
            'stability': stability,
            'score': score_value
        })

    if not summary_rows:
        return {
            'rows': [],
            'best_models_by_symbol': {},
            'best_by_pf': None,
            'best_by_maxdd': None,
            'best_by_return': None,
            'best_by_score': None
        }

    best_by_pf = max(summary_rows, key=lambda x: x['profit_factor'])
    best_by_maxdd = min(summary_rows, key=lambda x: x['max_drawdown_percent'])
    best_by_return = max(summary_rows, key=lambda x: x['return_pct'])
    best_by_score = max(summary_rows, key=lambda x: x['score'])

    return {
        'rows': summary_rows,
        'best_models_by_symbol': best_models_by_symbol,
        'best_by_pf': best_by_pf,
        'best_by_maxdd': best_by_maxdd,
        'best_by_return': best_by_return,
        'best_by_score': best_by_score
    }


def print_comparison_table(aggregated: dict[str, Any], use_robustness: bool = False):
    """Сравнительная таблица лучших моделей по инструментам"""
    rows = aggregated.get('rows', [])
    if not rows:
        logger.info(colorize("\n❌ Нет валидных результатов для сравнения", 'red', bright=True))
        return

    logger.info("\n" + colorize("=" * 120, 'magenta', bright=True))
    logger.info(colorize("СРАВНЕНИЕ ИНСТРУМЕНТОВ (ЛУЧШАЯ МОДЕЛЬ НА КАЖДЫЙ ИНСТРУМЕНТ)", 'magenta', bright=True))
    logger.info(colorize("=" * 120, 'magenta', bright=True))
    score_label = 'Robust' if use_robustness else 'Score'
    header = (
        f"{'Symbol':<10} {'Best Model':<15} {'PF':>8} {'PF_test':>8} "
        f"{'DD%':>8} {'Stability':>10} {score_label:>10}"
    )
    logger.info(header)
    logger.info("-" * 120)

    for row in sorted(rows, key=lambda x: x['score'], reverse=True):
        logger.info(
            f"{row['symbol']:<10} "
            f"{row['best_model']:<15} "
            f"{row.get('pf_train', row['profit_factor']):>8.2f} "
            f"{row['profit_factor']:>8.2f} "
            f"{row['max_drawdown_percent']:>8.2f} "
            f"{row.get('stability', 1.0):>10.2f} "
            f"{row['score']:>10.2f}"
        )
    logger.info(colorize("=" * 120, 'magenta', bright=True))

    best_pf = aggregated['best_by_pf']
    best_maxdd = aggregated['best_by_maxdd']
    best_return = aggregated['best_by_return']
    best_score = aggregated['best_by_score']

    logger.info(colorize(f"\n🏆 Лучший по Profit Factor: {best_pf['symbol']} ({best_pf['profit_factor']:.2f})", 'green', bright=True))
    logger.info(colorize(f"🏆 Лучший по MaxDD:         {best_maxdd['symbol']} ({best_maxdd['max_drawdown_percent']:.2f}%)", 'green', bright=True))
    logger.info(colorize(f"🏆 Лучший по Stability:     {best_score['symbol']} ({best_score.get('stability', 1.0):.2f})", 'green', bright=True))
    logger.info(colorize(f"🏆 Лучший по Return:        {best_return['symbol']} ({best_return['return_pct']:.2f}%)", 'green', bright=True))
    logger.info(colorize(f"\n🏆 Лучший инструмент: {best_score['symbol']}", 'green', bright=True))
    logger.info(colorize(f"🏆 Лучшая модель: {best_score['best_model']}", 'green', bright=True))
    logger.info(colorize(f"Score: {best_score['score']:.2f}", 'green', bright=True))


def run_optimization(df, deposit, ticker, signal_kwargs: dict | None = None, debug_filters: bool = False):
    """Запуск оптимизации - сравнение всех моделей"""
    print_separator(color='cyan')
    logger.info(colorize(f"🔄 ОПТИМИЗАЦИЯ МОДЕЛЕЙ ДЛЯ {ticker}", 'cyan', bright=True))
    print_separator(color='cyan')
    logger.info("\nЗапуск бэктеста для всех моделей...\n")

    results = []

    for model_name in MODELS.keys():
        logger.info(f"  Тестирование модели: {model_name}...")
        model = get_model(model_name)

        backtest_result = run_backtest(
            df=df,
            signal_generator=generate_signal,
            deposit=deposit,
            model=model,
            lookback_window=300,
            max_holding_candles=50,
            signal_kwargs=signal_kwargs,
            debug_filters=debug_filters
        )

        results.append(backtest_result)
        logger.info(f"✓ ({backtest_result.total_trades} сделок)")
        if debug_filters:
            print_filter_debug_stats(backtest_result.to_dict().get('filter_stats'))

    # Вывод сравнительной таблицы
    logger.info("\n" + compare_models_results(results))

    # Детальные результаты по каждой модели
    logger.info("\n" + colorize("="*80, 'magenta', bright=True))
    logger.info(colorize("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО МОДЕЛЯМ", 'magenta', bright=True))
    logger.info(colorize("="*80, 'magenta', bright=True) + "\n")

    for result in sorted(results, key=lambda x: x.expectancy, reverse=True):
        if result.total_trades > 0:
            print_backtest_report(result.to_dict(), show_details=False)


def _frange(start: float, stop: float, step: float) -> list[float]:
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 2))
        current += step
    return values


def run_grid_search_for_symbol(
    exchange: str,
    symbol: str,
    deposit: float,
    timeframe: str,
    board: str,
    start_date: Optional[str],
    end_date: Optional[str],
    adapter,
    base_model_name: str,
    signal_kwargs: Optional[dict],
    walk_forward: bool,
    monte_carlo_simulations: int,
    limit: Optional[int] = None,
    adaptive_regime: bool = False,
    filter_config: Optional[dict] = None,
) -> dict[str, Any]:
    """Grid search по RR/Volume/ATR для оценки устойчивых зон."""
    base_model = get_model(base_model_name)
    if filter_config:
        base_model = apply_filters_to_model(base_model, filter_config)

    data_result = _load_cli_dataset(
        exchange=exchange,
        ticker=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        board=board,
        adapter=adapter,
        limit=limit
    )
    df = data_result.df

    if df.empty:
        return {'symbol': symbol, 'entries': [], 'error': f'Нет данных для {symbol}'}

    rr_values = _frange(1.5, 2.5, 0.25)
    vol_values = _frange(1.0, 1.5, 0.1)
    atr_values = _frange(0.8, 1.5, 0.1)

    entries = []
    symbol_signal_kwargs = dict(signal_kwargs or {})
    spec = _resolve_contract_spec(
        exchange=exchange,
        adapter=adapter,
        symbol=symbol,
        board=board,
    )
    try:
        contract_margin_rub = float(spec.get("initial_margin"))
    except (TypeError, ValueError):
        contract_margin_rub = None
    try:
        contract_multiplier = float(spec.get("contract_multiplier"))
    except (TypeError, ValueError):
        contract_multiplier = None
    if contract_margin_rub is not None and contract_margin_rub > 0:
        symbol_signal_kwargs["contract_margin_rub"] = contract_margin_rub
    if contract_multiplier is not None and contract_multiplier > 0:
        symbol_signal_kwargs["contract_multiplier"] = contract_multiplier

    for rr in rr_values:
        for vol in vol_values:
            for atr in atr_values:
                tuned_model = replace(
                    base_model,
                    min_rr=rr,
                    min_volume_ratio=vol,
                    atr_multiplier_stop=atr
                )

                stats, _ = evaluate_model(
                    df=df,
                    deposit=deposit,
                    model=tuned_model,
                    signal_kwargs=symbol_signal_kwargs,
                    walk_forward=walk_forward,
                    monte_carlo_simulations=monte_carlo_simulations,
                    debug_filters=False,
                    adaptive_regime=adaptive_regime
                )
                score = stats.get('robustness_score', stats.get('score', 0))
                stability = stats.get('stability_ratio', stats.get('profit_factor', 0))

                entries.append({
                    'rr': rr,
                    'volume': vol,
                    'atr': atr,
                    'score': round(score, 4),
                    'stability': round(stability, 4)
                })

    entries_sorted = sorted(entries, key=lambda x: x['score'], reverse=True)
    top_n = max(1, int(len(entries_sorted) * 0.2))
    top_zone = entries_sorted[:top_n]
    return {'symbol': symbol, 'entries': entries_sorted, 'top_zone': top_zone, 'error': None}


def print_grid_search_heatmap(grid_result: dict[str, Any], walk_forward: bool):
    """Текстовый heatmap лучших зон устойчивости."""
    if grid_result.get('error'):
        logger.info(colorize(f"\n❌ Grid search: {grid_result['error']}", 'red', bright=True))
        return

    entries = grid_result.get('entries', [])
    top_zone = grid_result.get('top_zone', [])
    if not entries:
        logger.info(colorize("\n❌ Grid search не дал результатов", 'red', bright=True))
        return

    score_label = 'RobustScore' if walk_forward else 'Score'

    logger.info("\n" + colorize("=" * 100, 'magenta', bright=True))
    logger.info(colorize(f"GRID SEARCH HEATMAP ({grid_result['symbol']}) - лучшие зоны устойчивости", 'magenta', bright=True))
    logger.info(colorize("=" * 100, 'magenta', bright=True))
    logger.info(f"Top-zone size: {len(top_zone)}/{len(entries)}")

    logger.info(f"\nЛучшие комбинации ({score_label}):")
    logger.info(f"{'RR':>6} {'Vol':>6} {'ATR':>6} {score_label:>12}")
    logger.info("-" * 40)
    for row in top_zone[:15]:
        logger.info(f"{row['rr']:>6.2f} {row['volume']:>6.2f} {row['atr']:>6.2f} {row['score']:>12.2f}")

    rr_values = sorted({x['rr'] for x in entries})
    vol_values = sorted({x['volume'] for x in entries})
    atr_values = sorted({x['atr'] for x in entries})

    logger.info("\nHeatmap RR x Volume (mean score by ATR):")
    header = "RR\\Vol  " + " ".join(f"{v:>7.2f}" for v in vol_values)
    logger.info(header)
    logger.info("-" * len(header))
    for rr in rr_values:
        row = [f"{rr:>6.2f}"]
        for vol in vol_values:
            values = [x['score'] for x in entries if x['rr'] == rr and x['volume'] == vol]
            cell = sum(values) / len(values) if values else 0
            row.append(f"{cell:>7.2f}")
        logger.info(" ".join(row))

    logger.info("\nHeatmap RR x ATR (mean score by Volume):")
    header = "RR\\ATR  " + " ".join(f"{a:>7.2f}" for a in atr_values)
    logger.info(header)
    logger.info("-" * len(header))
    for rr in rr_values:
        row = [f"{rr:>6.2f}"]
        for atr in atr_values:
            values = [x['score'] for x in entries if x['rr'] == rr and x['atr'] == atr]
            cell = sum(values) / len(values) if values else 0
            row.append(f"{cell:>7.2f}")
        logger.info(" ".join(row))
    logger.info(colorize("=" * 100, 'magenta', bright=True))


def main():
    parser = argparse.ArgumentParser(
        description='Профессиональная торговая система с моделями',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Использование модели conservative
  python -m ports.cli.trading_cli CCH6 -d 100000 -e futures -m forts --model conservative

  # Использование модели scalp
  python -m ports.cli.trading_cli SBER -d 500000 --model scalp

  # Оптимизация - сравнение всех моделей
  python -m ports.cli.trading_cli CCH6 -d 100000 -e futures -m forts --optimize

  # Оптимизация по нескольким инструментам
  python -m ports.cli.trading_cli --symbols CCH6,RIH6,SiH6 -d 100000 -e futures -m forts --optimize

  # Показать доступные модели
  python -m ports.cli.trading_cli --list-models

  # JSON вывод
  python -m ports.cli.trading_cli SBER -d 100000 --model high_rr --json

  # Диагностика фильтров и адаптивный volume
  python -m ports.cli.trading_cli SBER -d 100000 --backtest --debug-filters --volume-mode adaptive

  # Walk-forward + Monte Carlo
  python -m ports.cli.trading_cli CCH6 -d 100000 -e futures -m forts --optimize --walk-forward --monte-carlo 1000

  # Grid search устойчивых зон
  python -m ports.cli.trading_cli CCH6 -d 100000 -e futures -m forts --optimize --walk-forward --grid-search
        """
    )

    parser.add_argument('ticker', nargs='?', type=str, help='Тикер инструмента')
    parser.add_argument('--symbols', type=str, default=None,
                        help='Список тикеров через запятую (например: CCH6,RIH6,SiH6). Если указан, ticker игнорируется')
    parser.add_argument('--deposit', '-d', type=float,
                        help='Размер депозита в рублях')
    parser.add_argument('--timeframe', '-t', type=str, default='10m',
                        help='Таймфрейм (по умолчанию: 10m)')
    parser.add_argument('--exchange', type=str, default='moex', choices=SUPPORTED_EXCHANGES,
                        help='Биржа/источник данных (по умолчанию: moex)')
    parser.add_argument('--engine', '-e', type=str, default='stock',
                        help='Движок: stock, futures (по умолчанию: stock)')
    parser.add_argument('--market', '-m', type=str, default='shares',
                        help='Рынок: shares, forts (по умолчанию: shares)')
    parser.add_argument('--board', '-b', type=str, default=None,
                        help='Режим торгов')
    parser.add_argument('--model', type=str, default='balanced',
                        help='Торговая модель: conservative, high_rr, aggressive, scalp, balanced (по умолчанию: balanced)')
    parser.add_argument('--backtest', action='store_true',
                        help='Запустить бэктест стратегии')
    parser.add_argument('--optimize', action='store_true',
                        help='Оптимизация - сравнить все модели')
    parser.add_argument('--no-signal', action='store_true',
                        help='Не показывать текущий сигнал')
    parser.add_argument('--json', action='store_true',
                        help='Вывод в формате JSON')
    parser.add_argument('--no-color', action='store_true',
                        help='Отключить цветной вывод CLI')
    parser.add_argument('--config', type=str, default=None,
                        help='Путь к YAML-конфигу фильтров стратегии (по умолчанию: strict.yaml, если есть)')
    parser.add_argument('--list-models', action='store_true',
                        help='Показать список доступных моделей')
    parser.add_argument('--compare-models', action='store_true',
                        help='Показать сравнительную таблицу моделей')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Дата начала (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='Дата окончания (YYYY-MM-DD)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Ограничение числа свечей (по умолчанию: без ограничения)')
    parser.add_argument('--debug-filters', action='store_true',
                        help='Показать диагностику фильтрации сигналов в бэктесте')
    parser.add_argument('--volume-mode', choices=['fixed', 'adaptive'], default=None,
                        help='Режим volume-фильтра: fixed | adaptive (по умолчанию из config)')
    parser.add_argument('--structure-mode', choices=['strict', 'simple'], default='strict',
                        help='Режим структуры: strict (HH/HL) | simple (MA50 vs MA200)')
    parser.add_argument('--disable-rr', action='store_true',
                        help='Отключить RR фильтр')
    parser.add_argument('--disable-volume', action='store_true',
                        help='Отключить volume фильтр')
    parser.add_argument('--disable-trend', action='store_true',
                        help='Отключить trend filter')
    parser.add_argument('--walk-forward', action='store_true',
                        help='Walk-forward тестирование: Train 70%% / Test 30%%')
    parser.add_argument('--monte-carlo', type=int, default=0,
                        help='Количество Monte Carlo симуляций (например: 1000)')
    parser.add_argument('--grid-search', action='store_true',
                        help='Перебор RR/Volume/ATR и вывод тепловых зон устойчивости')
    parser.add_argument('--adaptive-regime', action='store_true',
                        help='Адаптивная режимная стратегия (trend/range/high_volatility)')
    parser.add_argument('--regime-pf-window', type=int, default=20,
                        help='N последних сделок для контроля PF по режиму (по умолчанию: 20)')
    parser.add_argument('--generate-setups', action='store_true',
                        help='Сгенерировать структурированные торговые сетапы')
    parser.add_argument('--export-json', type=str, default=None,
                        help='Путь для экспорта/чтения JSON сетапов (например setups.json)')
    parser.add_argument('--monitor', action='store_true',
                        help='Мониторинг сетапов: entry/stop/tp/expiry с уведомлениями')
    parser.add_argument('--monitor-interval', type=int, default=15,
                        help='Интервал мониторинга в секундах (по умолчанию: 15)')
    parser.add_argument('--monitor-workers', type=int, default=10,
                        help='Количество параллельных воркеров для опроса цен (по умолчанию: 10)')
    parser.add_argument('--monitor-cycles', type=int, default=0,
                        help='Лимит циклов мониторинга, 0 = без лимита')
    parser.add_argument('--setup-expiry-candles', type=int, default=5,
                        help='Срок действия сетапа в свечах (по умолчанию: 5)')

    args = parser.parse_args()
    setup_cli_colors(enabled=(not args.json and not args.no_color))

    try:
        filter_config = load_config(args.config)
    except ValueError as e:
        parser.error(str(e))

    if args.grid_search and not args.optimize:
        parser.error("--grid-search requires --optimize")
    if args.monte_carlo < 0:
        parser.error("--monte-carlo must be >= 0")
    if args.walk_forward and 0 < args.monte_carlo < 1000 and not args.json:
        logger.info(colorize("⚠ Для устойчивой оценки рекомендуется --monte-carlo >= 1000", 'yellow', bright=True))
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be > 0")
    if args.regime_pf_window <= 0:
        parser.error("--regime-pf-window must be > 0")
    if args.monitor_interval <= 0:
        parser.error("--monitor-interval must be > 0")
    if args.monitor_workers <= 0:
        parser.error("--monitor-workers must be > 0")
    if args.monitor_cycles < 0:
        parser.error("--monitor-cycles must be >= 0")
    if args.setup_expiry_candles <= 0:
        parser.error("--setup-expiry-candles must be > 0")
    if args.generate_setups and (args.optimize or args.backtest):
        parser.error("--generate-setups нельзя комбинировать с --optimize/--backtest")

    # Список моделей
    if args.list_models:
        print_separator()
        logger.info("ДОСТУПНЫЕ ТОРГОВЫЕ МОДЕЛИ")
        print_separator()
        for name, model in MODELS.items():
            logger.info(f"\n{name:15} - {model.description}")
        print_separator()
        return

    # Сравнение моделей
    if args.compare_models:
        logger.info(compare_models())
        return

    if not args.deposit and not args.generate_setups and not args.monitor:
        parser.error("--deposit/-d is required")
    if args.grid_search:
        try:
            get_model(args.model)
        except ValueError as e:
            logger.info(colorize(f"❌ Ошибка: {e}", 'red', bright=True))
            return

    symbols = parse_symbols(args.symbols)

    # Определяем board
    board = args.board
    if board is None:
        board = _default_board(args.exchange, args.engine)

    try:
        adapter = _build_exchange_adapter(args.exchange, args.engine, args.market)
    except NotImplementedError as exc:
        parser.error(str(exc))

    cfg_filters = filter_config["filters"]
    volume_cfg = cfg_filters["volume"]
    trend_cfg = cfg_filters["trend"]
    rsi_cfg = cfg_filters["rsi"]
    atr_cfg = cfg_filters["atr"]
    regime_cfg = cfg_filters["regime"]
    risk_cfg = cfg_filters["risk"]
    costs_cfg = cfg_filters["costs"]
    scoring_cfg = cfg_filters.get("scoring", {})

    effective_volume_mode = args.volume_mode or volume_cfg["mode"]
    effective_disable_volume = args.disable_volume or (not volume_cfg["enabled"])
    effective_disable_trend = args.disable_trend or (not trend_cfg["enabled"])

    signal_kwargs = {
        'volume_mode': effective_volume_mode,
        'structure_mode': args.structure_mode,
        'disable_rr': args.disable_rr,
        'disable_volume': effective_disable_volume,
        'disable_trend': effective_disable_trend,
        'debug_filters': args.debug_filters,
        'regime_pf_window': args.regime_pf_window,
        'rsi_enabled': bool(rsi_cfg["enabled"]),
        'rsi_trend_confirmation_only': bool(rsi_cfg["trend_confirmation_only"]),
        'atr_enabled': bool(atr_cfg["enabled"]),
        'atr_min_percentile': int(atr_cfg["min_percentile"]),
        'max_daily_loss_pct': float(risk_cfg["max_daily_loss_pct"]),
        'equity_dd_stop_pct': float(risk_cfg["equity_dd_stop_pct"]),
        'partial_take_profit': bool(risk_cfg["partial_take_profit"]),
        'partial_tp_rr': float(risk_cfg["partial_tp_rr"]),
        'partial_close_fraction': float(risk_cfg["partial_close_fraction"]),
        'trail_after_partial': bool(risk_cfg["trail_after_partial"]),
        'trailing_atr_multiplier': float(risk_cfg["trailing_atr_multiplier"]),
        'sl_tp_only': True,
        'regime_thresholds': {
            'trend': float(regime_cfg["trend_pf_threshold"]),
            'range': float(regime_cfg["range_pf_threshold"]),
            'high_volatility': float(regime_cfg["high_vol_pf_threshold"]),
        } if regime_cfg.get("enabled", True) else None,
        'regime_enabled_flags': {
            'trend': bool(regime_cfg["trade_trend"]),
            'range': bool(regime_cfg["trade_range"]),
            'high_volatility': bool(regime_cfg["trade_high_vol"]),
        } if regime_cfg.get("enabled", True) else None,
        'min_regime_pf': float(regime_cfg["min_regime_pf"]),
        'filter_config': filter_config,
        'engine': args.engine,
        'costs_enabled': bool(costs_cfg["enabled"]),
        'futures_per_contract_rub': float(costs_cfg["futures_per_contract_rub"]),
        'futures_fee_mode': str(costs_cfg["futures_fee_mode"]),
        'securities_bps': float(costs_cfg["securities_bps"]),
        'settlement_bps': float(costs_cfg["settlement_bps"]),
        'slippage_bps': float(costs_cfg["slippage_bps"]),
        'min_expected_trades_per_month': float(scoring_cfg.get("min_expected_trades_per_month", 0.0)),
    }
    report_contract_multiplier = 1.0
    if args.ticker:
        spec = _resolve_contract_spec(
            exchange=args.exchange,
            adapter=adapter,
            symbol=args.ticker,
            board=board,
        )
        raw_margin = spec.get("initial_margin")
        raw_multiplier = spec.get("contract_multiplier")
        try:
            contract_margin_rub = float(raw_margin)
        except (TypeError, ValueError):
            contract_margin_rub = None
        try:
            resolved_multiplier = float(raw_multiplier)
        except (TypeError, ValueError):
            resolved_multiplier = None
        if contract_margin_rub is not None and contract_margin_rub > 0:
            signal_kwargs["contract_margin_rub"] = contract_margin_rub
        if resolved_multiplier is not None and resolved_multiplier > 0:
            report_contract_multiplier = resolved_multiplier
            signal_kwargs["contract_multiplier"] = resolved_multiplier
    signal_cost_config = {
        "enabled": bool(costs_cfg["enabled"]),
        "engine": args.engine,
        "futures_per_contract_rub": float(costs_cfg["futures_per_contract_rub"]),
        "futures_fee_mode": str(costs_cfg["futures_fee_mode"]),
        "securities_bps": float(costs_cfg["securities_bps"]),
        "settlement_bps": float(costs_cfg["settlement_bps"]),
    }

    if args.monitor and not args.generate_setups:
        if not args.export_json:
            parser.error("--monitor без --generate-setups требует --export-json <file>")

        setups_from_file = load_setups_json(args.export_json)
        if not setups_from_file:
            logger.info(colorize(f"❌ В файле {args.export_json} нет активных сетапов", 'red', bright=True))
            return

        if not args.json:
            logger.info(colorize(f"🔎 Загружено сетапов из {args.export_json}: {len(setups_from_file)}", 'cyan', bright=True))

        remaining = _monitor_setups(
            exchange=args.exchange,
            setups=setups_from_file,
            adapter=adapter,
            board=board,
            interval_seconds=args.monitor_interval,
            export_path=args.export_json,
            sender=AlertSender(enabled_console=not args.json),
            max_cycles=args.monitor_cycles,
            filter_config=filter_config,
            monitor_workers=args.monitor_workers,
        )
        if args.json:
            logger.info(json.dumps({"remaining_setups": remaining}, ensure_ascii=False, indent=2))
        return

    if args.generate_setups:
        target_symbols = symbols if symbols else ([args.ticker] if args.ticker else [])
        if not target_symbols:
            parser.error("ticker is required (or use --symbols) for --generate-setups")

        try:
            model = get_model(args.model)
        except ValueError as e:
            logger.info(colorize(f"❌ Ошибка: {e}", 'red', bright=True))
            return

        generated_setups: list[TradeSetup] = []
        rejected: dict[str, list[str]] = {}
        data_warnings: dict[str, list[str]] = {}

        for symbol in target_symbols:
            try:
                data_result = _load_cli_dataset(
                    exchange=args.exchange,
                    ticker=symbol,
                    timeframe=args.timeframe,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    board=board,
                    adapter=adapter,
                    limit=args.limit,
                )
            except Exception as exc:
                rejected[symbol] = [f"Ошибка загрузки данных: {exc}"]
                continue
            df = data_result.df
            if data_result.warnings:
                data_warnings[symbol] = list(data_result.warnings)

            if df.empty:
                rejected[symbol] = [f"Нет данных для {symbol}"]
                continue

            model_for_setup = model
            if args.adaptive_regime:
                regime = classify_current_regime(df)
                model_for_setup = apply_regime_risk(build_regime_model(regime), regime)
            if filter_config:
                setup_regime = classify_current_regime(df)
                model_for_setup = apply_filters_to_model(model_for_setup, filter_config, regime=setup_regime)

            generation = generate_trade_setup(
                symbol=symbol,
                df=df,
                model=model_for_setup,
                timeframe=args.timeframe,
                board=board,
                volume_mode=effective_volume_mode,
                structure_mode=args.structure_mode,
                expiry_candles=args.setup_expiry_candles,
                disable_rr=args.disable_rr,
                disable_volume=effective_disable_volume,
                disable_trend=effective_disable_trend,
                rsi_enabled=signal_kwargs['rsi_enabled'],
                rsi_trend_confirmation_only=signal_kwargs['rsi_trend_confirmation_only'],
                atr_enabled=signal_kwargs['atr_enabled'],
                atr_min_percentile=signal_kwargs['atr_min_percentile'],
            )
            if generation.setup is None:
                rejected[symbol] = generation.reasons or ["Условия сетапа не выполнены"]
                continue

            generated_setups.append(generation.setup)

        setup_dicts = [s.to_dict() for s in generated_setups]

        if args.export_json:
            export_setups_json(args.export_json, setup_dicts)
            if not args.json:
                logger.info(colorize(f"💾 Сетапы сохранены: {args.export_json}", 'cyan', bright=True))

        if args.json:
            logger.info(
                json.dumps(
                    {
                        "setups": setup_dicts,
                        "rejected": rejected,
                        "warnings": data_warnings,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            if setup_dicts:
                for setup in setup_dicts:
                    print_setup_detected(setup)
            else:
                logger.info(colorize("⚠ Валидные сетапы не найдены", 'yellow', bright=True))

            if rejected:
                logger.info(colorize("\nПричины отклонения:", 'yellow', bright=True))
                for symbol, reasons in rejected.items():
                    logger.info(colorize(f"  {symbol}:", 'yellow', bright=True))
                    for reason in reasons:
                        logger.info(colorize(f"    - {reason}", 'yellow'))

            if data_warnings:
                logger.info(colorize("\nПредупреждения по данным:", 'yellow', bright=True))
                for symbol, warnings in data_warnings.items():
                    for warning in warnings:
                        logger.info(colorize(f"  {symbol}: {warning}", 'yellow'))

        if args.monitor and setup_dicts:
            _monitor_setups(
                exchange=args.exchange,
                setups=setup_dicts,
                adapter=adapter,
                board=board,
                interval_seconds=args.monitor_interval,
                export_path=args.export_json,
                sender=AlertSender(enabled_console=not args.json),
                max_cycles=args.monitor_cycles,
                filter_config=filter_config,
                monitor_workers=args.monitor_workers,
            )
        return

    if not args.json and (args.disable_rr or effective_disable_volume or effective_disable_trend):
        logger.info(colorize("⚙️  Отключенные фильтры:", 'yellow', bright=True))
        if args.disable_rr:
            logger.info("   • RR")
        if effective_disable_volume:
            logger.info("   • Volume")
        if effective_disable_trend:
            logger.info("   • Trend")
        logger.info()

    # Оптимизация по нескольким инструментам
    if symbols:
        if not args.optimize:
            parser.error("--symbols поддерживается только в режиме --optimize")

        results: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        for symbol in symbols:
            symbol_result = run_backtest_for_symbol(
                exchange=args.exchange,
                symbol=symbol,
                deposit=args.deposit,
                timeframe=args.timeframe,
                board=board,
                start_date=args.start_date,
                end_date=args.end_date,
                adapter=adapter,
                signal_kwargs=signal_kwargs,
                debug_filters=args.debug_filters,
                verbose=not args.json,
                walk_forward=args.walk_forward,
                monte_carlo_simulations=args.monte_carlo,
                limit=args.limit,
                adaptive_regime=args.adaptive_regime,
                filter_config=filter_config,
            )

            if symbol_result['error']:
                errors[symbol] = symbol_result['error']
                continue

            # Требуемая структура: results[symbol] = { ...model_stats... }
            results[symbol] = symbol_result['models']

            if not args.json and symbol_result.get('results'):
                if args.walk_forward:
                    print_walk_forward_table(symbol_result['models'])
                else:
                    logger.info("\n" + compare_models_results(symbol_result['results']))

        aggregated = aggregate_results(results)

        if args.json:
            logger.info(
                json.dumps(
                    {
                        'results': results,
                        'errors': errors,
                        'summary': aggregated
                    },
                    indent=2,
                    ensure_ascii=False
                )
            )
        else:
            print_comparison_table(aggregated, use_robustness=args.walk_forward)
            if args.walk_forward:
                print_cross_instrument_stability(results)
            if errors:
                logger.info(colorize("\n⚠️  Символы с ошибками:", 'yellow', bright=True))
                for symbol, error in errors.items():
                    logger.info(colorize(f"   {symbol}: {error}", 'yellow'))

            if args.grid_search and results:
                logger.info("\nЗапуск grid-search по каждому инструменту...")
                for symbol in results.keys():
                    grid_result = run_grid_search_for_symbol(
                        exchange=args.exchange,
                        symbol=symbol,
                        deposit=args.deposit,
                        timeframe=args.timeframe,
                        board=board,
                        start_date=args.start_date,
                        end_date=args.end_date,
                        adapter=adapter,
                        base_model_name=args.model,
                        signal_kwargs=signal_kwargs,
                        walk_forward=args.walk_forward,
                        monte_carlo_simulations=args.monte_carlo,
                        limit=args.limit,
                        adaptive_regime=args.adaptive_regime,
                        filter_config=filter_config,
                    )
                    print_grid_search_heatmap(grid_result, walk_forward=args.walk_forward)
        return

    # Одиночный тикер обязателен, если --symbols не указан
    if not args.ticker:
        parser.error("ticker is required (or use --symbols)")

    # Оптимизация по одному инструменту
    if args.optimize:
        symbol_result = run_backtest_for_symbol(
            exchange=args.exchange,
            symbol=args.ticker,
            deposit=args.deposit,
            timeframe=args.timeframe,
            board=board,
            start_date=args.start_date,
            end_date=args.end_date,
            adapter=adapter,
            signal_kwargs=signal_kwargs,
            debug_filters=args.debug_filters,
            verbose=not args.json,
            walk_forward=args.walk_forward,
            monte_carlo_simulations=args.monte_carlo,
            limit=args.limit,
            adaptive_regime=args.adaptive_regime,
            filter_config=filter_config,
        )

        if symbol_result['error']:
            return

        if args.json:
            single_results = {args.ticker: symbol_result['models']}
            logger.info(
                json.dumps(
                    {
                        'results': single_results,
                        'summary': aggregate_results(single_results)
                    },
                    indent=2,
                    ensure_ascii=False
                )
            )
        else:
            if args.walk_forward:
                print_walk_forward_table(symbol_result['models'])
            else:
                logger.info("\n" + compare_models_results(symbol_result['results']))
                logger.info("\n" + colorize("=" * 80, 'magenta', bright=True))
                logger.info(colorize("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО МОДЕЛЯМ", 'magenta', bright=True))
                logger.info(colorize("=" * 80, 'magenta', bright=True) + "\n")
                for result in sorted(symbol_result['results'], key=lambda x: x.expectancy, reverse=True):
                    if result.total_trades > 0:
                        print_backtest_report(result.to_dict(), show_details=False)

            if args.grid_search:
                grid_result = run_grid_search_for_symbol(
                    exchange=args.exchange,
                    symbol=args.ticker,
                    deposit=args.deposit,
                    timeframe=args.timeframe,
                    board=board,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    adapter=adapter,
                    base_model_name=args.model,
                    signal_kwargs=signal_kwargs,
                    walk_forward=args.walk_forward,
                    monte_carlo_simulations=args.monte_carlo,
                    limit=args.limit,
                    adaptive_regime=args.adaptive_regime,
                    filter_config=filter_config,
                )
                print_grid_search_heatmap(grid_result, walk_forward=args.walk_forward)
        return

    # Получаем модель (для single signal/backtest режима)
    try:
        model = get_model(args.model)
    except ValueError as e:
        logger.info(colorize(f"❌ Ошибка: {e}", 'red', bright=True))
        return
    model = apply_filters_to_model(model, filter_config)

    # Загружаем данные
    if not args.json:
        print_separator(color='cyan')
        logger.info(colorize(f"📥 Загрузка данных для {args.ticker}", 'cyan', bright=True))
        logger.info(f"   Таймфрейм: {args.timeframe}")
        logger.info(f"   Движок: {args.engine}, Рынок: {args.market}, Режим: {board}")
        print_separator(color='cyan')

    try:
        data_result = _load_cli_dataset(
            exchange=args.exchange,
            ticker=args.ticker,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            board=board,
            adapter=adapter,
            limit=args.limit
        )
    except Exception as exc:
        logger.info(colorize(f"\n❌ Ошибка загрузки данных: {exc}", 'red', bright=True))
        return
    df = data_result.df

    if df.empty:
        logger.info(colorize(f"\n❌ Нет данных для {args.ticker}", 'red', bright=True))
        return

    if not args.json:
        logger.info(colorize(f"✅ Загружено {len(df)} свечей", 'green', bright=True))
        logger.info(f"   Период: {df.index[0]} - {df.index[-1]}\n")
        for warning in data_result.warnings:
            logger.info(warning)
        if data_result.warnings:
            logger.info()

    # Показываем информацию о модели
    if not args.json and not args.no_signal:
        print_model_info(model)

    # Генерация сигнала
    signal = None
    if not args.no_signal:
        if args.adaptive_regime:
            adaptive_engine = AdaptiveSignalEngine(
                volume_mode=effective_volume_mode,
                structure_mode=args.structure_mode,
                disable_rr=args.disable_rr,
                disable_volume=effective_disable_volume,
                disable_trend=effective_disable_trend,
                debug_filters=args.debug_filters,
                rsi_enabled=signal_kwargs['rsi_enabled'],
                rsi_trend_confirmation_only=signal_kwargs['rsi_trend_confirmation_only'],
                atr_enabled=signal_kwargs['atr_enabled'],
                atr_min_percentile=signal_kwargs['atr_min_percentile'],
                pf_window=args.regime_pf_window,
                disable_threshold_pf=signal_kwargs['min_regime_pf'],
                regime_thresholds=signal_kwargs['regime_thresholds'],
                regime_enabled_flags=signal_kwargs['regime_enabled_flags'],
                filter_config=filter_config,
                contract_margin_rub=signal_kwargs.get('contract_margin_rub'),
                contract_multiplier=signal_kwargs.get('contract_multiplier', 1.0),
            )
            signal = adaptive_engine.generate_signal(
                df=df,
                deposit=args.deposit,
                model=model
            )
        else:
            signal = generate_signal(
                df=df,
                deposit=args.deposit,
                model=model,
                volume_mode=effective_volume_mode,
                structure_mode=args.structure_mode,
                disable_rr=args.disable_rr,
                disable_volume=effective_disable_volume,
                disable_trend=effective_disable_trend,
                debug_filters=args.debug_filters,
                rsi_enabled=signal_kwargs['rsi_enabled'],
                rsi_trend_confirmation_only=signal_kwargs['rsi_trend_confirmation_only'],
                atr_enabled=signal_kwargs['atr_enabled'],
                atr_min_percentile=signal_kwargs['atr_min_percentile'],
                filter_config=filter_config,
                contract_margin_rub=signal_kwargs.get('contract_margin_rub'),
                contract_multiplier=signal_kwargs.get('contract_multiplier', 1.0),
                min_expected_trades_per_month=signal_kwargs['min_expected_trades_per_month'],
            )
        if signal is not None:
            regime_value = str(getattr(signal, "market_regime", "") or "").strip().lower()
            if regime_value in {"", "unknown"}:
                signal.market_regime = classify_current_regime(df)

    # Бэктест
    backtest_result = None
    walk_forward_result = None
    eval_stats = None
    if args.backtest:
        if not args.json:
            logger.info(colorize("🔄 Запуск бэктеста...\n", 'cyan', bright=True))

        eval_stats, raw_eval = evaluate_model(
            df=df,
            deposit=args.deposit,
            model=model,
            signal_kwargs=signal_kwargs,
            walk_forward=args.walk_forward,
            monte_carlo_simulations=args.monte_carlo,
            debug_filters=args.debug_filters,
            adaptive_regime=args.adaptive_regime
        )
        if args.walk_forward:
            walk_forward_result = raw_eval
        else:
            backtest_result = raw_eval

    # Вывод результатов
    if args.json:
        output = {}
        if signal:
            signal_dict = signal.to_dict()
            signal_dict["fee_estimate"] = round(estimate_signal_fee(signal_dict, signal_cost_config), 2)
            output['signal'] = signal_dict
        if backtest_result:
            output['backtest'] = backtest_result.to_dict()
        if walk_forward_result:
            output['walk_forward'] = walk_forward_result.to_dict()
        if eval_stats:
            output['risk_of_ruin'] = round(eval_stats.get('risk_of_ruin', 0), 2)
            output['market_regime_performance'] = eval_stats.get('market_regime_performance', {})
            output['best_regime'] = eval_stats.get('best_regime')
            output['best_regime_pf'] = eval_stats.get('best_regime_pf', 0)
            output['enabled_regimes'] = eval_stats.get('enabled_regimes', [])
            output['disabled_regimes'] = eval_stats.get('disabled_regimes', {})
            output['admission'] = eval_stats.get('admission', {})
            output['edge_found'] = eval_stats.get('edge_found', False)
        logger.info(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        if signal:
            print_signal_report(
                signal.to_dict(),
                cost_config=signal_cost_config,
                contract_multiplier=report_contract_multiplier,
            )
        if backtest_result:
            backtest_dict = backtest_result.to_dict()
            print_backtest_report(backtest_dict)
            if args.debug_filters:
                print_filter_debug_stats(backtest_dict.get('filter_stats'))
        if walk_forward_result:
            wf = walk_forward_result.to_dict()
            logger.info(colorize("\nWalk-Forward Summary:", 'magenta', bright=True))
            logger.info(
                f"Train PF: {wf['train']['profit_factor']:.2f}, "
                f"Test PF: {wf['test']['profit_factor']:.2f}, "
                f"Stability Ratio: {wf['robustness']['stability_ratio']:.2f}, "
                f"RobustScore: {wf['robustness']['robustness_score']:.2f}"
            )
            if wf['robustness']['stability_ratio'] < 0.7:
                logger.info(colorize("⚠️  unstable (Stability ratio < 0.7)", 'yellow'))
            if wf['robustness']['unstable_oos']:
                logger.info(colorize("⚠️  Out-of-sample unstable", 'yellow'))
            if wf['robustness']['overfit']:
                logger.info(colorize("⚠️  Overfitting detected", 'yellow'))
            if wf.get('monte_carlo'):
                mc = wf['monte_carlo']
                logger.info(
                    f"MonteCarlo: worst DD {mc['worst_drawdown_percent']:.2f}%, "
                    f"q5 equity {mc['quantile_5_equity']:.2f}, "
                    f"stability {mc['stability_score']:.4f}"
                )
                logger.info(f"📉 Вероятность просадки > 30%: {mc.get('probability_drawdown_over_30', 0):.2f}%")
            if eval_stats:
                logger.info(colorize(f"Risk of ruin: {eval_stats.get('risk_of_ruin', 0):.2f}%", 'cyan'))
                print_admission_report(eval_stats.get('admission'))

        if eval_stats and eval_stats.get('market_regime_performance'):
            regime_stats = eval_stats['market_regime_performance']
            total_regime_trades = int(sum(int(v.get('trades', 0)) for v in regime_stats.values()))
            if total_regime_trades > 0:
                print_regime_performance_summary(
                    regime_stats,
                    overall_pf=eval_stats.get('pf_test', eval_stats.get('profit_factor')),
                    overall_maxdd=eval_stats.get('maxdd_test', eval_stats.get('max_drawdown_percent'))
                )
                print_regime_final_decision(
                    best_regime=eval_stats.get('best_regime'),
                    best_regime_pf=eval_stats.get('best_regime_pf'),
                    enabled_regimes=eval_stats.get('enabled_regimes'),
                    disabled_regimes=eval_stats.get('disabled_regimes'),
                )
            else:
                logger.info(colorize("\nℹ Regime performance summary недоступен: в бэктесте нет сделок.", 'yellow', bright=True))
            print_edge_recommendations(eval_stats.get('admission'))


if __name__ == '__main__':
    main()
