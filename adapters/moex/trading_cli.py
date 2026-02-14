#!/usr/bin/env python
"""
CLI для профессиональной торговой системы с моделями
"""

import argparse
import json
from adapters.moex import load_data_with_indicators
from adapters.moex.moex import MOEXAdapter
from services.strategy_engine import (
    get_model,
    MODELS,
    compare_models,
    generate_signal,
    run_backtest,
    compare_models_results
)


def print_separator(char='=', length=80):
    print(char * length)


def print_model_info(model):
    """Вывод информации о модели"""
    print_separator()
    print(f"📊 АКТИВНАЯ МОДЕЛЬ: {model.name.upper()}")
    print_separator()
    print(f"\n📝 Описание: {model.description}")
    print(f"\n📉 Параметры:")
    print(f"   Min RR:              {model.min_rr}")
    print(f"   Max Risk:            {model.max_risk_percent}%")
    print(f"   Volume filter:       {model.min_volume_ratio}x")
    print(f"   Trend required:      {'Yes' if model.trend_required else 'No'}")
    print(f"   Allow range:         {'Yes' if model.allow_range else 'No'}")
    print(f"   ATR stop multiplier: {model.atr_multiplier_stop}")
    print(f"   Min confidence:      {model.min_confidence}")
    print(f"   Min trend strength:  {model.min_trend_strength}%")
    print_separator()


def print_signal_report(signal_dict: dict):
    """Красивый вывод торгового сигнала"""
    print_separator()
    print("📊 ТОРГОВЫЙ СИГНАЛ")
    print_separator()

    signal_emoji = {
        'long': '🟢 LONG',
        'short': '🔴 SHORT',
        'none': '⚪ НЕТ СИГНАЛА'
    }
    print(f"\nНаправление: {signal_emoji.get(signal_dict['signal'], signal_dict['signal'])}")
    print(f"Уверенность: {signal_dict['confidence'].upper()}")

    if signal_dict['signal'] != 'none':
        print(f"\n💰 ПАРАМЕТРЫ СДЕЛКИ:")
        print(f"   Вход:     {signal_dict['entry']:.2f}")
        print(f"   Стоп:     {signal_dict['stop']:.2f}")
        print(f"   Цель:     {signal_dict['target']:.2f}")
        print(f"   RR:       {signal_dict['rr']:.2f}")

        print(f"\n📈 РИСК-МЕНЕДЖМЕНТ:")
        print(f"   Размер позиции:  {signal_dict['position_size']:.0f} контрактов")
        print(f"   Риск в рублях:   {signal_dict['risk_rub']:.2f} ₽")
        print(f"   Риск в %:        {signal_dict['risk_percent']:.2f}%")
        potential_profit = abs(signal_dict['target'] - signal_dict['entry']) * signal_dict['position_size']
        print(f"   Потенциал:       {potential_profit:.2f} ₽")

    print(f"\n📉 СТРУКТУРА РЫНКА:")
    print(f"   Тренд:           {signal_dict['structure']}")
    print(f"   Фаза:            {signal_dict['phase']}")
    print(f"   ATR:             {signal_dict['atr']:.2f}")

    print(f"\n📊 ИНДИКАТОРЫ:")
    print(f"   RSI:             {signal_dict['rsi']:.1f}")
    print(f"   Расст. до MA50:  {signal_dict['distance_ma50_pct']:+.2f}%")
    print(f"   Расст. до MA200: {signal_dict['distance_ma200_pct']:+.2f}%")
    print(f"   Volume ratio:    {signal_dict['volume_ratio']:.2f}x")

    if signal_dict['warnings']:
        print(f"\n⚠️  ПРЕДУПРЕЖДЕНИЯ:")
        for warning in signal_dict['warnings']:
            print(f"   • {warning}")

    print_separator()


def print_backtest_report(backtest: dict, show_details: bool = True):
    """Красивый вывод результатов бэктеста"""
    print_separator()
    print(f"📈 РЕЗУЛЬТАТЫ БЭКТЕСТА - {backtest['model_name'].upper()}")
    print_separator()

    if backtest['total_trades'] == 0:
        print("\n❌ Сделок не найдено")
        return

    print(f"\n📊 СТАТИСТИКА:")
    print(f"   Всего сделок:     {backtest['total_trades']}")
    print(f"   Прибыльных:       {backtest['winning_trades']} ({backtest['winrate']:.1f}%)")
    print(f"   Убыточных:        {backtest['losing_trades']}")
    print(f"   Средняя длит.:    {backtest['avg_trade_duration']} свечей")

    if show_details:
        initial_deposit = backtest['final_balance'] - backtest['total_profit']
        print(f"\n💰 ФИНАНСЫ:")
        print(f"   Начальный депо:   {initial_deposit:.2f} ₽")
        print(f"   Конечный депо:    {backtest['final_balance']:.2f} ₽")
        print(f"   Чистая прибыль:   {backtest['total_profit']:+.2f} ₽")
        print(f"   Доходность:       {backtest['return_pct']:+.2f}%")

    print(f"\n📈 МЕТРИКИ:")
    print(f"   Средний выигрыш:  {backtest['avg_win']:.2f} ₽")
    print(f"   Средний проигрыш: {backtest['avg_loss']:.2f} ₽")
    print(f"   Лучшая сделка:    {backtest['best_trade']:.2f} ₽")
    print(f"   Худшая сделка:    {backtest['worst_trade']:.2f} ₽")
    print(f"   Expectancy:       {backtest['expectancy']:.2f} ₽")
    print(f"   Profit Factor:    {backtest['profit_factor']:.2f}")
    print(f"   Sharpe Ratio:     {backtest['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown:     {backtest['max_drawdown_percent']:.2f}%")

    if show_details:
        # Оценка системы
        print(f"\n🎯 ОЦЕНКА СИСТЕМЫ:")
        score = 0
        if backtest['winrate'] >= 40:
            score += 1
            print(f"   ✅ Winrate >= 40%")
        else:
            print(f"   ❌ Winrate < 40%")

        if backtest['profit_factor'] >= 1.5:
            score += 1
            print(f"   ✅ Profit Factor >= 1.5")
        else:
            print(f"   ❌ Profit Factor < 1.5")

        if backtest['expectancy'] > 0:
            score += 1
            print(f"   ✅ Expectancy > 0")
        else:
            print(f"   ❌ Expectancy <= 0")

        if backtest['max_drawdown_percent'] < 20:
            score += 1
            print(f"   ✅ Drawdown < 20%")
        else:
            print(f"   ⚠️  Drawdown >= 20%")

        print(f"\n   Итоговая оценка: {score}/4")

        if score >= 3:
            print("   🌟 СИСТЕМА ПЕРСПЕКТИВНА")
        elif score >= 2:
            print("   ⚠️  СИСТЕМА ТРЕБУЕТ ДОРАБОТКИ")
        else:
            print("   ❌ СИСТЕМА НЕ РЕКОМЕНДУЕТСЯ")

    print_separator()


def run_optimization(df, deposit, ticker):
    """Запуск оптимизации - сравнение всех моделей"""
    print_separator()
    print(f"🔄 ОПТИМИЗАЦИЯ МОДЕЛЕЙ ДЛЯ {ticker}")
    print_separator()
    print("\nЗапуск бэктеста для всех моделей...\n")

    results = []

    for model_name in MODELS.keys():
        print(f"  Тестирование модели: {model_name}...", end=' ')
        model = get_model(model_name)

        backtest_result = run_backtest(
            df=df,
            signal_generator=generate_signal,
            deposit=deposit,
            model=model,
            lookback_window=300,
            max_holding_candles=50
        )

        results.append(backtest_result)
        print(f"✓ ({backtest_result.total_trades} сделок)")

    # Вывод сравнительной таблицы
    print("\n" + compare_models_results(results))

    # Детальные результаты по каждой модели
    print("\n" + "="*80)
    print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО МОДЕЛЯМ")
    print("="*80 + "\n")

    for result in sorted(results, key=lambda x: x.expectancy, reverse=True):
        if result.total_trades > 0:
            print_backtest_report(result.to_dict(), show_details=False)


def main():
    parser = argparse.ArgumentParser(
        description='Профессиональная торговая система с моделями',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Использование модели conservative
  python -m adapters.moex.trading_cli CCH6 -d 100000 -e futures -m forts --model conservative

  # Использование модели scalp
  python -m adapters.moex.trading_cli SBER -d 500000 --model scalp

  # Оптимизация - сравнение всех моделей
  python -m adapters.moex.trading_cli CCH6 -d 100000 -e futures -m forts --optimize

  # Показать доступные модели
  python -m adapters.moex.trading_cli --list-models

  # JSON вывод
  python -m adapters.moex.trading_cli SBER -d 100000 --model high_rr --json
        """
    )

    parser.add_argument('ticker', nargs='?', type=str, help='Тикер инструмента')
    parser.add_argument('--deposit', '-d', type=float,
                        help='Размер депозита в рублях')
    parser.add_argument('--timeframe', '-t', type=str, default='10m',
                        help='Таймфрейм (по умолчанию: 10m)')
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
    parser.add_argument('--list-models', action='store_true',
                        help='Показать список доступных моделей')
    parser.add_argument('--compare-models', action='store_true',
                        help='Показать сравнительную таблицу моделей')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Дата начала (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='Дата окончания (YYYY-MM-DD)')

    args = parser.parse_args()

    # Список моделей
    if args.list_models:
        print_separator()
        print("ДОСТУПНЫЕ ТОРГОВЫЕ МОДЕЛИ")
        print_separator()
        for name, model in MODELS.items():
            print(f"\n{name:15} - {model.description}")
        print_separator()
        return

    # Сравнение моделей
    if args.compare_models:
        print(compare_models())
        return

    # Проверка обязательных параметров
    if not args.ticker:
        parser.error("ticker is required")
    if not args.deposit:
        parser.error("--deposit/-d is required")

    # Получаем модель
    try:
        model = get_model(args.model)
    except ValueError as e:
        print(f"❌ Ошибка: {e}")
        return

    # Определяем board
    board = args.board
    if board is None:
        board = 'RFUD' if args.engine == 'futures' else 'TQBR'

    # Загружаем данные
    if not args.json:
        print_separator()
        print(f"📥 Загрузка данных для {args.ticker}")
        print(f"   Таймфрейм: {args.timeframe}")
        print(f"   Движок: {args.engine}, Рынок: {args.market}, Режим: {board}")
        print_separator()

    adapter = MOEXAdapter(engine=args.engine, market=args.market)

    df, volume_stats = load_data_with_indicators(
        ticker=args.ticker,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        board=board,
        ma_periods=[50, 200],
        rsi_period=14,
        adapter=adapter
    )

    if df.empty:
        print(f"\n❌ Нет данных для {args.ticker}")
        return

    if not args.json:
        print(f"✅ Загружено {len(df)} свечей")
        print(f"   Период: {df.index[0]} - {df.index[-1]}\n")

    # Оптимизация
    if args.optimize:
        run_optimization(df, args.deposit, args.ticker)
        return

    # Показываем информацию о модели
    if not args.json and not args.no_signal:
        print_model_info(model)

    # Генерация сигнала
    signal = None
    if not args.no_signal:
        signal = generate_signal(
            df=df,
            deposit=args.deposit,
            model=model
        )

    # Бэктест
    backtest_result = None
    if args.backtest:
        if not args.json:
            print("🔄 Запуск бэктеста...\n")

        backtest_result = run_backtest(
            df=df,
            signal_generator=generate_signal,
            deposit=args.deposit,
            model=model
        )

    # Вывод результатов
    if args.json:
        output = {}
        if signal:
            output['signal'] = signal.to_dict()
        if backtest_result:
            output['backtest'] = backtest_result.to_dict()
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        if signal:
            print_signal_report(signal.to_dict())
        if backtest_result:
            print_backtest_report(backtest_result.to_dict())


if __name__ == '__main__':
    main()
