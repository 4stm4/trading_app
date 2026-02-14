#!/usr/bin/env python
"""
Скрипт для быстрой загрузки данных из MOEX с индикаторами
"""

import argparse
import pandas as pd
from adapters.moex import load_data_with_indicators, get_latest_signals
from adapters.moex.moex import MOEXAdapter


def main():
    parser = argparse.ArgumentParser(description='Загрузка данных MOEX с индикаторами')
    parser.add_argument('ticker', type=str, help='Тикер инструмента (например, SBER, CCH6)')
    parser.add_argument('--timeframe', '-t', type=str, default='10m',
                        help='Таймфрейм: 1m, 5m, 10m, 15m, 30m, 1h, 4h, 1d, 1w, 1M (по умолчанию: 10m)')
    parser.add_argument('--engine', '-e', type=str, default='stock',
                        help='Движок: stock, futures (по умолчанию: stock)')
    parser.add_argument('--market', '-m', type=str, default='shares',
                        help='Рынок: shares, forts (по умолчанию: shares)')
    parser.add_argument('--board', '-b', type=str, default=None,
                        help='Режим торгов: TQBR (акции), RFUD (фьючерсы)')
    parser.add_argument('--ma-periods', type=str, default='50,200',
                        help='Периоды MA через запятую (по умолчанию: 50,200)')
    parser.add_argument('--rsi-period', type=int, default=14,
                        help='Период RSI (по умолчанию: 14)')
    parser.add_argument('--rows', '-n', type=int, default=20,
                        help='Количество последних строк для вывода (по умолчанию: 20)')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Дата начала (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='Дата окончания (YYYY-MM-DD)')

    args = parser.parse_args()

    # Настройка pandas для красивого вывода
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.precision', 2)

    # Парсим периоды MA
    ma_periods = [int(p.strip()) for p in args.ma_periods.split(',')]

    # Определяем board автоматически, если не указан
    board = args.board
    if board is None:
        if args.engine == 'futures':
            board = 'RFUD'
        else:
            board = 'TQBR'

    print('='*80)
    print(f'Загрузка данных для {args.ticker} на таймфрейме {args.timeframe}')
    print(f'Движок: {args.engine}, Рынок: {args.market}, Режим торгов: {board}')
    print('='*80)

    # Создаем адаптер
    adapter = MOEXAdapter(engine=args.engine, market=args.market)

    # Загружаем данные
    df, volume_stats = load_data_with_indicators(
        ticker=args.ticker,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        board=board,
        ma_periods=ma_periods,
        rsi_period=args.rsi_period,
        adapter=adapter
    )

    if df.empty:
        print(f'\n❌ Нет данных для {args.ticker}')
        print('\nПопробуйте изменить параметры:')
        print('  - Для фьючерсов: --engine futures --market forts')
        print('  - Для акций: --engine stock --market shares')
        return

    # Формируем колонки для вывода
    ma_cols = [f'ma{p}' for p in ma_periods]
    display_cols = ['open', 'high', 'low', 'close', 'volume'] + ma_cols + ['rsi']
    display_cols = [col for col in display_cols if col in df.columns]

    print(f'\nПоследние {args.rows} записей:')
    print(df[display_cols].tail(args.rows))

    print('\n' + '='*80)
    print('Статистика объема:')
    print('='*80)
    for key, value in volume_stats.items():
        print(f'{key:20s}: {value:,.0f}')

    print('\n' + '='*80)
    print('Последние сигналы:')
    print('='*80)
    signals = get_latest_signals(df)
    for key, value in signals.items():
        if key == 'timestamp':
            print(f'{key:20s}: {value}')
        elif isinstance(value, (int, float)):
            print(f'{key:20s}: {value:.2f}')
        else:
            print(f'{key:20s}: {value}')

    print('\n' + '='*80)
    print(f'Всего загружено свечей: {len(df)}')
    print(f'Период: {df.index[0]} - {df.index[-1]}')
    print('='*80 + '\n')


if __name__ == '__main__':
    main()
