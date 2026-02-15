#!/usr/bin/env python
"""
Диагностическая CLI-утилита для быстрой проверки данных MOEX.
"""

from __future__ import annotations

import argparse

import pandas as pd

from adapters.moex.iss_client import MOEXAdapter
from adapters.moex.indicator_dataset import load_data_with_indicators
from services.strategy_engine.indicators import get_latest_signals


DEFAULT_MA_PERIODS = "50,200"
DEFAULT_ROWS = 20


def _parse_ma_periods(raw: str) -> list[int]:
    return [int(period.strip()) for period in raw.split(",") if period.strip()]


def _resolve_board(engine: str, board: str | None) -> str:
    if board:
        return board
    return "RFUD" if engine == "futures" else "TQBR"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Диагностическая загрузка данных MOEX с индикаторами",
    )
    parser.add_argument("ticker", type=str, help="Тикер инструмента (например, SBER, CCH6)")
    parser.add_argument(
        "--timeframe",
        "-t",
        type=str,
        default="10m",
        help="Таймфрейм: 1m, 5m, 10m, 15m, 30m, 1h, 4h, 1d, 1w, 1M",
    )
    parser.add_argument(
        "--engine",
        "-e",
        type=str,
        default="stock",
        help="Движок: stock, futures",
    )
    parser.add_argument(
        "--market",
        "-m",
        type=str,
        default="shares",
        help="Рынок: shares, forts",
    )
    parser.add_argument(
        "--board",
        "-b",
        type=str,
        default=None,
        help="Режим торгов: TQBR (акции), RFUD (фьючерсы)",
    )
    parser.add_argument(
        "--ma-periods",
        type=str,
        default=DEFAULT_MA_PERIODS,
        help="Периоды MA через запятую (например: 50,200)",
    )
    parser.add_argument(
        "--rsi-period",
        type=int,
        default=14,
        help="Период RSI",
    )
    parser.add_argument(
        "--rows",
        "-n",
        type=int,
        default=DEFAULT_ROWS,
        help="Количество последних строк для вывода",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Дата начала (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Дата окончания (YYYY-MM-DD)",
    )
    return parser


def run_probe(args: argparse.Namespace) -> int:
    ma_periods = _parse_ma_periods(args.ma_periods)
    board = _resolve_board(args.engine, args.board)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.precision', 2)

    print('='*80)
    print(f'Загрузка данных для {args.ticker} на таймфрейме {args.timeframe}')
    print(f'Движок: {args.engine}, Рынок: {args.market}, Режим торгов: {board}')
    print('='*80)

    adapter = MOEXAdapter(engine=args.engine, market=args.market)
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
        return 1

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
    return 0


def cli_main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_probe(args)


if __name__ == '__main__':
    raise SystemExit(cli_main())
