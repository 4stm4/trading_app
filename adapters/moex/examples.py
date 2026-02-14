"""
Примеры использования модуля загрузки данных MOEX с индикаторами
"""

from adapters.moex import load_data_with_indicators, get_latest_signals
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def example_1_basic_usage():
    """Пример 1: Базовая загрузка данных с индикаторами"""
    print("\n" + "="*70)
    print("Пример 1: Базовая загрузка данных с индикаторами")
    print("="*70 + "\n")

    # Загружаем данные для Газпрома на часовом таймфрейме
    df, volume_stats = load_data_with_indicators(
        ticker="GAZP",
        timeframe="1h"
    )

    if not df.empty:
        print("Последние 10 записей:")
        print(df[["close", "volume", "ma50", "ma200", "rsi"]].tail(10))

        print(f"\nСтатистика объема:")
        print(f"Общий объем: {volume_stats['total_volume']:,.0f}")
        print(f"Средний объем: {volume_stats['avg_volume']:,.0f}")
        print(f"Текущий объем: {volume_stats['current_volume']:,.0f}")


def example_2_custom_timeframes():
    """Пример 2: Различные таймфреймы"""
    print("\n" + "="*70)
    print("Пример 2: Различные таймфреймы")
    print("="*70 + "\n")

    timeframes = ["15m", "1h", "4h", "1d"]

    for tf in timeframes:
        try:
            df, _ = load_data_with_indicators(
                ticker="SBER",
                timeframe=tf,
                ma_periods=[20, 50]  # Используем MA20 и MA50
            )

            if not df.empty:
                last_row = df.iloc[-1]
                print(f"\n{tf:5s} | Цена: {last_row['close']:8.2f} | "
                      f"MA20: {last_row['ma20']:8.2f} | "
                      f"MA50: {last_row['ma50']:8.2f} | "
                      f"RSI: {last_row['rsi']:5.1f}")
        except Exception as e:
            print(f"{tf}: Ошибка - {e}")


def example_3_custom_date_range():
    """Пример 3: Загрузка данных за определенный период"""
    print("\n" + "="*70)
    print("Пример 3: Загрузка данных за определенный период")
    print("="*70 + "\n")

    # Загружаем данные за январь 2026
    df, volume_stats = load_data_with_indicators(
        ticker="LKOH",
        timeframe="1d",
        start_date="2026-01-01",
        end_date="2026-01-31"
    )

    if not df.empty:
        print(f"Загружено {len(df)} дневных свечей")
        print(f"Период: {df.index[0]} - {df.index[-1]}")
        print(f"\nМаксимальная цена: {df['close'].max():.2f}")
        print(f"Минимальная цена: {df['close'].min():.2f}")
        print(f"Средняя цена: {df['close'].mean():.2f}")


def example_4_trading_signals():
    """Пример 4: Получение торговых сигналов"""
    print("\n" + "="*70)
    print("Пример 4: Получение торговых сигналов")
    print("="*70 + "\n")

    tickers = ["SBER", "GAZP", "LKOH", "YNDX", "MGNT"]

    for ticker in tickers:
        try:
            df, _ = load_data_with_indicators(
                ticker=ticker,
                timeframe="1d"
            )

            if not df.empty:
                signals = get_latest_signals(df)

                print(f"\n{ticker}:")
                print(f"  Цена: {signals['close']:.2f}")
                print(f"  RSI: {signals['rsi']:.1f} ({signals['rsi_signal']})")
                print(f"  Позиция относительно MA50: {signals['price_vs_ma50']}")
                print(f"  Позиция относительно MA200: {signals['price_vs_ma200']}")
                print(f"  Тренд: {signals['ma_cross']}")

        except Exception as e:
            print(f"{ticker}: Ошибка - {e}")


def example_5_custom_ma_periods():
    """Пример 5: Кастомные периоды скользящих средних"""
    print("\n" + "="*70)
    print("Пример 5: Кастомные периоды скользящих средних")
    print("="*70 + "\n")

    # Используем MA10, MA20, MA50
    df, _ = load_data_with_indicators(
        ticker="SBER",
        timeframe="1h",
        ma_periods=[10, 20, 50],
        rsi_period=9  # Более чувствительный RSI
    )

    if not df.empty:
        print("Последние 5 записей с кастомными индикаторами:")
        print(df[["close", "ma10", "ma20", "ma50", "rsi"]].tail())


def example_6_volume_analysis():
    """Пример 6: Анализ объемов"""
    print("\n" + "="*70)
    print("Пример 6: Анализ объемов")
    print("="*70 + "\n")

    df, volume_stats = load_data_with_indicators(
        ticker="SBER",
        timeframe="1h"
    )

    if not df.empty:
        # Находим свечи с объемом выше среднего
        avg_vol = volume_stats['avg_volume']
        high_volume = df[df['volume'] > avg_vol * 1.5]

        print(f"Средний объем: {avg_vol:,.0f}")
        print(f"Свечей с повышенным объемом (>150% среднего): {len(high_volume)}")

        if len(high_volume) > 0:
            print(f"\nПоследние 5 свечей с повышенным объемом:")
            print(high_volume[["close", "volume"]].tail())


def example_7_screen_multiple_tickers():
    """Пример 7: Скрининг нескольких акций по критериям"""
    print("\n" + "="*70)
    print("Пример 7: Скрининг акций по критериям")
    print("="*70 + "\n")

    tickers = ["SBER", "GAZP", "LKOH", "YNDX", "MGNT", "ROSN", "NVTK"]

    print("Критерии: RSI < 30 (перепроданность) и цена выше MA200")
    print("-" * 70)

    found = 0
    for ticker in tickers:
        try:
            df, _ = load_data_with_indicators(ticker=ticker, timeframe="1d")

            if not df.empty:
                signals = get_latest_signals(df)

                # Проверяем критерии
                if (signals.get('rsi', 100) < 30 and
                    signals.get('price_vs_ma200') == 'above'):

                    print(f"{ticker}: Цена={signals['close']:.2f}, "
                          f"RSI={signals['rsi']:.1f} ✓")
                    found += 1

        except Exception as e:
            continue

    if found == 0:
        print("Нет акций, соответствующих критериям")
    else:
        print(f"\nНайдено акций: {found}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ МОДУЛЯ MOEX INDICATORS")
    print("="*70)

    # Запускаем все примеры
    example_1_basic_usage()
    example_2_custom_timeframes()
    example_3_custom_date_range()
    example_4_trading_signals()
    example_5_custom_ma_periods()
    example_6_volume_analysis()
    example_7_screen_multiple_tickers()

    print("\n" + "="*70)
    print("ВСЕ ПРИМЕРЫ ВЫПОЛНЕНЫ")
    print("="*70 + "\n")
