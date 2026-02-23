#!/usr/bin/env python
"""
Быстрый сканер рынка по инструментам с оценкой пригодности под текущий портфель.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Optional

from loguru import logger

from adapters import (
    build_exchange_adapter,
    load_cli_dataset_for_exchange,
    resolve_default_board,
)
from services.strategy_engine import MODELS, generate_signal, get_model
from services.strategy_engine.filter_config import apply_filters_to_model, load_config


DEFAULT_BINANCE_UNIVERSE = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "BCHUSDT",
    "UNIUSDT",
    "ATOMUSDT",
    "NEARUSDT",
    "APTUSDT",
    "SUIUSDT",
    "TONUSDT",
    "TRXUSDT",
    "ETCUSDT",
    "FILUSDT",
    "INJUSDT",
    "TIAUSDT",
    "ARBUSDT",
    "OPUSDT",
]


CONFIDENCE_SCORE = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}


def configure_cli_logger() -> None:
    logger.remove()
    logger.add(sys.stdout, format="{message}", level="INFO")


@dataclass
class ScanRow:
    symbol: str
    timeframe: str
    data_start: str
    data_end: str
    signal: str
    confidence: str
    close: float
    atr: float
    atr_pct: float
    volume_ratio: float
    trend: str
    phase: str
    risk_amount: Optional[float]
    potential_amount: Optional[float]
    risk_budget_amount: float
    count: float
    position_size: float
    warnings_count: int
    first_warning: str
    data_points: int


def _parse_symbols_inline(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    return [token.strip().upper() for token in raw.split(",") if token.strip()]


def _parse_symbols_file(path: Optional[str]) -> list[str]:
    if not path:
        return []
    result: list[str] = []
    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            token = line.strip().upper()
            if not token or token.startswith("#"):
                continue
            result.append(token)
    return result


def _deduplicate_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _load_moex_universe(board: str, max_symbols: int) -> list[str]:
    adapter = build_exchange_adapter("moex", "stock", "shares")
    get_securities = getattr(adapter, "get_securities", None)
    if not callable(get_securities):
        return []
    df = get_securities(board=board)
    if df.empty or "SECID" not in df.columns:
        return []
    symbols = [str(v).strip().upper() for v in df["SECID"].tolist() if str(v).strip()]
    return _deduplicate_keep_order(symbols)[:max_symbols]


def _resolve_universe(
    *,
    exchange: str,
    board: str,
    symbols_inline: Optional[str],
    symbols_file: Optional[str],
    max_symbols: int,
) -> list[str]:
    explicit = _parse_symbols_inline(symbols_inline) + _parse_symbols_file(symbols_file)
    if explicit:
        return _deduplicate_keep_order(explicit)[:max_symbols]

    if exchange == "binance":
        return DEFAULT_BINANCE_UNIVERSE[:max_symbols]
    if exchange == "moex":
        return _load_moex_universe(board=board, max_symbols=max_symbols)
    return []


def _resolve_contract_params(
    adapter: Any,
    symbol: str,
    board: str,
) -> tuple[dict[str, float], float, Optional[float]]:
    get_security_spec = getattr(adapter, "get_security_spec", None)
    if not callable(get_security_spec):
        return {}, 1.0, None

    try:
        spec = get_security_spec(symbol, board=board) or {}
    except Exception:
        return {}, 1.0, None

    signal_kwargs: dict[str, float] = {}
    contract_multiplier = 1.0
    contract_margin = None
    try:
        margin = float(spec.get("initial_margin"))
        if margin > 0:
            signal_kwargs["contract_margin_rub"] = margin
            contract_margin = margin
    except (TypeError, ValueError):
        pass
    try:
        multiplier = float(spec.get("contract_multiplier"))
        if multiplier > 0:
            signal_kwargs["contract_multiplier"] = multiplier
            contract_multiplier = multiplier
    except (TypeError, ValueError):
        pass
    return signal_kwargs, contract_multiplier, contract_margin


def _calculate_count_by_portfolio(
    *,
    deposit: float,
    close_price: float,
    contract_margin: Optional[float],
) -> float:
    dep = max(float(deposit), 0.0)
    if contract_margin is not None and contract_margin > 0:
        return float(max(0, math.floor(dep / contract_margin)))
    if close_price <= 0:
        return 0.0
    return dep / close_price


def _build_signal_kwargs(filter_config: dict[str, Any]) -> dict[str, Any]:
    filters = filter_config["filters"]
    volume_cfg = filters["volume"]
    trend_cfg = filters["trend"]
    rsi_cfg = filters["rsi"]
    atr_cfg = filters["atr"]
    scoring_cfg = filters.get("scoring", {})

    return {
        "volume_mode": volume_cfg["mode"],
        "structure_mode": "strict",
        "disable_rr": False,
        "disable_volume": not bool(volume_cfg["enabled"]),
        "disable_trend": not bool(trend_cfg["enabled"]),
        "debug_filters": False,
        "rsi_enabled": bool(rsi_cfg["enabled"]),
        "rsi_trend_confirmation_only": bool(rsi_cfg["trend_confirmation_only"]),
        "atr_enabled": bool(atr_cfg["enabled"]),
        "atr_min_percentile": int(atr_cfg["min_percentile"]),
        "filter_config": filter_config,
        "min_expected_trades_per_month": float(scoring_cfg.get("min_expected_trades_per_month", 0.0)),
    }


def _format_dt(value: Any) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _scan_symbol(
    *,
    exchange: str,
    engine: str,
    market: str,
    board: str,
    symbol: str,
    timeframe: str,
    deposit: float,
    model,
    signal_kwargs: dict[str, Any],
    limit: Optional[int],
) -> tuple[str, Optional[ScanRow], Optional[str]]:
    try:
        adapter = build_exchange_adapter(exchange, engine, market)
        data_result = load_cli_dataset_for_exchange(
            exchange=exchange,
            ticker=symbol,
            timeframe=timeframe,
            start_date=None,
            end_date=None,
            board=board,
            adapter=adapter,
            limit=limit,
        )
        df = data_result.df
        if df.empty:
            return symbol, None, "Нет данных"

        local_kwargs = dict(signal_kwargs)
        contract_risk_kwargs, contract_multiplier, contract_margin = _resolve_contract_params(adapter, symbol, board)
        local_kwargs.update(contract_risk_kwargs)

        signal = generate_signal(
            df=df,
            deposit=deposit,
            model=model,
            **local_kwargs,
        )
        close_price = float(df["close"].iloc[-1])
        atr_value = float(signal.atr)
        atr_pct = (atr_value / close_price * 100.0) if close_price > 0 else 0.0
        count = _calculate_count_by_portfolio(
            deposit=deposit,
            close_price=close_price,
            contract_margin=contract_margin,
        )
        risk_budget_amount = float(deposit) * float(model.max_risk_percent) / 100.0
        risk_amount: Optional[float] = None
        potential: Optional[float] = None
        if signal.signal != "none":
            risk_amount = float(signal.risk_rub)
            potential = (
                abs(float(signal.target) - float(signal.entry))
                * float(signal.position_size)
                * contract_multiplier
            )

        warnings = list(signal.warnings or [])
        row = ScanRow(
            symbol=symbol,
            timeframe=timeframe,
            data_start=_format_dt(df.index[0]),
            data_end=_format_dt(df.index[-1]),
            signal=signal.signal,
            confidence=signal.confidence,
            close=close_price,
            atr=atr_value,
            atr_pct=atr_pct,
            volume_ratio=float(signal.volume_ratio),
            trend=signal.structure,
            phase=signal.phase,
            risk_amount=risk_amount,
            potential_amount=potential,
            risk_budget_amount=risk_budget_amount,
            count=float(count),
            position_size=float(signal.position_size),
            warnings_count=len(warnings),
            first_warning=warnings[0] if warnings else "",
            data_points=len(df),
        )
        return symbol, row, None

    except Exception as exc:  # noqa: BLE001
        return symbol, None, str(exc)


def _print_table(title: str, rows: list[list[str]], headers: list[str]) -> None:
    logger.info("\n" + "=" * 110)
    logger.info(title)
    logger.info("=" * 110)
    table = [headers] + rows
    widths = [max(len(str(row[col])) for row in table) for col in range(len(headers))]

    def render(row: list[str]) -> str:
        return "  ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row))

    logger.info(render(headers))
    logger.info("-" * sum(widths) + "-" * (2 * (len(widths) - 1)))
    for row in rows:
        logger.info(render(row))


def _format_money(exchange: str) -> str:
    return "₽" if exchange == "moex" else "USDT"


def _format_count(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Быстрый рыночный сканер по инструментам")
    parser.add_argument("--exchange", type=str, default="binance", choices=("moex", "binance"))
    parser.add_argument("--engine", "-e", type=str, default="stock")
    parser.add_argument("--market", "-m", type=str, default="shares")
    parser.add_argument("--board", "-b", type=str, default=None)
    parser.add_argument("--deposit", "-d", type=float, required=True, help="Размер портфеля")
    parser.add_argument("--timeframe", "-t", type=str, default="1h")
    parser.add_argument("--model", type=str, default="balanced", choices=tuple(MODELS.keys()))
    parser.add_argument("--symbols", type=str, default=None, help="Тикеры через запятую")
    parser.add_argument("--symbols-file", type=str, default=None, help="Файл с тикерами (по одному на строку)")
    parser.add_argument("--max-symbols", type=int, default=25, help="Лимит инструментов в скане")
    parser.add_argument("--workers", type=int, default=8, help="Параллельные воркеры")
    parser.add_argument("--limit", type=int, default=None, help="Лимит свечей на инструмент")
    parser.add_argument("--config", type=str, default=None, help="Путь к YAML-конфигу (по умолчанию strict.yaml)")
    parser.add_argument("--top", type=int, default=10, help="Сколько строк показывать в отчете")
    parser.add_argument("--only-signals", action="store_true", help="Показывать только инструменты с signal != none")
    parser.add_argument("--json", action="store_true", help="JSON вывод")
    args = parser.parse_args()

    configure_cli_logger()

    if args.max_symbols <= 0:
        parser.error("--max-symbols must be > 0")
    if args.workers <= 0:
        parser.error("--workers must be > 0")
    if args.top <= 0:
        parser.error("--top must be > 0")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be > 0")

    board = args.board or resolve_default_board(args.exchange, args.engine)
    filter_config = load_config(args.config)
    model = apply_filters_to_model(get_model(args.model), filter_config)
    signal_kwargs = _build_signal_kwargs(filter_config)

    universe = _resolve_universe(
        exchange=args.exchange,
        board=board,
        symbols_inline=args.symbols,
        symbols_file=args.symbols_file,
        max_symbols=args.max_symbols,
    )

    if not universe:
        logger.info("❌ Не удалось сформировать список инструментов для скана.")
        return

    logger.info(
        f"Сканирование: exchange={args.exchange}, tf={args.timeframe}, model={args.model}, "
        f"symbols={len(universe)}, deposit={args.deposit:.2f}"
    )

    rows: list[ScanRow] = []
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _scan_symbol,
                exchange=args.exchange,
                engine=args.engine,
                market=args.market,
                board=board,
                symbol=symbol,
                timeframe=args.timeframe,
                deposit=args.deposit,
                model=model,
                signal_kwargs=signal_kwargs,
                limit=args.limit,
            ): symbol
            for symbol in universe
        }
        for future in as_completed(futures):
            symbol, row, error = future.result()
            if error:
                errors[symbol] = error
                continue
            if row is not None:
                rows.append(row)

    report_rows = [row for row in rows if row.signal != "none"] if args.only_signals else rows

    if args.json:
        payload = {
            "exchange": args.exchange,
            "timeframe": args.timeframe,
            "model": args.model,
            "deposit": args.deposit,
            "rows": [asdict(row) for row in report_rows],
            "scanned_count": len(rows),
            "returned_count": len(report_rows),
            "only_signals": bool(args.only_signals),
            "errors": errors,
        }
        logger.info(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    currency = _format_money(args.exchange)
    actionable = [row for row in report_rows if row.signal != "none"]
    actionable_sorted = sorted(
        actionable,
        key=lambda item: (
            CONFIDENCE_SCORE.get(item.confidence, 0),
            item.potential_amount or 0.0,
            item.volume_ratio,
            item.atr_pct,
        ),
        reverse=True,
    )[: args.top]

    volatile_sorted = sorted(report_rows, key=lambda item: item.atr_pct, reverse=True)[: args.top]
    liquid_sorted = sorted(report_rows, key=lambda item: item.volume_ratio, reverse=True)[: args.top]

    logger.info("")
    logger.info(
        f"Готово: получено {len(rows)} инструментов, после фильтра only_signals={bool(args.only_signals)} "
        f"осталось {len(report_rows)}, ошибок загрузки: {len(errors)}"
    )
    if errors:
        logger.info("Ошибки (первые 5):")
        for idx, (symbol, err) in enumerate(errors.items()):
            if idx >= 5:
                break
            logger.info(f"  {symbol}: {err}")

    if not report_rows:
        logger.info("⚠ Подходящих инструментов для вывода нет.")
        return

    action_rows = [
        [
            row.symbol,
            row.data_start,
            row.data_end,
            row.signal.upper(),
            row.confidence.upper(),
            f"{row.atr_pct:.2f}%",
            f"{row.volume_ratio:.2f}x",
            _format_count(row.count),
            f"{(row.risk_amount or 0.0):.2f} {currency}",
            f"{(row.potential_amount or 0.0):.2f} {currency}",
            row.first_warning[:46],
        ]
        for row in actionable_sorted
    ]
    _print_table(
        title="ТОП ПОДХОДЯЩИХ ИНСТРУМЕНТОВ (с текущим сигналом)",
        rows=action_rows if action_rows else [["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "Нет сигналов"]],
        headers=["Symbol", "Start", "End", "Signal", "Conf", "ATR%", "Vol", "Count", "Risk", "Potential", "Warning"],
    )

    volatile_rows = [
        [
            row.symbol,
            row.data_start,
            row.data_end,
            row.signal.upper(),
            f"{row.close:.4f}",
            f"{row.atr:.4f}",
            f"{row.atr_pct:.2f}%",
            f"{row.volume_ratio:.2f}x",
            _format_count(row.count),
            row.phase,
        ]
        for row in volatile_sorted
    ]
    _print_table(
        title="ТОП ВОЛАТИЛЬНЫХ ИНСТРУМЕНТОВ",
        rows=volatile_rows,
        headers=["Symbol", "Start", "End", "Signal", "Close", "ATR", "ATR%", "Vol", "Count", "Phase"],
    )

    liquid_rows = [
        [
            row.symbol,
            row.data_start,
            row.data_end,
            row.signal.upper(),
            f"{row.volume_ratio:.2f}x",
            f"{row.atr_pct:.2f}%",
            _format_count(row.count),
            row.trend,
            row.phase,
            row.first_warning[:46],
        ]
        for row in liquid_sorted
    ]
    _print_table(
        title="ТОП ИНСТРУМЕНТОВ ПО VOLUME RATIO",
        rows=liquid_rows,
        headers=["Symbol", "Start", "End", "Signal", "Vol", "ATR%", "Count", "Trend", "Phase", "Warning"],
    )


if __name__ == "__main__":
    main()
