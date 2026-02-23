"""
Composite registry for exchange adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class ExchangeComposite:
    name: str
    default_board: Callable[[str], str]
    build_adapter: Callable[[str, str], Any]
    load_data_with_indicators: Callable[..., tuple]
    load_cli_dataset: Callable[..., Any]
    monitor_setups: Optional[Callable[..., Any]] = None


def _moex_default_board(engine: str) -> str:
    return "RFUD" if str(engine).lower() == "futures" else "TQBR"


def _binance_default_board(engine: str) -> str:
    del engine
    return ""


def _build_moex_adapter(engine: str, market: str):
    from adapters.moex.iss_client import MOEXAdapter

    return MOEXAdapter(engine=engine, market=market)


def _build_binance_adapter(engine: str, market: str):
    from adapters.binance import BinanceAdapter

    return BinanceAdapter(engine=engine, market=market)


def _load_moex_data_with_indicators(**kwargs):
    from adapters.moex import load_data_with_indicators

    return load_data_with_indicators(**kwargs)


def _load_binance_data_with_indicators(**kwargs):
    from adapters.binance import load_data_with_indicators

    return load_data_with_indicators(**kwargs)


def _load_moex_cli_dataset(**kwargs):
    from adapters.moex.cli_dataset_loader import load_cli_dataset

    return load_cli_dataset(**kwargs)


def _load_binance_cli_dataset(**kwargs):
    from adapters.binance.cli_dataset_loader import load_cli_dataset

    return load_cli_dataset(**kwargs)


def _monitor_moex_setups(**kwargs):
    from adapters.moex.setup_monitor import monitor_setups

    return monitor_setups(**kwargs)


_REGISTRY: dict[str, ExchangeComposite] = {
    "moex": ExchangeComposite(
        name="moex",
        default_board=_moex_default_board,
        build_adapter=_build_moex_adapter,
        load_data_with_indicators=_load_moex_data_with_indicators,
        load_cli_dataset=_load_moex_cli_dataset,
        monitor_setups=_monitor_moex_setups,
    ),
    "binance": ExchangeComposite(
        name="binance",
        default_board=_binance_default_board,
        build_adapter=_build_binance_adapter,
        load_data_with_indicators=_load_binance_data_with_indicators,
        load_cli_dataset=_load_binance_cli_dataset,
        monitor_setups=None,
    ),
}


def get_exchange_composite(exchange: str) -> ExchangeComposite:
    normalized = str(exchange).lower()
    try:
        return _REGISTRY[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_REGISTRY))
        raise NotImplementedError(f"Exchange '{exchange}' is not supported. Available: {supported}") from exc


def resolve_default_board(exchange: str, engine: str) -> str:
    composite = get_exchange_composite(exchange)
    return composite.default_board(engine)


def build_exchange_adapter(exchange: str, engine: str, market: str):
    composite = get_exchange_composite(exchange)
    return composite.build_adapter(engine, market)


def load_data_with_indicators_for_exchange(
    exchange: str,
    *,
    ticker: str,
    timeframe: str,
    start_date: Optional[str],
    end_date: Optional[str],
    board: str,
    adapter: Any,
    limit: Optional[int] = None,
):
    composite = get_exchange_composite(exchange)
    return composite.load_data_with_indicators(
        ticker=ticker,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        board=board,
        adapter=adapter,
        limit=limit,
    )


def load_cli_dataset_for_exchange(
    exchange: str,
    *,
    ticker: str,
    timeframe: str,
    start_date: Optional[str],
    end_date: Optional[str],
    board: str,
    adapter: Any,
    limit: Optional[int] = None,
):
    composite = get_exchange_composite(exchange)
    return composite.load_cli_dataset(
        ticker=ticker,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        board=board,
        adapter=adapter,
        limit=limit,
    )


def monitor_setups_for_exchange(
    exchange: str,
    *,
    setups: list[dict[str, Any]],
    adapter: Any,
    board: str,
    interval_seconds: int,
    export_path: str | None,
    sender: Any,
    max_cycles: int,
    filter_config: dict[str, Any] | None,
    monitor_workers: int,
):
    composite = get_exchange_composite(exchange)
    if composite.monitor_setups is None:
        raise NotImplementedError(f"Setup monitor is not implemented for exchange '{exchange}'.")

    return composite.monitor_setups(
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

