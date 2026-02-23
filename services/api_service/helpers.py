"""Compatibility facade for shared API-service helper functions."""

from .conversions import (
    coerce_float,
    coerce_int,
    json_safe,
    to_bool,
    to_float_or_none,
    to_int_or_none,
    to_json_object,
    to_unix_timestamp,
)
from .data_access import (
    build_adapter,
    build_equity_and_drawdown_curves,
    get_db_session_factory,
    get_model_or_raise,
    load_dataset,
    load_dataset_from_db,
    resolve_contract_risk_params,
    volume_context_series,
    with_atr,
)
from .request_params import (
    DEFAULT_GUEST_PORTFOLIO_BALANCE,
    DEFAULT_GUEST_PORTFOLIO_CURRENCY,
    normalize_system_config,
    parse_market_request_params,
    parse_request_params,
    require_fields,
    validate_trade_plan,
)

__all__ = [
    "DEFAULT_GUEST_PORTFOLIO_BALANCE",
    "DEFAULT_GUEST_PORTFOLIO_CURRENCY",
    "build_adapter",
    "build_equity_and_drawdown_curves",
    "coerce_float",
    "coerce_int",
    "get_db_session_factory",
    "get_model_or_raise",
    "json_safe",
    "load_dataset",
    "load_dataset_from_db",
    "normalize_system_config",
    "parse_market_request_params",
    "parse_request_params",
    "require_fields",
    "resolve_contract_risk_params",
    "to_bool",
    "to_float_or_none",
    "to_int_or_none",
    "to_json_object",
    "to_unix_timestamp",
    "validate_trade_plan",
    "volume_context_series",
    "with_atr",
]
