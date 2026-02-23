"""Public API service facade split by bounded use-cases."""

from .auth import (
    build_auth_login_response,
    build_auth_me_response,
    build_auth_register_response,
    resolve_authenticated_user_email,
)
from .core import build_health_response, build_models_response
from .errors import (
    ApiConflictError,
    ApiNotFoundError,
    ApiServiceError,
    ApiUnauthorizedError,
    ApiValidationError,
)
from .market import (
    build_backtest_response,
    build_candles_response,
    build_dashboard_backtest_response,
    build_dashboard_market_response,
    build_dashboard_robustness_response,
    build_moex_instruments_response,
    build_optimize_response,
    build_signal_response,
)
from .systems import (
    build_portfolio_response,
    build_portfolio_update_response,
    build_system_create_response,
    build_system_run_artifacts_response,
    build_system_runs_response,
    build_system_set_current_response,
    build_system_update_config_response,
    build_systems_response,
)

__all__ = [
    "ApiServiceError",
    "ApiValidationError",
    "ApiNotFoundError",
    "ApiUnauthorizedError",
    "ApiConflictError",
    "build_health_response",
    "build_models_response",
    "build_auth_register_response",
    "build_auth_login_response",
    "build_auth_me_response",
    "resolve_authenticated_user_email",
    "build_systems_response",
    "build_system_create_response",
    "build_system_update_config_response",
    "build_system_set_current_response",
    "build_portfolio_response",
    "build_portfolio_update_response",
    "build_system_runs_response",
    "build_system_run_artifacts_response",
    "build_moex_instruments_response",
    "build_candles_response",
    "build_dashboard_market_response",
    "build_dashboard_backtest_response",
    "build_dashboard_robustness_response",
    "build_signal_response",
    "build_backtest_response",
    "build_optimize_response",
]
