"""
Strategy Engine - универсальный движок торговых стратегий

Может использоваться со всеми адаптерами: MOEX, Binance, Alfa Invest и др.
"""

from .models import TradingModel, get_model, MODELS, list_models, compare_models
from .signals import TradingSignal, generate_signal
from .risk import (
    RiskParameters,
    calculate_position_risk,
    calculate_stop_by_atr,
    calculate_target_by_rr,
    calculate_kelly_criterion,
    get_risk_summary
)
from .backtest import (
    Trade,
    BacktestResults,
    FilterStats,
    run_backtest,
    compare_models_results
)
from .core import (
    calculate_atr,
    calculate_structure,
    calculate_distance_to_ma,
    calculate_volume_stats
)
from .monte_carlo import MonteCarloResult, run_monte_carlo
from .robustness import RobustnessMetrics, evaluate_robustness, calculate_robustness_score
from .walk_forward import WalkForwardResult, split_train_test, run_walk_forward
from .risk_model import estimate_risk_of_ruin, estimate_risk_of_ruin_from_backtest
from .regime_detection import (
    REGIMES,
    classify_market_regime,
    classify_current_regime,
    performance_by_regime,
)
from .regime_engine import RegimeEngine
from .risk_manager import RISK_ALLOCATION_MULTIPLIERS, apply_regime_risk
from .adaptive_model import build_regime_model
from .adaptive_signal_engine import AdaptiveSignalEngine
from .backtest_engine import evaluate_model
from .setup_generator import (
    TradeSetup,
    SetupGenerationResult,
    generate_trade_setup,
    export_setups_json,
    load_setups_json,
    timeframe_to_timedelta,
)
from .conservative_v2 import (
    ConservativeModelV2,
    ConservativeScoreResult,
    ConservativeV2Config,
    ScoringWeights,
    load_conservative_v2_config,
    should_use_conservative_v2,
)
from .monitor_core import (
    MonitorContext,
    MonitorCycleResult,
    MonitorEvent,
    MonitorRiskConfig,
    drawdown_pct,
    prepare_active_setups,
    process_monitor_cycle,
)
from .monitor_service import MonitorRunConfig, run_monitoring
from .ports import MonitorNotifierPort, MonitorPriceFeedPort, SetupPersistencePort
from .filter_config import (
    DEFAULT_FILTERS,
    apply_filters_to_model,
    deep_merge,
    load_config,
    resolve_config_path,
    validate_filter_config,
)

__all__ = [
    # Models
    "TradingModel",
    "get_model",
    "MODELS",
    "list_models",
    "compare_models",
    # Signals
    "TradingSignal",
    "generate_signal",
    # Risk
    "RiskParameters",
    "calculate_position_risk",
    "calculate_stop_by_atr",
    "calculate_target_by_rr",
    "calculate_kelly_criterion",
    "get_risk_summary",
    # Backtest
    "Trade",
    "BacktestResults",
    "FilterStats",
    "run_backtest",
    "compare_models_results",
    # Core
    "calculate_atr",
    "calculate_structure",
    "calculate_distance_to_ma",
    "calculate_volume_stats",
    # Monte Carlo
    "MonteCarloResult",
    "run_monte_carlo",
    # Robustness
    "RobustnessMetrics",
    "evaluate_robustness",
    "calculate_robustness_score",
    # Walk-forward
    "WalkForwardResult",
    "split_train_test",
    "run_walk_forward",
    # Adaptive regime / robustness layer
    "estimate_risk_of_ruin",
    "estimate_risk_of_ruin_from_backtest",
    "REGIMES",
    "classify_market_regime",
    "classify_current_regime",
    "performance_by_regime",
    "RegimeEngine",
    "RISK_ALLOCATION_MULTIPLIERS",
    "apply_regime_risk",
    "build_regime_model",
    "AdaptiveSignalEngine",
    "evaluate_model",
    "TradeSetup",
    "SetupGenerationResult",
    "generate_trade_setup",
    "export_setups_json",
    "load_setups_json",
    "timeframe_to_timedelta",
    # Conservative V2
    "ConservativeModelV2",
    "ConservativeScoreResult",
    "ConservativeV2Config",
    "ScoringWeights",
    "load_conservative_v2_config",
    "should_use_conservative_v2",
    # Monitor core
    "MonitorContext",
    "MonitorCycleResult",
    "MonitorEvent",
    "MonitorRiskConfig",
    "drawdown_pct",
    "prepare_active_setups",
    "process_monitor_cycle",
    "MonitorRunConfig",
    "run_monitoring",
    "MonitorPriceFeedPort",
    "MonitorNotifierPort",
    "SetupPersistencePort",
    # Filter config
    "DEFAULT_FILTERS",
    "deep_merge",
    "load_config",
    "resolve_config_path",
    "validate_filter_config",
    "apply_filters_to_model",
]
