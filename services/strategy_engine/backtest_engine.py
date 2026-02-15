"""
Backtest-движок для CLI: классический и адаптивный режимы.
"""

from typing import Any, Optional

import pandas as pd

from .adaptive_signal_engine import AdaptiveSignalEngine
from .backtest import run_backtest
from .models import TradingModel
from .monte_carlo import run_monte_carlo
from .robustness import evaluate_robustness
from .signals import generate_signal
from .walk_forward import WalkForwardResult, run_walk_forward, split_train_test

from .regime_detection import classify_market_regime, performance_by_regime
from .risk_model import estimate_risk_of_ruin_from_backtest


DEFAULT_MAX_HOLDING_CANDLES = 50
DEFAULT_LOOKBACK_WINDOW = 300
DEFAULT_TRAIN_RATIO = 0.7

RUNTIME_KWARGS_KEYS = {
    "max_holding_candles",
    "lookback_window",
    "train_ratio",
}

SIGNAL_KWARGS_KEYS = {
    "volume_mode",
    "structure_mode",
    "disable_rr",
    "disable_volume",
    "disable_trend",
    "debug_filters",
    "rsi_enabled",
    "rsi_trend_confirmation_only",
    "atr_enabled",
    "atr_min_percentile",
    "contract_margin_rub",
    "contract_multiplier",
    "filter_config",
    "min_expected_trades_per_month",
    "max_holding_candles",
    "lookback_window",
    "train_ratio",
}


def evaluate_model(
    df: pd.DataFrame,
    deposit: float,
    model: TradingModel,
    signal_kwargs: Optional[dict] = None,
    walk_forward: bool = False,
    monte_carlo_simulations: int = 0,
    debug_filters: bool = False,
    adaptive_regime: bool = False
) -> tuple[dict[str, Any], Any]:
    """
    Возвращает (stats_dict, raw_result_object).
    """
    signal_kwargs = signal_kwargs or {}
    runtime_params = _extract_runtime_params(signal_kwargs)
    max_holding_candles = runtime_params['max_holding_candles']
    lookback_window = runtime_params['lookback_window']
    train_ratio = runtime_params['train_ratio']

    runtime = _build_signal_runtime(
        adaptive_regime=adaptive_regime,
        signal_kwargs=signal_kwargs,
    )

    if walk_forward:
        if adaptive_regime:
            wf = _run_walk_forward_adaptive(
                df=df,
                deposit=deposit,
                model=model,
                runtime=runtime,
                monte_carlo_simulations=monte_carlo_simulations
            )
        else:
            wf = run_walk_forward(
                df=df,
                signal_generator=runtime['signal_generator'],
                deposit=deposit,
                model=model,
                signal_kwargs=runtime['signal_kwargs'],
                lookback_window=lookback_window,
                max_holding_candles=max_holding_candles,
                train_ratio=train_ratio,
                monte_carlo_simulations=monte_carlo_simulations,
                risk_constraints=runtime.get('risk_constraints'),
                execution_config=runtime.get('execution_config'),
                cost_config=runtime.get('cost_config'),
            )

        wf_dict = wf.to_dict()
        train_stats = wf_dict['train']
        test_stats = wf_dict['test']
        robustness = wf_dict['robustness']
        monte_carlo = wf_dict.get('monte_carlo')

        _, test_df = split_train_test(df, train_ratio=train_ratio)
        regimes = classify_market_regime(test_df)
        regime_stats = performance_by_regime(
            wf.test_results.trades,
            regimes,
            initial_balance=deposit
        )
        trades_by_regime = {k: v.get('trades', 0) for k, v in regime_stats.items()}

        risk_of_ruin = estimate_risk_of_ruin_from_backtest(
            winrate_pct=test_stats.get('winrate', 0),
            avg_win=test_stats.get('avg_win', 0),
            avg_loss=test_stats.get('avg_loss', 0),
            risk_percent=model.max_risk_percent
        )

        stats = {
            'model_name': model.name,
            'pf_train': train_stats.get('profit_factor', 0),
            'pf_test': test_stats.get('profit_factor', 0),
            'expectancy_test': test_stats.get('expectancy', 0),
            'maxdd_test': test_stats.get('max_drawdown_percent', 0),
            'return_pct_test': test_stats.get('return_pct', 0),
            'stability_ratio': robustness.get('stability_ratio', 0),
            'unstable': robustness.get('stability_ratio', 0) < 0.7,
            'unstable_oos': robustness.get('unstable_oos', False),
            'overfit': robustness.get('overfit', False),
            'robustness_score': robustness.get('robustness_score', 0),
            'score': robustness.get('robustness_score', 0),
            'risk_of_ruin': round(risk_of_ruin, 2),
            'trades_by_regime': trades_by_regime,
            'market_regime_performance': regime_stats,
            'train': train_stats,
            'test': test_stats,
            'monte_carlo': monte_carlo,
            'enabled_regimes': wf_dict.get('enabled_regimes', []),
            'disabled_regimes': wf_dict.get('disabled_regimes', {}),
            'train_regime_performance': wf_dict.get('regime_train_performance', {}),
        }
        stats['best_regime'], stats['best_regime_pf'] = _best_regime_from_stats(regime_stats)
        stats['admission'] = _admission_checks(
            pf=stats['pf_test'],
            expectancy=stats['expectancy_test'],
            sharpe=test_stats.get('sharpe_ratio', 0),
            maxdd=stats['maxdd_test'],
            stability=stats['stability_ratio'],
            risk_of_ruin=stats['risk_of_ruin'],
        )
        stats['edge_found'] = bool(stats['admission']['all_passed'])
        return stats, wf

    backtest = run_backtest(
        df=df,
        signal_generator=runtime['signal_generator'],
        deposit=deposit,
        model=model,
        lookback_window=lookback_window,
        max_holding_candles=max_holding_candles,
        signal_kwargs=runtime['signal_kwargs'],
        debug_filters=debug_filters,
        on_trade_close=runtime['on_trade_close'],
        risk_constraints=runtime.get('risk_constraints'),
        execution_config=runtime.get('execution_config'),
        cost_config=runtime.get('cost_config'),
    )
    backtest_dict = backtest.to_dict()

    regimes = classify_market_regime(df)
    regime_stats = performance_by_regime(
        backtest.trades,
        regimes,
        initial_balance=deposit
    )
    trades_by_regime = {k: v.get('trades', 0) for k, v in regime_stats.items()}

    risk_of_ruin = estimate_risk_of_ruin_from_backtest(
        winrate_pct=backtest_dict.get('winrate', 0),
        avg_win=backtest_dict.get('avg_win', 0),
        avg_loss=backtest_dict.get('avg_loss', 0),
        risk_percent=model.max_risk_percent
    )

    stats = {
        **backtest_dict,
        'score': _classic_score(backtest_dict),
        'risk_of_ruin': round(risk_of_ruin, 2),
        'trades_by_regime': trades_by_regime,
        'market_regime_performance': regime_stats,
    }
    stats['best_regime'], stats['best_regime_pf'] = _best_regime_from_stats(regime_stats)
    stats['admission'] = _admission_checks(
        pf=float(backtest_dict.get('profit_factor', 0)),
        expectancy=float(backtest_dict.get('expectancy', 0)),
        sharpe=float(backtest_dict.get('sharpe_ratio', 0)),
        maxdd=float(backtest_dict.get('max_drawdown_percent', 0)),
        stability=None,
        risk_of_ruin=float(stats['risk_of_ruin']),
    )
    stats['edge_found'] = bool(stats['admission']['all_passed'])
    return stats, backtest


def _build_signal_runtime(adaptive_regime: bool, signal_kwargs: Optional[dict]) -> dict[str, Any]:
    signal_kwargs = signal_kwargs or {}

    if not adaptive_regime:
        filtered_kwargs = {
            k: v
            for k, v in signal_kwargs.items()
            if k in SIGNAL_KWARGS_KEYS and k not in RUNTIME_KWARGS_KEYS
        }
        return {
            'signal_generator': generate_signal,
            'signal_kwargs': filtered_kwargs,
            'on_trade_close': None,
            'risk_constraints': _extract_risk_constraints(signal_kwargs),
            'execution_config': _extract_execution_config(signal_kwargs),
            'cost_config': _extract_cost_config(signal_kwargs),
        }

    engine = AdaptiveSignalEngine(
        volume_mode=signal_kwargs.get('volume_mode', 'fixed'),
        structure_mode=signal_kwargs.get('structure_mode', 'strict'),
        disable_rr=signal_kwargs.get('disable_rr', False),
        disable_volume=signal_kwargs.get('disable_volume', False),
        disable_trend=signal_kwargs.get('disable_trend', False),
        debug_filters=signal_kwargs.get('debug_filters', False),
        rsi_enabled=signal_kwargs.get('rsi_enabled', True),
        rsi_trend_confirmation_only=signal_kwargs.get('rsi_trend_confirmation_only', False),
        atr_enabled=signal_kwargs.get('atr_enabled', True),
        atr_min_percentile=signal_kwargs.get('atr_min_percentile', 0),
        contract_margin_rub=signal_kwargs.get('contract_margin_rub'),
        contract_multiplier=signal_kwargs.get('contract_multiplier', 1.0),
        pf_window=signal_kwargs.get('regime_pf_window', 20),
        disable_threshold_pf=signal_kwargs.get('min_regime_pf', 1.1),
        regime_thresholds=signal_kwargs.get('regime_thresholds'),
        enabled_regimes=signal_kwargs.get('enabled_regimes'),
        disabled_regimes_train=signal_kwargs.get('disabled_regimes_train'),
        regime_enabled_flags=signal_kwargs.get('regime_enabled_flags'),
        filter_config=signal_kwargs.get('filter_config'),
    )
    return {
        'signal_generator': engine.generate_signal,
        'signal_kwargs': None,
        'on_trade_close': engine.on_trade_close,
        'risk_constraints': _extract_risk_constraints(signal_kwargs),
        'execution_config': _extract_execution_config(signal_kwargs),
        'cost_config': _extract_cost_config(signal_kwargs),
        'adaptive_signal_kwargs': signal_kwargs
    }


def _run_walk_forward_adaptive(
    df: pd.DataFrame,
    deposit: float,
    model: TradingModel,
    runtime: dict[str, Any],
    monte_carlo_simulations: int
) -> WalkForwardResult:
    adaptive_kwargs = runtime.get('adaptive_signal_kwargs', {})
    runtime_params = _extract_runtime_params(adaptive_kwargs)
    max_holding_candles = runtime_params['max_holding_candles']
    lookback_window = runtime_params['lookback_window']
    train_ratio = runtime_params['train_ratio']

    train_df, test_df = split_train_test(df, train_ratio=train_ratio)
    train_lookback = _adaptive_lookback(len(train_df), lookback_window, max_holding_candles)
    test_lookback = _adaptive_lookback(len(test_df), lookback_window, max_holding_candles)

    train_backtest = run_backtest(
        df=train_df,
        signal_generator=runtime['signal_generator'],
        deposit=deposit,
        model=model,
        lookback_window=train_lookback,
        max_holding_candles=max_holding_candles,
        signal_kwargs=runtime['signal_kwargs'],
        debug_filters=False,
        on_trade_close=runtime['on_trade_close'],
        risk_constraints=runtime.get('risk_constraints'),
        execution_config=runtime.get('execution_config'),
        cost_config=runtime.get('cost_config'),
    )

    train_regimes = classify_market_regime(train_df)
    train_regime_stats = performance_by_regime(
        train_backtest.trades,
        train_regimes,
        initial_balance=deposit
    )
    enabled_regimes, disabled_regimes = _derive_enabled_regimes(
        train_regime_stats=train_regime_stats,
        signal_kwargs=runtime.get('adaptive_signal_kwargs', {}),
    )

    # Для test запускаем отдельный экземпляр adaptive runtime без train leakage
    test_signal_kwargs = dict(runtime.get('adaptive_signal_kwargs', {}))
    test_signal_kwargs['enabled_regimes'] = enabled_regimes
    test_signal_kwargs['disabled_regimes_train'] = disabled_regimes
    test_runtime = _build_signal_runtime(
        adaptive_regime=True,
        signal_kwargs=test_signal_kwargs
    )
    test_backtest = run_backtest(
        df=test_df,
        signal_generator=test_runtime['signal_generator'],
        deposit=deposit,
        model=model,
        lookback_window=test_lookback,
        max_holding_candles=max_holding_candles,
        signal_kwargs=test_runtime['signal_kwargs'],
        debug_filters=False,
        on_trade_close=test_runtime['on_trade_close'],
        risk_constraints=test_runtime.get('risk_constraints'),
        execution_config=test_runtime.get('execution_config'),
        cost_config=test_runtime.get('cost_config'),
    )

    monte_carlo = None
    if monte_carlo_simulations > 0:
        monte_carlo = run_monte_carlo(
            trades=test_backtest.trades,
            initial_balance=deposit,
            simulations=monte_carlo_simulations
        )

    robustness = evaluate_robustness(
        pf_train=train_backtest.profit_factor,
        pf_test=test_backtest.profit_factor,
        maxdd_test=test_backtest.max_drawdown_percent,
        monte_carlo=monte_carlo
    )

    return WalkForwardResult(
        model_name=model.name,
        train_results=train_backtest,
        test_results=test_backtest,
        robustness=robustness,
        monte_carlo=monte_carlo,
        enabled_regimes=sorted(enabled_regimes),
        disabled_regimes=disabled_regimes,
        regime_train_performance=train_regime_stats,
    )


def _adaptive_lookback(df_len: int, default_lookback: int, max_holding_candles: int) -> int:
    if df_len <= max_holding_candles + 20:
        return 20
    max_allowed = max(20, df_len - max_holding_candles - 1)
    return min(default_lookback, max_allowed)


def _extract_runtime_params(signal_kwargs: dict[str, Any]) -> dict[str, Any]:
    max_holding_candles = _safe_int(
        signal_kwargs.get('max_holding_candles', DEFAULT_MAX_HOLDING_CANDLES),
        default=DEFAULT_MAX_HOLDING_CANDLES,
        min_value=1,
    )
    lookback_window = _safe_int(
        signal_kwargs.get('lookback_window', DEFAULT_LOOKBACK_WINDOW),
        default=DEFAULT_LOOKBACK_WINDOW,
        min_value=20,
    )
    train_ratio = _safe_float(
        signal_kwargs.get('train_ratio', DEFAULT_TRAIN_RATIO),
        default=DEFAULT_TRAIN_RATIO,
        min_value=0.05,
        max_value=0.95,
    )
    return {
        'max_holding_candles': max_holding_candles,
        'lookback_window': lookback_window,
        'train_ratio': train_ratio,
    }


def _safe_int(value: Any, default: int, min_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, parsed)


def _safe_float(value: Any, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < min_value:
        return min_value
    if parsed > max_value:
        return max_value
    return parsed


def _classic_score(model_stats: dict[str, Any]) -> float:
    pf = model_stats.get('profit_factor', 0)
    ret = model_stats.get('return_pct', 0)
    maxdd = model_stats.get('max_drawdown_percent', 0)
    return (pf * 0.5) + (ret * 0.3) - (maxdd * 0.2)


def _extract_risk_constraints(signal_kwargs: dict[str, Any]) -> dict[str, float]:
    return {
        'max_daily_loss_pct': float(signal_kwargs.get('max_daily_loss_pct', 0.0)),
        'equity_dd_stop_pct': float(signal_kwargs.get('equity_dd_stop_pct', 0.0)),
    }


def _extract_execution_config(signal_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        'sl_tp_only': bool(signal_kwargs.get('sl_tp_only', True)),
        'partial_take_profit': bool(signal_kwargs.get('partial_take_profit', False)),
        'partial_tp_rr': float(signal_kwargs.get('partial_tp_rr', 1.0)),
        'partial_close_fraction': float(signal_kwargs.get('partial_close_fraction', 0.5)),
        'trail_after_partial': bool(signal_kwargs.get('trail_after_partial', False)),
        'trailing_atr_multiplier': float(signal_kwargs.get('trailing_atr_multiplier', 1.0)),
    }


def _extract_cost_config(signal_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        'enabled': bool(signal_kwargs.get('costs_enabled', True)),
        'engine': str(signal_kwargs.get('engine', 'stock')),
        'futures_per_contract_rub': float(signal_kwargs.get('futures_per_contract_rub', 2.0)),
        'futures_fee_mode': str(signal_kwargs.get('futures_fee_mode', 'round_trip')),
        'securities_bps': float(signal_kwargs.get('securities_bps', 4.9)),
        'settlement_bps': float(signal_kwargs.get('settlement_bps', 0.0)),
        'slippage_bps': float(signal_kwargs.get('slippage_bps', 0.0)),
    }


def _derive_enabled_regimes(
    train_regime_stats: dict[str, dict[str, float]],
    signal_kwargs: dict[str, Any],
) -> tuple[set[str], dict[str, str]]:
    thresholds = _normalize_regime_thresholds(signal_kwargs.get('regime_thresholds'))
    flags = signal_kwargs.get('regime_enabled_flags', {})
    min_regime_pf = float(signal_kwargs.get('min_regime_pf', 1.1))

    enabled: set[str] = set()
    disabled: dict[str, str] = {}

    for regime in ("trend", "range", "high_volatility"):
        if flags and not bool(flags.get(regime, True)):
            disabled[regime] = "disabled_by_config"
            continue

        pf = float(train_regime_stats.get(regime, {}).get('pf', 0.0))
        threshold = thresholds.get(regime, min_regime_pf)
        if pf >= threshold:
            enabled.add(regime)
        else:
            disabled[regime] = f"PF_train {pf:.2f} < {threshold:.2f}"

    return enabled, disabled


def _normalize_regime_thresholds(raw_thresholds: Any) -> dict[str, float]:
    result = {
        'trend': 1.1,
        'range': 1.1,
        'high_volatility': 1.05,
    }
    if isinstance(raw_thresholds, dict):
        for key in result:
            if key in raw_thresholds:
                try:
                    result[key] = float(raw_thresholds[key])
                except (TypeError, ValueError):
                    continue
    return result


def _best_regime_from_stats(regime_stats: dict[str, dict[str, float]]) -> tuple[str, float]:
    best_regime = "none"
    best_pf = 0.0
    for regime, metrics in regime_stats.items():
        pf = float(metrics.get('pf', 0.0))
        if pf > best_pf:
            best_pf = pf
            best_regime = regime
    return best_regime, best_pf


def _admission_checks(
    pf: float,
    expectancy: float,
    sharpe: float,
    maxdd: float,
    stability: Optional[float],
    risk_of_ruin: float,
) -> dict[str, Any]:
    checks = {
        'pf': pf >= 1.2,
        'expectancy': expectancy > 0,
        'sharpe': sharpe >= 0.8,
        'maxdd': maxdd <= 25.0,
        'risk_of_ruin': risk_of_ruin < 5.0,
    }
    if stability is not None:
        checks['stability'] = stability >= 0.7

    failed = [name for name, passed in checks.items() if not passed]
    return {
        'checks': checks,
        'all_passed': len(failed) == 0,
        'failed': failed,
    }
