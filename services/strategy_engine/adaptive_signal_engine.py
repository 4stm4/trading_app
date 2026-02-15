"""
Адаптивный signal engine с переключением модели по режиму.
"""

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from .filter_config import apply_filters_to_model
from .models import TradingModel
from .signals import TradingSignal, generate_signal

from .adaptive_model import build_regime_model
from .regime_engine import RegimeEngine
from .risk_manager import apply_regime_risk


@dataclass
class AdaptiveSignalEngine:
    volume_mode: str = "fixed"
    structure_mode: str = "strict"
    disable_rr: bool = False
    disable_volume: bool = False
    disable_trend: bool = False
    debug_filters: bool = False
    rsi_enabled: bool = True
    rsi_trend_confirmation_only: bool = False
    atr_enabled: bool = True
    atr_min_percentile: int = 0
    contract_margin_rub: Optional[float] = None
    contract_multiplier: float = 1.0
    pf_window: int = 20
    disable_threshold_pf: float = 1.0
    regime_thresholds: Optional[dict[str, float]] = None
    regime_enabled_flags: Optional[dict[str, bool]] = None
    enabled_regimes: Optional[set[str]] = None
    disabled_regimes_train: Optional[dict[str, str]] = None
    filter_config: Optional[dict[str, Any]] = None

    def __post_init__(self):
        enabled = self.enabled_regimes
        if enabled is None and self.regime_enabled_flags is not None:
            enabled = {
                key for key, is_enabled in self.regime_enabled_flags.items()
                if is_enabled
            }
        self.regime_engine = RegimeEngine(
            pf_window=self.pf_window,
            disable_threshold_pf=self.disable_threshold_pf,
            thresholds_by_regime=self.regime_thresholds,
            enabled_regimes=enabled,
        )

    def generate_signal(self, df: pd.DataFrame, deposit: float, model: TradingModel) -> TradingSignal:
        regime = self.regime_engine.detect_regime(df)

        if not self.regime_engine.is_regime_enabled(regime):
            reason = "режим отключен по перформансу"
            if self.disabled_regimes_train and regime in self.disabled_regimes_train:
                reason = self.disabled_regimes_train[regime]
            return _create_regime_blocked_signal(
                df=df,
                model_name=model.name,
                regime=regime,
                reason=reason,
            )

        precheck_block = _regime_precheck(df, regime)
        if precheck_block is not None:
            precheck_block.model_name = model.name
            precheck_block.market_regime = regime
            return precheck_block

        regime_model = build_regime_model(regime)
        regime_model = apply_regime_risk(regime_model, regime)
        if self.filter_config:
            regime_model = apply_filters_to_model(regime_model, self.filter_config, regime=regime)

        signal = generate_signal(
            df=df,
            deposit=deposit,
            model=regime_model,
            volume_mode=self.volume_mode,
            structure_mode=self.structure_mode,
            disable_rr=self.disable_rr,
            disable_volume=self.disable_volume,
            disable_trend=self.disable_trend,
            debug_filters=self.debug_filters,
            rsi_enabled=self.rsi_enabled,
            rsi_trend_confirmation_only=self.rsi_trend_confirmation_only,
            atr_enabled=self.atr_enabled,
            atr_min_percentile=self.atr_min_percentile,
            filter_config=self.filter_config,
            contract_margin_rub=self.contract_margin_rub,
            contract_multiplier=self.contract_multiplier,
        )

        signal.market_regime = regime
        signal.model_name = regime_model.name
        signal.warnings = list(signal.warnings) + [f"Regime: {regime}"]
        return signal

    def on_trade_close(self, trade):
        regime = getattr(trade, "regime", None)
        pnl = getattr(trade, "pnl", None)
        if regime is None or pnl is None:
            return
        self.regime_engine.register_trade_result(regime, pnl)


def _create_regime_blocked_signal(df: pd.DataFrame, model_name: str, regime: str, reason: str) -> TradingSignal:
    if df.empty:
        price = 0.0
        rsi = 0.0
        atr = 0.0
    else:
        price = float(df["close"].iloc[-1])
        rsi = float(df["rsi"].iloc[-1]) if "rsi" in df.columns and pd.notna(df["rsi"].iloc[-1]) else 50.0
        atr = float(df["atr"].iloc[-1]) if "atr" in df.columns and pd.notna(df["atr"].iloc[-1]) else 0.0

    return TradingSignal(
        signal="none",
        entry=price,
        stop=0.0,
        target=0.0,
        rr=0.0,
        risk_rub=0.0,
        risk_percent=0.0,
        position_size=0.0,
        structure="unknown",
        phase="unknown",
        volume_ratio=0.0,
        atr=atr,
        distance_ma50_pct=0.0,
        distance_ma200_pct=0.0,
        rsi=rsi,
        confidence="none",
        warnings=[f"Торговля в режиме {regime} отключена: {reason}"],
        model_name=model_name,
        market_regime=regime,
        debug_potential_setup=False,
        debug_filter_stage="filtered_trend",
    )


def _regime_precheck(df: pd.DataFrame, regime: str) -> Optional[TradingSignal]:
    if df.empty or len(df) < 220:
        return None

    close = float(df["close"].iloc[-1])
    ma50 = _safe_last(df, "ma50", close)
    ma200 = _safe_last(df, "ma200", close)
    atr = _safe_last(df, "atr", 0.0)
    rsi = _safe_last(df, "rsi", 50.0)
    prev_high = float(df["high"].iloc[-2]) if len(df) > 1 else close
    prev_low = float(df["low"].iloc[-2]) if len(df) > 1 else close
    trend_strength = abs(((ma50 - ma200) / ma200) * 100) if ma200 else 0.0
    atr_percentile = _atr_percentile(df)

    if regime == "trend":
        pullback_to_ma50 = abs(close - ma50) <= max(atr, close * 0.002)
        if not (ma50 > ma200 and trend_strength >= 2.0 and pullback_to_ma50 and 40.0 <= rsi <= 60.0):
            return _blocked_signal_from_check(
                close=close,
                atr=atr,
                rsi=rsi,
                reason="Trend precheck: MA50>MA200, strength>=2, pullback to MA50, RSI 40-60",
                stage="filtered_trend",
            )

    elif regime == "range":
        tail = df.tail(40)
        rng_high = float(tail["high"].max())
        rng_low = float(tail["low"].min())
        in_channel = (close >= (rng_low - atr)) and (close <= (rng_high + atr))
        near_boundary = abs(close - rng_low) <= atr or abs(close - rng_high) <= atr
        bounce = False
        if len(df) > 1:
            prev_close = float(df["close"].iloc[-2])
            bounce = (abs(close - rng_low) <= atr and close > prev_close) or (
                abs(close - rng_high) <= atr and close < prev_close
            )

        if not (in_channel and 30.0 <= rsi <= 70.0 and near_boundary and bounce):
            return _blocked_signal_from_check(
                close=close,
                atr=atr,
                rsi=rsi,
                reason="Range precheck: channel±ATR, RSI 30-70, boundary bounce",
                stage="no_setup",
            )

    elif regime == "high_volatility":
        breakout = close > prev_high or close < prev_low
        if not (atr_percentile > 70.0 and breakout):
            return _blocked_signal_from_check(
                close=close,
                atr=atr,
                rsi=rsi,
                reason="HighVol precheck: ATR percentile>70 and breakout only",
                stage="filtered_atr",
            )

    return None


def _blocked_signal_from_check(close: float, atr: float, rsi: float, reason: str, stage: str) -> TradingSignal:
    return TradingSignal(
        signal="none",
        entry=close,
        stop=0.0,
        target=0.0,
        rr=0.0,
        risk_rub=0.0,
        risk_percent=0.0,
        position_size=0.0,
        structure="unknown",
        phase="unknown",
        volume_ratio=0.0,
        atr=atr,
        distance_ma50_pct=0.0,
        distance_ma200_pct=0.0,
        rsi=rsi,
        confidence="none",
        warnings=[reason],
        model_name="adaptive_regime",
        market_regime="unknown",
        debug_potential_setup=False,
        debug_filter_stage=stage,
    )


def _safe_last(df: pd.DataFrame, column: str, default: float) -> float:
    if column not in df.columns:
        return default
    value = df[column].iloc[-1]
    if pd.isna(value):
        return default
    return float(value)


def _atr_percentile(df: pd.DataFrame) -> float:
    if "atr" not in df.columns:
        return 0.0
    series = (df["atr"] / df["close"]).replace([float("inf"), float("-inf")], pd.NA).dropna()
    if series.empty:
        return 0.0
    value = float(series.rank(pct=True).iloc[-1] * 100)
    if pd.isna(value):
        return 0.0
    return value
