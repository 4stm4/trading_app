"""
Scoring engine для ConservativeModelV2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .models import TradingModel


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class ScoringWeights:
    trend_alignment: float = 0.25
    trend_strength: float = 0.25
    volume: float = 0.15
    rsi: float = 0.10
    pullback: float = 0.25
    ma_distance: float = 0.0
    structure: float = 0.0

    def normalized(self) -> "ScoringWeights":
        total = (
            self.trend_alignment
            + self.trend_strength
            + self.volume
            + self.rsi
            + self.pullback
            + self.ma_distance
            + self.structure
        )
        if total <= 0:
            return ScoringWeights()
        return ScoringWeights(
            trend_alignment=self.trend_alignment / total,
            trend_strength=self.trend_strength / total,
            volume=self.volume / total,
            rsi=self.rsi / total,
            pullback=self.pullback / total,
            ma_distance=self.ma_distance / total,
            structure=self.structure / total,
        )


@dataclass(frozen=True)
class ConservativeV2Config:
    enabled: bool = True
    min_score: float = 0.65
    max_relaxation: float = 0.20
    atr_relaxation_bonus: float = 0.08
    trend_relaxation_bonus: float = 0.05
    range_relaxation_bonus: float = 0.03
    min_expected_trades_per_month: float = 35.0
    weights: ScoringWeights = field(default_factory=ScoringWeights)


@dataclass(frozen=True)
class ConservativeScoreResult:
    total_score: float
    threshold: float
    confidence: str
    pullback_profile: str
    rejection_stage: str
    components: dict[str, float]
    notes: list[str]


class ConservativeModelV2:
    """
    Weighted scoring модель:
    - hard constraints проверяются снаружи (RR/risk/ATR),
    - soft quality фильтры дают общий score и confidence.
    """

    def __init__(self, model: TradingModel, config: ConservativeV2Config):
        self.model = model
        self.config = config
        self.weights = config.weights.normalized()

    def score(
        self,
        *,
        df: pd.DataFrame,
        direction: str,
        structure: str,
        phase: str,
        trend_strength_pct: float,
        current_price: float,
        ma50: float | None,
        ma200: float | None,
        atr: float,
        rsi: float,
        volume_ratio: float,
        required_volume_ratio: float,
        volume_is_impulse: bool,
        dist_ma50_pct: float,
        max_distance_ma50_pct: float,
        atr_percentile: float,
        disable_volume: bool,
        disable_trend: bool,
        rsi_enabled: bool,
        rsi_trend_confirmation_only: bool,
        allow_range: bool,
        soft_threshold_relaxation: float = 0.0,
    ) -> ConservativeScoreResult:
        slope_score = _slope_strength_score(df=df, atr=atr, price=current_price)

        trend_alignment = self._trend_alignment_score(
            direction=direction,
            ma50=ma50,
            ma200=ma200,
            structure=structure,
            disable_trend=disable_trend,
        )
        trend_strength = self._trend_strength_score(
            trend_strength_pct=trend_strength_pct,
            slope_score=slope_score,
            disable_trend=disable_trend,
        )
        volume = self._volume_score(
            volume_ratio=volume_ratio,
            required_volume_ratio=required_volume_ratio,
            atr_percentile=atr_percentile,
            volume_is_impulse=volume_is_impulse,
            disable_volume=disable_volume,
        )
        rsi_score = self._rsi_score(
            direction=direction,
            rsi=rsi,
            trend_strength=trend_strength,
            rsi_enabled=rsi_enabled,
            rsi_trend_confirmation_only=rsi_trend_confirmation_only,
            phase=phase,
        )
        pullback, pullback_profile = self._pullback_score(
            dist_ma50_pct=dist_ma50_pct,
            phase=phase,
            max_distance_ma50_pct=max_distance_ma50_pct,
        )
        ma_distance = self._ma_distance_score(
            dist_ma50_pct=dist_ma50_pct,
            max_distance_ma50_pct=max_distance_ma50_pct,
        )
        structure_quality = self._structure_score(
            structure=structure,
            allow_range=allow_range,
        )

        components = {
            "trend_alignment": trend_alignment,
            "trend_strength": trend_strength,
            "volume": volume,
            "rsi": rsi_score,
            "pullback": pullback,
            "ma_distance": ma_distance,
            "structure": structure_quality,
        }

        score = (
            self.weights.trend_alignment * trend_alignment
            + self.weights.trend_strength * trend_strength
            + self.weights.volume * volume
            + self.weights.rsi * rsi_score
            + self.weights.pullback * pullback
            + self.weights.ma_distance * ma_distance
            + self.weights.structure * structure_quality
        )

        threshold = self._dynamic_threshold(
            structure=structure,
            atr_percentile=atr_percentile,
            trend_strength=trend_strength,
            soft_threshold_relaxation=soft_threshold_relaxation,
            allow_range=allow_range,
        )
        confidence = _score_to_confidence(score, threshold)
        rejection_stage = _infer_rejection_stage(components)

        notes = [
            (
                "Score "
                f"{score:.3f} (threshold {threshold:.3f}) | "
                f"trend={trend_alignment:.2f}/{trend_strength:.2f} "
                f"vol={volume:.2f} rsi={rsi_score:.2f} "
                f"pullback={pullback:.2f} dist={ma_distance:.2f}"
            ),
            f"Pullback profile: {pullback_profile}",
        ]

        return ConservativeScoreResult(
            total_score=float(score),
            threshold=float(threshold),
            confidence=confidence,
            pullback_profile=pullback_profile,
            rejection_stage=rejection_stage,
            components=components,
            notes=notes,
        )

    def _dynamic_threshold(
        self,
        *,
        structure: str,
        atr_percentile: float,
        trend_strength: float,
        soft_threshold_relaxation: float,
        allow_range: bool,
    ) -> float:
        base = float(self.config.min_score)
        dynamic_bonus = 0.0
        if atr_percentile >= 70.0:
            dynamic_bonus += float(self.config.atr_relaxation_bonus)
        if trend_strength >= 0.80:
            dynamic_bonus += float(self.config.trend_relaxation_bonus)
        if structure == "range" and allow_range:
            dynamic_bonus += float(self.config.range_relaxation_bonus)

        relaxed = base - max(0.0, soft_threshold_relaxation) - dynamic_bonus
        min_floor = base - float(self.config.max_relaxation)
        return max(min_floor, relaxed)

    def _trend_alignment_score(
        self,
        *,
        direction: str,
        ma50: float | None,
        ma200: float | None,
        structure: str,
        disable_trend: bool,
    ) -> float:
        if disable_trend:
            return 1.0
        if ma50 is None or ma200 is None:
            return 0.2
        if structure == "range":
            return 0.75 if self.model.allow_range else 0.25
        if direction == "long":
            return 1.0 if ma50 > ma200 else 0.2
        if direction == "short":
            return 1.0 if ma50 < ma200 else 0.2
        return 0.2

    def _trend_strength_score(
        self,
        *,
        trend_strength_pct: float,
        slope_score: float,
        disable_trend: bool,
    ) -> float:
        if disable_trend:
            return 1.0
        threshold = max(float(self.model.min_trend_strength), 0.1)
        pct_score = _clip01(trend_strength_pct / threshold)
        return _clip01(pct_score * 0.55 + slope_score * 0.45)

    def _volume_score(
        self,
        *,
        volume_ratio: float,
        required_volume_ratio: float,
        atr_percentile: float,
        volume_is_impulse: bool,
        disable_volume: bool,
    ) -> float:
        if disable_volume:
            return 1.0
        required = max(required_volume_ratio, 0.01)
        if atr_percentile >= 70.0:
            required *= 0.85
        raw = volume_ratio / required
        score = _clip01(raw)
        if volume_is_impulse:
            score = _clip01(score + 0.10)
        return score

    def _rsi_score(
        self,
        *,
        direction: str,
        rsi: float,
        trend_strength: float,
        rsi_enabled: bool,
        rsi_trend_confirmation_only: bool,
        phase: str,
    ) -> float:
        if (not rsi_enabled) or (rsi_trend_confirmation_only and phase != "trend"):
            return 1.0

        if direction == "long":
            if 40 <= rsi <= 60:
                score = 1.0
            elif 35 <= rsi <= 70:
                score = 0.75
            elif rsi >= self.model.rsi_overbought:
                score = 0.20
            else:
                score = 0.45
        else:
            if 40 <= rsi <= 60:
                score = 1.0
            elif 30 <= rsi <= 65:
                score = 0.75
            elif rsi <= self.model.rsi_oversold:
                score = 0.20
            else:
                score = 0.45

        # В очень сильном тренде снижаем штраф RSI.
        if trend_strength >= 0.85:
            score = _clip01(score + 0.08)
        return score

    def _pullback_score(
        self,
        *,
        dist_ma50_pct: float,
        phase: str,
        max_distance_ma50_pct: float,
    ) -> tuple[float, str]:
        dist = abs(dist_ma50_pct)
        if phase == "pullback":
            if dist <= 0.3:
                return 0.60, "shallow"
            if dist <= min(2.0, max_distance_ma50_pct):
                return 1.00, "optimal"
            if dist <= max_distance_ma50_pct:
                return 0.65, "deep"
            return 0.20, "invalid"

        if phase == "trend":
            return 0.70, "momentum"
        if phase == "range":
            return 0.75, "range-bounce"
        return 0.40, "unknown"

    @staticmethod
    def _ma_distance_score(*, dist_ma50_pct: float, max_distance_ma50_pct: float) -> float:
        max_dist = max(max_distance_ma50_pct, 0.1)
        ratio = abs(dist_ma50_pct) / max_dist
        return _clip01(1.0 - ratio)

    def _structure_score(self, *, structure: str, allow_range: bool) -> float:
        if structure in {"uptrend", "downtrend"}:
            return 1.0
        if structure == "range":
            return 0.85 if allow_range else 0.25
        return 0.30


def _score_to_confidence(score: float, threshold: float) -> str:
    if score >= max(0.82, threshold + 0.12):
        return "high"
    if score >= max(0.68, threshold + 0.02):
        return "medium"
    if score >= threshold:
        return "low"
    return "none"


def _infer_rejection_stage(components: dict[str, float]) -> str:
    if components.get("volume", 1.0) < 0.35:
        return "filtered_volume"
    if min(
        components.get("trend_alignment", 1.0),
        components.get("trend_strength", 1.0),
        components.get("structure", 1.0),
    ) < 0.35:
        return "filtered_trend"
    return "no_setup"


def _slope_strength_score(df: pd.DataFrame, atr: float, price: float) -> float:
    if "ma50" not in df.columns or "close" not in df.columns or len(df) < 80:
        return 0.5

    ma50 = df["ma50"].dropna()
    close = df["close"].dropna()
    if len(ma50) < 40 or len(close) < 40:
        return 0.5

    now = float(ma50.iloc[-1])
    prev = float(ma50.iloc[-20]) if len(ma50) >= 21 else float(ma50.iloc[0])
    raw_delta = now - prev

    atr_safe = max(float(atr), max(float(price), 1.0) * 1e-5)
    atr_norm = abs(raw_delta) / atr_safe
    atr_norm_score = _clip01(atr_norm / 3.0)

    returns = close.pct_change().dropna().tail(100)
    vol = float(returns.std()) if not returns.empty else 0.0
    slope_pct = abs(raw_delta) / max(abs(prev), 1e-9)
    vol_norm_score = _clip01(slope_pct / max(vol * np.sqrt(20), 1e-5))

    return _clip01(atr_norm_score * 0.55 + vol_norm_score * 0.45)


def should_use_conservative_v2(model_name: str, filter_config: dict[str, Any] | None) -> bool:
    filters = (filter_config or {}).get("filters", {})
    scoring = filters.get("scoring", {})
    enabled = bool(scoring.get("conservative_v2_enabled", True))
    if not enabled:
        return False
    normalized = model_name.lower()
    return "conservative" in normalized


def load_conservative_v2_config(filter_config: dict[str, Any] | None) -> ConservativeV2Config:
    filters = (filter_config or {}).get("filters", {})
    scoring = filters.get("scoring", {})
    raw_weights = scoring.get("weights", {}) if isinstance(scoring.get("weights"), dict) else {}
    weights = ScoringWeights(
        trend_alignment=float(raw_weights.get("trend_alignment", 0.25)),
        trend_strength=float(raw_weights.get("trend_strength", 0.25)),
        volume=float(raw_weights.get("volume", 0.15)),
        rsi=float(raw_weights.get("rsi", 0.10)),
        pullback=float(raw_weights.get("pullback", 0.25)),
        ma_distance=float(raw_weights.get("ma_distance", 0.0)),
        structure=float(raw_weights.get("structure", 0.0)),
    )
    return ConservativeV2Config(
        enabled=bool(scoring.get("conservative_v2_enabled", True)),
        min_score=float(scoring.get("conservative_min_score", 0.65)),
        max_relaxation=float(scoring.get("max_threshold_relaxation", 0.20)),
        atr_relaxation_bonus=float(scoring.get("atr_relaxation_bonus", 0.08)),
        trend_relaxation_bonus=float(scoring.get("trend_relaxation_bonus", 0.05)),
        range_relaxation_bonus=float(scoring.get("range_relaxation_bonus", 0.03)),
        min_expected_trades_per_month=float(scoring.get("min_expected_trades_per_month", 35.0)),
        weights=weights,
    )
