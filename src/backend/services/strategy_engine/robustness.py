"""
Оценка переобучения и устойчивости стратегии
"""

from dataclasses import dataclass
from typing import Any, Optional

from .monte_carlo import MonteCarloResult


@dataclass
class RobustnessMetrics:
    """Метрики устойчивости модели"""
    pf_train: float
    pf_test: float
    maxdd_test: float
    stability_ratio: float
    unstable_oos: bool
    overfit: bool
    monte_carlo_stability: float
    robustness_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            'pf_train': round(self.pf_train, 2),
            'pf_test': round(self.pf_test, 2),
            'maxdd_test': round(self.maxdd_test, 2),
            'stability_ratio': round(self.stability_ratio, 2),
            'unstable_oos': self.unstable_oos,
            'overfit': self.overfit,
            'monte_carlo_stability': round(self.monte_carlo_stability, 4),
            'robustness_score': round(self.robustness_score, 4),
        }


def evaluate_robustness(
    pf_train: float,
    pf_test: float,
    maxdd_test: float,
    monte_carlo: Optional[MonteCarloResult] = None
) -> RobustnessMetrics:
    stability_ratio = calculate_stability_ratio(pf_train, pf_test)
    unstable_oos = is_unstable_oos(pf_train, pf_test)
    overfit = stability_ratio < 0.7
    monte_carlo_stability = monte_carlo.stability_score if monte_carlo else 0.0

    robustness_score = calculate_robustness_score(
        pf_test=pf_test,
        maxdd_test=maxdd_test,
        stability_ratio=stability_ratio,
        monte_carlo_stability=monte_carlo_stability
    )

    return RobustnessMetrics(
        pf_train=pf_train,
        pf_test=pf_test,
        maxdd_test=maxdd_test,
        stability_ratio=stability_ratio,
        unstable_oos=unstable_oos,
        overfit=overfit,
        monte_carlo_stability=monte_carlo_stability,
        robustness_score=robustness_score
    )


def calculate_stability_ratio(pf_train: float, pf_test: float) -> float:
    if pf_train <= 0:
        return 0.0
    return pf_test / pf_train


def is_unstable_oos(pf_train: float, pf_test: float) -> bool:
    return pf_train > 1.5 and pf_test < 1.2


def calculate_robustness_score(
    pf_test: float,
    maxdd_test: float,
    stability_ratio: float,
    monte_carlo_stability: float
) -> float:
    """
    Robustness Score =
      (PF_test * 0.4) +
      ((1 - MaxDD) * 0.3) +
      (Stability * 0.3)

    where MaxDD is normalized to [0..1] from percent.
    """
    maxdd_normalized = max(0.0, min(1.0, maxdd_test / 100.0))
    return (
        (pf_test * 0.4)
        + ((1 - maxdd_normalized) * 0.3)
        + (stability_ratio * 0.3)
    )
