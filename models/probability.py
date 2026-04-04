"""
models/probability.py
---------------------
Computes P(total_delivery_time ≤ SLA) under pluggable distributional
assumptions. Default: Normal approximation (closed-form, fast).
Designed so the backend can be swapped for Monte Carlo or empirical
methods without changing the calling interface.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable
import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Protocol — defines the interface every backend must satisfy
# ---------------------------------------------------------------------------

@runtime_checkable
class SLAProbabilityModel(Protocol):
    """
    Interface for SLA probability backends.
    Any callable class implementing `prob_within_sla` qualifies.
    """

    def prob_within_sla(
        self,
        expected_time: float,
        variance: float,
        sla_threshold: float,
    ) -> float:
        """
        Return P(T ≤ sla_threshold) given the distribution parameters.

        Args:
            expected_time:  E[T] in minutes.
            variance:       Var[T] in minutes².
            sla_threshold:  SLA deadline in minutes.

        Returns:
            Probability in [0.0, 1.0].
        """
        ...


# ---------------------------------------------------------------------------
# Backend 1: Normal approximation (default)
# ---------------------------------------------------------------------------

class NormalSLAModel:
    """
    Closed-form P(T ≤ SLA) assuming T ~ Normal(μ, σ²).
    Fast and analytically tractable; reasonable when σ is small relative to μ.
    """

    def prob_within_sla(
        self,
        expected_time: float,
        variance: float,
        sla_threshold: float,
    ) -> float:
        std_dev = math.sqrt(variance)
        if std_dev < 1e-9:
            # Degenerate: deterministic delivery time
            return 1.0 if expected_time <= sla_threshold else 0.0
        return float(norm.cdf(sla_threshold, loc=expected_time, scale=std_dev))


# ---------------------------------------------------------------------------
# Backend 2: Monte Carlo (placeholder — swap in when needed)
# ---------------------------------------------------------------------------

class MonteCarloSLAModel:
    """
    Empirical P(T ≤ SLA) via Monte Carlo sampling.
    More accurate when the Normal assumption breaks down (heavy tails,
    skewed prep times, etc.).

    Args:
        n_samples:  Number of simulation draws.
        rng:        Seeded numpy Generator for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_samples = n_samples
        self.rng = rng or np.random.default_rng()

    def prob_within_sla(
        self,
        expected_time: float,
        variance: float,
        sla_threshold: float,
    ) -> float:
        std_dev = math.sqrt(variance)
        samples = self.rng.normal(
            loc=expected_time,
            scale=std_dev if std_dev > 1e-9 else 1e-9,
            size=self.n_samples,
        )
        return float(np.mean(samples <= sla_threshold))


# ---------------------------------------------------------------------------
# Primary public function — thin dispatch layer
# ---------------------------------------------------------------------------

def prob_within_sla(
    expected_time: float,
    variance: float,
    sla_threshold: float,
    model: SLAProbabilityModel | None = None,
) -> float:
    """
    Compute P(total_delivery_time ≤ sla_threshold).

    Args:
        expected_time:  E[T] — mean total time (minutes).
        variance:       Var[T] — variance of total time (minutes²).
        sla_threshold:  SLA deadline (minutes), e.g. 30.0.
        model:          Backend to use. Defaults to NormalSLAModel.

    Returns:
        Probability in [0.0, 1.0].

    Examples:
        >>> prob_within_sla(expected_time=25.0, variance=9.0, sla_threshold=30.0)
        0.9522...
        >>> prob_within_sla(25.0, 9.0, 30.0, model=MonteCarloSLAModel(n_samples=50_000))
        ~0.952  # converges with more samples
    """
    if model is None:
        model = NormalSLAModel()
    return model.prob_within_sla(expected_time, variance, sla_threshold)


def meets_sla_target(
    expected_time: float,
    variance: float,
    sla_threshold: float,
    probability_target: float,
    model: SLAProbabilityModel | None = None,
) -> bool:
    """
    Return True if P(T ≤ sla_threshold) ≥ probability_target.
    Convenience wrapper for binary SLA feasibility checks.

    Args:
        probability_target: Required confidence level, e.g. 0.95.
    """
    p = prob_within_sla(expected_time, variance, sla_threshold, model)
    return p >= probability_target