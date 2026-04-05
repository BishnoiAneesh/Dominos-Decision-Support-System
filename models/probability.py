"""
models/probability.py
---------------------
Computes P(total_delivery_time ≤ SLA) under pluggable distributional
assumptions.

System decomposition
--------------------
Total delivery time has three components:

    T = queue_delay + prep_time + delivery_time

Under the batch-based store model:
  - queue_delay and prep_time are deterministic (batch ceiling arithmetic).
  - delivery_time is stochastic ~ Normal(μ_d, σ_d²).

Therefore:
    total_mean     = queue_delay + prep_time + μ_d   (all deterministic except delivery)
    total_variance = σ_d²                            (only delivery contributes variance)

P(T ≤ SLA) is computed via a Normal CDF on these parameters.
The public interface accepts (expected_time, variance, sla_threshold) so callers
compose the mean themselves — this module stays a pure probability layer.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Protocol — interface every backend must satisfy
# ---------------------------------------------------------------------------

@runtime_checkable
class SLAProbabilityModel(Protocol):
    """
    Interface for SLA probability backends.
    Any class implementing `prob_within_sla` qualifies — no inheritance needed.
    """

    def prob_within_sla(
        self,
        expected_time: float,
        variance: float,
        sla_threshold: float,
    ) -> float:
        """
        Return P(T ≤ sla_threshold).

        Args:
            expected_time:  E[T] = queue_delay + prep_time + E[delivery_time] (minutes).
            variance:       Var[T] = Var[delivery_time] only (minutes²).
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

    Because queue_delay and prep_time are deterministic, variance comes
    entirely from delivery_time. The Normal assumption holds well when
    travel distances are moderate and speed noise is Gaussian.

    Degenerates cleanly to a step function when σ → 0 (e.g. fixed-speed
    courier or zero-distance delivery).
    """

    def prob_within_sla(
        self,
        expected_time: float,
        variance: float,
        sla_threshold: float,
    ) -> float:
        std_dev = math.sqrt(max(variance, 0.0))
        if std_dev < 1e-9:
            # Fully deterministic: binary outcome
            return 1.0 if expected_time <= sla_threshold else 0.0
        return float(norm.cdf(sla_threshold, loc=expected_time, scale=std_dev))


# ---------------------------------------------------------------------------
# Backend 2: Monte Carlo
# ---------------------------------------------------------------------------

class MonteCarloSLAModel:
    """
    Empirical P(T ≤ SLA) via Monte Carlo sampling over delivery_time.

    Samples delivery_time ~ Normal(μ_d, σ_d²) and adds the deterministic
    queue_delay + prep_time offset via expected_time. Useful for validating
    the Normal approximation or when delivery noise is non-Gaussian.

    Args:
        n_samples:  Number of draws per query (higher = more accurate, slower).
        rng:        Seeded numpy Generator for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_samples = n_samples
        self.rng       = rng or np.random.default_rng()

    def prob_within_sla(
        self,
        expected_time: float,
        variance: float,
        sla_threshold: float,
    ) -> float:
        std_dev = math.sqrt(max(variance, 0.0))
        # Sample total time: deterministic offset already baked into expected_time
        samples = self.rng.normal(
            loc   = expected_time,
            scale = max(std_dev, 1e-9),
            size  = self.n_samples,
        )
        return float(np.mean(samples <= sla_threshold))


# ---------------------------------------------------------------------------
# Primary public API
# ---------------------------------------------------------------------------

def prob_within_sla(
    expected_time: float,
    variance: float,
    sla_threshold: float,
    model: SLAProbabilityModel | None = None,
) -> float:
    """
    Compute P(total_delivery_time ≤ sla_threshold).

    Callers should compose expected_time as:
        expected_time = queue_delay + prep_time + expected_delivery_time

    And variance as:
        variance = delivery_variance   # queue/prep are deterministic

    Args:
        expected_time:  E[T] in minutes (deterministic + stochastic components).
        variance:       Var[T] in minutes² (delivery noise only).
        sla_threshold:  SLA deadline in minutes.
        model:          Probability backend. Defaults to NormalSLAModel.

    Returns:
        Probability in [0.0, 1.0].

    Examples:
        >>> # queue_delay=5, prep_time=8, E[delivery]=14, Var[delivery]=9
        >>> prob_within_sla(expected_time=27.0, variance=9.0, sla_threshold=30.0)
        0.8413...
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

    Args:
        probability_target: Required confidence level, e.g. 0.95.

    Convenience wrapper for binary feasibility checks in the assignment engine.
    """
    return prob_within_sla(expected_time, variance, sla_threshold, model) >= probability_target


def decompose_and_evaluate(
    queue_delay:       float,
    prep_time:         float,
    expected_delivery: float,
    delivery_variance: float,
    sla_threshold:     float,
    probability_target: float,
    model: SLAProbabilityModel | None = None,
) -> tuple[float, float, bool]:
    """
    Convenience function that accepts the three pipeline components explicitly.

    Composes total_mean and total_variance internally, then evaluates SLA.
    Useful in contexts where components are already computed separately
    (e.g. diagnostics, UI breakdown tables).

    Args:
        queue_delay:        Deterministic queue wait (minutes).
        prep_time:          Deterministic batch prep time (minutes).
        expected_delivery:  E[delivery_time] (minutes).
        delivery_variance:  Var[delivery_time] (minutes²).
        sla_threshold:      SLA deadline (minutes).
        probability_target: Required P(T ≤ SLA) to be considered feasible.
        model:              Probability backend. Defaults to NormalSLAModel.

    Returns:
        (total_mean, sla_probability, meets_target) as a named tuple.
    """
    total_mean = queue_delay + prep_time + expected_delivery
    p          = prob_within_sla(total_mean, delivery_variance, sla_threshold, model)
    return total_mean, p, p >= probability_target