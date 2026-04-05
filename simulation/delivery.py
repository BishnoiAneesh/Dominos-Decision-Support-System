"""
delivery.py
-----------
Analytical delivery time estimation with configurable stochastic noise.
Returns both expected time and variance to support downstream probability modelling.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from config import DeliveryConfig, RandomnessConfig


Point = Tuple[float, float]


@dataclass(frozen=True)
class DeliveryEstimate:
    """
    Result of a single delivery time estimation.

    Attributes:
        expected_time:  E[T] — mean delivery duration (minutes).
        variance:       Var[T] — variance of delivery duration (minutes²).
        distance_km:    Euclidean distance between origin and destination.
    """
    expected_time: float
    variance:      float
    distance_km:   float

    @property
    def std_dev(self) -> float:
        return math.sqrt(self.variance)

    def prob_within_sla(self, sla_minutes: float) -> float:
        """P(T ≤ sla_minutes) under a Normal approximation."""
        if self.std_dev < 1e-9:
            return 1.0 if self.expected_time <= sla_minutes else 0.0
        from scipy.stats import norm
        return float(norm.cdf(sla_minutes, loc=self.expected_time, scale=self.std_dev))


def euclidean_distance(a: Point, b: Point) -> float:
    """Euclidean distance between two (x, y) coordinate pairs."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def estimate_delivery_time(
    origin: Point,
    destination: Point,
    delivery_cfg: DeliveryConfig,
    randomness_cfg: RandomnessConfig,
    rng: np.random.Generator | None = None,
) -> DeliveryEstimate:
    """
    Estimate delivery time between two points with distance-scaled noise.

    Args:
        origin:          Departure point (store location).
        destination:     Delivery point (order location).
        delivery_cfg:    Speed and radius parameters.
        randomness_cfg:  Noise controls.
        rng:             Optional numpy Generator for a stochastic sample.

    Returns:
        DeliveryEstimate with expected_time, variance, and distance_km.
    """
    distance_km = euclidean_distance(origin, destination)
    speed_kmpm  = delivery_cfg.average_speed_kmph / 60.0
    base_time   = distance_km / speed_kmpm if speed_kmpm > 0 else float("inf")

    noise_std = (
        randomness_cfg.travel_time_noise_std
        + delivery_cfg.speed_variance * distance_km / delivery_cfg.average_speed_kmph
    )
    variance = noise_std ** 2

    if rng is not None:
        expected_time = max(float(rng.normal(loc=base_time, scale=noise_std)), 0.0)
    else:
        expected_time = base_time

    return DeliveryEstimate(
        expected_time=expected_time,
        variance=variance,
        distance_km=distance_km,
    )


def compute_delivery_cost(distance_km: float, cost_per_km: float) -> float:
    """Return the monetary cost of a delivery given distance and per-km rate."""
    return distance_km * cost_per_km


def batch_estimate(
    origin: Point,
    destinations: list[Point],
    delivery_cfg: DeliveryConfig,
    randomness_cfg: RandomnessConfig,
    rng: np.random.Generator | None = None,
) -> list[DeliveryEstimate]:
    """Estimate delivery times from one origin to multiple destinations."""
    return [
        estimate_delivery_time(origin, dest, delivery_cfg, randomness_cfg, rng)
        for dest in destinations
    ]