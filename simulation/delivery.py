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
    origin:         Point,
    destination:    Point,
    delivery_cfg:   DeliveryConfig,
    randomness_cfg: RandomnessConfig,
    rng:            np.random.Generator | None = None,
    graph:          object | None = None,
    use_network_distance: bool = False,
) -> DeliveryEstimate:
    """
    Estimate delivery time between two points with distance-scaled noise.

    Args:
        origin:               Departure point (store location) as (lat, lon) or (x, y).
        destination:          Delivery point (order location).
        delivery_cfg:         Speed and radius parameters.
        randomness_cfg:       Noise controls.
        rng:                  Optional numpy Generator for a stochastic sample.
        graph:                OSMnx MultiDiGraph for road-network routing.
                              Required when use_network_distance=True.
        use_network_distance: If True, use shortest road-network path distance.
                              Falls back to Euclidean if graph is None.

    Returns:
        DeliveryEstimate with expected_time, variance, and distance_km.
    """
    if use_network_distance and graph is not None:
        distance_km = network_distance_km(origin, destination, graph)
    else:
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


def network_distance_km(
    origin:      Point,
    destination: Point,
    graph:       object,
) -> float:
    """
    Compute shortest-path road distance (km) between two (lat, lon) points.

    Uses OSMnx to snap both points to the nearest road nodes, then queries
    the shortest weighted path via NetworkX.

    Args:
        origin:      (lat, lon) of the departure point.
        destination: (lat, lon) of the delivery address.
        graph:       Pre-loaded OSMnx MultiDiGraph.

    Returns:
        Road distance in kilometres. Falls back to Euclidean distance if
        no path exists between the snapped nodes.
    """
    try:
        import osmnx as ox
        import networkx as nx
    except ImportError as e:
        raise ImportError("OSMnx and NetworkX are required for network routing.") from e

    # ox.nearest_nodes expects (X=lon, Y=lat)
    orig_node = ox.nearest_nodes(graph, origin[1],      origin[0])
    dest_node = ox.nearest_nodes(graph, destination[1], destination[0])

    if orig_node == dest_node:
        return 0.0

    try:
        length_m = nx.shortest_path_length(graph, orig_node, dest_node, weight="length")
        return length_m / 1000.0
    except nx.NetworkXNoPath:
        # Fallback to Euclidean if graph is disconnected at these nodes
        return euclidean_distance(origin, destination)


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