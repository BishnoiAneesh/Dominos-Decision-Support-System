"""
delivery.py
-----------
Road-network delivery time estimation backed by precomputed Dijkstra distances.

All distance calculations are purely graph-based (OSMnx + NetworkX).
There is NO Euclidean fallback anywhere in this module.

Precomputed distances
---------------------
Call ``precompute_store_distances(stores, graph)`` once before the simulation
loop.  It runs ``networkx.single_source_dijkstra_path_length`` from each
store's nearest road node and stores the result as:

    StoreDistanceIndex = Dict[store_id, Dict[node_id, distance_metres]]

Pass the returned index into ``estimate_delivery_time()`` via the
``store_node_distances`` argument.  Per-order distance lookups then become
an O(1) dict access rather than a full shortest-path query.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from config import DeliveryConfig, RandomnessConfig


Point = Tuple[float, float]

# Type aliases
NodeDistances      = Dict[int, float]          # node_id  → distance in metres
StoreDistanceIndex = Dict[int, NodeDistances]  # store_id → NodeDistances


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeliveryEstimate:
    """
    Result of a single delivery time estimation.

    Attributes:
        expected_time:  E[T] — mean delivery duration (minutes).
        variance:       Var[T] — variance of delivery duration (minutes²).
        distance_km:    Road-network distance between origin and destination (km).
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


# ---------------------------------------------------------------------------
# Node snapping (internal helper)
# ---------------------------------------------------------------------------

def _nearest_node(graph: object, lat: float, lon: float) -> int:
    """Return the id of the OSMnx graph node nearest to (lat, lon)."""
    try:
        import osmnx as ox
    except ImportError as exc:
        raise ImportError(
            "OSMnx is required for road-network delivery estimation. "
            "Install it with: pip install osmnx"
        ) from exc
    # ox.nearest_nodes expects (X=lon, Y=lat)
    return int(ox.nearest_nodes(graph, lon, lat))


# ---------------------------------------------------------------------------
# Precomputation
# ---------------------------------------------------------------------------

def precompute_store_distances(
    stores: list,   # List[Store] — string annotation avoids circular import
    graph:  object,
) -> StoreDistanceIndex:
    """
    Run single-source Dijkstra from each store node and cache all results.

    This is the **only** place shortest-path computation happens.
    The returned index is passed to ``estimate_delivery_time()`` so that
    every subsequent distance lookup is a simple dict ``get()``.

    Args:
        stores: List of Store objects (need ``.id`` and ``.location``).
        graph:  OSMnx MultiDiGraph with ``'length'`` edge weights (metres).

    Returns:
        ``{store_id: {node_id: distance_metres, ...}, ...}``

    Raises:
        ImportError: If NetworkX or OSMnx is not installed.
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "NetworkX is required for distance precomputation. "
            "Install it with: pip install networkx"
        ) from exc

    index: StoreDistanceIndex = {}
    for store in stores:
        lat, lon   = store.location
        store_node = _nearest_node(graph, lat, lon)
        distances  = nx.single_source_dijkstra_path_length(
            graph, store_node, weight="length"
        )
        # nx returns a generator-like dict; materialise it once
        index[store.id] = dict(distances)
    return index


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

def estimate_delivery_time(
    origin_store_id:      int,
    destination:          Point,
    delivery_cfg:         DeliveryConfig,
    randomness_cfg:       RandomnessConfig,
    graph:                object,
    store_node_distances: StoreDistanceIndex,
    rng:                  np.random.Generator | None = None,
) -> DeliveryEstimate:
    """
    Estimate delivery time using a precomputed road-network distance index.

    Args:
        origin_store_id:      ID of the dispatching store.
        destination:          (lat, lon) of the delivery address.
        delivery_cfg:         Speed and variance parameters.
        randomness_cfg:       Noise controls.
        graph:                OSMnx MultiDiGraph — used only to snap the
                              destination to its nearest road node.
        store_node_distances: Output of ``precompute_store_distances()``.
        rng:                  Optional numpy Generator for a stochastic sample.
                              When provided, ``expected_time`` is a single draw
                              from Normal(base_time, noise_std); otherwise it
                              equals the deterministic base_time.

    Returns:
        DeliveryEstimate with expected_time, variance, and distance_km.

    Notes:
        If the destination node is unreachable from the store node (graph
        disconnected), the maximum known distance for that store is used as a
        conservative upper bound so the simulation degrades gracefully.
    """
    dest_node  = _nearest_node(graph, destination[0], destination[1])
    node_dists = store_node_distances[origin_store_id]

    distance_m = node_dists.get(dest_node)
    if distance_m is None:
        # Conservative fallback for disconnected nodes
        distance_m = max(node_dists.values()) if node_dists else 0.0

    distance_km = distance_m / 1000.0
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


# ---------------------------------------------------------------------------
# Economic helper
# ---------------------------------------------------------------------------

def compute_delivery_cost(distance_km: float, cost_per_km: float) -> float:
    """Return the monetary cost of a delivery given distance and per-km rate."""
    return distance_km * cost_per_km


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def batch_estimate(
    origin_store_id:      int,
    destinations:         List[Point],
    delivery_cfg:         DeliveryConfig,
    randomness_cfg:       RandomnessConfig,
    graph:                object,
    store_node_distances: StoreDistanceIndex,
    rng:                  np.random.Generator | None = None,
) -> List[DeliveryEstimate]:
    """Estimate delivery times from one store origin to multiple destinations."""
    return [
        estimate_delivery_time(
            origin_store_id      = origin_store_id,
            destination          = dest,
            delivery_cfg         = delivery_cfg,
            randomness_cfg       = randomness_cfg,
            graph                = graph,
            store_node_distances = store_node_distances,
            rng                  = rng,
        )
        for dest in destinations
    ]