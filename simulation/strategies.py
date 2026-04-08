"""
strategies.py
-------------
Assignment strategy hierarchy for store selection.

All distance calculations use precomputed road-network distances
(StoreDistanceIndex from simulation.delivery).
There is NO Euclidean / straight-line distance anywhere in this module.

The engine calls ``strategy.set_distance_index(index)`` once after
precomputation, before the order-processing loop begins.

Extend by subclassing AssignmentStrategy and implementing select_store().
Register new strategies in STRATEGY_REGISTRY for UI dropdown support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from config import SimConfig
from simulation.order import Order
from simulation.store import Store
from simulation.delivery import (
    estimate_delivery_time,
    compute_delivery_cost,
    StoreDistanceIndex,
    _nearest_node,
)
from simulation.assignment_types import AssignmentResult, StoreEvaluation
from models.probability import (
    prob_within_sla,
    meets_sla_target,
    NormalSLAModel,
    SLAProbabilityModel,
)


# ---------------------------------------------------------------------------
# Shared evaluation helper
# ---------------------------------------------------------------------------

def _full_evaluation(
    order:                Order,
    store:                Store,
    config:               SimConfig,
    graph:                object,
    store_node_distances: StoreDistanceIndex,
    assignment_cost:      float = 0.0,
) -> StoreEvaluation:
    """
    Compute a complete StoreEvaluation for one (order, store) pair.

    All distances come from the precomputed ``store_node_distances`` index.
    The ``assignment_cost`` field is injected by the calling strategy so
    each strategy can define its own objective without duplicating the
    pipeline logic here.
    """
    queue_delay  = store.estimate_queue_delay(order, order.arrival_time)
    prep_time    = store.estimate_prep_time(order)
    delivery_est = estimate_delivery_time(
        origin_store_id      = store.id,
        destination          = order.location,
        delivery_cfg         = config.delivery,
        randomness_cfg       = config.randomness,
        graph                = graph,
        store_node_distances = store_node_distances,
    )

    expected_total = queue_delay + prep_time + delivery_est.expected_time
    total_variance = delivery_est.variance

    p_sla   = prob_within_sla(
        expected_time = expected_total,
        variance      = total_variance,
        sla_threshold = config.sla.max_delivery_minutes,
    )
    sla_met = meets_sla_target(
        expected_time      = expected_total,
        variance           = total_variance,
        sla_threshold      = config.sla.max_delivery_minutes,
        probability_target = config.sla.probability_threshold,
    )

    return StoreEvaluation(
        store            = store,
        queue_delay      = queue_delay,
        prep_time        = prep_time,
        travel_time      = delivery_est.expected_time,
        expected_total   = expected_total,
        total_variance   = total_variance,
        sla_probability  = p_sla,
        meets_sla_target = sla_met,
        assignment_cost  = assignment_cost,
        # Stash distance_km on the evaluation for the economic objective
        # by re-using delivery_est.distance_km via a helper below
    )


def _road_distance_km(
    store:                Store,
    order:                Order,
    graph:                object,
    store_node_distances: StoreDistanceIndex,
) -> float:
    """
    Road-network distance (km) from a store to an order location.

    Precompute mode: O(1) lookup from the full node-distance dict.
    Real-time mode:  on-the-fly Dijkstra using the stored node id.
    """
    dest_node = _nearest_node(graph, order.location[0], order.location[1])

    if store.id in store_node_distances:
        # Precompute mode
        node_dists = store_node_distances[store.id]
        distance_m = node_dists.get(dest_node)
        if distance_m is None:
            distance_m = max(node_dists.values()) if node_dists else 0.0
    else:
        # Real-time mode: use stored node id and run targeted Dijkstra
        import networkx as nx
        store_node = store_node_distances.get(f"_node_{store.id}")
        if store_node is None:
            return 0.0
        try:
            distance_m, _ = nx.single_source_dijkstra(
                graph, store_node, dest_node, weight="length"
            )
        except Exception:
            distance_m = 0.0

    return distance_m / 1000.0


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class AssignmentStrategy(ABC):
    """
    Base class for all store-assignment strategies.

    The engine injects the precomputed distance index via
    ``set_distance_index()`` once before the simulation loop, so strategies
    never need to trigger shortest-path computation themselves.
    """

    def __init__(self, graph: object) -> None:
        self._graph:          object             = graph
        self._distance_index: StoreDistanceIndex = {}

    def set_distance_index(self, index: StoreDistanceIndex) -> None:
        """Called once by the engine after ``precompute_store_distances()``."""
        self._distance_index = index

    @property
    def name(self) -> str:
        """Human-readable strategy name (used in results and UI)."""
        return self.__class__.__name__

    @abstractmethod
    def select_store(
        self,
        order:  Order,
        stores: List[Store],
        config: SimConfig,
    ) -> AssignmentResult:
        """
        Choose the best store for an incoming order.

        Args:
            order:   The order to be assigned.
            stores:  All candidate stores.
            config:  Master simulation config.

        Returns:
            AssignmentResult with selected store and full diagnostics.

        Raises:
            ValueError: If stores is empty.
        """
        ...


# ---------------------------------------------------------------------------
# Strategy 1 — Nearest Store (baseline)
# ---------------------------------------------------------------------------

class NearestStoreStrategy(AssignmentStrategy):
    """
    Selects the store with the shortest road-network distance to the order.

    Selection uses the O(1) precomputed distance lookup.
    Full pipeline metrics are computed for all stores after selection so
    the AssignmentResult carries complete diagnostics.
    """

    def __init__(self, graph: object) -> None:
        super().__init__(graph)

    def select_store(
        self,
        order:  Order,
        stores: List[Store],
        config: SimConfig,
    ) -> AssignmentResult:
        if not stores:
            raise ValueError("NearestStoreStrategy requires at least one store.")

        nearest = min(
            stores,
            key=lambda s: _road_distance_km(s, order, self._graph, self._distance_index),
        )

        evaluations = [
            _full_evaluation(
                order                = order,
                store                = s,
                config               = config,
                graph                = self._graph,
                store_node_distances = self._distance_index,
                assignment_cost      = _road_distance_km(
                    s, order, self._graph, self._distance_index
                ),
            )
            for s in stores
        ]
        selected_ev = next(e for e in evaluations if e.store.id == nearest.id)

        return AssignmentResult(
            order           = order,
            selected_store  = nearest,
            evaluation      = selected_ev,
            all_evaluations = evaluations,
            feasible        = selected_ev.meets_sla_target,
            strategy_name   = self.name,
        )


# ---------------------------------------------------------------------------
# Strategy 2 — Optimized (economic objective)
# ---------------------------------------------------------------------------

class OptimizedStrategy(AssignmentStrategy):
    """
    Selects the store minimising an economic objective:

        delivery_cost    = distance_km * cost_per_km
        expected_penalty = (1 - P(SLA)) * sla_penalty_factor * order_value
        objective        = delivery_cost + expected_penalty

    Filters to stores where P(SLA) >= probability_threshold first.
    Falls back to highest P(SLA) if no store meets the threshold.
    """

    def __init__(
        self,
        graph:             object,
        probability_model: SLAProbabilityModel | None = None,
    ) -> None:
        super().__init__(graph)
        self.probability_model = probability_model or NormalSLAModel()

    def select_store(
        self,
        order:  Order,
        stores: List[Store],
        config: SimConfig,
    ) -> AssignmentResult:
        if not stores:
            raise ValueError("OptimizedStrategy requires at least one store.")

        eco   = config.economics
        value = order.order_value(eco.main_item_price, eco.side_item_price)

        evaluations = []
        for store in stores:
            distance_km = _road_distance_km(
                store, order, self._graph, self._distance_index
            )

            ev = _full_evaluation(
                order                = order,
                store                = store,
                config               = config,
                graph                = self._graph,
                store_node_distances = self._distance_index,
                assignment_cost      = 0.0,   # will be replaced below
            )

            delivery_cost    = compute_delivery_cost(distance_km, eco.cost_per_km)
            expected_penalty = (1.0 - ev.sla_probability) * eco.sla_penalty_factor * value
            objective        = delivery_cost + expected_penalty

            evaluations.append(StoreEvaluation(
                store            = ev.store,
                queue_delay      = ev.queue_delay,
                prep_time        = ev.prep_time,
                travel_time      = ev.travel_time,
                expected_total   = ev.expected_total,
                total_variance   = ev.total_variance,
                sla_probability  = ev.sla_probability,
                meets_sla_target = ev.meets_sla_target,
                assignment_cost  = objective,
            ))

        feasible = [e for e in evaluations if e.meets_sla_target]

        if feasible:
            selected    = min(feasible, key=lambda e: e.assignment_cost)
            is_feasible = True
        else:
            selected    = max(evaluations, key=lambda e: e.sla_probability)
            is_feasible = False

        return AssignmentResult(
            order           = order,
            selected_store  = selected.store,
            evaluation      = selected,
            all_evaluations = evaluations,
            feasible        = is_feasible,
            strategy_name   = self.name,
        )


# ---------------------------------------------------------------------------
# Strategy registry — UI dropdown name → class
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, type[AssignmentStrategy]] = {
    "nearest":   NearestStoreStrategy,
    "optimized": OptimizedStrategy,
}