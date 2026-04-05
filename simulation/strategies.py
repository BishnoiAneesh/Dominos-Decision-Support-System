"""
strategies.py
-------------
Assignment strategy hierarchy for store selection.

Extend by subclassing AssignmentStrategy and implementing select_store().
Register new strategies in STRATEGY_REGISTRY for UI dropdown support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from config import SimConfig
from simulation.order import Order
from simulation.store import Store
from simulation.delivery import estimate_delivery_time, euclidean_distance, compute_delivery_cost
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
    order:             Order,
    store:             Store,
    config:            SimConfig,
    assignment_cost:   float = 0.0,
) -> StoreEvaluation:
    """
    Compute a complete StoreEvaluation for one (order, store) pair.
    assignment_cost is passed in by the calling strategy so each strategy
    can use its own objective without duplicating the pipeline logic.
    """
    queue_delay  = store.estimate_queue_delay(order)
    prep_time    = store.estimate_prep_time(order)
    delivery_est = estimate_delivery_time(
        origin         = store.location,
        destination    = order.location,
        delivery_cfg   = config.delivery,
        randomness_cfg = config.randomness,
        rng            = None,
    )

    expected_total = queue_delay + prep_time + delivery_est.expected_time
    total_variance = delivery_est.variance

    p_sla = prob_within_sla(
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
    )


def _distance_to(store: Store, order: Order) -> float:
    return euclidean_distance(store.location, order.location)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class AssignmentStrategy(ABC):
    """
    Base class for all store-assignment strategies.
    Subclasses implement select_store() and return a fully populated
    AssignmentResult. The engine calls this once per order.
    """

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
# Strategy 1 — Nearest Store (baseline, unchanged)
# ---------------------------------------------------------------------------

class NearestStoreStrategy(AssignmentStrategy):
    """
    Selects the store with minimum Euclidean distance to the order.
    Full pipeline metrics are computed after selection for accurate reporting.
    """

    def select_store(
        self,
        order:  Order,
        stores: List[Store],
        config: SimConfig,
    ) -> AssignmentResult:
        if not stores:
            raise ValueError("NearestStoreStrategy requires at least one store.")

        nearest = min(stores, key=lambda s: _distance_to(s, order))

        evaluations = [
            _full_evaluation(order, s, config, assignment_cost=_distance_to(s, order))
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

    Args:
        probability_model: Pluggable SLA probability backend.
    """

    def __init__(
        self,
        probability_model: SLAProbabilityModel | None = None,
    ) -> None:
        self.probability_model = probability_model or NormalSLAModel()

    def select_store(
        self,
        order:  Order,
        stores: List[Store],
        config: SimConfig,
    ) -> AssignmentResult:
        if not stores:
            raise ValueError("OptimizedStrategy requires at least one store.")

        eco = config.economics
        value = order.order_value(eco.main_item_price, eco.side_item_price)

        evaluations = []
        for store in stores:
            distance_km = _distance_to(store, order)

            # Compute pipeline metrics first (needed for p_sla)
            ev = _full_evaluation(order, store, config, assignment_cost=0.0)

            # Economic objective
            delivery_cost    = compute_delivery_cost(distance_km, eco.cost_per_km)
            expected_penalty = (1.0 - ev.sla_probability) * eco.sla_penalty_factor * value
            objective        = delivery_cost + expected_penalty

            # Re-attach objective as assignment_cost
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