"""
assignment.py
-------------
Core decision engine: assigns an incoming order to the best available store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from config import DeliveryConfig, RandomnessConfig, SLAConfig
from simulation.order import Order
from simulation.store import Store
from simulation.delivery import estimate_delivery_time
from models.probability import prob_within_sla, meets_sla_target, SLAProbabilityModel


@dataclass
class StoreEvaluation:
    """Full diagnostic snapshot for one store candidate."""
    store:            Store
    queue_delay:      float
    prep_time:        float
    travel_time:      float
    expected_total:   float
    total_variance:   float
    sla_probability:  float
    meets_sla_target: bool
    assignment_cost:  float


@dataclass
class AssignmentResult:
    """Output of the assignment engine for one order."""
    order:           Order
    selected_store:  Store
    evaluation:      StoreEvaluation
    all_evaluations: List[StoreEvaluation]
    feasible:        bool


class AssignmentEngine:
    """
    Assigns orders to stores using a probability-weighted cost objective.

    Objective (feasible stores):  cost = base_cost + λ * (1 - P(SLA))
    Fallback (no feasible store): argmax P(SLA)
    """

    def __init__(
        self,
        sla_config:         SLAConfig,
        delivery_config:    DeliveryConfig,
        randomness_config:  RandomnessConfig,
        lambda_penalty:     float = 1.0,
        arrival_rate:       float = 1.0,
        probability_model:  SLAProbabilityModel | None = None,
    ) -> None:
        self.sla_config        = sla_config
        self.delivery_config   = delivery_config
        self.randomness_config = randomness_config
        self.lambda_penalty    = lambda_penalty
        self.arrival_rate      = arrival_rate
        self.probability_model = probability_model

    def assign(self, order: Order, stores: List[Store]) -> AssignmentResult:
        """Evaluate all candidate stores and return the optimal assignment."""
        if not stores:
            raise ValueError("Assignment requires at least one candidate store.")

        evaluations = [self._evaluate(order, store) for store in stores]
        feasible    = [e for e in evaluations if e.meets_sla_target]

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
        )

    def _evaluate(self, order: Order, store: Store) -> StoreEvaluation:
        """Build a full StoreEvaluation for one (order, store) pair."""
        queue_delay  = store.estimate_queue_delay(self.arrival_rate)
        prep_time    = store.estimate_prep_time(order)
        delivery_est = estimate_delivery_time(
            origin         = store.location,
            destination    = order.location,
            delivery_cfg   = self.delivery_config,
            randomness_cfg = self.randomness_config,
            rng            = None,
        )

        expected_total = queue_delay + prep_time + delivery_est.expected_time
        total_variance = delivery_est.variance

        p_sla  = prob_within_sla(
            expected_time = expected_total,
            variance      = total_variance,
            sla_threshold = self.sla_config.max_delivery_minutes,
            model         = self.probability_model,
        )
        sla_met = meets_sla_target(
            expected_time      = expected_total,
            variance           = total_variance,
            sla_threshold      = self.sla_config.max_delivery_minutes,
            probability_target = self.sla_config.probability_threshold,
            model              = self.probability_model,
        )

        base_cost       = self._base_cost(store, order, delivery_est.distance_km)
        assignment_cost = base_cost + self.lambda_penalty * (1.0 - p_sla)

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

    def _base_cost(self, store: Store, order: Order, distance_km: float) -> float:
        """Base cost proportional to travel distance. Extend here for richer models."""
        return distance_km
