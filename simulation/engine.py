"""
engine.py
---------
Simulation orchestrator for the last-mile delivery system.
Runs a sequential event loop: demand generation → assignment → metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np

from config import SimConfig
from simulation.order import Order
from simulation.store import Store, SimpleQueueModel
from simulation.demand import DemandGenerator, DemandGeneratorConfig
from simulation.assignment import AssignmentEngine, AssignmentResult
from models.probability import NormalSLAModel


# ---------------------------------------------------------------------------
# Simulation outputs
# ---------------------------------------------------------------------------

@dataclass
class StoreMetrics:
    """Per-store summary collected over the simulation run."""
    store_id:          int
    orders_assigned:   int   = 0
    total_queue_delay: float = 0.0
    total_prep_time:   float = 0.0
    sla_met_count:     int   = 0

    @property
    def utilization(self) -> float:
        """Avg total store time per order (minutes)."""
        if self.orders_assigned == 0:
            return 0.0
        return (self.total_queue_delay + self.total_prep_time) / self.orders_assigned

    @property
    def sla_rate(self) -> float:
        if self.orders_assigned == 0:
            return 0.0
        return self.sla_met_count / self.orders_assigned


@dataclass
class SimulationResult:
    """Aggregate output of one full simulation run."""
    orders:             List[Order]
    assignment_results: List[AssignmentResult]
    store_metrics:      Dict[int, StoreMetrics]
    total_orders:       int
    sla_met:            int
    sla_rate:           float
    avg_delivery_time:  float
    feasibility_rate:   float


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SimulationEngine:
    """
    Orchestrates one complete simulation run.

    Args:
        config:        Master SimConfig.
        stores:        Pre-built list of Store objects.
        demand_config: Optional demand override (e.g. from Streamlit UI).
    """

    def __init__(
        self,
        config:        SimConfig,
        stores:        List[Store],
        demand_config: DemandGeneratorConfig | None = None,
    ) -> None:
        self.config        = config
        self.stores        = stores
        self._demand_config = demand_config or DemandGeneratorConfig.from_sim_config(config)

    def run(self) -> SimulationResult:
        """Execute the simulation and return aggregated metrics."""
        orders = self._generate_orders()
        engine = self._build_assignment_engine()

        store_metrics: Dict[int, StoreMetrics] = {
            s.id: StoreMetrics(store_id=s.id) for s in self.stores
        }
        assignment_results: List[AssignmentResult] = []

        for order in sorted(orders, key=lambda o: o.arrival_time):
            result = engine.assign(order, self.stores)
            self._apply_assignment(order, result, store_metrics)
            assignment_results.append(result)

        return self._build_result(orders, assignment_results, store_metrics)

    def _generate_orders(self) -> List[Order]:
        return DemandGenerator(self._demand_config).generate()

    def _build_assignment_engine(self) -> AssignmentEngine:
        cfg = self.config
        return AssignmentEngine(
            sla_config        = cfg.sla,
            delivery_config   = cfg.delivery,
            randomness_config = cfg.randomness,
            lambda_penalty    = 1.0,
            arrival_rate      = cfg.demand.poisson_lambda,
            probability_model = NormalSLAModel(),
        )

    def _apply_assignment(
        self,
        order:         Order,
        result:        AssignmentResult,
        store_metrics: Dict[int, StoreMetrics],
    ) -> None:
        ev  = result.evaluation
        m   = store_metrics[result.selected_store.id]

        order.ready_time     = order.arrival_time + ev.queue_delay + ev.prep_time
        order.delivered_time = order.ready_time + ev.travel_time

        result.selected_store.enqueue(order)
        result.selected_store.dequeue()

        m.orders_assigned   += 1
        m.total_queue_delay += ev.queue_delay
        m.total_prep_time   += ev.prep_time
        if ev.meets_sla_target:
            m.sla_met_count += 1

    def _build_result(
        self,
        orders:             List[Order],
        assignment_results: List[AssignmentResult],
        store_metrics:      Dict[int, StoreMetrics],
    ) -> SimulationResult:
        total        = len(orders)
        sla_met      = sum(1 for r in assignment_results if r.evaluation.meets_sla_target)
        avg_delivery = float(np.mean([r.evaluation.expected_total for r in assignment_results])) if assignment_results else 0.0
        feasible     = sum(1 for r in assignment_results if r.feasible)

        return SimulationResult(
            orders             = orders,
            assignment_results = assignment_results,
            store_metrics      = store_metrics,
            total_orders       = total,
            sla_met            = sla_met,
            sla_rate           = sla_met / total if total else 0.0,
            avg_delivery_time  = avg_delivery,
            feasibility_rate   = feasible / total if total else 0.0,
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def run_simulation(
    config:          SimConfig,
    store_locations: List[tuple[float, float]],
    demand_config:   DemandGeneratorConfig | None = None,
) -> SimulationResult:
    """
    Build stores from locations and run one simulation.

    Example::

        result = run_simulation(SimConfig(), [(2.0, 3.0), (7.0, 8.0)])
        print(result.sla_rate)
    """
    stores = [
        Store(id=i, location=loc, prep_config=config.prep, queue_model=SimpleQueueModel())
        for i, loc in enumerate(store_locations)
    ]
    return SimulationEngine(config=config, stores=stores, demand_config=demand_config).run()
