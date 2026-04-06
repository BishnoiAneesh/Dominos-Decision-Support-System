"""
engine.py
---------
Simulation orchestrator for the last-mile delivery system.
Runs a sequential event loop: demand generation → assignment → metrics.

Key entry points
----------------
run_simulation(config, store_locations, strategy)  → SimulationResult
compare_strategies(config, store_locations, ...)   → ComparisonResult
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Dict

import numpy as np

from config import SimConfig
from simulation.order import Order
from simulation.store import Store
from simulation.demand import DemandGenerator, DemandGeneratorConfig
from simulation.assignment_types import AssignmentResult
from simulation.strategies import AssignmentStrategy, NearestStoreStrategy, OptimizedStrategy


# ---------------------------------------------------------------------------
# Per-run outputs
# ---------------------------------------------------------------------------

@dataclass
class StoreMetrics:
    """Per-store summary collected over one simulation run."""
    store_id:          int
    orders_assigned:   int   = 0
    total_queue_delay: float = 0.0
    total_prep_time:   float = 0.0
    sla_met_count:     int   = 0

    @property
    def utilization(self) -> float:
        """Avg total store time per assigned order (minutes)."""
        if self.orders_assigned == 0:
            return 0.0
        return (self.total_queue_delay + self.total_prep_time) / self.orders_assigned

    @property
    def sla_rate(self) -> float:
        """Fraction of assigned orders that met SLA."""
        if self.orders_assigned == 0:
            return 0.0
        return self.sla_met_count / self.orders_assigned


@dataclass
class SimulationResult:
    """Aggregate output of one full simulation run."""
    strategy_name:      str
    total_orders:       int
    sla_met:            int
    sla_rate:           float
    avg_delivery_time:  float
    feasibility_rate:   float
    store_metrics:      Dict[int, StoreMetrics]
    orders:             List[Order]
    assignment_results: List[AssignmentResult]


# ---------------------------------------------------------------------------
# Comparison output
# ---------------------------------------------------------------------------

@dataclass
class StrategyComparison:
    """Head-to-head metrics for one pair of strategies."""
    strategy_name:    str
    sla_rate:         float
    avg_delivery_time: float
    store_utilization: Dict[int, float]   # store_id → avg store time (minutes)


@dataclass
class ComparisonResult:
    """
    Output of compare_strategies().
    Contains one StrategyComparison per strategy run, in the same order
    they were passed in. Full SimulationResults are also kept for deep dives.
    """
    comparisons:        List[StrategyComparison]
    full_results:       List[SimulationResult]

    def best_sla(self) -> StrategyComparison:
        """Return the strategy with the highest SLA rate."""
        return max(self.comparisons, key=lambda c: c.sla_rate)

    def best_speed(self) -> StrategyComparison:
        """Return the strategy with the lowest average delivery time."""
        return min(self.comparisons, key=lambda c: c.avg_delivery_time)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SimulationEngine:
    """
    Orchestrates one complete simulation run for a given strategy.

    Args:
        config:        Master SimConfig.
        stores:        Pre-built Store objects (will be mutated during run).
        strategy:      Assignment strategy. Defaults to OptimizedStrategy.
        demand_config: Optional demand override (e.g. from Streamlit UI).
    """

    def __init__(
        self,
        config:        SimConfig,
        stores:        List[Store],
        strategy:      AssignmentStrategy | None = None,
        demand_config: DemandGeneratorConfig | None = None,
    ) -> None:
        self.config         = config
        self.stores         = stores
        self.strategy       = strategy or OptimizedStrategy()
        self._demand_config = demand_config or DemandGeneratorConfig.from_sim_config(config)

    def run(self, orders: List[Order] | None = None) -> SimulationResult:
        """
        Execute the simulation and return aggregated metrics.

        Args:
            orders: Pre-generated orders to process. If None, generates
                    a fresh batch from the demand config. Passing orders
                    explicitly allows compare_strategies() to run both
                    strategies on identical demand.
        """
        if orders is None:
            orders = DemandGenerator(self._demand_config).generate()

        store_metrics: Dict[int, StoreMetrics] = {
            s.id: StoreMetrics(store_id=s.id) for s in self.stores
        }
        assignment_results: List[AssignmentResult] = []

        for order in sorted(orders, key=lambda o: o.arrival_time):
            result = self.strategy.select_store(order, self.stores, self.config)
            self._apply_assignment(order, result, store_metrics)
            assignment_results.append(result)

        return self._build_result(orders, assignment_results, store_metrics)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_assignment(
        self,
        order:         Order,
        result:        AssignmentResult,
        store_metrics: Dict[int, StoreMetrics],
    ) -> None:
        """Stamp order timestamps, update store workload, accumulate metrics."""
        ev = result.evaluation
        m  = store_metrics[result.selected_store.id]

        order.ready_time     = order.arrival_time + ev.queue_delay + ev.prep_time
        order.delivered_time = order.ready_time + ev.travel_time

        # Drain workload to current time and commit new order atomically
        result.selected_store.commit(order, order.arrival_time)

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
            strategy_name      = self.strategy.name,
            total_orders       = total,
            sla_met            = sla_met,
            sla_rate           = sla_met / total if total else 0.0,
            avg_delivery_time  = avg_delivery,
            feasibility_rate   = feasible / total if total else 0.0,
            store_metrics      = store_metrics,
            orders             = orders,
            assignment_results = assignment_results,
        )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def _build_stores(config: SimConfig, locations: List[tuple[float, float]]) -> List[Store]:
    """Construct a fresh list of Store objects from coordinate pairs."""
    return [
        Store(id=i, location=loc, prep_config=config.prep)
        for i, loc in enumerate(locations)
    ]


def run_simulation(
    config:          SimConfig,
    store_locations: List[tuple[float, float]],
    strategy:        AssignmentStrategy | None = None,
    demand_config:   DemandGeneratorConfig | None = None,
) -> SimulationResult:
    """
    Build stores and run one simulation with the given strategy.

    Args:
        config:           Master SimConfig.
        store_locations:  (x, y) coordinates for each store.
        strategy:         Assignment strategy. Defaults to OptimizedStrategy.
        demand_config:    Optional demand override (UI-supplied).

    Example::

        result = run_simulation(SimConfig(), [(2.0, 3.0), (7.0, 8.0)])
        print(result.sla_rate, result.strategy_name)
    """
    stores = _build_stores(config, store_locations)
    return SimulationEngine(
        config        = config,
        stores        = stores,
        strategy      = strategy,
        demand_config = demand_config,
    ).run()


def compare_strategies(
    config:          SimConfig,
    store_locations: List[tuple[float, float]],
    strategies:      List[AssignmentStrategy] | None = None,
    demand_config:   DemandGeneratorConfig | None = None,
) -> ComparisonResult:
    """
    Run the simulation once per strategy on identical demand, then
    return side-by-side comparison metrics.

    Both strategies see the same generated orders so results differ only
    due to assignment decisions, not demand randomness.

    Args:
        config:           Master SimConfig.
        store_locations:  (x, y) coordinates for each store.
        strategies:       List of strategy objects to compare.
                          Defaults to [NearestStoreStrategy(), OptimizedStrategy()].
        demand_config:    Optional demand override (UI-supplied).

    Returns:
        ComparisonResult with per-strategy StrategyComparison summaries
        and full SimulationResults for deep inspection.

    Example::

        result = compare_strategies(SimConfig(), [(2.0, 2.0), (8.0, 8.0)])
        print(result.best_sla().strategy_name)
        print(result.best_speed().strategy_name)
    """
    if strategies is None:
        strategies = [NearestStoreStrategy(), OptimizedStrategy()]

    # Generate demand once — shared across all strategy runs
    demand_cfg  = demand_config or DemandGeneratorConfig.from_sim_config(config)
    shared_orders = DemandGenerator(demand_cfg).generate()

    full_results: List[SimulationResult] = []

    for strategy in strategies:
        # Each strategy gets its own fresh stores (clean workload state)
        stores  = _build_stores(config, store_locations)
        # Deep-copy orders so timestamp fields don't bleed across runs
        orders  = copy.deepcopy(shared_orders)
        engine  = SimulationEngine(config=config, stores=stores, strategy=strategy)
        result  = engine.run(orders=orders)
        full_results.append(result)

    comparisons = [
        StrategyComparison(
            strategy_name     = r.strategy_name,
            sla_rate          = r.sla_rate,
            avg_delivery_time = r.avg_delivery_time,
            store_utilization = {sid: m.utilization for sid, m in r.store_metrics.items()},
        )
        for r in full_results
    ]

    return ComparisonResult(comparisons=comparisons, full_results=full_results)