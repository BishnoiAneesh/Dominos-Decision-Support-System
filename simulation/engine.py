"""
engine.py
---------
Simulation orchestrator for the last-mile delivery system.
Runs a sequential event loop: demand generation → assignment → metrics.

Speed-up: road-network distances from every store to every graph node are
precomputed once via ``precompute_store_distances()`` before the order loop
starts.  The resulting StoreDistanceIndex is injected into each strategy
so per-order lookups are O(1) dict accesses.

Key entry points
----------------
run_simulation(config, store_locations, graph, strategy)       → SimulationResult
compare_strategies(config, store_locations, graph, ...)        → ComparisonResult
run_simulation_with_events(config, store_locations, graph, ..) → SimulationResult
    (dual-strategy run that fires event_callback per order for live UI)

Execution modes (``precompute`` flag)
--------------------------------------
True  — "Fast runtime":  precompute_store_distances() runs once before the
        order loop; every distance lookup is an O(1) dict access.
False — "Real-time":     strategies receive an empty distance index and fall
        back to on-the-fly Dijkstra inside estimate_delivery_time().  Slower
        but avoids the upfront cost, which can be noticeable on large graphs.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from config import SimConfig
from simulation.order import Order
from simulation.store import Store
from simulation.demand import DemandGenerator, DemandGeneratorConfig
from simulation.assignment_types import AssignmentResult, StoreEvaluation
from simulation.delivery import precompute_store_distances, build_realtime_node_index, StoreDistanceIndex
from simulation.strategies import (
    AssignmentStrategy,
    NearestStoreStrategy,
    OptimizedStrategy,
)


# ---------------------------------------------------------------------------
# Per-run outputs
# ---------------------------------------------------------------------------

@dataclass
class StoreMetrics:
    """Per-store summary collected over one simulation run."""
    store_id:           int
    orders_assigned:    int   = 0
    total_queue_delay:  float = 0.0
    total_prep_time:    float = 0.0
    total_travel_time:  float = 0.0   # sum of travel_time across assigned orders
    sla_met_count:      int   = 0

    @property
    def utilization(self) -> float:
        """Average store time (queue + prep) per order."""
        if self.orders_assigned == 0:
            return 0.0
        return (self.total_queue_delay + self.total_prep_time) / self.orders_assigned

    @property
    def avg_total_time(self) -> float:
        """Average end-to-end time (store + travel) per order."""
        if self.orders_assigned == 0:
            return 0.0
        return (self.total_queue_delay + self.total_prep_time + self.total_travel_time) / self.orders_assigned

    @property
    def sla_rate(self) -> float:
        if self.orders_assigned == 0:
            return 0.0
        return self.sla_met_count / self.orders_assigned


@dataclass
class SimulationResult:
    """Aggregate output of one full simulation run."""
    strategy_name:        str
    total_orders:         int
    sla_met:              int
    sla_rate:             float
    avg_delivery_time:    float
    feasibility_rate:     float
    store_metrics:        Dict[int, StoreMetrics]
    orders:               List[Order]
    assignment_results:   List[AssignmentResult]
    # Only populated by run_simulation_with_events (dual-strategy run)
    nearest_store_metrics: Dict[int, StoreMetrics] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.nearest_store_metrics is None:
            self.nearest_store_metrics = {}


# ---------------------------------------------------------------------------
# Comparison output
# ---------------------------------------------------------------------------

@dataclass
class StrategyComparison:
    """Head-to-head metrics for one strategy."""
    strategy_name:     str
    sla_rate:          float
    avg_delivery_time: float
    store_utilization: Dict[int, float]


@dataclass
class ComparisonResult:
    """Output of compare_strategies()."""
    comparisons:  List[StrategyComparison]
    full_results: List[SimulationResult]

    def best_sla(self) -> StrategyComparison:
        return max(self.comparisons, key=lambda c: c.sla_rate)

    def best_speed(self) -> StrategyComparison:
        return min(self.comparisons, key=lambda c: c.avg_delivery_time)


# ---------------------------------------------------------------------------
# Event helpers (used by run_simulation_with_events)
# ---------------------------------------------------------------------------

def _delivery_color(delivered_time: Optional[float], arrival_time: float) -> str:
    """Hex colour based on total delivery duration."""
    if delivered_time is None:
        return "#2196F3"          # blue  — in flight / just created
    duration = delivered_time - arrival_time
    if duration < 25.0:
        return "#4CAF50"          # green
    if duration <= 30.0:
        return "#FFC107"          # yellow
    return "#F44336"              # red   — SLA breach


def _make_event(
    order:        Order,
    status:       str,            # "created" | "delivered"
    nearest_ev:   StoreEvaluation,
    optimized_ev: StoreEvaluation,
) -> dict:
    """
    Unified event dict consumed by the Streamlit UI callback.

    Keys
    ----
    order_id, created_time, mains, sides,
    nearest_store, nearest_store_time, nearest_delivery_time, nearest_total_time,
    optimized_store, optimized_store_time, optimized_delivery_time, optimized_total_time,
    status, lat, lon, final_color
    """
    delivered_time = order.delivered_time if status == "delivered" else None
    return {
        # Identity
        "order_id":                order.id,
        "created_time":            round(order.arrival_time, 2),
        "mains":                   order.main_items,
        "sides":                   order.side_items,
        # Nearest strategy columns
        "nearest_store":           nearest_ev.store.id,
        "nearest_store_time":      round(nearest_ev.queue_delay + nearest_ev.prep_time, 2),
        "nearest_delivery_time":   round(nearest_ev.travel_time, 2),
        "nearest_total_time":      round(nearest_ev.expected_total, 2),
        # Optimized strategy columns
        "optimized_store":         optimized_ev.store.id,
        "optimized_store_time":    round(optimized_ev.queue_delay + optimized_ev.prep_time, 2),
        "optimized_delivery_time": round(optimized_ev.travel_time, 2),
        "optimized_total_time":    round(optimized_ev.expected_total, 2),
        # Geo & lifecycle
        "status":                  status,
        "lat":                     order.location[0],
        "lon":                     order.location[1],
        "final_color":             _delivery_color(delivered_time, order.arrival_time),
    }


# ---------------------------------------------------------------------------
# Engine (unchanged public interface)
# ---------------------------------------------------------------------------

class SimulationEngine:
    """
    Orchestrates one complete simulation run for a given strategy.

    Precomputes road-network distances once, injects them into the strategy,
    then processes orders sequentially.
    """

    def __init__(
        self,
        config:        SimConfig,
        stores:        List[Store],
        graph:         object,
        strategy:      AssignmentStrategy | None = None,
        demand_config: DemandGeneratorConfig | None = None,
        precompute:    bool = True,
    ) -> None:
        self.config         = config
        self.stores         = stores
        self._graph         = graph
        self._demand_config = demand_config or DemandGeneratorConfig.from_sim_config(config)

        if precompute:
            self._distance_index: StoreDistanceIndex = precompute_store_distances(stores, graph)
        else:
            # Real-time mode: store only the nearest node id per store.
            # estimate_delivery_time() runs Dijkstra on-the-fly when it finds
            # only "_node_<id>" keys (no full distance dict for the store id).
            self._distance_index = build_realtime_node_index(stores, graph)

        if strategy is None:
            strategy = OptimizedStrategy(graph=graph)
        strategy.set_distance_index(self._distance_index)
        self.strategy = strategy

    def run(self, orders: List[Order] | None = None) -> SimulationResult:
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

    def _apply_assignment(
        self,
        order:         Order,
        result:        AssignmentResult,
        store_metrics: Dict[int, StoreMetrics],
    ) -> None:
        ev = result.evaluation
        m  = store_metrics[result.selected_store.id]

        order.ready_time     = order.arrival_time + ev.queue_delay + ev.prep_time
        order.delivered_time = order.ready_time + ev.travel_time

        result.selected_store.commit(order, order.arrival_time)

        m.orders_assigned   += 1
        m.total_queue_delay += ev.queue_delay
        m.total_prep_time   += ev.prep_time
        m.total_travel_time += ev.travel_time
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
        avg_delivery = (
            float(np.mean([r.evaluation.expected_total for r in assignment_results]))
            if assignment_results else 0.0
        )
        feasible = sum(1 for r in assignment_results if r.feasible)

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

def _build_stores(config: SimConfig, locations: List[tuple]) -> List[Store]:
    return [
        Store(id=i, location=loc, prep_config=config.prep)
        for i, loc in enumerate(locations)
    ]


def run_simulation(
    config:          SimConfig,
    store_locations: List[tuple],
    graph:           object,
    strategy:        AssignmentStrategy | None = None,
    demand_config:   DemandGeneratorConfig | None = None,
    precompute:      bool = True,
) -> SimulationResult:
    """Build stores and run one simulation with the given strategy."""
    stores = _build_stores(config, store_locations)
    return SimulationEngine(
        config        = config,
        stores        = stores,
        graph         = graph,
        strategy      = strategy,
        demand_config = demand_config,
        precompute    = precompute,
    ).run()


def compare_strategies(
    config:          SimConfig,
    store_locations: List[tuple],
    graph:           object,
    strategies:      List[AssignmentStrategy] | None = None,
    demand_config:   DemandGeneratorConfig | None = None,
    precompute:      bool = True,
) -> ComparisonResult:
    """Run the simulation once per strategy on identical demand."""
    if strategies is None:
        strategies = [NearestStoreStrategy(graph=graph), OptimizedStrategy(graph=graph)]

    demand_cfg    = demand_config or DemandGeneratorConfig.from_sim_config(config)
    shared_orders = DemandGenerator(demand_cfg).generate()
    full_results: List[SimulationResult] = []

    for strategy in strategies:
        stores = _build_stores(config, store_locations)
        orders = copy.deepcopy(shared_orders)
        engine = SimulationEngine(
            config        = config,
            stores        = stores,
            graph         = graph,
            strategy      = strategy,
            demand_config = demand_cfg,
            precompute    = precompute,
        )
        full_results.append(engine.run(orders=orders))

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


# ---------------------------------------------------------------------------
# Dual-strategy event-streaming run  (used by Streamlit app)
# ---------------------------------------------------------------------------

def _update_metrics(metrics: Dict[int, StoreMetrics], result: AssignmentResult) -> None:
    ev = result.evaluation
    m  = metrics[result.selected_store.id]
    m.orders_assigned   += 1
    m.total_queue_delay += ev.queue_delay
    m.total_prep_time   += ev.prep_time
    m.total_travel_time += ev.travel_time
    if ev.meets_sla_target:
        m.sla_met_count += 1


def run_simulation_with_events(
    config:          SimConfig,
    store_locations: List[tuple],
    graph:           object,
    demand_config:   DemandGeneratorConfig | None = None,
    event_callback:  Callable[[dict], None] | None = None,
    precompute:      bool = True,
) -> SimulationResult:
    """
    Run NearestStore and Optimized in lock-step on identical demand,
    emitting a unified event dict per order so the UI can update live.

    The returned SimulationResult reflects OptimizedStrategy decisions
    (store workload commits, metrics).  NearestStore evaluations are
    computed read-only and included in the event for comparison display.

    Parameters
    ----------
    precompute:
        True  → "Fast runtime": run Dijkstra once per store up-front.
        False → "Real-time": run Dijkstra per order inside the loop.

    Event lifecycle
    ---------------
    For each order, two events are fired in sequence:
      • "created"   — both strategy evaluations are ready; timestamps not yet set.
      • "delivered" — order.delivered_time has been stamped; final_color is live.

    If event_callback is None this function behaves like run_simulation()
    with OptimizedStrategy.
    """
    demand_cfg = demand_config or DemandGeneratorConfig.from_sim_config(config)
    orders     = DemandGenerator(demand_cfg).generate()

    # Independent store instances so workloads never cross-contaminate
    nearest_stores   = _build_stores(config, store_locations)
    optimized_stores = _build_stores(config, store_locations)

    # ------------------------------------------------------------------
    # Distance index — precompute or leave empty for on-the-fly routing
    # ------------------------------------------------------------------
    if precompute:
        # Precompute once; both strategies share the read-only index
        distance_index: StoreDistanceIndex = precompute_store_distances(optimized_stores, graph)
    else:
        # Real-time mode: node id index only; Dijkstra runs per-order inside
        # estimate_delivery_time() and _road_distance_km()
        distance_index = build_realtime_node_index(optimized_stores, graph)

    nearest_strategy   = NearestStoreStrategy(graph=graph)
    optimized_strategy = OptimizedStrategy(graph=graph)
    nearest_strategy.set_distance_index(distance_index)
    optimized_strategy.set_distance_index(distance_index)

    nearest_metrics:   Dict[int, StoreMetrics] = {s.id: StoreMetrics(s.id) for s in nearest_stores}
    optimized_metrics: Dict[int, StoreMetrics] = {s.id: StoreMetrics(s.id) for s in optimized_stores}
    assignment_results: List[AssignmentResult] = []

    for order in sorted(orders, key=lambda o: o.arrival_time):
        nearest_result   = nearest_strategy.select_store(order, nearest_stores,   config)
        optimized_result = optimized_strategy.select_store(order, optimized_stores, config)

        nearest_ev   = nearest_result.evaluation
        optimized_ev = optimized_result.evaluation

        # "created" event — no timestamps yet
        if event_callback:
            event_callback(_make_event(order, "created", nearest_ev, optimized_ev))

        # Stamp timestamps using optimized evaluation
        order.ready_time     = order.arrival_time + optimized_ev.queue_delay + optimized_ev.prep_time
        order.delivered_time = order.ready_time + optimized_ev.travel_time

        # Commit workloads to each strategy's own stores
        nearest_order_copy = copy.copy(order)
        nearest_result.selected_store.commit(nearest_order_copy, order.arrival_time)
        optimized_result.selected_store.commit(order, order.arrival_time)

        _update_metrics(nearest_metrics,   nearest_result)
        _update_metrics(optimized_metrics, optimized_result)
        assignment_results.append(optimized_result)

        # "delivered" event — final_color is now accurate
        if event_callback:
            event_callback(_make_event(order, "delivered", nearest_ev, optimized_ev))

    total        = len(orders)
    sla_met      = sum(1 for r in assignment_results if r.evaluation.meets_sla_target)
    avg_delivery = (
        float(np.mean([r.evaluation.expected_total for r in assignment_results]))
        if assignment_results else 0.0
    )
    feasible = sum(1 for r in assignment_results if r.feasible)

    return SimulationResult(
        strategy_name         = "OptimizedStrategy",
        total_orders          = total,
        sla_met               = sla_met,
        sla_rate              = sla_met / total if total else 0.0,
        avg_delivery_time     = avg_delivery,
        feasibility_rate      = feasible / total if total else 0.0,
        store_metrics         = optimized_metrics,
        nearest_store_metrics = nearest_metrics,
        orders                = orders,
        assignment_results    = assignment_results,
    )