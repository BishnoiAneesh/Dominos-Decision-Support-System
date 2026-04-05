"""simulation — core entities and engine for last-mile delivery simulation."""

from simulation.order import Order
from simulation.store import Store
from simulation.demand import DemandGenerator, DemandGeneratorConfig
from simulation.delivery import estimate_delivery_time, DeliveryEstimate
from simulation.assignment_types import StoreEvaluation, AssignmentResult
from simulation.strategies import (
    AssignmentStrategy,
    NearestStoreStrategy,
    OptimizedStrategy,
    STRATEGY_REGISTRY,
)
from simulation.engine import (
    SimulationEngine,
    SimulationResult,
    StoreMetrics,
    ComparisonResult,
    StrategyComparison,
    run_simulation,
    compare_strategies,
)

__all__ = [
    # Entities
    "Order",
    "Store",
    # Demand
    "DemandGenerator",
    "DemandGeneratorConfig",
    # Delivery
    "estimate_delivery_time",
    "DeliveryEstimate",
    # Assignment
    "StoreEvaluation",
    "AssignmentResult",
    "AssignmentStrategy",
    "NearestStoreStrategy",
    "OptimizedStrategy",
    "STRATEGY_REGISTRY",
    # Engine
    "SimulationEngine",
    "SimulationResult",
    "StoreMetrics",
    "ComparisonResult",
    "StrategyComparison",
    "run_simulation",
    "compare_strategies",
]