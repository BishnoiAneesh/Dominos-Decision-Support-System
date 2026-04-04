"""simulation — core entities and engine for last-mile delivery simulation."""

from simulation.order import Order
from simulation.store import Store
from simulation.demand import DemandGenerator, DemandGeneratorConfig
from simulation.delivery import estimate_delivery_time, DeliveryEstimate
from simulation.assignment import AssignmentEngine, AssignmentResult
from simulation.engine import SimulationEngine, SimulationResult, run_simulation

__all__ = [
    "Order",
    "Store",
    "DemandGenerator",
    "DemandGeneratorConfig",
    "estimate_delivery_time",
    "DeliveryEstimate",
    "AssignmentEngine",
    "AssignmentResult",
    "SimulationEngine",
    "SimulationResult",
    "run_simulation",
]