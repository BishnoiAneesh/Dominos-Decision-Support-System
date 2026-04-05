"""
assignment_types.py
-------------------
Shared data types for the assignment pipeline.
Kept separate so strategies, engine, and UI can all import them
without circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from simulation.order import Order
from simulation.store import Store


@dataclass
class StoreEvaluation:
    """
    Full diagnostic snapshot for one (order, store) candidate pair.
    Populated by whichever strategy performed the evaluation.
    Fields that a strategy does not compute should be left as 0.0.
    """
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
    """Output of one assignment decision."""
    order:           Order
    selected_store:  Store
    evaluation:      StoreEvaluation
    all_evaluations: List[StoreEvaluation]
    feasible:        bool
    strategy_name:   str               # Which strategy produced this result
