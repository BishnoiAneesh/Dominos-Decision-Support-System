"""
assignment.py
-------------
Backward-compatible re-export shim.

All assignment logic now lives in:
  simulation.strategies       — AssignmentStrategy, NearestStoreStrategy, OptimizedStrategy
  simulation.assignment_types — StoreEvaluation, AssignmentResult

Import from those modules directly for new code.
This file preserves existing imports across the codebase.
"""

from simulation.assignment_types import StoreEvaluation, AssignmentResult   # noqa: F401
from simulation.strategies import (                                          # noqa: F401
    AssignmentStrategy,
    NearestStoreStrategy,
    OptimizedStrategy,
    STRATEGY_REGISTRY,
)

__all__ = [
    "StoreEvaluation",
    "AssignmentResult",
    "AssignmentStrategy",
    "NearestStoreStrategy",
    "OptimizedStrategy",
    "STRATEGY_REGISTRY",
]
