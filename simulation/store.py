"""
store.py
--------
Defines the Store entity with batch-based capacity modelling and time-aware
workload progression.

Capacity model
--------------
    effective_items = main_items + side_weight * side_items

Batches: each holds up to `capacity_per_batch` items, taking `batch_time_minutes`.

Queue progression
-----------------
`_workload` tracks committed equivalent items not yet processed.
`_last_update_time` is the simulation clock when workload was last drained.

Two usage patterns:
  - estimate_queue_delay(order, current_time): read-only; computes effective
    workload at current_time without mutating state. Safe to call on all
    candidate stores during strategy evaluation.
  - commit(order, current_time): drains workload to current_time, then adds
    the order's items. Called by the engine only on the assigned store.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from config import PrepConfig
from simulation.order import Order


@dataclass
class Store:
    """
    Represents a fulfilment store in the delivery network.

    Attributes:
        id:          Unique store identifier.
        location:    (x, y) coordinates of the store.
        prep_config: Batch capacity parameters from SimConfig.
    """

    id:          int
    location:    tuple[float, float]
    prep_config: PrepConfig
    _workload:         float = field(default=0.0, init=False, repr=False)
    _last_update_time: float = field(default=0.0, init=False, repr=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _effective_workload_at(self, current_time: float) -> float:
        """
        Compute remaining workload at `current_time` without mutating state.
        Used by estimation methods so candidate stores are never modified
        during strategy evaluation.
        """
        if self._workload <= 0.0:
            return 0.0
        elapsed          = max(0.0, current_time - self._last_update_time)
        throughput_rate  = self.prep_config.capacity_per_batch / self.prep_config.batch_time_minutes
        processed        = elapsed * throughput_rate
        return max(0.0, self._workload - processed)

    def update_workload(self, current_time: float) -> None:
        """
        Drain processed items from workload up to `current_time` (mutates state).
        Called by the engine on the assigned store before committing a new order.
        """
        self._workload         = self._effective_workload_at(current_time)
        self._last_update_time = current_time

    # ------------------------------------------------------------------
    # Item conversion
    # ------------------------------------------------------------------

    def effective_items(self, order: Order) -> float:
        """effective_items = main_items + side_weight * side_items"""
        return order.main_items + self.prep_config.side_weight * order.side_items

    # ------------------------------------------------------------------
    # Core estimation API (read-only — safe to call on any candidate store)
    # ------------------------------------------------------------------

    def estimate_queue_delay(self, order: Order, current_time: float = 0.0) -> float:
        """
        Estimate queue wait before this order's batch begins (minutes).

        Uses effective workload at `current_time` — does not mutate state.

            batches_ahead = ceil(remaining_workload / capacity_per_batch)
            queue_delay   = batches_ahead * batch_time_minutes
        """
        cfg      = self.prep_config
        workload = self._effective_workload_at(current_time)
        if workload <= 0.0:
            return 0.0
        batches_ahead = math.ceil(workload / cfg.capacity_per_batch)
        return batches_ahead * cfg.batch_time_minutes

    def estimate_prep_time(self, order: Order) -> float:
        """
        Estimate preparation time for this order alone (minutes).

            own_batches = ceil(effective_items / capacity_per_batch)
            prep_time   = own_batches * batch_time_minutes
        """
        cfg   = self.prep_config
        items = self.effective_items(order)
        own_batches = math.ceil(items / cfg.capacity_per_batch)
        return own_batches * cfg.batch_time_minutes

    def estimate_total_store_time(self, order: Order, current_time: float = 0.0) -> float:
        """Queue delay + prep time for this order (minutes)."""
        return self.estimate_queue_delay(order, current_time) + self.estimate_prep_time(order)

    # ------------------------------------------------------------------
    # Workload commitment (mutating — call only on the assigned store)
    # ------------------------------------------------------------------

    def commit(self, order: Order, current_time: float) -> None:
        """
        Drain workload to current_time then add this order's items.
        This is the only method that mutates workload during a simulation run.
        Called by the engine after assignment is decided.
        """
        self.update_workload(current_time)
        self._workload += self.effective_items(order)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def current_workload(self) -> float:
        """Last-committed workload (equivalent items). May be stale; call update_workload first."""
        return self._workload

    @property
    def batches_in_progress(self) -> int:
        """Batches needed to clear current committed workload."""
        if self._workload <= 0.0:
            return 0
        return math.ceil(self._workload / self.prep_config.capacity_per_batch)

    def __repr__(self) -> str:
        return (
            f"Store(id={self.id}, location={self.location}, "
            f"workload={self._workload:.1f} items, "
            f"batches={self.batches_in_progress})"
        )