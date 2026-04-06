"""
store.py
--------
Defines the Store entity with a parallel-server queue model (M/D/c approximation).

Queue model
-----------
Each store has `c = capacity_per_batch` parallel processing slots.
Each item takes exactly `T = batch_time_minutes` once it starts.
Up to c items are processed simultaneously; excess items queue.

    effective_items = main_items + side_weight * side_items

Queue delay (waiting before service starts):

    queue_delay = max(0, (workload - c) / c) * T

Interpretation: items beyond the c currently being served must wait
for one full service cycle per c-item block of overflow.
No ceil() — delay increases smoothly with workload.

Prep time (service duration for this order's items):

    if effective_items <= c:
        prep_time = T                               # fits in one parallel pass
    else:
        prep_time = ceil(effective_items / c) * T   # needs multiple passes

Time-aware workload decay
-------------------------
`_workload` tracks committed equivalent items not yet processed.
`_last_update_time` is the simulation clock at the last commit.

`_effective_workload_at(t)` computes remaining workload at time t
without mutating state — safe to call on all candidate stores during
strategy evaluation.

    processed_items = (elapsed / T) * c
    remaining       = max(0, workload - processed_items)

`commit(order, t)` is the single mutation point: drains workload to t
then adds the new order's items. Called only by the engine on the winner.
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
        prep_config: Capacity parameters (capacity_per_batch, batch_time_minutes, side_weight).
    """

    id:          int
    location:    tuple[float, float]
    prep_config: PrepConfig
    _workload:         float = field(default=0.0, init=False, repr=False)
    _last_update_time: float = field(default=0.0, init=False, repr=False)

    # ------------------------------------------------------------------
    # Internal: time-aware workload decay (read-only)
    # ------------------------------------------------------------------

    def _effective_workload_at(self, current_time: float) -> float:
        """
        Return remaining workload at `current_time` without mutating state.

            processed_items = (elapsed / batch_time) * capacity_per_batch
            remaining       = max(0, workload - processed_items)
        """
        if self._workload <= 0.0:
            return 0.0
        cfg       = self.prep_config
        elapsed   = max(0.0, current_time - self._last_update_time)
        processed = (elapsed / cfg.batch_time_minutes) * cfg.capacity_per_batch
        return max(0.0, self._workload - processed)

    # ------------------------------------------------------------------
    # Item conversion
    # ------------------------------------------------------------------

    def effective_items(self, order: Order) -> float:
        """effective_items = main_items + side_weight * side_items"""
        return order.main_items + self.prep_config.side_weight * order.side_items

    # ------------------------------------------------------------------
    # Core estimation API (read-only — safe on any candidate store)
    # ------------------------------------------------------------------

    def estimate_queue_delay(self, order: Order, current_time: float = 0.0) -> float:
        """
        Estimate waiting time before this order's items start being served (minutes).

        Parallel-server model: items beyond the c busy slots must wait.

            queue_delay = max(0, (workload - c) / c) * T

        Delay is zero when workload <= c (slots available).
        Increases smoothly — no artificial ceil() jumps.

        Args:
            order:        Incoming order (interface consistency; not used in formula).
            current_time: Simulation time at which this order arrives.
        """
        cfg      = self.prep_config
        workload = self._effective_workload_at(current_time)
        overflow = workload - cfg.capacity_per_batch
        if overflow <= 0.0:
            return 0.0
        return (overflow / cfg.capacity_per_batch) * cfg.batch_time_minutes

    def estimate_prep_time(self, order: Order) -> float:
        """
        Estimate service duration for this order's items (minutes).

        Small orders (items <= c) complete in one parallel pass (T minutes).
        Large orders require multiple passes.

            if effective_items <= c:  prep_time = T
            else:                     prep_time = ceil(effective_items / c) * T
        """
        cfg   = self.prep_config
        items = self.effective_items(order)
        if items <= cfg.capacity_per_batch:
            return cfg.batch_time_minutes
        return math.ceil(items / cfg.capacity_per_batch) * cfg.batch_time_minutes

    def estimate_total_store_time(self, order: Order, current_time: float = 0.0) -> float:
        """Queue delay + prep time for this order (minutes)."""
        return self.estimate_queue_delay(order, current_time) + self.estimate_prep_time(order)

    # ------------------------------------------------------------------
    # Workload commitment (mutating — engine only, assigned store only)
    # ------------------------------------------------------------------

    def update_workload(self, current_time: float) -> None:
        """Drain processed items up to `current_time` and update the clock."""
        self._workload         = self._effective_workload_at(current_time)
        self._last_update_time = current_time

    def commit(self, order: Order, current_time: float) -> None:
        """
        Drain workload to `current_time` then add this order's effective items.
        The only method that grows workload during a simulation run.
        Called by the engine after assignment is decided — never during evaluation.
        """
        self.update_workload(current_time)
        self._workload += self.effective_items(order)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def current_workload(self) -> float:
        """Last-committed workload in equivalent items (may be stale between commits)."""
        return self._workload

    @property
    def slots_busy(self) -> int:
        """Parallel slots currently occupied (capped at capacity)."""
        return min(int(math.ceil(self._workload)), int(self.prep_config.capacity_per_batch))

    def __repr__(self) -> str:
        return (
            f"Store(id={self.id}, location={self.location}, "
            f"workload={self._workload:.1f}, slots_busy={self.slots_busy})"
        )