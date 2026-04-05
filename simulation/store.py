"""
store.py
--------
Defines the Store entity with batch-based capacity modelling.

Capacity model
--------------
Orders are converted to equivalent items:
    effective_items = main_items + side_weight * side_items

The store processes items in fixed-size batches:
    - Each batch holds up to `capacity_per_batch` equivalent items.
    - Each batch takes exactly `batch_time_minutes` to complete.

Queue state is tracked as a single float (`_workload`) — the total
equivalent items currently ahead in the pipeline — rather than a list
of order objects. This makes delay estimation O(1).
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
    _workload:   float = field(default=0.0, init=False, repr=False)

    # ------------------------------------------------------------------
    # Item conversion
    # ------------------------------------------------------------------

    def effective_items(self, order: Order) -> float:
        """
        Convert an order's item counts to a single equivalent-item value.

        effective_items = main_items + side_weight * side_items
        """
        return order.main_items + self.prep_config.side_weight * order.side_items

    # ------------------------------------------------------------------
    # Core estimation API
    # ------------------------------------------------------------------

    def estimate_queue_delay(self, order: Order) -> float:
        """
        Estimate waiting time before this order's batch begins (minutes).

        Logic:
            batches_ahead = ceil(workload_ahead / capacity_per_batch)
            queue_delay   = batches_ahead * batch_time_minutes

        The order's own items are NOT included in workload_ahead —
        only items already committed to the pipeline count.
        """
        cfg = self.prep_config
        if self._workload <= 0.0:
            return 0.0
        batches_ahead = math.ceil(self._workload / cfg.capacity_per_batch)
        return batches_ahead * cfg.batch_time_minutes

    def estimate_prep_time(self, order: Order) -> float:
        """
        Estimate preparation time for this order alone (minutes).

        Logic:
            own_batches = ceil(effective_items / capacity_per_batch)
            prep_time   = own_batches * batch_time_minutes
        """
        cfg   = self.prep_config
        items = self.effective_items(order)
        own_batches = math.ceil(items / cfg.capacity_per_batch)
        return own_batches * cfg.batch_time_minutes

    def estimate_total_store_time(self, order: Order) -> float:
        """Queue delay + prep time for this order (minutes)."""
        return self.estimate_queue_delay(order) + self.estimate_prep_time(order)

    # ------------------------------------------------------------------
    # Workload management
    # ------------------------------------------------------------------

    def add_order(self, order: Order) -> None:
        """
        Commit an order to the store pipeline.
        Increments workload by the order's effective item count.
        Call this after assignment is confirmed.
        """
        self._workload += self.effective_items(order)

    def complete_order(self, order: Order) -> None:
        """
        Remove a completed order from the pipeline.
        Decrements workload; clamps to zero to avoid floating-point drift.
        """
        self._workload = max(0.0, self._workload - self.effective_items(order))

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def current_workload(self) -> float:
        """Total equivalent items currently in the pipeline."""
        return self._workload

    @property
    def batches_in_progress(self) -> int:
        """Number of batches needed to clear the current workload."""
        if self._workload <= 0.0:
            return 0
        return math.ceil(self._workload / self.prep_config.capacity_per_batch)

    def __repr__(self) -> str:
        return (
            f"Store(id={self.id}, location={self.location}, "
            f"workload={self._workload:.1f} items, "
            f"batches={self.batches_in_progress})"
        )