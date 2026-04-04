"""
store.py
--------
Defines the Store entity and its analytical time-estimation methods.
No discrete event simulation here — only closed-form / queueing approximations.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Protocol

from config import PrepConfig
from simulation.order import Order


# ---------------------------------------------------------------------------
# Queue model abstraction
# ---------------------------------------------------------------------------

class QueueModel(Protocol):
    """Interface for pluggable queueing models (M/M/1, M/G/1, …)."""

    def estimate_wait(
        self,
        queue: Deque[Order],
        arrival_rate: float,
        service_rate: float,
    ) -> float: ...


class SimpleQueueModel:
    """Deterministic wait = total queued items / service_rate."""

    def estimate_wait(
        self,
        queue: Deque[Order],
        arrival_rate: float,
        service_rate: float,
    ) -> float:
        if not queue:
            return 0.0
        total_items = sum(o.total_items() for o in queue)
        return total_items / service_rate if service_rate > 0 else float("inf")


class MM1QueueModel:
    """M/M/1 approximation: Wq = λ / (μ(μ − λ))."""

    def estimate_wait(
        self,
        queue: Deque[Order],
        arrival_rate: float,
        service_rate: float,
    ) -> float:
        rho = arrival_rate / service_rate if service_rate > 0 else float("inf")
        if rho >= 1.0:
            return float("inf")
        return rho / (service_rate * (1.0 - rho))


class MG1QueueModel:
    """M/G/1 Pollaczek–Khinchine approximation."""

    def __init__(self, service_time_cv: float = 1.0) -> None:
        self.cv = service_time_cv

    def estimate_wait(
        self,
        queue: Deque[Order],
        arrival_rate: float,
        service_rate: float,
    ) -> float:
        rho = arrival_rate / service_rate if service_rate > 0 else float("inf")
        if rho >= 1.0:
            return float("inf")
        mean_service = 1.0 / service_rate
        return (rho * mean_service * (1.0 + self.cv ** 2)) / (2.0 * (1.0 - rho))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

@dataclass
class Store:
    """
    Represents a fulfilment store in the delivery network.

    Attributes:
        id:           Unique store identifier.
        location:     (x, y) coordinates of the store.
        prep_config:  Prep rates sourced from SimConfig.
        queue_model:  Pluggable queueing model for wait-time estimation.
    """

    id:          int
    location:    tuple[float, float]
    prep_config: PrepConfig
    queue_model: QueueModel        = field(default_factory=SimpleQueueModel)
    _queue:      Deque[Order]      = field(default_factory=deque, init=False, repr=False)

    def enqueue(self, order: Order) -> None:
        """Add an incoming order to the store queue."""
        self._queue.append(order)

    def dequeue(self) -> Order | None:
        """Remove and return the next order (FIFO)."""
        return self._queue.popleft() if self._queue else None

    @property
    def queue_length(self) -> int:
        return len(self._queue)

    @property
    def queue_snapshot(self) -> list[Order]:
        return list(self._queue)

    def estimate_prep_time(self, order: Order) -> float:
        """Estimate preparation time for a single order (minutes)."""
        cfg = self.prep_config
        main_time = (
            order.main_items / cfg.main_item_prep_rate
            if cfg.main_item_prep_rate > 0 else float("inf")
        )
        side_time = (
            order.side_items / cfg.side_item_prep_rate
            if cfg.side_item_prep_rate > 0 else float("inf")
        )
        return main_time + side_time

    def estimate_queue_delay(self, arrival_rate: float) -> float:
        """Estimate waiting time in queue before prep begins (minutes)."""
        avg_items_per_order   = 2.0
        effective_service_rate = self.prep_config.main_item_prep_rate / avg_items_per_order
        return self.queue_model.estimate_wait(
            queue        = self._queue,
            arrival_rate = arrival_rate,
            service_rate = effective_service_rate,
        )

    def estimate_total_store_time(self, order: Order, arrival_rate: float) -> float:
        """Queue delay + prep time (minutes)."""
        return self.estimate_queue_delay(arrival_rate) + self.estimate_prep_time(order)
