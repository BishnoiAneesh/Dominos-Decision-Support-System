"""
order.py
--------
Defines the Order entity for the delivery simulation.
Responsible only for representing an order — no generation logic here.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Order:
    """
    Represents a single customer delivery order.

    Attributes:
        id:           Unique order identifier.
        location:     (x, y) coordinates of the delivery address.
        arrival_time: Time (minutes) at which the order enters the system.
        main_items:   Number of main items in the order.
        side_items:   Number of side items in the order.
    """
    id:           int
    location:     Tuple[float, float]
    arrival_time: float
    main_items:   int
    side_items:   int

    ready_time:     float | None = field(default=None, repr=False)
    delivered_time: float | None = field(default=None, repr=False)

    def total_items(self) -> int:
        """Return the combined item count across all categories."""
        return self.main_items + self.side_items

    def delivery_duration(self) -> float | None:
        """Return end-to-end time (minutes) from arrival to delivery."""
        if self.delivered_time is None:
            return None
        return self.delivered_time - self.arrival_time

    def met_sla(self, sla_minutes: float) -> bool | None:
        """Check whether the order was delivered within the SLA window."""
        duration = self.delivery_duration()
        if duration is None:
            return None
        return duration <= sla_minutes

    def __repr__(self) -> str:
        return (
            f"Order(id={self.id}, items={self.total_items()}, "
            f"arrival={self.arrival_time:.2f}, location={self.location})"
        )
