"""
demand.py
---------
Configurable demand generation for the delivery simulation.
Uses the Strategy pattern so arrival processes, location samplers, and
item-count distributions are each independently swappable.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List

from config import DemandConfig
from simulation.order import Order


# ---------------------------------------------------------------------------
# Strategy interfaces
# ---------------------------------------------------------------------------

class ArrivalProcess(ABC):
    """Generates inter-arrival times (minutes) for incoming orders."""

    @abstractmethod
    def inter_arrival_times(self, n: int, rng: np.random.Generator) -> np.ndarray: ...

    def arrival_times(
        self, n: int, rng: np.random.Generator, start_time: float = 0.0
    ) -> np.ndarray:
        return start_time + np.cumsum(self.inter_arrival_times(n, rng))


class LocationSampler(ABC):
    """Samples (x, y) delivery locations."""

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray: ...


class ItemCountSampler(ABC):
    """Samples main-item and side-item counts per order."""

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Arrival processes
# ---------------------------------------------------------------------------

class PoissonArrivalProcess(ArrivalProcess):
    """Inter-arrival times ~ Exponential(1/λ)."""

    def __init__(self, lam: float) -> None:
        if lam <= 0:
            raise ValueError(f"Poisson lambda must be > 0, got {lam}")
        self.lam = lam

    def inter_arrival_times(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.exponential(scale=1.0 / self.lam, size=n)


class UniformArrivalProcess(ArrivalProcess):
    """Orders arrive uniformly at random within [0, horizon]."""

    def __init__(self, horizon: float) -> None:
        self.horizon = horizon

    def inter_arrival_times(self, n: int, rng: np.random.Generator) -> np.ndarray:
        times = np.sort(rng.uniform(0.0, self.horizon, size=n))
        return np.diff(times, prepend=0.0)


class CustomArrivalProcess(ArrivalProcess):
    """User-supplied sampler: callable (rng, n) → np.ndarray."""

    def __init__(self, sampler_fn: Callable[[np.random.Generator, int], np.ndarray]) -> None:
        self.sampler_fn = sampler_fn

    def inter_arrival_times(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.sampler_fn(rng, n)


# ---------------------------------------------------------------------------
# Location samplers
# ---------------------------------------------------------------------------

class UniformLocationSampler(LocationSampler):
    """Uniform random locations within an axis-aligned bounding box."""

    def __init__(
        self,
        x_range: tuple[float, float] = (0.0, 10.0),
        y_range: tuple[float, float] = (0.0, 10.0),
    ) -> None:
        self.x_range = x_range
        self.y_range = y_range

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        x = rng.uniform(*self.x_range, size=n)
        y = rng.uniform(*self.y_range, size=n)
        return np.column_stack([x, y])


class GaussianLocationSampler(LocationSampler):
    """Locations clustered around a centre — models dense urban demand."""

    def __init__(
        self,
        centre:  tuple[float, float] = (5.0, 5.0),
        std_dev: float = 2.0,
    ) -> None:
        self.centre  = centre
        self.std_dev = std_dev

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(loc=self.centre, scale=self.std_dev, size=(n, 2))


class RoadNetworkLocationSampler(LocationSampler):
    """
    Samples delivery locations uniformly along real road edges.

    Requires OSMnx and a loaded graph (from geo.map_loader.load_city_graph).
    Edges are extracted once at construction time for performance.

    Args:
        graph: OSMnx MultiDiGraph (pass the cached singleton from map_loader).
    """

    def __init__(self, graph: object) -> None:
        from geo.road_sampler import extract_edges
        self._graph = graph
        self._edges = extract_edges(graph)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        from geo.road_sampler import sample_point_on_edge
        points = [
            sample_point_on_edge(
                self._edges[int(rng.integers(0, len(self._edges)))],
                self._graph,
                rng,
            )
            for _ in range(n)
        ]
        return np.array(points, dtype=float)   # shape (n, 2): col 0=lat, col 1=lon


# ---------------------------------------------------------------------------
# Item count samplers
# ---------------------------------------------------------------------------

class PoissonItemSampler(ItemCountSampler):
    """Item counts ~ Poisson, clipped to minimum of 1."""

    def __init__(self, main_lam: float = 2.0, side_lam: float = 3.0) -> None:
        self.main_lam = main_lam
        self.side_lam = side_lam

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        main = np.clip(rng.poisson(self.main_lam, size=n), 1, None)
        side = np.clip(rng.poisson(self.side_lam, size=n), 1, None)
        return np.column_stack([main, side])


class FixedItemSampler(ItemCountSampler):
    """Every order has the same fixed item counts. Useful for baselines."""

    def __init__(self, main_items: int = 2, side_items: int = 2) -> None:
        self.main_items = main_items
        self.side_items = side_items

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.tile([self.main_items, self.side_items], (n, 1))


# ---------------------------------------------------------------------------
# Strategy registries — UI dropdown name → class
# ---------------------------------------------------------------------------

ARRIVAL_REGISTRY: dict[str, type[ArrivalProcess]] = {
    "poisson": PoissonArrivalProcess,
    "uniform": UniformArrivalProcess,
    "custom":  CustomArrivalProcess,
}

LOCATION_REGISTRY: dict[str, type[LocationSampler]] = {
    "uniform":    UniformLocationSampler,
    "gaussian":   GaussianLocationSampler,
    "road_network": RoadNetworkLocationSampler,
}

ITEM_REGISTRY: dict[str, type[ItemCountSampler]] = {
    "poisson": PoissonItemSampler,
    "fixed":   FixedItemSampler,
}


# ---------------------------------------------------------------------------
# DemandGeneratorConfig
# ---------------------------------------------------------------------------

@dataclass
class DemandGeneratorConfig:
    """
    Full configuration for DemandGenerator.
    Every field maps 1-to-1 with a Streamlit widget.
    """
    arrival_process:  ArrivalProcess   = field(default_factory=lambda: PoissonArrivalProcess(lam=5.0))
    location_sampler: LocationSampler  = field(default_factory=UniformLocationSampler)
    item_sampler:     ItemCountSampler = field(default_factory=PoissonItemSampler)
    time_horizon:     float            = 480.0
    seed:             int | None       = 42
    use_real_map:     bool             = False   # If True, location_sampler must be RoadNetworkLocationSampler

    @classmethod
    def from_ui_inputs(
        cls,
        arrival_type:    str            = "poisson",
        arrival_params:  dict | None    = None,
        location_type:   str            = "uniform",
        location_params: dict | None    = None,
        item_type:       str            = "poisson",
        item_params:     dict | None    = None,
        time_horizon:    float          = 480.0,
        seed:            int | None     = 42,
    ) -> "DemandGeneratorConfig":
        """
        Construct from flat UI inputs (dropdown names + param dicts).

        Example (Streamlit)::

            cfg = DemandGeneratorConfig.from_ui_inputs(
                arrival_type   = "poisson",
                arrival_params = {"lam": st.slider("λ", 1.0, 20.0, 5.0)},
                location_type  = "gaussian",
                location_params= {"centre": (5, 5), "std_dev": st.slider("σ", 0.5, 5.0, 2.0)},
            )
        """
        def _build(registry, key, params):
            cls_ = registry.get(key)
            if cls_ is None:
                raise ValueError(f"Unknown strategy '{key}'. Choose from: {list(registry)}")
            return cls_(**(params or {}))

        return cls(
            arrival_process  = _build(ARRIVAL_REGISTRY,  arrival_type,  arrival_params),
            location_sampler = _build(LOCATION_REGISTRY, location_type, location_params),
            item_sampler     = _build(ITEM_REGISTRY,     item_type,     item_params),
            time_horizon     = time_horizon,
            seed             = seed,
        )

    @classmethod
    def from_real_map(
        cls,
        graph:          object,
        arrival_type:   str            = "poisson",
        arrival_params: dict | None    = None,
        item_type:      str            = "poisson",
        item_params:    dict | None    = None,
        time_horizon:   float          = 480.0,
        seed:           int | None     = 42,
    ) -> "DemandGeneratorConfig":
        """
        Build a DemandGeneratorConfig that samples order locations from a
        real road-network graph.

        Args:
            graph:          Pre-loaded OSMnx MultiDiGraph (use load_city_graph()).
            arrival_type:   Arrival process key (e.g. "poisson").
            arrival_params: Parameters for the arrival process.
            item_type:      Item count sampler key.
            item_params:    Parameters for the item sampler.
            time_horizon:   Simulation duration (minutes).
            seed:           RNG seed.

        Example::

            from geo.map_loader import load_city_graph
            from simulation.demand import DemandGeneratorConfig

            G   = load_city_graph()
            cfg = DemandGeneratorConfig.from_real_map(G, arrival_params={"lam": 2.0})
        """
        def _build(registry, key, params):
            cls_ = registry.get(key)
            if cls_ is None:
                raise ValueError(f"Unknown strategy '{key}'. Choose from: {list(registry)}")
            return cls_(**(params or {}))

        return cls(
            arrival_process  = _build(ARRIVAL_REGISTRY, arrival_type, arrival_params),
            location_sampler = RoadNetworkLocationSampler(graph),
            item_sampler     = _build(ITEM_REGISTRY, item_type, item_params),
            time_horizon     = time_horizon,
            seed             = seed,
            use_real_map     = True,
        )

    @classmethod
    def from_sim_config(cls, sim_config) -> "DemandGeneratorConfig":
        """Bootstrap a generator config from the master SimConfig."""
        dc = sim_config.demand
        return cls.from_ui_inputs(
            arrival_type   = dc.distribution,
            arrival_params = {"lam": dc.poisson_lambda, **dc.custom_params},
            time_horizon   = sim_config.simulation.time_horizon_minutes,
            seed           = sim_config.simulation.random_seed,
        )


# ---------------------------------------------------------------------------
# DemandGenerator
# ---------------------------------------------------------------------------

class DemandGenerator:
    """
    Generates a stream of Order objects using pluggable strategies
    for arrivals, locations, and item counts.
    """

    def __init__(self, config: DemandGeneratorConfig) -> None:
        self.config = config
        self._rng   = np.random.default_rng(config.seed)

    def generate(self, n_orders: int | None = None) -> List[Order]:
        """
        Generate orders up to the configured time horizon.

        Args:
            n_orders: Fixed count override. If None, derived from horizon.

        Returns:
            List of Order objects sorted by arrival_time.
        """
        n = n_orders or self._estimate_order_count()

        arrival_times = self.config.arrival_process.arrival_times(n, self._rng)
        locations     = self.config.location_sampler.sample(n, self._rng)
        item_counts   = self.config.item_sampler.sample(n, self._rng)

        orders = []
        for i, (t, loc, items) in enumerate(zip(arrival_times, locations, item_counts)):
            if t > self.config.time_horizon:
                break
            orders.append(Order(
                id           = i,
                location     = (float(loc[0]), float(loc[1])),
                arrival_time = float(t),
                main_items   = int(items[0]),
                side_items   = int(items[1]),
            ))

        return orders

    def _estimate_order_count(self) -> int:
        if isinstance(self.config.arrival_process, PoissonArrivalProcess):
            expected = self.config.arrival_process.lam * self.config.time_horizon
            return int(expected * 2) + 10
        return 10_000