"""
config.py
---------
Central configuration for the stochastic last-mile delivery simulation.
All parameters are grouped by concern and can be overridden dynamically
(e.g. from a Streamlit UI) by passing updated dicts to SimConfig.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Sub-configs (logical groupings)
# ---------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    """Top-level simulation controls."""
    time_horizon_minutes: float = 180.0   # Total sim duration (e.g. 3-hour shift)
    random_seed: int | None = 42          # None = non-deterministic


@dataclass
class DemandConfig:
    """
    Demand arrival process.
    Default: Poisson with configurable lambda.
    'distribution' key lets the UI swap in alternate generators later.
    """
    distribution: str = "poisson"         # "poisson" | "uniform" | "custom"
    poisson_lambda: float = 15.0           # Mean arrivals per minute
    custom_params: dict = field(default_factory=dict)  # Reserved for non-Poisson params


@dataclass
class SLAConfig:
    """Service Level Agreement thresholds."""
    max_delivery_minutes: float = 30.0    # Hard SLA window
    probability_threshold: float = 0.95   # Target P(delivery ≤ SLA)


@dataclass
class PrepConfig:
    """
    Store-side preparation rates.
    Rates are in items-per-minute; durations are derived as 1/rate.
    """
    main_item_prep_rate: float = 1.0      # Main items prepared per minute
    side_item_prep_rate: float = 1.5      # Side items prepared per minute
    prep_variance: float = 0.1            # Gaussian noise std-dev on prep time


@dataclass
class DeliveryConfig:
    """Courier movement and delivery parameters."""
    average_speed_kmph: float = 20.0      # Average courier speed
    speed_variance: float = 2.0           # Std-dev of speed (kmph)
    max_delivery_radius_km: float = 5.0   # Hard cap on delivery zone radius


@dataclass
class RandomnessConfig:
    """Global noise / stochasticity controls."""
    demand_noise_std: float = 0.5         # Noise on inter-arrival times
    travel_time_noise_std: float = 1.0    # Extra variance on travel time (minutes)
    prep_time_noise_std: float = 0.2      # Extra variance on prep time (minutes)


# ---------------------------------------------------------------------------
# Master config — single object passed throughout the simulation
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """
    Aggregated simulation configuration.

    Usage::

        cfg = SimConfig()                          # all defaults
        cfg.demand.poisson_lambda = 8.0            # override one field
        cfg = SimConfig(sla=SLAConfig(max_delivery_minutes=20))  # override section
    """
    simulation: SimulationConfig  = field(default_factory=SimulationConfig)
    demand:     DemandConfig      = field(default_factory=DemandConfig)
    sla:        SLAConfig         = field(default_factory=SLAConfig)
    prep:       PrepConfig        = field(default_factory=PrepConfig)
    delivery:   DeliveryConfig    = field(default_factory=DeliveryConfig)
    randomness: RandomnessConfig  = field(default_factory=RandomnessConfig)

    # ------------------------------------------------------------------
    # UI-facing helper: build a SimConfig from flat key=value dicts,
    # one dict per sub-config (mirrors what Streamlit widgets emit).
    # ------------------------------------------------------------------
    @classmethod
    def from_ui_inputs(
        cls,
        simulation: dict | None = None,
        demand:     dict | None = None,
        sla:        dict | None = None,
        prep:       dict | None = None,
        delivery:   dict | None = None,
        randomness: dict | None = None,
    ) -> "SimConfig":
        """Construct a SimConfig from optional UI-supplied override dicts."""
        def _merge(dataclass_cls, overrides):
            obj = dataclass_cls()
            for k, v in (overrides or {}).items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
            return obj

        return cls(
            simulation = _merge(SimulationConfig, simulation),
            demand     = _merge(DemandConfig,     demand),
            sla        = _merge(SLAConfig,        sla),
            prep       = _merge(PrepConfig,       prep),
            delivery   = _merge(DeliveryConfig,   delivery),
            randomness = _merge(RandomnessConfig, randomness),
        )


# ---------------------------------------------------------------------------
# Default singleton — import this for zero-config usage
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = SimConfig()