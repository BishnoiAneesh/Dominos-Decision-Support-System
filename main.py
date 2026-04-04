"""
main.py
-------
Entry point for running a quick end-to-end simulation test.
Initialises stores, demand, runs the engine, and prints summary metrics.
"""

from config import SimConfig
from simulation.demand import DemandGeneratorConfig
from simulation.engine import run_simulation


def main() -> None:
    # --- Config ---
    config = SimConfig()
    demand_config = DemandGeneratorConfig.from_sim_config(config)
    
    # --- Stores: 2 stores placed across a 10×10 km grid ---
    store_locations = [
        (2.0, 2.0),
        (5.0, 8.0)
    ]

    # --- Demand: Poisson arrivals, Gaussian locations, Poisson items ---
    demand_config = DemandGeneratorConfig.from_ui_inputs(
        arrival_type    = "poisson",
        arrival_params  = {"lam": config.demand.poisson_lambda},
        location_type   = "gaussian",
        location_params = {"centre": (5.0, 5.0), "std_dev": 2.5},
        item_type       = "poisson",
        item_params     = {"main_lam": 2.0, "side_lam": 3.0},
        time_horizon    = config.simulation.time_horizon_minutes,
        seed            = config.simulation.random_seed,
    )

    # --- Run ---
    print("Running simulation...")
    result = run_simulation(
        config          = config,
        store_locations = store_locations,
        demand_config   = demand_config,
    )

    # --- Summary ---
    print(f"\n{'='*40}")
    print(f"  SIMULATION SUMMARY")
    print(f"{'='*40}")
    print(f"  Total orders        : {result.total_orders}")
    print(f"  SLA met             : {result.sla_met} / {result.total_orders}")
    print(f"  SLA success rate    : {result.sla_rate:.1%}")
    print(f"  Avg delivery time   : {result.avg_delivery_time:.2f} min")
    print(f"  Feasibility rate    : {result.feasibility_rate:.1%}")

    print(f"\n{'='*40}")
    print(f"  PER-STORE BREAKDOWN")
    print(f"{'='*40}")
    for sid, m in result.store_metrics.items():
        print(
            f"  Store {sid} | "
            f"Orders: {m.orders_assigned:>4} | "
            f"SLA rate: {m.sla_rate:.1%} | "
            f"Avg store time: {m.utilization:.2f} min"
        )

    print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
