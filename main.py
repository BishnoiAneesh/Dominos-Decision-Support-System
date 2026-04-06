"""
main.py
-------
Runs a strategy comparison and prints a side-by-side summary.
"""

from config import SimConfig
from simulation.demand import DemandGeneratorConfig
from simulation.engine import compare_strategies


def main() -> None:
    # --- Config ---
    config = SimConfig()
    config.simulation.time_horizon_minutes = 180.0
    config.demand.poisson_lambda           = 1    
    config.sla.max_delivery_minutes        = 30.0
    config.sla.probability_threshold       = 0.95

    store_locations = [(2.0, 2.0), (5.0, 8.0), (8.0, 3.0)]

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

    # --- Run comparison (both strategies, same demand) ---
    comparison = compare_strategies(
        config          = config,
        store_locations = store_locations,
        demand_config   = demand_config,
    )

    # --- Print summary ---
    total_orders = comparison.full_results[0].total_orders
    print(f"\n{'='*52}")
    print(f"  STRATEGY COMPARISON  ({total_orders} orders)")
    print(f"{'='*52}")
    print(f"  {'Strategy':<22} {'SLA Rate':>9} {'Avg Time':>10}")
    print(f"  {'-'*46}")

    for c in comparison.comparisons:
        print(f"  {c.strategy_name:<22} {c.sla_rate:>8.1%} {c.avg_delivery_time:>9.2f} min")

    print(f"\n  Best SLA   → {comparison.best_sla().strategy_name}")
    print(f"  Best Speed → {comparison.best_speed().strategy_name}")

    # --- Per-store breakdown ---
    for result in comparison.full_results:
        print(f"\n  [{result.strategy_name}] Store Utilization")
        print(f"  {'-'*46}")
        for sid, m in result.store_metrics.items():
            print(
                f"    Store {sid} | "
                f"Orders: {m.orders_assigned:>4} | "
                f"SLA: {m.sla_rate:.1%} | "
                f"Avg store time: {m.utilization:.2f} min"
            )

    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()