"""
main.py
-------
Runs a strategy comparison using real road-network demand,
then renders a side-by-side map comparison.
"""

from config import SimConfig
from simulation.demand import DemandGeneratorConfig
from simulation.engine import compare_strategies
from geo.map_loader import load_city_graph
from visualization.map_viz import show_comparison

def main() -> None:
    # --- Config ---
    config = SimConfig()
    config.simulation.time_horizon_minutes = 180.0
    config.demand.poisson_lambda           = 1.5
    config.sla.max_delivery_minutes        = 30.0
    config.sla.probability_threshold       = 0.95

    # --- Load map (cached after first run) ---
    print("Loading map...")
    G = load_city_graph()

    # --- Store locations (lat, lon within bounding box) ---
    store_locations = [
        (28.58, 77.35),
        (28.57, 77.40),
        (28.55, 77.34),
        (28.54, 77.45),
        (28.52, 77.37),
        (28.50, 77.43)
    ]

    # --- Real-map demand generation ---
    demand_config = DemandGeneratorConfig.from_real_map(
        graph          = G,
        arrival_params = {"lam": config.demand.poisson_lambda},
        time_horizon   = config.simulation.time_horizon_minutes,
        seed           = config.simulation.random_seed,
    )

    # --- Run comparison (graph is now a required argument) ---
    print("Precomputing store distances and running simulation...")
    comparison = compare_strategies(
        config          = config,
        store_locations = store_locations,
        graph           = G,
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

    # --- Map visualisation ---
    print("Rendering comparison map...")
    show_comparison(
        comparison      = comparison,
        graph           = G,
        store_locations = store_locations,
        save_path       = "comparison_map.png",   # set to None to display interactively
    )


if __name__ == "__main__":
    main()