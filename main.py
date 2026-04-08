"""
main.py
-------
Runs a strategy comparison using real road-network demand,
then renders a side-by-side map comparison.

Execution mode
--------------
Set EXEC_MODE to control routing behaviour:
  "precompute"  ("Fast runtime")  — Dijkstra runs once per store up-front.
  "direct"      ("Real-time")     — Dijkstra runs per order inside the loop.
"""

from config import SimConfig
from simulation.demand import DemandGeneratorConfig
from simulation.engine import compare_strategies
from geo.map_loader import load_city_graph
from visualization.map_viz import show_comparison
from geo.geo_constants import MAP_BOUNDS, MAP_NORTH, MAP_SOUTH, MAP_EAST, MAP_WEST


# ---------------------------------------------------------------------------
# Execution mode: "precompute" (fast) | "direct" (real-time)
# ---------------------------------------------------------------------------
EXEC_MODE = "direct"


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

    # --- Store locations (lat, lon) — must be within MAP_BOUNDS ---
    store_locations = [
        (28.5598, 77.3712),
        (28.5553, 77.4076),
        (28.5366, 77.4083),
        (28.5381, 77.3678),
        (28.5158, 77.3753),
        (28.5176, 77.4076)
    ]

    # Validate all stores fall within the shared bounding box
    invalid = [
        loc for loc in store_locations
        if not (MAP_SOUTH <= loc[0] <= MAP_NORTH and MAP_WEST <= loc[1] <= MAP_EAST)
    ]
    if invalid:
        raise ValueError(
            f"Store location(s) outside map bounds {MAP_BOUNDS}: {invalid}"
        )

    # --- Real-map demand generation ---
    demand_config = DemandGeneratorConfig.from_real_map(
        graph          = G,
        arrival_params = {"lam": config.demand.poisson_lambda},
        time_horizon   = config.simulation.time_horizon_minutes,
        seed           = config.simulation.random_seed,
    )

    precompute = (EXEC_MODE == "precompute")
    mode_label = "Fast runtime (precompute)" if precompute else "Real-time (direct Dijkstra)"
    print(f"Execution mode: {mode_label}")

    # --- Run comparison ---
    print("Running simulation...")
    comparison = compare_strategies(
        config          = config,
        store_locations = store_locations,
        graph           = G,
        demand_config   = demand_config,
        precompute      = precompute,
    )

    # --- Print summary ---
    total_orders = comparison.full_results[0].total_orders
    print(f"\n{'='*52}")
    print(f"  STRATEGY COMPARISON  ({total_orders} orders)  [{mode_label}]")
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