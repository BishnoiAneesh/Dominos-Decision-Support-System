"""
visualization/map_viz.py
------------------------
Matplotlib-based map visualisation for the last-mile delivery simulation.

Renders:
  • Road network (OSMnx graph edges)
  • Store locations (star markers)
  • Order locations colour-coded by delivery outcome:
        Blue   — order created / undelivered
        Green  — delivered in < 25 min
        Yellow — delivered in 25–30 min
        Red    — delivered in > 30 min

Public API
----------
plot_simulation_result(result, graph, store_locations)
    Full static plot of one SimulationResult.

plot_comparison(comparison_result, graph, store_locations)
    Side-by-side subplots for each strategy in a ComparisonResult.

animate_orders(result, graph, store_locations, interval_ms)
    Step-wise animation: orders appear one at a time in arrival order.

All functions return the Matplotlib Figure so callers can save or display it::

    fig = plot_simulation_result(result, G, store_locations)
    fig.savefig("sim_output.png", dpi=150)
    plt.show()
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import numpy as np

from simulation.engine import SimulationResult, ComparisonResult
from simulation.order import Order


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_BLUE   = "#2196F3"   # created / in-flight
_GREEN  = "#4CAF50"   # delivered fast  (< 25 min)
_YELLOW = "#FFC107"   # borderline      (25–30 min)
_RED    = "#F44336"   # SLA breach      (> 30 min)

_STORE_COLOR  = "#9C27B0"
_ROAD_COLOR   = "#CCCCCC"


def _order_color(order: Order, sla_minutes: float = 30.0) -> str:
    """Return the display colour for a single order based on delivery time."""
    if order.delivered_time is None:
        return _BLUE
    duration = order.delivered_time - order.arrival_time
    if duration < 25.0:
        return _GREEN
    if duration <= sla_minutes:
        return _YELLOW
    return _RED


# ---------------------------------------------------------------------------
# Road network drawing
# ---------------------------------------------------------------------------

def _draw_road_network(ax: plt.Axes, graph: object, alpha: float = 0.3) -> None:
    """Draw all road edges as thin grey lines."""
    for u, v, data in graph.edges(data=True):
        geom = data.get("geometry")
        if geom is not None:
            xs, ys = geom.xy
            ax.plot(xs, ys, color=_ROAD_COLOR, linewidth=0.5, alpha=alpha, zorder=1)
        else:
            # Fallback: straight line between endpoint nodes
            nodes = graph.nodes
            if u in nodes and v in nodes:
                x0, y0 = nodes[u]["x"], nodes[u]["y"]
                x1, y1 = nodes[v]["x"], nodes[v]["y"]
                ax.plot([x0, x1], [y0, y1],
                        color=_ROAD_COLOR, linewidth=0.5, alpha=alpha, zorder=1)


def _draw_stores(
    ax: plt.Axes,
    store_locations: List[Tuple[float, float]],
) -> None:
    """Plot store locations as large star markers."""
    for i, (lat, lon) in enumerate(store_locations):
        ax.plot(
            lon, lat,
            marker="*", markersize=14,
            color=_STORE_COLOR, markeredgecolor="white", markeredgewidth=0.8,
            zorder=5, label=f"Store {i}" if i == 0 else "_nolegend_",
        )
        ax.annotate(
            f"S{i}",
            xy=(lon, lat),
            xytext=(3, 3), textcoords="offset points",
            fontsize=7, color=_STORE_COLOR, fontweight="bold", zorder=6,
        )


def _draw_orders(
    ax: plt.Axes,
    orders: List[Order],
    sla_minutes: float = 30.0,
    size: float = 18,
) -> None:
    """Scatter-plot all orders colour-coded by delivery outcome."""
    # Group by colour for a single scatter call each (faster rendering)
    buckets: dict[str, tuple[list, list]] = {
        _BLUE:   ([], []),
        _GREEN:  ([], []),
        _YELLOW: ([], []),
        _RED:    ([], []),
    }
    for o in orders:
        col = _order_color(o, sla_minutes)
        lat, lon = o.location
        buckets[col][0].append(lon)
        buckets[col][1].append(lat)

    labels = {
        _BLUE:   "Created / in-flight",
        _GREEN:  "Delivered < 25 min",
        _YELLOW: "Delivered 25–30 min",
        _RED:    "Delivered > 30 min (SLA breach)",
    }
    for col, (lons, lats) in buckets.items():
        if lons:
            ax.scatter(lons, lats, c=col, s=size, alpha=0.75,
                       edgecolors="none", zorder=4, label=labels[col])


# ---------------------------------------------------------------------------
# Legend builder
# ---------------------------------------------------------------------------

def _add_legend(ax: plt.Axes, show_stores: bool = True) -> None:
    patches = [
        mpatches.Patch(color=_GREEN,  label="Delivered < 25 min"),
        mpatches.Patch(color=_YELLOW, label="Delivered 25–30 min"),
        mpatches.Patch(color=_RED,    label="SLA breach (> 30 min)"),
        mpatches.Patch(color=_BLUE,   label="Created / in-flight"),
    ]
    if show_stores:
        patches.append(
            mpatches.Patch(color=_STORE_COLOR, label="Store")
        )
    ax.legend(handles=patches, loc="lower left", fontsize=7,
              framealpha=0.85, edgecolor="grey")


# ---------------------------------------------------------------------------
# Primary public function: single result
# ---------------------------------------------------------------------------

def plot_simulation_result(
    result:          SimulationResult,
    graph:           object,
    store_locations: List[Tuple[float, float]],
    sla_minutes:     float = 30.0,
    figsize:         Tuple[float, float] = (10, 8),
    title:           Optional[str] = None,
) -> plt.Figure:
    """
    Produce a static map for one SimulationResult.

    Args:
        result:           Output of run_simulation() or engine.run().
        graph:            OSMnx MultiDiGraph used during the simulation.
        store_locations:  List of (lat, lon) tuples — same order as stores.
        sla_minutes:      SLA threshold for colour coding (default 30).
        figsize:          Matplotlib figure size.
        title:            Custom plot title.  Auto-generated if None.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    _draw_road_network(ax, graph)
    _draw_stores(ax, store_locations)
    _draw_orders(ax, result.orders, sla_minutes)
    _add_legend(ax)

    _title = title or (
        f"Strategy: {result.strategy_name}  |  "
        f"SLA met: {result.sla_met}/{result.total_orders} "
        f"({result.sla_rate:.1%})  |  "
        f"Avg delivery: {result.avg_delivery_time:.1f} min"
    )
    ax.set_title(_title, fontsize=10, pad=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Comparison plot: one subplot per strategy
# ---------------------------------------------------------------------------

def plot_comparison(
    comparison:      ComparisonResult,
    graph:           object,
    store_locations: List[Tuple[float, float]],
    sla_minutes:     float = 30.0,
    figsize:         Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Side-by-side subplots comparing all strategies in a ComparisonResult.

    Args:
        comparison:       Output of compare_strategies().
        graph:            OSMnx MultiDiGraph.
        store_locations:  List of (lat, lon) tuples.
        sla_minutes:      SLA threshold for colour coding.
        figsize:          Override figure size; auto-computed if None.

    Returns:
        matplotlib.figure.Figure
    """
    n = len(comparison.full_results)
    if figsize is None:
        figsize = (9 * n, 8)

    fig, axes = plt.subplots(1, n, figsize=figsize, dpi=100,
                              sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, comparison.full_results):
        _draw_road_network(ax, graph)
        _draw_stores(ax, store_locations)
        _draw_orders(ax, result.orders, sla_minutes)
        _add_legend(ax, show_stores=False)

        ax.set_title(
            f"{result.strategy_name}\n"
            f"SLA {result.sla_rate:.1%}  |  "
            f"Avg {result.avg_delivery_time:.1f} min",
            fontsize=9,
        )
        ax.set_xlabel("Longitude")
        ax.set_aspect("equal")

    axes[0].set_ylabel("Latitude")
    fig.suptitle("Strategy Comparison", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Animation: orders appear in arrival order
# ---------------------------------------------------------------------------

def animate_orders(
    result:          SimulationResult,
    graph:           object,
    store_locations: List[Tuple[float, float]],
    sla_minutes:     float = 30.0,
    interval_ms:     int   = 120,
    figsize:         Tuple[float, float] = (10, 8),
) -> FuncAnimation:
    """
    Step-wise animation: orders appear one at a time, sorted by arrival time.

    Each frame adds the next order.  Colour reflects final delivery outcome
    so the viewer sees the result build up progressively.

    Args:
        result:          SimulationResult.
        graph:           OSMnx MultiDiGraph.
        store_locations: List of (lat, lon) tuples.
        sla_minutes:     SLA threshold for colour coding.
        interval_ms:     Milliseconds between frames.
        figsize:         Figure size.

    Returns:
        matplotlib.animation.FuncAnimation  (call plt.show() or .save() on it)

    Example::

        anim = animate_orders(result, G, store_locations)
        anim.save("delivery_animation.gif", fps=10)
        plt.show()
    """
    orders_sorted = sorted(result.orders, key=lambda o: o.arrival_time)

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    _draw_road_network(ax, graph, alpha=0.25)
    _draw_stores(ax, store_locations)
    _add_legend(ax)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")

    # Pre-compute bounding box so the view doesn't jump
    all_lons = [o.location[1] for o in orders_sorted]
    all_lats = [o.location[0] for o in orders_sorted]
    store_lons = [loc[1] for loc in store_locations]
    store_lats = [loc[0] for loc in store_locations]
    all_lons += store_lons
    all_lats += store_lats
    margin = 0.005
    ax.set_xlim(min(all_lons) - margin, max(all_lons) + margin)
    ax.set_ylim(min(all_lats) - margin, max(all_lats) + margin)

    scatter_artists: List = []
    title_template = "Order {i}/{total}  |  t = {t:.1f} min  |  Strategy: {s}"

    def _update(frame: int):
        order = orders_sorted[frame]
        col   = _order_color(order, sla_minutes)
        lat, lon = order.location
        sc = ax.scatter([lon], [lat], c=col, s=22, alpha=0.85,
                        edgecolors="none", zorder=4)
        scatter_artists.append(sc)
        ax.set_title(
            title_template.format(
                i     = frame + 1,
                total = len(orders_sorted),
                t     = order.arrival_time,
                s     = result.strategy_name,
            ),
            fontsize=9,
        )
        return scatter_artists

    anim = FuncAnimation(
        fig,
        _update,
        frames   = len(orders_sorted),
        interval = interval_ms,
        blit     = False,
        repeat   = False,
    )
    fig.tight_layout()
    return anim


# ---------------------------------------------------------------------------
# Convenience: called directly from main.py
# ---------------------------------------------------------------------------

def show_comparison(
    comparison:      ComparisonResult,
    graph:           object,
    store_locations: List[Tuple[float, float]],
    save_path:       Optional[str] = None,
    sla_minutes:     float = 30.0,
) -> None:
    """
    Plot a comparison and either save it or display it interactively.

    Args:
        comparison:       Output of compare_strategies().
        graph:            OSMnx MultiDiGraph.
        store_locations:  List of (lat, lon) tuples.
        save_path:        If given, saves to this file path (e.g. 'out.png').
                          Otherwise calls plt.show().
        sla_minutes:      SLA threshold for colour coding.
    """
    fig = plot_comparison(comparison, graph, store_locations, sla_minutes=sla_minutes)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[map_viz] Saved comparison map → {save_path}")
    else:
        plt.show()
    plt.close(fig)