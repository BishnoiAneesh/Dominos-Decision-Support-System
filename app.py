"""
app.py
------
Streamlit interactive UI for the last-mile delivery simulation.

Layout
------
Left sidebar  → configuration panel + store placement
Main area     → live order table + live map

Run with:
    streamlit run app.py

Threading model
---------------
The simulation runs in a background daemon thread.  Session state must
NEVER be written from that thread (Streamlit raises ScriptRunContext errors).
Instead the thread pushes event dicts onto a stdlib ``queue.Queue`` stored in
session state as a plain Python object.  The main Streamlit thread drains the
queue on every rerun and merges events into session state safely.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Delivery Simulation",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Map boundary constants — single source of truth in geo_constants.py
# ---------------------------------------------------------------------------
from geo.geo_constants import MAP_BOUNDS, MAP_CENTER, MAP_ZOOM


def _clamp_to_bounds(lat: float, lon: float) -> tuple[float, float]:
    """Clamp a (lat, lon) point into the valid map bounding box."""
    lat = max(MAP_BOUNDS["south"], min(MAP_BOUNDS["north"], lat))
    lon = max(MAP_BOUNDS["west"],  min(MAP_BOUNDS["east"],  lon))
    return lat, lon


def _within_bounds(lat: float, lon: float) -> bool:
    return (
        MAP_BOUNDS["south"] <= lat <= MAP_BOUNDS["north"]
        and MAP_BOUNDS["west"] <= lon <= MAP_BOUNDS["east"]
    )


def _add_boundary_rect(fmap: folium.Map) -> None:
    """Draw the valid store-placement boundary on a Folium map."""
    folium.Rectangle(
        bounds=[
            [MAP_BOUNDS["south"], MAP_BOUNDS["west"]],
            [MAP_BOUNDS["north"], MAP_BOUNDS["east"]],
        ],
        color="#9C27B0",
        weight=2,
        fill=True,
        fill_color="#9C27B0",
        fill_opacity=0.04,
        tooltip="Valid store placement area",
        dash_array="6 4",
    ).add_to(fmap)


# ---------------------------------------------------------------------------
# Lazy simulation imports
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading road network graph…")
def _load_graph():
    from geo.map_loader import load_city_graph
    return load_city_graph()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REDRAW_EVERY = 5

TABLE_COLUMNS = [
    "Order ID", "Created Time", "Main Items", "Side Items",
    "Nearest Store", "Nearest Store Time", "Nearest Delivery Time", "Nearest Total Time",
    "Optimized Store", "Optimized Store Time", "Optimized Delivery Time", "Optimized Total Time",
    "Status",
]

COLOR_CSS = {
    "#2196F3": "background-color:#2196F3;color:white",
    "#4CAF50": "background-color:#4CAF50;color:white",
    "#FFC107": "background-color:#FFC107;color:black",
    "#F44336": "background-color:#F44336;color:white",
}

_FOLIUM_COLORS = {
    "#2196F3": "blue",
    "#4CAF50": "green",
    "#FFC107": "orange",
    "#F44336": "red",
}


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults: dict[str, Any] = {
        "store_locations": [],
        "sim_running":     False,
        "sim_done":        False,
        "order_rows":      {},        # order_id → latest row dict
        "event_count":     0,
        "map_version":     0,
        "result":          None,
        # Thread-safe queue: written by background thread, read by main thread
        # We store it as a plain Python object so Streamlit never serialises it.
        "_event_queue":    None,
        # Execution mode: "precompute" | "direct"
        "exec_mode":       "precompute",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Ensure queue object exists (survives hot-reloads)
    if st.session_state["_event_queue"] is None:
        st.session_state["_event_queue"] = queue.Queue()

_init_state()


# ---------------------------------------------------------------------------
# Drain the thread-safe queue into session state (called every rerun)
# ---------------------------------------------------------------------------
def _drain_queue() -> int:
    """Pull all pending events from the background queue into session state.
    Returns the number of new events processed."""
    q: queue.Queue = st.session_state["_event_queue"]
    processed = 0
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break

        # Sentinel: simulation finished
        if item is None:
            st.session_state.sim_running = False
            st.session_state.sim_done    = True
            break

        # Result object sent as a 2-tuple ("result", SimulationResult)
        if isinstance(item, tuple) and item[0] == "result":
            st.session_state.result      = item[1]
            st.session_state.map_version += 1
            continue

        # Normal event dict
        event    = item
        order_id = event["order_id"]
        row = {
            "Order ID":                order_id,
            "Created Time":            event["created_time"],
            "Main Items":              event["mains"],
            "Side Items":              event["sides"],
            "Nearest Store":           event["nearest_store"],
            "Nearest Store Time":      event["nearest_store_time"],
            "Nearest Delivery Time":   event["nearest_delivery_time"],
            "Nearest Total Time":      event["nearest_total_time"],
            "Optimized Store":         event["optimized_store"],
            "Optimized Store Time":    event["optimized_store_time"],
            "Optimized Delivery Time": event["optimized_delivery_time"],
            "Optimized Total Time":    event["optimized_total_time"],
            "Status":                  event["status"],
            "_color":                  event["final_color"],
            "_lat":                    event["lat"],
            "_lon":                    event["lon"],
        }
        st.session_state.order_rows[order_id] = row
        st.session_state.event_count += 1
        processed += 1

    if processed > 0 and processed % REDRAW_EVERY == 0:
        st.session_state.map_version += 1

    return processed


# ---------------------------------------------------------------------------
# Background simulation thread
# ---------------------------------------------------------------------------
def _build_config(time_horizon, lam, sla_max, sla_prob):
    from config import SimConfig
    cfg = SimConfig()
    cfg.simulation.time_horizon_minutes = float(time_horizon)
    cfg.demand.poisson_lambda           = float(lam)
    cfg.sla.max_delivery_minutes        = float(sla_max)
    cfg.sla.probability_threshold       = float(sla_prob)
    return cfg


def _run_simulation_thread(config, store_locations, graph, demand_config, event_queue, exec_mode):
    """
    Runs entirely in a background thread.
    All communication back to the UI goes through event_queue — never
    through st.session_state directly.
    """
    from simulation.engine import run_simulation_with_events

    def handle_event(event: dict) -> None:
        event_queue.put(event)

    try:
        result = run_simulation_with_events(
            config          = config,
            store_locations = store_locations,
            graph           = graph,
            demand_config   = demand_config,
            event_callback  = handle_event,
            precompute      = (exec_mode == "precompute"),
        )
        event_queue.put(("result", result))
    except Exception as exc:
        # Put a sentinel so the UI doesn't hang, then re-raise for logging
        event_queue.put(None)
        raise
    finally:
        event_queue.put(None)   # sentinel: done


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Simulation Config")

    st.subheader("Demand & Time")
    time_horizon = st.slider("Time Horizon (minutes)", 30, 480, 180, step=10)
    lam          = st.slider("Arrival Rate λ (orders/min)", 0.1, 5.0, 0.6, step=0.1)

    st.subheader("SLA")
    sla_max  = st.slider("Max Delivery Time (min)", 15, 60, 30, step=5)
    sla_prob = st.slider("SLA Probability Threshold", 0.50, 0.99, 0.95, step=0.01)

    st.subheader("⚡ Execution Mode")
    exec_mode_label = st.radio(
        "Routing strategy",
        options=["Preconfigure Dijkstra - Slow Prep, Fast simulation", "Real-time Dijkstra - Fast Prep, Slow simulation"],
        index=0,
        help=(
            "**Preconfigure Dijkstra** — precomputes all store→node distances before the "
            "simulation loop (Dijkstra once per store). Best for long runs.\n\n"
            "**Real-time Dijkstra** — runs Dijkstra on-the-fly per order. Slower but "
            "avoids the upfront precomputation cost."
        ),
    )
    exec_mode = "precompute" if exec_mode_label == "Preconfigure Dijkstra - Slow Prep, Fast simulation" else "direct"
    st.session_state.exec_mode = exec_mode

    st.divider()
    st.subheader("🏪 Place Stores")
    st.caption(
        "Click inside the **purple boundary** to add a store. "
        "Points outside are clamped to the valid region."
    )

    # Store placement map with boundary rectangle
    placement_map = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, height=320)
    _add_boundary_rect(placement_map)
    for i, loc in enumerate(st.session_state.store_locations):
        folium.Marker(
            location=loc,
            tooltip=f"Store {i}",
            icon=folium.Icon(color="red", icon="home", prefix="fa"),
        ).add_to(placement_map)

    placement_data = st_folium(
        placement_map,
        key="placement_map",
        height=320,
        returned_objects=["last_clicked"],
    )

    if placement_data and placement_data.get("last_clicked"):
        click   = placement_data["last_clicked"]
        raw_lat = click["lat"]
        raw_lon = click["lng"]

        # Clamp to valid boundary
        new_lat, new_lon = _clamp_to_bounds(raw_lat, raw_lon)
        new_loc = [new_lat, new_lon]

        if not _within_bounds(raw_lat, raw_lon):
            st.warning(
                f"Click ({raw_lat:.4f}, {raw_lon:.4f}) was outside the valid area "
                f"and was clamped to ({new_lat:.4f}, {new_lon:.4f}).",
                icon="⚠️",
            )

        if new_loc not in st.session_state.store_locations:
            st.session_state.store_locations.append(new_loc)
            st.rerun()

    if st.session_state.store_locations:
        st.write(f"**{len(st.session_state.store_locations)} store(s) placed:**")
        for i, loc in enumerate(st.session_state.store_locations):
            col_a, col_b = st.columns([3, 1])
            col_a.write(f"S{i}: ({loc[0]:.4f}, {loc[1]:.4f})")
            if col_b.button("✕", key=f"del_store_{i}"):
                st.session_state.store_locations.pop(i)
                st.rerun()
    else:
        st.info("No stores placed yet.")

    if st.button("Clear All Stores"):
        st.session_state.store_locations = []
        st.rerun()

    st.divider()

    run_disabled = (
        st.session_state.sim_running
        or len(st.session_state.store_locations) == 0
    )
    run_clicked = st.button(
        "▶ Run Simulation",
        disabled=run_disabled,
        type="primary",
        width='stretch',
    )

    if run_disabled and not st.session_state.sim_running:
        st.caption("Place at least one store to enable the simulation.")

    if st.button("🔄 Reset", width='stretch'):
        st.session_state.sim_running = False
        st.session_state.sim_done    = False
        st.session_state.order_rows  = {}
        st.session_state.event_count = 0
        st.session_state.map_version = 0
        st.session_state.result      = None
        st.session_state["_event_queue"] = queue.Queue()
        st.rerun()


# ---------------------------------------------------------------------------
# Drain pending events before rendering (runs on every rerun)
# ---------------------------------------------------------------------------
_drain_queue()


# ---------------------------------------------------------------------------
# Trigger simulation
# ---------------------------------------------------------------------------
if run_clicked and not st.session_state.sim_running:
    st.session_state.order_rows      = {}
    st.session_state.event_count     = 0
    st.session_state.map_version     = 0
    st.session_state.result          = None
    st.session_state.sim_done        = False
    st.session_state.sim_running     = True
    st.session_state["_event_queue"] = queue.Queue()

    graph  = _load_graph()
    config = _build_config(time_horizon, lam, sla_max, sla_prob)

    from simulation.demand import DemandGeneratorConfig
    demand_config = DemandGeneratorConfig.from_real_map(
        graph          = graph,
        arrival_params = {"lam": lam},
        time_horizon   = time_horizon,
        seed           = config.simulation.random_seed,
    )

    store_locs = [tuple(s) for s in st.session_state.store_locations]

    t = threading.Thread(
        target = _run_simulation_thread,
        args   = (
            config,
            store_locs,
            graph,
            demand_config,
            st.session_state["_event_queue"],
            st.session_state.exec_mode,
        ),
        daemon = True,
    )
    t.start()
    st.rerun()


# ---------------------------------------------------------------------------
# Auto-refresh while simulation is running
# ---------------------------------------------------------------------------
if st.session_state.sim_running:
    time.sleep(0.4)
    st.rerun()


# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------
st.title("🚚 Last-Mile Delivery Simulation")

# ---- Status banner ----
if st.session_state.sim_running:
    mode_label = "Fast runtime" if st.session_state.exec_mode == "precompute" else "Real-time"
    st.info(
        f"⏳ Simulation running [{mode_label}]… "
        f"{st.session_state.event_count} events received"
    )
elif st.session_state.sim_done and st.session_state.result:
    r = st.session_state.result
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Orders",  r.total_orders)
    c2.metric("SLA Met",       f"{r.sla_rate:.1%}")
    c3.metric("Avg Delivery",  f"{r.avg_delivery_time:.1f} min")
elif not st.session_state.sim_done:
    st.caption(
        "Configure parameters in the sidebar and place stores on the map, "
        "then click **▶ Run Simulation**."
    )

st.divider()

col_table, col_map = st.columns([1.1, 1], gap="medium")

# ================================================================
# LEFT: Live order table
# ================================================================
with col_table:
    st.subheader("📋 Order Table")

    rows = list(st.session_state.order_rows.values())

    if not rows:
        st.caption("Orders will appear here as the simulation runs.")
    else:
        display_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows]
        df = pd.DataFrame(display_rows, columns=TABLE_COLUMNS)

        # --- Feature 3: Color only the two total-time columns independently ---
        def _sla_color_css(value: float) -> str:
            """Return CSS for a single SLA time cell."""
            try:
                v = float(value)
            except (TypeError, ValueError):
                return ""
            if v < 25.0:
                return "background-color:#4CAF50;color:white"
            if v <= 30.0:
                return "background-color:#FFC107;color:black"
            return "background-color:#F44336;color:white"

        def _style_total_columns(df: pd.DataFrame) -> pd.DataFrame:
            """Return a DataFrame of CSS strings; only color the two total columns."""
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for col in ("Nearest Total Time", "Optimized Total Time"):
                if col in df.columns:
                    styles[col] = df[col].apply(_sla_color_css)
            return styles

        styled = df.style.apply(_style_total_columns, axis=None)
        st.dataframe(styled, width='stretch', height=520)

        leg_cols = st.columns(3)
        legend = [
            ("🟢", "< 25 min"),
            ("🟡", "25–30 min"),
            ("🔴", "> 30 min"),
        ]
        for col, (icon, label) in zip(leg_cols, legend):
            col.caption(f"{icon} {label}")

# ================================================================
# RIGHT: Live map  — single map (live during simulation)
# ================================================================
with col_map:
    st.subheader("🗺️ Live Map")

    fmap = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM)
    _add_boundary_rect(fmap)

    for i, loc in enumerate(st.session_state.store_locations):
        folium.Marker(
            location = loc,
            tooltip  = f"Store {i}",
            icon     = folium.Icon(color="purple", icon="home", prefix="fa"),
        ).add_to(fmap)

    for row in list(st.session_state.order_rows.values()):
        color    = row.get("_color", "#2196F3")
        lat, lon = row["_lat"], row["_lon"]
        folium.CircleMarker(
            location     = [lat, lon],
            radius       = 5,
            color        = color,
            fill         = True,
            fill_color   = color,
            fill_opacity = 0.8,
            tooltip      = (
                f"Order {row['Order ID']} | {row['Status']} | "
                f"Optimized Total: {row['Optimized Total Time']} min"
            ),
        ).add_to(fmap)

    st_folium(
        fmap,
        key              = f"live_map_{st.session_state.map_version}",
        height           = 540,
        returned_objects = [],
    )

    st.caption(
        "🟣 Store &nbsp;&nbsp; 🔵 Created &nbsp;&nbsp; "
        "🟢 <25 min &nbsp;&nbsp; 🟡 25–30 min &nbsp;&nbsp; 🔴 >30 min"
    )

# ================================================================
# Feature 2: Dual maps — shown after simulation completes
# ================================================================
if st.session_state.sim_done and st.session_state.order_rows:
    st.divider()
    st.subheader("🗺️ Strategy Comparison Maps")

    rows_all = list(st.session_state.order_rows.values())

    def _sla_color_for_time(total_time) -> str:
        try:
            v = float(total_time)
        except (TypeError, ValueError):
            return "#2196F3"
        if v < 25.0:
            return "#4CAF50"
        if v <= 30.0:
            return "#FFC107"
        return "#F44336"

    def _build_strategy_map(
        store_locations: list,
        order_rows: list,
        total_time_key: str,
        store_key: str,
        map_key: str,
    ) -> folium.Map:
        fmap = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM)
        _add_boundary_rect(fmap)
        for i, loc in enumerate(store_locations):
            folium.Marker(
                location = loc,
                tooltip  = f"Store {i}",
                icon     = folium.Icon(color="purple", icon="home", prefix="fa"),
            ).add_to(fmap)
        for row in order_rows:
            color    = _sla_color_for_time(row.get(total_time_key))
            lat, lon = row["_lat"], row["_lon"]
            folium.CircleMarker(
                location     = [lat, lon],
                radius       = 5,
                color        = color,
                fill         = True,
                fill_color   = color,
                fill_opacity = 0.8,
                tooltip      = (
                    f"Order {row['Order ID']} | "
                    f"Store {row[store_key]} | "
                    f"Total: {row[total_time_key]} min"
                ),
            ).add_to(fmap)
        return fmap

    map_col1, map_col2 = st.columns(2, gap="small")

    with map_col1:
        st.markdown("**Nearest Strategy**")
        nearest_map = _build_strategy_map(
            store_locations = st.session_state.store_locations,
            order_rows      = rows_all,
            total_time_key  = "Nearest Total Time",
            store_key       = "Nearest Store",
            map_key         = "nearest_strategy_map",
        )
        st_folium(
            nearest_map,
            key              = f"nearest_map_{st.session_state.map_version}",
            height           = 420,
            returned_objects = [],
        )

    with map_col2:
        st.markdown("**Optimized Strategy**")
        optimized_map = _build_strategy_map(
            store_locations = st.session_state.store_locations,
            order_rows      = rows_all,
            total_time_key  = "Optimized Total Time",
            store_key       = "Optimized Store",
            map_key         = "optimized_strategy_map",
        )
        st_folium(
            optimized_map,
            key              = f"optimized_map_{st.session_state.map_version}",
            height           = 420,
            returned_objects = [],
        )

    st.caption(
        "🟢 <25 min &nbsp;&nbsp; 🟡 25–30 min &nbsp;&nbsp; 🔴 >30 min &nbsp;&nbsp; "
        "Colors reflect each strategy's own store selection and total time."
    )

# ================================================================
# Feature 4: Store-wise Strategy Statistics (shown after sim done)
# ================================================================
if st.session_state.sim_done and st.session_state.result:
    result = st.session_state.result
    st.divider()
    st.subheader("📊 Strategy Statistics")

    # --- Summary row ---
    nearest_sla_met   = sum(m.sla_met_count   for m in result.nearest_store_metrics.values())
    nearest_orders    = sum(m.orders_assigned  for m in result.nearest_store_metrics.values())
    nearest_avg_total = (
        sum(m.total_queue_delay + m.total_prep_time + m.total_travel_time
            for m in result.nearest_store_metrics.values()) / nearest_orders
        if nearest_orders else 0.0
    )
    nearest_avg_store = (
        sum(m.total_queue_delay + m.total_prep_time
            for m in result.nearest_store_metrics.values()) / nearest_orders
        if nearest_orders else 0.0
    )

    opt_sla_met   = sum(m.sla_met_count   for m in result.store_metrics.values())
    opt_orders    = sum(m.orders_assigned  for m in result.store_metrics.values())
    opt_avg_total = (
        sum(m.total_queue_delay + m.total_prep_time + m.total_travel_time
            for m in result.store_metrics.values()) / opt_orders
        if opt_orders else 0.0
    )
    opt_avg_store = (
        sum(m.total_queue_delay + m.total_prep_time
            for m in result.store_metrics.values()) / opt_orders
        if opt_orders else 0.0
    )

    st.markdown("#### Overall Summary")
    sum_col1, sum_col2 = st.columns(2, gap="medium")

    with sum_col1:
        st.markdown("**Nearest Strategy**")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total Orders",    nearest_orders)
        sc2.metric("SLA Rate",        f"{nearest_sla_met / nearest_orders:.1%}" if nearest_orders else "—")
        sc3.metric("Avg Store Time",  f"{nearest_avg_store:.1f} min")
        sc4.metric("Avg Total Time",  f"{nearest_avg_total:.1f} min")

    with sum_col2:
        st.markdown("**Optimized Strategy**")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Total Orders",    opt_orders)
        sc2.metric("SLA Rate",        f"{opt_sla_met / opt_orders:.1%}" if opt_orders else "—")
        sc3.metric("Avg Store Time",  f"{opt_avg_store:.1f} min")
        sc4.metric("Avg Total Time",  f"{opt_avg_total:.1f} min")

    # --- Per-store breakdown table ---
    st.markdown("#### Per-Store Breakdown")

    all_store_ids = sorted(
        set(result.nearest_store_metrics) | set(result.store_metrics)
    )

    store_rows = []
    for sid in all_store_ids:
        nm = result.nearest_store_metrics.get(sid)
        om = result.store_metrics.get(sid)
        store_rows.append({
            "Store": f"S{sid}",
            # Nearest columns
            "N Orders":     nm.orders_assigned if nm else 0,
            "N Avg Store":  f"{nm.utilization:.1f}"    if nm else "—",
            "N Avg Total":  f"{nm.avg_total_time:.1f}" if nm else "—",
            "N SLA Rate":   f"{nm.sla_rate:.1%}"       if nm else "—",
            # Optimized columns
            "O Orders":     om.orders_assigned if om else 0,
            "O Avg Store":  f"{om.utilization:.1f}"    if om else "—",
            "O Avg Total":  f"{om.avg_total_time:.1f}" if om else "—",
            "O SLA Rate":   f"{om.sla_rate:.1%}"       if om else "—",
        })

    store_df = pd.DataFrame(store_rows)
    store_df.columns = pd.MultiIndex.from_tuples([
        ("", "Store"),
        ("Nearest Strategy", "Orders"), ("Nearest Strategy", "Avg Store Time"),
        ("Nearest Strategy", "Avg Total Time"), ("Nearest Strategy", "SLA Rate"),
        ("Optimized Strategy", "Orders"), ("Optimized Strategy", "Avg Store Time"),
        ("Optimized Strategy", "Avg Total Time"), ("Optimized Strategy", "SLA Rate"),
    ])
    st.dataframe(store_df, width='stretch')