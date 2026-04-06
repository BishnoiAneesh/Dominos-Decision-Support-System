"""
geo/road_sampler.py
--------------------
Edge-based road-network sampling.

All sampled points lie on actual road edges — not just at nodes — giving
a realistic spatial distribution of customer locations.

Key functions
-------------
extract_edges(G)                 → list of edge dicts with geometry
sample_point_on_edge(edge, rng)  → (lat, lon)
sample_random_road_point(G, rng) → (lat, lon)
snap_to_nearest_node(G, lat, lon)→ (lat, lon)  — for store placement
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

# Shapely is a dependency of OSMnx; always available when OSMnx is installed.
try:
    from shapely.geometry import LineString, Point as ShapelyPoint
    _SHAPELY_AVAILABLE = True
except ImportError:
    _SHAPELY_AVAILABLE = False

LatLon = Tuple[float, float]


# ---------------------------------------------------------------------------
# Edge extraction
# ---------------------------------------------------------------------------

def extract_edges(G: Any) -> list[dict]:
    """
    Return a list of edge attribute dicts, each augmented with
    'u' (start node id) and 'v' (end node id) keys.

    Only edges with both endpoint nodes present in G are included
    (guards against malformed graphs).

    Args:
        G: OSMnx MultiDiGraph.

    Returns:
        List of dicts with at minimum keys: 'u', 'v', and optionally 'geometry'.
    """
    nodes = G.nodes(data=True)
    node_ids = set(dict(nodes).keys())

    edges = []
    for u, v, data in G.edges(data=True):
        if u in node_ids and v in node_ids:
            edges.append({"u": u, "v": v, **data})
    return edges


# ---------------------------------------------------------------------------
# Point sampling on a single edge
# ---------------------------------------------------------------------------

def sample_point_on_edge(edge: dict, G: Any, rng: np.random.Generator) -> LatLon:
    """
    Sample a uniformly random point along a road edge.

    If the edge has a Shapely LineString geometry, interpolates at a
    random fraction of its total length. Otherwise falls back to linear
    interpolation between start and end node coordinates.

    Args:
        edge: Edge dict as returned by extract_edges().
        G:    OSMnx MultiDiGraph (used to look up node coordinates).
        rng:  Seeded numpy Generator for reproducibility.

    Returns:
        (lat, lon) of the sampled point.
    """
    t = float(rng.uniform(0.0, 1.0))

    # --- Geometry-based interpolation (preferred) ---
    if _SHAPELY_AVAILABLE and "geometry" in edge:
        geom = edge["geometry"]
        if isinstance(geom, LineString):
            point: ShapelyPoint = geom.interpolate(t, normalized=True)
            # OSMnx stores coordinates as (lon, lat)
            return (point.y, point.x)

    # --- Fallback: linear interpolation between endpoint nodes ---
    nodes = dict(G.nodes(data=True))
    u_data = nodes[edge["u"]]
    v_data = nodes[edge["v"]]
    lat = u_data["y"] + t * (v_data["y"] - u_data["y"])
    lon = u_data["x"] + t * (v_data["x"] - u_data["x"])
    return (lat, lon)


# ---------------------------------------------------------------------------
# Random road point sampler
# ---------------------------------------------------------------------------

def sample_random_road_point(
    G:     Any,
    rng:   np.random.Generator,
    edges: list[dict] | None = None,
) -> LatLon:
    """
    Sample a uniformly random point along any road edge in the graph.

    Edges are chosen with equal probability (not weighted by length).
    For length-weighted sampling, pre-weight edges by their 'length'
    attribute before passing them in.

    Args:
        G:     OSMnx MultiDiGraph.
        rng:   Seeded numpy Generator.
        edges: Pre-extracted edge list (pass this to avoid re-extracting
               on every call — critical for performance in tight loops).

    Returns:
        (lat, lon) of the sampled point.
    """
    if edges is None:
        edges = extract_edges(G)
    if not edges:
        raise ValueError("Graph has no valid edges to sample from.")

    edge = edges[int(rng.integers(0, len(edges)))]
    return sample_point_on_edge(edge, G, rng)


# ---------------------------------------------------------------------------
# Node snapping (for store placement)
# ---------------------------------------------------------------------------

def snap_to_nearest_node(G: Any, lat: float, lon: float) -> LatLon:
    """
    Return the (lat, lon) of the road node nearest to (lat, lon).

    Use this to place stores at valid road-network locations.

    Args:
        G:   OSMnx MultiDiGraph.
        lat: Latitude of the raw location.
        lon: Longitude of the raw location.

    Returns:
        (lat, lon) of the nearest graph node.
    """
    try:
        import osmnx as ox
    except ImportError as e:
        raise ImportError("OSMnx required for snap_to_nearest_node.") from e

    node_id = ox.nearest_nodes(G, lon, lat)   # OSMnx takes (X=lon, Y=lat)
    node    = G.nodes[node_id]
    return (node["y"], node["x"])