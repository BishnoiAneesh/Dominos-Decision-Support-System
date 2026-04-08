"""
geo/map_loader.py
-----------------
Loads a drivable road-network graph for a 15 km bounding box centred on
Noida / South-East Delhi using OSMnx.

Graph is cached to disk as a GraphML file so it is downloaded only once.
A module-level singleton (_GRAPH_CACHE) ensures the graph is loaded only
once per process, regardless of how many times load_city_graph() is called.

Usage::

    from geo.map_loader import load_city_graph
    G = load_city_graph()          # first call downloads if file absent
    G = load_city_graph()          # subsequent calls return cached object
"""

from __future__ import annotations

import logging
import os
from typing import Any
from shapely.geometry import box

from geo.geo_constants import MAP_LAT, MAP_LON, MAP_DIST, MAP_NORTH, MAP_SOUTH, MAP_EAST, MAP_WEST

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Geography constants (imported from geo_constants — do not duplicate here)
# ---------------------------------------------------------------------------

LAT   = MAP_LAT
LON   = MAP_LON
DIST  = MAP_DIST

NORTH = MAP_NORTH
SOUTH = MAP_SOUTH
EAST  = MAP_EAST
WEST  = MAP_WEST

GRAPHML_PATH = f"graph_{LAT}_{LON}_{DIST}.graphml"

# ---------------------------------------------------------------------------
# Module-level cache — graph loaded at most once per process
# ---------------------------------------------------------------------------

_GRAPH_CACHE: Any | None = None


def load_city_graph(path: str = GRAPHML_PATH) -> Any:
    print("NORTH:", NORTH)
    print("SOUTH:", SOUTH)
    print("EAST:", EAST)
    print("WEST:", WEST)
    print("DELTA LAT:", NORTH - SOUTH)
    print("DELTA LON:", EAST - WEST)
    """
    Return the drivable road-network graph for the configured bounding box.

    Load order:
      1. Return module-level cache if already loaded.
      2. Load from GraphML file at `path` if it exists.
      3. Download from OSM, save to `path`, then return.

    Args:
        path: Filesystem path for the cached GraphML file.

    Returns:
        A NetworkX MultiDiGraph as returned by OSMnx.

    Raises:
        ImportError: If OSMnx is not installed.
    """
    global _GRAPH_CACHE
    if _GRAPH_CACHE is not None:
        return _GRAPH_CACHE

    try:
        import osmnx as ox
    except ImportError as e:
        raise ImportError(
            "OSMnx is required for real-map mode. "
            "Install it with: pip install osmnx"
        ) from e

    if os.path.exists(path):
        logger.info("Loading road graph from %s …", path)
        G = ox.load_graphml(path)
    else:
        logger.info(
            "Downloading road graph (bbox: N=%.4f S=%.4f E=%.4f W=%.4f) …",
            NORTH, SOUTH, EAST, WEST,
        )
        polygon = box(MAP_WEST, MAP_SOUTH, MAP_EAST, MAP_NORTH)

        G = ox.graph_from_polygon(
            polygon,
            network_type="drive",
        )

        ox.save_graphml(G, path)
        logger.info("Road graph saved to %s", path)

    _GRAPH_CACHE = G
    return G


def clear_cache() -> None:
    """Reset the module-level graph cache (useful for testing)."""
    global _GRAPH_CACHE
    _GRAPH_CACHE = None
