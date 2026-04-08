"""
geo_constants.py
----------------
Single source of truth for the map bounding box used by:
  • geo/map_loader.py   — graph download area
  • app.py              — valid store placement area + order generation area
  • main.py             — store_locations sanity reference

All store placements and order generation locations must fall within
[SOUTH, NORTH] × [WEST, EAST].  Any code that places stores or samples
demand should import from here rather than hard-coding lat/lon values.
"""

# Centre point — Noida / South-East Delhi
MAP_LAT  = 28.5355
MAP_LON  = 77.3910

# Half-extent in degrees (~7 km per 0.07°lat → ~15 km bounding box)
MAP_DIST = 0.03

# Derived bounding box — use these everywhere
MAP_NORTH  = MAP_LAT + MAP_DIST
MAP_SOUTH  = MAP_LAT - MAP_DIST
MAP_EAST   = MAP_LON + MAP_DIST
MAP_WEST   = MAP_LON - MAP_DIST

MAP_CENTER = [MAP_LAT, MAP_LON]
MAP_ZOOM   = 13

MAP_BOUNDS = {
    "north": MAP_NORTH,
    "south": MAP_SOUTH,
    "east":  MAP_EAST,
    "west":  MAP_WEST,
}
