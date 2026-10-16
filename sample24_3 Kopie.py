# ── SSL certificate fix (Python 3.14 on macOS) ───────────────────────────────
# Must be done before importing requests/urllib3.
import os
import ssl
try:
    import certifi as _certifi_early
    _CA = _certifi_early.where()
    os.environ["REQUESTS_CA_BUNDLE"] = _CA
    os.environ["SSL_CERT_FILE"]      = _CA
    os.environ["CURL_CA_BUNDLE"]     = _CA
except ImportError:
    _CA = None

import ssl
import json
import math
import time
import requests
import cv2
import numpy as np
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString, Polygon
from shapely.ops import unary_union
from pyproj import Transformer
from geopy.distance import geodesic

# Build a persistent session that always uses the certifi CA bundle.
# All HTTP calls in this script go through _SESSION.get / _SESSION.post
# so the verify= parameter is applied consistently everywhere.
_SESSION = requests.Session()
if _CA:
    _SESSION.verify = _CA


def _get(url, **kwargs):
    """Wrapper around _SESSION.get with SSL-error retry logic.
    On transient SSL errors, waits briefly and retries up to 3 times
    so a single certificate hiccup does not crash the whole script."""
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            return _SESSION.get(url, **kwargs)
        except requests.exceptions.SSLError as e:
            if attempt == max_retries:
                raise
            wait = attempt * 5
            print(f"  SSL error (attempt {attempt}/{max_retries}), "
                  f"retrying in {wait}s: {e}")
            time.sleep(wait)


def _post(url, **kwargs):
    """Same retry wrapper for POST."""
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            return _SESSION.post(url, **kwargs)
        except requests.exceptions.SSLError as e:
            if attempt == max_retries:
                raise
            wait = attempt * 5
            print(f"  SSL error (attempt {attempt}/{max_retries}), "
                  f"retrying in {wait}s: {e}")
            time.sleep(wait)

# ======================================================
# CONFIGURATION
# ======================================================

API_KEY = "DEIN API SCHLÜSSEL HIER"

# The 4 streets that form the Gründerzeit block (order does not matter)
STREET_NAMES = [
    "Laxenburger Straße",
    "Landgutgasse",
    "Jagdgasse",
    "Dampfgasse"
]

CITY = "Vienna, Austria"

SCAN_DISTANCE_M   = 2.5
RAY_LENGTH_M      = 60
MAX_ANGLE_DEG     = 75.0
MAX_FACADE_DIST_M = 15.0   # metres — panos further than this from the facade
                            # (e.g. on the far lane of a wide road) are skipped

BASE_OUT_DIR = "samples"

CREATE_SESSION_URL = "https://tile.googleapis.com/v1/createSession"
META_URL           = "https://tile.googleapis.com/v1/streetview/metadata"
TILE_URL           = "https://tile.googleapis.com/v1/streetview/tiles"

# ======================================================
# GEOMETRY UTILITIES
# ======================================================

def bearing(p1, p2):
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def unit_vec(angle):
    r = math.radians(angle)
    return math.sin(r), math.cos(r)

def angle_diff(a, b):
    """Absolute angular difference between two bearings, in [0, 180]."""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d

def cols_for_bearings(ray_start_abs, ray_end_abs, camera_heading, deg_per_tile, tiles_x):
    """
    Return tile column indices covering the angular window between two
    absolute geographic bearings.

    ray_start_abs : bearing of the start-side fan ray  (geographic, 0-360°)
    ray_end_abs   : bearing of the end-side fan ray    (geographic, 0-360°)
    camera_heading: panorama camera heading from Street View metadata
    deg_per_tile  : degrees per tile column = 360 / tiles_x

    EMPIRICALLY VERIFIED (2026-06-30, pano DNnFJAxRBGLEdssduRiEtw):
    geo_bearing(col) = (col * deg_per_tile + camera_heading) % 360
    i.e. column 0 corresponds to camera_heading, NOT geographic North.
    Confirmed by downloading single tiles at col=0, col=9, col=25 and
    visually comparing facade vs. street-view content against expected
    bearings.  Do NOT remove the camera_heading offset again without
    re-running that empirical test — a previous attempt to do so based on
    a flawed test script caused a regression (Buchengasse/Rotenhofgasse
    panos showed the street instead of the facade).
    """
    a = (ray_start_abs - camera_heading) % 360
    b = (ray_end_abs   - camera_heading) % 360
    col_a = int(a / deg_per_tile) % tiles_x
    col_b = int(b / deg_per_tile) % tiles_x

    # Choose the arc direction (CW from a to b, or CW from b to a)
    # that spans the shorter angle — the facade window is always < 180°.
    cw_span = (col_b - col_a) % tiles_x   # columns from a → b clockwise
    if cw_span <= tiles_x // 2:
        # CW arc a → b
        if col_a <= col_b:
            cols = list(range(col_a, col_b + 1))
        else:
            cols = list(range(col_a, tiles_x)) + list(range(0, col_b + 1))
    else:
        # CCW arc a → b  (= CW arc b → a)
        if col_b <= col_a:
            cols = list(range(col_b, col_a + 1))
        else:
            cols = list(range(col_b, tiles_x)) + list(range(0, col_a + 1))

    if not cols:
        cols = [col_a]
    return cols


def resolve_facade_column(pano_pt, inner_normal_deg, cols_primary, cols_opposite,
                           buildings_all, probe_dist=12.0):
    """
    Determine whether cols_primary or cols_opposite is the column set that
    actually points at our target block's facade - using the Google
    Geocoding API (reverse geocoding) as the deciding signal, not pixel
    content and not a pure in-house distance-to-buildings_all check alone.

    WHY NOT JUST GEOMETRY (buildings_all distance):
    Empirically verified (2026-06-30) that inner_normal can point
    correctly at the block (geometrically verified against block_polygon)
    while cols_primary - derived from inner_normal via the
    camera_heading-based cols_for_bearings formula - silently points at
    the opposite (wrong) side in the ACTUAL downloaded tile content, with
    no predictor found in heading/tilt/roll metadata (pano
    GK1kS4tfUAYF1WtGPOUOPw, Neilreichgasse, is the confirmed case: a pure
    buildings_all-distance check confidently scored cols_primary as
    correct with dist=0.0m, yet the real tile showed the wrong building).
    A geometry-only check cannot catch this because it never looks at
    what the formula-selected column actually corresponds to in reality.

    WHY NOT A PROBE POINT NEAR THE PANO (radius-based panoId search):
    A point ~12m orthogonal to the street, in a dense Gruenderzeit block,
    usually lands inside a building footprint or an inner courtyard -
    locations where Street View almost never has a panorama standing.
    Searching for "the nearest panoId" from such a point is therefore
    unreliable; it may find a much more distant pano or no result at all.
    Reverse GEOCODING does not have this problem: it does not require a
    panorama to exist near the probe point, only a mapped building/address
    to exist there - which Google's address database covers far more
    densely than its Street View coverage.

    Method:
      1. Probe a point ~probe_dist metres from pano_pt in the
         inner_normal direction ("primary") and in the 180 degree
         opposite direction ("opposite").
      2. Reverse-geocode BOTH probe points via the Google Geocoding API
         (https://maps.googleapis.com/maps/api/geocode/json?latlng=...),
         which returns the nearest real-world address regardless of
         whether a Street View panorama exists there.
      3. Compare each geocoded (housenumber, street) pair against the
         addresses already present in buildings_all (which is restricted
         to buildings whose centroid lies inside block_polygon - i.e.
         "known to belong to our block"). A match means that probe side
         is confirmed to be our block; a non-match (different street, or
         a housenumber not in buildings_all) means that side is NOT our
         block.
      4. If exactly one side matches a buildings_all address, choose it.
         If both or neither match, fall back to the buildings_all
         distance check as a secondary signal, and flag the result as
         "undetermined" if that is also inconclusive - these cases should
         be reviewed manually.

    This uses one extra Google Geocoding API call per probe (2 per pano,
    independent of the Street View Tiles quota), in addition to the
    existing tile downloads.

    Returns: (chosen_cols, matched_side, detail_str)
      matched_side: "primary", "opposite", or "undetermined"
    """
    GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

    def _reverse_geocode(probe_pt_25833):
        """EPSG:25833 point -> (housenumber, street) via Google Geocoding API,
        or (None, None) if no result / request failed."""
        lng, lat = _proj_to_wgs.transform(probe_pt_25833.x, probe_pt_25833.y)
        try:
            r = _get(
                GEOCODE_URL,
                params={"latlng": f"{lat},{lng}", "key": API_KEY,
                        "result_type": "street_address"}
            )
            if r.status_code != 200:
                return None, None
            data = r.json()
        except Exception:
            return None, None

        if data.get("status") != "OK" or not data.get("results"):
            return None, None

        hn, st = None, None
        for comp in data["results"][0].get("address_components", []):
            types = comp.get("types", [])
            if "street_number" in types:
                hn = comp.get("long_name")
            if "route" in types:
                st = comp.get("long_name")
        return hn, st

    def _is_known_block_address(hn, st):
        """Check whether (hn, st) is in the geocoding-derived BLOCK_ADDRESSES
        set (built once at startup from probe points along block_polygon).
        This is more reliable than OSM addr:housenumber tags, which are
        often absent for dense Gruenderzeit blocks."""
        if not hn or not st:
            return False
        hn_norm = str(hn).strip()
        st_norm = str(st).strip().lower().replace("-", "").replace(" ", "")
        return (hn_norm, st_norm) in BLOCK_ADDRESSES

    def _nearest_block_building_dist(probe_pt):
        if buildings_all is None or len(buildings_all) == 0:
            return None
        dists = buildings_all.geometry.distance(probe_pt)
        return dists.min()

    nx_p, ny_p = unit_vec(inner_normal_deg)
    probe_primary  = Point(pano_pt.x + nx_p * probe_dist, pano_pt.y + ny_p * probe_dist)
    opp_deg = (inner_normal_deg + 180) % 360
    nx_o, ny_o = unit_vec(opp_deg)
    probe_opposite = Point(pano_pt.x + nx_o * probe_dist, pano_pt.y + ny_o * probe_dist)

    hn_p, st_p = _reverse_geocode(probe_primary)
    hn_o, st_o = _reverse_geocode(probe_opposite)

    match_p = _is_known_block_address(hn_p, st_p)
    match_o = _is_known_block_address(hn_o, st_o)

    detail = (f"primary: geocode={hn_p or '?'} {st_p or '?'} (in_block={match_p})  |  "
              f"opposite: geocode={hn_o or '?'} {st_o or '?'} (in_block={match_o})")

    if match_p and not match_o:
        return cols_primary, "primary", detail
    if match_o and not match_p:
        return cols_opposite, "opposite", detail

    # Geocoding was inconclusive (both/neither matched a known block
    # address) - fall back to the buildings_all distance check as a
    # secondary signal.
    d_p = _nearest_block_building_dist(probe_primary)
    d_o = _nearest_block_building_dist(probe_opposite)
    detail += f"  |  fallback distances: primary={d_p if d_p is None else round(d_p,1)}m opposite={d_o if d_o is None else round(d_o,1)}m"

    if d_p is not None and d_o is not None and d_p != d_o:
        if d_p < d_o:
            return cols_primary, "primary (fallback)", detail
        return cols_opposite, "opposite (fallback)", detail

    return cols_primary, "undetermined", detail


def inner_normal(street_bear, pano_pt, block_polygon, probe_dist=40.0):
    """Return whichever of the two perpendicular normals (left/right of street)
    points INTO the block polygon.

    The pano sits on the street, outside the block polygon.  A short probe in
    the correct normal direction will land inside the polygon; the wrong
    direction lands outside.  This is geometrically unambiguous and requires
    no building data — only the block polygon formed from street intersections.

    probe_dist: how far to shoot the probe (metres, EPSG:25833).  Must be long
    enough to cross the street and reach the polygon interior (~20-50 m is
    safe for Viennese Gründerzeit blocks).
    """
    n_left  = (street_bear + 90) % 360
    n_right = (street_bear - 90) % 360

    def _inside(angle):
        nx, ny = unit_vec(angle)
        probe = Point(pano_pt.x + nx * probe_dist,
                      pano_pt.y + ny * probe_dist)
        return block_polygon.contains(probe)

    if _inside(n_left):
        return n_left
    if _inside(n_right):
        return n_right
    # Fallback: neither probe lands inside (pano at corner / gap).
    # Return the normal whose probe is closer to the polygon boundary.
    nl_nx, nl_ny = unit_vec(n_left)
    nr_nx, nr_ny = unit_vec(n_right)
    probe_l = Point(pano_pt.x + nl_nx * probe_dist, pano_pt.y + nl_ny * probe_dist)
    probe_r = Point(pano_pt.x + nr_nx * probe_dist, pano_pt.y + nr_ny * probe_dist)
    dist_l  = block_polygon.exterior.distance(probe_l)
    dist_r  = block_polygon.exterior.distance(probe_r)
    print(f"  WARNING inner_normal fallback: neither probe inside block "
          f"(dist_l={dist_l:.1f} m  dist_r={dist_r:.1f} m) — choosing closer side")
    return n_left if dist_l <= dist_r else n_right

def flatten_geometry(geom):
    coords = []
    if isinstance(geom, LineString):
        coords.extend([(y, x) for x, y in geom.coords])
    elif isinstance(geom, MultiLineString):
        for g in geom.geoms:
            coords.extend([(y, x) for x, y in g.coords])
    else:
        raise TypeError("Unsupported geometry")
    return coords

def compute_dist_to_facade(pano_pt, in_normal, buildings, last_valid_dist):
    """Ray-cast from pano toward inner facade, return distance to nearest building edge.
    Falls back to last_valid_dist if no building is hit."""
    x, y  = pano_pt.x, pano_pt.y
    nx, ny = unit_vec(in_normal)
    ray   = LineString([pano_pt, Point(x + nx * RAY_LENGTH_M, y + ny * RAY_LENGTH_M)])
    hits  = [pano_pt.distance(g) for g in buildings.intersection(ray) if not g.is_empty]
    if hits:
        return min(hits)
    return last_valid_dist if last_valid_dist else 15.0

def corner_angle_for_edge_pano(pano_pt, inner_normal_deg, dist_to_facade,
                               building_block_outline,
                               direction_deg):
    """
    For a boundary pano (first or last on a street), snap the open angle
    to the building corner that is:
      a) on the facade side (close to the inner_normal ray), AND
      b) in direction_deg from the pano (street direction toward the gap).

    Strategy (revised):
      1. Collect all building outline corners.
      2. Keep only corners that lie on the FACADE SIDE of the pano:
         project (corner - pano) onto the inner_normal axis; keep corners
         with a positive projection (i.e. in front of the pano).
      3. From those, keep corners that lie in direction_deg from the pano:
         project (corner - pano) onto direction_deg axis; keep corners
         with a positive projection (i.e. ahead in street direction).
      4. Among the remaining corners, pick the one closest to the pano.
         This is the building corner at the near edge of the facade.
      5. Compute the LATERAL distance (perpendicular to inner_normal)
         from pano to that corner → atan(d_lateral / dist_to_facade).

    Returns (angle_deg, corner_Point) or (None, None) on failure.
    """
    if building_block_outline is None or building_block_outline.is_empty:
        return None, None

    # ── Collect all building outline corners ────────────────────────────
    if building_block_outline.geom_type == "Polygon":
        outline_coords = list(building_block_outline.exterior.coords[:-1])
    elif building_block_outline.geom_type == "MultiPolygon":
        outline_coords = []
        for part in building_block_outline.geoms:
            outline_coords.extend(list(part.exterior.coords[:-1]))
    else:
        return None, None

    if not outline_coords:
        return None, None

    # Direction vectors
    norm_rad = math.radians(inner_normal_deg)
    norm_vec = (math.sin(norm_rad), math.cos(norm_rad))   # toward facade

    dir_rad  = math.radians(direction_deg)
    dir_vec  = (math.sin(dir_rad), math.cos(dir_rad))     # along street

    # Minimum depth: a corner must be at least 20% of dist_to_facade
    # away from the pano in the normal direction to count as "facade side".
    min_depth = max(1.0, dist_to_facade * 0.2)

    candidates = []
    for cx, cy in outline_coords:
        dx = cx - pano_pt.x
        dy = cy - pano_pt.y
        depth  = dx * norm_vec[0] + dy * norm_vec[1]   # along inner_normal
        along  = dx * dir_vec[0]  + dy * dir_vec[1]    # along street dir
        if depth >= min_depth and along > 0:
            dist_to_pano = math.hypot(dx, dy)
            candidates.append((dist_to_pano, Point(cx, cy), depth, along))

    if not candidates:
        return None, None

    # Pick the corner closest to the pano (not the block corner far away)
    candidates.sort(key=lambda t: t[0])
    _, best_corner, _, _ = candidates[0]

    # ── Lateral distance (perpendicular to inner_normal) ────────────────
    lat_rad = math.radians(inner_normal_deg + 90)
    lat_vec = (math.sin(lat_rad), math.cos(lat_rad))
    dx = best_corner.x - pano_pt.x
    dy = best_corner.y - pano_pt.y
    d_lateral = abs(dx * lat_vec[0] + dy * lat_vec[1])

    if dist_to_facade <= 0:
        return None, None
    angle = math.degrees(math.atan(d_lateral / dist_to_facade))
    return min(angle, MAX_ANGLE_DEG), best_corner



# ======================================================
# CREATE TILE SESSION
# ======================================================

print("\n=== Creating Street View Tiles session ===")
resp = _post(
    CREATE_SESSION_URL,
    params={"key": API_KEY},
    json={"mapType": "streetview", "language": "en-US", "region": "AT"}
)
resp.raise_for_status()
session = resp.json()

session_id  = session["session"]
tile_width  = session.get("tileWidth",  512)
tile_height = session.get("tileHeight", 512)

print(f"Session created: {session_id}")
print(f"Tile size: {tile_width} x {tile_height}")

# ======================================================
# LOAD ALL 4 STREETS + BUILDINGS + COMPUTE BLOCK CENTROID
# ======================================================

print("\n=== Loading OSM streets + buildings ===")
transformer = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)

print(f"  Downloading OSM street network for {CITY} ...")
G     = ox.graph_from_place(CITY, network_type="drive", simplify=True)
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
print(f"  Network loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

street_geoms_latlon = {}   # street_name → unified geometry in EPSG:4326
street_geoms_proj   = {}   # street_name → unified geometry in EPSG:25833

for name in STREET_NAMES:
    seg = edges[edges["name"] == name]
    if seg.empty:
        raise ValueError(f"Street not found in OSM: '{name}'")
    geom_4326  = unary_union(seg.geometry)
    geom_25833 = (
        gpd.GeoDataFrame(geometry=[geom_4326], crs=edges.crs)
        .to_crs(epsg=25833)
        .geometry.iloc[0]
    )
    street_geoms_latlon[name] = geom_4326
    street_geoms_proj[name]   = geom_25833
    print(f"  loaded street: {name}")

print("  Computing block polygon from street intersections ...")
# --- Block polygon from pairwise intersections of the 4 street lines ---
street_list = list(street_geoms_proj.values())
corner_pts  = []
for i in range(len(street_list)):
    for j in range(i + 1, len(street_list)):
        inter = street_list[i].intersection(street_list[j])
        if inter.is_empty:
            continue
        if inter.geom_type == "Point":
            corner_pts.append(inter)
        elif inter.geom_type == "MultiPoint":
            corner_pts.extend(inter.geoms)

if len(corner_pts) < 3:
    raise ValueError(
        f"Only {len(corner_pts)} intersection point(s) found between the 4 streets. "
        "Check that the street names form a closed block."
    )

print("\nDIAG: individual corner points (EPSG:25833):")
for k, cp in enumerate(corner_pts):
    print(f"  corner[{k}] = ({cp.x:.1f}, {cp.y:.1f})")

print("\nDIAG: street geometry extents (EPSG:25833):")
for name, geom in street_geoms_proj.items():
    gb = geom.bounds
    glen = geom.length
    print(f"  {name:20s}  bounds=({gb[0]:.1f},{gb[1]:.1f})-({gb[2]:.1f},{gb[3]:.1f})  length={glen:.1f}m  geom_type={geom.geom_type}")

block_polygon = Polygon([(p.x, p.y) for p in corner_pts]).convex_hull
centroid_proj = block_polygon.centroid
print(f"\nBlock centroid (EPSG:25833): {centroid_proj.x:.1f}, {centroid_proj.y:.1f}")
print(f"Corner points found: {len(corner_pts)}")
_b = block_polygon.bounds
print(f"Block polygon bounds:  minx={_b[0]:.1f}  miny={_b[1]:.1f}  maxx={_b[2]:.1f}  maxy={_b[3]:.1f}")
print(f"Block polygon area:    {block_polygon.area:.0f} m²  (expected ~10 000–80 000 for a Viennese block)")

# --- Buildings within the block polygon ---
# features_from_polygon queries only the area of the polygon — much
# faster than features_from_place which downloads the entire city.
print("  Querying OSM buildings inside block polygon ...")
block_polygon_wgs = (
    gpd.GeoDataFrame(geometry=[block_polygon], crs="EPSG:25833")
    .to_crs(epsg=4326)
    .geometry.iloc[0]
)
buildings_all_raw = ox.features_from_polygon(
    block_polygon_wgs, tags={"building": True}
).to_crs(epsg=25833)
buildings_all_raw["geometry"] = buildings_all_raw.geometry.simplify(1.5)

# features_from_polygon returns all buildings that touch the bounding box,
# including buildings on the OPPOSITE side of the street whose footprint
# merely grazes the query polygon boundary.  Filter to only buildings whose
# centroid lies strictly inside the block polygon so that raycasts from
# street-side panos cannot accidentally hit an across-the-street building.
buildings_all = buildings_all_raw[
    buildings_all_raw.geometry.centroid.within(block_polygon)
].copy()
print(f"  Buildings raw (bbox): {len(buildings_all_raw)}  ->  inside block: {len(buildings_all)}")

# ── Build block address set via Google Geocoding API ──────────────────────────
# OSM buildings inside our block_polygon have no addr:housenumber tags, so
# we geocode the centroid of each building in buildings_all (which are already
# guaranteed to lie inside block_polygon) to obtain their real-world addresses.
# No new probe points are generated — we reuse the geometry that is already
# available from the buildings_all GeoDataFrame (the same centroids that were
# used to filter buildings into the block in the first place).
# The resulting BLOCK_ADDRESSES frozenset is used by _is_known_block_address
# inside resolve_facade_column to decide which tile-column side is correct.

_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
# 25833->4326 transformer needed here (before _proj_to_wgs is defined later).
_proj_to_wgs_init = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)

def _geocode_point(x_25833, y_25833, api_key):
    """Reverse-geocode a single EPSG:25833 point. Returns (housenumber, street)
    normalised for comparison, or (None, None) on failure."""
    lng, lat = _proj_to_wgs_init.transform(x_25833, y_25833)
    try:
        r = _get(
            _GEOCODE_URL,
            params={"latlng": f"{lat},{lng}", "key": api_key,
                    "result_type": "street_address"},
            timeout=10
        )
        if r.status_code != 200:
            return None, None
        data = r.json()
    except Exception:
        return None, None

    if data.get("status") != "OK" or not data.get("results"):
        return None, None

    hn, st = None, None
    for comp in data["results"][0].get("address_components", []):
        types = comp.get("types", [])
        if "street_number" in types:
            hn = comp.get("long_name", "").strip()
        if "route" in types:
            st = comp.get("long_name", "").strip()
    return hn, st

print("  Building block address set from dense grid via Geocoding API ...")
# Sample a dense regular grid of points inside block_polygon (step ~6m).
# We inset the polygon by GRID_INSET_M before sampling so that all probe
# points are well away from the street boundary — without this, points near
# the polygon edge can be geocoded to the building on the OPPOSITE side of
# the street (Google snaps to the nearest address, which may be across the
# road). An inset of ~8m keeps every probe point safely inside the block.
_GRID_STEP_M  = 6.0
_GRID_INSET_M = 8.0
_sample_polygon = block_polygon.buffer(-_GRID_INSET_M)
if _sample_polygon.is_empty:
    print("  WARNING: inset polygon is empty — falling back to original polygon")
    _sample_polygon = block_polygon
_minx, _miny, _maxx, _maxy = _sample_polygon.bounds
_block_addr_raw = {}  # (hn_norm, st_norm) -> (hn_display, st_display)
_n_probed = 0
_xs = list(np.arange(_minx, _maxx, _GRID_STEP_M))
_ys = list(np.arange(_miny, _maxy, _GRID_STEP_M))
for _gx in _xs:
    for _gy in _ys:
        _gpt = Point(_gx, _gy)
        if not _sample_polygon.contains(_gpt):
            continue
        _n_probed += 1
        _hn, _st = _geocode_point(_gx, _gy, API_KEY)
        if _hn and _st:
            _st_norm = _st.lower().replace("-", "").replace(" ", "")
            _key = (_hn, _st_norm)
            if _key not in _block_addr_raw:
                _block_addr_raw[_key] = (_hn, _st)

print(f"  Probed {_n_probed} grid points -> {len(_block_addr_raw)} unique addresses:")
for (_hn_n, _st_n), (_hn_d, _st_d) in sorted(_block_addr_raw.items()):
    print(f"    {_hn_d} {_st_d}")

# BLOCK_ADDRESSES: frozenset of (hn_norm, st_norm) for fast membership test
# BLOCK_ADDRESS_LABELS: dict (hn_norm, st_norm) -> "hn st" for display/labels
BLOCK_ADDRESSES     = frozenset(_block_addr_raw.keys())
BLOCK_ADDRESS_LABELS = {k: f"{v[0]} {v[1]}" for k, v in _block_addr_raw.items()}
print(f"  Block address set ready: {len(BLOCK_ADDRESSES)} entries")

# ======================================================
# FACADE LENGTH ANALYSIS
# ======================================================
#
# Algorithm (per building, per non-party edge):
#
#   STEP 1 — Party walls (Brandwände):
#             Edge overlaps a neighbour by ≥ 20% of its length.
#
#   STEP 2 — Outer vs Inner:
#             For each non-party edge:
#             a) Find the nearest side of the block polygon to the
#                edge midpoint.
#             b) Compute the edge normal pointing toward that block side.
#             c) Cast a ray from the midpoint along that normal toward
#                the block side.
#             d) If the ray intersects ANY building edge in the block
#                before reaching the block side → INNER facade.
#                Otherwise → OUTER facade.
#
# No pano data needed — runs before the per-street loop.
#
# ======================================================

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SHARED_EDGE_SNAP_M = 0.05   # tight tolerance for shared-wall detection
EDGE_MIN_LENGTH_M  = 0.5    # ignore slivers

def edge_segments_indexed(polygon):
    coords = list(polygon.exterior.coords)
    segs = []
    for i, (a, b) in enumerate(zip(coords[:-1], coords[1:])):
        seg = LineString([a, b])
        if seg.length >= EDGE_MIN_LENGTH_M:
            segs.append((i, seg, seg.length))
    return segs

def _block_sides(block_polygon):
    """Return the 4 sides of the block polygon as LineStrings."""
    coords = list(block_polygon.exterior.coords)
    return [LineString([coords[i], coords[i+1]]) for i in range(len(coords)-1)]

def _nearest_block_side(midpt, block_sides):
    """Return the block side LineString closest to midpt."""
    return min(block_sides, key=lambda s: midpt.distance(s))

def _edge_normal_toward(seg, target_geom):
    """
    Return the unit normal of seg (as bearing in degrees) that points
    toward target_geom. The two candidates are +90° and -90° from the
    edge bearing.
    """
    dx = seg.coords[-1][0] - seg.coords[0][0]
    dy = seg.coords[-1][1] - seg.coords[0][1]
    edge_bear = (math.degrees(math.atan2(dx, dy)) + 360) % 360
    n1 = (edge_bear + 90) % 360
    n2 = (edge_bear - 90) % 360

    midpt = seg.interpolate(0.5, normalized=True)
    cx1 = midpt.x + math.sin(math.radians(n1))
    cy1 = midpt.y + math.cos(math.radians(n1))
    cx2 = midpt.x + math.sin(math.radians(n2))
    cy2 = midpt.y + math.cos(math.radians(n2))

    d1 = Point(cx1, cy1).distance(target_geom)
    d2 = Point(cx2, cy2).distance(target_geom)
    return n1 if d1 <= d2 else n2

def run_facade_analysis(block_buildings, block_polygon,
                        centroid_proj, BASE_OUT_DIR, STREET_NAMES):
    """
    Classify every building edge as outer, inner, or party wall.
    Returns (building_facades, vis_segments).
    """
    print("\n=== Facade length analysis ===")

    block_sides = _block_sides(block_polygon)

    # Pre-compute other_union per building for party wall detection
    other_unions = {}
    for bid, row in block_buildings.iterrows():
        others = block_buildings.loc[block_buildings.index != bid, "geometry"]
        other_unions[bid] = unary_union(others.values) if len(others) else None

    # Union of ALL edges of ALL buildings — used for ray blocking check
    all_edges = []
    for _, brow in block_buildings.iterrows():
        if brow.geometry is None or brow.geometry.is_empty:
            continue
        coords = list(brow.geometry.exterior.coords)
        for a, b in zip(coords[:-1], coords[1:]):
            seg = LineString([a, b])
            if seg.length >= EDGE_MIN_LENGTH_M:
                all_edges.append(seg)
    all_edges_union = unary_union(all_edges)

    building_facades = []
    vis_segments     = []
    n_bld = len(block_buildings)

    for b_idx, (bid, row) in enumerate(block_buildings.iterrows(), 1):
        print(f"  Analysing building {b_idx}/{n_bld}: {bid}")
        geom = row.geometry
        if geom is None or geom.is_empty:
            print("    WARNING: empty geometry — skipping")
            continue

        segs   = edge_segments_indexed(geom)
        n_segs = len(segs)
        if not n_segs:
            continue

        other_union = other_unions[bid]

        # ── STEP 1: party walls — with partial-overlap split ────────────
        # A segment that FULLY overlaps a neighbour building edge → party.
        # A segment that PARTIALLY overlaps → split at the overlap boundary:
        #   - the overlapping sub-segment → party
        #   - the non-overlapping sub-segment(s) → passed to Step 2
        # This handles fire walls ("Brandwände") that extend beyond the
        # neighbour building and become inner/outer walls at the overhang.

        split_segs  = []   # list of (sub_seg, is_party_flag)

        for idx, seg, length in segs:
            if other_union is None:
                split_segs.append((seg, False))
                continue

            buffered_other = other_union.buffer(SHARED_EDGE_SNAP_M)
            overlap_geom   = seg.intersection(buffered_other)
            non_overlap_geom = seg.difference(buffered_other)

            overlap_len     = overlap_geom.length if not overlap_geom.is_empty else 0.0
            non_overlap_len = non_overlap_geom.length if not non_overlap_geom.is_empty else 0.0

            if overlap_len < length * 0.1:
                # Negligible overlap — treat entire segment as non-party
                split_segs.append((seg, False))
            elif non_overlap_len < length * 0.1:
                # Almost entirely overlapping — treat as full party wall
                split_segs.append((seg, True))
            else:
                # Genuine partial overlap — split into sub-segments
                def _extract_linestrings(geom):
                    """Flatten any geometry into a list of LineStrings."""
                    if geom.is_empty:
                        return []
                    if geom.geom_type == "LineString":
                        return [geom]
                    if geom.geom_type in ("MultiLineString", "GeometryCollection"):
                        result = []
                        for g in geom.geoms:
                            result.extend(_extract_linestrings(g))
                        return result
                    return []

                for sub in _extract_linestrings(overlap_geom):
                    if sub.length >= EDGE_MIN_LENGTH_M * 0.5:
                        split_segs.append((sub, True))   # party sub-segment
                for sub in _extract_linestrings(non_overlap_geom):
                    if sub.length >= EDGE_MIN_LENGTH_M * 0.5:
                        split_segs.append((sub, False))  # non-party sub-segment

        # ── STEP 2: outer vs inner via block-side ray ────────────────────
        # Ray is checked against all_edges_union (all buildings in block).
        # Own edge is excluded via a small buffer around the segment itself.
        is_outer = []
        is_inner = []

        outer_m = 0.0
        inner_m = 0.0

        for seg, is_party_seg in split_segs:
            length = seg.length
            if is_party_seg:
                is_outer.append(False)
                is_inner.append(False)
                vis_segments.append((seg, "party"))
                continue

            midpt = seg.interpolate(0.5, normalized=True)

            # Normal toward centroid (used to derive the outward direction)
            normal_toward_centroid = _edge_normal_toward(seg, centroid_proj)

            # Shoot a ray from the midpoint in the outward normal direction
            # (away from the block centroid, toward the street side).
            #
            # outer: ray reaches the block boundary (street) unobstructed.
            # inner: ray is blocked by another building wall before reaching
            #        the street — this includes:
            #          - courtyard walls facing the Innenhof
            #          - walls orthogonal to the street: their outward normal
            #            runs parallel to the street and will hit the corner
            #            building's outer wall (or a party wall) before
            #            reaching the block boundary, correctly classifying
            #            them as inner.

            normal_out = (normal_toward_centroid + 180) % 360
            rad_out    = math.radians(normal_out)

            block_boundary = LineString(list(block_polygon.exterior.coords))
            self_exclusion = seg.buffer(0.1)
            ray_length     = block_polygon.exterior.length

            far = Point(midpt.x + math.sin(rad_out) * ray_length,
                        midpt.y + math.cos(rad_out) * ray_length)
            ray_t = LineString([midpt, far]).difference(self_exclusion)

            d_street = midpt.distance(ray_t.intersection(block_boundary)) \
                       if not ray_t.intersection(block_boundary).is_empty \
                       else float("inf")
            d_wall   = midpt.distance(ray_t.intersection(all_edges_union)) \
                       if not ray_t.intersection(all_edges_union).is_empty \
                       else float("inf")

            if d_street <= d_wall:
                is_outer.append(True)
                is_inner.append(False)
                outer_m += length
                vis_segments.append((seg, "outer"))
            else:
                is_inner.append(True)
                is_outer.append(False)
                inner_m += length
                vis_segments.append((seg, "inner"))

        ratio = (inner_m / outer_m) if outer_m > 0 else None
        n_party = sum(1 for _, flag in split_segs if flag)
        print(
            f"    outer={outer_m:.1f} m  inner={inner_m:.1f} m  "
            f"party={n_party} segments  "
            + (f"ratio={ratio:.3f}" if ratio else "ratio=n/a")
        )

        # Look up address via geocoding (OSM addr tags are absent for this block).
        # Geocode the building centroid, then normalise and check against
        # BLOCK_ADDRESS_LABELS which was built from the dense grid earlier.
        _b_cx = row.geometry.centroid.x
        _b_cy = row.geometry.centroid.y
        _b_hn, _b_st = _geocode_point(_b_cx, _b_cy, API_KEY)
        if _b_hn and _b_st:
            _b_st_norm = _b_st.lower().replace("-", "").replace(" ", "")
            _b_label = BLOCK_ADDRESS_LABELS.get((_b_hn, _b_st_norm))
        else:
            _b_hn, _b_st, _b_label = None, None, None

        building_facades.append({
            "building_id":    str(bid),
            "facade_outer_m": round(outer_m, 2),
            "facade_inner_m": round(inner_m, 2),
            "facade_ratio":   round(ratio, 3) if ratio is not None else None,
            "housenumber":    _b_hn,
            "street":         _b_st,
        })

    print(f"  Facade analysis complete: {len(building_facades)} buildings")

    # ── visualisation ───────────────────────────────────────────────────
    COLOR = {"outer":   "#2ecc71",   # green  — street-facing facade
             "inner":   "#e74c3c",   # red    — courtyard-facing facade
             "party":   "#e67e22",   # orange — shared fire wall
             "unknown": "#dddddd"}   # grey   — unclassified
    WIDTH = {"outer": 3.0, "inner": 3.0, "party": 2.0, "unknown": 0.8}
    STYLE = {"outer": "-",  "inner": "--", "party": ":",  "unknown": "-."}

    fig, ax = plt.subplots(figsize=(12, 12))

    bx, by = block_polygon.exterior.xy
    ax.fill(bx, by, alpha=0.08, color="steelblue", zorder=0)
    ax.plot(bx, by, color="steelblue", linewidth=1.2, linestyle="--", zorder=1)

    for _, row in block_buildings.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        gx, gy = row.geometry.exterior.xy
        ax.fill(gx, gy, alpha=0.15, color="tan", zorder=1)

    for seg, label in vis_segments:
        xs, ys = seg.xy
        ax.plot(xs, ys, color=COLOR[label], linewidth=WIDTH[label],
                linestyle=STYLE[label], zorder=3)

    ax.plot(centroid_proj.x, centroid_proj.y, "b+", markersize=14,
            markeredgewidth=2, zorder=5)

    legend_elements = [
        Line2D([0],[0], color=COLOR["outer"], linewidth=3.0, linestyle="-",
               label="Außenwand (straßenseitig)"),
        Line2D([0],[0], color=COLOR["inner"], linewidth=3.0, linestyle="--",
               label="Innenwand (hofseitig)"),
        Line2D([0],[0], color=COLOR["party"], linewidth=2.0, linestyle=":",
               label="Brandwand (Nachbargebäude)"),
        Line2D([0],[0], color=COLOR["unknown"], linewidth=0.8, linestyle="-.",
               label="Nicht klassifiziert"),
        Line2D([0],[0], color="steelblue", linewidth=1.2,
               linestyle="--", label="Block-Polygon"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
    ax.set_aspect("equal")
    ax.set_title("Fassadenklassifikation — " + ", ".join(STREET_NAMES), fontsize=12)
    ax.set_xlabel("Easting (m, EPSG:25833)")
    ax.set_ylabel("Northing (m, EPSG:25833)")

    # Save a preliminary version (without pano overlays)
    os.makedirs(BASE_OUT_DIR, exist_ok=True)
    vis_path = os.path.join(BASE_OUT_DIR, "facade_classification.png")
    fig.savefig(vis_path, dpi=150, bbox_inches="tight")
    print(f"  Facade base visualisation saved → {vis_path}")
    # Return fig and ax so pano overlays can be added later
    return building_facades, vis_segments, fig, ax, vis_path

# ======================================================
# PANO OVERLAY VISUALISATION
# ======================================================

def add_pano_overlays(fig, ax, vis_path, street_exports):
    """
    Overlay pano positions and fan rays onto the facade classification figure.

    Each pano has two rays stored as absolute geographic bearings:
      ray_toward_start  → toward street position 0
      ray_toward_end    → toward street position N

    These are drawn directly — no sign arithmetic, no prev_sign/next_sign.
    Snapped edge rays are drawn in magenta with a dotted line to the
    snapped building corner.
    """
    STREET_COLORS = ["#e67e22", "#8e44ad", "#2980b9", "#16a085"]
    SNAP_COLOR    = "#e91e63"
    FAN_ALPHA     = 0.35
    FAN_ALPHA_SNAP = 0.85
    FAN_LW        = 1.2
    FAN_LW_SNAP   = 2.0
    DOT_SIZE      = 6
    LABEL_SIZE    = 5

    snap_legend_added = False

    for s_idx, (street_name, (export, _)) in enumerate(street_exports.items()):
        color = STREET_COLORS[s_idx % len(STREET_COLORS)]
        panos = export["panos"]

        for p in panos:
            if p["skip"]:
                continue

            px     = p["x_25833"]
            py     = p["y_25833"]
            normal = p["inner_normal"]
            dist   = p["dist_to_facade"]
            r_start = p["ray_toward_start"]
            r_end   = p["ray_toward_end"]

            # Pano dot — larger + numbered for start/end edges
            if p.get("is_start_edge") or p.get("is_end_edge"):
                ax.plot(px, py, "D", color=color, markersize=DOT_SIZE + 2,
                        zorder=6, markeredgewidth=0.8, markeredgecolor="white")
            else:
                ax.plot(px, py, "o", color=color, markersize=DOT_SIZE,
                        zorder=6, markeredgewidth=0.5, markeredgecolor="white")

            # Draw both fan rays
            for ray_bear, is_snapped, snap_corner_xy in (
                (r_start, p.get("start_snapped", False), p.get("snap_start_corner")),
                (r_end,   p.get("end_snapped",   False), p.get("snap_end_corner")),
            ):
                if ray_bear is None:
                    continue
                rad = math.radians(ray_bear)
                ex  = px + math.sin(rad) * dist
                ey  = py + math.cos(rad) * dist

                if is_snapped:
                    ax.plot([px, ex], [py, ey],
                            color=SNAP_COLOR, linewidth=FAN_LW_SNAP,
                            alpha=FAN_ALPHA_SNAP, zorder=8,
                            label="Snapped Randwinkel" if not snap_legend_added else "_nolegend_")
                    snap_legend_added = True
                    if snap_corner_xy:
                        cx, cy = snap_corner_xy
                        ax.plot([px, cx], [py, cy],
                                color=SNAP_COLOR, linewidth=1.0,
                                linestyle=":", alpha=0.7, zorder=7)
                        ax.plot(cx, cy, "*", color=SNAP_COLOR,
                                markersize=9, zorder=9,
                                markeredgewidth=0.5, markeredgecolor="white")
                else:
                    ax.plot([px, ex], [py, ey], color=color,
                            linewidth=FAN_LW, alpha=FAN_ALPHA, zorder=5)

            # Centre ray along inner_normal
            rad = math.radians(normal)
            ex  = px + math.sin(rad) * dist
            ey  = py + math.cos(rad) * dist
            ax.plot([px, ex], [py, ey], color=color,
                    linewidth=FAN_LW * 0.7, alpha=FAN_ALPHA * 0.7,
                    linestyle="--", zorder=5)

            # Position label — show pos index + short panoId
            lbl = f"{p.get('pos','?')}:{p['panoId'][:5]}"
            ax.annotate(lbl, xy=(px, py), xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=LABEL_SIZE, color=color, zorder=7, clip_on=True)

        ax.plot([], [], "o-", color=color, linewidth=FAN_LW,
                label=street_name, markersize=DOT_SIZE)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=8)
    fig.savefig(vis_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pano overlay visualisation saved → {vis_path}")

# ======================================================
# PER-STREET PROCESSING
# ======================================================

# ======================================================
# RUN FACADE ANALYSIS (before downloading — no pano data needed)
# ======================================================
block_buildings  = buildings_all.copy()
building_facades, _, facade_fig, facade_ax, facade_vis_path = run_facade_analysis(
    block_buildings, block_polygon,
    centroid_proj, BASE_OUT_DIR, STREET_NAMES
)

# Precompute the outline of the entire Bebauungsblock (union of all buildings).
# Used by corner_angle_for_edge_pano to snap boundary-pano angles to real
# building corners rather than individual building corners.
building_block_outline = unary_union(
    [r.geometry for _, r in block_buildings.iterrows()
     if r.geometry is not None and not r.geometry.is_empty]
)
print(f"  Building block outline computed: {building_block_outline.geom_type}")

# Global set of already-seen pano IDs — avoids re-downloading corner panos
global_seen_panos = set()

street_exports = {}   # street_name → (export_dict, collect_dir)

for street_name in STREET_NAMES:

    print(f"\n{'='*60}")
    print(f"  STREET: {street_name}")
    print(f"{'='*60}")

    street_slug = street_name.replace(" ", "_").replace("/", "-")
    collect_dir = os.path.join(BASE_OUT_DIR, street_slug)
    os.makedirs(collect_dir, exist_ok=True)

    # Clip street geometry to the block polygon before scanning.
    # A small buffer avoids losing points exactly on the boundary.
    CLIP_BUFFER_M = 5.0
    street_proj_clipped = street_geoms_proj[street_name].intersection(
        block_polygon.buffer(CLIP_BUFFER_M)
    )

    if street_proj_clipped.is_empty:
        print(f"  WARNING: clipped geometry is empty for {street_name} — skipping")
        continue

    street_latlon = flatten_geometry(
        gpd.GeoDataFrame(geometry=[street_proj_clipped], crs="EPSG:25833")
        .to_crs(4326)
        .geometry.iloc[0]
    )

    # --------------------------------------------------
    # PHASE 0 — DISCOVER PANOS (block segment only)
    # --------------------------------------------------
    print(f"\n=== Phase 0: Discover pano IDs [{street_name}] ===")
    print(f"  Street clipped to block: {street_proj_clipped.length:.1f} m")

    scan_points = []
    for a, b in zip(street_latlon[:-1], street_latlon[1:]):
        dist = geodesic(a, b).meters
        n    = max(1, int(dist / SCAN_DISTANCE_M))
        for i in range(n):
            t = i / n
            scan_points.append((
                a[0] + t * (b[0] - a[0]),
                a[1] + t * (b[1] - a[1])
            ))

    panos       = []
    street_seen = set()

    for lat, lon in scan_points:
        r = _get(
            META_URL,
            params={"lat": lat, "lng": lon, "session": session_id, "key": API_KEY}
        )
        if r.status_code != 200:
            continue
        meta = r.json()
        pid  = meta.get("panoId")
        if not pid or pid in street_seen or pid in global_seen_panos:
            continue
        street_seen.add(pid)
        panos.append({"panoId": pid, "lat": meta["lat"], "lon": meta["lng"]})
        print(f"  discovered pano: {pid}")

    global_seen_panos.update(street_seen)
    print(f"  Scan points queried: {len(scan_points)}")
    print(f"  Total unique panos discovered: {len(panos)}")

    # --------------------------------------------------
    # PHASE 1 — GEOMETRY (bearing + inner normal + width)
    # --------------------------------------------------
    print(f"\n=== Phase 1: Geometry [{street_name}] ===")

    # Sort panos by their geographic position projected onto the street axis.
    # OSM may return segments in any direction; sorting ensures pano_data[0]
    # is always the geographically first pano and pano_data[-1] the last,
    # regardless of OSM edge orientation.
    if len(panos) >= 2:
        # Use the raw bearing from first to last discovered pano as a
        # provisional street axis for sorting.  After sorting we recompute
        # bearings from neighbours so any residual error is eliminated.
        ref_bear = bearing(
            (panos[0]["lat"], panos[0]["lon"]),
            (panos[-1]["lat"], panos[-1]["lon"])
        )
        ref_rad  = math.radians(ref_bear)
        ref_vec  = (math.sin(ref_rad), math.cos(ref_rad))
        # Project each pano onto the street axis (in projected coords)
        def _proj_along(p_entry):
            x, y = transformer.transform(p_entry["lon"], p_entry["lat"])
            return x * ref_vec[0] + y * ref_vec[1]
        panos = sorted(panos, key=_proj_along)
        print(f"  Panos sorted along street axis (bearing {ref_bear:.1f}°): "
              f"{[p['panoId'][:6] for p in panos]}")

    pano_data       = []
    last_valid_dist = None

    for i in range(len(panos)):
        p = panos[i]

        # Street bearing: use prev→next when available, else nearest endpoint
        if i == 0:
            if len(panos) > 1:
                street_bear = bearing((p["lat"], p["lon"]),
                                      (panos[i + 1]["lat"], panos[i + 1]["lon"]))
            else:
                street_bear = 0.0
        elif i == len(panos) - 1:
            street_bear = bearing((panos[i - 1]["lat"], panos[i - 1]["lon"]),
                                  (p["lat"], p["lon"]))
        else:
            street_bear = bearing((panos[i - 1]["lat"], panos[i - 1]["lon"]),
                                  (panos[i + 1]["lat"], panos[i + 1]["lon"]))

        x, y    = transformer.transform(p["lon"], p["lat"])
        pano_pt = Point(x, y)

        # Determine which side of the street faces the block interior.
        in_normal       = inner_normal(street_bear, pano_pt, block_polygon)
        dist_to_facade  = compute_dist_to_facade(pano_pt, in_normal, buildings_all, last_valid_dist)
        last_valid_dist = dist_to_facade

        # ── Maximum facade distance filter ───────────────────────────────
        # Panos on the far lane of a wide road have a larger dist_to_facade
        # than those on the near lane. Excluding panos beyond the threshold
        # removes far-lane rows without affecting near-lane panos.
        if dist_to_facade > MAX_FACADE_DIST_M:
            print(
                f"  [{i+1}/{len(panos)}] pano {p['panoId'][:8]}  "
                f"SKIPPED — dist_to_facade={dist_to_facade:.1f} m "
                f"(> MAX_FACADE_DIST_M={MAX_FACADE_DIST_M} m)"
            )
            continue

        print(
            f"  [{i+1}/{len(panos)}] pano {p['panoId']}  "
            f"bearing={street_bear:.1f}°  inner_normal={in_normal:.1f}°  "
            f"dist_to_facade={dist_to_facade:.1f} m"
        )

        pano_data.append({
            "idx":            i,
            "pos":            i,           # position index along sorted street (0 = start)
            "panoId":         p["panoId"],
            "lat":            p["lat"],
            "lon":            p["lon"],
            "street_bearing": street_bear,
            "inner_normal":   in_normal,
            "pano_pt":        pano_pt,
            "dist_to_facade": dist_to_facade,
            # Absolute geographic bearings of the two fan rays.
            # ray_toward_start points toward pos=0 (start of street).
            # ray_toward_end   points toward pos=N (end of street).
            # These are never sign-ambiguous — they are direct bearings.
            "ray_toward_start": None,
            "ray_toward_end":   None,
            "is_start_edge":  False,       # True only for pos==0
            "is_end_edge":    False,       # True only for pos==N-1
            "start_snapped":  False,       # True when start ray was corner-snapped
            "end_snapped":    False,
            "snap_start_corner": None,     # Point(x,y) EPSG:25833
            "snap_end_corner":   None,
            "skip":           False,
        })

    # --------------------------------------------------
    # PHASE 2a — HEALTH CHECK
    # --------------------------------------------------
    print(f"\n=== Phase 2a: Health check [{street_name}] ===")
    print(f"  Checking {len(pano_data)} panos ...")

    n_panos = len(pano_data)
    for p_idx, p in enumerate(pano_data, 1):
        print(f"\n  [{p_idx}/{n_panos}] Health check pano {p['panoId']}")

        meta = _get(
            META_URL,
            params={"panoId": p["panoId"], "session": session_id, "key": API_KEY}
        ).json()

        img_w, img_h = meta["imageWidth"], meta["imageHeight"]
        tiles_x      = math.ceil(img_w / tile_width)
        tiles_y      = math.ceil(img_h / tile_height)
        zoom         = int(max(math.log2(tiles_x), math.log2(tiles_y)))
        deg_per_tile = 360 / tiles_x

        # --------------------------------------------------
        # Geometric check: does inner_normal point at a building?
        # If the raycast hits nothing, the pano is looking into
        # empty space (e.g. it is on the wrong side of the street
        # or at a corner where the clipping left a stray pano).
        # --------------------------------------------------
        pano_pt_check = p["pano_pt"]
        rad_check     = math.radians(p["inner_normal"])
        ray_check     = LineString([
            pano_pt_check,
            Point(pano_pt_check.x + math.sin(rad_check) * RAY_LENGTH_M,
                  pano_pt_check.y + math.cos(rad_check) * RAY_LENGTH_M)
        ])
        hits_building = any(
            not buildings_all.iloc[k].geometry.intersection(ray_check).is_empty
            for k in range(len(buildings_all))
        )
        if not hits_building:
            # Determine which end of the sorted pano list this pano is
            # closer to. If it is in the start half, all panos from the
            # beginning up to and including this one are invalid (stray
            # panos before the block facade starts). If it is in the end
            # half, all panos from this one to the end are invalid.
            # This propagates the failed check outward toward the nearest
            # edge — reflecting the assumption that invalid panos cluster
            # at the extremes (corners, stray panos past the block edge).
            midpoint = (n_panos - 1) / 2.0
            idx0     = p_idx - 1   # 0-based index into pano_data
            if idx0 <= midpoint:
                print(f"  pano {p['panoId'][:8]} INVALID (wall check) — "
                      f"cascading skip to START: invalidating panos 1..{p_idx}")
                for _j in range(p_idx):        # 0-based: 0 .. p_idx-1
                    pano_data[_j]["skip"] = True
            else:
                print(f"  pano {p['panoId'][:8]} INVALID (wall check) — "
                      f"cascading skip to END: invalidating panos {p_idx}..{n_panos}")
                for _j in range(idx0, n_panos):   # 0-based: idx0 .. n_panos-1
                    pano_data[_j]["skip"] = True
            time.sleep(2.0)
            continue

        # --------------------------------------------------
        # Incidence angle check: is the normal hitting the
        # facade at a steep enough angle?
        # A pano in an intersection often fires its normal at
        # a neighbouring facade at a grazing angle (<55°).
        # Such images are geometrically degenerate for facade
        # analysis and should be excluded.
        # Method: find the specific building edge that the ray
        # actually INTERSECTS (not just the nearest edge to the
        # hit point, which can be a parallel wall nearby), then
        # compute the angle between the ray direction and that edge.
        # --------------------------------------------------
        MIN_INCIDENCE_DEG = 55.0

        _best_angle = None
        _best_dist  = float("inf")

        for _k in range(len(buildings_all)):
            _bgeom = buildings_all.iloc[_k].geometry
            if _bgeom.intersection(ray_check).is_empty:
                continue

            # Iterate over exterior edges and find which one(s)
            # the ray actually intersects — use that edge for the angle.
            _rings = []
            if _bgeom.geom_type == "Polygon":
                _rings = [_bgeom.exterior]
            elif _bgeom.geom_type == "MultiPolygon":
                for _part in _bgeom.geoms:
                    _rings.append(_part.exterior)

            for _ring in _rings:
                _coords = list(_ring.coords)
                for _a, _b in zip(_coords[:-1], _coords[1:]):
                    _eseg = LineString([_a, _b])
                    _hit  = _eseg.intersection(ray_check)
                    if _hit.is_empty:
                        continue
                    # This edge is actually intersected by the ray.
                    # Distance from pano to intersection point.
                    if _hit.geom_type == "Point":
                        _hit_pt = _hit
                    else:
                        _hit_pt = _hit.centroid
                    _d = _hit_pt.distance(pano_pt_check)
                    if _d >= _best_dist:
                        continue
                    _best_dist = _d
                    # Angle between ray direction and this edge.
                    _ex = _b[0] - _a[0]
                    _ey = _b[1] - _a[1]
                    _rx = math.sin(rad_check)
                    _ry = math.cos(rad_check)
                    _cross = abs(_rx * _ey - _ry * _ex)
                    _denom = math.hypot(_rx, _ry) * math.hypot(_ex, _ey)
                    if _denom > 0:
                        _best_angle = math.degrees(
                            math.asin(min(1.0, _cross / _denom))
                        )

        if _best_angle is not None and _best_angle < MIN_INCIDENCE_DEG:
            midpoint = (n_panos - 1) / 2.0
            idx0     = p_idx - 1
            print(f"  pano {p['panoId'][:8]} INVALID (incidence {_best_angle:.1f}° < "
                  f"{MIN_INCIDENCE_DEG}°) — pano likely in intersection, "
                  f"normal grazes neighbouring facade")
            if idx0 <= midpoint:
                print(f"    cascading skip to START: invalidating panos 1..{p_idx}")
                for _j in range(p_idx):
                    pano_data[_j]["skip"] = True
            else:
                print(f"    cascading skip to END: invalidating panos {p_idx}..{n_panos}")
                for _j in range(idx0, n_panos):
                    pano_data[_j]["skip"] = True
            time.sleep(2.0)
            continue

        print(f"  incidence angle: {_best_angle:.1f}°  ✓"
              if _best_angle is not None else "  incidence angle: n/a")

        test_col = int(p["inner_normal"] / deg_per_tile) % tiles_x

        test_rows = [
            int(tiles_y * 0.15),
            int(tiles_y * 0.50),
            int(tiles_y * 0.85),
        ]

        failures = 0

        for row in test_rows:
            success = False

            for attempt in range(3):
                r = _get(
                    f"{TILE_URL}/{zoom}/{test_col}/{row}",
                    params={
                        "session": session_id,
                        "panoId":  p["panoId"],
                        "zoom":    zoom,
                        "key":     API_KEY,
                    }
                )

                if r.status_code == 200:
                    tile = cv2.imdecode(
                        np.frombuffer(r.content, np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    if tile is not None and tile.size > 0:
                        print(f"    test tile OK z={zoom} x={test_col} y={row}")
                        success = True
                        break
                    else:
                        print(f"    INVALID PAYLOAD z={zoom} x={test_col} y={row}")
                        break

                elif r.status_code == 429:
                    print(f"    RATE LIMITED z={zoom} x={test_col} y={row} (attempt {attempt+1})")
                    time.sleep(0.6 * (attempt + 1))

                else:
                    print(f"    test tile FAILED z={zoom} x={test_col} y={row} (HTTP {r.status_code})")
                    break

            if not success:
                failures += 1
                print(f"    test tile FAILED permanently at y={row}")

            time.sleep(0.2)

        print(f"  health summary: {failures}/3 failed")

        if failures >= 2:
            print("  pano INVALID — marking as skip")
            p["skip"] = True
        else:
            print("  pano VALID")

        time.sleep(2.0)

    n_valid = sum(1 for p in pano_data if not p["skip"])
    n_skip  = len(pano_data) - n_valid
    print(f"\n  Phase 2a complete: {n_valid} valid, {n_skip} skipped")

    # --------------------------------------------------
    # PHASE 2b — RAY BEARING COMPUTATION (valid panos only)
    # --------------------------------------------------
    # Each pano gets two absolute geographic bearings:
    #   ray_toward_start : points toward the START of the street (pos=0)
    #   ray_toward_end   : points toward the END of the street (pos=N-1)
    #
    # For a middle pano:
    #   ray_toward_start = bearing(this_pano → prev_valid_pano)
    #   ray_toward_end   = bearing(this_pano → next_valid_pano)
    #
    # For the start-edge pano (pos==0):
    #   ray_toward_end   = bearing(this → next)   [normal midpoint ray]
    #   ray_toward_start = bearing to snapped building corner in back direction
    #
    # For the end-edge pano (pos==N-1):
    #   ray_toward_start = bearing(this → prev)   [normal midpoint ray]
    #   ray_toward_end   = bearing to snapped building corner in fwd direction
    #
    # The visualisation and tile download use these bearings directly —
    # no sign arithmetic, no prev_sign/next_sign, no ambiguity.
    # --------------------------------------------------
    print(f"\n=== Phase 2b: Ray bearing computation [{street_name}] ===")

    valid_indices = [i for i, p in enumerate(pano_data) if not p["skip"]]
    n_valid       = len(valid_indices)
    print(f"  Computing rays for {n_valid} valid panos ...")

    # Mark start/end edges
    if n_valid > 0:
        pano_data[valid_indices[0]]["is_start_edge"]  = True
    if n_valid > 1:
        pano_data[valid_indices[-1]]["is_end_edge"]   = True

    # The "back" direction (toward street start) is the bearing from
    # first valid pano to last valid pano, reversed.
    if n_valid >= 2:
        p_first = pano_data[valid_indices[0]]
        p_last  = pano_data[valid_indices[-1]]
        street_fwd_bear = bearing(
            (p_first["lat"], p_first["lon"]),
            (p_last["lat"],  p_last["lon"])
        )
        street_back_bear = (street_fwd_bear + 180) % 360
    else:
        street_fwd_bear  = pano_data[valid_indices[0]]["street_bearing"]
        street_back_bear = (street_fwd_bear + 180) % 360

    print(f"  Street forward bearing: {street_fwd_bear:.1f}°  "
          f"back bearing: {street_back_bear:.1f}°")

    for pos, i in enumerate(valid_indices):
        p    = pano_data[i]
        dist = p["dist_to_facade"]

        prev_valid = pano_data[valid_indices[pos - 1]] if pos > 0 else None
        next_valid = pano_data[valid_indices[pos + 1]] if pos < n_valid - 1 else None

        is_start = p["is_start_edge"]
        is_end   = p["is_end_edge"]

        def _facade_ray(nbr, w_self):
            """
            Compute the absolute bearing of the fan ray toward the facade
            midpoint between this pano and a neighbour pano.

            The facade midpoint is the point on the facade plane that lies
            at the lateral midpoint between this pano and its neighbour.
            We project it explicitly in EPSG:25833 coordinates:

              1. Foot-of-normal from this pano to the facade:
                 foot_self = pano_pt + w_self * unit_vec(inner_normal)

              2. Foot-of-normal from neighbour to the facade
                 (using THIS pano's inner_normal so both feet are on the
                 same facade plane, avoiding cross-overs when inner_normal
                 differs between adjacent panos near a street bend):
                 foot_nbr  = nbr_pt + w_self * unit_vec(inner_normal)

              3. Facade midpoint:
                 mid = (foot_self + foot_nbr) / 2

              4. Bearing from this pano to mid.

            This avoids the sign/lateral-projection approach entirely, which
            was the source of cross-overs: when inner_normal differs between
            two adjacent panos (e.g. near a street corner), the old approach
            computed the angle in two different coordinate systems and
            produced rays that crossed each other at the facade.
            """
            from geopy.distance import geodesic as _geodesic
            d = _geodesic((p["lat"], p["lon"]),
                          (nbr["lat"], nbr["lon"])).meters
            if w_self <= 0 or d <= 0:
                return p["inner_normal"]

            # Unit vector toward facade (this pano's inner_normal)
            nx, ny = unit_vec(p["inner_normal"])

            # Foot of normal from this pano on the facade plane
            foot_self_x = p["pano_pt"].x + nx * w_self
            foot_self_y = p["pano_pt"].y + ny * w_self

            # Foot of normal from neighbour on the SAME facade plane
            # (project neighbour along this pano's inner_normal, not its own)
            nbr_x, nbr_y = transformer.transform(nbr["lon"], nbr["lat"])
            foot_nbr_x   = nbr_x + nx * w_self
            foot_nbr_y   = nbr_y + ny * w_self

            # Midpoint on facade
            mid_x = (foot_self_x + foot_nbr_x) / 2.0
            mid_y = (foot_self_y + foot_nbr_y) / 2.0

            # Bearing from this pano to the facade midpoint
            dx = mid_x - p["pano_pt"].x
            dy = mid_y - p["pano_pt"].y
            bear = (math.degrees(math.atan2(dx, dy)) + 360) % 360

            # Safety: clamp to MAX_ANGLE_DEG from inner_normal
            diff = (bear - p["inner_normal"] + 180) % 360 - 180
            if abs(diff) > MAX_ANGLE_DEG:
                bear = (p["inner_normal"] + math.copysign(MAX_ANGLE_DEG, diff)) % 360

            return bear

        # ── ray_toward_start ────────────────────────────────────────
        if prev_valid:
            # Middle pano: ray to facade midpoint between this and prev
            p["ray_toward_start"] = _facade_ray(prev_valid, dist)
        elif is_start:
            # Start-edge pano: snap to building corner behind us
            snapped, snap_corner = corner_angle_for_edge_pano(
                p["pano_pt"], p["inner_normal"], dist,
                building_block_outline, street_back_bear
            )
            if snapped is not None and snap_corner is not None:
                dx = snap_corner.x - p["pano_pt"].x
                dy = snap_corner.y - p["pano_pt"].y
                corner_bear = (math.degrees(math.atan2(dx, dy)) + 360) % 360
                p["ray_toward_start"] = corner_bear
                p["start_snapped"]    = True
                p["snap_start_corner"] = snap_corner
                print(f"    ray_toward_start snapped: bearing={corner_bear:.1f}°  "
                      f"corner=({snap_corner.x:.1f}, {snap_corner.y:.1f})")
            else:
                p["ray_toward_start"] = None
        else:
            p["ray_toward_start"] = street_back_bear  # shouldn't happen

        # ── ray_toward_end ──────────────────────────────────────────
        if next_valid:
            # Middle pano: ray to facade midpoint between this and next
            p["ray_toward_end"] = _facade_ray(next_valid, dist)
        elif is_end:
            # End-edge pano: snap to building corner ahead of us
            snapped, snap_corner = corner_angle_for_edge_pano(
                p["pano_pt"], p["inner_normal"], dist,
                building_block_outline, street_fwd_bear
            )
            if snapped is not None and snap_corner is not None:
                dx = snap_corner.x - p["pano_pt"].x
                dy = snap_corner.y - p["pano_pt"].y
                corner_bear = (math.degrees(math.atan2(dx, dy)) + 360) % 360
                p["ray_toward_end"]   = corner_bear
                p["end_snapped"]      = True
                p["snap_end_corner"]  = snap_corner
                print(f"    ray_toward_end snapped: bearing={corner_bear:.1f}°  "
                      f"corner=({snap_corner.x:.1f}, {snap_corner.y:.1f})")
            else:
                p["ray_toward_end"] = None
        else:
            p["ray_toward_end"] = street_fwd_bear  # shouldn't happen

        # ── fallback: mirror if one side failed ─────────────────────
        if p["ray_toward_start"] is None:
            # Mirror ray_toward_end around the inner_normal
            if p["ray_toward_end"] is not None:
                p["ray_toward_start"] = (2 * p["inner_normal"] - p["ray_toward_end"]) % 360
            else:
                p["ray_toward_start"] = street_back_bear
            print(f"    ray_toward_start fallback: {p['ray_toward_start']:.1f}°")

        if p["ray_toward_end"] is None:
            if p["ray_toward_start"] is not None:
                p["ray_toward_end"] = (2 * p["inner_normal"] - p["ray_toward_start"]) % 360
            else:
                p["ray_toward_end"] = street_fwd_bear
            print(f"    ray_toward_end fallback: {p['ray_toward_end']:.1f}°")

        print(
            f"  [{pos+1}/{n_valid}] pano {p['panoId'][:8]}  "
            f"ray_start={p['ray_toward_start']:.1f}°  "
            f"ray_end={p['ray_toward_end']:.1f}°  "
            f"inner_normal={p['inner_normal']:.1f}°  "
            f"{'[START EDGE]' if is_start else ''}{'[END EDGE]' if is_end else ''}"
        )

    # --------------------------------------------------
    # EXPORT — stored for writing after facade analysis
    # --------------------------------------------------
    centroid_to_wgs = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)
    centroid_lon, centroid_lat = centroid_to_wgs.transform(centroid_proj.x, centroid_proj.y)

    export = {
        "street":   street_name,
        "city":     CITY,
        "centroid": {
            "lat":      centroid_lat,
            "lon":      centroid_lon,
            "x_25833":  centroid_proj.x,
            "y_25833":  centroid_proj.y,
        },
        "building_facades": [],   # filled after facade analysis
        "panos": [
            {
                "panoId":           p["panoId"],
                "pos":              p["pos"],
                "lat":              p["lat"],
                "lon":              p["lon"],
                "x_25833":          p["pano_pt"].x,
                "y_25833":          p["pano_pt"].y,
                "street_bearing":   p["street_bearing"],
                "inner_normal":     p["inner_normal"],
                "dist_to_facade":   p["dist_to_facade"],
                "ray_toward_start": p["ray_toward_start"],
                "ray_toward_end":   p["ray_toward_end"],
                "is_start_edge":    p["is_start_edge"],
                "is_end_edge":      p["is_end_edge"],
                "start_snapped":    p["start_snapped"],
                "end_snapped":      p["end_snapped"],
                "snap_start_corner": ([p["snap_start_corner"].x, p["snap_start_corner"].y]
                                      if p["snap_start_corner"] else None),
                "snap_end_corner":   ([p["snap_end_corner"].x, p["snap_end_corner"].y]
                                      if p["snap_end_corner"] else None),
                "skip":             p["skip"],
                "image_path":       os.path.join(collect_dir, f"pt_{p['idx']:03d}_inner.png"),
            }
            for p in pano_data
        ]
    }
    street_exports[street_name] = (export, collect_dir)
    print(f"  Phase 2b complete — metadata queued for writing")

    n_queued = sum(1 for p in pano_data if not p["skip"])
    print(f"  Phase 0–2 complete: {n_queued} panos queued for download")



# Add pano overlays to facade visualisation
print("\n=== Adding pano overlays to visualisation ===")
add_pano_overlays(facade_fig, facade_ax, facade_vis_path, street_exports)

# Write all metadata.json files
print("\n=== Writing metadata JSON files ===")

# Serialise building polygons once — exterior ring as list of [x, y] pairs
# in EPSG:25833. Also include WGS84 centroid of each polygon for convenience.
_proj_to_wgs = Transformer.from_crs("EPSG:25833", "EPSG:4326", always_xy=True)

building_polygons = []
for bid, row in block_buildings.iterrows():
    geom = row.geometry
    if geom is None or geom.is_empty:
        continue
    coords_25833 = [[round(x, 3), round(y, 3)]
                    for x, y in geom.exterior.coords]
    cen          = geom.centroid
    cen_lon, cen_lat = _proj_to_wgs.transform(cen.x, cen.y)
    building_polygons.append({
        "building_id":   str(bid),
        "coords_25833":  coords_25833,          # [[x, y], …] exterior ring
        "centroid_lat":  round(cen_lat, 8),
        "centroid_lon":  round(cen_lon, 8),
        "centroid_x":    round(cen.x, 3),
        "centroid_y":    round(cen.y, 3),
    })

print(f"  Building polygons serialised: {len(building_polygons)}")

# ======================================================
# BUILDING LABEL MAP + UNIT MERGE
# ======================================================
# Computed once after facade analysis, before writing metadata.json.
# Labels (e.g. "Karmarschgasse_12") and building unit definitions are
# stored in metadata.json and consumed directly by the inference script.

import math as _math

def _sample_edge_bearing(seg):
    dx = seg.coords[-1][0] - seg.coords[0][0]
    dy = seg.coords[-1][1] - seg.coords[0][1]
    return (_math.degrees(_math.atan2(dx, dy)) + 360) % 360 % 180

def _sample_bearings_parallel(b1, b2, tol=15.0):
    diff = abs(b1 - b2) % 180
    return diff < tol or diff > (180 - tol)

# Rebuild edge classification from block_buildings geometry
_edge_classification = {}
_edge_segments_map   = {}

_block_sides_sample = []
for _i in range(len(list(block_polygon.exterior.coords)) - 1):
    _c = list(block_polygon.exterior.coords)
    _block_sides_sample.append(LineString([_c[_i], _c[_i + 1]]))

_all_edge_segs_sample = []
for _, _brow in block_buildings.iterrows():
    if _brow.geometry is None or _brow.geometry.is_empty:
        continue
    _c2 = list(_brow.geometry.exterior.coords)
    for _a, _b in zip(_c2[:-1], _c2[1:]):
        _seg = LineString([_a, _b])
        if _seg.length >= 0.5:
            _all_edge_segs_sample.append(_seg)
_all_edges_union_sample = unary_union(_all_edge_segs_sample) if _all_edge_segs_sample else None

_bid_index_list  = list(block_buildings.index)
_geom_list_s     = [block_buildings.loc[bid, "geometry"] for bid in _bid_index_list]
_other_unions_s  = {
    str(bid): (unary_union([g for j, g in enumerate(_geom_list_s) if j != i
                            and g is not None and not g.is_empty]) or None)
    for i, bid in enumerate(_bid_index_list)
}

def _sample_classify_edge(seg, other_union):
    if other_union is not None:
        try:
            if seg.intersection(other_union.buffer(0.05)).length >= seg.length * 0.2:
                return "party"
        except Exception:
            pass
    if not _block_sides_sample or _all_edges_union_sample is None:
        return "outer"
    midpt        = seg.interpolate(0.5, normalized=True)
    nearest_side = min(_block_sides_sample, key=lambda s: midpt.distance(s))
    dx  = nearest_side.centroid.x - midpt.x
    dy  = nearest_side.centroid.y - midpt.y
    nrm = (_math.degrees(_math.atan2(dx, dy)) + 360) % 360
    dist = midpt.distance(nearest_side)
    ray  = LineString([
        (midpt.x, midpt.y),
        (midpt.x + _math.sin(_math.radians(nrm)) * dist,
         midpt.y + _math.cos(_math.radians(nrm)) * dist),
    ])
    inter = ray.difference(seg.buffer(0.1)).intersection(_all_edges_union_sample)
    return "inner" if not inter.is_empty else "outer"

for _bid_raw, _brow in block_buildings.iterrows():
    _bid_str = str(_bid_raw)
    if _brow.geometry is None or _brow.geometry.is_empty:
        continue
    _coords2     = list(_brow.geometry.exterior.coords)
    _other_union = _other_unions_s.get(_bid_str)
    for _ei in range(len(_coords2) - 1):
        _seg = LineString([_coords2[_ei], _coords2[_ei + 1]])
        if _seg.length < 0.5:
            continue
        _key = (_bid_str, _ei)
        _edge_segments_map[_key]   = _seg
        _edge_classification[_key] = _sample_classify_edge(_seg, _other_union)

_n_o = sum(1 for v in _edge_classification.values() if v == "outer")
_n_i = sum(1 for v in _edge_classification.values() if v == "inner")
_n_p = sum(1 for v in _edge_classification.values() if v == "party")
print(f"  Edge classification (for labels): {_n_o} outer, {_n_i} inner, {_n_p} party")

# --- Hausnummern aus building_facades ---
_hn_map     = {b["building_id"]: b.get("housenumber") for b in building_facades}
_street_osm = {b["building_id"]: b.get("street")      for b in building_facades}

# --- Nearest street via pano positions ---
_street_pano_pts_s = {}
for _sn, (_exp, _) in street_exports.items():
    _pts = [(_p["x_25833"], _p["y_25833"])
            for _p in _exp["panos"] if not _p["skip"]]
    if _pts:
        _street_pano_pts_s[_sn] = _pts

def _nearest_street_for_bid(bid_str):
    outer_pts = []
    for (_b, _ei), _lbl in _edge_classification.items():
        if _b != bid_str or _lbl != "outer":
            continue
        _seg = _edge_segments_map.get((_b, _ei))
        if _seg:
            _mp = _seg.interpolate(0.5, normalized=True)
            outer_pts.append((_mp.x, _mp.y))
    if not outer_pts or not _street_pano_pts_s:
        return None
    from shapely.geometry import Point as _SP
    cx  = sum(p[0] for p in outer_pts) / len(outer_pts)
    cy  = sum(p[1] for p in outer_pts) / len(outer_pts)
    ctr = _SP(cx, cy)
    best, best_d = None, float("inf")
    for _sn, _pts in _street_pano_pts_s.items():
        d = min(ctr.distance(_SP(px, py)) for px, py in _pts)
        if d < best_d:
            best_d, best = d, _sn
    return best

# --- Union-Find merge ---
_all_bids_s = [str(bid) for bid in _bid_index_list]
_parent_s   = {b: b for b in _all_bids_s}

def _find_s(x):
    while _parent_s[x] != x:
        _parent_s[x] = _parent_s[_parent_s[x]]
        x = _parent_s[x]
    return x

def _union_s(x, y):
    _parent_s[_find_s(x)] = _find_s(y)

def _has_inner_s(bid_str):
    return any(lbl == "inner"
               for (b, _), lbl in _edge_classification.items() if b == bid_str)

def _outer_bearing_s(bid_str):
    bearings = []
    for (b, ei), lbl in _edge_classification.items():
        if b != bid_str or lbl != "outer":
            continue
        seg = _edge_segments_map.get((b, ei))
        if seg and seg.length >= 0.5:
            bearings.append(_sample_edge_bearing(seg))
    if not bearings:
        return None
    sines   = sum(_math.sin(_math.radians(2 * b)) for b in bearings)
    cosines = sum(_math.cos(_math.radians(2 * b)) for b in bearings)
    return (_math.degrees(_math.atan2(sines, cosines)) / 2) % 180

MERGE_ANGLE_DEG = 15.0

# Adjacency: pairs sharing a party edge
_adjacency_s  = {}
_bid_items_s  = [(str(bid), list(block_buildings.loc[bid, "geometry"].exterior.coords))
                 for bid in _bid_index_list
                 if block_buildings.loc[bid, "geometry"] is not None
                 and not block_buildings.loc[bid, "geometry"].is_empty]

for _i2, (_ba, _ca) in enumerate(_bid_items_s):
    for _j2, (_bb, _cb) in enumerate(_bid_items_s):
        if _j2 <= _i2:
            continue
        _pair = frozenset({_ba, _bb})
        for _eia in range(len(_ca) - 1):
            if _edge_classification.get((_ba, _eia)) != "party":
                continue
            _sa = _edge_segments_map.get((_ba, _eia))
            if _sa is None:
                continue
            for _eib in range(len(_cb) - 1):
                if _edge_classification.get((_bb, _eib)) != "party":
                    continue
                _sb = _edge_segments_map.get((_bb, _eib))
                if _sb is None:
                    continue
                try:
                    _ov = _sa.intersection(_sb.buffer(0.15)).length
                except Exception:
                    continue
                _shorter = min(_sa.length, _sb.length)
                if _shorter > 0 and _ov / _shorter >= 0.80:
                    _adjacency_s.setdefault(_pair, []).append(
                        ((_ba, _eia), (_bb, _eib)))

_changed_s = True
while _changed_s:
    _changed_s = False
    for _pair, _shared in _adjacency_s.items():
        _ba, _bb = tuple(_pair)
        if _has_inner_s(_ba) and _has_inner_s(_bb):
            continue
        _merge_ok = False
        for (_ka, _kb) in _shared:
            _sp = _edge_segments_map.get(_ka)
            if _sp is None:
                continue
            _pb = _sample_edge_bearing(_sp)
            for _ob in filter(None, [_outer_bearing_s(_ba), _outer_bearing_s(_bb)]):
                if _sample_bearings_parallel(_pb, _ob, MERGE_ANGLE_DEG):
                    _merge_ok = True
                    break
            if _merge_ok:
                break
        if _merge_ok and _find_s(_ba) != _find_s(_bb):
            _union_s(_ba, _bb)
            _changed_s = True

# Build unit list
_groups_s = {}
for _bid_s in _all_bids_s:
    _groups_s.setdefault(_find_s(_bid_s), []).append(_bid_s)

_unit_defs = []
_bid_to_unit_s = {}
for _root, _bids in sorted(_groups_s.items()):
    _outer_edges_u = [(_b, _ei) for (_b, _ei), _lbl in _edge_classification.items()
                      if _b in _bids and _lbl == "outer"]
    _inner_edges_u = [(_b, _ei) for (_b, _ei), _lbl in _edge_classification.items()
                      if _b in _bids and _lbl == "inner"]
    _outer_bid_u   = ({_b for _b, _ in _outer_edges_u} or {_bids[0]}).pop()
    _inner_bid_u   = ({_b for _b, _ in _inner_edges_u} or {_bids[0]}).pop()
    _uidx          = len(_unit_defs)
    _unit_defs.append({"bids": _bids, "outer_bid": _outer_bid_u, "inner_bid": _inner_bid_u})
    for _b in _bids:
        _bid_to_unit_s[_b] = _uidx

_n_merged_s = sum(1 for u in _unit_defs if len(u["bids"]) > 1)
print(f"  Building units: {len(_unit_defs)} total, {_n_merged_s} merged groups")

# --- Assign labels ---
_street_counters_s  = {}
_bid_label_map_s    = {}

def _base_label_s(bid_str):
    hn     = _hn_map.get(bid_str)
    street = _street_osm.get(bid_str) or _nearest_street_for_bid(bid_str) or "Gebaeude"
    slug   = street.replace(" ", "_").replace("/", "-")
    if hn:
        return slug, str(hn)
    _street_counters_s[slug] = _street_counters_s.get(slug, 0) + 1
    return slug, f"{_street_counters_s[slug]:02d}"

# Outer bids first for deterministic numbering
for _unit in _unit_defs:
    _ob = _unit["outer_bid"]
    if _ob not in _bid_label_map_s:
        _slug, _num = _base_label_s(_ob)
        _bid_label_map_s[_ob] = f"{_slug}_{_num}"

# Hinterhaus members
def _bid_geom_centroid(bid_str):
    try:
        _bid_raw = next(bid for bid in block_buildings.index if str(bid) == bid_str)
        _g = block_buildings.loc[_bid_raw, "geometry"]
        return _g.centroid if _g and not _g.is_empty else None
    except StopIteration:
        return None

for _unit in _unit_defs:
    _ob   = _unit["outer_bid"]
    _base = _bid_label_map_s[_ob]
    _hinten = [_b for _b in _unit["bids"] if _b != _ob]
    if len(_hinten) == 1:
        _bid_label_map_s[_hinten[0]] = f"{_base}_hinten"
    elif len(_hinten) > 1:
        _ob_ctr = _bid_geom_centroid(_ob)
        if _ob_ctr:
            _hinten.sort(key=lambda b: (
                _bid_geom_centroid(b).distance(_ob_ctr)
                if _bid_geom_centroid(b) else 0))
        for _depth, _hb in enumerate(_hinten, 1):
            _bid_label_map_s[_hb] = f"{_base}_hinten{_depth}"

# Any unassigned (standalone buildings not outer_bid)
for _bid_s in _all_bids_s:
    if _bid_s not in _bid_label_map_s:
        _slug, _num = _base_label_s(_bid_s)
        _bid_label_map_s[_bid_s] = f"{_slug}_{_num}"

print("  Building labels:")
for _bid_s, _lbl in sorted(_bid_label_map_s.items()):
    print(f"    {_bid_s} -> {_lbl}")

# Attach label to building_facades entries
for _bf in building_facades:
    _bf["label"] = _bid_label_map_s.get(_bf["building_id"], _bf["building_id"])

# Draw building labels on facade visualisation
for _bid_raw, _brow in block_buildings.iterrows():
    _bid_str = str(_bid_raw)
    if _brow.geometry is None or _brow.geometry.is_empty:
        continue
    _ctr = _brow.geometry.centroid
    _lbl = _bid_label_map_s.get(_bid_str, _bid_str)
    facade_ax.annotate(
        _lbl, xy=(_ctr.x, _ctr.y),
        ha="center", va="center",
        fontsize=6, color="#222222", zorder=6, clip_on=True,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, ec="none"),
    )
facade_fig.savefig(facade_vis_path, dpi=150, bbox_inches="tight")
print(f"  Facade visualisation updated with building labels -> {facade_vis_path}")

# Serialise unit definitions for export
_building_units_export = [
    {
        "unit_idx":    _uidx,
        "bids":        _u["bids"],
        "labels":      [_bid_label_map_s.get(_b, _b) for _b in _u["bids"]],
        "outer_bid":   _u["outer_bid"],
        "inner_bid":   _u["inner_bid"],
        "outer_label": _bid_label_map_s.get(_u["outer_bid"], _u["outer_bid"]),
        "inner_label": _bid_label_map_s.get(_u["inner_bid"], _u["inner_bid"]),
        "merged":      len(_u["bids"]) > 1,
    }
    for _uidx, _u in enumerate(_unit_defs)
]

for sname, (export, collect_dir) in street_exports.items():
    export["building_facades"]  = building_facades
    export["building_polygons"] = building_polygons
    export["building_units"]    = _building_units_export
    export["bid_label_map"]     = _bid_label_map_s
    export["block_polygon_25833"] = [
        [round(x, 3), round(y, 3)]
        for x, y in block_polygon.exterior.coords
    ]
    json_path = os.path.join(collect_dir, "metadata.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    print(f"  Saved -> {json_path}")

# ======================================================
# PHASE 3 — TILE DOWNLOAD + STITCH (all streets)
# ======================================================
print("\n=== Phase 3: Tile download (all streets) ===")

for street_name in STREET_NAMES:
    export_data, collect_dir = street_exports.get(street_name, (None, None))
    if export_data is None:
        continue

    street_slug = street_name.replace(" ", "_").replace("/", "-")
    pano_list   = export_data["panos"]
    n_active    = sum(1 for p in pano_list if not p["skip"])

    # ── Resume check: street level ───────────────────────────────────────
    # If every non-skipped pano already has a downloaded image on disk,
    # this street was completed in a previous run — skip it entirely.
    already_done = [
        os.path.exists(os.path.join(collect_dir, f"{p['panoId']}_inner.png"))
        for p in pano_list if not p["skip"]
    ]
    if already_done and all(already_done):
        print(f"\n  Street: {street_name}  — already complete ({n_active} panos), skipping")
        continue

    n_done = sum(already_done)
    print(f"\n  Street: {street_name}  ({n_active} panos"
          + (f", {n_done} already downloaded — resuming" if n_done > 0 else "")
          + ")")

    for p in pano_list:
        if p["skip"]:
            print(f"  Skipping pano {p['panoId']}")
            continue

        # ── Resume check: pano level ─────────────────────────────────────
        img_out_check = os.path.join(collect_dir, f"{p['panoId']}_inner.png")
        if os.path.exists(img_out_check):
            print(f"  Pano {p['panoId'][:8]} already downloaded, skipping")
            continue

        pano_start = time.time()
        print(f"\n  --- Downloading pano {p['panoId']} ---")

        meta = _get(
            META_URL,
            params={"panoId": p["panoId"], "session": session_id, "key": API_KEY}
        ).json()

        img_w, img_h = meta["imageWidth"], meta["imageHeight"]
        tiles_x      = math.ceil(img_w / tile_width)
        tiles_y      = math.ceil(img_h / tile_height)
        zoom         = int(max(math.log2(tiles_x), math.log2(tiles_y)))
        deg_per_tile = 360 / tiles_x

        # Tile columns are indexed relative to the camera's own heading,
        # not relative to geographic north. Convert inner_normal (absolute)
        # to camera-relative before computing which columns to download.
        camera_heading  = meta.get("heading", 0.0)

        cols_primary = cols_for_bearings(
            p["ray_toward_start"], p["ray_toward_end"],
            camera_heading, deg_per_tile, tiles_x
        )
        pano_pt_p3    = Point(p["x_25833"], p["y_25833"])
        half_shift    = tiles_x // 2
        cols_opposite = [(c + half_shift) % tiles_x for c in cols_primary]

        # IMPORTANT — API inversion:
        # cols_primary (computed from inner_normal via cols_for_bearings) has
        # been empirically verified to correspond to the OPPOSITE side in the
        # actual API response — i.e. when the calibration says "primary" (the
        # geocoding-verified block-interior direction), the API actually serves
        # the tiles for cols_opposite. The root cause is an unresolvable
        # inconsistency in the GSW Tiles API column-to-bearing mapping that
        # varies per pano and has no reliable predictor in metadata fields.
        # Fix: pass cols_opposite as the "primary" candidate and cols_primary
        # as the "opposite" candidate so that resolve_facade_column's "primary"
        # result correctly selects the API-block-interior column set.
        cols, matched_side, resolve_detail = resolve_facade_column(
            pano_pt_p3, p["inner_normal"],
            cols_opposite,   # <- API reality: this is what inner_normal actually loads
            cols_primary,    # <- API reality: this is the opposite side
            buildings_all
        )

        total_tiles = len(cols) * tiles_y
        print(f"  INNER cols={len(cols)} rows={tiles_y}  "
              f"normal={p['inner_normal']:.1f}°  camera_heading={camera_heading:.1f}°  "
              f"ray_start={p['ray_toward_start']:.1f}°  ray_end={p['ray_toward_end']:.1f}°  "
              f"total_tiles={total_tiles}")
        print(f"  CALIBRATION matched_side={matched_side}  {resolve_detail}")
        if matched_side == "undetermined":
            print(f"  WARNING: could not determine block-interior side (buildings_all "
                  f"empty or equidistant) - using primary (geometry-only) choice, "
                  f"using primary (geometry-only) choice, please verify this pano manually")

        canvas = np.zeros((tiles_y * tile_height, len(cols) * tile_width, 3), np.uint8)
        tiles_done = 0

        for ix, col in enumerate(cols):
            for yrow in range(tiles_y):
                for attempt in range(3):
                    r = _get(
                        f"{TILE_URL}/{zoom}/{col}/{yrow}",
                        params={
                            "session": session_id,
                            "panoId":  p["panoId"],
                            "zoom":    zoom,
                            "key":     API_KEY,
                        }
                    )

                    if r.status_code == 200:
                        tile = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                        if tile is not None:
                            canvas[
                                yrow * tile_height:(yrow + 1) * tile_height,
                                ix   * tile_width :(ix   + 1) * tile_width
                            ] = tile
                            tiles_done += 1
                            print(f"      tile {tiles_done}/{total_tiles}  col={col} row={yrow} ✓", end="\r")
                        break
                    elif r.status_code == 429:
                        print(f"      RATE LIMITED z={zoom} x={col} y={yrow} attempt={attempt+1}")
                        time.sleep(0.3 * (attempt + 1))
                    else:
                        print(f"      TILE FAILED z={zoom} x={col} y={yrow} → {r.status_code}")
                        break

            time.sleep(0.3)

        sample_dir = os.path.join(BASE_OUT_DIR, f"{street_slug}_pt_{p['panoId']}")
        os.makedirs(sample_dir, exist_ok=True)

        img_out = os.path.join(collect_dir, f"{p['panoId']}_inner.png")
        cv2.imwrite(os.path.join(sample_dir, "facade_inner.png"), canvas)
        cv2.imwrite(img_out, canvas)

        # Update image_path in the export so the JSON reflects actual file
        for ep in export_data["panos"]:
            if ep["panoId"] == p["panoId"]:
                ep["image_path"] = img_out

        print(f"  pano completed: {tiles_done}/{total_tiles} tiles in {time.time() - pano_start:.1f}s")
        time.sleep(2.0)

    print(f"  Street {street_name} download complete")

print("\n" + "="*60)
print("  ALL DONE")
print(f"  Blocks processed : {len(STREET_NAMES)} streets")
print(f"  Output directory : {BASE_OUT_DIR}")
print(f"  Visualisation    : {os.path.join(BASE_OUT_DIR, 'facade_classification.png')}")
print("="*60)
