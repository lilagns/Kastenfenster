import os
import sys
import json
import csv
import math
import argparse
import glob

import cv2
import numpy as np
from ultralytics import YOLO
from pyproj import Transformer
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union

# --------------------------------------------------
# Argument parsing
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="YOLO inference on Tiles-API facade images with metadata-driven deduplication"
)
parser.add_argument("--model",       required=True,  help="Path to YOLO model (.pt)")
parser.add_argument("--samples_dir", required=True,  help="Root samples directory (contains per-street subfolders)")
parser.add_argument("--thresh",      type=float, default=0.5, help="Confidence threshold")
parser.add_argument("--hit_dedup_m", type=float, default=0.8,
                    help="Wall-hit deduplication radius in metres (default: 0.8)")
parser.add_argument("--cx_axis_tol", type=float, default=0.04,
                    help="Normalised-cx tolerance for vertical axis grouping within one pano (default: 0.04)")
parser.add_argument("--floor_band",  type=float, default=0.10,
                    help="Normalised-cy band height for floor-layer matching in hit-space dedup (default: 0.10)")
parser.add_argument("--resolution",  default=None,   help="Optional WxH display resolution e.g. 1280x720")
parser.add_argument("--show",        action="store_true", help="Show annotated images after inference")

args = parser.parse_args()

# --------------------------------------------------
# Validation
# --------------------------------------------------

if not os.path.exists(args.model):
    print("ERROR: Model file not found.")
    sys.exit(1)

if not os.path.isdir(args.samples_dir):
    print("ERROR: samples_dir not found.")
    sys.exit(1)

# --------------------------------------------------
# Load model
# --------------------------------------------------

model  = YOLO(args.model, task="detect")
labels = model.names

# --------------------------------------------------
# Class colors
# --------------------------------------------------

bbox_colors = [
    (164,120,87), (68,148,228), (93,97,209), (178,182,133),
    (88,159,106), (96,202,231), (159,124,168), (169,162,241),
    (98,118,150), (172,176,184)
]

# --------------------------------------------------
# Optional resize
# --------------------------------------------------

resize = False
if args.resolution:
    resize = True
    resW, resH = map(int, args.resolution.split("x"))

# --------------------------------------------------
# Discover all metadata.json files
# --------------------------------------------------

meta_files = sorted(glob.glob(
    os.path.join(args.samples_dir, "*", "metadata.json")
))

if not meta_files:
    print("ERROR: No metadata.json files found under samples_dir.")
    sys.exit(1)

print(f"\nFound {len(meta_files)} street(s):")
for mf in meta_files:
    print(f"  {mf}")

# --------------------------------------------------
# Cache all metadata.json files (read each file exactly once)
# --------------------------------------------------
# All subsequent code reads from _meta_cache[mf] instead of reopening files.

_meta_cache = {}
for mf in meta_files:
    with open(mf, encoding="utf-8") as f:
        _meta_cache[mf] = json.load(f)

# --------------------------------------------------
# Load block geometry from first metadata.json
# --------------------------------------------------

_first_meta = _meta_cache[meta_files[0]]

_c = _first_meta["centroid"]
centroid_x = _c["x_25833"]
centroid_y = _c["y_25833"]

building_facades = _first_meta.get("building_facades", [])
if not building_facades:
    print("WARNING: No building_facades found in metadata. Building assignment will be skipped.")

# Transformer: WGS84 -> EPSG:25833
to_proj = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)

facade_by_id = {b["building_id"]: b for b in building_facades}

RAY_LENGTH_M = 80.0

print(f"Block centroid loaded: ({centroid_x:.1f}, {centroid_y:.1f})")
print(f"Buildings with facade data: {len(building_facades)}")

# --------------------------------------------------
# Image path resolution
# --------------------------------------------------
# sample20.py writes  <street_slug>/<panoId>_inner.png  and stores that
# path in metadata["panos"][n]["image_path"].
# If the stored path no longer exists we try several fallbacks so that
# both old (pt_NNN_inner.png) and new (panoId_inner.png) samples work.

def resolve_image_path(pano_meta, collect_dir):
    """
    Return the filesystem path to the pano's stitched image, or None.

    sample24_3 writes image_path into the export dict after each pano
    download, so pano_meta["image_path"] is the authoritative source.

    Fallback chain (for older samples):
      1. pano_meta["image_path"] as stored            (sample24_3 / sample20)
      2. <collect_dir>/<panoId>_inner.png             (sample20 naming)
      3. <collect_dir>/pt_<pos:03d>_inner.png         (sample15 legacy)
      4. Any *.png in collect_dir containing the panoId
    """
    stored = pano_meta.get("image_path", "")
    if stored and os.path.exists(stored):
        return stored

    pid = pano_meta["panoId"]

    # sample24_3 / sample20 canonical name
    candidate = os.path.join(collect_dir, f"{pid}_inner.png")
    if os.path.exists(candidate):
        return candidate

    # sample15 legacy: pt_NNN_inner.png
    pos = pano_meta.get("pos", pano_meta.get("idx"))
    if pos is not None:
        legacy = os.path.join(collect_dir, f"pt_{int(pos):03d}_inner.png")
        if os.path.exists(legacy):
            return legacy

    # Fuzzy: any png containing the panoId
    try:
        for fname in os.listdir(collect_dir):
            if pid in fname and fname.endswith(".png"):
                return os.path.join(collect_dir, fname)
    except OSError:
        pass

    return None


# --------------------------------------------------
# IoU helper — defined once at module level
# --------------------------------------------------

def _iou(a, b):
    """Intersection-over-union for two YOLO detection boxes."""
    ax1, ay1, ax2, ay2 = a.xyxy.cpu().numpy().squeeze()
    bx1, by1, bx2, by2 = b.xyxy.cpu().numpy().squeeze()
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih   = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter    = iw * ih
    area_a   = (ax2 - ax1) * (ay2 - ay1)
    area_b   = (bx2 - bx1) * (by2 - by1)
    union    = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------
# Storage
# --------------------------------------------------

all_street_detections = {}   # street_name -> list of detection dicts
all_annotated         = []   # (street_name, img_path, pano_id, raw_frame)

# --------------------------------------------------
# Per-street inference
# --------------------------------------------------

for meta_file in meta_files:

    meta = _meta_cache[meta_file]

    street_name = meta["street"]
    collect_dir = os.path.dirname(meta_file)

    # Sort panos by street position (pos field from sample20; idx fallback for sample15)
    panos = [p for p in meta["panos"] if not p["skip"]]
    panos.sort(key=lambda p: p.get("pos", p.get("idx", 0)))

    print(f"\n{'='*60}")
    print(f"  Inference: {street_name}  ({len(panos)} valid panos)")
    print(f"{'='*60}")

    street_detections = []

    for pano in panos:
        img_path = resolve_image_path(pano, collect_dir)

        if img_path is None:
            print(f"  WARNING: image not found for pano {pano['panoId']} — skipping")
            continue

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  WARNING: could not read — {img_path}")
            continue

        if resize:
            frame = cv2.resize(frame, (resW, resH))

        results    = model(frame, verbose=False)
        detections = results[0].boxes

        img_h, img_w = frame.shape[:2]

        # ---- Per-image IoU deduplication (>80% overlap -> keep higher-conf) ----
        raw_boxes = []
        for det in detections:
            conf = det.conf.item()
            if conf < args.thresh:
                continue
            raw_boxes.append(det)

        raw_boxes.sort(key=lambda d: d.conf.item(), reverse=True)
        keep = []
        suppressed = set()
        for i, det_i in enumerate(raw_boxes):
            if i in suppressed:
                continue
            keep.append(det_i)
            for j, det_j in enumerate(raw_boxes):
                if j <= i or j in suppressed:
                    continue
                if _iou(det_i, det_j) > 0.80:
                    suppressed.add(j)

        for det in keep:
            conf_det  = det.conf.item()
            xmin, ymin, xmax, ymax = det.xyxy.cpu().numpy().squeeze().astype(int)
            classidx  = int(det.cls.item())
            classname = labels[classidx]

            cx = ((xmin + xmax) / 2) / img_w
            cy = ((ymin + ymax) / 2) / img_h
            h  = (ymax - ymin) / img_h

            street_detections.append({
                "class":            classname,
                "confidence":       conf_det,
                "panoId":           pano["panoId"],
                "lat":              pano["lat"],
                "lon":              pano["lon"],
                "inner_normal":     pano["inner_normal"],
                # sample20 ray system — absolute geographic bearings
                "ray_toward_start": pano.get("ray_toward_start"),
                "ray_toward_end":   pano.get("ray_toward_end"),
                # sample15 legacy fields (kept for fallback in compute_ray_angle)
                "ang_prev":         pano.get("ang_prev"),
                "ang_next":         pano.get("ang_next"),
                "cx":               cx,
                "cy":               cy,
                "h":                h,
                "_bbox":            (xmin, ymin, xmax, ymax),
                "_classidx":        classidx,
                "_img_path":        img_path,
            })

        all_annotated.append((street_name, img_path, pano["panoId"], frame.copy()))

    all_street_detections[street_name] = street_detections
    print(f"  Raw detections: {len(street_detections)}")

# --------------------------------------------------
# Build building geometry lookup
# --------------------------------------------------

block_buildings_geom = {}
_all_osm_geoms       = []

_building_polygons = _first_meta.get("building_polygons", [])

if not _building_polygons:
    print("WARNING: No building_polygons found in metadata.json. "
          "Re-run sample20.py to regenerate metadata with polygon coordinates.")
else:
    for bp in _building_polygons:
        bid    = bp["building_id"]
        coords = bp["coords_25833"]
        if len(coords) < 3:
            continue
        try:
            poly = Polygon([(c[0], c[1]) for c in coords])
            if poly.is_valid and not poly.is_empty:
                block_buildings_geom[bid] = poly
                _all_osm_geoms.append(poly)
        except Exception:
            continue
    print(f"Building geometries loaded from metadata: {len(block_buildings_geom)}")

# --------------------------------------------------
# Building labels and units — loaded from metadata.json
# --------------------------------------------------
# Both are computed at sampling time (sample20.py) and stored in
# metadata.json under "bid_label_map" and "building_units".
# The inference script reads them directly — no recomputation needed.

_bid_label_map = _first_meta.get("bid_label_map", {})

if not _bid_label_map:
    print("WARNING: bid_label_map not found in metadata.json. "
          "Re-run sample24_3.py to generate labels.")

print("Building labels (from metadata):")
for bid, lbl in sorted(_bid_label_map.items()):
    print(f"  {bid} -> {lbl}")

# building_units: list of unit dicts as exported by sample20
# Each entry: {unit_idx, bids, labels, outer_bid, inner_bid,
#              outer_label, inner_label, merged}
_building_units_raw = _first_meta.get("building_units", [])

# Reconstruct edge key lists per unit from edge_classification
# (edge_classification is built below — deferred; units are used later)
# We store the raw unit dicts now and materialise edge lists after
# edge_classification is available.
_bid_to_unit = {}   # bid -> unit_idx (populated after edge_classification)
for u in _building_units_raw:
    for bid in u["bids"]:
        _bid_to_unit[bid] = u["unit_idx"]

_n_merged = sum(1 for u in _building_units_raw if u.get("merged", False))
print(f"\nBuilding units (from metadata): {len(_building_units_raw)} total, "
      f"{_n_merged} merged groups")
for u in _building_units_raw:
    if u.get("merged", False):
        print(f"  Merged unit: {' + '.join(u['labels'])}")

RAY_WIN_M = 50.0

from shapely.geometry import LineString as _LS_agg, Point as _Pt_agg, Polygon as _Poly_agg
from shapely.ops import unary_union as _uu_agg

_bp_coords = _first_meta.get("block_polygon_25833", [])
if _bp_coords and len(_bp_coords) >= 3:
    _block_polygon_agg = _Poly_agg([(c[0], c[1]) for c in _bp_coords])
    print(f"Block polygon loaded from metadata: {len(_bp_coords)-1} corners")
else:
    _block_polygon_agg = _uu_agg(list(block_buildings_geom.values())).convex_hull \
                         if block_buildings_geom else None
    print("WARNING: block_polygon_25833 not in metadata — using building convex hull as fallback")

def _block_sides_agg(bp):
    coords = list(bp.exterior.coords)
    return [_LS_agg([coords[i], coords[i+1]]) for i in range(len(coords)-1)]

def _edge_normal_toward_agg(seg, target):
    dx = seg.coords[-1][0] - seg.coords[0][0]
    dy = seg.coords[-1][1] - seg.coords[0][1]
    eb = (math.degrees(math.atan2(dx, dy)) + 360) % 360
    n1, n2 = (eb + 90) % 360, (eb - 90) % 360
    mp = seg.interpolate(0.5, normalized=True)
    d1 = _Pt_agg(mp.x + math.sin(math.radians(n1)),
                 mp.y + math.cos(math.radians(n1))).distance(target)
    d2 = _Pt_agg(mp.x + math.sin(math.radians(n2)),
                 mp.y + math.cos(math.radians(n2))).distance(target)
    return n1 if d1 <= d2 else n2

_block_sides_list_agg = _block_sides_agg(_block_polygon_agg) \
                        if _block_polygon_agg and not _block_polygon_agg.is_empty else []

_all_edge_segs_agg = []
for _geom in block_buildings_geom.values():
    _coords = list(_geom.exterior.coords)
    for _a, _b in zip(_coords[:-1], _coords[1:]):
        _seg = _LS_agg([_a, _b])
        if _seg.length >= 0.5:
            _all_edge_segs_agg.append(_seg)
_all_edges_union_agg = _uu_agg(_all_edge_segs_agg) if _all_edge_segs_agg else None

_bid_list_agg  = list(block_buildings_geom.keys())
_geom_list_agg = [block_buildings_geom[b] for b in _bid_list_agg]
_other_unions_agg = {
    bid: (_uu_agg([g for j, g in enumerate(_geom_list_agg) if j != i]) or None)
    for i, bid in enumerate(_bid_list_agg)
}

def _classify_edge_agg(seg, other_union):
    if other_union is not None:
        overlap = seg.intersection(other_union.buffer(0.05)).length
        if overlap >= seg.length * 0.2:
            return "party"
    if not _block_sides_list_agg or _all_edges_union_agg is None:
        return "outer"
    midpt        = seg.interpolate(0.5, normalized=True)
    nearest_side = min(_block_sides_list_agg, key=lambda s: midpt.distance(s))
    normal_deg   = _edge_normal_toward_agg(seg, nearest_side)
    normal_rad   = math.radians(normal_deg)
    dist_to_blk  = midpt.distance(nearest_side)
    ray_clipped  = _LS_agg([
        (midpt.x, midpt.y),
        (midpt.x + math.sin(normal_rad) * dist_to_blk,
         midpt.y + math.cos(normal_rad) * dist_to_blk),
    ])
    self_excl = seg.buffer(0.1)
    inter     = ray_clipped.difference(self_excl).intersection(_all_edges_union_agg)
    return "inner" if not inter.is_empty else "outer"

edge_classification = {}
edge_segments_map   = {}

for bid, geom in block_buildings_geom.items():
    coords      = list(geom.exterior.coords)
    other_union = _other_unions_agg.get(bid)
    for i in range(len(coords) - 1):
        seg = _LS_agg([coords[i], coords[i+1]])
        if seg.length < 0.5:
            continue
        key = (bid, i)
        edge_segments_map[key]   = seg
        edge_classification[key] = _classify_edge_agg(seg, other_union)

_n_outer = sum(1 for v in edge_classification.values() if v == "outer")
_n_inner = sum(1 for v in edge_classification.values() if v == "inner")
_n_party = sum(1 for v in edge_classification.values() if v == "party")
print(f"Edge classification: {_n_outer} outer, {_n_inner} inner, {_n_party} party")

# Build an R-tree spatial index over all edge segments so that
# _find_hit_edge only tests the small subset of edges whose bounding
# boxes intersect the ray, rather than iterating over every edge.
from shapely.strtree import STRtree

_edge_keys  = list(edge_segments_map.keys())   # [(bid, ei), ...]
_edge_geoms = [edge_segments_map[k] for k in _edge_keys]
_edge_tree  = STRtree(_edge_geoms)

print(f"STRtree built over {len(_edge_keys)} edge segments")

# Materialise building_units with concrete edge key lists now that
# edge_classification is available. _building_units_raw came from metadata.json.
building_units = []
for u_raw in _building_units_raw:
    bids       = u_raw["bids"]
    outer_bid  = u_raw["outer_bid"]
    inner_bid  = u_raw["inner_bid"]
    outer_edges = [(b, ei) for (b, ei), lbl in edge_classification.items()
                   if b in bids and lbl == "outer"]
    inner_edges = [(b, ei) for (b, ei), lbl in edge_classification.items()
                   if b in bids and lbl == "inner"]
    building_units.append({
        "bids":        bids,
        "outer_bid":   outer_bid,
        "inner_bid":   inner_bid,
        "outer_edges": outer_edges,
        "inner_edges": inner_edges,
        "merged":      u_raw.get("merged", False),
    })

print(f"Building units materialised: {len(building_units)} "
      f"({sum(1 for u in building_units if u['merged'])} merged)")


def _find_hit_edge(cpx, cpy, ray_angle):
    """
    Shoot a ray from (cpx, cpy) at ray_angle (radians).
    Returns (hit_Point, building_id, edge_index) or (None, None, None).

    Uses the STRtree index to query only edges whose bounding boxes
    intersect the ray envelope — O(log n) instead of O(n).
    """
    if not _edge_keys:
        return None, None, None

    ray = _LS_agg([
        (cpx, cpy),
        (cpx + math.sin(ray_angle) * RAY_WIN_M,
         cpy + math.cos(ray_angle) * RAY_WIN_M),
    ])
    pano_pt     = _Pt_agg(cpx, cpy)
    ray_trimmed = ray.difference(pano_pt.buffer(0.5))

    best_pt   = None
    best_dist = float("inf")
    best_key  = (None, None)

    for idx in _edge_tree.query(ray_trimmed):
        seg   = _edge_geoms[idx]
        inter = ray_trimmed.intersection(seg)
        if inter.is_empty:
            continue
        if inter.geom_type == "Point":
            candidate = inter
        elif inter.geom_type == "LineString":
            candidate = inter.interpolate(0.5, normalized=True)
        elif inter.geom_type in ("MultiPoint", "GeometryCollection"):
            pts = [g for g in inter.geoms if g.geom_type == "Point"]
            if not pts:
                continue
            candidate = min(pts, key=lambda p: pano_pt.distance(p))
        else:
            continue
        d = pano_pt.distance(candidate)
        if d < best_dist:
            best_dist = d
            best_pt   = candidate
            best_key  = _edge_keys[idx]

    return best_pt, best_key[0], best_key[1]


# --------------------------------------------------
# All pano metadata lookup
# --------------------------------------------------

_all_panos_meta = {}
for mf in meta_files:
    for p in _meta_cache[mf]["panos"]:
        _all_panos_meta[p["panoId"]] = p


# --------------------------------------------------
# Ray angle computation — sample20 system
# --------------------------------------------------
#
# sample20 stores two ABSOLUTE geographic bearings per pano:
#
#   ray_toward_start  bearing of the fan ray toward street start (pos=0)
#   ray_toward_end    bearing of the fan ray toward street end   (pos=N-1)
#
# cols_for_bearings() in sample20 always downloads the SHORTER arc between
# the two rays (the facade window, which is always < 180°).  The resulting
# image has:
#   cx = 0  →  left edge of that arc
#   cx = 1  →  right edge of that arc
#
# We determine left/right purely from the arc direction:
#   - Compute the CW span from ray_toward_start to ray_toward_end.
#   - If CW span <= 180° the image was built CW (start is left, end is right).
#   - If CW span > 180° the image was built CCW (end is left, start is right).
#
# Then interpolate cx linearly within the chosen arc.
#
# Falls back to the sample15 asymmetric ang_prev/ang_next mapping when the
# new fields are absent.

def compute_ray_angle(cx_norm, pano_meta):
    """
    Convert normalised image x position (cx_norm in [0,1]) to an absolute
    geographic bearing in radians.
    """
    nr = math.radians(pano_meta["inner_normal"])

    rs = pano_meta.get("ray_toward_start")
    re = pano_meta.get("ray_toward_end")

    # ---- sample20: absolute bearing rays --------------------------------
    if rs is not None and re is not None:
        cw_span = (re - rs) % 360   # CW arc from rs to re, in degrees

        if cw_span <= 180:
            # Short arc is CW from rs to re  ->  cx=0 is rs, cx=1 is re
            bear_deg = (rs + cx_norm * cw_span) % 360
        else:
            # Short arc is CCW from rs to re  (= CW from re to rs)
            ccw_span = 360 - cw_span           # length of the short arc
            # cx=0 is re, cx=1 is rs
            bear_deg = (re + cx_norm * ccw_span) % 360

        return math.radians(bear_deg)

    # ---- sample15 legacy: ang_prev / ang_next ---------------------------
    ap_deg = pano_meta.get("ang_prev") or 20.0
    an_deg = pano_meta.get("ang_next") or 20.0
    ap = math.radians(ap_deg)
    an = math.radians(an_deg)
    # Asymmetric linear mapping: cx=0 -> normal-ang_prev, cx=1 -> normal+ang_next
    aoff = -ap + cx_norm * (ap + an)
    return nr + aoff


# ==============================================================
# DEDUPLICATION PIPELINE  (3 stages)
# ==============================================================

# ---- Stage 1: Vertical axis grouping per pano (image space) ----
#
# Detections with similar cx (within cx_axis_tol) within the same pano
# are grouped into a vertical axis. Each detection keeps its own class
# and cy — no cross-floor voting here.

all_axes = []

for street_name, detections in all_street_detections.items():

    by_pano = {}
    for det in detections:
        by_pano.setdefault(det["panoId"], []).append(det)

    street_axis_count = 0

    for pid, pano_dets in by_pano.items():
        pano_dets.sort(key=lambda d: d["cx"])

        axes_this_pano = []

        for det in pano_dets:
            assigned = False
            for axis in axes_this_pano:
                if abs(det["cx"] - axis["cx_rep"]) <= args.cx_axis_tol:
                    axis["members"].append(det)
                    total_w = sum(m["confidence"] for m in axis["members"])
                    axis["cx_rep"] = (
                        sum(m["cx"] * m["confidence"] for m in axis["members"]) / total_w
                    )
                    assigned = True
                    break
            if not assigned:
                axes_this_pano.append({"cx_rep": det["cx"], "members": [det]})

        for axis in axes_this_pano:
            all_axes.append({
                "street":  street_name,
                "panoId":  pid,
                "cx_rep":  axis["cx_rep"],
                "members": axis["members"],
            })
            street_axis_count += 1

    print(f"\n{street_name}: {len(detections)} detections -> "
          f"{street_axis_count} vertical axes after Stage 1")


# ---- Stage 2: Ray angle + wall-hit filter (metric space) ----
#
# Each detection gets its own ray angle via compute_ray_angle().
# Detections whose ray does not strike any building edge are rejected.

accepted_detections = []

for axis in all_axes:
    pid      = axis["panoId"]
    pmeta    = _all_panos_meta.get(pid)
    det0     = axis["members"][0]
    pano_lat = det0["lat"]
    pano_lon = det0["lon"]
    cpx, cpy = to_proj.transform(pano_lon, pano_lat)

    for det in axis["members"]:
        # Use metadata.json pano record if available; else fall back to det fields
        meta_src  = pmeta if pmeta is not None else det
        ray_angle = compute_ray_angle(det["cx"], meta_src)

        hit_pt, hit_bid, hit_ei = _find_hit_edge(cpx, cpy, ray_angle)

        if hit_pt is None:
            det["_accepted"]  = False
            det["_hit_pt"]    = None
            det["_ray_angle"] = ray_angle
            det["_cpx"]       = cpx
            det["_cpy"]       = cpy
            continue

        det["_accepted"]  = True
        det["_hit_pt"]    = (hit_pt.x, hit_pt.y)
        det["_ray_angle"] = ray_angle
        det["_cpx"]       = cpx
        det["_cpy"]       = cpy
        det["_hit_bid"]   = hit_bid
        det["_hit_ei"]    = hit_ei

        accepted_detections.append(det)

n_rejected_stage2 = sum(
    1 for ax in all_axes for det in ax["members"]
    if not det.get("_accepted", False)
)
print(f"\nStage 2: {len(accepted_detections)} accepted, "
      f"{n_rejected_stage2} rejected (no wall hit)")


# ---- Stage 3: hit_pt deduplication (metric space, global) ----
#
# Two detections are duplicates when:
#   |hit_pt distance| < hit_dedup_m  AND  |delta_cy| < floor_band
# Sorted by confidence descending — greedy pass keeps the best detection.

accepted_detections.sort(key=lambda d: d["confidence"], reverse=True)

final_detections = []
suppressed_ids   = set()

for i, det_i in enumerate(accepted_detections):
    if i in suppressed_ids:
        continue
    final_detections.append(det_i)
    hx_i, hy_i = det_i["_hit_pt"]
    pt_i        = _Pt_agg(hx_i, hy_i)

    for j in range(i + 1, len(accepted_detections)):
        if j in suppressed_ids:
            continue
        det_j       = accepted_detections[j]
        hx_j, hy_j = det_j["_hit_pt"]

        if pt_i.distance(_Pt_agg(hx_j, hy_j)) >= args.hit_dedup_m:
            continue
        if abs(det_i["cy"] - det_j["cy"]) >= args.floor_band:
            continue
        suppressed_ids.add(j)

print(f"Stage 3: {len(accepted_detections)} accepted -> "
      f"{len(final_detections)} after hit_pt deduplication "
      f"({len(accepted_detections) - len(final_detections)} duplicates removed)")


# --------------------------------------------------
# Accepted detection lookup for annotation (exact bbox key)
# --------------------------------------------------

_accepted_det_ids = set()
for det in final_detections:
    _accepted_det_ids.add((det["panoId"], det["_bbox"]))


# --------------------------------------------------
# panoId -> street map
# --------------------------------------------------

_pano_to_street = {}
for street_name, detections in all_street_detections.items():
    for det in detections:
        _pano_to_street[det["panoId"]] = street_name


# --------------------------------------------------
# Edge reclassification + results assembly
# --------------------------------------------------

results_by_street = {}

for det in final_detections:
    det["_street"] = _pano_to_street.get(det["panoId"], "unknown")
    hit_key = (det["_hit_bid"], det["_hit_ei"])
    if edge_classification.get(hit_key) == "inner":
        edge_classification[hit_key] = "outer"
        print(f"  Reclassified edge {hit_key} inner->outer "
              f"(window ray hit, class={det['class']})")

for det in final_detections:
    sname  = det["_street"]
    hit_pt = det["_hit_pt"]
    raw_bid = det["_hit_bid"]
    results_by_street.setdefault(sname, []).append({
        "class":           det["class"],
        "confidence_mean": round(det["confidence"], 4),
        "n_votes":         1,
        "building_id":     _bid_label_map.get(raw_bid, str(raw_bid)),
        "hit_pt":          (round(hit_pt[0], 3), round(hit_pt[1], 3)),
    })

for street_name in all_street_detections:
    results_by_street.setdefault(street_name, [])

for street_name, windows in results_by_street.items():
    print(f"\n{street_name}: {len(windows)} windows accepted")


# --------------------------------------------------
# Aggregate across all streets
# --------------------------------------------------

class_stats               = {}
kastenfenster_total       = 0
nicht_kastenfenster_total = 0

for street_name, windows in results_by_street.items():
    for w in windows:
        cls  = w["class"]
        conf = w["confidence_mean"]
        if cls not in class_stats:
            class_stats[cls] = {"count": 0, "conf_sum": 0.0}
        class_stats[cls]["count"]    += 1
        class_stats[cls]["conf_sum"] += conf
        if "Kastenfenster" in cls:
            kastenfenster_total += 1
        elif cls != "angeschnitten":
            nicht_kastenfenster_total += 1

gesamt = kastenfenster_total + nicht_kastenfenster_total
anteil = (kastenfenster_total / gesamt * 100) if gesamt > 0 else 0.0

# --------------------------------------------------
# Console summary
# --------------------------------------------------

print("\n\nFinal window counts (all streets combined)")
print("=" * 65)
print(f"{'Class':<30s} | {'Count':>6} | {'Anteil':>8} | {'Oe Conf':>7}")
print("-" * 65)

total_count = sum(s["count"] for s in class_stats.values())

for cls, stats in sorted(class_stats.items(), key=lambda x: -x[1]["count"]):
    count     = stats["count"]
    mean_conf = stats["conf_sum"] / count
    share     = count / total_count * 100 if total_count else 0.0
    print(f"{cls:<30s} | {count:>6} | {share:>7.1f}% | {mean_conf:>7.3f}")

print("-" * 65)
print(f"{'Kastenfenster total':<30s} | {kastenfenster_total:>6} | {anteil:>7.1f}%")
print(f"{'Nicht-Kastenfenster total':<30s} | {nicht_kastenfenster_total:>6}")
print("=" * 65)

# --------------------------------------------------
# JSON + CSV export
# --------------------------------------------------

def _aggregate_by_building(results_by_street, building_facades, label_map,
                           building_units, bid_to_unit):
    """
    Aggregate detected windows per building and estimate courtyard (inner) windows.
    Uses merged building units for ratio computation so that Vorderhaus buildings
    without their own inner wall get the unit-level ratio
    (inner_m of the whole unit / outer_m of the whole unit).
    """
    # Pre-compute unit-level outer_m and inner_m from building_facades
    facade_by_bid = {b["building_id"]: b for b in building_facades}
    unit_outer_m  = {}   # unit_idx -> total outer_m across all members
    unit_inner_m  = {}   # unit_idx -> total inner_m across all members
    for unit_idx, unit in enumerate(building_units):
        om = sum(float(facade_by_bid[b]["facade_outer_m"])
                 for b in unit["bids"] if b in facade_by_bid)
        im = sum(float(facade_by_bid[b]["facade_inner_m"])
                 for b in unit["bids"] if b in facade_by_bid)
        unit_outer_m[unit_idx] = om
        unit_inner_m[unit_idx] = im

    win_counts = {}   # label -> {class -> count}
    for windows in results_by_street.values():
        for w in windows:
            lbl = w.get("building_id")
            if lbl is None:
                continue
            win_counts.setdefault(lbl, {})
            cls = w["class"]
            win_counts[lbl][cls] = win_counts[lbl].get(cls, 0) + 1

    result = {}
    for b in building_facades:
        bid      = b["building_id"]
        lbl      = label_map.get(bid, str(bid))
        outer_m  = float(b["facade_outer_m"])
        inner_m  = float(b["facade_inner_m"])

        # Use unit-level ratio if this building belongs to a merged unit
        unit_idx = bid_to_unit.get(bid)
        if unit_idx is not None and len(building_units[unit_idx]["bids"]) > 1:
            u_outer = unit_outer_m.get(unit_idx, outer_m)
            u_inner = unit_inner_m.get(unit_idx, inner_m)
            ratio   = (u_inner / u_outer) if u_outer > 0 else None
            merged  = True
        else:
            ratio_raw = b["facade_ratio"]
            if ratio_raw is not None:
                ratio = float(ratio_raw)
                if ratio > 10:   # stored as percentage — divide back
                    ratio = ratio / 100.0
            else:
                ratio = None
            merged = False

        by_class     = win_counts.get(lbl, {})
        total_street = sum(by_class.values())
        wins_per_m   = (total_street / outer_m) if outer_m > 0 else None

        est_inner_by_class   = {}
        grand_total_by_class = {}
        for cls, cnt in by_class.items():
            est = cnt * ratio if ratio is not None else None
            est_inner_by_class[cls]   = round(est, 1) if est is not None else None
            grand_total_by_class[cls] = round(cnt + est, 1) if est is not None else cnt

        total_est_inner = round(sum(v for v in est_inner_by_class.values()
                                    if v is not None), 1)
        total_grand     = round(total_street + total_est_inner, 1)

        result[lbl] = {
            "facade_outer_m":          round(outer_m, 2),
            "facade_inner_m":          round(inner_m, 2),
            "facade_ratio":            round(ratio, 3) if ratio is not None else None,
            "ratio_source":            "unit_merged" if merged else "building",
            "street_window_count":     total_street,
            "street_by_class":         by_class,
            "windows_per_outer_m":     round(wins_per_m, 4) if wins_per_m else None,
            "est_inner_total":         total_est_inner,
            "est_inner_by_class":      est_inner_by_class,
            "grand_total":             total_grand,
            "grand_total_by_class":    grand_total_by_class,
        }
    return result

by_building = _aggregate_by_building(
    results_by_street, building_facades, _bid_label_map,
    building_units, _bid_to_unit,
)

# Block-level summary: sum street + estimated inner across all buildings
_all_classes  = sorted({cls for b in by_building.values()
                         for cls in b["street_by_class"]})
_block_street = {cls: sum(b["street_by_class"].get(cls, 0)
                           for b in by_building.values())
                  for cls in _all_classes}
_block_inner  = {cls: round(sum(b["est_inner_by_class"].get(cls) or 0
                                 for b in by_building.values()), 1)
                  for cls in _all_classes}
_block_grand  = {cls: round(_block_street[cls] + _block_inner[cls], 1)
                  for cls in _all_classes}

json_out = {
    "summary": {
        "kastenfenster_total":       kastenfenster_total,
        "nicht_kastenfenster_total": nicht_kastenfenster_total,
        "kastenfenster_anteil_pct":  round(anteil, 2),
        "by_class": {
            cls: {
                "count":           s["count"],
                "confidence_mean": round(s["conf_sum"] / s["count"], 4),
            }
            for cls, s in class_stats.items()
        },
        "block_totals": {
            "street_detected":    _block_street,
            "est_inner":          _block_inner,
            "grand_total":        _block_grand,
        },
    },
    "by_street": {
        street: [
            {
                "class":           w["class"],
                "confidence_mean": w["confidence_mean"],
                "n_votes":         w["n_votes"],
                "building_id":     w.get("building_id"),
            }
            for w in windows
        ]
        for street, windows in results_by_street.items()
    },
    "by_building": by_building,
}

json_path = os.path.join(args.samples_dir, "results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(json_out, f, indent=2, ensure_ascii=False)
print(f"\nResults saved -> {json_path}")

csv_path = os.path.join(args.samples_dir, "results.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    # ---- Section 1: Class overview (street detections) ----
    writer.writerow(["=== STRASSENFENSTER — Klassenübersicht ==="])
    writer.writerow(["Klasse", "Anzahl", "Anteil_%", "Ø_Konfidenz"])
    for cls, stats in sorted(class_stats.items(), key=lambda x: -x[1]["count"]):
        count     = stats["count"]
        mean_conf = stats["conf_sum"] / count
        share     = count / total_count * 100 if total_count else 0.0
        writer.writerow([cls, count, round(share, 2), round(mean_conf, 4)])
    writer.writerow([])
    writer.writerow(["Kastenfenster gesamt",       kastenfenster_total,       round(anteil, 2), ""])
    writer.writerow(["Nicht-Kastenfenster gesamt", nicht_kastenfenster_total, "", ""])
    writer.writerow([])

    # ---- Section 2: Per-building breakdown ----
    writer.writerow(["=== GEBÄUDEWEISE AUSWERTUNG ==="])

    # Collect all class names that appear anywhere
    all_cls_sorted = sorted(_all_classes)

    # Header: fixed columns + one triple per class
    header_fixed = [
        "Gebäude_ID",
        "Außenfassade_m", "Hofseite_m", "Verhältnis", "Verhältnis_Quelle",
        "Straße_gesamt", "Hof_geschätzt", "Gesamt",
    ]
    header_cls = []
    for cls in all_cls_sorted:
        short = cls[:18]
        header_cls += [f"Str_{short}", f"Hof_{short}", f"Ges_{short}"]
    writer.writerow(header_fixed + header_cls)

    for bid, bd in sorted(by_building.items()):
        row_fixed = [
            bid,
            bd["facade_outer_m"],
            bd["facade_inner_m"],
            bd["facade_ratio"] if bd["facade_ratio"] is not None else "",
            bd.get("ratio_source", ""),
            bd["street_window_count"],
            bd["est_inner_total"],
            bd["grand_total"],
        ]
        row_cls = []
        for cls in all_cls_sorted:
            str_cnt = bd["street_by_class"].get(cls, 0)
            hof_cnt = bd["est_inner_by_class"].get(cls, "")
            ges_cnt = bd["grand_total_by_class"].get(cls, "")
            row_cls += [str_cnt, hof_cnt, ges_cnt]
        writer.writerow(row_fixed + row_cls)

    writer.writerow([])

    # ---- Section 3: Block-level totals ----
    writer.writerow(["=== BLOCK-GESAMTBILANZ ==="])
    writer.writerow(["Kategorie"] + all_cls_sorted + ["Gesamt"])

    row_str   = ["Straße erkannt"]   + [_block_street.get(c, 0) for c in all_cls_sorted] \
                + [sum(_block_street.values())]
    row_inner = ["Hof geschätzt"]    + [_block_inner.get(c, 0)  for c in all_cls_sorted] \
                + [round(sum(_block_inner.values()), 1)]
    row_grand = ["Gesamt (Str+Hof)"] + [_block_grand.get(c, 0)  for c in all_cls_sorted] \
                + [round(sum(_block_grand.values()), 1)]
    writer.writerow(row_str)
    writer.writerow(row_inner)
    writer.writerow(row_grand)

print(f"Table saved      -> {csv_path}")

# --------------------------------------------------
# Floor-plan visualization
# --------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    STREET_COLORS = ["#e67e22", "#8e44ad", "#2980b9", "#16a085",
                     "#e74c3c", "#1abc9c", "#f39c12", "#7f8c8d"]
    SNAP_COLOR = "#e91e63"
    EDGE_COLOR = {"outer": "#2ecc71", "inner": "#e74c3c",
                  "party": "#e67e22", "unknown": "#dddddd"}
    EDGE_WIDTH = {"outer": 3.0, "inner": 3.0, "party": 2.0, "unknown": 0.8}
    EDGE_STYLE = {"outer": "-",  "inner": "--", "party": ":",  "unknown": "-."}

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect("equal")
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")

    from shapely.geometry import LineString as _LS, Point as _Pt
    from shapely.ops import unary_union as _uu

    _EDGE_MIN = 0.5

    street_panos_map = {}
    for mf in meta_files:
        _m = _meta_cache[mf]
        street_panos_map[_m["street"]] = [p for p in _m["panos"] if not p["skip"]]

    if block_buildings_geom:
        _block_hull = _uu(list(block_buildings_geom.values())).convex_hull
        bx, by = _block_hull.exterior.xy
        ax.fill(bx, by, alpha=0.08, color="steelblue", zorder=0)
        ax.plot(bx, by, color="steelblue", linewidth=1.2, linestyle="--", zorder=1)

    for bid, geom in block_buildings_geom.items():
        gx, gy = geom.exterior.xy
        ax.fill(gx, gy, alpha=0.15, color="tan", zorder=1)
        coords = list(geom.exterior.coords)
        for i in range(len(coords) - 1):
            seg = _LS([coords[i], coords[i+1]])
            if seg.length < _EDGE_MIN:
                continue
            label = edge_classification.get((bid, i), "unknown")
            xs, ys = seg.xy
            ax.plot(xs, ys, color=EDGE_COLOR[label],
                    linewidth=EDGE_WIDTH[label],
                    linestyle=EDGE_STYLE[label], zorder=3)

    ax.plot(centroid_x, centroid_y, "b+", markersize=14, markeredgewidth=2, zorder=10)

    legend_handles = []
    RAY_EXTEND_M   = 50.0

    for s_idx, (street_name, _dets) in enumerate(all_street_detections.items()):
        color      = STREET_COLORS[s_idx % len(STREET_COLORS)]
        panos_meta = {p["panoId"]: p for p in street_panos_map.get(street_name, [])}

        for pano in panos_meta.values():
            px      = pano["x_25833"]
            py      = pano["y_25833"]
            normal  = pano["inner_normal"]
            dist    = pano["dist_to_facade"]
            r_start = pano.get("ray_toward_start")
            r_end   = pano.get("ray_toward_end")

            is_edge  = pano.get("is_start_edge") or pano.get("is_end_edge")
            dot_size = 9 if is_edge else 6
            ax.plot(px, py, "o", color=color, markersize=dot_size, zorder=6,
                    markeredgewidth=0.5, markeredgecolor="white")

            # Fan rays — sample20: use absolute bearings directly
            if r_start is not None and r_end is not None:
                for bear_deg, snap_flag, corner_key in [
                    (r_start, pano.get("start_snapped", False), "snap_start_corner"),
                    (r_end,   pano.get("end_snapped",   False), "snap_end_corner"),
                ]:
                    rad   = math.radians(bear_deg)
                    end_x = px + math.sin(rad) * dist
                    end_y = py + math.cos(rad) * dist
                    lw    = 2.0 if snap_flag else 1.2
                    alpha = 0.85 if snap_flag else 0.35
                    lc    = SNAP_COLOR if snap_flag else color
                    ax.plot([px, end_x], [py, end_y],
                            color=lc, linewidth=lw, alpha=alpha, zorder=5)

                    if snap_flag:
                        sc = pano.get(corner_key)
                        if sc is not None:
                            scx = sc[0] if isinstance(sc, (list, tuple)) else sc.x
                            scy = sc[1] if isinstance(sc, (list, tuple)) else sc.y
                            ax.plot([end_x, scx], [end_y, scy],
                                    color=SNAP_COLOR, linewidth=1.0,
                                    linestyle=":", alpha=0.7, zorder=5)
            else:
                # Legacy sample15 fan rays
                ang_prev = pano.get("ang_prev") or 0.0
                ang_next = pano.get("ang_next") or 0.0
                for aoff in (-ang_prev, ang_next):
                    rad = math.radians(normal + aoff)
                    ax.plot([px, px + math.sin(rad) * dist],
                            [py, py + math.cos(rad) * dist],
                            color=color, linewidth=1.2, alpha=0.35, zorder=5)

            # Inner-normal guide ray
            rad = math.radians(normal)
            ax.plot([px, px + math.sin(rad) * dist],
                    [py, py + math.cos(rad) * dist],
                    color=color, linewidth=0.8, alpha=0.25,
                    linestyle="--", zorder=5)

            ax.annotate(pano["panoId"][:6], xy=(px, py), xytext=(3, 3),
                        textcoords="offset points", fontsize=5,
                        color=color, zorder=7, clip_on=True)

        legend_handles.append(
            Line2D([0], [0], marker="o", color=color, markerfacecolor=color,
                   markeredgecolor="white", markersize=7, label=street_name,
                   linewidth=1.2)
        )

        # Accepted window rays
        for det in final_detections:
            if det.get("_street") != street_name:
                continue
            cpx       = det["_cpx"]
            cpy       = det["_cpy"]
            ray_angle = det["_ray_angle"]
            hit_pt    = det["_hit_pt"]
            is_kf     = "Kastenfenster" in det["class"]
            if hit_pt is not None:
                hx, hy = hit_pt
                ax.plot([cpx, hx], [cpy, hy],
                        color=color, linewidth=0.8, alpha=0.6, zorder=7)
                ax.plot(hx, hy,
                        marker="*" if is_kf else "s",
                        color=color,
                        markeredgecolor="#111111", markeredgewidth=0.5,
                        markersize=11 if is_kf else 6,
                        zorder=9, alpha=0.95)

        # Rejected rays (wall-hit failures)
        for ax_entry in all_axes:
            if _pano_to_street.get(ax_entry["panoId"]) != street_name:
                continue
            for det in ax_entry["members"]:
                if det.get("_accepted", False):
                    continue
                cpx       = det.get("_cpx", 0.0)
                cpy       = det.get("_cpy", 0.0)
                ray_angle = det.get("_ray_angle", 0.0)
                ax.plot([cpx, cpx + math.sin(ray_angle) * RAY_EXTEND_M],
                        [cpy, cpy + math.cos(ray_angle) * RAY_EXTEND_M],
                        color="#dd2222", linewidth=0.7, alpha=0.5,
                        linestyle="--", zorder=6)

    legend_handles += [
        Line2D([0], [0], color=EDGE_COLOR["outer"], linewidth=3.0, linestyle="-",
               label="Aussenwand (strassenseitig)"),
        Line2D([0], [0], color=EDGE_COLOR["inner"], linewidth=3.0, linestyle="--",
               label="Innenwand (hofseitig)"),
        Line2D([0], [0], color=EDGE_COLOR["party"], linewidth=2.0, linestyle=":",
               label="Brandwand (Nachbargebaeude)"),
        Line2D([0], [0], color="steelblue", linewidth=1.2,
               linestyle="--", label="Block-Polygon"),
        Line2D([0], [0], color=SNAP_COLOR, linewidth=2.0,
               label="Ecken-Snap (Eckpano)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#888888",
               markeredgecolor="#111111", markersize=12, label="Kastenfenster"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#888888",
               markeredgecolor="#111111", markersize=7,  label="Nicht-Kastenfenster"),
        Line2D([0], [0], color="#dd2222", linewidth=1.0, linestyle="--",
               label="Kein Wandtreffer (ausgeschlossen)"),
    ]

    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              framealpha=0.9, edgecolor="#AAAAAA")
    ax.set_title("Grundriss - erkannte Fenster nach Inferenz", fontsize=13, pad=12)
    ax.set_xlabel("Easting (m, EPSG:25833)")
    ax.set_ylabel("Northing (m, EPSG:25833)")
    ax.ticklabel_format(style="plain", useOffset=False)
    plt.tight_layout()

    fp_path = os.path.join(args.samples_dir, "facade_inference.png")
    fig.savefig(fp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFloor-plan visualization saved -> {fp_path}")

    # ------------------------------------------------------------------
    # Separate twin-window figure
    # ------------------------------------------------------------------
    # A clean second plot showing only:
    #   - Building footprints with classified edges (same as main)
    #   - Accepted street-side window markers (street-colour, filled)
    #   - Orthogonal twin rays and estimated courtyard markers (grey)
    # Rejected rays and pano fans are omitted to keep it readable.

    TWIN_COLOR = "#444444"

    def _inward_normal_deg_twin(seg, building_geom):
        dx = seg.coords[-1][0] - seg.coords[0][0]
        dy = seg.coords[-1][1] - seg.coords[0][1]
        eb   = (math.degrees(math.atan2(dx, dy)) + 360) % 360
        n1   = (eb + 90) % 360
        n2   = (eb - 90) % 360
        midp = seg.interpolate(0.5, normalized=True)
        ctr  = building_geom.centroid
        d1 = _Pt(midp.x + math.sin(math.radians(n1)),
                 midp.y + math.cos(math.radians(n1))).distance(ctr)
        d2 = _Pt(midp.x + math.sin(math.radians(n2)),
                 midp.y + math.cos(math.radians(n2))).distance(ctr)
        return n1 if d1 <= d2 else n2

    fig2, ax2 = plt.subplots(figsize=(14, 14))
    ax2.set_aspect("equal")
    ax2.set_facecolor("#ffffff")
    fig2.patch.set_facecolor("#ffffff")

    # Align with sample24_3 colour scheme
    EDGE_COLOR2 = {"outer": "#2ecc71", "inner": "#e74c3c",
                   "party": "#e67e22", "unknown": "#dddddd"}
    EDGE_WIDTH2 = {"outer": 3.0, "inner": 3.0, "party": 2.0, "unknown": 0.8}
    EDGE_STYLE2 = {"outer": "-",  "inner": "--", "party": ":",  "unknown": "-."}

    # Block hull
    if block_buildings_geom:
        _block_hull2 = _uu(list(block_buildings_geom.values())).convex_hull
        bx2, by2 = _block_hull2.exterior.xy
        ax2.fill(bx2, by2, alpha=0.06, color="steelblue", zorder=0)
        ax2.plot(bx2, by2, color="steelblue", linewidth=1.0,
                 linestyle="--", zorder=1)

    # Building footprints + classified edges
    for bid2, geom2 in block_buildings_geom.items():
        gx2, gy2 = geom2.exterior.xy
        ax2.fill(gx2, gy2, alpha=0.12, color="tan", zorder=1)
        coords2 = list(geom2.exterior.coords)
        for i2 in range(len(coords2) - 1):
            seg2 = _LS([coords2[i2], coords2[i2 + 1]])
            if seg2.length < 0.5:
                continue
            lbl2 = edge_classification.get((bid2, i2), "unknown")
            xs2, ys2 = seg2.xy
            ax2.plot(xs2, ys2,
                     color=EDGE_COLOR2[lbl2],
                     linewidth=EDGE_WIDTH2[lbl2],
                     linestyle=EDGE_STYLE2[lbl2], zorder=3)

    # Building labels on footprint centroids
    for bid2, geom2 in block_buildings_geom.items():
        ctr2  = geom2.centroid
        lbl2  = _bid_label_map.get(bid2, str(bid2))
        ax2.annotate(lbl2, xy=(ctr2.x, ctr2.y),
                     ha="center", va="center",
                     fontsize=6, color="#333333",
                     zorder=4, clip_on=True)

    ax2.plot(centroid_x, centroid_y, "b+",
             markersize=14, markeredgewidth=2, zorder=10)

    TWIN_RAY_M2 = 60.0
    twin_legend  = []

    for s_idx2, street_name2 in enumerate(all_street_detections):
        color2 = STREET_COLORS[s_idx2 % len(STREET_COLORS)]

        for det in final_detections:
            if det.get("_street") != street_name2:
                continue
            hit_pt  = det.get("_hit_pt")
            hit_bid = det.get("_hit_bid")
            hit_ei  = det.get("_hit_ei")
            if hit_pt is None or hit_bid is None or hit_ei is None:
                continue

            hx, hy  = hit_pt
            is_kf   = "Kastenfenster" in det["class"]

            # Street-side window (filled, street colour)
            ax2.plot(hx, hy,
                     marker="*" if is_kf else "s",
                     color=color2,
                     markeredgecolor="#111111", markeredgewidth=0.5,
                     markersize=11 if is_kf else 6,
                     zorder=9, alpha=0.95)

            hit_seg  = edge_segments_map.get((hit_bid, hit_ei))
            hit_geom = block_buildings_geom.get(hit_bid)
            if hit_seg is None or hit_geom is None:
                continue

            inward_rad2 = math.radians(
                _inward_normal_deg_twin(hit_seg, hit_geom))
            twin_ray2 = _LS([
                (hx, hy),
                (hx + math.sin(inward_rad2) * TWIN_RAY_M2,
                 hy + math.cos(inward_rad2) * TWIN_RAY_M2),
            ])
            twin_ray2_cl = twin_ray2.difference(_Pt(hx, hy).buffer(0.3))

            # Collect candidate inner edges from the WHOLE building unit
            # (not just the hit building). This ensures that Vorderhaus
            # windows project all the way to the Hinterhaus inner wall.
            unit_idx2    = _bid_to_unit.get(hit_bid)
            unit2        = building_units[unit_idx2] if unit_idx2 is not None else None
            unit_bids2   = unit2["bids"] if unit2 else [hit_bid]
            unit_inner2  = unit2["inner_edges"] if unit2 else []

            # Build candidate edge list: inner edges of unit + party edges
            # of the hit building (to cross intermediate walls if needed)
            candidate_edges = []
            for (b_c, ei_c) in unit_inner2:
                seg_c = edge_segments_map.get((b_c, ei_c))
                if seg_c and seg_c.length >= 0.5:
                    candidate_edges.append(((b_c, ei_c), seg_c, "inner"))
            # Also add party edges of non-hit unit members as fallback
            for b_c in unit_bids2:
                if b_c == hit_bid:
                    continue
                geom_c  = block_buildings_geom.get(b_c)
                if geom_c is None:
                    continue
                coords_c = list(geom_c.exterior.coords)
                for ei_c in range(len(coords_c) - 1):
                    lbl_c = edge_classification.get((b_c, ei_c), "unknown")
                    if lbl_c == "party":
                        seg_c = edge_segments_map.get((b_c, ei_c))
                        if seg_c and seg_c.length >= 0.5:
                            candidate_edges.append(((b_c, ei_c), seg_c, "party"))

            twin_pt2   = None
            twin_dist2 = float("inf")

            for (key_c, seg_c, _lbl_c) in candidate_edges:
                if key_c == (hit_bid, hit_ei):
                    continue
                inter_b2 = twin_ray2_cl.intersection(seg_c)
                if inter_b2.is_empty:
                    continue
                cand2 = (inter_b2 if inter_b2.geom_type == "Point"
                         else inter_b2.interpolate(0.5, normalized=True))
                d2_ = _Pt(hx, hy).distance(cand2)
                if d2_ < twin_dist2:
                    twin_dist2 = d2_
                    twin_pt2   = cand2

            if twin_pt2 is None:
                continue

            tx2, ty2 = twin_pt2.x, twin_pt2.y

            # Orthogonal dashed line
            ax2.plot([hx, tx2], [hy, ty2],
                     color=TWIN_COLOR, linewidth=1.0, linestyle="--",
                     alpha=0.6, zorder=8)
            # Estimated courtyard marker (white-filled)
            ax2.plot(tx2, ty2,
                     marker="*" if is_kf else "s",
                     markerfacecolor="white",
                     markeredgecolor=TWIN_COLOR,
                     markeredgewidth=1.4,
                     markersize=11 if is_kf else 6,
                     zorder=10, alpha=0.9)

        twin_legend.append(
            Line2D([0], [0], marker="o", color=color2,
                   markerfacecolor=color2, markeredgecolor="white",
                   markersize=7, label=street_name2, linewidth=0))

    twin_legend += [
        Line2D([0], [0], color=EDGE_COLOR2["outer"], linewidth=3.0, linestyle="-",
               label="Aussenwand (strassenseitig)"),
        Line2D([0], [0], color=EDGE_COLOR2["inner"], linewidth=3.0, linestyle="--",
               label="Innenwand (hofseitig)"),
        Line2D([0], [0], color=EDGE_COLOR2["party"], linewidth=2.0, linestyle=":",
               label="Brandwand (Nachbargebaeude)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#888888",
               markeredgecolor="#111111", markersize=12,
               label="Strassenfenster (KF)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#888888",
               markeredgecolor="#111111", markersize=7,
               label="Strassenfenster (NKF)"),
        Line2D([0], [0], color=TWIN_COLOR, linewidth=1.0, linestyle="--",
               label="Orthogonale (Hoffenster-Schaetzung)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="white",
               markeredgecolor=TWIN_COLOR, markeredgewidth=1.4,
               markersize=12, label="Geschaetztes Hoffenster (KF)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="white",
               markeredgecolor=TWIN_COLOR, markeredgewidth=1.4,
               markersize=7,  label="Geschaetztes Hoffenster (NKF)"),
    ]

    ax2.legend(handles=twin_legend, loc="upper right", fontsize=8,
               framealpha=0.9, edgecolor="#AAAAAA")
    _merge_note = f"{_n_merged} Baueinheiten zusammengeführt" if _n_merged else "keine Zusammenführung"
    ax2.set_title(f"Grundriss - Strassenfenster & geschätzte Hoffenster [{_merge_note}]",
                  fontsize=12, pad=12)
    ax2.set_xlabel("Easting (m, EPSG:25833)")
    ax2.set_ylabel("Northing (m, EPSG:25833)")
    ax2.ticklabel_format(style="plain", useOffset=False)
    plt.tight_layout()

    tw_path = os.path.join(args.samples_dir, "twin_windows.png")
    fig2.savefig(tw_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Twin-window visualization saved -> {tw_path}")

except Exception as _e:
    import traceback
    print(f"\nWARNING: Floor-plan visualization skipped - {_e}")
    traceback.print_exc()


# --------------------------------------------------
# Deferred annotation — only final accepted detections
# --------------------------------------------------

annotated_dir = os.path.join(args.samples_dir, "annotated")
os.makedirs(annotated_dir, exist_ok=True)

print(f"\nSaving annotated images -> {annotated_dir}/")

for street_name, img_path, pano_id, frame in all_annotated:
    img_h, img_w    = frame.shape[:2]
    annotated_frame = frame.copy()

    for det in all_street_detections.get(street_name, []):
        if det["panoId"] != pano_id:
            continue
        if (pano_id, det["_bbox"]) not in _accepted_det_ids:
            continue

        xmin, ymin, xmax, ymax = det["_bbox"]
        classidx  = det["_classidx"]
        classname = det["class"]
        conf      = det["confidence"]

        color = bbox_colors[classidx % len(bbox_colors)]
        cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), color, 4)
        cv2.putText(
            annotated_frame,
            f"{classname} {conf:.2f}",
            (xmin, max(ymin - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1, color, 2
        )

    safe_street = street_name.replace(" ", "_").replace("/", "-")
    basename    = os.path.splitext(os.path.basename(img_path))[0]
    out_path    = os.path.join(annotated_dir, f"{safe_street}__{basename}.jpg")
    cv2.imwrite(out_path, annotated_frame)

print(f"  {len(all_annotated)} annotated image(s) saved.")

# --------------------------------------------------
# Optional interactive visualization
# --------------------------------------------------

if args.show:
    for street_name, img_path, pano_id, frame in all_annotated:
        cv2.imshow(f"{street_name} - {os.path.basename(img_path)}", frame)
        key = cv2.waitKey(0)
        if key in [ord("q"), ord("Q")]:
            break
    cv2.destroyAllWindows()
