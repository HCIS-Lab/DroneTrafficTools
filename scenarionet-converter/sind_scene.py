#!/usr/bin/env python3
import os
import math
import csv
import argparse
import pandas as pd
import numpy as np
import pickle
import shutil
import copy
from collections import defaultdict

from numpy.linalg import norm
from shapely.geometry import Polygon
from scipy.interpolate import interp1d
import utm

# ScenarioNet's internal imports
try:
    from metadrive.scenario import ScenarioDescription as SD
    from metadrive.type import MetaDriveType
except ImportError:
    raise ImportError("Please install ScenarioNet / MetaDrive environment to use these imports.")

# ===========================================================================
# 1. Basic Utility/Mapping (unchanged)
# ===========================================================================
AGENT_TYPE_MAPPING = {
    'car': MetaDriveType.VEHICLE,
    'bus': MetaDriveType.VEHICLE,
    'truck': MetaDriveType.VEHICLE,
    'pedestrian': MetaDriveType.PEDESTRIAN,
    'bicycle': MetaDriveType.CYCLIST,
    'motorcycle': MetaDriveType.CYCLIST,
    'tricycle': MetaDriveType.CYCLIST,
}

WAYMO_DT = 0.1
WAYMO_SCENARIO_LENGTH = 91
WAYMO_CURRENT_TIME_INDEX = 10

import math

def lonlat_to_local(lat, lon, lat0, lon0):

    # metres per degree of latitude on the WGS-84 ellipsoid
    DEG_TO_M_LAT = 111_132.954

    # metres per degree of longitude varies with latitude
    cos_lat0 = math.cos(math.radians(lat0))
    DEG_TO_M_LON = 111_319.459 * cos_lat0

    dx = (lon - lon0) * DEG_TO_M_LON   # east-west offset
    dy = (lat - lat0) * DEG_TO_M_LAT   # north-south offset
    return dx, dy



def map_osm_to_md_type(osm_type, subtype=None):
    """Comprehensive mapping from OSM type/subtype to MetaDriveType for SinD dataset."""
    if osm_type == 'lanelet':
        return MetaDriveType.LANE_SURFACE_STREET
    elif osm_type == 'line_thin':
        if subtype == 'solid':
            return MetaDriveType.LINE_SOLID_SINGLE_WHITE
        elif subtype == 'dashed':
            return MetaDriveType.LINE_BROKEN_SINGLE_WHITE
        else:
            return MetaDriveType.LINE_SOLID_SINGLE_WHITE
    elif osm_type == 'line_thick':
        if subtype == 'solid':
            return MetaDriveType.LINE_SOLID_DOUBLE_WHITE
        elif subtype == 'dashed':
            return MetaDriveType.LINE_BROKEN_SINGLE_WHITE
        else:
            return MetaDriveType.LINE_SOLID_DOUBLE_WHITE
    elif osm_type == 'guard_rail':
        return MetaDriveType.GUARDRAIL
    elif osm_type == 'zebra':
        return MetaDriveType.CROSSWALK
    elif osm_type == 'zebra_marking':
        # SinD's version of crosswalks
        return MetaDriveType.CROSSWALK
    elif osm_type == 'stop_line':
        # Stop lines are thick road markings
        return MetaDriveType.LINE_SOLID_SINGLE_WHITE
    elif osm_type == 'curbstone':
        return MetaDriveType.BOUNDARY_LINE
    elif osm_type == 'virtual':
        # Virtual lines are usually lane dividers
        return MetaDriveType.LINE_BROKEN_SINGLE_WHITE
    elif osm_type == 'traffic_light':
        # Traffic lights are not map features, skip them
        return None
    elif osm_type == 'origin':
        return None
    else:
        # Unknown types should be skipped, not mapped to LINE_UNKNOWN
        return None

def are_boundaries_aligned(left_coords, right_coords):
    """Check if left/right boundary directions are consistent; reverse if needed."""
    left_dir = np.array(left_coords[-1]) - np.array(left_coords[0])
    right_dir = np.array(right_coords[-1]) - np.array(right_coords[0])
    dot_product = np.dot(left_dir, right_dir)
    return dot_product >= 0

def resample_coords(coords, num_points):
    """Uniformly resample a list of points to `num_points` via linear interpolation."""
    if len(coords) < 2:
        return np.array(coords)
    distance = np.cumsum([0] + [
        norm(np.array(coords[i]) - np.array(coords[i-1]))
        for i in range(1, len(coords))
    ])
    if distance[-1] <= 0:
        return np.array(coords)
    distance /= distance[-1]
    interpolator = interp1d(distance, np.array(coords), axis=0, kind='linear')
    new_distances = np.linspace(0, 1, num_points)
    return interpolator(new_distances)

def compute_continuous_valid_length(valid_array):
    max_length = current_length = 0
    for v in valid_array:
        if v:
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 0
    return max_length


def compute_lane_direction(polyline, eps=1e-6):
    """Compute a 2D unit direction vector for a lane polyline."""
    if polyline is None:
        return None
    pts = np.asarray(polyline)
    if pts.shape[0] < 2:
        return None
    start = pts[0][:2]
    for next_pt in pts[1:]:
        vec = next_pt[:2] - start
        norm = np.linalg.norm(vec)
        if norm > eps:
            return vec / norm
    return None

# ===========================================================================
# 2. SinD Reading & Splitting Data (MODIFIED)
# ===========================================================================
def read_sind_data(sind_dir):
    """
    Reads the five SinD CSV files from a SinD subdirectory.
    Expected files:
      - recoding_metas.csv
      - Veh_smoothed_tracks.csv and Veh_tracks_meta.csv
      - Ped_smoothed_tracks.csv and Ped_tracks_meta.csv
    Returns:
      frame_rate, dt, xUtmOrigin, yUtmOrigin, rec_meta_df, tracks_meta, agents
    """
    rec_meta_path = os.path.join(sind_dir, "recoding_metas.csv")
    rec_meta_df = pd.read_csv(rec_meta_path)
    meta_row = rec_meta_df.iloc[0]
    frame_rate = float(meta_row["Raw frame rate"])  # fixed at 29.97hz in SinD
    dt = 1.0 / frame_rate
    xUtmOrigin, yUtmOrigin = 0.0, 0.0

    # Vehicles
    veh_meta_path = os.path.join(sind_dir, "Veh_tracks_meta.csv")
    veh_meta_df = pd.read_csv(veh_meta_path)
    veh_tracks_meta = {}
    for _, row in veh_meta_df.iterrows():
        track_id = str(row["trackId"])
        veh_tracks_meta[track_id] = {
            "initialFrame": int(row["initialFrame"]),
            "finalFrame": int(row["finalFrame"]),
            "Frame_nums": int(row["Frame_nums"]),
            "length": float(row["length"]),
            "width": float(row["width"]),
            "agent_type": str(row["class"]).strip(),
            "CrossType": row.get("CrossType", ""),
            "Signal_Violation_Behavior": row.get("Signal_Violation_Behavior", "")
        }
    veh_tracks_path = os.path.join(sind_dir, "Veh_smoothed_tracks.csv")
    veh_tracks_df = pd.read_csv(veh_tracks_path)
    veh_agents = {}
    for _, row in veh_tracks_df.iterrows():
        track_id = str(row["track_id"])
        rec = {
            "frame": int(row["frame_id"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "heading": float(row["heading_rad"]),  # already in radians
            "vx": float(row["vx"]),
            "vy": float(row["vy"])
        }
        veh_agents.setdefault(track_id, []).append(rec)

    # Pedestrians
    ped_meta_path = os.path.join(sind_dir, "Ped_tracks_meta.csv")
    ped_meta_df = pd.read_csv(ped_meta_path)
    ped_tracks_meta = {}
    for _, row in ped_meta_df.iterrows():
        track_id = str(row["trackId"])
        ped_tracks_meta[track_id] = {
            "initialFrame": int(row["initialFrame"]),
            "finalFrame": int(row["finalFrame"]),
            "Frame_nums": int(row["Frame_nums"]),
            "agent_type": "pedestrian"
        }
    ped_tracks_path = os.path.join(sind_dir, "Ped_smoothed_tracks.csv")
    ped_tracks_df = pd.read_csv(ped_tracks_path)
    ped_agents = {}
    for _, row in ped_tracks_df.iterrows():
        track_id = str(row["track_id"])
        rec = {
            "frame": int(row["frame_id"]),
            "x": float(row["x"]),
            "y": float(row["y"]),
            "heading": 0.0,
            "vx": float(row["vx"]),
            "vy": float(row["vy"])
        }
        ped_agents.setdefault(track_id, []).append(rec)

    tracks_meta = {}
    agents = {}
    for track_id, meta in veh_tracks_meta.items():
        tracks_meta[track_id] = meta
        agents[track_id] = veh_agents.get(track_id, [])
    for track_id, meta in ped_tracks_meta.items():
        tracks_meta[track_id] = meta
        agents[track_id] = ped_agents.get(track_id, [])

    return frame_rate, dt, xUtmOrigin, yUtmOrigin, rec_meta_df, tracks_meta, agents

import logging
# Setup logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

def process_agents_direct_sind(tracks_meta, agents):
    rows = []
    default_ped_width = 0.3
    default_ped_height = 0.3
    for track_id, records in agents.items():
        records = sorted(records, key=lambda r: r['frame'])
        if track_id in tracks_meta:
            meta = tracks_meta[track_id]
            if meta["agent_type"].lower() == "pedestrian":
                avg_width = default_ped_width
                avg_height = default_ped_height
            else:
                avg_width = meta["width"]
                avg_height = meta["length"]
            agent_type = meta["agent_type"].lower()
        else:
            avg_width, avg_height = 0.0, 0.0
            agent_type = "unknown"
        
        for r in records:

            psi_rad = r['heading'] # Already in radians
            # *** ADD THIS NORMALIZATION ***
            while psi_rad > math.pi:
                psi_rad -= 2 * math.pi
            while psi_rad <= -math.pi:
                psi_rad += 2 * math.pi

            row = {
                'agent_id': track_id,
                'frame_number': r['frame'],
                'agent_type': agent_type,
                'x_position_m': r['x'],
                'y_position_m': r['y'],
                'avg_width_m': avg_width,
                'avg_height_m': avg_height,
                'psi_rad_rad': psi_rad,
                'vx_m_s': r['vx'],
                'vy_m_s': r['vy']
            }
            rows.append(row)
    logging.debug("Processed %d agent rows", len(rows))
    return rows

def split_into_segments(rows, segment_size):
    if not rows:
        return []
    min_frame = min(r['frame_number'] for r in rows)
    max_frame = max(r['frame_number'] for r in rows)
    total_frames = max_frame - min_frame + 1
    num_segments = math.ceil(total_frames / segment_size)
    segments = [[] for _ in range(num_segments)]
    for r in rows:
        seg_index = (r['frame_number'] - min_frame) // segment_size
        seg_index = max(0, min(seg_index, num_segments - 1))
        segments[seg_index].append(r)
    return [seg for seg in segments if seg]


def compute_waymo_frame_offsets(frame_rate, target_dt=WAYMO_DT, scenario_length=WAYMO_SCENARIO_LENGTH):
    offsets = [int(round(i * target_dt * frame_rate)) for i in range(scenario_length)]
    if len(set(offsets)) != len(offsets):
        raise ValueError(f"frame_rate={frame_rate} is too low to sample unique {scenario_length} steps at dt={target_dt}")
    return offsets


def split_rows_into_waymo_segments(rows, frame_rate, scenario_length=WAYMO_SCENARIO_LENGTH, target_dt=WAYMO_DT):
    if not rows:
        return []
    frame_offsets = compute_waymo_frame_offsets(frame_rate, target_dt=target_dt, scenario_length=scenario_length)
    raw_window = frame_offsets[-1] + 1
    min_frame = min(r["frame_number"] for r in rows)
    max_frame = max(r["frame_number"] for r in rows)
    segments = []
    start_frame = min_frame
    while start_frame + raw_window - 1 <= max_frame:
        sampled_frames = [start_frame + offset for offset in frame_offsets]
        sampled_set = set(sampled_frames)
        seg_rows = [r for r in rows if r["frame_number"] in sampled_set]
        if seg_rows:
            segments.append({"rows": seg_rows, "frames": sampled_frames})
        start_frame += raw_window
    return segments

# ===========================================================================
# 3. Map Handling (MODIFIED for SinD)
# ===========================================================================
osm_cache = {}

def parse_osm_map(osm_file, xUtmOrigin, yUtmOrigin):
    """Read OSM, convert nodes to local coords, parse ways/relations to map_features."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(osm_file)
    root = tree.getroot()

    # 1) Build node dict (OSM ID -> local coords)
    nodes = {}
    for node in root.findall('node'):
        node_id = node.attrib['id']
        lat = float(node.attrib['lat'])
        lon = float(node.attrib['lon'])
        local_x, local_y = lonlat_to_local(lat, lon, xUtmOrigin, yUtmOrigin)
        nodes[node_id] = (local_x, local_y)

    # 2) Build ways
    ways = {}
    for way in root.findall('way'):
        way_id = way.attrib['id']
        nd_refs = [nd.attrib['ref'] for nd in way.findall('nd')]
        tags = {tag.attrib['k']: tag.attrib['v'] for tag in way.findall('tag')}
        ways[way_id] = {
            'nd_refs': nd_refs,
            'tags': tags
        }

    # 3) Build relations
    relations = []
    
    for rel in root.findall('relation'):
        rel_id = rel.attrib['id']
        members = []
        for m in rel.findall('member'):
            members.append({
                'type': m.attrib['type'],
                'ref':  m.attrib['ref'],
                'role': m.attrib['role']
            })
        tags = {tag.attrib['k']: tag.attrib['v'] for tag in rel.findall('tag')}
        relations.append({'id': rel_id, 'members': members, 'tags': tags})

    # 4) Convert ways to map_features
    from shapely.geometry import Polygon
    map_features = {}
    for way_id, wdata in ways.items():
        osm_type = wdata['tags'].get('type')
        subtype  = wdata['tags'].get('subtype')
        md_type  = map_osm_to_md_type(osm_type, subtype)
        if md_type is None:
            continue
        coords = [nodes[ref] for ref in wdata['nd_refs'] if ref in nodes]
        if len(coords) < 2:
            continue
            
        coords_array = np.asarray(coords, dtype=float)
        
        # Special handling for crosswalks - they need polygons, not polylines
        if md_type == MetaDriveType.CROSSWALK:
            # For crosswalks, create a rectangular polygon from the line
            if len(coords_array) == 2:
                # Create a rectangle from a 2-point line (pedestrian crossing)
                p1, p2 = coords_array[0], coords_array[1]
                # Calculate perpendicular direction (using only x,y)
                direction = p2[:2] - p1[:2]
                length = np.linalg.norm(direction)
                if length > 0:
                    direction = direction / length
                    # Create perpendicular vector (rotate 90 degrees)
                    perp = np.array([-direction[1], direction[0]])
                    # Make crosswalk 3 meters wide
                    width = 3.0
                    half_width = width / 2.0
                    # Create rectangle corners (2D only for crosswalks)
                    coords_array = np.array([
                        p1[:2] + perp * half_width,
                        p2[:2] + perp * half_width,
                        p2[:2] - perp * half_width,
                        p1[:2] - perp * half_width,
                        p1[:2] + perp * half_width  # Close the polygon
                    ], dtype=np.float32)
            else:
                # For multi-point crosswalks, create a polygon by offsetting the line
                coords_2d = coords_array[:, :2]  # Take only x,y
                
                # Calculate average direction for the line
                total_direction = np.array([0.0, 0.0])
                for i in range(len(coords_2d) - 1):
                    seg_dir = coords_2d[i + 1] - coords_2d[i]
                    seg_len = np.linalg.norm(seg_dir)
                    if seg_len > 0:
                        total_direction += seg_dir / seg_len
                
                if np.linalg.norm(total_direction) > 0:
                    total_direction = total_direction / np.linalg.norm(total_direction)
                    # Create perpendicular vector (rotate 90 degrees)
                    perp = np.array([-total_direction[1], total_direction[0]])
                    
                    # Make crosswalk 3 meters wide
                    width = 3.0
                    half_width = width / 2.0
                    
                    # Create offset lines on both sides
                    left_line = coords_2d + perp * half_width
                    right_line = coords_2d - perp * half_width
                    
                    # Build polygon: left line + reversed right line + close
                    coords_array = np.vstack([
                        left_line,
                        right_line[::-1],  # Reverse right line
                        left_line[0:1]     # Close polygon
                    ]).astype(np.float32)
                else:
                    # Fallback: just ensure polygon is closed and 2D
                    if len(coords_2d) > 2 and not np.array_equal(coords_2d[0], coords_2d[-1]):
                        coords_array = np.vstack([coords_2d, coords_2d[0:1]]).astype(np.float32)
                    else:
                        coords_array = coords_2d.astype(np.float32)
            
            map_features[way_id] = {'type': md_type, 'polygon': coords_array}
        else:
            # All other features use polyline
            map_features[way_id] = {'type': md_type, 'polyline': coords_array}

    # 5) Lanelets from relations
    from shapely.geometry   import LineString, MultiLineString, Polygon
    from shapely.ops        import linemerge

    lane_boundary_refs = {}

    for rel in relations:
        if rel['tags'].get('type') != 'lanelet':
            continue

        # 1) Gather all left/right way-IDs
        left_ids  = [m['ref'] for m in rel['members']
                    if m['type']=='way' and m['role']=='left']
        right_ids = [m['ref'] for m in rel['members']
                    if m['type']=='way' and m['role']=='right']
        if not left_ids or not right_ids:
            continue

        # 2) Build and merge Shapely LineStrings
        left_lines  = [ LineString([nodes[n] for n in ways[w]['nd_refs']])
                        for w in left_ids ]
        right_lines = [ LineString([nodes[n] for n in ways[w]['nd_refs']])
                        for w in right_ids ]
        left_merged  = linemerge(MultiLineString(left_lines))
        right_merged = linemerge(MultiLineString(right_lines))

        # Extract coords (handles both LineString and MultiLineString)
        def extract_coords(geom):
            if geom.geom_type == 'LineString':
                return list(geom.coords)
            # if it's still multiple parts, pick the longest
            parts = list(geom)
            longest = max(parts, key=lambda g: g.length)
            return list(longest.coords)

        left_coords  = extract_coords(left_merged)
        right_coords = extract_coords(right_merged)

        # 3) Align orientation of right to left
        if not are_boundaries_aligned(left_coords, right_coords):
            right_coords.reverse()

        # 4) Resample to same number of points
        num_pts = max(len(left_coords), len(right_coords))
        l_res   = resample_coords(left_coords, num_pts)
        r_res   = resample_coords(right_coords, num_pts)

        # 5) Geometric "is-left-on-left?" test & flip centerline if needed
        center = (l_res + r_res) / 2
        def is_left_on_left(center, left, step=5):
            signs = []
            for i in range(0, len(center)-1, step):
                v     = center[i+1] - center[i]
                l_off = left[i] - center[i]
                signs.append(np.sign(v[0]*l_off[1] - v[1]*l_off[0]))
            return np.mean(signs) > 0

        if not is_left_on_left(center, l_res):
            center = center[::-1]
            l_res  = l_res[::-1]
            r_res  = r_res[::-1]

        # 6) Build the full polygon shell (no gaps)
        shell = np.vstack([
            l_res,
            r_res[::-1],
            l_res[0:1]          # close the loop
        ])
        lane_poly = Polygon(shell)

        # 7) Fix invalid geometry, if any
        if not lane_poly.is_valid:
            lane_poly = lane_poly.buffer(0)

        poly_coords = np.array(lane_poly.exterior.coords)

        # 8) Register the lane feature
        lid = f"{rel['id']}"
        map_features[lid] = {
            'type':            MetaDriveType.LANE_SURFACE_STREET,
            'polyline':        center.astype(np.float32),
            'entry_lanes':     [],
            'exit_lanes':      [],
            'left_neighbor':   [],
            'right_neighbor':  [],
            'speed_limit_kmh': 50.0,
            'interpolating':   True,
            'width':           np.zeros((len(center), 2), dtype=np.float32),
            '_left_boundary_coords': l_res.astype(np.float32),
            '_right_boundary_coords': r_res.astype(np.float32),
            '_left_boundary_ids': [str(w) for w in left_ids],
            '_right_boundary_ids': [str(w) for w in right_ids],
        }
        lane_boundary_refs[lid] = {
            'left': left_ids,
            'right': right_ids
        }

    boundary_to_lanes = defaultdict(list)
    for lane_id, bounds in lane_boundary_refs.items():
        for way_id in bounds['left']:
            boundary_to_lanes[way_id].append((lane_id, 'left'))
        for way_id in bounds['right']:
            boundary_to_lanes[way_id].append((lane_id, 'right'))

    lane_direction_cache = {}
    for lane_id, feat in map_features.items():
        if feat.get('type') == MetaDriveType.LANE_SURFACE_STREET and 'polyline' in feat:
            lane_direction_cache[lane_id] = compute_lane_direction(feat['polyline'])
        else:
            lane_direction_cache[lane_id] = None

    left_neighbor_sets = defaultdict(set)
    right_neighbor_sets = defaultdict(set)

    for lane_id, bounds in lane_boundary_refs.items():
        for way_id in bounds['left']:
            for other_lane, side in boundary_to_lanes.get(way_id, []):
                if other_lane == lane_id or side != 'right':
                    continue
                left_neighbor_sets[lane_id].add(other_lane)
                right_neighbor_sets[other_lane].add(lane_id)
        for way_id in bounds['right']:
            for other_lane, side in boundary_to_lanes.get(way_id, []):
                if other_lane == lane_id or side != 'left':
                    continue
                right_neighbor_sets[lane_id].add(other_lane)
                left_neighbor_sets[other_lane].add(lane_id)

    for lane_id, feat in map_features.items():
        if feat.get('type') != MetaDriveType.LANE_SURFACE_STREET:
            continue
        feat['_left_neighbor_ids'] = sorted(left_neighbor_sets.get(lane_id, []), key=str)
        feat['_right_neighbor_ids'] = sorted(right_neighbor_sets.get(lane_id, []), key=str)

    # 6) Connect lane endpoints
    lane_endpoints = {}
    angle_cos_threshold = math.cos(math.radians(45.0))
    for mf_id, feat in map_features.items():
        if feat['type'] == MetaDriveType.LANE_SURFACE_STREET and 'polyline' in feat:
            pl = feat['polyline']
            if len(pl) >= 2:
                start_vec = pl[1, :2] - pl[0, :2]
                end_vec = pl[-1, :2] - pl[-2, :2]
                start_norm = np.linalg.norm(start_vec)
                end_norm = np.linalg.norm(end_vec)
                lane_endpoints[mf_id] = {
                    'start': pl[0],
                    'end': pl[-1],
                    'start_dir': start_vec / start_norm if start_norm > 1e-6 else None,
                    'end_dir': end_vec / end_norm if end_norm > 1e-6 else None,
                }

    dist_thr = 2.0
    for lane_id, se in lane_endpoints.items():
        start_pt = se['start']
        end_pt   = se['end']
        entry_lanes=[]
        exit_lanes=[]
        for other_id, ose in lane_endpoints.items():
            if other_id == lane_id:
                continue
            if np.linalg.norm(start_pt - ose['end']) < dist_thr:
                aligned = True
                if se.get('start_dir') is not None and ose.get('end_dir') is not None:
                    dot = np.dot(se['start_dir'], ose['end_dir'])
                    aligned = dot > angle_cos_threshold
                if aligned and other_id not in entry_lanes:
                    entry_lanes.append(other_id)
            if np.linalg.norm(end_pt - ose['start']) < dist_thr:
                aligned = True
                if se.get('end_dir') is not None and ose.get('start_dir') is not None:
                    dot = np.dot(se['end_dir'], ose['start_dir'])
                    aligned = dot > angle_cos_threshold
                if aligned and other_id not in exit_lanes:
                    exit_lanes.append(other_id)
        map_features[lane_id]['entry_lanes'] = entry_lanes
        map_features[lane_id]['exit_lanes']  = exit_lanes

    return map_features, (0,0)

def get_sind_map(osm_file, xUtmOrigin, yUtmOrigin):
    if not os.path.exists(osm_file):
        print(f"[WARN] OSM file {osm_file} not found; returning empty map features.")
        return {}, (0, 0)
    map_features, map_center = parse_osm_map(osm_file, xUtmOrigin, yUtmOrigin)
    return map_features, map_center


def build_waymo_boundary_descriptors(boundary_ids, map_features, lane_end_index):
    return [{
        "lane_start_index": "0",
        "lane_end_index": str(lane_end_index),
        "boundary_type": str(map_features.get(boundary_id, {}).get("type", "UNKNOWN")),
        "boundary_feature_id": str(boundary_id),
    } for boundary_id in boundary_ids]

# ===========================================================================
# 4. create_scenario_from_csv (mostly unchanged)
# ===========================================================================

def create_scenario_from_csv(scenario_data, map_features, map_center, scenario_id,
                             dataset_version, xUtmOrigin, yUtmOrigin, frame_rate, sampled_frames):
    scenario = SD()
    scenario[SD.ID] = scenario_id
    scenario[SD.VERSION] = dataset_version
    scenario[SD.METADATA] = {}
    scenario[SD.METADATA][SD.COORDINATE] = "metadrive"
    scenario[SD.METADATA]["dataset"] = "SinD"
    scenario[SD.METADATA]["scenario_id"] = scenario_id
    scenario[SD.METADATA]["metadrive_processed"] = False
    scenario[SD.METADATA]['id'] = scenario_id
    scenario_map = copy.deepcopy(map_features)
    frames = list(sampled_frames)
    num_frames = len(frames)
    scenario[SD.LENGTH] = num_frames
    scenario[SD.METADATA][SD.TIMESTEP] = np.arange(num_frames, dtype=np.float32) * np.float32(WAYMO_DT)
    scenario[SD.METADATA]['ts'] = scenario[SD.METADATA][SD.TIMESTEP]
    frame_to_idx = {f: i for i, f in enumerate(frames)}

    lane_boundary_refs = {}
    left_neighbor_sets = defaultdict(set)
    right_neighbor_sets = defaultdict(set)
    for feat_id, feat in scenario_map.items():
        if feat.get("type") != MetaDriveType.LANE_SURFACE_STREET or "polyline" not in feat:
            continue
        lane_boundary_refs[feat_id] = {
            "left": list(feat.pop("_left_boundary_ids", [])),
            "right": list(feat.pop("_right_boundary_ids", [])),
        }
        left_neighbor_sets[feat_id].update(feat.pop("_left_neighbor_ids", []))
        right_neighbor_sets[feat_id].update(feat.pop("_right_neighbor_ids", []))
        polyline = feat["polyline"]
        if polyline.shape[-1] == 2:
            polyline = np.concatenate([polyline.astype(np.float32), np.zeros((polyline.shape[0], 1), dtype=np.float32)], axis=1)
            feat["polyline"] = polyline
        lane_end_index = len(polyline) - 1
        feat["left_boundaries"] = build_waymo_boundary_descriptors(lane_boundary_refs[feat_id]["left"], scenario_map, lane_end_index)
        feat["right_boundaries"] = build_waymo_boundary_descriptors(lane_boundary_refs[feat_id]["right"], scenario_map, lane_end_index)
        feat["left_neighbor"] = [{
            "feature_id": str(nid),
            "self_start_index": "0",
            "self_end_index": str(lane_end_index),
            "neighbor_start_index": "0",
            "neighbor_end_index": str(len(scenario_map[nid]["polyline"]) - 1),
            "boundaries": [{"lane_start_index": "0", "lane_end_index": str(lane_end_index), "boundary_type": "UNKNOWN", "boundary_feature_id": "0"}],
        } for nid in sorted(left_neighbor_sets[feat_id], key=str) if nid in scenario_map and "polyline" in scenario_map[nid]]
        feat["right_neighbor"] = [{
            "feature_id": str(nid),
            "self_start_index": "0",
            "self_end_index": str(lane_end_index),
            "neighbor_start_index": "0",
            "neighbor_end_index": str(len(scenario_map[nid]["polyline"]) - 1),
            "boundaries": [{"lane_start_index": "0", "lane_end_index": str(lane_end_index), "boundary_type": "UNKNOWN", "boundary_feature_id": "0"}],
        } for nid in sorted(right_neighbor_sets[feat_id], key=str) if nid in scenario_map and "polyline" in scenario_map[nid]]
        left_boundary = feat.pop("_left_boundary_coords", None)
        right_boundary = feat.pop("_right_boundary_coords", None)
        center_xy = polyline[:, :2]
        left_width = np.zeros(len(polyline), dtype=np.float32)
        right_width = np.zeros(len(polyline), dtype=np.float32)
        if isinstance(left_boundary, np.ndarray) and left_boundary.shape[0] == len(polyline):
            left_width = np.linalg.norm(center_xy - left_boundary[:, :2], axis=1).astype(np.float32)
        if isinstance(right_boundary, np.ndarray) and right_boundary.shape[0] == len(polyline):
            right_width = np.linalg.norm(right_boundary[:, :2] - center_xy, axis=1).astype(np.float32)
        feat["width"] = np.stack([left_width, right_width], axis=1).astype(np.float32)
        feat["speed_limit_kmh"] = float(feat.get("speed_limit_kmh", 0.0))
        feat["speed_limit_mph"] = float(feat["speed_limit_kmh"] / 1.60934)

    scenario[SD.MAP_FEATURES] = scenario_map

    agent_dict = defaultdict(list)
    agent_types = {}
    for r in scenario_data:
        agent_id = r['agent_id']
        agent_dict[agent_id].append(r)
        agent_types[agent_id] = r['agent_type'].lower()

    scenario[SD.TRACKS] = {}
    object_summary = {}

    for agent_id, recs in agent_dict.items():
        recs = sorted(recs, key=lambda x: int(x['frame_number']))
        positions = np.zeros((num_frames, 3), dtype=np.float32)
        headings = np.zeros(num_frames, dtype=np.float32)
        velocities = np.zeros((num_frames, 2), dtype=np.float32)
        lengths = np.zeros(num_frames, dtype=np.float32)
        widths = np.zeros(num_frames, dtype=np.float32)
        heights = np.zeros(num_frames, dtype=np.float32)
        valid = np.zeros(num_frames, dtype=bool)

        for rec in recs:
            fn = int(rec['frame_number'])
            idx = frame_to_idx[fn]
            positions[idx, 0] = float(rec['x_position_m'])
            positions[idx, 1] = float(rec['y_position_m'])
            headings[idx] = float(rec['psi_rad_rad'])
            velocities[idx, 0] = float(rec['vx_m_s'])
            velocities[idx, 1] = float(rec['vy_m_s'])
            lengths[idx] = float(rec['avg_height_m'])
            widths[idx] = float(rec['avg_width_m'])
            heights[idx] = 1.5
            valid[idx] = True

        raw_type = agent_types[agent_id]
        agent_type = AGENT_TYPE_MAPPING.get(raw_type, MetaDriveType.OTHER)

        if len(positions[valid > 0]) >= 2:
            deltas = np.diff(positions[valid > 0][:, :2], axis=0)
            dist = float(np.sum(np.linalg.norm(deltas, axis=1)))
        else:
            dist = 0.0

        valid_length = int(np.sum(valid))
        cval_len = compute_continuous_valid_length(valid)

        object_summary[agent_id] = {
            'type': str(agent_type),
            'valid_length': valid_length,
            'continuous_valid_length': cval_len,
            'track_length': num_frames,
            'moving_distance': dist,
            'object_id': agent_id
        }
        scenario[SD.TRACKS][agent_id] = {
            SD.TYPE: agent_type,
            SD.STATE: {
                'position': positions,
                'heading': headings,
                'velocity': velocities,
                'length': lengths,
                'width': widths,
                'height': heights,
                'valid': valid,
            },
            SD.METADATA: {
                'track_length': num_frames,
                'type': str(agent_type),
                'object_id': agent_id,
                'dataset': 'SinD',
                'original_id': agent_id
            }
        }
    scenario[SD.METADATA]['object_summary'] = object_summary
    num_summary = {
        "num_objects": len(scenario[SD.TRACKS]),
        "object_types": set(),
        "num_objects_each_type": defaultdict(int),
        "num_moving_objects": 0,
        "num_moving_objects_each_type": defaultdict(int),
        "num_traffic_lights": 0,
        "num_traffic_light_types": set(),
        "num_traffic_light_each_step": {},
        "num_map_features": len(scenario_map),
        "map_height_diff": 0.0,
    }
    for aid in scenario[SD.TRACKS]:
        a_type = scenario[SD.TRACKS][aid][SD.TYPE]
        num_summary["object_types"].add(a_type)
        num_summary["num_objects_each_type"][a_type] += 1
        if object_summary[aid]["moving_distance"] > 0:
            num_summary["num_moving_objects"] += 1
            num_summary["num_moving_objects_each_type"][a_type] += 1
    scenario[SD.METADATA]["number_summary"] = num_summary

    eligible_sdc_ids = [aid for aid, track in scenario[SD.TRACKS].items() if track[SD.STATE]['valid'][WAYMO_CURRENT_TIME_INDEX]]
    fallback_id = max(eligible_sdc_ids if eligible_sdc_ids else object_summary, key=lambda aid: object_summary[aid]['continuous_valid_length'])
    valuable_ids = [fallback_id]
    scenario[SD.METADATA]['current_time_index'] = WAYMO_CURRENT_TIME_INDEX
    scenario[SD.METADATA]['sdc_track_index'] = list(scenario[SD.TRACKS].keys()).index(fallback_id)
    scenario[SD.METADATA]['objects_of_interest'] = []
    scenario[SD.METADATA]['source_file'] = scenario_id
    scenario[SD.METADATA]['track_length'] = num_frames

    # --- build exactly one scenario variant per vehicle in valuable_ids ---
    scenario_variants = []
    for agent_id in valuable_ids:
        sc_copy = copy.deepcopy(scenario)
        sc_copy[SD.METADATA][SD.SDC_ID] = agent_id
        sc_copy[SD.METADATA]['tracks_to_predict'] = {
            agent_id: {
                'track_id': agent_id,
                'object_type': sc_copy[SD.TRACKS][agent_id][SD.TYPE],
                'difficulty': 0,
                'track_index': list(sc_copy[SD.TRACKS].keys()).index(agent_id)
            }
        }
        sc_copy[SD.DYNAMIC_MAP_STATES] = {}
        scenario_variants.append(sc_copy)

    for sc in scenario_variants:
        sdc_id = sc[SD.METADATA][SD.SDC_ID]
        ego = sc[SD.TRACKS][sdc_id][SD.STATE]
        # find first valid ego-frame
        first_i   = int(np.where(ego['valid'] > 0)[0][0])
        origin_xy = ego['position'][first_i, :2]

        # shift all map_features
        for feat in sc[SD.MAP_FEATURES].values():
            for k in ('polyline','polygon'):
                if k in feat and isinstance(feat[k], np.ndarray):
                    feat[k] = feat[k].copy()
                    if feat[k].shape[-1] >= 2:
                        feat[k][:, :2] -= origin_xy

        # shift every track's positions
        for tr in sc[SD.TRACKS].values():
            pts = tr[SD.STATE]['position']   # shape (T,3)
            pts[:, :2] -= origin_xy
            tr[SD.STATE]['position'] = pts

        # now compute the ego's initial heading and build a 2×2 rot matrix
        psi0 = ego['heading'][first_i]
        c, s = math.cos(-psi0), math.sin(-psi0)
        R = np.array([[c, -s],
                      [s,  c]], dtype=float)

        # rotate all map features
        for feat in sc[SD.MAP_FEATURES].values():
            for k in ('polyline','polygon'):
                if k in feat and isinstance(feat[k], np.ndarray):
                    pts = feat[k].copy()
                    if pts.shape[-1] >= 2:
                        pts[:, :2] = (R @ pts[:, :2].T).T
                        feat[k] = pts

        # rotate every object's positions & velocities
        for tr in sc[SD.TRACKS].values():
            P = tr[SD.STATE]['position']   # (T,3)
            V = tr[SD.STATE]['velocity']   # (T,2)
            P[:,:2] = (R @ P[:,:2].T).T
            V      = (R @ V.T).T
            tr[SD.STATE]['position'][:,0] = P[:,0]
            tr[SD.STATE]['position'][:,1] = P[:,1]
            tr[SD.STATE]['velocity'] = V.astype(np.float32)

        # finally, make the ego's starting yaw zero
        for tr in sc[SD.TRACKS].values():
            
            tr[SD.STATE]['heading'] -= psi0

    return scenario_variants


def save_summary_and_mapping(summary_path, mapping_path, summary, mapping):
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    with open(mapping_path, 'wb') as f:
        pickle.dump(mapping, f)

def write_scenarios_to_directory(scenarios, output_dir, dataset_name, dataset_version):
    if os.path.exists(output_dir):
        ans = input(f"Output dir {output_dir} exists. Overwrite? (y/n): ")
        if ans.lower() != 'y':
            print("Aborting.")
            return
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    summary = {}
    mapping = {}
    for sc in scenarios:
        sc_id = sc[SD.ID]
        pkl_name = SD.get_export_file_name(dataset_name, dataset_version, sc_id)
        summary[pkl_name] = sc[SD.METADATA]
        mapping[pkl_name] = ""
        sc_dict = sc.to_dict()
        SD.sanity_check(sc_dict)
        pkl_path = os.path.join(output_dir, pkl_name)
        with open(pkl_path, 'wb') as pf:
            pickle.dump(sc_dict, pf)
    summary_path = os.path.join(output_dir, "dataset_summary.pkl")
    mapping_path = os.path.join(output_dir, "dataset_mapping.pkl")
    save_summary_and_mapping(summary_path, mapping_path, summary, mapping)

# ===========================================================================
# 5. MAIN WRAPPER (MODIFIED for SinD)
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Convert SinD data (in multiple subdirectories) => ScenarioNet format."
    )
    parser.add_argument("--root_dir", required=True,
                        help="Path to SinD dataset root, containing subdirectories (e.g. '8_2_1', '8_2_2', ...).")
    parser.add_argument("--segment_size", type=int, default=271,
                        help="Raw frames per scenario window. Default 271 frames ~= 9.0s at 29.97Hz -> 91 steps at 10Hz.")
    parser.add_argument("--output_dir", default=None,
                        help="Where to put final ScenarioNet PKLs. Default: <root_dir>/converted_scenarios")
    args = parser.parse_args()
    root_dir = args.root_dir
    segment_size = args.segment_size
    output_dir = args.output_dir or os.path.join(root_dir, "converted_scenarios")

    # Determine the shared OSM file path from the SinD root folder
    shared_osm_path = os.path.join(root_dir, "map.osm")
    if os.path.exists(shared_osm_path):
        print(f"Using shared OSM file: {shared_osm_path}")
        # Parse the shared OSM file once (using (0,0) as the origin)
        shared_map_features, shared_map_center = get_sind_map(shared_osm_path, 0.0, 0.0)
    else:
        print("No shared OSM file found; map features will be empty.")
        shared_map_features, shared_map_center = {}, (0, 0)

    # Get all subdirectories containing the SinD CSV files.
    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
               if os.path.isdir(os.path.join(root_dir, d))]
    if not subdirs:
        print(f"[ERROR] No subdirectories found in {root_dir}")
        return

    dataset_name = "SinD"
    dataset_version = "1.0"
    all_scenarios = []

    for sind_dir in subdirs:
        print(f"Processing SinD directory: {sind_dir}")
        try:
            (frame_rate, dt, xUtm, yUtm, rec_meta,
             tracks_meta, agents) = read_sind_data(sind_dir)
        except Exception as e:
            print(f"[WARN] Skipping directory {sind_dir} due to error: {e}")
            continue

        rows = process_agents_direct_sind(tracks_meta, agents)
        print(f"Found {len(rows)} rows in {sind_dir}")

        expected_raw_window = compute_waymo_frame_offsets(frame_rate)[-1] + 1
        if segment_size != expected_raw_window:
            print(
                f"[WARN] segment_size={segment_size} is ignored for Waymo alignment; "
                f"using raw_window={expected_raw_window} from frame_rate={frame_rate}"
            )
        segments = split_rows_into_waymo_segments(rows, frame_rate)
        if not segments:
            continue

        # Use the shared OSM map for every subdirectory.
        map_features, map_center = shared_map_features, shared_map_center

        for i, seg in enumerate(segments, start=1):
            scenario_id = f"{os.path.basename(sind_dir)}_seg{i}"
            scenario_variants = create_scenario_from_csv(
                seg["rows"],
                map_features, map_center,
                scenario_id,
                dataset_version,
                xUtm, yUtm,
                frame_rate,
                seg["frames"],
            )
            for j, variant in enumerate(scenario_variants, start=1):
                variant_id = f"{scenario_id}_ego_{j}"
                variant[SD.ID] = variant_id
                all_scenarios.append(variant)

    if not all_scenarios:
        print("[INFO] No scenarios were produced.")
        return

    write_scenarios_to_directory(all_scenarios, output_dir, dataset_name, dataset_version)
    print(f"[DONE] Wrote {len(all_scenarios)} scenario PKLs into {output_dir}")

if __name__ == "__main__":
    main()
