#!/usr/bin/env python3

import os
import math
import csv
import argparse
import pandas as pd
import numpy as np
import pickle
from lxml import etree
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
# 1. Basic Utility/Mapping
# ===========================================================================

AGENT_TYPE_MAPPING = {
    'car': MetaDriveType.VEHICLE,
    'truck_bus': MetaDriveType.VEHICLE,
    'truck': MetaDriveType.VEHICLE,
    'bus': MetaDriveType.VEHICLE,
    'pedestrian': MetaDriveType.PEDESTRIAN,
    'bicycle': MetaDriveType.CYCLIST,
    'motorcycle': MetaDriveType.CYCLIST,
    'unknown': MetaDriveType.OTHER,  # FIX: Added proper fallback
}

WAYMO_DT = 0.1
WAYMO_SCENARIO_LENGTH = 91
WAYMO_CURRENT_TIME_INDEX = 10

def lonlat_to_local(lat, lon, xUtmOrigin, yUtmOrigin):
    """Convert lat/lon to local meters using UTM offset by (xUtmOrigin, yUtmOrigin)."""
    easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
    local_x = easting - xUtmOrigin
    local_y = northing - yUtmOrigin
    return local_x, local_y

def map_osm_to_md_type(osm_type, subtype=None):
    """Basic mapping from OSM type/subtype to MetaDriveType."""
    if osm_type == 'lanelet':
        return MetaDriveType.LANE_SURFACE_STREET
    elif osm_type == 'line_thin':
        if subtype == 'solid' or subtype == 'solid_solid':
            return MetaDriveType.LINE_SOLID_SINGLE_WHITE
        elif subtype == 'dashed':
            return MetaDriveType.LINE_BROKEN_SINGLE_WHITE
        else:
            return MetaDriveType.LINE_SOLID_SINGLE_WHITE
    elif osm_type == 'line_thick':
        if subtype == 'solid':
            return MetaDriveType.LINE_SOLID_DOUBLE_WHITE
        elif subtype == 'dashed':
            # There's no dashed double line in MetaDrive, use solid double yellow
            return MetaDriveType.LINE_SOLID_DOUBLE_YELLOW
        else:
            return MetaDriveType.LINE_SOLID_DOUBLE_WHITE
    elif osm_type == 'wall':
        return MetaDriveType.BOUNDARY_LINE  # This equals "ROAD_EDGE_BOUNDARY"
    elif osm_type == 'zebra_marking':
        return MetaDriveType.CROSSWALK
    elif osm_type == 'virtual':
        # Virtual lines are usually lane dividers, use broken white line
        return MetaDriveType.LINE_BROKEN_SINGLE_WHITE
    elif osm_type == 'road_border':
        # Physical road borders/edges
        return MetaDriveType.BOUNDARY_LINE  # "ROAD_EDGE_BOUNDARY"
    elif osm_type == 'curbstone':
        # Curbs separating road from sidewalk
        return MetaDriveType.BOUNDARY_LINE  # "ROAD_EDGE_BOUNDARY"
    elif osm_type == 'fence':
        # Fencing around road area
        return MetaDriveType.GUARDRAIL
    elif osm_type == 'arrow':
        # Lane direction markings - skip for map features (lane metadata instead)
        return None
    elif osm_type == 'traffic_sign':
        # Traffic signs - skip for map features
        return None
    elif osm_type == 'de274-30':
        # German traffic sign - skip
        return None
    elif osm_type == 'traffic_light':
        # Traffic lights are not map features in MetaDrive, skip them
        return None
    elif osm_type == 'multipolygon':
        # Buildings and other polygons, skip them
        return None
    elif osm_type == 'regulatory_element':
        # Regulatory elements like signs, skip them
        return None
    else:
        # Default to None for unknown types
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
    """Compute unit direction vector for a lane polyline."""
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
# 2. inD Reading & Splitting Data (NO disk-based segmentation)
# ===========================================================================

def read_ind_data(prefix, data_dir):
    """
    Reads the three inD CSV files for a given prefix ("00", "01") in data_dir.
    Returns:
      - frame_rate, dt
      - xUtmOrigin, yUtmOrigin
      - location_id
      - tracks_meta: dict track_id -> meta info
      - agents: dict track_id -> list of per-frame records
    """
    rec_meta_path    = os.path.join(data_dir, f"{prefix}_recordingMeta.csv")
    tracks_path      = os.path.join(data_dir, f"{prefix}_tracks.csv")
    tracks_meta_path = os.path.join(data_dir, f"{prefix}_tracksMeta.csv")

    rec_meta_df = pd.read_csv(rec_meta_path)
    meta_row = rec_meta_df.iloc[0]

    # Frame rate
    if 'frameRate' in meta_row:
        frame_rate = float(meta_row['frameRate'])
    elif 'frame_rate' in meta_row:
        frame_rate = float(meta_row['frame_rate'])
    elif 'FrameRate' in meta_row:
        frame_rate = float(meta_row['FrameRate'])
    else:
        raise KeyError("Frame rate column not found in recording meta CSV.")

    dt = 1.0 / frame_rate


    xUtmOrigin = float(meta_row['xUtmOrigin'])
    yUtmOrigin = float(meta_row['yUtmOrigin'])

    if 'locationId' not in meta_row:
        raise KeyError("No 'locationId' column found in recordingMeta!")
    location_id = int(meta_row['locationId'])

    # tracksMeta
    tracks_meta_df = pd.read_csv(tracks_meta_path)
    tracks_meta = {}
    for _, row in tracks_meta_df.iterrows():
        track_id = str(int(row['trackId']))
        tracks_meta[track_id] = {
            'initialFrame': int(row['initialFrame']),
            'finalFrame': int(row['finalFrame']),
            'width': float(row['width']),
            'length': float(row['length']),
            'agent_type': str(row['class']).strip()
        }

    # tracks
    tracks_df = pd.read_csv(tracks_path)
    agents = {}
    for _, row in tracks_df.iterrows():
        track_id = str(int(row['trackId']))
        rec = {
            'frame': int(row['frame']),
            'x': float(row['xCenter']),
            'y': float(row['yCenter']),
            'heading': float(row['heading']),  # degrees
            'vx': float(row['xVelocity']),
            'vy': float(row['yVelocity'])
        }
        agents.setdefault(track_id, []).append(rec)

    return frame_rate, dt, xUtmOrigin, yUtmOrigin, location_id, tracks_meta, agents

def process_agents_direct(tracks_meta, agents):
    """
    Convert raw agent data into a flat list of dict rows.
    Each row: {
      'agent_id', 'frame_number', 'agent_type',
      'x_position_m','y_position_m','avg_width_m','avg_height_m',
      'psi_rad_rad','vx_m_s','vy_m_s'
    }
    """
    rows = []
    for track_id, records in agents.items():
        # sort by frame
        records = sorted(records, key=lambda r: r['frame'])
        if track_id in tracks_meta:
            meta = tracks_meta[track_id]
            avg_width = meta['width']
            avg_height = meta['length']
            agent_type = meta['agent_type']
        else:
            avg_width, avg_height = 0.0, 0.0
            agent_type = "unknown"

        for r in records:
            psi_rad = math.radians(r['heading'])
            if psi_rad > math.pi:
                psi_rad -= 2*math.pi
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
    return rows

def split_into_segments(rows, segment_size):
    """
    Segments the in-memory `rows` list into smaller "chunks" by frame range.
    Returns a list of 'segments', each is a list of rows.
    """
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

    # filter out empty segments
    segments = [seg for seg in segments if seg]
    return segments


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
# 3. Parse OSM ONCE per location & Create Scenarios
# ===========================================================================

# We keep a global cache: location_id -> (map_features, map_center)
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
        nodes[node_id] = (local_x, local_y, 0.0)  # FIX: Add z-coordinate

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

    # 3) Build relations and collect lanelet member ways
    relations = []
    lanelet_ways = set()  # Track ways that are part of lanelet relations
    
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
        
        # If this is a lanelet, track its member ways
        if tags.get('type') == 'lanelet':
            for m in members:
                if m['type'] == 'way' and m['role'] in ['left', 'right']:
                    lanelet_ways.add(m['ref'])

    # 4) Process ways that are members of lanelets
    from shapely.geometry import Polygon
    map_features = {}
    
    # Process lanelet member ways
    for way_id in lanelet_ways:
        if way_id not in ways:
            continue
        wdata = ways[way_id]
        osm_type = wdata['tags'].get('type', 'line_thin')  # Default to thin line
        subtype = wdata['tags'].get('subtype', 'solid')
        md_type = map_osm_to_md_type(osm_type, subtype)
        
        if md_type:
            coords = [nodes[ref] for ref in wdata['nd_refs'] if ref in nodes]
            if coords:
                coords_array = np.array(coords, dtype=np.float32)
                
                # Crosswalks should use polygon field
                if md_type == MetaDriveType.CROSSWALK:
                    # For crosswalks, we need to create a rectangular polygon from the line
                    if len(coords_array) == 2:
                        # Create a rectangle from a line (zebra crossing)
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
                        # For crosswalks with more points, ensure polygon is closed and 2D
                        coords_array = coords_array[:, :2]  # Take only x,y
                        if len(coords_array) > 2 and not np.array_equal(coords_array[0], coords_array[-1]):
                            coords_array = np.vstack([coords_array, coords_array[0:1]])
                    feature = {'type': md_type, 'polygon': coords_array}
                else:
                    # All other features use polyline
                    feature = {'type': md_type, 'polyline': coords_array}
                
                map_features[way_id] = feature

    # Process standalone ways with important road types (not part of lanelets)
    important_standalone_types = {
        'road_border', 'curbstone', 'fence', 'wall', 'line_thin', 'line_thick', 'zebra_marking', 'virtual'
    }
    
    for way_id, wdata in ways.items():
        # Skip if already processed as lanelet member
        if way_id in lanelet_ways:
            continue
            
        osm_type = wdata['tags'].get('type')
        if osm_type in important_standalone_types:
            subtype = wdata['tags'].get('subtype', 'solid')
            md_type = map_osm_to_md_type(osm_type, subtype)
            
            if md_type:
                coords = [nodes[ref] for ref in wdata['nd_refs'] if ref in nodes]
                if coords:
                    coords_array = np.array(coords, dtype=np.float32)
                    
                    # Crosswalks should use polygon field
                    if md_type == MetaDriveType.CROSSWALK:
                        # For crosswalks, create rectangular polygon from line
                        if len(coords_array) == 2:
                            p1, p2 = coords_array[0], coords_array[1]
                            direction = p2[:2] - p1[:2]
                            length = np.linalg.norm(direction)
                            if length > 0:
                                direction = direction / length
                                perp = np.array([-direction[1], direction[0]])
                                width = 3.0
                                half_width = width / 2.0
                                coords_array = np.array([
                                    p1[:2] + perp * half_width,
                                    p2[:2] + perp * half_width,
                                    p2[:2] - perp * half_width,
                                    p1[:2] - perp * half_width,
                                    p1[:2] + perp * half_width
                                ], dtype=np.float32)
                        else:
                            coords_array = coords_array[:, :2]
                            if len(coords_array) > 2 and not np.array_equal(coords_array[0], coords_array[-1]):
                                coords_array = np.vstack([coords_array, coords_array[0:1]])
                        feature = {'type': md_type, 'polygon': coords_array}
                    else:
                        # All other features use polyline
                        feature = {'type': md_type, 'polyline': coords_array}
                    
                    map_features[way_id] = feature

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
        left_lines  = []
        right_lines = []
        
        for w in left_ids:
            if w in ways:
                coords = [nodes[n] for n in ways[w]['nd_refs'] if n in nodes]
                if len(coords) >= 2:
                    left_lines.append(LineString(coords))
                    
        for w in right_ids:
            if w in ways:
                coords = [nodes[n] for n in ways[w]['nd_refs'] if n in nodes]
                if len(coords) >= 2:
                    right_lines.append(LineString(coords))
        
        if not left_lines or not right_lines:
            continue
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

        poly_coords = np.array(lane_poly.exterior.coords, dtype=np.float32)

        # 8) Register the lane feature
        lid = f"{rel['id']}"
        map_features[lid] = {
            'type':            'LANE_SURFACE_STREET',
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
        if feat.get('type') == 'LANE_SURFACE_STREET' and 'polyline' in feat:
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
        if feat.get('type') != 'LANE_SURFACE_STREET':
            continue
        feat['_left_neighbor_ids'] = sorted(left_neighbor_sets.get(lane_id, []), key=str)
        feat['_right_neighbor_ids'] = sorted(right_neighbor_sets.get(lane_id, []), key=str)

    # 6) Connect lane endpoints
    lane_endpoints = {}
    angle_cos_threshold = math.cos(math.radians(45.0))
    for mf_id, feat in map_features.items():
        if feat['type'] == 'LANE_SURFACE_STREET' and 'polyline' in feat:
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
        

    
    return map_features, (0, 0)


def get_osm_map_for_location(loc_id, osm_file, xUtmOrigin, yUtmOrigin):
    """
    Caches parse_osm_map() results so we only parse once per location.
    """
    if loc_id not in osm_cache:
        map_features, map_center = parse_osm_map(osm_file, xUtmOrigin, yUtmOrigin)
        osm_cache[loc_id] = (map_features, map_center)
    return osm_cache[loc_id]


def build_waymo_boundary_descriptors(boundary_ids, map_features, lane_end_index):
    descriptors = []
    for boundary_id in boundary_ids:
        boundary_feature = map_features.get(boundary_id, {})
        descriptors.append(
            {
                "lane_start_index": "0",
                "lane_end_index": str(lane_end_index),
                "boundary_type": str(boundary_feature.get("type", "UNKNOWN")),
                "boundary_feature_id": str(boundary_id),
            }
        )
    return descriptors


def build_waymo_neighbor_descriptors(neighbor_ids, lane_id, lane_feat, map_features, lane_boundary_refs, side):
    lane_end_index = len(lane_feat["polyline"]) - 1
    descriptors = []
    for neighbor_id in sorted(neighbor_ids, key=str):
        neighbor_feat = map_features.get(neighbor_id)
        if neighbor_feat is None or "polyline" not in neighbor_feat:
            continue
        neighbor_end_index = len(neighbor_feat["polyline"]) - 1
        lane_bounds = lane_boundary_refs.get(lane_id, {"left": [], "right": []})
        neighbor_bounds = lane_boundary_refs.get(neighbor_id, {"left": [], "right": []})
        if side == "left":
            shared_boundaries = sorted(set(lane_bounds["left"]).intersection(neighbor_bounds["right"]), key=str)
        else:
            shared_boundaries = sorted(set(lane_bounds["right"]).intersection(neighbor_bounds["left"]), key=str)

        boundaries = build_waymo_boundary_descriptors(shared_boundaries, map_features, lane_end_index) if shared_boundaries else [
            {
                "lane_start_index": "0",
                "lane_end_index": str(lane_end_index),
                "boundary_type": "UNKNOWN",
                "boundary_feature_id": "0",
            }
        ]
        descriptors.append(
            {
                "feature_id": str(neighbor_id),
                "self_start_index": "0",
                "self_end_index": str(lane_end_index),
                "neighbor_start_index": "0",
                "neighbor_end_index": str(neighbor_end_index),
                "boundaries": boundaries,
            }
        )
    return descriptors

def create_scenario_from_csv(
    scenario_data,
    map_features,
    map_center,
    scenario_id,
    dataset_version,
    xUtmOrigin,
    yUtmOrigin,
    frame_rate,
    sampled_frames,
    source_file=None
):
    """
    Parse scenario rows => build scenario => produce multiple variants (up to 3 EGO).
    We assume map_features and map_center are provided (already parsed).
    """
    scenario = SD()
    scenario[SD.ID] = scenario_id
    scenario[SD.VERSION] = dataset_version
    scenario[SD.METADATA] = {}
    scenario[SD.METADATA][SD.COORDINATE] = "metadrive"
    scenario[SD.METADATA]["dataset"] = "inD"
    scenario[SD.METADATA]["scenario_id"] = scenario_id
    scenario[SD.METADATA]["metadrive_processed"] = False
    scenario[SD.METADATA]['id'] = scenario_id

    scenario_map = copy.deepcopy(map_features)
    frames = list(sampled_frames)
    num_frames = len(frames)
    scenario[SD.LENGTH] = num_frames
    ts = np.arange(num_frames, dtype=np.float32) * np.float32(WAYMO_DT)
    scenario[SD.METADATA][SD.TIMESTEP] = ts
    scenario[SD.METADATA]['ts'] = ts
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
        feat["left_neighbor"] = build_waymo_neighbor_descriptors(left_neighbor_sets[feat_id], feat_id, feat, scenario_map, lane_boundary_refs, "left")
        feat["right_neighbor"] = build_waymo_neighbor_descriptors(right_neighbor_sets[feat_id], feat_id, feat, scenario_map, lane_boundary_refs, "right")
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
        positions = np.zeros((num_frames,3), dtype=np.float32)
        headings  = np.zeros(num_frames, dtype=np.float32)
        velocities= np.zeros((num_frames,2), dtype=np.float32)  # Keep float32 for consistency
        # FIX: Correct array shapes - no extra dimension
        lengths   = np.zeros(num_frames, dtype=np.float32)
        widths    = np.zeros(num_frames, dtype=np.float32)
        heights   = np.zeros(num_frames, dtype=np.float32)
        valid     = np.zeros(num_frames, dtype=bool)  # FIX: Boolean dtype

        for rec in recs:
            fn  = int(rec['frame_number'])
            idx = frame_to_idx[fn]
            positions[idx,0] = float(rec['x_position_m'])
            positions[idx,1] = float(rec['y_position_m'])
            positions[idx,2] = 0.0  # FIX: Add z-coordinate
            headings[idx]    = float(rec['psi_rad_rad'])
            velocities[idx,0]= float(rec['vx_m_s'])
            velocities[idx,1]= float(rec['vy_m_s'])
            # FIX: Direct assignment without indexing
            lengths[idx]   = float(rec['avg_height_m'])
            widths[idx]    = float(rec['avg_width_m'])
            heights[idx]   = 1.5
            valid[idx]     = True

        raw_type   = agent_types[agent_id]
        # FIX: Use proper fallback
        agent_type = AGENT_TYPE_MAPPING.get(raw_type, MetaDriveType.OTHER)

        # compute total distance traveled
        valid_pos = positions[valid > 0]
        if len(valid_pos) >= 2:
            deltas = np.diff(valid_pos[:,:2], axis=0)
            dist   = float(np.sum(np.linalg.norm(deltas, axis=1)))  # Convert to Python float
        else:
            dist = 0.0

        valid_length = int(np.sum(valid))
        cval_len     = compute_continuous_valid_length(valid)

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
                'length':  lengths,
                'width':   widths,
                'height':  heights,
                'valid':   valid,
            },
            SD.METADATA: {
                'track_length': num_frames,
                'type': str(agent_type),
                'object_id': agent_id,
                'dataset': 'inD',
                'original_id': agent_id
            }
        }

    scenario[SD.METADATA]['object_summary'] = object_summary

    # Summaries
    num_summary = {}
    track_ids = list(scenario[SD.TRACKS].keys())
    num_summary['num_objects'] = len(track_ids)
    num_summary['object_types'] = set()
    num_summary['num_objects_each_type'] = defaultdict(int)
    num_summary['num_moving_objects'] = 0
    num_summary['num_moving_objects_each_type'] = defaultdict(int)
    num_summary['num_traffic_lights'] = 0
    num_summary['num_traffic_light_types'] = set()
    num_summary['num_traffic_light_each_step'] = {}
    num_summary['num_map_features'] = len(scenario_map)
    num_summary['map_height_diff'] = 0.0

    for aid in track_ids:
        a_type = scenario[SD.TRACKS][aid][SD.TYPE]
        num_summary['object_types'].add(a_type)
        num_summary['num_objects_each_type'][a_type] += 1
        dist = object_summary[aid]['moving_distance']
        if dist > 0:
            num_summary['num_moving_objects'] += 1
            num_summary['num_moving_objects_each_type'][a_type] += 1

    scenario[SD.METADATA]['number_summary'] = num_summary

    eligible_sdc_ids = [aid for aid, track in scenario[SD.TRACKS].items() if track[SD.STATE]['valid'][WAYMO_CURRENT_TIME_INDEX]]
    candidate_ids = eligible_sdc_ids if eligible_sdc_ids else list(object_summary.keys())
    fallback_id = max(candidate_ids, key=lambda aid: object_summary[aid]['continuous_valid_length'])
    valuable_ids = [fallback_id]

    scenario[SD.METADATA]['current_time_index'] = WAYMO_CURRENT_TIME_INDEX
    scenario[SD.METADATA]['sdc_track_index'] = list(scenario[SD.TRACKS].keys()).index(fallback_id)
    scenario[SD.METADATA]['objects_of_interest'] = []
    scenario[SD.METADATA]['source_file'] = source_file or "inD_tracks.csv"
    scenario[SD.METADATA]['track_length'] = num_frames

    scenario_variants = []
    for agent_id in valuable_ids:
        sc_copy = copy.deepcopy(scenario)
        sc_copy[SD.METADATA][SD.SDC_ID] = agent_id
        sc_copy[SD.METADATA]['sdc_track_index'] = list(sc_copy[SD.TRACKS].keys()).index(agent_id)
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
        first_i = int(np.where(ego['valid'] > 0)[0][0])
        origin_xy = ego['position'][first_i, :2]

        # shift all map_features
        for feat in sc[SD.MAP_FEATURES].values():
            # Handle simple arrays (polyline and polygon)
            for k in ('polyline', 'polygon'):
                if k in feat and isinstance(feat[k], np.ndarray):
                    feat[k] = feat[k].copy()
                    if k == 'polygon' and feat['type'] == MetaDriveType.CROSSWALK:
                        # Crosswalk polygons are 2D
                        if feat[k].shape[-1] == 2:
                            feat[k] -= origin_xy
                    else:
                        # Other features are 3D
                        if feat[k].shape[-1] >= 2:
                            feat[k][:, :2] -= origin_xy
            
            # Handle list of arrays (boundaries)
            for k in ('left_boundaries', 'right_boundaries'):
                if k in feat and isinstance(feat[k], list):
                    for i, boundary in enumerate(feat[k]):
                        if isinstance(boundary, np.ndarray) and boundary.shape[-1] >= 2:
                            feat[k][i] = boundary.copy()
                            feat[k][i][:, :2] -= origin_xy

        # shift every track's positions
        for tr in sc[SD.TRACKS].values():
            pts = tr[SD.STATE]['position'].copy()   # shape (T,3)
            pts[:, :2] -= origin_xy
            tr[SD.STATE]['position'] = pts

        # now compute the ego's initial heading and build a 2×2 rot matrix
        psi0 = ego['heading'][first_i]
        c, s = math.cos(-psi0), math.sin(-psi0)
        R = np.array([[c, -s],
                      [s,  c]], dtype=float)

        # rotate all map features
        for feat in sc[SD.MAP_FEATURES].values():
            # Handle simple arrays (polyline and polygon)
            for k in ('polyline', 'polygon'):
                if k in feat and isinstance(feat[k], np.ndarray):
                    pts = feat[k].copy()  # already translated
                    if k == 'polygon' and feat['type'] == MetaDriveType.CROSSWALK:
                        # Crosswalk polygons are 2D
                        if pts.shape[-1] == 2:
                            feat[k] = (R @ pts.T).T
                    else:
                        # Other features are 3D
                        if pts.shape[-1] >= 2:
                            pts[:, :2] = (R @ pts[:, :2].T).T
                            feat[k] = pts
            
            # Handle list of arrays (boundaries)
            for k in ('left_boundaries', 'right_boundaries'):
                if k in feat and isinstance(feat[k], list):
                    for i, boundary in enumerate(feat[k]):
                        if isinstance(boundary, np.ndarray) and boundary.shape[-1] >= 2:
                            pts = boundary.copy()
                            pts[:, :2] = (R @ pts[:, :2].T).T
                            feat[k][i] = pts

        # rotate every object's positions & velocities
        for tr in sc[SD.TRACKS].values():
            P = tr[SD.STATE]['position'].copy()   # (T,3)
            V = tr[SD.STATE]['velocity'].copy()   # (T,2)
            P[:,:2] = (R @ P[:,:2].T).T
            V      = (R @ V.T).T
            tr[SD.STATE]['position'] = P
            tr[SD.STATE]['velocity'] = V.astype(np.float32)

        # finally, make the ego's starting yaw zero
        for tr in sc[SD.TRACKS].values():
            tr[SD.STATE]['heading'] = tr[SD.STATE]['heading'] - psi0

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
        # e.g. inD_v1.0_<sc_id>.pkl
        pkl_name = SD.get_export_file_name(dataset_name, dataset_version, sc_id)
        summary[pkl_name] = sc[SD.METADATA]
        mapping[pkl_name] = ""  # store something if needed

        sc_dict = sc.to_dict()
        SD.sanity_check(sc_dict)
        pkl_path = os.path.join(output_dir, pkl_name)
        with open(pkl_path, 'wb') as pf:
            pickle.dump(sc_dict, pf)

    summary_path = os.path.join(output_dir, "dataset_summary.pkl")
    mapping_path = os.path.join(output_dir, "dataset_mapping.pkl")
    save_summary_and_mapping(summary_path, mapping_path, summary, mapping)

# ===========================================================================
# 4. MAIN WRAPPER
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert inD data + OSM maps => ScenarioNet format, with caching and in-memory segmentation."
    )
    parser.add_argument("--root_dir", required=True,
                        help="Path to inD-dataset-v1.1 root, which should contain 'data/' & 'maps/' subfolders.")
    parser.add_argument("--segment_size", type=int, default=226,
                        help="Raw frames per scenario window. Default 226 frames = 9.0s at 25Hz -> 91 steps at 10Hz.")
    parser.add_argument("--output_dir", default=None,
                        help="Where to put final ScenarioNet PKLs. Default: <root_dir>/converted_scenarios")

    args = parser.parse_args()
    root_dir     = args.root_dir
    segment_size = args.segment_size
    output_dir   = args.output_dir or os.path.join(root_dir, "converted_scenarios")

    data_dir = os.path.join(root_dir, "data")
    maps_dir = os.path.join(root_dir, "maps", "lanelets")  # or just 'maps' if needed

    if not os.path.isdir(data_dir):
        print(f"[ERROR] No 'data' subfolder found in {root_dir}")
        return
    if not os.path.isdir(maps_dir):
        print(f"[ERROR] No 'maps/lanelets' folder found in {root_dir}")
        return

    all_files = os.listdir(data_dir)
    prefixes = []
    for fname in all_files:
        if fname.endswith("_tracks.csv"):
            pfx = fname.replace("_tracks.csv", "")
            prefixes.append(pfx)
    prefixes = sorted(set(prefixes))

    dataset_name    = "inD"
    dataset_version = "1.0"

    # We will store all scenario PKLs from all segments here
    all_scenarios = []

    def find_osm_file_for_location(loc_id):
        """
        Example logic: if your map folder is named like "04_aseag" for locationId=4,
        and the OSM is "location4.osm" inside it, do that.
        Adjust as needed for your folder naming.
        """
        possible_folders = [f for f in os.listdir(maps_dir) if f.startswith(f"{loc_id:02d}_")]
        if not possible_folders:
            return None
        folder = possible_folders[0]
        osm_candidate = os.path.join(maps_dir, folder, f"location{loc_id}.osm")
        if os.path.isfile(osm_candidate):
            return osm_candidate
        return None

    for prefix in prefixes:
        try:
            (frame_rate, dt, xUtm, yUtm, loc_id,
             tracks_meta, agents) = read_ind_data(prefix, data_dir)
        except Exception as e:
            print(f"[WARN] Skipping prefix {prefix} due to error: {e}")
            continue

        # Convert to 'rows' in memory
        rows = process_agents_direct(tracks_meta, agents)
        print(f"[{prefix}] Found {len(rows)} rows, locationId={loc_id}")

        expected_raw_window = compute_waymo_frame_offsets(frame_rate)[-1] + 1
        if segment_size != expected_raw_window:
            print(
                f"[WARN] segment_size={segment_size} is ignored for Waymo alignment; "
                f"using raw_window={expected_raw_window} from frame_rate={frame_rate}"
            )

        segments = split_rows_into_waymo_segments(rows, frame_rate)
        if not segments:
            continue

        # Parse the OSM once for this location (cached)
        osm_file = find_osm_file_for_location(loc_id)
        if osm_file is None:
            print(f"[WARN] No OSM file found for locationId={loc_id}")
            continue
        map_features, map_center = get_osm_map_for_location(loc_id, osm_file, xUtm, yUtm)

        # Convert each segment => scenario
        for i, seg in enumerate(segments, start=1):
            scenario_id = f"{prefix}_loc{loc_id}_seg{i}"
            source_file = f"{prefix}_tracks.csv"
            scenario_variants = create_scenario_from_csv(
                seg["rows"],
                map_features, map_center,
                scenario_id,
                dataset_version,
                xUtm, yUtm,
                frame_rate,
                seg["frames"],
                source_file
            )

            # add scenario variants to final list
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
