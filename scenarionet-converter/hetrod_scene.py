#!/usr/bin/env python3

import os
import math
import argparse
import pandas as pd
import numpy as np
import pickle
import shutil
import copy
import re
from collections import defaultdict

from numpy.linalg import norm
from scipy.interpolate import interp1d
import utm

try:
    from metadrive.scenario import ScenarioDescription as SD
    from metadrive.type import MetaDriveType
except ImportError:
    raise ImportError("Please install ScenarioNet / MetaDrive environment to use these imports.")


AGENT_TYPE_MAPPING = {
    "car": MetaDriveType.VEHICLE,
    "truck_bus": MetaDriveType.VEHICLE,
    "truck": MetaDriveType.VEHICLE,
    "bus": MetaDriveType.VEHICLE,
    "pedestrian": MetaDriveType.PEDESTRIAN,
    "bicycle": MetaDriveType.CYCLIST,
    "motorcycle": MetaDriveType.CYCLIST,
    "unknown": MetaDriveType.OTHER,
}


def lonlat_to_local(lat, lon, xUtmOrigin, yUtmOrigin):
    easting, northing, _, _ = utm.from_latlon(lat, lon)
    return easting - xUtmOrigin, northing - yUtmOrigin


def map_osm_to_md_type(osm_type, subtype=None):
    if osm_type == "lanelet":
        return MetaDriveType.LANE_SURFACE_STREET
    if osm_type == "line_thin":
        if subtype == "solid_solid":
            return MetaDriveType.LINE_SOLID_DOUBLE_YELLOW
        if subtype == "solid":
            return MetaDriveType.LINE_SOLID_SINGLE_WHITE
        if subtype == "dashed":
            return MetaDriveType.LINE_BROKEN_SINGLE_WHITE
        return MetaDriveType.LINE_SOLID_SINGLE_WHITE
    if osm_type == "line_thick":
        if subtype == "solid":
            return MetaDriveType.LINE_SOLID_DOUBLE_WHITE
        if subtype == "dashed":
            return MetaDriveType.LINE_SOLID_DOUBLE_YELLOW
        return MetaDriveType.LINE_SOLID_DOUBLE_WHITE
    if osm_type in {"wall", "curbstone", "fence", "road_border"}:
        return MetaDriveType.BOUNDARY_LINE
    if osm_type in {"zebra_marking", "zebra"}:
        return MetaDriveType.CROSSWALK
    if osm_type == "virtual":
        return MetaDriveType.LINE_BROKEN_SINGLE_WHITE
    if osm_type in {"traffic_light", "multipolygon", "regulatory_element"}:
        return None
    return None


def parse_speed_limit_kmh(tags, default=50.0):
    raw = tags.get("speed_limit") or tags.get("maxspeed")
    if not raw:
        return float(default)

    value = raw.strip().lower()
    match = re.search(r"([0-9]+(?:\.[0-9]*)?)", value)
    if not match:
        return float(default)

    speed = float(match.group(1))
    if "mph" in value or "mile" in value:
        speed *= 1.60934
    return speed


def are_boundaries_aligned(left_coords, right_coords):
    left_dir = np.array(left_coords[-1]) - np.array(left_coords[0])
    right_dir = np.array(right_coords[-1]) - np.array(right_coords[0])
    return np.dot(left_dir, right_dir) >= 0


def resample_coords(coords, num_points):
    if len(coords) < 2:
        return np.array(coords)

    distance = np.cumsum(
        [0] + [norm(np.array(coords[i]) - np.array(coords[i - 1])) for i in range(1, len(coords))]
    )
    if distance[-1] <= 0:
        return np.array(coords)

    distance /= distance[-1]
    interpolator = interp1d(distance, np.array(coords), axis=0, kind="linear")
    return interpolator(np.linspace(0, 1, num_points))


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
    if polyline is None:
        return None
    pts = np.asarray(polyline)
    if pts.shape[0] < 2:
        return None
    start = pts[0][:2]
    for next_pt in pts[1:]:
        vec = next_pt[:2] - start
        vec_norm = np.linalg.norm(vec)
        if vec_norm > eps:
            return vec / vec_norm
    return None


def read_hetrod_data(prefix, data_dir):
    rec_meta_path = os.path.join(data_dir, f"{prefix}_recordingMeta.csv")
    tracks_path = os.path.join(data_dir, f"{prefix}_tracks.csv")
    tracks_meta_path = os.path.join(data_dir, f"{prefix}_tracksMeta.csv")

    rec_meta_df = pd.read_csv(rec_meta_path)
    meta_row = rec_meta_df.iloc[0]

    if "frameRate" in meta_row:
        frame_rate = float(meta_row["frameRate"])
    elif "frame_rate" in meta_row:
        frame_rate = float(meta_row["frame_rate"])
    elif "FrameRate" in meta_row:
        frame_rate = float(meta_row["FrameRate"])
    else:
        raise KeyError("Frame rate column not found in recording meta CSV.")

    dt = 1.0 / frame_rate
    xUtmOrigin = float(meta_row["xUtmOrigin"])
    yUtmOrigin = float(meta_row["yUtmOrigin"])

    if "locationId" not in meta_row:
        raise KeyError("No 'locationId' column found in recordingMeta!")
    location_id = int(meta_row["locationId"])

    tracks_meta_df = pd.read_csv(tracks_meta_path)
    tracks_meta = {}
    for _, row in tracks_meta_df.iterrows():
        track_id = str(int(row["trackId"]))
        tracks_meta[track_id] = {
            "initialFrame": int(row["initialFrame"]),
            "finalFrame": int(row["finalFrame"]),
            "width": float(row["width"]),
            "length": float(row["length"]),
            "agent_type": str(row["class"]).strip(),
        }

    tracks_df = pd.read_csv(tracks_path)
    agents = {}
    for _, row in tracks_df.iterrows():
        track_id = str(int(row["trackId"]))
        rec = {
            "frame": int(row["frame"]),
            "x": float(row["xCenter"]),
            "y": float(row["yCenter"]),
            "heading": float(row["heading"]),
            "vx": float(row["xVelocity"]),
            "vy": float(row["yVelocity"]),
        }
        agents.setdefault(track_id, []).append(rec)

    return frame_rate, dt, xUtmOrigin, yUtmOrigin, location_id, tracks_meta, agents


def process_agents_direct(tracks_meta, agents):
    rows = []
    for track_id, records in agents.items():
        records = sorted(records, key=lambda r: r["frame"])
        if track_id in tracks_meta:
            meta = tracks_meta[track_id]
            avg_width = meta["width"]
            avg_height = meta["length"]
            agent_type = meta["agent_type"]
        else:
            avg_width, avg_height = 0.0, 0.0
            agent_type = "unknown"

        for r in records:
            psi_rad = math.radians(r["heading"])
            if psi_rad > math.pi:
                psi_rad -= 2 * math.pi
            rows.append(
                {
                    "agent_id": track_id,
                    "frame_number": r["frame"],
                    "agent_type": agent_type,
                    "x_position_m": r["x"],
                    "y_position_m": r["y"],
                    "avg_width_m": avg_width,
                    "avg_height_m": avg_height,
                    "psi_rad_rad": psi_rad,
                    "vx_m_s": r["vx"],
                    "vy_m_s": r["vy"],
                }
            )
    return rows


def split_into_segments(rows, segment_size):
    if not rows:
        return []

    min_frame = min(r["frame_number"] for r in rows)
    max_frame = max(r["frame_number"] for r in rows)
    num_segments = math.ceil((max_frame - min_frame + 1) / segment_size)

    segments = [[] for _ in range(num_segments)]
    for r in rows:
        seg_index = (r["frame_number"] - min_frame) // segment_size
        seg_index = max(0, min(seg_index, num_segments - 1))
        segments[seg_index].append(r)
    return [seg for seg in segments if seg]


osm_cache = {}


def parse_osm_map(osm_file, xUtmOrigin, yUtmOrigin):
    import xml.etree.ElementTree as ET
    from shapely.geometry import LineString, MultiLineString, Polygon
    from shapely.ops import linemerge

    tree = ET.parse(osm_file)
    root = tree.getroot()

    nodes = {}
    for node in root.findall("node"):
        node_id = node.attrib["id"]
        lat = float(node.attrib["lat"])
        lon = float(node.attrib["lon"])
        local_x, local_y = lonlat_to_local(lat, lon, xUtmOrigin, yUtmOrigin)
        nodes[node_id] = (local_x, local_y, 0.0)

    ways = {}
    for way in root.findall("way"):
        way_id = way.attrib["id"]
        nd_refs = [nd.attrib["ref"] for nd in way.findall("nd")]
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way.findall("tag")}
        ways[way_id] = {"nd_refs": nd_refs, "tags": tags}

    relations = []
    lanelet_ways = set()
    for rel in root.findall("relation"):
        rel_id = rel.attrib["id"]
        members = []
        for m in rel.findall("member"):
            members.append({"type": m.attrib["type"], "ref": m.attrib["ref"], "role": m.attrib["role"]})
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in rel.findall("tag")}
        relations.append({"id": rel_id, "members": members, "tags": tags})
        if tags.get("type") == "lanelet":
            for m in members:
                if m["type"] == "way" and m["role"] in ["left", "right"]:
                    lanelet_ways.add(m["ref"])

    map_features = {}
    for way_id in lanelet_ways:
        if way_id not in ways:
            continue
        wdata = ways[way_id]
        osm_type = wdata["tags"].get("type", "line_thin")
        subtype = wdata["tags"].get("subtype", "solid")
        md_type = map_osm_to_md_type(osm_type, subtype)
        if not md_type:
            continue

        coords = [nodes[ref] for ref in wdata["nd_refs"] if ref in nodes]
        if not coords:
            continue

        coords_array = np.array(coords, dtype=np.float32)
        if md_type == MetaDriveType.CROSSWALK:
            if len(coords_array) == 2:
                p1, p2 = coords_array[0], coords_array[1]
                direction = p2[:2] - p1[:2]
                length = np.linalg.norm(direction)
                if length > 0:
                    direction = direction / length
                    perp = np.array([-direction[1], direction[0]])
                    half_width = 1.5
                    coords_array = np.array(
                        [
                            p1[:2] + perp * half_width,
                            p2[:2] + perp * half_width,
                            p2[:2] - perp * half_width,
                            p1[:2] - perp * half_width,
                            p1[:2] + perp * half_width,
                        ],
                        dtype=np.float32,
                    )
            else:
                coords_array = coords_array[:, :2]
                if len(coords_array) > 2 and not np.array_equal(coords_array[0], coords_array[-1]):
                    coords_array = np.vstack([coords_array, coords_array[0:1]])
            feature = {"type": md_type, "polygon": coords_array}
        else:
            feature = {"type": md_type, "polyline": coords_array}
        map_features[way_id] = feature

    lane_boundary_refs = {}
    for rel in relations:
        if rel["tags"].get("type") != "lanelet":
            continue

        left_ids = [m["ref"] for m in rel["members"] if m["type"] == "way" and m["role"] == "left"]
        right_ids = [m["ref"] for m in rel["members"] if m["type"] == "way" and m["role"] == "right"]
        if not left_ids or not right_ids:
            continue

        left_lines = []
        right_lines = []
        for w in left_ids:
            if w in ways:
                coords = [nodes[n] for n in ways[w]["nd_refs"] if n in nodes]
                if len(coords) >= 2:
                    left_lines.append(LineString(coords))
        for w in right_ids:
            if w in ways:
                coords = [nodes[n] for n in ways[w]["nd_refs"] if n in nodes]
                if len(coords) >= 2:
                    right_lines.append(LineString(coords))
        if not left_lines or not right_lines:
            continue

        left_merged = linemerge(MultiLineString(left_lines))
        right_merged = linemerge(MultiLineString(right_lines))

        def extract_coords(geom):
            if geom.geom_type == "LineString":
                return list(geom.coords)
            parts = list(geom)
            longest = max(parts, key=lambda g: g.length)
            return list(longest.coords)

        left_coords = extract_coords(left_merged)
        right_coords = extract_coords(right_merged)

        if not are_boundaries_aligned(left_coords, right_coords):
            right_coords.reverse()

        num_pts = max(len(left_coords), len(right_coords))
        l_res = resample_coords(left_coords, num_pts)
        r_res = resample_coords(right_coords, num_pts)

        center = (l_res + r_res) / 2

        def is_left_on_left(centerline, left, step=5):
            signs = []
            for i in range(0, len(centerline) - 1, step):
                v = centerline[i + 1] - centerline[i]
                l_off = left[i] - centerline[i]
                signs.append(np.sign(v[0] * l_off[1] - v[1] * l_off[0]))
            return np.mean(signs) > 0

        if not is_left_on_left(center, l_res):
            center = center[::-1]
            l_res = l_res[::-1]
            r_res = r_res[::-1]

        shell = np.vstack([l_res, r_res[::-1], l_res[0:1]])
        lane_poly = Polygon(shell)
        if not lane_poly.is_valid:
            lane_poly = lane_poly.buffer(0)

        speed_limit_kmh = parse_speed_limit_kmh(rel["tags"])
        lane_id = f"{rel['id']}"
        map_features[lane_id] = {
            "type": MetaDriveType.LANE_SURFACE_STREET,
            "polyline": center.astype(np.float32),
            "left_boundaries": [l_res.astype(np.float32)],
            "right_boundaries": [r_res.astype(np.float32)],
            "entry_lanes": [],
            "exit_lanes": [],
            "left_neighbor": [],
            "right_neighbor": [],
            "speed_limit_kmh": [float(speed_limit_kmh)],
            "interpolating": True,
            "width": float(np.mean(np.linalg.norm(r_res - l_res, axis=1))),
        }
        lane_boundary_refs[lane_id] = {"left": left_ids, "right": right_ids}

    boundary_to_lanes = defaultdict(list)
    for lane_id, bounds in lane_boundary_refs.items():
        for way_id in bounds["left"]:
            boundary_to_lanes[way_id].append((lane_id, "left"))
        for way_id in bounds["right"]:
            boundary_to_lanes[way_id].append((lane_id, "right"))

    lane_direction_cache = {}
    for lane_id, feat in map_features.items():
        if feat.get("type") == MetaDriveType.LANE_SURFACE_STREET and "polyline" in feat:
            lane_direction_cache[lane_id] = compute_lane_direction(feat["polyline"])
        else:
            lane_direction_cache[lane_id] = None

    left_neighbor_sets = defaultdict(set)
    right_neighbor_sets = defaultdict(set)
    for lane_id, bounds in lane_boundary_refs.items():
        for way_id in bounds["left"]:
            for other_lane, side in boundary_to_lanes.get(way_id, []):
                if other_lane == lane_id or side != "right":
                    continue
                left_neighbor_sets[lane_id].add(other_lane)
                right_neighbor_sets[other_lane].add(lane_id)
        for way_id in bounds["right"]:
            for other_lane, side in boundary_to_lanes.get(way_id, []):
                if other_lane == lane_id or side != "left":
                    continue
                right_neighbor_sets[lane_id].add(other_lane)
                left_neighbor_sets[other_lane].add(lane_id)

    for lane_id, feat in map_features.items():
        if feat.get("type") != MetaDriveType.LANE_SURFACE_STREET:
            continue
        feat["left_neighbor"] = sorted(left_neighbor_sets.get(lane_id, []), key=lambda x: str(x))
        feat["right_neighbor"] = sorted(right_neighbor_sets.get(lane_id, []), key=lambda x: str(x))

    lane_endpoints = {}
    angle_cos_threshold = math.cos(math.radians(45.0))
    for mf_id, feat in map_features.items():
        if feat["type"] == MetaDriveType.LANE_SURFACE_STREET and "polyline" in feat:
            pl = feat["polyline"]
            if len(pl) >= 2:
                start_vec = pl[1, :2] - pl[0, :2]
                end_vec = pl[-1, :2] - pl[-2, :2]
                start_norm = np.linalg.norm(start_vec)
                end_norm = np.linalg.norm(end_vec)
                lane_endpoints[mf_id] = {
                    "start": pl[0],
                    "end": pl[-1],
                    "start_dir": start_vec / start_norm if start_norm > 1e-6 else None,
                    "end_dir": end_vec / end_norm if end_norm > 1e-6 else None,
                }

    dist_thr = 2.0
    for lane_id, se in lane_endpoints.items():
        start_pt = se["start"]
        end_pt = se["end"]
        entry_lanes = []
        exit_lanes = []
        for other_id, ose in lane_endpoints.items():
            if other_id == lane_id:
                continue
            if np.linalg.norm(start_pt - ose["end"]) < dist_thr:
                aligned = True
                if se.get("start_dir") is not None and ose.get("end_dir") is not None:
                    aligned = np.dot(se["start_dir"], ose["end_dir"]) > angle_cos_threshold
                if aligned and other_id not in entry_lanes:
                    entry_lanes.append(other_id)
            if np.linalg.norm(end_pt - ose["start"]) < dist_thr:
                aligned = True
                if se.get("end_dir") is not None and ose.get("start_dir") is not None:
                    aligned = np.dot(se["end_dir"], ose["start_dir"]) > angle_cos_threshold
                if aligned and other_id not in exit_lanes:
                    exit_lanes.append(other_id)
        map_features[lane_id]["entry_lanes"] = entry_lanes
        map_features[lane_id]["exit_lanes"] = exit_lanes

    return map_features, (0, 0)


def get_osm_map_for_location(loc_id, osm_file, xUtmOrigin, yUtmOrigin):
    if loc_id not in osm_cache:
        osm_cache[loc_id] = parse_osm_map(osm_file, xUtmOrigin, yUtmOrigin)
    return osm_cache[loc_id]


def create_scenario_from_csv(
    scenario_data,
    map_features,
    map_center,
    scenario_id,
    dataset_version,
    xUtmOrigin,
    yUtmOrigin,
    source_file=None,
):
    scenario = SD()
    scenario[SD.ID] = scenario_id
    scenario[SD.VERSION] = dataset_version
    scenario[SD.METADATA] = {}
    scenario[SD.METADATA][SD.COORDINATE] = "right-handed"
    scenario[SD.METADATA]["dataset"] = "HetroD"
    scenario[SD.METADATA]["scenario_id"] = scenario_id
    scenario[SD.METADATA]["metadrive_processed"] = False
    scenario[SD.METADATA]["id"] = scenario_id

    sample_rate = 0.034
    time_step = sample_rate
    scenario_map = copy.deepcopy(map_features)
    scenario[SD.MAP_FEATURES] = scenario_map

    frames = sorted(set(int(r["frame_number"]) for r in scenario_data))
    num_frames = len(frames)
    scenario[SD.LENGTH] = num_frames

    ts = np.linspace(0, (num_frames - 1) * time_step, num_frames, dtype=np.float64)
    scenario[SD.METADATA][SD.TIMESTEP] = ts
    scenario[SD.METADATA]["ts"] = ts
    frame_to_idx = {f: i for i, f in enumerate(frames)}

    agent_dict = defaultdict(list)
    agent_types = {}
    for r in scenario_data:
        agent_id = r["agent_id"]
        agent_dict[agent_id].append(r)
        agent_types[agent_id] = r["agent_type"].lower()

    scenario[SD.TRACKS] = {}
    object_summary = {}
    for agent_id, recs in agent_dict.items():
        recs = sorted(recs, key=lambda x: int(x["frame_number"]))
        positions = np.zeros((num_frames, 3), dtype=np.float32)
        headings = np.zeros(num_frames, dtype=np.float32)
        velocities = np.zeros((num_frames, 2), dtype=np.float32)
        lengths = np.zeros(num_frames, dtype=np.float32)
        widths = np.zeros(num_frames, dtype=np.float32)
        heights = np.zeros(num_frames, dtype=np.float32)
        valid = np.zeros(num_frames, dtype=bool)

        for rec in recs:
            idx = frame_to_idx[int(rec["frame_number"])]
            positions[idx, 0] = float(rec["x_position_m"])
            positions[idx, 1] = float(rec["y_position_m"])
            headings[idx] = float(rec["psi_rad_rad"])
            velocities[idx, 0] = float(rec["vx_m_s"])
            velocities[idx, 1] = float(rec["vy_m_s"])
            lengths[idx] = float(rec["avg_height_m"])
            widths[idx] = float(rec["avg_width_m"])
            heights[idx] = 1.5
            valid[idx] = True

        raw_type = agent_types[agent_id]
        agent_type = AGENT_TYPE_MAPPING.get(raw_type, MetaDriveType.OTHER)

        valid_pos = positions[valid > 0]
        if len(valid_pos) >= 2:
            deltas = np.diff(valid_pos[:, :2], axis=0)
            dist = float(np.sum(np.linalg.norm(deltas, axis=1)))
        else:
            dist = 0.0

        valid_idx = np.where(valid > 0)[0]
        valid_headings = headings[valid_idx]
        if len(valid_headings) >= 2:
            total_heading_change = float(np.sum(np.abs(np.diff(valid_headings))))
        else:
            total_heading_change = 0.0

        challenge_score = dist + total_heading_change
        valid_length = int(np.sum(valid))
        cval_len = compute_continuous_valid_length(valid)
        object_summary[agent_id] = {
            "type": agent_type,
            "valid_length": valid_length,
            "continuous_valid_length": cval_len,
            "track_length": num_frames,
            "moving_distance": dist,
            "total_heading_change": total_heading_change,
            "challenge_score": challenge_score,
            "object_id": agent_id,
        }

        scenario[SD.TRACKS][agent_id] = {
            SD.TYPE: agent_type,
            SD.STATE: {
                "position": positions,
                "heading": headings,
                "velocity": velocities,
                "length": lengths,
                "width": widths,
                "height": heights,
                "valid": valid,
            },
            SD.METADATA: {
                "track_length": valid_length,
                "type": agent_type,
                "object_id": agent_id,
                "dataset": "HetroD",
                "original_id": agent_id,
            },
        }

    scenario[SD.METADATA]["object_summary"] = object_summary

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
        "map_height_diff": float("-inf"),
    }
    for aid in scenario[SD.TRACKS]:
        a_type = scenario[SD.TRACKS][aid][SD.TYPE]
        num_summary["object_types"].add(a_type)
        num_summary["num_objects_each_type"][a_type] += 1
        if object_summary[aid]["moving_distance"] > 0:
            num_summary["num_moving_objects"] += 1
            num_summary["num_moving_objects_each_type"][a_type] += 1
    scenario[SD.METADATA]["number_summary"] = num_summary

    fallback_id = max(object_summary, key=lambda aid: object_summary[aid]["continuous_valid_length"])
    scenario[SD.METADATA]["current_time_index"] = 25
    scenario[SD.METADATA]["sdc_track_index"] = list(scenario[SD.TRACKS].keys()).index(fallback_id)
    scenario[SD.METADATA]["objects_of_interest"] = []
    scenario[SD.METADATA]["source_file"] = source_file or "hetroD_tracks.csv"
    scenario[SD.METADATA]["track_length"] = num_frames

    scenario_variants = []
    for agent_id in [fallback_id]:
        sc_copy = copy.deepcopy(scenario)
        if agent_id != "ego":
            sc_copy[SD.TRACKS]["ego"] = sc_copy[SD.TRACKS].pop(agent_id)
            sc_copy[SD.TRACKS]["ego"][SD.METADATA][SD.OBJECT_ID] = "ego"
            sc_copy[SD.TRACKS]["ego"][SD.METADATA]["original_id"] = agent_id

        sc_copy[SD.METADATA][SD.SDC_ID] = "ego"
        sc_copy[SD.METADATA]["sdc_track_index"] = list(sc_copy[SD.TRACKS].keys()).index("ego")
        sc_copy[SD.METADATA]["tracks_to_predict"] = {
            "ego": {
                "track_id": "ego",
                "object_type": sc_copy[SD.TRACKS]["ego"][SD.TYPE],
                "difficulty": 0,
                "track_index": list(sc_copy[SD.TRACKS].keys()).index("ego"),
            }
        }
        sc_copy[SD.DYNAMIC_MAP_STATES] = {}
        scenario_variants.append(sc_copy)

    for sc in scenario_variants:
        ego = sc[SD.TRACKS]["ego"][SD.STATE]
        first_i = int(np.where(ego["valid"] > 0)[0][0])
        origin_xy = ego["position"][first_i, :2]

        for feat in sc[SD.MAP_FEATURES].values():
            for k in ("polyline", "polygon"):
                if k in feat and isinstance(feat[k], np.ndarray):
                    feat[k] = feat[k].copy()
                    if k == "polygon" and feat["type"] == MetaDriveType.CROSSWALK:
                        if feat[k].shape[-1] == 2:
                            feat[k] -= origin_xy
                    elif feat[k].shape[-1] >= 2:
                        feat[k][:, :2] -= origin_xy
            for k in ("left_boundaries", "right_boundaries"):
                if k in feat and isinstance(feat[k], list):
                    for i, boundary in enumerate(feat[k]):
                        if isinstance(boundary, np.ndarray) and boundary.shape[-1] >= 2:
                            feat[k][i] = boundary.copy()
                            feat[k][i][:, :2] -= origin_xy

        for tr in sc[SD.TRACKS].values():
            pts = tr[SD.STATE]["position"].copy()
            pts[:, :2] -= origin_xy
            tr[SD.STATE]["position"] = pts

        psi0 = ego["heading"][first_i]
        c, s = math.cos(-psi0), math.sin(-psi0)
        R = np.array([[c, -s], [s, c]], dtype=float)

        for feat in sc[SD.MAP_FEATURES].values():
            for k in ("polyline", "polygon"):
                if k in feat and isinstance(feat[k], np.ndarray):
                    pts = feat[k].copy()
                    if k == "polygon" and feat["type"] == MetaDriveType.CROSSWALK:
                        if pts.shape[-1] == 2:
                            feat[k] = (R @ pts.T).T
                    elif pts.shape[-1] >= 2:
                        pts[:, :2] = (R @ pts[:, :2].T).T
                        feat[k] = pts
            for k in ("left_boundaries", "right_boundaries"):
                if k in feat and isinstance(feat[k], list):
                    for i, boundary in enumerate(feat[k]):
                        if isinstance(boundary, np.ndarray) and boundary.shape[-1] >= 2:
                            pts = boundary.copy()
                            pts[:, :2] = (R @ pts[:, :2].T).T
                            feat[k][i] = pts

        for tr in sc[SD.TRACKS].values():
            P = tr[SD.STATE]["position"].copy()
            V = tr[SD.STATE]["velocity"].copy()
            P[:, :2] = (R @ P[:, :2].T).T
            V = (R @ V.T).T
            tr[SD.STATE]["position"] = P
            tr[SD.STATE]["velocity"] = V
            tr[SD.STATE]["heading"] = tr[SD.STATE]["heading"] - psi0

    return scenario_variants


def save_summary_and_mapping(summary_path, mapping_path, summary, mapping):
    with open(summary_path, "wb") as f:
        pickle.dump(summary, f)
    with open(mapping_path, "wb") as f:
        pickle.dump(mapping, f)


def write_scenarios_to_directory(scenarios, output_dir, dataset_name, dataset_version):
    if os.path.exists(output_dir):
        ans = input(f"Output dir {output_dir} exists. Overwrite? (y/n): ")
        if ans.lower() != "y":
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
        with open(os.path.join(output_dir, pkl_name), "wb") as pf:
            pickle.dump(sc_dict, pf)

    save_summary_and_mapping(
        os.path.join(output_dir, "dataset_summary.pkl"),
        os.path.join(output_dir, "dataset_mapping.pkl"),
        summary,
        mapping,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert HetroD data + OSM maps => ScenarioNet format, with caching and in-memory segmentation."
    )
    parser.add_argument(
        "--root_dir",
        required=True,
        help="Path to HetroD-dataset-v1.1 root, which should contain 'data/' & 'maps/' subfolders.",
    )
    parser.add_argument(
        "--segment_size",
        type=int,
        default=276,
        help="Number of frames per CSV scenario segment.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Where to put final ScenarioNet PKLs. Default: <root_dir>/converted_scenarios",
    )

    args = parser.parse_args()
    root_dir = args.root_dir
    segment_size = args.segment_size
    output_dir = args.output_dir or os.path.join(root_dir, "converted_scenarios")

    data_dir = os.path.join(root_dir, "data")
    maps_dir = os.path.join(root_dir, "maps", "lanelets")
    if not os.path.isdir(data_dir):
        print(f"[ERROR] No 'data' subfolder found in {root_dir}")
        return
    if not os.path.isdir(maps_dir):
        print(f"[ERROR] No 'maps/lanelets' folder found in {root_dir}")
        return

    prefixes = sorted(
        set(fname.replace("_tracks.csv", "") for fname in os.listdir(data_dir) if fname.endswith("_tracks.csv"))
    )

    dataset_name = "HetroD"
    dataset_version = "1.0"
    all_scenarios = []

    def find_osm_file_for_location(loc_id):
        possible_folders = [f for f in os.listdir(maps_dir) if f.startswith(f"{loc_id:02d}_")]
        if not possible_folders:
            return None
        osm_candidate = os.path.join(maps_dir, possible_folders[0], f"location{loc_id}.osm")
        return osm_candidate if os.path.isfile(osm_candidate) else None

    for prefix in prefixes:
        try:
            frame_rate, dt, xUtm, yUtm, loc_id, tracks_meta, agents = read_hetrod_data(prefix, data_dir)
        except Exception as e:
            print(f"[WARN] Skipping prefix {prefix} due to error: {e}")
            continue

        rows = process_agents_direct(tracks_meta, agents)
        print(f"[{prefix}] Found {len(rows)} rows, locationId={loc_id}")

        segments = split_into_segments(rows, segment_size)
        if not segments:
            continue

        osm_file = find_osm_file_for_location(loc_id)
        if osm_file is None:
            print(f"[WARN] No OSM file found for locationId={loc_id}")
            continue
        map_features, map_center = get_osm_map_for_location(loc_id, osm_file, xUtm, yUtm)

        for i, seg_data in enumerate(segments, start=1):
            scenario_id = f"{prefix}_loc{loc_id}_seg{i}"
            scenario_variants = create_scenario_from_csv(
                seg_data,
                map_features,
                map_center,
                scenario_id,
                dataset_version,
                xUtm,
                yUtm,
                f"{prefix}_tracks.csv",
            )
            for j, variant in enumerate(scenario_variants, start=1):
                variant[SD.ID] = f"{scenario_id}_ego_{j}"
                all_scenarios.append(variant)

    if not all_scenarios:
        print("[INFO] No scenarios were produced.")
        return

    write_scenarios_to_directory(all_scenarios, output_dir, dataset_name, dataset_version)
    print(f"[DONE] Wrote {len(all_scenarios)} scenario PKLs into {output_dir}")


if __name__ == "__main__":
    main()
