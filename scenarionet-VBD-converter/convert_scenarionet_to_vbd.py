#!/usr/bin/env python3
"""
Convert ScenarioNet pickle files to VBD format with train/test split.

Usage:
    python convert_scenarionet_to_vbd.py --input_dir /path/to/scenarionet/pkl/files --output_dir /path/to/vbd/output --split_ratio 0.9
"""

import argparse
import glob
import math
import os
import pickle
from typing import Optional

import numpy as np
from tqdm import tqdm
import matplotlib as mpl
from matplotlib import pyplot as plt

# 1. Import scenarionet pkl file format, trajectory, map (need centerline(polyline with direction), exit and entry tags)

# 2. Process Scenarionet trajectories 

# 3. Process Scenarionet maps (need centerline(polyline with direction), exit and entry tags)



def wrap_to_pi(angle):
    """
    Wrap an angle to the range [-pi, pi].

    Args:
        angle (float): The input angle.

    Returns:
        float: The wrapped angle.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


    

def calculate_relations(agents, polylines, traffic_lights):
    """
    Calculate the relations between agents, polylines, and traffic lights.

    Args:
        agents (numpy.ndarray): Array of agent positions and orientations.
        polylines (numpy.ndarray): Array of polyline positions.
        traffic_lights (numpy.ndarray): Array of traffic light positions.

    Returns:
        numpy.ndarray: Array of relations between the elements.
    """
    n_agents = agents.shape[0]
    n_polylines = polylines.shape[0]
    n_traffic_lights = traffic_lights.shape[0]
    n = n_agents + n_polylines + n_traffic_lights

    # Prepare a single array to hold all elements
    all_elements = np.concatenate([
        agents[:, -1, :3],
        polylines[:, 0, :3],
        np.concatenate([traffic_lights[:, :2], np.zeros((n_traffic_lights, 1))], axis=1)
    ], axis=0)

    # Compute pairwise differences using broadcasting
    pos_diff = all_elements[:, :2][:, None, :] - all_elements[:, :2][None, :, :]

    # Compute local positions and angle differences
    cos_theta = np.cos(all_elements[:, 2])[:, None]
    sin_theta = np.sin(all_elements[:, 2])[:, None]
    local_pos_x = pos_diff[..., 0] * cos_theta + pos_diff[..., 1] * sin_theta
    local_pos_y = -pos_diff[..., 0] * sin_theta + pos_diff[..., 1] * cos_theta
    theta_diff = wrap_to_pi(all_elements[:, 2][:, None] - all_elements[:, 2][None, :])

    # Set theta_diff to zero for traffic lights
    start_idx = n_agents + n_polylines
    theta_diff = np.where((np.arange(n) >= start_idx)[:, None] | (np.arange(n) >= start_idx)[None, :], 0, theta_diff)

    # Set the diagonal of the differences to a very small value
    diag_mask = np.eye(n, dtype=bool)
    epsilon = 0.01
    local_pos_x = np.where(diag_mask, epsilon, local_pos_x)
    local_pos_y = np.where(diag_mask, epsilon, local_pos_y)
    theta_diff = np.where(diag_mask, epsilon, theta_diff)

    # Conditions for zero coordinates
    zero_mask = np.logical_or(all_elements[:, 0][:, None] == 0, all_elements[:, 0][None, :] == 0)

    # Initialize relations array
    relations = np.stack([local_pos_x, local_pos_y, theta_diff], axis=-1)

    # Apply zero mask
    relations = np.where(zero_mask[..., None], 0.0, relations)

    return relations



############################
# ScenarioNet → VBD helpers
############################

CURRENT_INDEX = 10
HISTORY_LENGTH = CURRENT_INDEX + 1
FUTURE_LENGTH = 81
MAX_OBJECTS = 64
MAX_POLYLINES = 256
NUM_POINTS_POLYLINE = 30
MAX_TRAFFIC_LIGHTS = 16
TARGET_FRAME_RATE = 10.0
DEFAULT_FRAME_RATE = 10.0

TYPE_TO_ID = {
    'VEHICLE': 1,
    'CAR': 1,
    'TRUCK': 1,
    'BUS': 1,
    'PEDESTRIAN': 2,
    'CYCLIST': 3,
    'BICYCLE': 3,
    'MOTORCYCLE': 3,
    'UNKNOWN': 0,
}

MAP_TYPE_TO_ID = {
    'LANE_SURFACE_STREET': 9,
    'LANE_SURFACE_UNSTRUCTURE': 9,
    'LANE_SURFACE_HIGHWAY': 9,
    'ROAD_LINE_SOLID_SINGLE_WHITE': 2,
    'ROAD_LINE_SOLID_SINGLE_YELLOW': 3,
    'ROAD_LINE_SOLID_DOUBLE_WHITE': 4,
    'ROAD_LINE_SOLID_DOUBLE_YELLOW': 5,
    'ROAD_LINE_BROKEN_SINGLE_WHITE': 6,
    'ROAD_LINE_BROKEN_SINGLE_YELLOW': 7,
    'ROAD_EDGE_BOUNDARY': 1,
    'BOUNDARY_LINE': 1,
    'CROSSWALK': 12,
}


def _ensure_min_length(arr, target_len, axis=0):
    if arr.shape[axis] >= target_len:
        slicer = [slice(None)] * arr.ndim
        slicer[axis] = slice(0, target_len)
        return arr[tuple(slicer)]
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, target_len - arr.shape[axis])
    return np.pad(arr, pad_width, mode='constant')


def resample_polyline(points, target=NUM_POINTS_POLYLINE):
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        return np.zeros((target, pts.shape[-1] if pts.ndim > 1 else 3), dtype=np.float32)
    if pts.shape[0] == 1:
        return np.repeat(pts, target, axis=0)
    xy = pts[:, :2]
    seg_lens = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cumdist = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cumdist[-1]
    if total < 1e-6:
        return np.repeat(pts[:1], target, axis=0)
    target_dist = np.linspace(0.0, total, target)
    new_pts = []
    for dim in range(pts.shape[1]):
        new_pts.append(np.interp(target_dist, cumdist, pts[:, dim]))
    return np.stack(new_pts, axis=-1)


def compute_heading(xy):
    diffs = np.diff(xy, axis=0, prepend=xy[:1])
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    return headings.astype(np.float32)


def process_map_features(scenario_map):
    polylines = []
    for feat in scenario_map.values():
        poly = feat.get('polyline')
        if poly is None or len(poly) == 0:
            continue
        poly = np.asarray(poly, dtype=np.float32)
        poly = resample_polyline(poly, NUM_POINTS_POLYLINE)
        heading = wrap_to_pi(compute_heading(poly[:, :2]))
        lane_type = MAP_TYPE_TO_ID.get(feat.get('type', ''), 0)
        poly_feat = np.zeros((NUM_POINTS_POLYLINE, 5), dtype=np.float32)
        poly_feat[:, 0:2] = poly[:, :2]
        poly_feat[:, 2] = heading
        poly_feat[:, 3] = 0.0  # traffic light state unavailable
        poly_feat[:, 4] = lane_type
        polylines.append(poly_feat)
    if len(polylines) == 0:
        polylines = [np.zeros((NUM_POINTS_POLYLINE, 5), dtype=np.float32)]
    polylines = polylines[:MAX_POLYLINES]
    polylines_valid = np.zeros((MAX_POLYLINES,), dtype=np.int32)
    polylines_valid[:len(polylines)] = 1
    if len(polylines) < MAX_POLYLINES:
        padding = [np.zeros((NUM_POINTS_POLYLINE, 5), dtype=np.float32)
                   for _ in range(MAX_POLYLINES - len(polylines))]
        polylines.extend(padding)
    return np.stack(polylines, axis=0), polylines_valid


def map_agent_type(agent_type):
    if agent_type is None:
        return 0
    key = agent_type.upper()
    return TYPE_TO_ID.get(key, 0)


def _downsample_array(arr, stride: int):
    if stride <= 1:
        return arr
    return arr[::stride]


def infer_frame_rate(scenario, fallback: float = DEFAULT_FRAME_RATE) -> float:
    ts = scenario.get('metadata', {}).get('ts')
    if ts is None:
        ts = scenario.get('metadata', {}).get('timestep')
    if ts is not None:
        ts = np.asarray(ts, dtype=np.float32)
        if ts.size >= 2:
            dt = float(ts[1] - ts[0])
            if dt > 1e-6:
                return 1.0 / dt
    return float(fallback)


def process_trajectories(scenario, frame_rate: float = DEFAULT_FRAME_RATE):
    tracks = scenario['tracks']
    metadata = scenario['metadata']
    sdc_id = metadata.get('sdc_id')
    objects_of_interest = set(str(o) for o in metadata.get('objects_of_interest', []))
    total_length = scenario['length']

    stride = max(1, int(round(frame_rate / TARGET_FRAME_RATE)))
    target_length = int(math.ceil(total_length / stride))
    if target_length <= CURRENT_INDEX:
        raise ValueError('Scenario shorter than required window.')

    def to_raw_index(index: int, series_length: int) -> int:
        raw = int(round(index * frame_rate / TARGET_FRAME_RATE))
        return min(raw, series_length - 1)

    def get_position(track_id, index):
        track = tracks[track_id]
        pos = track['state']['position']
        valid = track['state']['valid']
        raw_index = to_raw_index(index, pos.shape[0])
        if raw_index >= pos.shape[0] or not valid[raw_index]:
            return None
        return pos[raw_index, :2]
    if sdc_id in tracks:
        sdc_pos = get_position(sdc_id, CURRENT_INDEX)
    else:
        first_id = next(iter(tracks))
        sdc_id = first_id
        sdc_pos = get_position(sdc_id, CURRENT_INDEX)
    candidates = []
    for track_id, track in tracks.items():
        pos = track['state']['position']
        if pos.shape[0] <= to_raw_index(CURRENT_INDEX, pos.shape[0]):
            continue
        valid = track['state']['valid']
        agent_pos = get_position(track_id, CURRENT_INDEX)
        if agent_pos is None or sdc_pos is None:
            dist = np.inf
        else:
            dist = np.linalg.norm(agent_pos - sdc_pos)
        candidates.append((dist, track_id))
    candidates.sort(key=lambda x: x[0])
    selected = [tid for _, tid in candidates[:MAX_OBJECTS]]
    agents_history = np.zeros((MAX_OBJECTS, HISTORY_LENGTH, 8), dtype=np.float32)
    agents_future = np.zeros((MAX_OBJECTS, FUTURE_LENGTH, 5), dtype=np.float32)
    agents_interested = np.zeros((MAX_OBJECTS,), dtype=np.int32)
    agents_type = np.zeros((MAX_OBJECTS,), dtype=np.int32)
    agents_id = np.zeros((MAX_OBJECTS,), dtype=np.int32)
    for idx, track_id in enumerate(selected):
        track = tracks[track_id]
        state = track['state']
        pos = _downsample_array(state['position'].astype(np.float32), stride)
        heading = wrap_to_pi(_downsample_array(state['heading'].astype(np.float32), stride))
        velocity = _downsample_array(state['velocity'].astype(np.float32), stride)
        length = _downsample_array(state['length'].astype(np.float32), stride)
        width = _downsample_array(state['width'].astype(np.float32), stride)
        height = _downsample_array(state['height'].astype(np.float32), stride)
        valid = _downsample_array(state['valid'].astype(np.bool_), stride)
        hist_pos = _ensure_min_length(pos, HISTORY_LENGTH)
        hist_heading = _ensure_min_length(heading[:, None], HISTORY_LENGTH, axis=0).reshape(HISTORY_LENGTH)
        hist_vel = _ensure_min_length(velocity, HISTORY_LENGTH)
        hist_length = _ensure_min_length(length[:, None], HISTORY_LENGTH, axis=0).reshape(HISTORY_LENGTH)
        hist_width = _ensure_min_length(width[:, None], HISTORY_LENGTH, axis=0).reshape(HISTORY_LENGTH)
        hist_height = _ensure_min_length(height[:, None], HISTORY_LENGTH, axis=0).reshape(HISTORY_LENGTH)
        hist_valid = _ensure_min_length(valid[:, None], HISTORY_LENGTH, axis=0).reshape(HISTORY_LENGTH)
        history = np.column_stack([
            hist_pos[:HISTORY_LENGTH, 0],
            hist_pos[:HISTORY_LENGTH, 1],
            hist_heading[:HISTORY_LENGTH],
            hist_vel[:HISTORY_LENGTH, 0],
            hist_vel[:HISTORY_LENGTH, 1],
            hist_length[:HISTORY_LENGTH],
            hist_width[:HISTORY_LENGTH],
            hist_height[:HISTORY_LENGTH],
        ])
        history[~hist_valid[:HISTORY_LENGTH]] = 0.0
        agents_history[idx] = history
        future_end = min(pos.shape[0], CURRENT_INDEX + FUTURE_LENGTH)
        future_slice = slice(CURRENT_INDEX, future_end)
        fut_pos = _ensure_min_length(pos[future_slice], FUTURE_LENGTH)
        fut_heading = _ensure_min_length(heading[future_slice, None], FUTURE_LENGTH, axis=0).reshape(FUTURE_LENGTH)
        fut_vel = _ensure_min_length(velocity[future_slice], FUTURE_LENGTH)
        fut_valid = _ensure_min_length(valid[future_slice, None], FUTURE_LENGTH, axis=0).reshape(FUTURE_LENGTH)
        future = np.column_stack([
            fut_pos[:, 0],
            fut_pos[:, 1],
            fut_heading,
            fut_vel[:, 0],
            fut_vel[:, 1],
        ])
        future[:, 2] = wrap_to_pi(future[:, 2])
        future[~fut_valid] = 0.0
        agents_future[idx] = future
        agents_type[idx] = map_agent_type(track.get('type'))
        if str(track_id) == str(sdc_id) or track_id == sdc_id:
            agents_interested[idx] = 10
        elif str(track_id) in objects_of_interest:
            agents_interested[idx] = 10
        else:
            agents_interested[idx] = 1
        try:
            agents_id[idx] = int(track['metadata'].get('original_id', track_id))
        except (TypeError, ValueError):
            agents_id[idx] = idx
    return agents_history, agents_future, agents_interested, agents_type, agents_id


def scenarionet_to_vbd(scenario, include_raw=False, frame_rate: Optional[float] = None):
    if frame_rate is None:
        frame_rate = infer_frame_rate(scenario, DEFAULT_FRAME_RATE)
    agents_history, agents_future, agents_interested, agents_type, agents_id = process_trajectories(
        scenario, frame_rate=frame_rate
    )
    polylines, polylines_valid = process_map_features(scenario['map_features'])
    traffic_light_points = np.zeros((MAX_TRAFFIC_LIGHTS, 3), dtype=np.float32)
    relations = calculate_relations(agents_history, polylines, traffic_light_points)
    data_dict = {
        'agents_history': agents_history,
        'agents_future': agents_future,
        'agents_interested': agents_interested,
        'agents_type': agents_type.astype(np.int32),
        'agents_id': agents_id.astype(np.int32),
        'traffic_light_points': traffic_light_points,
        'polylines': polylines,
        'polylines_valid': polylines_valid,
        'relations': relations.astype(np.float32),
    }
    if include_raw:
        data_dict['scenario_raw'] = scenario
    data_dict['scenario_id'] = scenario.get('id', scenario['metadata'].get('scenario_id', 'unknown'))
    return data_dict


def convert_directory(input_dir, output_dir, include_raw=False, frame_rate: Optional[float] = None):
    os.makedirs(output_dir, exist_ok=True)
    pkl_files = sorted(glob.glob(os.path.join(input_dir, '*.pkl')))
    if len(pkl_files) == 0:
        raise FileNotFoundError(f'No ScenarioNet pkl files found in {input_dir}')
    for path in tqdm(pkl_files, desc='Converting ScenarioNet'):
        with open(path, 'rb') as f:
            scenario = pickle.load(f)
        if not isinstance(scenario, dict) or 'tracks' not in scenario or 'map_features' not in scenario:
            continue
        try:
            data_dict = scenarionet_to_vbd(scenario, include_raw=include_raw, frame_rate=frame_rate)
        except ValueError as err:
            print(f'[WARN] Skipping {path}: {err}')
            continue
        scenario_id = data_dict['scenario_id']
        out_path = os.path.join(output_dir, f'scenario_{scenario_id}.pkl')
        with open(out_path, 'wb') as wf:
            pickle.dump(data_dict, wf)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert ScenarioNet PKL files to VBD format.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing ScenarioNet PKL files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save converted VBD pickles.')
    parser.add_argument('--include_raw', action='store_true', help='Store original ScenarioNet scenario inside output.')
    parser.add_argument('--frame_rate', type=float, default=None, help='Optional source frame rate override; default infers from scenario metadata ts.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert_directory(
        args.input_dir,
        args.output_dir,
        include_raw=args.include_raw,
        frame_rate=args.frame_rate,
    )
