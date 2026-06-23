#!/usr/bin/env python3

import argparse
import hashlib
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math

import numpy as np
import torch
from scipy.interpolate import interp1d

CATK_ROOT = Path("/home/hcis-s26/Yuhsiang/catk")
if CATK_ROOT.exists():
    sys.path.insert(0, CATK_ROOT.as_posix())

from src.smart.utils.preprocess import get_polylines_from_polygon, preprocess_map  # noqa: E402

"""
conda activate catk

  python /home/hcis-s26/Yuhsiang/drone-tool/scenarionet-smart/scenarionet_to_smart.py \
    --input_dir /home/hcis-s26/Yuhsiang/HetroD/scenarionet_converter/hetrod_scene_level \
    --output_dir /home/hcis-s26/Yuhsiang/catk/cache/SMART_hetrod_crosswalkbox_broken_final_all \
    --prune_crosswalk_box_broken_roadline


"""

SCENARIONET_TRACK_TYPE_TO_SMART = {
    "VEHICLE": 0,
    "PEDESTRIAN": 1,
    "CYCLIST": 2,
}

SCENARIONET_MAP_TYPE_TO_SMART = {
    "LANE_FREEWAY": ("lane", 0),
    "LANE_SURFACE_STREET": ("lane", 1),
    "LANE_BIKE_LANE": ("lane", 3),
    "ROAD_EDGE_BOUNDARY": ("road_edge", 4),
    "ROAD_EDGE_MEDIAN": ("road_edge", 5),
    "ROAD_LINE_BROKEN_SINGLE_WHITE": ("road_line", 6),
    "ROAD_LINE_BROKEN_SINGLE_YELLOW": ("road_line", 6),
    "ROAD_LINE_BROKEN_DOUBLE_YELLOW": ("road_line", 6),
    "ROAD_LINE_SOLID_SINGLE_WHITE": ("road_line", 7),
    "ROAD_LINE_SOLID_SINGLE_YELLOW": ("road_line", 7),
    "ROAD_LINE_SOLID_DOUBLE_WHITE": ("road_line", 8),
    "ROAD_LINE_SOLID_DOUBLE_YELLOW": ("road_line", 8),
    "ROAD_LINE_PASSING_DOUBLE_YELLOW": ("road_line", 8),
    "CROSSWALK": ("crosswalk", 9),
}

SKIP_FILENAMES = {"dataset_summary.pkl", "dataset_mapping.pkl"}

# Agents with total displacement below this threshold (in meters) are considered
# stationary (e.g. parked cars) and will NOT receive ego/predict roles.
MOVING_DISPLACEMENT_THRESHOLD = 2.0  # meters

# HetroD map features are often split into many short sparse polylines. Stitch nearby
# same-type pieces before CATK preprocessing so they do not appear visually disconnected.
POLYLINE_STITCH_ENDPOINT_DISTANCE_THRESHOLD = 2.0
POLYLINE_STITCH_ANGLE_THRESHOLD_DEG = 20.0
POLYLINE_STITCH_WINDOW_DISTANCE_THRESHOLD = 1.0
POLYLINE_STITCH_HEADING_WINDOW = 2
LANE_STITCH_ENDPOINT_DISTANCE_THRESHOLD = 2.5
LANE_STITCH_ANGLE_THRESHOLD_DEG = 28.0
LANE_STITCH_WINDOW_DISTANCE_THRESHOLD = 1.5
LANE_STITCH_HEADING_WINDOW = 3
ROAD_EDGE_STITCH_ENDPOINT_DISTANCE_THRESHOLD = 1.75
ROAD_EDGE_STITCH_ANGLE_THRESHOLD_DEG = 20.0
ROAD_EDGE_STITCH_WINDOW_DISTANCE_THRESHOLD = 1.25
ROAD_EDGE_STITCH_HEADING_WINDOW = 3
BROKEN_ROAD_LINE_STITCH_ENDPOINT_DISTANCE_THRESHOLD = 4.5
BROKEN_ROAD_LINE_STITCH_ANGLE_THRESHOLD_DEG = 55.0
BROKEN_ROAD_LINE_STITCH_WINDOW_DISTANCE_THRESHOLD = 2.0
SOLID_ROAD_LINE_STITCH_ENDPOINT_DISTANCE_THRESHOLD = 3.5
SOLID_ROAD_LINE_STITCH_ANGLE_THRESHOLD_DEG = 40.0
SOLID_ROAD_LINE_STITCH_WINDOW_DISTANCE_THRESHOLD = 1.5
DOUBLE_ROAD_LINE_STITCH_ENDPOINT_DISTANCE_THRESHOLD = 2.25
DOUBLE_ROAD_LINE_STITCH_ANGLE_THRESHOLD_DEG = 20.0
DOUBLE_ROAD_LINE_STITCH_WINDOW_DISTANCE_THRESHOLD = 1.0
ROAD_LINE_STITCH_HEADING_WINDOW = 4


def stable_int_id(raw_id: Any) -> int:
    try:
        return int(raw_id)
    except (TypeError, ValueError):
        digest = hashlib.blake2b(str(raw_id).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)


def get_agent_features(
    track_infos: Dict[str, np.ndarray], num_historical_steps: int, num_steps: int
) -> Dict[str, Any]:
    idx_agents_to_add = []
    for i in range(len(track_infos["object_id"])):
        if track_infos["valid"][i, num_historical_steps - 1]:
            idx_agents_to_add.append(i)

    num_agents = len(idx_agents_to_add)
    out_dict = {
        "num_nodes": num_agents,
        "valid_mask": torch.zeros([num_agents, num_steps], dtype=torch.bool),
        "role": torch.zeros([num_agents, 3], dtype=torch.bool),
        "id": torch.zeros(num_agents, dtype=torch.int64) - 1,
        "type": torch.zeros(num_agents, dtype=torch.uint8),
        "position": torch.zeros([num_agents, num_steps, 3], dtype=torch.float32),
        "heading": torch.zeros([num_agents, num_steps], dtype=torch.float32),
        "velocity": torch.zeros([num_agents, num_steps, 2], dtype=torch.float32),
        "shape": torch.zeros([num_agents, 3], dtype=torch.float32),
    }

    for i, idx in enumerate(idx_agents_to_add):
        out_dict["role"][i] = torch.from_numpy(track_infos["role"][idx])
        out_dict["id"][i] = int(track_infos["object_id"][idx])
        out_dict["type"][i] = int(track_infos["object_type"][idx])

        valid = track_infos["valid"][idx]
        states = track_infos["states"][idx]
        valid_steps = np.where(valid)[0]
        if valid_steps.size == 0:
            continue

        object_shape = states[:, 3:6]
        object_shape = object_shape[valid].mean(axis=0)
        out_dict["shape"][i] = torch.from_numpy(object_shape)

        position = states[:, :3]
        velocity = states[:, 7:9]
        heading = states[:, 6]
        if valid.sum() > 1:
            t_start, t_end = valid_steps[0], valid_steps[-1]
            f_pos = interp1d(valid_steps, position[valid], axis=0)
            f_vel = interp1d(valid_steps, velocity[valid], axis=0)
            f_yaw = interp1d(valid_steps, np.unwrap(heading[valid], axis=0), axis=0)
            t_in = np.arange(t_start, t_end + 1)
            out_dict["valid_mask"][i, t_start : t_end + 1] = True
            out_dict["position"][i, t_start : t_end + 1] = torch.from_numpy(f_pos(t_in))
            out_dict["velocity"][i, t_start : t_end + 1] = torch.from_numpy(f_vel(t_in))
            out_dict["heading"][i, t_start : t_end + 1] = torch.from_numpy(f_yaw(t_in))
        else:
            t = valid_steps[0]
            out_dict["valid_mask"][i, t] = True
            out_dict["position"][i, t] = torch.from_numpy(position[t])
            out_dict["velocity"][i, t] = torch.from_numpy(velocity[t])
            out_dict["heading"][i, t] = torch.tensor(heading[t])

    return out_dict


def _tracks_to_predict_ids(raw: Any) -> set[str]:
    if isinstance(raw, dict):
        out = set()
        for k, v in raw.items():
            if isinstance(v, dict):
                out.add(str(v.get("track_id", k)))
            else:
                out.add(str(v))
        return out
    if isinstance(raw, (list, tuple, set)):
        out = set()
        for v in raw:
            if isinstance(v, dict):
                out.add(str(v.get("track_id", v.get("id", ""))))
            else:
                out.add(str(v))
        return {x for x in out if x}
    return set()


def scenario_track_infos_to_smart(scenario: Dict[str, Any]) -> Dict[str, np.ndarray]:
    tracks = scenario["tracks"]
    track_ids = list(tracks.keys())
    metadata = scenario["metadata"]
    current_time_index = int(metadata["current_time_index"])
    length = int(scenario["length"])

    sdc_id = str(metadata.get("sdc_id", ""))
    sdc_track_index = metadata.get("sdc_track_index", None)

    predict_ids = _tracks_to_predict_ids(metadata.get("tracks_to_predict", {}))
    objects_of_interest = {str(x) for x in metadata.get("objects_of_interest", [])}

    object_id = []
    object_type = []
    states = []
    valid = []
    role = []

    kept_track_ids = []
    for track_id in track_ids:
        tr = tracks[track_id]
        tr_type = str(tr["type"])
        if tr_type not in SCENARIONET_TRACK_TYPE_TO_SMART:
            continue

        st = tr["state"]
        pos = np.asarray(st["position"], dtype=np.float32)
        heading = np.asarray(st["heading"], dtype=np.float32)
        vel = np.asarray(st["velocity"], dtype=np.float32)
        length_arr = np.asarray(st["length"], dtype=np.float32)
        width_arr = np.asarray(st["width"], dtype=np.float32)
        height_arr = np.asarray(st["height"], dtype=np.float32)
        valid_arr = np.asarray(st["valid"], dtype=bool)

        state = np.zeros((length, 9), dtype=np.float32)
        state[:, :2] = pos[:, :2]
        if pos.shape[1] >= 3:
            state[:, 2] = pos[:, 2]
        state[:, 3] = length_arr
        state[:, 4] = width_arr
        state[:, 5] = height_arr
        state[:, 6] = heading
        state[:, 7:9] = vel[:, :2]

        object_id.append(stable_int_id(track_id))
        object_type.append(SCENARIONET_TRACK_TYPE_TO_SMART[tr_type])
        states.append(state)
        valid.append(valid_arr)
        kept_track_ids.append(str(track_id))

    # -- Compute per-agent displacement to identify moving vs stationary agents --
    # HetroD is a drone dataset: the original sdc_id points to a fake stationary
    # reference point at [0,0,0].  Instead we pick the most-displaced agent as ego
    # and mark all moving agents as predict targets.
    displacements = np.zeros(len(kept_track_ids), dtype=np.float32)
    for agent_i in range(len(kept_track_ids)):
        v = valid[agent_i]
        if v.sum() < 2:
            continue
        pos_xy = states[agent_i][:, :2]
        valid_steps = np.where(v)[0]
        start_pos = pos_xy[valid_steps[0]]
        end_pos = pos_xy[valid_steps[-1]]
        displacements[agent_i] = np.linalg.norm(end_pos - start_pos)

    moving_mask = displacements >= MOVING_DISPLACEMENT_THRESHOLD

    # Ego must be valid at current_time_index (get_agent_features filters by this).
    valid_at_current = np.array(
        [valid[i][current_time_index] if len(valid[i]) > current_time_index else False
         for i in range(len(kept_track_ids))],
        dtype=bool,
    )

    # Pick the moving agent with the largest displacement that is valid at
    # current_time_index.  Fall back to original sdc_id if nothing qualifies.
    ego_track_id = None
    eligible = moving_mask & valid_at_current
    if eligible.any():
        ego_agent_i = int(np.where(eligible)[0][np.argmax(displacements[eligible])])
        ego_track_id = kept_track_ids[ego_agent_i]
    else:
        # No moving + valid-at-current agent — fall back.
        if sdc_id and sdc_id in kept_track_ids:
            ego_track_id = sdc_id
        elif sdc_track_index is not None:
            try:
                idx = int(sdc_track_index)
                if 0 <= idx < len(track_ids):
                    candidate = str(track_ids[idx])
                    if candidate in kept_track_ids:
                        ego_track_id = candidate
            except (TypeError, ValueError):
                pass
        if ego_track_id is None and kept_track_ids:
            # Last resort: any agent valid at current_time_index.
            if valid_at_current.any():
                best_i = int(np.where(valid_at_current)[0][np.argmax(displacements[valid_at_current])])
            else:
                best_i = int(np.argmax(displacements))
            ego_track_id = kept_track_ids[best_i]

    for agent_i, track_id in enumerate(kept_track_ids):
        is_ego = (str(track_id) == ego_track_id)
        is_moving = bool(moving_mask[agent_i])
        role.append(
            [
                is_ego,                # role[0]: ego
                is_ego,                # role[1]: interest (same as ego)
                is_moving or is_ego,   # role[2]: predict — all moving agents
            ]
        )

    track_infos = {
        "object_id": np.asarray(object_id, dtype=np.int64),
        "object_type": np.asarray(object_type, dtype=np.uint8),
        "states": np.asarray(states, dtype=np.float32),
        "valid": np.asarray(valid, dtype=bool),
        "role": np.asarray(role, dtype=bool),
        "current_time_index": current_time_index,
    }
    return track_infos


def _normalize_polygon_for_catk(polygon: np.ndarray) -> Optional[np.ndarray]:
    if polygon.ndim != 2 or polygon.shape[0] < 4 or polygon.shape[1] < 2:
        return None

    polygon = np.asarray(polygon, dtype=np.float32)
    if np.allclose(polygon[0, :2], polygon[-1, :2]):
        polygon = polygon[:-1]
    if polygon.shape[0] != 4:
        return None

    xyz = np.zeros((4, 3), dtype=np.float32)
    xyz[:, :2] = polygon[:, :2]
    if polygon.shape[1] >= 3:
        xyz[:, 2] = polygon[:, 2]
    return xyz


def _prepare_polyline(polyline: np.ndarray) -> Optional[np.ndarray]:
    if polyline.ndim != 2 or polyline.shape[0] < 2 or polyline.shape[1] < 2:
        return None

    polyline = np.asarray(polyline, dtype=np.float32)
    keep = [0]
    for idx in range(1, polyline.shape[0]):
        if not np.allclose(polyline[idx, :2], polyline[keep[-1], :2]):
            keep.append(idx)
    polyline = polyline[keep]
    if polyline.shape[0] < 2:
        return None

    densified = [polyline[0]]
    for idx in range(polyline.shape[0] - 1):
        start = polyline[idx]
        end = polyline[idx + 1]
        length = np.linalg.norm(end[:2] - start[:2])
        n_steps = max(int(np.ceil(length / 2.5)), 1)
        for step in range(1, n_steps + 1):
            alpha = step / n_steps
            densified.append(start * (1.0 - alpha) + end * alpha)
    polyline = np.asarray(densified, dtype=np.float32)

    if polyline.shape[0] < 4:
        diffs = polyline[1:, :2] - polyline[:-1, :2]
        seg_len = np.linalg.norm(diffs, axis=1)
        dist = np.concatenate([[0.0], np.cumsum(seg_len)])
        if dist[-1] <= 1e-6:
            return None
        targets = np.linspace(0.0, dist[-1], 4, dtype=np.float32)
        resampled = []
        for t in targets:
            idx = np.searchsorted(dist, t, side='right') - 1
            idx = min(max(idx, 0), len(dist) - 2)
            denom = dist[idx + 1] - dist[idx]
            alpha = 0.0 if denom <= 1e-6 else (t - dist[idx]) / denom
            resampled.append(polyline[idx] * (1.0 - alpha) + polyline[idx + 1] * alpha)
        polyline = np.asarray(resampled, dtype=np.float32)
    return polyline



def _polyline_endpoint_heading(polyline: np.ndarray, at_start: bool) -> float:
    if polyline.shape[0] < 2:
        return 0.0
    if at_start:
        delta = polyline[1, :2] - polyline[0, :2]
    else:
        delta = polyline[-1, :2] - polyline[-2, :2]
    return float(math.atan2(delta[1], delta[0]))


def _polyline_window_heading(polyline: np.ndarray, at_start: bool, window: int = 4) -> float:
    if polyline.shape[0] < 2:
        return 0.0
    span = max(1, min(window, polyline.shape[0] - 1))
    if at_start:
        delta = polyline[span, :2] - polyline[0, :2]
    else:
        delta = polyline[-1, :2] - polyline[-1 - span, :2]
    if float(np.linalg.norm(delta)) <= 1e-6:
        return _polyline_endpoint_heading(polyline, at_start=at_start)
    return float(math.atan2(delta[1], delta[0]))


def _polyline_window_min_distance(poly_a: np.ndarray, poly_b: np.ndarray, window: int = 4) -> float:
    a_cnt = max(1, min(window, poly_a.shape[0]))
    b_cnt = max(1, min(window, poly_b.shape[0]))
    a_pts = poly_a[-a_cnt:, :2]
    b_pts = poly_b[:b_cnt, :2]
    d = a_pts[:, None, :] - b_pts[None, :, :]
    return float(np.min(np.linalg.norm(d, axis=2)))


def _wrap_angle_diff(angle_a: float, angle_b: float) -> float:
    diff = angle_a - angle_b
    return float(abs((diff + math.pi) % (2 * math.pi) - math.pi))


def _merge_oriented_polylines(poly_a: np.ndarray, poly_b: np.ndarray) -> np.ndarray:
    if np.allclose(poly_a[-1, :2], poly_b[0, :2]):
        return np.concatenate([poly_a, poly_b[1:]], axis=0)
    return np.concatenate([poly_a, poly_b], axis=0)


def _stitch_thresholds_for_feature(
    polygon_kind: str,
    point_type: int,
) -> Tuple[float, float, float, int]:
    if polygon_kind == "lane":
        return (
            LANE_STITCH_ENDPOINT_DISTANCE_THRESHOLD,
            LANE_STITCH_ANGLE_THRESHOLD_DEG,
            LANE_STITCH_WINDOW_DISTANCE_THRESHOLD,
            LANE_STITCH_HEADING_WINDOW,
        )
    if polygon_kind == "road_edge":
        return (
            ROAD_EDGE_STITCH_ENDPOINT_DISTANCE_THRESHOLD,
            ROAD_EDGE_STITCH_ANGLE_THRESHOLD_DEG,
            ROAD_EDGE_STITCH_WINDOW_DISTANCE_THRESHOLD,
            ROAD_EDGE_STITCH_HEADING_WINDOW,
        )
    if polygon_kind == "road_line":
        if point_type == 6:
            return (
                BROKEN_ROAD_LINE_STITCH_ENDPOINT_DISTANCE_THRESHOLD,
                BROKEN_ROAD_LINE_STITCH_ANGLE_THRESHOLD_DEG,
                BROKEN_ROAD_LINE_STITCH_WINDOW_DISTANCE_THRESHOLD,
                ROAD_LINE_STITCH_HEADING_WINDOW,
            )
        if point_type == 7:
            return (
                SOLID_ROAD_LINE_STITCH_ENDPOINT_DISTANCE_THRESHOLD,
                SOLID_ROAD_LINE_STITCH_ANGLE_THRESHOLD_DEG,
                SOLID_ROAD_LINE_STITCH_WINDOW_DISTANCE_THRESHOLD,
                ROAD_LINE_STITCH_HEADING_WINDOW,
            )
        if point_type == 8:
            return (
                DOUBLE_ROAD_LINE_STITCH_ENDPOINT_DISTANCE_THRESHOLD,
                DOUBLE_ROAD_LINE_STITCH_ANGLE_THRESHOLD_DEG,
                DOUBLE_ROAD_LINE_STITCH_WINDOW_DISTANCE_THRESHOLD,
                3,
            )
    return (
        POLYLINE_STITCH_ENDPOINT_DISTANCE_THRESHOLD,
        POLYLINE_STITCH_ANGLE_THRESHOLD_DEG,
        POLYLINE_STITCH_WINDOW_DISTANCE_THRESHOLD,
        POLYLINE_STITCH_HEADING_WINDOW,
    )


def _stitch_adjacent_polylines(
    polylines: List[np.ndarray],
    endpoint_dist_thresh: float = POLYLINE_STITCH_ENDPOINT_DISTANCE_THRESHOLD,
    angle_thresh_deg: float = POLYLINE_STITCH_ANGLE_THRESHOLD_DEG,
    window_dist_thresh: float = POLYLINE_STITCH_ENDPOINT_DISTANCE_THRESHOLD,
    heading_window: int = 2,
) -> List[np.ndarray]:
    if len(polylines) <= 1:
        return [np.asarray(polyline, dtype=np.float32) for polyline in polylines]

    remaining = [np.asarray(polyline, dtype=np.float32) for polyline in polylines]
    angle_thresh_rad = math.radians(angle_thresh_deg)

    changed = True
    while changed and len(remaining) > 1:
        changed = False
        best = None
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                for rev_i in (False, True):
                    for rev_j in (False, True):
                        poly_i = remaining[i][::-1] if rev_i else remaining[i]
                        poly_j = remaining[j][::-1] if rev_j else remaining[j]
                        dist = float(np.linalg.norm(poly_i[-1, :2] - poly_j[0, :2]))
                        if dist > endpoint_dist_thresh:
                            continue
                        window_dist = _polyline_window_min_distance(poly_i, poly_j, window=heading_window)
                        if window_dist > window_dist_thresh:
                            continue
                        heading_i = _polyline_window_heading(poly_i, at_start=False, window=heading_window)
                        heading_j = _polyline_window_heading(poly_j, at_start=True, window=heading_window)
                        angle_diff = _wrap_angle_diff(heading_i, heading_j)
                        if angle_diff > angle_thresh_rad:
                            continue
                        score = (window_dist, dist, angle_diff)
                        if best is None or score < best[0]:
                            best = (score, i, j, rev_i, rev_j)
        if best is None:
            break

        _, i, j, rev_i, rev_j = best
        poly_i = remaining[i][::-1] if rev_i else remaining[i]
        poly_j = remaining[j][::-1] if rev_j else remaining[j]
        merged = _merge_oriented_polylines(poly_i, poly_j)
        new_remaining = []
        for idx, poly in enumerate(remaining):
            if idx not in (i, j):
                new_remaining.append(poly)
        new_remaining.append(merged.astype(np.float32))
        remaining = new_remaining
        changed = True

    return remaining


def _append_stitched_polyline_features(
    feature_entries: List[Tuple[int, np.ndarray]],
    point_type: int,
    polygon_kind: str,
    container: Dict[str, List[Dict[str, Any]]],
    polylines: List[np.ndarray],
    point_cnt: int,
) -> int:
    if not feature_entries:
        return point_cnt

    endpoint_dist_thresh, angle_thresh_deg, window_dist_thresh, heading_window = _stitch_thresholds_for_feature(
        polygon_kind,
        point_type,
    )
    stitched_polylines = _stitch_adjacent_polylines(
        [poly for _, poly in feature_entries],
        endpoint_dist_thresh=endpoint_dist_thresh,
        angle_thresh_deg=angle_thresh_deg,
        window_dist_thresh=window_dist_thresh,
        heading_window=heading_window,
    )
    for idx, polyline in enumerate(stitched_polylines):
        feature_id = stable_int_id(f'{polygon_kind}:{point_type}:{idx}:{polyline[0, :2].tolist()}:{polyline[-1, :2].tolist()}')
        point_cnt = append_polyline_feature(
            polyline,
            feature_id,
            point_type,
            polygon_kind,
            container,
            polylines,
            point_cnt,
        )
    return point_cnt


def append_polyline_feature(
    polyline: np.ndarray,
    feature_id: int,
    point_type: int,
    polygon_kind: str,
    container: Dict[str, List[Dict[str, Any]]],
    polylines: List[np.ndarray],
    point_cnt: int,
) -> int:
    polyline = _prepare_polyline(polyline)
    if polyline is None:
        return point_cnt

    xyz = np.zeros((polyline.shape[0], 3), dtype=np.float32)
    xyz[:, :2] = polyline[:, :2]
    if polyline.shape[1] >= 3:
        xyz[:, 2] = polyline[:, 2]

    typed_polyline = np.concatenate(
        [
            xyz,
            np.full((xyz.shape[0], 1), point_type, dtype=np.float32),
            np.full((xyz.shape[0], 1), feature_id, dtype=np.float32),
        ],
        axis=1,
    )
    container[polygon_kind].append(
        {
            "id": feature_id,
            "type": point_type,
            "polyline_index": (point_cnt, point_cnt + len(typed_polyline)),
        }
    )
    polylines.append(typed_polyline)
    return point_cnt + len(typed_polyline)




def _compute_crosswalk_bbox_from_processed_map(
    data: Dict[str, Any],
    margin: float,
) -> Optional[torch.Tensor]:
    traj_pos = data["map_save"]["traj_pos"]
    pl_type = data["pt_token"]["pl_type"]
    crosswalk_mask = pl_type == 3
    if int(crosswalk_mask.sum()) == 0:
        return None
    xy = traj_pos[crosswalk_mask, :, :2].reshape(-1, 2)
    xy_min = xy.min(dim=0).values - margin
    xy_max = xy.max(dim=0).values + margin
    return torch.stack([xy_min, xy_max], dim=0)


def _prune_crosswalk_box_broken_roadline_segments(
    data: Dict[str, Any],
    enabled: bool,
    margin: float,
    min_keep: int = 1,
) -> Dict[str, Any]:
    if not enabled:
        return data

    bbox = _compute_crosswalk_bbox_from_processed_map(data, margin=margin)
    if bbox is None:
        return data

    traj_pos = data["map_save"]["traj_pos"]
    traj_theta = data["map_save"]["traj_theta"]
    pt_type = data["pt_token"]["type"]
    pl_type = data["pt_token"]["pl_type"]
    light_type = data["pt_token"]["light_type"]

    broken_road_line = (pl_type == 2) & (pt_type == 6)
    inside = (
        (traj_pos[..., 0] >= bbox[0, 0])
        & (traj_pos[..., 0] <= bbox[1, 0])
        & (traj_pos[..., 1] >= bbox[0, 1])
        & (traj_pos[..., 1] <= bbox[1, 1])
    ).any(dim=1)
    drop_mask = broken_road_line & inside
    keep_mask = ~drop_mask
    if int(keep_mask.sum()) < min_keep:
        return data
    if int(drop_mask.sum()) == 0:
        return data

    out = dict(data)
    out["map_save"] = dict(data["map_save"])
    out["pt_token"] = dict(data["pt_token"])
    out["map_save"]["traj_pos"] = traj_pos[keep_mask]
    out["map_save"]["traj_theta"] = traj_theta[keep_mask]
    out["pt_token"]["type"] = pt_type[keep_mask]
    out["pt_token"]["pl_type"] = pl_type[keep_mask]
    out["pt_token"]["light_type"] = light_type[keep_mask]
    out["pt_token"]["num_nodes"] = int(keep_mask.sum().item())
    return out


def scenario_map_to_smart(
    scenario: Dict[str, Any],
    prune_crosswalk_box_broken_roadline: bool = False,
    crosswalk_box_margin: float = 2.0,
) -> Dict[str, Any]:
    map_infos = {"lane": [], "road_edge": [], "road_line": [], "crosswalk": []}
    polylines = []
    point_cnt = 0
    grouped_polyline_features: Dict[Tuple[str, int], List[Tuple[int, np.ndarray]]] = {}
    for feature_id_raw, feat in scenario["map_features"].items():
        feat_type = str(feat.get("type", ""))
        mapping = SCENARIONET_MAP_TYPE_TO_SMART.get(feat_type)
        if mapping is None:
            continue

        polygon_kind, point_type = mapping
        feature_id = stable_int_id(feature_id_raw)

        if "polyline" in feat and isinstance(feat["polyline"], np.ndarray):
            polyline_np = np.asarray(feat["polyline"], dtype=np.float32)
            grouped_polyline_features.setdefault((polygon_kind, point_type), []).append((feature_id, polyline_np))
        elif "polygon" in feat and isinstance(feat["polygon"], np.ndarray):
            polygon = np.asarray(feat["polygon"], dtype=np.float32)
            if polygon_kind == "crosswalk":
                polygon_for_catk = _normalize_polygon_for_catk(polygon)
                if polygon_for_catk is not None:
                    crosswalk_polylines = get_polylines_from_polygon(polygon_for_catk)
                    for polyline in np.array_split(crosswalk_polylines, 4):
                        point_cnt = append_polyline_feature(
                            np.asarray(polyline, dtype=np.float32),
                            feature_id,
                            point_type,
                            polygon_kind,
                            map_infos,
                            polylines,
                            point_cnt,
                        )
                    continue

            # Fallback for polygon-typed features that cannot be normalized to CATK's 4-corner assumption.
            if polygon.shape[0] >= 3 and not np.allclose(polygon[0, :2], polygon[-1, :2]):
                polygon = np.concatenate([polygon, polygon[:1]], axis=0)
            point_cnt = append_polyline_feature(
                polygon,
                feature_id,
                point_type,
                polygon_kind,
                map_infos,
                polylines,
                point_cnt,
            )

    for (polygon_kind, point_type), feature_entries in grouped_polyline_features.items():
        point_cnt = _append_stitched_polyline_features(
            feature_entries,
            point_type,
            polygon_kind,
            map_infos,
            polylines,
            point_cnt,
        )

    if polylines:
        map_infos["all_polylines"] = np.concatenate(polylines, axis=0).astype(np.float32)
    else:
        map_infos["all_polylines"] = np.zeros((0, 5), dtype=np.float32)

    num_polygons = sum(len(map_infos[k]) for k in ("lane", "road_edge", "road_line", "crosswalk"))
    polygon_type = torch.zeros(num_polygons, dtype=torch.uint8)
    polygon_light_type = torch.zeros(num_polygons, dtype=torch.uint8)
    point_position: List[Optional[torch.Tensor]] = [None] * num_polygons
    point_type_list: List[Optional[torch.Tensor]] = [None] * num_polygons

    polygon_type_lookup = {"lane": 0, "road_edge": 1, "road_line": 2, "crosswalk": 3}
    segments = [
        (key, seg)
        for key in ("lane", "road_edge", "road_line", "crosswalk")
        for seg in map_infos[key]
    ]

    for idx, (key, seg) in enumerate(segments):
        centerline = map_infos["all_polylines"][seg["polyline_index"][0] : seg["polyline_index"][1]]
        centerline = torch.from_numpy(centerline).float()
        polygon_type[idx] = polygon_type_lookup[key]
        point_position[idx] = centerline[:-1, :2]
        point_type_list[idx] = torch.full((len(centerline) - 1,), seg["type"], dtype=torch.uint8)

    num_points = torch.tensor([point.size(0) for point in point_position if point is not None], dtype=torch.long)
    if len(num_points) == 0:
        map_data = {
            "map_polygon": {"num_nodes": 0, "type": torch.tensor([], dtype=torch.uint8), "light_type": torch.tensor([], dtype=torch.uint8)},
            "map_point": {"num_nodes": 0, "position": torch.tensor([], dtype=torch.float32), "type": torch.tensor([], dtype=torch.uint8)},
            ("map_point", "to", "map_polygon"): {"edge_index": torch.zeros((2, 0), dtype=torch.long)},
        }
        return _prune_crosswalk_box_broken_roadline_segments(
            preprocess_map(map_data),
            enabled=prune_crosswalk_box_broken_roadline,
            margin=crosswalk_box_margin,
        )

    point_to_polygon_edge_index = torch.stack(
        [
            torch.arange(num_points.sum(), dtype=torch.long),
            torch.arange(num_polygons, dtype=torch.long).repeat_interleave(num_points),
        ],
        dim=0,
    )

    map_data = {
        "map_polygon": {
            "num_nodes": num_polygons,
            "type": polygon_type,
            "light_type": polygon_light_type,
        },
        "map_point": {
            "num_nodes": num_points.sum().item(),
            "position": torch.cat(point_position, dim=0),
            "type": torch.cat(point_type_list, dim=0),
        },
        ("map_point", "to", "map_polygon"): {
            "edge_index": point_to_polygon_edge_index
        },
    }
    return _prune_crosswalk_box_broken_roadline_segments(
        preprocess_map(map_data),
        enabled=prune_crosswalk_box_broken_roadline,
        margin=crosswalk_box_margin,
    )


def convert_scenario(
    scenario: Dict[str, Any],
    output_path: Path,
    prune_crosswalk_box_broken_roadline: bool = False,
    crosswalk_box_margin: float = 2.0,
) -> Tuple[str, int]:
    track_infos = scenario_track_infos_to_smart(scenario)
    num_steps = int(scenario["length"])
    current_time_index = int(track_infos["current_time_index"])
    map_dict = scenario_map_to_smart(
        scenario,
        prune_crosswalk_box_broken_roadline=prune_crosswalk_box_broken_roadline,
        crosswalk_box_margin=crosswalk_box_margin,
    )
    map_dict["agent"] = get_agent_features(
        track_infos,
        num_historical_steps=current_time_index + 1,
        num_steps=num_steps,
    )
    map_dict["scenario_id"] = scenario["id"]

    with open(output_path, "wb") as f:
        pickle.dump(map_dict, f)

    return scenario["id"], int(map_dict["agent"]["num_nodes"])


def list_scenario_files(input_dir: Path) -> List[Path]:
    return sorted(p for p in input_dir.glob("*.pkl") if p.name not in SKIP_FILENAMES)


def main():
    parser = argparse.ArgumentParser(description="Convert ScenarioNet PKLs to CAT-K SMART cache PKLs.")
    parser.add_argument("--input_dir", required=True, help="ScenarioNet directory containing scenario PKLs.")
    parser.add_argument("--output_dir", required=True, help="Output SMART cache split directory, e.g. cache/SMART/validation.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of files to convert for quick testing.")
    parser.add_argument(
        "--prune_crosswalk_box_broken_roadline",
        action="store_true",
        help="Prune ROAD_LINE_BROKEN segments whose points fall inside the crosswalk bounding box.",
    )
    parser.add_argument("--crosswalk_box_margin", type=float, default=2.0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_scenario_files(input_dir)
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        print("[INFO] No ScenarioNet PKLs found.")
        return

    converted = 0
    for path in files:
        try:
            with open(path, "rb") as f:
                scenario = pickle.load(f)
            out_path = output_dir / f"{scenario['id']}.pkl"
            scenario_id, num_agents = convert_scenario(
                scenario,
                out_path,
                prune_crosswalk_box_broken_roadline=args.prune_crosswalk_box_broken_roadline,
                crosswalk_box_margin=args.crosswalk_box_margin,
            )
        except Exception as e:
            print(f"[WARN] Failed to convert {path.name}: {e}")
            continue
        converted += 1
        print(f"[OK] {scenario_id} -> {out_path.name} (agents={num_agents})")

    print(f"[DONE] Converted {converted} scenarios into {output_dir}")


if __name__ == "__main__":
    main()
