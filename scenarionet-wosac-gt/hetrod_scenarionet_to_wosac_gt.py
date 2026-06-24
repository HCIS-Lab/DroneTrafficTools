#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

# NumPy 2 pickles refer to numpy._core, while older environments may still use
# NumPy 1.x where the same modules live under numpy.core.
if not hasattr(np, "_core"):
    import numpy.core
    import numpy.core.multiarray
    import numpy.core.numeric

    sys.modules.setdefault("numpy._core", numpy.core)
    sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)

try:
    from waymo_open_dataset.protos import scenario_pb2

    TYPE_UNSET = scenario_pb2.Track.ObjectType.TYPE_UNSET
    TYPE_VEHICLE = scenario_pb2.Track.ObjectType.TYPE_VEHICLE
    TYPE_PEDESTRIAN = scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN
    TYPE_CYCLIST = scenario_pb2.Track.ObjectType.TYPE_CYCLIST
except Exception:
    # Waymo Track.ObjectType enum values. Keeping the fallback makes this script
    # usable for conversion even when only the output consumer has Waymo installed.
    TYPE_UNSET = 0
    TYPE_VEHICLE = 1
    TYPE_PEDESTRIAN = 2
    TYPE_CYCLIST = 3


NUM_STEPS = 91
CURRENT_TIME_INDEX = 10
SKIP_FILENAMES = {"dataset_summary.pkl", "dataset_mapping.pkl"}

TRACK_TYPE_TO_WAYMO = {
    "VEHICLE": TYPE_VEHICLE,
    "PEDESTRIAN": TYPE_PEDESTRIAN,
    "CYCLIST": TYPE_CYCLIST,
}

LANE_TYPES = {
    "LANE_FREEWAY",
    "LANE_SURFACE_STREET",
    "LANE_BIKE_LANE",
}
ROAD_EDGE_TYPES = {
    "ROAD_EDGE_BOUNDARY",
    "ROAD_EDGE_MEDIAN",
}

PredictAgentPolicy = Literal["all", "current_valid", "tracks_to_predict"]
AgentFilterPolicy = Literal["current_valid", "all"]


def stable_int_id(raw_id: Any) -> int:
    try:
        return int(raw_id)
    except (TypeError, ValueError):
        digest = hashlib.blake2b(str(raw_id).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_pickle_atomic(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def scenario_id_from_path(path: Path) -> str | None:
    name = path.name
    if name in SKIP_FILENAMES:
        return None
    stem = path.stem
    prefix = "sd_HetroD_1.0_"
    return stem[len(prefix) :] if stem.startswith(prefix) else stem


def scenario_output_path(input_path: Path, output_dir: Path) -> Path | None:
    scenario_id = scenario_id_from_path(input_path)
    if scenario_id is None:
        return None
    return output_dir / f"{scenario_id}.pkl"


def list_scenario_files(input_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(input_dir.glob("*.pkl"))
        if path.name not in SKIP_FILENAMES
    ]


def _to_xyz_tensor(points: Any) -> torch.Tensor | None:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    if arr.shape[1] == 2:
        arr = np.concatenate(
            [arr, np.zeros((arr.shape[0], 1), dtype=np.float32)],
            axis=1,
        )
    else:
        arr = arr[:, :3]
    if not np.isfinite(arr).all():
        return None
    return torch.from_numpy(arr.copy()).float()


def _signed_distance_to_polyline(points: torch.Tensor, polyline: torch.Tensor) -> torch.Tensor:
    """Signed 2D distance using the same sign convention as WOSAC road-edge metric."""
    if points.numel() == 0 or len(polyline) < 2:
        return torch.empty(points.shape[0], dtype=torch.float32)
    points = points.float()
    polyline = polyline.float()
    segment_start = polyline[:-1]
    segment_end = polyline[1:]
    segment = segment_end - segment_start
    start_to_point = points[:, None, :3] - segment_start[None, :, :3]
    denom = (segment[None, :, :2] * segment[None, :, :2]).sum(dim=-1).clamp_min(1e-9)
    rel_t = ((start_to_point[:, :, :2] * segment[None, :, :2]).sum(dim=-1) / denom).clamp(0.0, 1.0)
    projected = segment_start[None, :, :3] + rel_t[:, :, None] * segment[None, :, :3]
    distance = (points[:, None, :2] - projected[:, :, :2]).norm(dim=-1)
    closest_segment = distance.argmin(dim=1)
    row = torch.arange(points.shape[0])
    closest_start_to_point = points[:, :3] - segment_start[closest_segment, :3]
    closest_segment_vec = segment[closest_segment]
    cross = (
        closest_start_to_point[:, 0] * closest_segment_vec[:, 1]
        - closest_start_to_point[:, 1] * closest_segment_vec[:, 0]
    )
    sign = torch.sign(cross)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return sign * distance[row, closest_segment]


def orient_road_edges_by_lanes(
    road_edges: list[torch.Tensor],
    lane_polylines: list[torch.Tensor],
    *,
    local_evidence_radius: float = 20.0,
    min_evidence_points: int = 3,
) -> list[torch.Tensor]:
    if not road_edges or not lane_polylines:
        return road_edges
    lane_points = torch.cat([lane[:, :3] for lane in lane_polylines if len(lane) > 0], dim=0)
    oriented = []
    for edge in road_edges:
        signed_distance = _signed_distance_to_polyline(lane_points, edge)
        nearby = signed_distance.abs() < local_evidence_radius
        if int(nearby.sum().item()) < min_evidence_points:
            oriented.append(edge)
            continue
        positive_fraction = (signed_distance[nearby] > 0).float().mean()
        oriented.append(edge.flip(0) if positive_fraction > 0.5 else edge)
    return oriented


def extract_map_features(scenario: dict[str, Any], *, orient_road_edges: bool) -> tuple[list[int], list[torch.Tensor], list[torch.Tensor]]:
    lane_ids: list[int] = []
    lane_polylines: list[torch.Tensor] = []
    road_edges: list[torch.Tensor] = []

    for raw_id, feature in scenario.get("map_features", {}).items():
        feature_type = str(feature.get("type", ""))
        polyline = feature.get("polyline")
        if polyline is None:
            continue
        tensor = _to_xyz_tensor(polyline)
        if tensor is None:
            continue
        if feature_type in LANE_TYPES:
            lane_ids.append(stable_int_id(raw_id))
            lane_polylines.append(tensor)
        elif feature_type in ROAD_EDGE_TYPES:
            road_edges.append(tensor)

    if orient_road_edges:
        road_edges = orient_road_edges_by_lanes(road_edges, lane_polylines)
    return lane_ids, lane_polylines, road_edges


def _tracks_to_predict_ids(raw: Any) -> set[int]:
    if isinstance(raw, dict):
        out = set()
        for key, value in raw.items():
            if isinstance(value, dict):
                out.add(stable_int_id(value.get("track_id", key)))
            else:
                out.add(stable_int_id(value))
        return out
    if isinstance(raw, (list, tuple, set)):
        return {stable_int_id(value.get("track_id", value.get("id", "")) if isinstance(value, dict) else value) for value in raw}
    return set()


def build_tracks(
    scenario: dict[str, Any],
    *,
    agent_filter: AgentFilterPolicy,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    tracks_in = scenario["tracks"]
    all_track_ids = list(tracks_in.keys())
    num_steps = int(scenario.get("length", NUM_STEPS))
    if num_steps != NUM_STEPS:
        raise ValueError(f"Expected {NUM_STEPS} timesteps, got {num_steps}")

    current_time_index = int(scenario.get("metadata", {}).get("current_time_index", CURRENT_TIME_INDEX))
    if agent_filter == "current_valid":
        track_ids = [
            track_id
            for track_id in all_track_ids
            if bool(np.asarray(tracks_in[track_id]["state"]["valid"], dtype=bool)[current_time_index])
        ]
    elif agent_filter == "all":
        track_ids = all_track_ids
    else:
        raise ValueError(f"Unsupported agent filter: {agent_filter}")

    num_agents = len(track_ids)

    tracks = torch.zeros((num_agents, num_steps, 9), dtype=torch.float32)
    track_masks = torch.zeros((num_agents, num_steps), dtype=torch.bool)
    object_ids = torch.zeros((num_agents,), dtype=torch.int32)
    object_types = torch.full((num_agents,), TYPE_UNSET, dtype=torch.int64)

    for index, raw_track_id in enumerate(track_ids):
        track = tracks_in[raw_track_id]
        state = track["state"]
        object_ids[index] = stable_int_id(raw_track_id)
        object_types[index] = TRACK_TYPE_TO_WAYMO.get(str(track.get("type", "")), TYPE_UNSET)

        position = np.asarray(state["position"], dtype=np.float32)
        heading = np.asarray(state["heading"], dtype=np.float32)
        velocity = np.asarray(state["velocity"], dtype=np.float32)
        length = np.asarray(state["length"], dtype=np.float32)
        width = np.asarray(state["width"], dtype=np.float32)
        height = np.asarray(state["height"], dtype=np.float32)
        valid = np.asarray(state["valid"], dtype=bool)

        if position.shape[0] != num_steps or heading.shape[0] != num_steps or valid.shape[0] != num_steps:
            raise ValueError(f"Track {raw_track_id} has inconsistent timestep length.")

        tracks[index, :, 0:2] = torch.from_numpy(position[:, :2])
        if position.shape[1] >= 3:
            tracks[index, :, 2] = torch.from_numpy(position[:, 2])
        tracks[index, :, 3] = torch.from_numpy(length)
        tracks[index, :, 4] = torch.from_numpy(width)
        tracks[index, :, 5] = torch.from_numpy(height)
        tracks[index, :, 6] = torch.from_numpy(heading)
        tracks[index, :, 7:9] = torch.from_numpy(velocity[:, :2])
        track_masks[index] = torch.from_numpy(valid)

    return tracks, track_masks, object_ids, object_types, track_ids


def select_sdc_track_index(scenario: dict[str, Any], object_ids: torch.Tensor, track_masks: torch.Tensor) -> int:
    metadata = scenario.get("metadata", {})
    raw_index = metadata.get("sdc_track_index")
    if raw_index is not None:
        try:
            index = int(raw_index)
            if 0 <= index < object_ids.numel():
                return index
        except (TypeError, ValueError):
            pass
    sdc_id = metadata.get("sdc_id")
    if sdc_id is not None:
        matches = torch.nonzero(object_ids == stable_int_id(sdc_id), as_tuple=False).flatten()
        if matches.numel() > 0:
            return int(matches[0].item())
    current = int(metadata.get("current_time_index", CURRENT_TIME_INDEX))
    valid_now = torch.nonzero(track_masks[:, current], as_tuple=False).flatten()
    return int(valid_now[0].item()) if valid_now.numel() else 0


def select_predict_agent_ids(
    scenario: dict[str, Any],
    object_ids: torch.Tensor,
    track_masks: torch.Tensor,
    policy: PredictAgentPolicy,
) -> torch.Tensor:
    if policy == "all":
        return torch.sort(object_ids)[0].int()
    if policy == "current_valid":
        current = int(scenario.get("metadata", {}).get("current_time_index", CURRENT_TIME_INDEX))
        return torch.sort(object_ids[track_masks[:, current]])[0].int()
    if policy == "tracks_to_predict":
        ids = _tracks_to_predict_ids(scenario.get("metadata", {}).get("tracks_to_predict", {}))
        if ids:
            id_tensor = torch.tensor(sorted(ids), dtype=torch.int32)
            present = torch.isin(id_tensor, object_ids.int())
            if bool(present.all()):
                return id_tensor
        current = int(scenario.get("metadata", {}).get("current_time_index", CURRENT_TIME_INDEX))
        return torch.sort(object_ids[track_masks[:, current]])[0].int()
    raise ValueError(f"Unsupported predict-agent policy: {policy}")


def convert_scenario_to_gt(
    scenario: dict[str, Any],
    *,
    agent_filter: AgentFilterPolicy,
    predict_agent_policy: PredictAgentPolicy,
    orient_road_edges: bool,
) -> dict[str, Any]:
    scenario_id = str(scenario.get("id") or scenario.get("metadata", {}).get("id") or scenario.get("metadata", {}).get("scenario_id"))
    current_time_index = int(scenario.get("metadata", {}).get("current_time_index", CURRENT_TIME_INDEX))
    if current_time_index != CURRENT_TIME_INDEX:
        raise ValueError(f"Expected current_time_index={CURRENT_TIME_INDEX}, got {current_time_index}")

    tracks, track_masks, object_ids, object_types, _ = build_tracks(
        scenario,
        agent_filter=agent_filter,
    )
    lane_ids, lane_polylines, road_edges = extract_map_features(
        scenario,
        orient_road_edges=orient_road_edges,
    )
    sim_agent_ids = torch.sort(object_ids[track_masks[:, current_time_index]])[0].int()
    predict_agent_ids = select_predict_agent_ids(
        scenario,
        object_ids,
        track_masks,
        predict_agent_policy,
    )
    predict_index = torch.nonzero(torch.isin(object_ids.int(), predict_agent_ids), as_tuple=False).flatten().int()
    objects_of_interest = [
        stable_int_id(value)
        for value in scenario.get("metadata", {}).get("objects_of_interest", [])
    ]
    timestamps = scenario.get("metadata", {}).get("ts")
    if timestamps is None:
        timestamps_seconds = [0.1 * i for i in range(NUM_STEPS)]
    else:
        timestamps_seconds = [float(x) for x in np.asarray(timestamps, dtype=np.float32).tolist()]

    return {
        "scenario_id": scenario_id,
        "timestamps_seconds": timestamps_seconds,
        "current_time_index": current_time_index,
        "sdc_track_index": select_sdc_track_index(scenario, object_ids, track_masks),
        "objects_of_interest": sorted(objects_of_interest),
        "tracks": tracks,
        "track_masks": track_masks,
        "object_ids": object_ids.int(),
        "object_types": object_types,
        "road_edges": road_edges,
        "predict_index": predict_index,
        "sim_agent_ids": sim_agent_ids,
        "predict_agent_ids": predict_agent_ids,
        "lane_ids": lane_ids,
        "lane_polylines": lane_polylines,
        "traffic_signals": [],
    }


def validate_gt(gt: dict[str, Any]) -> None:
    required = {
        "scenario_id",
        "tracks",
        "track_masks",
        "object_ids",
        "object_types",
        "road_edges",
        "sim_agent_ids",
        "predict_agent_ids",
        "lane_polylines",
        "traffic_signals",
    }
    missing = required - set(gt)
    if missing:
        raise ValueError(f"GT is missing required keys: {sorted(missing)}")
    if gt["tracks"].shape[-2:] != (NUM_STEPS, 9):
        raise ValueError(f"tracks must have shape [num_agents, {NUM_STEPS}, 9], got {gt['tracks'].shape}")
    if gt["track_masks"].shape != gt["tracks"].shape[:2]:
        raise ValueError("track_masks shape does not match tracks.")
    if gt["object_ids"].numel() != gt["tracks"].shape[0]:
        raise ValueError("object_ids length does not match tracks.")
    if torch.unique(gt["object_ids"]).numel() != gt["object_ids"].numel():
        raise ValueError("object_ids must be unique.")
    if not torch.isfinite(gt["tracks"]).all():
        raise ValueError("tracks contain NaN or Inf.")


def convert_one(
    input_path: Path,
    output_dir: Path,
    predict_agent_policy: PredictAgentPolicy,
    agent_filter: AgentFilterPolicy,
    orient_road_edges: bool,
    overwrite: bool,
) -> tuple[str, int, int, int]:
    output_path = scenario_output_path(input_path, output_dir)
    if output_path is None:
        raise ValueError(f"Unsupported ScenarioNet file name: {input_path.name}")
    if output_path.is_file() and not overwrite:
        return output_path.stem, -1, -1, -1
    scenario = load_pickle(input_path)
    gt = convert_scenario_to_gt(
        scenario,
        agent_filter=agent_filter,
        predict_agent_policy=predict_agent_policy,
        orient_road_edges=orient_road_edges,
    )
    validate_gt(gt)
    write_pickle_atomic(gt, output_path)
    return (
        str(gt["scenario_id"]),
        int(gt["object_ids"].numel()),
        int(gt["sim_agent_ids"].numel()),
        len(gt["road_edges"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert HetroD ScenarioNet PKLs to WOSAC-style GT pickle files.",
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--predict-agent-policy",
        choices=("all", "current_valid", "tracks_to_predict"),
        default="all",
        help="Policy for predict_agent_ids. HetroD challenge defaults to all agents.",
    )
    parser.add_argument(
        "--agent-filter",
        choices=("current_valid", "all"),
        default="current_valid",
        help=(
            "Which ScenarioNet tracks to keep in object_ids/tracks. HetroD challenge "
            "defaults to current_valid so required submission agents match sim_agent_ids. "
            "Use all for Waymo-style full-track GT pickles."
        ),
    )
    parser.add_argument(
        "--no-orient-road-edges",
        action="store_true",
        help="Disable lane-based road-edge orientation correction.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = list_scenario_files(input_dir)
    if args.limit is not None:
        files = files[: args.limit]
    if not args.overwrite:
        original = len(files)
        files = [
            path
            for path in files
            if (scenario_output_path(path, output_dir) is None or not scenario_output_path(path, output_dir).is_file())
        ]
        skipped = original - len(files)
        if skipped:
            print(f"[INFO] Resuming: skipped {skipped} existing GT files.")

    if not files:
        print("[INFO] No ScenarioNet scenario files to convert.")
        return 0

    converted = 0
    skipped = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                convert_one,
                path,
                output_dir,
                args.predict_agent_policy,
                args.agent_filter,
                not args.no_orient_road_edges,
                args.overwrite,
            ): path
            for path in files
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                scenario_id, num_agents, num_sim_agents, num_road_edges = future.result()
            except Exception as exc:
                failed += 1
                print(f"[WARN] Failed {path.name}: {type(exc).__name__}: {exc}", flush=True)
                continue
            if num_agents < 0:
                skipped += 1
            else:
                converted += 1
                print(
                    f"[OK] {scenario_id}.pkl agents={num_agents} "
                    f"sim_agents={num_sim_agents} road_edges={num_road_edges} "
                    f"progress={converted + skipped + failed}/{len(files)}",
                    flush=True,
                )

    print(
        f"[DONE] converted={converted} skipped={skipped} failed={failed} "
        f"output_dir={output_dir}",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
