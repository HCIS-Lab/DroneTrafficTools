#!/usr/bin/env python3
"""
Convert ScenarioNet-formatted pickles to Scenario Dreamer autoencoder preprocess pickles.
"""
import glob

import argparse
import math
import os
import pickle
import random
from typing import Dict, List

import numpy as np


def load_scenarionet_pickle(path: str) -> Dict:
    with open(path, "rb") as f:
        scenario = pickle.load(f)
    return scenario


def build_waymo_raw_dict(scenario: Dict, agent_type_map: Dict[str, str]) -> Dict:
    if "tracks" not in scenario or "map_features" not in scenario:
        raise KeyError("ScenarioNet pickle missing 'tracks' or 'map_features'.")

    tracks = scenario["tracks"]

    metadata = scenario.get("metadata", {})
    sdc_track_index = metadata.get("sdc_track_index")
    sdc_id = metadata.get("sdc_id")
    tracks_numeric: Dict[int, Dict] = {}
    warnings = []
    for key, value in tracks.items():
        key_str = str(key)
        if key_str.isdigit():
            numeric_key = int(key_str)
        elif sdc_id is not None and key_str == str(sdc_id):
            numeric_key = int(key_str) if key_str.isdigit() else int(sdc_track_index)
        elif key_str.lower() == "ego" and sdc_track_index is not None:
            numeric_key = int(sdc_track_index)
        else:
            warnings.append(key_str)
            continue
        if numeric_key in tracks_numeric:
            warnings.append(f"{key_str}(duplicate)")
            continue
        tracks_numeric[numeric_key] = value
    if warnings:
        pass
        #print(f"[WARN] skipped non-numeric track ids {warnings}")

    track_keys = sorted(tracks_numeric.keys())

    objects_sorted: List[Dict] = []
    key_to_idx: Dict[str, int] = {}

    for idx, track_id in enumerate(track_keys):
        track = tracks_numeric[track_id]
        state = track["state"]
        T = state["position"].shape[0]
        position_10hz = state["position"]
        heading_10hz = state["heading"]
        valid_10hz = state["valid"]
        if "velocity" in state:
            velocity_10hz = state["velocity"]
        else:
            velocity_10hz = np.zeros((T, 2), dtype=np.float32)

        position = [{"x": float(pos[0]), "y": float(pos[1])} for pos in position_10hz]
        velocity = [{"x": float(vel[0]), "y": float(vel[1])} for vel in velocity_10hz]
        headings_deg = list(np.degrees(heading_10hz))

        length_series = state.get("length")
        width_series = state.get("width")
        current_time_index = int(metadata.get("current_time_index", min(10, max(T - 1, 0))))
        size_timestep = min(current_time_index, T - 1)

        if length_series is None:
            avg_length = track["metadata"].get("avg_height_m", 4.5)
            length_value = float(avg_length)
        else:
            length_value = float(length_series[min(size_timestep, len(length_series) - 1)])

        if width_series is None:
            avg_width = track["metadata"].get("avg_width_m", 1.8)
            width_value = float(avg_width)
        else:
            width_value = float(width_series[min(size_timestep, len(width_series) - 1)])

        waymo_type = agent_type_map.get(track["type"].upper(), "other")

        # ═══════════════════════════════════════════════════════════════
        # DEBUG: Check if heading changes over time for MULTIPLE agents
        # ═══════════════════════════════════════════════════════════════
        if idx < 5 and len(headings_deg) > 1:
            heading_std = np.std(heading_10hz)
            heading_range = np.degrees(heading_10hz.max() - heading_10hz.min())
            pos_x_range = position_10hz[:, 0].max() - position_10hz[:, 0].min()
            pos_y_range = position_10hz[:, 1].max() - position_10hz[:, 1].min()

            # Only print if there's actual movement
            is_moving = pos_x_range > 0.1 or pos_y_range > 0.1 or heading_range > 0.1

            print(f"[DEBUG] Agent {idx} (track {track_id}, type={waymo_type}):")
            print(f"  Timesteps: {T}")
            print(f"  {'✓ MOVING' if is_moving else '✗ STATIC'}")
            print(f"  Heading: range={heading_range:.2f}°, std={np.degrees(heading_std):.2f}°")
            print(f"  Position: ΔX={pos_x_range:.2f}m, ΔY={pos_y_range:.2f}m")

        obj = {
            "position": position,
            "velocity": velocity,
            "heading": headings_deg,
            "length": length_value,
            "width": width_value,
            "valid": [bool(v) for v in valid_10hz],
            "type": waymo_type,
        }
        objects_sorted.append(obj)
        key_to_idx[str(track_id)] = idx

    time_stamps = scenario["metadata"].get("ts")
    if time_stamps is not None:
        time_stamps = np.asarray(time_stamps, dtype=np.float32)

    map_features = scenario["map_features"]
    lanes = {}
    pre_pairs = {}
    suc_pairs = {}
    left_pairs = {}
    right_pairs = {}

    for lane_id_str, feature in map_features.items():
        if feature["type"] != "LANE_SURFACE_STREET":
            continue

        lane_id = int(lane_id_str)
        lanes[lane_id] = np.asarray(feature["polyline"])[:, :2]
        pre_pairs[lane_id] = [int(x) for x in feature.get("entry_lanes", [])]
        suc_pairs[lane_id] = [int(x) for x in feature.get("exit_lanes", [])]
        left_pairs[lane_id] = [int(x["feature_id"]) if isinstance(x, dict) else int(x) for x in feature.get("left_neighbor", [])]
        right_pairs[lane_id] = [int(x["feature_id"]) if isinstance(x, dict) else int(x) for x in feature.get("right_neighbor", [])]

    lane_graph = {
        "lanes": lanes,
        "pre_pairs": pre_pairs,
        "suc_pairs": suc_pairs,
        "left_pairs": left_pairs,
        "right_pairs": right_pairs,
    }

    sdc_key = str(metadata.get("sdc_id", metadata.get("sdc_track_index")))
    if sdc_key not in key_to_idx:
        raise ValueError(f"SDC track id {sdc_key} not found in tracks.")
    av_idx = key_to_idx[sdc_key]

    return {
        "objects": objects_sorted,
        "lane_graph": lane_graph,
        "av_idx": av_idx,
        "time_stamps": time_stamps,
    }


def convert_file(src_path: str, tmp_dir: str, dataset_obj) -> bool:
    scenario = load_scenarionet_pickle(src_path)

    agent_type_map = {
        "VEHICLE": "vehicle",
        "PEDESTRIAN": "pedestrian",
        "CYCLIST": "cyclist",
    }


    try:
        raw_dict = build_waymo_raw_dict(scenario, agent_type_map)
    except KeyError as e:
        print(f"[WARN] {src_path} missing required key ({e}), skipping.")
        return False

    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(dataset_obj.preprocessed_dir, exist_ok=True)

    tmp_basename = f"_tmp_{os.getpid()}_{os.path.splitext(os.path.basename(src_path))[0]}"
    tmp_path = os.path.join(tmp_dir, f"{tmp_basename}.pkl")
    with open(tmp_path, "wb") as f:
        pickle.dump(raw_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    original_files = list(dataset_obj.files)
    try:
        dataset_obj.files = [tmp_path]
        result = dataset_obj.get(0)
    finally:
        dataset_obj.files = original_files
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    pre_dir = dataset_obj.preprocessed_dir
    original_base = os.path.splitext(os.path.basename(src_path))[0]
    tmp_pattern = os.path.join(pre_dir, f"{tmp_basename}_*.pkl")

    for tmp_file in glob.glob(tmp_pattern):
        new_filename = os.path.basename(tmp_file).replace(tmp_basename, original_base, 1)
        new_path = os.path.join(pre_dir, new_filename)
        os.rename(tmp_file, new_path)

    if isinstance(result, dict):
        if not result.get("valid_scene", False):
            print(f"[WARN] Scene {src_path} deemed invalid, skipping.")
            return False
        return True

    return True



def main():
    from omegaconf import OmegaConf
    from datasets.waymo.dataset_autoencoder_waymo_temporal import WaymoDatasetAutoEncoder

    parser = argparse.ArgumentParser(
        description="Convert ScenarioNet pickles into Scenario Dreamer preprocess pickles."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing ScenarioNet *.pkl")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save Scenario Dreamer preprocess pickles (train/val/test subfolders will be created)",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="cfgs/dataset/waymo_autoencoder_temporal.yaml",
        help="Hydra dataset config path",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Portion of scenes assigned to the train split")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Portion of scenes assigned to the validation split")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Portion of scenes assigned to the test split")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic splitting")
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6):
        parser.error("Train/val/test ratios must sum to 1.0.")

    auto_cfg = OmegaConf.load(args.cfg_path)
    base_cfg_path = os.path.join(os.path.dirname(args.cfg_path), "waymo_base.yaml")
    base_cfg = OmegaConf.load(base_cfg_path)
    dataset_cfg = OmegaConf.merge(base_cfg, auto_cfg)
    if "defaults" in dataset_cfg:
        del dataset_cfg["defaults"]
    OmegaConf.set_struct(dataset_cfg, False)
    dataset_cfg.dataset_path = args.input_dir
    dataset_cfg.preprocess = False
    dataset_cfg.preprocess_dir = args.output_dir
    OmegaConf.set_struct(dataset_cfg, True)

    os.makedirs(args.output_dir, exist_ok=True)

    filenames = [fname for fname in os.listdir(args.input_dir) if fname.endswith(".pkl")]
    if not filenames:
        print(f"No ScenarioNet pickles found in {args.input_dir}.")
        return

    rng = random.Random(args.seed)
    rng.shuffle(filenames)

    total_files = len(filenames)
    train_count = int(total_files * args.train_ratio)
    val_count = int(total_files * args.val_ratio)
    split_to_files = {
        "train": sorted(filenames[:train_count]),
        "val": sorted(filenames[train_count:train_count + val_count]),
        "test": sorted(filenames[train_count + val_count:]),
    }

    dataset_objs = {}
    for split_name in split_to_files.keys():
        cfg_clone = OmegaConf.create(OmegaConf.to_container(dataset_cfg, resolve=True))
        OmegaConf.set_struct(cfg_clone, True)
        dataset_objs[split_name] = WaymoDatasetAutoEncoder(cfg_clone, split_name=split_name, mode="eval")
        os.makedirs(dataset_objs[split_name].preprocessed_dir, exist_ok=True)

    converted_counts = {split: 0 for split in split_to_files}

    for split_name, split_files in split_to_files.items():
        dataset_obj = dataset_objs[split_name]
        for filename in split_files:
            src_path = os.path.join(args.input_dir, filename)
            success = convert_file(src_path, args.output_dir, dataset_obj)
            if success:
                converted_counts[split_name] += 1

    total_converted = sum(converted_counts.values())
    print("Conversion completed.")
    for split_name in ("train", "val", "test"):
        target_dir = dataset_objs[split_name].preprocessed_dir
        converted = converted_counts[split_name]
        assigned = len(split_to_files[split_name])
        print(f"  {split_name}: {converted}/{assigned} scenes -> {target_dir}")
    print(f"Total converted: {total_converted} ScenarioNet scenes.")


if __name__ == "__main__":
    main()
