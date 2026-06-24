#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Literal


SCENARIONET_PREFIX = "sd_HetroD_1.0_"
SCENARIO_RE = re.compile(r"^(?P<date>\d+)_loc(?P<location>\d+)_seg(?P<segment>\d+)_ego_(?P<ego>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create synchronized train/valid/test splits for HetroD WOSAC GT and ScenarioNet PKLs.",
    )
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--scenarionet-dir", type=Path, required=True)
    parser.add_argument(
        "--metrics-json",
        type=Path,
        required=True,
        help="Full GT-oracle metrics JSON. Old runners may store no-selected scenarios under errors.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--test-holdout-locations",
        default="2",
        help=(
            "Comma-separated location ids to reserve for test. Default loc2 is "
            "the only location with no no-selected scenarios in the current data."
        ),
    )
    parser.add_argument("--seed", type=str, default="hetrod-v1")
    parser.add_argument(
        "--materialize",
        choices=("symlink", "copy", "none"),
        default="symlink",
        help="How to populate split directories. Manifests are always written.",
    )
    parser.add_argument("--overwrite-links", action="store_true")
    return parser.parse_args()


def scenario_id_from_gt(path: Path) -> str:
    return path.stem


def scenarionet_path_for_id(scenarionet_dir: Path, scenario_id: str) -> Path:
    return scenarionet_dir / f"{SCENARIONET_PREFIX}{scenario_id}.pkl"


def location_of(scenario_id: str) -> str:
    match = SCENARIO_RE.match(scenario_id)
    if not match:
        return "unknown"
    return match.group("location")


def stable_key(seed: str, scenario_id: str) -> str:
    return hashlib.blake2b(f"{seed}:{scenario_id}".encode("utf-8"), digest_size=16).hexdigest()


def _difficulty_bin(score: float, low_threshold: float, high_threshold: float) -> str:
    if score >= high_threshold:
        return "high"
    if score >= low_threshold:
        return "medium"
    return "normal"


def _scenario_type_counts(report: dict) -> dict[str, int]:
    metric = report["kinematic_realism"]["metrics"]["linear_speed"]
    return {
        type_name: int(metric["num_agents_by_type"].get(type_name, 0))
        for type_name in ("vehicle", "two_wheeler", "pedestrian")
    }


def _scenario_pair_counts(report: dict) -> dict[str, int]:
    pair_scores = report["cross_type_interaction"].get("pair_type_scores", {})
    return {
        pair_name: int(pair_scores.get(pair_name, {}).get("num_pairs", 0))
        for pair_name in (
            "vehicle_pedestrian",
            "vehicle_two_wheeler",
            "pedestrian_two_wheeler",
        )
    }


def _raw_challenge_score(report: dict) -> float:
    selected = float(report.get("num_selected_agents", 0))
    included_pairs = float(report["cross_type_interaction"].get("num_included_pairs", 0))
    pair_counts = _scenario_pair_counts(report)
    pedestrian_pairs = float(
        pair_counts["vehicle_pedestrian"] + pair_counts["pedestrian_two_wheeler"]
    )
    safety_risk = 1.0 - float(report["safety"]["score"])
    return (
        selected
        + 0.08 * included_pairs
        + 0.20 * pedestrian_pairs
        + 30.0 * safety_risk
    )


def load_metrics_metadata(metrics_json: Path) -> tuple[set[str], dict[str, dict]]:
    with metrics_json.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    skipped = {
        item["scenario_id"]
        for item in report.get("skipped_scenarios", [])
        if item.get("status") == "skipped_no_selected_agents"
    }
    for item in report.get("errors", []):
        if "no agents selected by the HetroD filters" in item.get("error", ""):
            skipped.add(item["scenario_id"])
    raw_scores = {
        item["scenario_id"]: _raw_challenge_score(item)
        for item in report.get("scenarios", [])
    }
    if raw_scores:
        ordered = sorted(raw_scores.values())
        low_threshold = ordered[int(0.33 * (len(ordered) - 1))]
        high_threshold = ordered[int(0.67 * (len(ordered) - 1))]
    else:
        low_threshold = high_threshold = 0.0

    metadata = {}
    for item in report.get("scenarios", []):
        scenario_id = item["scenario_id"]
        challenge_score = raw_scores[scenario_id]
        metadata[scenario_id] = {
            "scenario_id": scenario_id,
            "location": location_of(scenario_id),
            "num_selected_agents": int(item.get("num_selected_agents", 0)),
            "type_counts": _scenario_type_counts(item),
            "pair_counts": _scenario_pair_counts(item),
            "num_included_pairs": int(item["cross_type_interaction"].get("num_included_pairs", 0)),
            "safety_score": float(item["safety"]["score"]),
            "challenge_score": challenge_score,
            "difficulty_bin": _difficulty_bin(challenge_score, low_threshold, high_threshold),
        }
    return skipped, metadata


def _take_ordered(
    ids: list[str],
    *,
    stats: dict[str, dict],
    count: int,
    seed: str,
    difficulty: str,
) -> list[str]:
    if count <= 0:
        return []
    if difficulty == "high":
        key = lambda sid: (-stats[sid]["challenge_score"], stable_key(seed, sid))
    elif difficulty == "normal":
        key = lambda sid: (stats[sid]["challenge_score"], stable_key(seed, sid))
    else:
        key = lambda sid: (
            abs(stats[sid]["challenge_score"]),
            stable_key(seed, sid),
        )
    return sorted(ids, key=key)[:count]


def _select_by_difficulty_mix(
    candidates: set[str],
    *,
    stats: dict[str, dict],
    target: int,
    fractions: dict[str, float],
    seed: str,
) -> set[str]:
    target = min(target, len(candidates))
    if target <= 0:
        return set()

    by_bin = {
        name: [sid for sid in candidates if stats[sid]["difficulty_bin"] == name]
        for name in ("normal", "medium", "high")
    }
    selected: set[str] = set()
    for name in ("high", "medium", "normal"):
        quota = round(target * fractions.get(name, 0.0))
        picked = _take_ordered(
            [sid for sid in by_bin[name] if sid not in selected],
            stats=stats,
            count=quota,
            seed=f"{seed}:{name}",
            difficulty=name,
        )
        selected.update(picked)

    if len(selected) < target:
        remainder = sorted(
            candidates - selected,
            key=lambda sid: (
                -stats[sid]["challenge_score"],
                stable_key(f"{seed}:remainder", sid),
            ),
        )
        selected.update(remainder[: target - len(selected)])
    elif len(selected) > target:
        selected = set(
            sorted(
                selected,
                key=lambda sid: (
                    -stats[sid]["challenge_score"],
                    stable_key(f"{seed}:trim", sid),
                ),
            )[:target]
        )
    return selected


def assign_challenge_splits(
    all_ids: list[str],
    no_selected_ids: set[str],
    stats: dict[str, dict],
    *,
    valid_ratio: float,
    test_ratio: float,
    seed: str,
    test_holdout_locations: set[str],
) -> dict[str, list[str]]:
    if not 0.0 <= valid_ratio < 1.0 or not 0.0 <= test_ratio < 1.0:
        raise ValueError("Split ratios must be in [0, 1).")
    if valid_ratio + test_ratio >= 1.0:
        raise ValueError("valid_ratio + test_ratio must be < 1.")

    all_set = set(all_ids)
    evaluable = all_set - no_selected_ids
    missing_stats = sorted(evaluable - set(stats))
    if missing_stats:
        raise ValueError(f"Missing metrics stats for evaluable scenarios: {missing_stats[:10]}")

    target_valid = round(len(evaluable) * valid_ratio)
    target_test = round(len(evaluable) * test_ratio)

    holdout_test = {
        sid
        for sid in evaluable
        if location_of(sid) in test_holdout_locations
    }
    test = set(holdout_test)
    test_pool = evaluable - test
    extra_test_needed = max(0, target_test - len(test))
    test.update(
        _select_by_difficulty_mix(
            test_pool,
            stats=stats,
            target=extra_test_needed,
            fractions={"normal": 0.20, "medium": 0.50, "high": 0.30},
            seed=f"{seed}:test",
        )
    )

    valid_pool = evaluable - test
    valid = _select_by_difficulty_mix(
        valid_pool,
        stats=stats,
        target=target_valid,
        fractions={"normal": 0.25, "medium": 0.55, "high": 0.20},
        seed=f"{seed}:valid",
    )

    train = all_set - test - valid
    train.update(no_selected_ids)
    return {
        "train": sorted(train),
        "valid": sorted(valid),
        "test": sorted(test),
    }


def write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")


def materialize_one(
    src: Path,
    dst: Path,
    *,
    mode: Literal["symlink", "copy", "none"],
    overwrite_links: bool,
) -> None:
    if mode == "none":
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite_links:
            return
        if dst.is_dir() and not dst.is_symlink():
            raise IsADirectoryError(f"Refusing to overwrite directory: {dst}")
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported materialize mode: {mode}")


def summarize_split(
    splits: dict[str, list[str]],
    no_selected_ids: set[str],
    stats: dict[str, dict],
) -> dict[str, dict]:
    summary = {}
    for name, ids in splits.items():
        location_counts = Counter(location_of(sid) for sid in ids)
        evaluable_ids = [sid for sid in ids if sid in stats]
        difficulty_counts = Counter(stats[sid]["difficulty_bin"] for sid in evaluable_ids)
        type_counts = Counter()
        pair_counts = Counter()
        for sid in evaluable_ids:
            type_counts.update(stats[sid]["type_counts"])
            pair_counts.update(stats[sid]["pair_counts"])
        summary[name] = {
            "num_scenarios": len(ids),
            "num_evaluable": len(evaluable_ids),
            "num_no_selected_agents": sum(sid in no_selected_ids for sid in ids),
            "location_counts": dict(sorted(location_counts.items())),
            "difficulty_counts": dict(sorted(difficulty_counts.items())),
            "selected_agent_type_counts": dict(sorted(type_counts.items())),
            "cross_type_pair_counts": dict(sorted(pair_counts.items())),
            "mean_selected_agents": (
                sum(stats[sid]["num_selected_agents"] for sid in evaluable_ids) / len(evaluable_ids)
                if evaluable_ids
                else 0.0
            ),
            "mean_included_pairs": (
                sum(stats[sid]["num_included_pairs"] for sid in evaluable_ids) / len(evaluable_ids)
                if evaluable_ids
                else 0.0
            ),
        }
    return summary


def main() -> int:
    args = parse_args()
    gt_dir = args.gt_dir.resolve()
    scenarionet_dir = args.scenarionet_dir.resolve()
    output_dir = args.output_dir.resolve()

    gt_paths = sorted(gt_dir.glob("*.pkl"))
    if not gt_paths:
        raise FileNotFoundError(f"No GT PKLs found in {gt_dir}")
    all_ids = [scenario_id_from_gt(path) for path in gt_paths]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate scenario ids found in GT directory.")

    missing_scenarionet = [
        sid for sid in all_ids if not scenarionet_path_for_id(scenarionet_dir, sid).is_file()
    ]
    if missing_scenarionet:
        preview = ", ".join(missing_scenarionet[:10])
        raise FileNotFoundError(
            f"Missing ScenarioNet files for {len(missing_scenarionet)} scenario(s): {preview}"
        )

    no_selected_ids, metrics_stats = load_metrics_metadata(args.metrics_json)
    unknown_no_selected = sorted(no_selected_ids - set(all_ids))
    if unknown_no_selected:
        preview = ", ".join(unknown_no_selected[:10])
        raise ValueError(
            f"Metrics JSON references no-selected ids not present in GT dir: {preview}"
        )

    test_holdout_locations = {
        item.strip().replace("loc", "")
        for item in args.test_holdout_locations.split(",")
        if item.strip()
    }
    splits = assign_challenge_splits(
        all_ids,
        no_selected_ids,
        metrics_stats,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        test_holdout_locations=test_holdout_locations,
    )
    evaluable_ids = sorted(set(all_ids) - no_selected_ids)

    assigned = set().union(*[set(ids) for ids in splits.values()])
    if assigned != set(all_ids):
        raise AssertionError("Split assignment does not cover all scenarios exactly once.")
    if set(splits["train"]) & set(splits["valid"]) or set(splits["train"]) & set(splits["test"]) or set(splits["valid"]) & set(splits["test"]):
        raise AssertionError("Split assignment contains overlap.")

    manifests_dir = output_dir / "manifests"
    for split_name, ids in splits.items():
        write_lines(manifests_dir / f"{split_name}.txt", ids)
        write_lines(
            manifests_dir / f"{split_name}_gt_paths.txt",
            [str(gt_dir / f"{sid}.pkl") for sid in ids],
        )
        write_lines(
            manifests_dir / f"{split_name}_scenarionet_paths.txt",
            [str(scenarionet_path_for_id(scenarionet_dir, sid)) for sid in ids],
        )

        for sid in ids:
            materialize_one(
                gt_dir / f"{sid}.pkl",
                output_dir / "gt" / split_name / f"{sid}.pkl",
                mode=args.materialize,
                overwrite_links=args.overwrite_links,
            )
            materialize_one(
                scenarionet_path_for_id(scenarionet_dir, sid),
                output_dir / "scenarionet" / split_name / f"{SCENARIONET_PREFIX}{sid}.pkl",
                mode=args.materialize,
                overwrite_links=args.overwrite_links,
            )

    summary = {
        "source": {
            "gt_dir": str(gt_dir),
            "scenarionet_dir": str(scenarionet_dir),
            "metrics_json": str(args.metrics_json.resolve()),
        },
        "config": {
            "valid_ratio": args.valid_ratio,
            "test_ratio": args.test_ratio,
            "train_ratio_effective": len(splits["train"]) / len(all_ids),
            "seed": args.seed,
            "materialize": args.materialize,
            "no_selected_policy": "force_train",
            "test_holdout_locations": sorted(test_holdout_locations),
            "evaluable_assignment": "loc_holdout_plus_metrics_aware_difficulty_mix",
            "test_extra_mix": {"normal": 0.20, "medium": 0.50, "high": 0.30},
            "valid_mix": {"normal": 0.25, "medium": 0.55, "high": 0.20},
        },
        "total": {
            "num_scenarios": len(all_ids),
            "num_evaluable": len(evaluable_ids),
            "num_no_selected_agents": len(no_selected_ids),
        },
        "splits": summarize_split(splits, no_selected_ids, metrics_stats),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
