#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree


ROAD_EDGE_TYPES = {"ROAD_EDGE_BOUNDARY", "ROAD_EDGE_MEDIAN"}
PANEL_WIDTH = 720
PANEL_HEIGHT = 720
HEADER_HEIGHT = 78
CANVAS_MARGIN = 36


def _install_numpy_pickle_aliases() -> None:
    if hasattr(np, "_core"):
        return

    import numpy.core
    import numpy.core.multiarray
    import numpy.core.numeric

    sys.modules.setdefault("numpy._core", numpy.core)
    sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)


def load_pickle(path: Path):
    _install_numpy_pickle_aliases()
    with path.open("rb") as handle:
        return pickle.load(handle)


def scenario_road_edges(scenario) -> list[np.ndarray]:
    edges = []
    for feature in scenario["map_features"].values():
        if str(feature.get("type", "")) not in ROAD_EDGE_TYPES:
            continue
        polyline = np.asarray(feature.get("polyline"), dtype=np.float32)
        if polyline.ndim == 2 and polyline.shape[0] >= 2:
            edges.append(polyline[:, :2])
    return edges


def smart_road_edges(cache) -> list[np.ndarray]:
    positions = cache["map_save"]["traj_pos"].detach().cpu().numpy()
    polygon_types = cache["pt_token"]["pl_type"].detach().cpu().numpy()
    return [positions[index] for index in np.flatnonzero(polygon_types == 1)]


def load_smart_road_edges(path: Path) -> list[np.ndarray]:
    if path.suffix == ".npz":
        with np.load(path) as data:
            offsets = data["offsets"]
            points = data["points"]
        return [
            points[offsets[index] : offsets[index + 1]]
            for index in range(len(offsets) - 1)
        ]
    return smart_road_edges(load_pickle(path))


def densify(lines: list[np.ndarray], spacing: float = 0.25) -> np.ndarray:
    sampled = []
    for line in lines:
        for start, end in zip(line[:-1], line[1:]):
            distance = float(np.linalg.norm(end - start))
            count = max(2, int(np.ceil(distance / spacing)) + 1)
            alpha = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
            sampled.append(start[None] * (1.0 - alpha) + end[None] * alpha)
    if not sampled:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(sampled, axis=0)


def symmetric_distance(source: list[np.ndarray], converted: list[np.ndarray]) -> dict:
    source_points = densify(source)
    converted_points = densify(converted)
    if len(source_points) == 0 or len(converted_points) == 0:
        return {"mean": float("inf"), "p95": float("inf"), "max": float("inf")}

    source_to_converted = cKDTree(converted_points).query(source_points, workers=-1)[0]
    converted_to_source = cKDTree(source_points).query(converted_points, workers=-1)[0]
    distances = np.concatenate([source_to_converted, converted_to_source])
    return {
        "mean": float(np.mean(distances)),
        "p95": float(np.percentile(distances, 95)),
        "max": float(np.max(distances)),
    }


def compute_view(lines: list[np.ndarray]) -> tuple[np.ndarray, float]:
    points = np.concatenate(lines, axis=0)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) * 0.5
    span = max(float(np.max(maximum - minimum)), 1.0)
    return center, span * 1.10


def to_pixels(points: np.ndarray, center: np.ndarray, span: float) -> np.ndarray:
    draw_size = PANEL_WIDTH - 2 * CANVAS_MARGIN
    minimum = center - span * 0.5
    pixels = (points - minimum) * (draw_size / span)
    pixels[:, 0] += CANVAS_MARGIN
    pixels[:, 1] = PANEL_HEIGHT - CANVAS_MARGIN - pixels[:, 1]
    return np.rint(pixels).astype(np.int32)


def draw_lines(
    panel: np.ndarray,
    lines: list[np.ndarray],
    center: np.ndarray,
    span: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    for line in lines:
        pixels = to_pixels(line.copy(), center, span)
        cv2.polylines(panel, [pixels], False, color, thickness, cv2.LINE_AA)


def put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.58,
    color: tuple[int, int, int] = (35, 35, 35),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def render(
    scenario_path: Path,
    smart_path: Path,
    output_path: Path,
    gt_edges_path: Path | None = None,
) -> dict:
    scenario = load_pickle(scenario_path)
    source = scenario_road_edges(scenario)
    converted = load_smart_road_edges(smart_path)
    metrics = symmetric_distance(source, converted)

    optional_gt = None
    gt_metrics = None
    if gt_edges_path is not None:
        with np.load(gt_edges_path) as gt_data:
            offsets = gt_data["offsets"]
            points = gt_data["points"]
        optional_gt = [
            points[offsets[index] : offsets[index + 1]]
            for index in range(len(offsets) - 1)
        ]
        gt_metrics = symmetric_distance(optional_gt, converted)

    all_lines = source + converted + (optional_gt or [])
    center, span = compute_view(all_lines)
    panels = []

    source_panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 255, dtype=np.uint8)
    draw_lines(source_panel, source, center, span, (220, 100, 20), 2)
    put_text(source_panel, "ScenarioNet source road edges", (24, 30), 0.70, thickness=2)
    put_text(source_panel, f"{len(source)} polylines", (24, 57))
    panels.append(source_panel)

    smart_panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 255, dtype=np.uint8)
    draw_lines(smart_panel, converted, center, span, (30, 50, 220), 2)
    put_text(smart_panel, "SMART road-edge tokens", (24, 30), 0.70, thickness=2)
    put_text(smart_panel, f"{len(converted)} token segments", (24, 57))
    panels.append(smart_panel)

    overlay_panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 255, dtype=np.uint8)
    draw_lines(overlay_panel, converted, center, span, (30, 50, 220), 4)
    draw_lines(overlay_panel, source, center, span, (220, 100, 20), 2)
    if optional_gt:
        draw_lines(overlay_panel, optional_gt, center, span, (20, 150, 20), 1)
    put_text(overlay_panel, "Overlay", (24, 30), 0.70, thickness=2)
    put_text(
        overlay_panel,
        f"source vs SMART: mean {metrics['mean']:.3f} m, p95 {metrics['p95']:.3f} m",
        (24, 57),
    )
    put_text(overlay_panel, "blue=source  red=SMART  green=WOSAC GT", (24, 82), 0.50)
    if gt_metrics is not None:
        put_text(
            overlay_panel,
            f"WOSAC GT vs SMART: mean {gt_metrics['mean']:.3f} m, p95 {gt_metrics['p95']:.3f} m",
            (24, 107),
            0.50,
        )
    panels.append(overlay_panel)

    image = np.concatenate(panels, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to write {output_path}")

    result = {
        "scenario": scenario["id"],
        "source_polylines": len(source),
        "smart_segments": len(converted),
        "source_vs_smart": metrics,
        "output": str(output_path),
    }
    if gt_metrics is not None:
        result["wosac_gt_polylines"] = len(optional_gt)
        result["wosac_gt_vs_smart"] = gt_metrics
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render ScenarioNet and SMART road-edge conversion side by side."
    )
    parser.add_argument("--scenarionet-pkl", required=True, type=Path)
    parser.add_argument(
        "--smart-pkl",
        required=True,
        type=Path,
        help="SMART cache pickle, or an isolated road-edge NPZ export.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--wosac-gt-edges-npz",
        type=Path,
        help="Optional flattened WOSAC GT road-edge arrays for a third overlay.",
    )
    args = parser.parse_args()

    result = render(
        args.scenarionet_pkl,
        args.smart_pkl,
        args.output,
        args.wosac_gt_edges_npz,
    )
    print(result)


if __name__ == "__main__":
    main()
