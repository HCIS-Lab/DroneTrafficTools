#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCENARIONET_COLOR = {
    "LANE_SURFACE_STREET": "#666666",
    "LANE_FREEWAY": "#444444",
    "LANE_BIKE_LANE": "#2f855a",
    "ROAD_EDGE_BOUNDARY": "#111111",
    "ROAD_EDGE_MEDIAN": "#222222",
    "ROAD_LINE_BROKEN_SINGLE_WHITE": "#4299e1",
    "ROAD_LINE_BROKEN_SINGLE_YELLOW": "#d69e2e",
    "ROAD_LINE_BROKEN_DOUBLE_YELLOW": "#b7791f",
    "ROAD_LINE_SOLID_SINGLE_WHITE": "#3182ce",
    "ROAD_LINE_SOLID_SINGLE_YELLOW": "#dd6b20",
    "ROAD_LINE_SOLID_DOUBLE_WHITE": "#2b6cb0",
    "ROAD_LINE_SOLID_DOUBLE_YELLOW": "#c05621",
    "ROAD_LINE_PASSING_DOUBLE_YELLOW": "#9c4221",
    "CROSSWALK": "#e53e3e",
}

SMART_PT_COLOR = {
    1: "#666666",
    4: "#111111",
    5: "#333333",
    6: "#4299e1",
    7: "#3182ce",
    8: "#2b6cb0",
    9: "#e53e3e",
}

SMART_PL_LABEL = {
    0: "lane",
    1: "road_edge",
    2: "road_line",
    3: "crosswalk",
}


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def scenario_centerlines(scenario):
    lines = []
    for feat_id, feat in scenario["map_features"].items():
        feat_type = str(feat.get("type", ""))
        poly = feat.get("polyline")
        if isinstance(poly, np.ndarray) and poly.ndim == 2 and poly.shape[0] >= 2 and poly.shape[1] >= 2:
            lines.append((str(feat_id), feat_type, poly[:, :2]))
    return lines


def smart_segments(data):
    traj_pos = data["map_save"]["traj_pos"]
    pt_type = data["pt_token"]["type"]
    pl_type = data["pt_token"]["pl_type"]

    if hasattr(traj_pos, "numpy"):
        traj_pos = traj_pos.numpy()
    if hasattr(pt_type, "numpy"):
        pt_type = pt_type.numpy()
    if hasattr(pl_type, "numpy"):
        pl_type = pl_type.numpy()

    return traj_pos, pt_type, pl_type


def collect_bounds(lines, traj_pos):
    pts = []
    for _, _, xy in lines:
        pts.append(xy)
    if traj_pos is not None and len(traj_pos) > 0:
        pts.append(traj_pos.reshape(-1, 2))
    pts = np.concatenate(pts, axis=0)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    span = max(max_xy - min_xy)
    pad = max(20.0, 0.1 * span)
    return (
        center[0] - span / 2 - pad,
        center[0] + span / 2 + pad,
        center[1] - span / 2 - pad,
        center[1] + span / 2 + pad,
    )


def add_scenarionet_legend(ax):
    labels = [
        "LANE_SURFACE_STREET",
        "ROAD_EDGE_BOUNDARY",
        "ROAD_LINE_BROKEN_SINGLE_WHITE",
        "ROAD_LINE_SOLID_SINGLE_WHITE",
        "ROAD_LINE_SOLID_DOUBLE_YELLOW",
        "CROSSWALK",
    ]
    handles = [
        plt.Line2D([0], [0], color=SCENARIONET_COLOR[k], lw=2, label=k)
        for k in labels
        if k in SCENARIONET_COLOR
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)


def add_smart_legend(ax):
    handles = [
        plt.Line2D([0], [0], color=SMART_PT_COLOR[k], lw=2, label=f"pt_type={k}")
        for k in sorted(SMART_PT_COLOR)
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)


def draw_debug_figure(scenario, smart_data, output_path: Path):
    lines = scenario_centerlines(scenario)
    traj_pos, pt_type, pl_type = smart_segments(smart_data)
    x0, x1, y0, y1 = collect_bounds(lines, traj_pos)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=120)

    ax = axes[0]
    for _, feat_type, xy in lines:
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=SCENARIONET_COLOR.get(feat_type, "#999999"),
            linewidth=1.2,
            alpha=0.95,
        )
    ax.set_title(f"ScenarioNet Map\n{scenario['id']}", fontsize=12)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    add_scenarionet_legend(ax)

    ax = axes[1]
    for idx in range(len(traj_pos)):
        seg = traj_pos[idx]
        ax.plot(
            seg[:, 0],
            seg[:, 1],
            color=SMART_PT_COLOR.get(int(pt_type[idx]), "#999999"),
            linewidth=2.0,
            alpha=0.95,
        )
        ax.scatter(
            seg[0, 0],
            seg[0, 1],
            s=6,
            color=SMART_PT_COLOR.get(int(pt_type[idx]), "#999999"),
            alpha=0.8,
        )
    uniq_pl = sorted(set(int(x) for x in pl_type.tolist()))
    pl_text = ", ".join(f"{k}:{SMART_PL_LABEL.get(k, 'unknown')}" for k in uniq_pl)
    ax.set_title(f"SMART Map Tokens\npl_type: {pl_text}", fontsize=12)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    add_smart_legend(ax)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visual debug for ScenarioNet -> SMART map conversion.")
    parser.add_argument("--scenarionet_pkl", required=True, help="Input ScenarioNet pkl path.")
    parser.add_argument("--smart_pkl", required=True, help="Converted SMART cache pkl path.")
    parser.add_argument("--output", required=True, help="Output png path.")
    args = parser.parse_args()

    scenarionet_pkl = Path(args.scenarionet_pkl)
    smart_pkl = Path(args.smart_pkl)
    output = Path(args.output)

    scenario = load_pickle(scenarionet_pkl)
    smart_data = load_pickle(smart_pkl)
    draw_debug_figure(scenario, smart_data, output)
    print(output)


if __name__ == "__main__":
    main()
