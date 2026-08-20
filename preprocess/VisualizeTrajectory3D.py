#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render unified trajectory JSONL with head/camera path and glove 21 points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    from preprocess.Timebase import repeat_counts
except ModuleNotFoundError:
    from Timebase import repeat_counts

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess.VisualizeHandKpts import MP_HAND_BONES


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def hand_points_world(row: Dict, side: Optional[str] = None) -> Optional[np.ndarray]:
    if side is None:
        glove = row.get("glove") or {}
    else:
        glove = (((row.get("hands") or {}).get(side) or {}).get("glove") or {})
    pts = glove.get("kpts_3d_world_m")
    if pts is None:
        return None
    arr = np.asarray(pts, dtype=np.float64)
    if arr.shape[0] < 21 or arr.shape[1] < 3 or not np.isfinite(arr[:21, :3]).all():
        return None
    return arr[:21, :3]


def c2w(row: Dict) -> Optional[np.ndarray]:
    mat = (row.get("head_pose") or {}).get("c2w") or (row.get("camera") or {}).get("c2w")
    if mat is None:
        return None
    arr = np.asarray(mat, dtype=np.float64)
    if arr.shape != (4, 4) or not np.isfinite(arr).all():
        return None
    return arr


def collect_limits(hand_series: List[Optional[np.ndarray]], cam_series: List[Optional[np.ndarray]], pad_ratio: float) -> Dict[str, Tuple[float, float]]:
    chunks = [p for p in hand_series if p is not None]
    cam_pts = [m[:3, 3] for m in cam_series if m is not None]
    if cam_pts:
        chunks.append(np.asarray(cam_pts, dtype=np.float64))
    if not chunks:
        raise RuntimeError("No valid hand/camera points for trajectory visualization.")
    flat = np.concatenate([c.reshape(-1, 3) for c in chunks], axis=0)
    lo = np.percentile(flat, 1, axis=0)
    hi = np.percentile(flat, 99, axis=0)
    span = np.maximum(hi - lo, 1e-3)
    pad = span * pad_ratio
    mid = (lo + hi) * 0.5
    max_span = float(np.max(span + 2.0 * pad))
    half = max_span * 0.5
    return {
        "x": (float(mid[0] - half), float(mid[0] + half)),
        "y": (float(mid[1] - half), float(mid[1] + half)),
        "z": (float(mid[2] - half), float(mid[2] + half)),
    }


def finger_color(i: int) -> str:
    if i <= 4:
        return "#ff8a3d"
    if i <= 8:
        return "#24b36b"
    if i <= 12:
        return "#3b82f6"
    if i <= 16:
        return "#d4a017"
    return "#c061cb"


def hand_color(side: str, joint: int) -> str:
    if side == "left":
        return "#2563eb"
    if side == "right":
        return "#ea580c"
    return finger_color(joint)


def draw_hand_3d(ax, pts: np.ndarray, side: str) -> None:
    for a, b in MP_HAND_BONES:
        ax.plot(
            [pts[a, 0], pts[b, 0]],
            [pts[a, 1], pts[b, 1]],
            [pts[a, 2], pts[b, 2]],
            color=hand_color(side, b),
            linewidth=2.0,
        )
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=[hand_color(side, i) for i in range(21)], s=18,
        edgecolors="black", linewidths=0.25, label=f"{side} hand" if side in {"left", "right"} else "hand",
    )


def draw_hand_2d(ax, pts: np.ndarray, dims: Tuple[int, int], side: str) -> None:
    a_dim, b_dim = dims
    for a, b in MP_HAND_BONES:
        ax.plot([pts[a, a_dim], pts[b, a_dim]], [pts[a, b_dim], pts[b, b_dim]], color=hand_color(side, b), linewidth=2.0)
    ax.scatter(pts[:, a_dim], pts[:, b_dim], c=[hand_color(side, i) for i in range(21)], s=18, edgecolors="black", linewidths=0.25)


def draw_camera_axes(ax, mat: np.ndarray, scale: float) -> None:
    origin = mat[:3, 3]
    axes = mat[:3, :3]
    colors = ("#ef4444", "#22c55e", "#3b82f6")
    for i, color in enumerate(colors):
        end = origin + axes[:, i] * scale
        ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]], color=color, linewidth=2.5)
    ax.scatter([origin[0]], [origin[1]], [origin[2]], c="black", s=18)


def set_3d_limits(ax, limits: Dict[str, Tuple[float, float]]) -> None:
    ax.set_xlim(*limits["x"])
    ax.set_ylim(*limits["y"])
    ax.set_zlim(*limits["z"])
    ax.set_xlabel("X world (m)")
    ax.set_ylabel("Y world (m)")
    ax.set_zlabel("Z world (m)")


def set_2d_limits(ax, limits: Dict[str, Tuple[float, float]], dims: Tuple[int, int], title: str) -> None:
    keys = ("x", "y", "z")
    labels = ("X", "Y", "Z")
    ax.set_xlim(*limits[keys[dims[0]]])
    ax.set_ylim(*limits[keys[dims[1]]])
    ax.set_xlabel(f"{labels[dims[0]]} world (m)")
    ax.set_ylabel(f"{labels[dims[1]]} world (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")


def render(args: argparse.Namespace) -> Dict:
    rows = read_jsonl(Path(args.trajectory_jsonl).expanduser().resolve())
    if args.max_frames is not None:
        rows = rows[: args.max_frames]
    dual_mode = any(isinstance(row.get("hands"), dict) for row in rows)
    sides = ("left", "right") if dual_mode else ("hand",)
    hand_series = {
        side: [hand_points_world(row, None if side == "hand" else side) for row in rows]
        for side in sides
    }
    cam_series = [c2w(r) for r in rows]
    valid_hands = {side: [p for p in series if p is not None] for side, series in hand_series.items()}
    if not any(valid_hands.values()):
        raise RuntimeError("No valid glove.kpts_3d_world_m in trajectory.")
    all_hand_series = [points for series in hand_series.values() for points in series]
    limits = collect_limits(all_hand_series, cam_series, args.pad_ratio)

    out = Path(args.out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out}")

    fig_w = args.width / 100.0
    fig_h = args.height / 100.0
    last_hand = {side: (values[0] if values else None) for side, values in valid_hands.items()}
    cam_path: List[np.ndarray] = []
    wrist_paths: Dict[str, List[np.ndarray]] = {side: [] for side in sides}
    written = 0
    repeats = repeat_counts(rows, output_fps=args.fps)

    for i, (row, cam) in enumerate(tqdm(list(zip(rows, cam_series)), desc="Rendering trajectory 3D")):
        frame_hands: Dict[str, np.ndarray] = {}
        wrist_trails: Dict[str, np.ndarray] = {}
        for side in sides:
            hand = hand_series[side][i]
            if hand is None:
                hand = last_hand[side]
            else:
                last_hand[side] = hand
            if hand is None:
                continue
            frame_hands[side] = hand
            wrist_paths[side].append(hand[0].copy())
            wrist_trails[side] = np.asarray(wrist_paths[side][-args.trail_len:], dtype=np.float64)
        if cam is not None:
            cam_path.append(cam[:3, 3].copy())
        cam_trail = np.asarray(cam_path[-args.trail_len:], dtype=np.float64) if cam_path else np.zeros((0, 3))

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0])
        ax3d = fig.add_subplot(gs[:, 0], projection="3d")
        ax_xy = fig.add_subplot(gs[0, 1])
        ax_xz = fig.add_subplot(gs[1, 1])
        fig.patch.set_facecolor("white")

        set_3d_limits(ax3d, limits)
        ax3d.view_init(elev=args.elev, azim=args.azim)
        ax3d.set_title("World trajectory")
        if len(cam_trail) > 1:
            ax3d.plot(cam_trail[:, 0], cam_trail[:, 1], cam_trail[:, 2], color="#111111", linewidth=1.4, label="head/camera")
        for side, wrist_trail in wrist_trails.items():
            if len(wrist_trail) > 1:
                ax3d.plot(wrist_trail[:, 0], wrist_trail[:, 1], wrist_trail[:, 2], color=hand_color(side, 0), linewidth=1.4, label=f"{side} wrist")
        for side, hand in frame_hands.items():
            draw_hand_3d(ax3d, hand, side)
        if cam is not None:
            draw_camera_axes(ax3d, cam, args.camera_axis_len)
        ax3d.legend(loc="upper right")

        set_2d_limits(ax_xy, limits, (0, 1), "XY projection")
        set_2d_limits(ax_xz, limits, (0, 2), "XZ projection")
        if len(cam_trail) > 1:
            ax_xy.plot(cam_trail[:, 0], cam_trail[:, 1], color="#111111", linewidth=1.1)
            ax_xz.plot(cam_trail[:, 0], cam_trail[:, 2], color="#111111", linewidth=1.1)
        for side, wrist_trail in wrist_trails.items():
            if len(wrist_trail) > 1:
                ax_xy.plot(wrist_trail[:, 0], wrist_trail[:, 1], color=hand_color(side, 0), linewidth=1.1)
                ax_xz.plot(wrist_trail[:, 0], wrist_trail[:, 2], color=hand_color(side, 0), linewidth=1.1)
        for side, hand in frame_hands.items():
            draw_hand_2d(ax_xy, hand, (0, 1), side)
            draw_hand_2d(ax_xz, hand, (0, 2), side)

        mode_label = "dual hand" if dual_mode else "single hand"
        fig.suptitle(f"trajectory | frame {row.get('frame')} | {mode_label} glove world 21pts", fontsize=14)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        if bgr.shape[1] != args.width or bgr.shape[0] != args.height:
            bgr = cv2.resize(bgr, (args.width, args.height), interpolation=cv2.INTER_AREA)
        for _ in range(repeats[i]):
            writer.write(bgr)
            written += 1
        plt.close(fig)

    writer.release()
    summary = {
        "trajectory_jsonl": str(Path(args.trajectory_jsonl).expanduser().resolve()),
        "out_path": str(out),
        "frames": len(rows),
        "mode": "dual_hand" if dual_mode else "single_hand",
        "valid_hand_world": {side: len(values) for side, values in valid_hands.items()},
        "valid_camera_pose": sum(1 for c in cam_series if c is not None),
        "written": written,
        "source_frames": len(rows),
        "timebase": "rgb_stamp_ns",
        "limits": limits,
    }
    if args.summary_json:
        with Path(args.summary_json).expanduser().resolve().open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
            f.write("\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize unified trajectory in world coordinates.")
    parser.add_argument("--trajectory_jsonl", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--trail_len", type=int, default=80)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--pad_ratio", type=float, default=0.18)
    parser.add_argument("--camera_axis_len", type=float, default=0.05)
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = render(args)
    print(f"[VisualizeTrajectory3D] video: {args.out_path}")
    print(f"[VisualizeTrajectory3D] summary: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
