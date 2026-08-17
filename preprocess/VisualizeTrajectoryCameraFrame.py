#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render glove hand trajectory directly in camera coordinates."""

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


def hand_points_camera(row: Dict) -> Optional[np.ndarray]:
    glove = row.get("glove") or row.get("glove_fk21") or {}
    pts = glove.get("kpts_3d_camera_m")
    if pts is None:
        return None
    arr = np.asarray(pts, dtype=np.float64)
    if arr.shape[0] < 21 or arr.shape[1] < 3 or not np.isfinite(arr[:21, :3]).all():
        return None
    return arr[:21, :3]


def collect_limits(hand_series: List[Optional[np.ndarray]], pad_ratio: float) -> Dict[str, Tuple[float, float]]:
    chunks = [p for p in hand_series if p is not None]
    if not chunks:
        raise RuntimeError("No valid glove camera-frame points for visualization.")
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


def draw_hand_3d(ax, pts: np.ndarray) -> None:
    for a, b in MP_HAND_BONES:
        ax.plot(
            [pts[a, 0], pts[b, 0]],
            [pts[a, 1], pts[b, 1]],
            [pts[a, 2], pts[b, 2]],
            color=finger_color(b),
            linewidth=2.0,
        )
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        c=[finger_color(i) for i in range(21)],
        s=18,
        edgecolors="black",
        linewidths=0.25,
    )


def draw_hand_2d(ax, pts: np.ndarray, dims: Tuple[int, int]) -> None:
    a_dim, b_dim = dims
    for a, b in MP_HAND_BONES:
        ax.plot([pts[a, a_dim], pts[b, a_dim]], [pts[a, b_dim], pts[b, b_dim]], color=finger_color(b), linewidth=2.0)
    ax.scatter(pts[:, a_dim], pts[:, b_dim], c=[finger_color(i) for i in range(21)], s=18, edgecolors="black", linewidths=0.25)


def set_3d_limits(ax, limits: Dict[str, Tuple[float, float]]) -> None:
    ax.set_xlim(*limits["x"])
    ax.set_ylim(*limits["y"])
    ax.set_zlim(*limits["z"])
    ax.set_xlabel("X camera (m)")
    ax.set_ylabel("Y camera (m)")
    ax.set_zlabel("Z camera (m)")


def set_2d_limits(ax, limits: Dict[str, Tuple[float, float]], dims: Tuple[int, int], title: str) -> None:
    keys = ("x", "y", "z")
    labels = ("X", "Y", "Z")
    ax.set_xlim(*limits[keys[dims[0]]])
    ax.set_ylim(*limits[keys[dims[1]]])
    ax.set_xlabel(f"{labels[dims[0]]} camera (m)")
    ax.set_ylabel(f"{labels[dims[1]]} camera (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")


def summarize_steps(points: List[np.ndarray]) -> Dict:
    if len(points) < 2:
        return {}
    pts = np.asarray(points, dtype=np.float64)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return {
        "median": float(np.median(steps)),
        "p95": float(np.percentile(steps, 95)),
        "max": float(steps.max()),
    }


def render(args: argparse.Namespace) -> Dict:
    rows = read_jsonl(Path(args.trajectory_jsonl).expanduser().resolve())
    if args.max_frames is not None:
        rows = rows[: args.max_frames]
    hand_series = [hand_points_camera(r) for r in rows]
    valid_hands = [p for p in hand_series if p is not None]
    if not valid_hands:
        raise RuntimeError("No valid glove.kpts_3d_camera_m in trajectory.")
    limits = collect_limits(hand_series, args.pad_ratio)

    out = Path(args.out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out}")

    fig_w = args.width / 100.0
    fig_h = args.height / 100.0
    last_hand = valid_hands[0]
    wrist_path: List[np.ndarray] = []
    written = 0

    for row, hand in tqdm(list(zip(rows, hand_series)), desc="Rendering camera-frame trajectory"):
        if hand is None:
            hand = last_hand
        else:
            last_hand = hand
        wrist_path.append(hand[0].copy())
        wrist_trail = np.asarray(wrist_path[-args.trail_len:], dtype=np.float64)

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0])
        ax3d = fig.add_subplot(gs[:, 0], projection="3d")
        ax_xy = fig.add_subplot(gs[0, 1])
        ax_xz = fig.add_subplot(gs[1, 1])
        fig.patch.set_facecolor("white")

        set_3d_limits(ax3d, limits)
        ax3d.view_init(elev=args.elev, azim=args.azim)
        ax3d.set_title("Camera-frame hand trajectory")
        if len(wrist_trail) > 1:
            ax3d.plot(wrist_trail[:, 0], wrist_trail[:, 1], wrist_trail[:, 2], color="#8b5cf6", linewidth=1.5, label="wrist")
        ax3d.scatter([0], [0], [0], c="black", s=18, label="camera")
        draw_hand_3d(ax3d, hand)
        ax3d.legend(loc="upper right")

        set_2d_limits(ax_xy, limits, (0, 1), "XY projection")
        set_2d_limits(ax_xz, limits, (0, 2), "XZ projection")
        if len(wrist_trail) > 1:
            ax_xy.plot(wrist_trail[:, 0], wrist_trail[:, 1], color="#8b5cf6", linewidth=1.1)
            ax_xz.plot(wrist_trail[:, 0], wrist_trail[:, 2], color="#8b5cf6", linewidth=1.1)
        ax_xy.scatter([0], [0], c="black", s=18)
        ax_xz.scatter([0], [0], c="black", s=18)
        draw_hand_2d(ax_xy, hand, (0, 1))
        draw_hand_2d(ax_xz, hand, (0, 2))

        fig.suptitle(f"camera frame | frame {row.get('frame')} | glove 21pts", fontsize=14)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        if bgr.shape[1] != args.width or bgr.shape[0] != args.height:
            bgr = cv2.resize(bgr, (args.width, args.height), interpolation=cv2.INTER_AREA)
        writer.write(bgr)
        plt.close(fig)
        written += 1

    writer.release()
    summary = {
        "trajectory_jsonl": str(Path(args.trajectory_jsonl).expanduser().resolve()),
        "out_path": str(out),
        "frames": len(rows),
        "valid_hand_camera": len(valid_hands),
        "written": written,
        "limits": limits,
        "wrist_camera_step_m": summarize_steps([p[0] for p in valid_hands]),
    }
    if args.summary_json:
        with Path(args.summary_json).expanduser().resolve().open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
            f.write("\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize glove hand trajectory in camera coordinates.")
    parser.add_argument("--trajectory_jsonl", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--trail_len", type=int, default=80)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--pad_ratio", type=float, default=0.18)
    parser.add_argument("--elev", type=float, default=24.0)
    parser.add_argument("--azim", type=float, default=-58.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = render(args)
    print(f"[VisualizeTrajectoryCameraFrame] video: {args.out_path}")
    print(f"[VisualizeTrajectoryCameraFrame] summary: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
