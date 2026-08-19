#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build and render RGB-rate camera trajectory from RTAB-Map Node poses."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from preprocess.Timebase import repeat_counts
except ModuleNotFoundError:
    from Timebase import repeat_counts


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def pose_blob_to_mat(blob: bytes) -> np.ndarray:
    if len(blob) != 48:
        raise ValueError(f"Expected RTAB-Map Transform blob with 12 float32 values, got {len(blob)} bytes")
    vals = struct.unpack("<12f", blob)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :4] = np.asarray(vals, dtype=np.float64).reshape(3, 4)
    return mat


def load_rtabmap_poses(db_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute(
            "select stamp, pose from Node where pose is not null order by stamp"
        ).fetchall()
    finally:
        con.close()
    stamps: List[float] = []
    mats: List[np.ndarray] = []
    for stamp, blob in rows:
        if stamp is None or blob is None:
            continue
        mat = pose_blob_to_mat(blob)
        if np.isfinite(mat).all():
            stamps.append(float(stamp))
            mats.append(mat)
    if len(mats) < 2:
        raise RuntimeError(f"Need at least two RTAB-Map poses, got {len(mats)} from {db_path}")
    return np.asarray(stamps, dtype=np.float64), np.asarray(mats, dtype=np.float64)


def mat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    m = np.asarray(rot, dtype=np.float64)
    tr = float(np.trace(m))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return q / max(np.linalg.norm(q), 1e-12)


def quat_to_mat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q / max(np.linalg.norm(q), 1e-12)]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def slerp_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = q0 / max(np.linalg.norm(q0), 1e-12)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / max(np.linalg.norm(q), 1e-12)
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


def interpolate_pose(
    stamps: np.ndarray,
    mats: np.ndarray,
    quats: np.ndarray,
    stamp: float,
    max_gap_sec: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], str, Optional[float]]:
    if stamp < float(stamps[0]) or stamp > float(stamps[-1]):
        return None, "outside_range", None
    hi = int(np.searchsorted(stamps, stamp, side="left"))
    if hi == 0:
        return mats[0].copy(), "ok", 0.0
    if hi >= len(stamps):
        return mats[-1].copy(), "ok", 0.0
    if float(stamps[hi]) == stamp:
        return mats[hi].copy(), "ok", 0.0
    lo = hi - 1
    denom = float(stamps[hi] - stamps[lo])
    if max_gap_sec is not None and max_gap_sec > 0.0 and denom > max_gap_sec:
        return None, "gap_exceeded", denom
    alpha = 0.0 if denom <= 0.0 else float((stamp - stamps[lo]) / denom)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, 3] = (1.0 - alpha) * mats[lo, :3, 3] + alpha * mats[hi, :3, 3]
    mat[:3, :3] = quat_to_mat_wxyz(slerp_wxyz(quats[lo], quats[hi], alpha))
    return mat, "ok", denom


def compute_limits(points: np.ndarray, pad_ratio: float) -> Dict[str, Tuple[float, float]]:
    lo = np.percentile(points, 1, axis=0)
    hi = np.percentile(points, 99, axis=0)
    span = np.maximum(hi - lo, 1e-3)
    pad = span * pad_ratio
    mid = (lo + hi) * 0.5
    half = float(np.max(span + 2.0 * pad)) * 0.5
    return {
        "x": (float(mid[0] - half), float(mid[0] + half)),
        "y": (float(mid[1] - half), float(mid[1] + half)),
        "z": (float(mid[2] - half), float(mid[2] + half)),
    }


def draw_axes(ax, mat: np.ndarray, scale: float) -> None:
    origin = mat[:3, 3]
    axes = mat[:3, :3]
    for i, color in enumerate(("#ef4444", "#22c55e", "#3b82f6")):
        end = origin + axes[:, i] * scale
        ax.plot([origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]], color=color, linewidth=2.0)


def render_video(rows: List[Dict], out_path: Path, summary_path: Path, fps: float, width: int, height: int, trail_len: int, camera_axis_len: float, draw_camera_axes_flag: bool) -> Dict:
    mats = [np.asarray(r["camera"]["c2w"], dtype=np.float64) for r in rows]
    points = np.asarray([m[:3, 3] for m in mats], dtype=np.float64)
    limits = compute_limits(points, 0.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")
    fig_w = width / 100.0
    fig_h = height / 100.0
    repeats = repeat_counts(rows, output_fps=fps)
    written = 0
    for i, mat in enumerate(mats):
        trail = points[max(0, i - trail_len + 1): i + 1]
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(*limits["x"])
        ax.set_ylim(*limits["y"])
        ax.set_zlim(*limits["z"])
        ax.set_xlabel("X world (m)")
        ax.set_ylabel("Y world (m)")
        ax.set_zlabel("Z world (m)")
        ax.view_init(elev=24.0, azim=-58.0)
        ax.set_title(f"RTAB-Map camera trajectory | frame {rows[i]['frame']}")
        if len(trail) > 1:
            ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], color="#111111", linewidth=1.6)
        ax.scatter([points[i, 0]], [points[i, 1]], [points[i, 2]], c="#2563eb", s=24)
        if draw_camera_axes_flag:
            draw_axes(ax, mat, camera_axis_len)
        fig.tight_layout()
        fig.canvas.draw()
        bgr = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        if bgr.shape[1] != width or bgr.shape[0] != height:
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
        for _ in range(repeats[i]):
            writer.write(bgr)
            written += 1
        plt.close(fig)
    writer.release()
    summary = {"mp4": str(out_path), "frames": len(rows), "written": written, "timebase": "rgb_stamp_ns", "limits": limits, "draw_camera_axes": draw_camera_axes_flag}
    write_json(summary_path, summary)
    return summary


def build(args: argparse.Namespace) -> Dict:
    if args.max_interp_gap_sec <= 0.0:
        raise ValueError("--max_interp_gap_sec must be greater than zero")
    db_path = Path(args.rtabmap_db).expanduser().resolve()
    timestamps_path = Path(args.timestamps_jsonl).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rtab_stamps, rtab_mats = load_rtabmap_poses(db_path)
    rtab_quats = np.asarray([mat_to_quat_wxyz(m[:3, :3]) for m in rtab_mats], dtype=np.float64)
    ts_rows = read_jsonl(timestamps_path)

    out_rows: List[Dict] = []
    skipped_outside: List[Dict] = []
    skipped_gap: List[Dict] = []
    used_bracketing_gaps: List[float] = []
    for row in ts_rows:
        rgb_stamp_ns = int(row["rgb_stamp_ns"])
        stamp = rgb_stamp_ns / 1e9
        mat, reason, bracket_gap_sec = interpolate_pose(
            rtab_stamps,
            rtab_mats,
            rtab_quats,
            stamp,
            args.max_interp_gap_sec,
        )
        if mat is None:
            skipped = {
                "frame": row.get("frame"),
                "rgb_stamp_ns": rgb_stamp_ns,
                "stamp": stamp,
            }
            if reason == "gap_exceeded":
                skipped["bracketing_gap_sec"] = bracket_gap_sec
                skipped_gap.append(skipped)
            else:
                skipped_outside.append(skipped)
            continue
        if bracket_gap_sec is not None:
            used_bracketing_gaps.append(float(bracket_gap_sec))
        out_rows.append({
            "frame": row.get("frame"),
            "rgb_stamp_ns": rgb_stamp_ns,
            "stamp": stamp,
            "camera": {
                "frame_id": args.frame_id,
                "c2w": mat.tolist(),
            },
        })

    coverage_ratio = float(len(out_rows) / len(ts_rows)) if ts_rows else 0.0
    coverage_info = {
        "rgb_frames_total": int(len(ts_rows)),
        "rgb_frames_interpolated": int(len(out_rows)),
        "coverage_ratio": coverage_ratio,
        "outside_range_count": len(skipped_outside),
        "gap_exceeded_count": len(skipped_gap),
        "outside_range_frames": skipped_outside[:20],
        "gap_exceeded_frames": skipped_gap[:20],
        "max_interp_gap_sec": args.max_interp_gap_sec,
        "max_bracketing_gap_sec_used": max(used_bracketing_gaps, default=0.0),
        "require_full_coverage": bool(args.require_full_coverage),
    }
    summary_path = out_dir / "rtabmap_camera_pose_rgb30_interp_summary.json"
    if not out_rows:
        write_json(summary_path, {**coverage_info, "status": "failed", "reason": "no overlapping RGB frames"})
        raise RuntimeError("No RGB timestamps overlap RTAB-Map pose time range")
    if args.require_full_coverage and len(out_rows) != len(ts_rows):
        write_json(summary_path, {**coverage_info, "status": "failed", "reason": "incomplete RTAB-Map coverage"})
        raise RuntimeError(
            "Incomplete RTAB-Map coverage: "
            f"{len(out_rows)}/{len(ts_rows)} RGB frames; "
            f"outside_range={len(skipped_outside)}, gap_exceeded={len(skipped_gap)}, "
            f"max_interp_gap_sec={args.max_interp_gap_sec}"
        )

    jsonl_path = out_dir / "rtabmap_camera_pose_rgb30_interp.jsonl"
    txt_path = out_dir / "poses_rgb30_interp.txt"
    source_txt_path = out_dir / "poses_rtabmap_source_overlap.txt"
    write_jsonl(jsonl_path, out_rows)
    with txt_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            mat = np.asarray(row["camera"]["c2w"], dtype=np.float64)
            vals = [row["stamp"], *mat[:3, 3].tolist(), *mat_to_quat_wxyz(mat[:3, :3]).tolist()]
            f.write(" ".join(f"{v:.9f}" for v in vals) + "\n")

    rgb_stamps = np.asarray([r["stamp"] for r in out_rows], dtype=np.float64)
    overlap = (rtab_stamps >= rgb_stamps[0]) & (rtab_stamps <= rgb_stamps[-1])
    with source_txt_path.open("w", encoding="utf-8") as f:
        for stamp, mat in zip(rtab_stamps[overlap], rtab_mats[overlap]):
            vals = [float(stamp), *mat[:3, 3].tolist(), *mat_to_quat_wxyz(mat[:3, :3]).tolist()]
            f.write(" ".join(f"{v:.9f}" for v in vals) + "\n")

    points = np.asarray([np.asarray(r["camera"]["c2w"], dtype=np.float64)[:3, 3] for r in out_rows], dtype=np.float64)
    source_points = rtab_mats[overlap, :3, 3]
    rgb_dt = np.diff(rgb_stamps)
    rtab_dt = np.diff(rtab_stamps)
    interp_step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    source_step = np.linalg.norm(np.diff(source_points, axis=0), axis=1) if len(source_points) > 1 else np.asarray([], dtype=np.float64)

    def stats(values: np.ndarray) -> Dict[str, float]:
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return {}
        out = {"median": float(np.median(values)), "min": float(np.min(values)), "max": float(np.max(values))}
        if np.all(values > 0):
            out["fps_median"] = float(1.0 / np.median(values))
        return out

    def step_stats(values: np.ndarray) -> Dict[str, float]:
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return {}
        return {
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    summary = {
        **coverage_info,
        "status": "complete",
        "source_bag": str(Path(args.source_bag).expanduser().resolve()) if args.source_bag else None,
        "source_rtabmap_db": str(db_path),
        "note": "Interpolated RTAB-Map Node.pose to RGB rosbag message timestamps; no hand_frame used.",
        "rtabmap_total_nodes": int(len(rtab_stamps)),
        "rtabmap_overlap_nodes": int(np.count_nonzero(overlap)),
        "rtabmap_time_range_sec": [float(rtab_stamps[0]), float(rtab_stamps[-1])],
        "rgb_time_range_sec": [float(rgb_stamps[0]), float(rgb_stamps[-1])],
        "rtabmap_dt_sec": stats(rtab_dt),
        "rgb_dt_sec": stats(rgb_dt),
        "rtabmap_source_step_m": step_stats(source_step),
        "interpolated_step_m": step_stats(interp_step),
        "position_range_xyz_m": (np.max(points, axis=0) - np.min(points, axis=0)).tolist(),
        "limits": compute_limits(points, 0.18),
        "jsonl": str(jsonl_path),
        "txt": str(txt_path),
        "source_overlap_txt": str(source_txt_path),
        "render_videos": bool(args.render_videos),
        "mp4": str(out_dir / "rtabmap_camera_pose_rgb30_interp.mp4") if args.render_videos else None,
        "mp4_no_axes": str(out_dir / "rtabmap_camera_pose_rgb30_interp_no_axes.mp4") if args.render_videos else None,
    }
    write_json(summary_path, summary)
    if args.render_videos:
        render_video(out_rows, out_dir / "rtabmap_camera_pose_rgb30_interp.mp4", out_dir / "rtabmap_camera_pose_rgb30_interp_render_summary.json", args.fps, args.width, args.height, args.trail_len, args.camera_axis_len, True)
        render_video(out_rows, out_dir / "rtabmap_camera_pose_rgb30_interp_no_axes.mp4", out_dir / "rtabmap_camera_pose_rgb30_interp_no_axes_summary.json", args.fps, args.width, args.height, args.trail_len, args.camera_axis_len, False)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interpolate RTAB-Map Node poses to RGB frame timestamps and render camera trajectory.")
    parser.add_argument("--rtabmap_db", required=True)
    parser.add_argument("--timestamps_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--source_bag", default=None)
    parser.add_argument("--frame_id", default="rtabmap_pose")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--trail_len", type=int, default=120)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--camera_axis_len", type=float, default=0.08)
    parser.add_argument(
        "--render_videos",
        action="store_true",
        help="Render the two RTAB-Map trajectory preview MP4 files (disabled by default).",
    )
    parser.add_argument(
        "--max_interp_gap_sec",
        type=float,
        default=0.25,
        help="Reject interpolation across RTAB-Map pose gaps larger than this; must be > 0.",
    )
    parser.add_argument(
        "--require_full_coverage",
        action="store_true",
        help="Fail unless every RGB timestamp receives an RTAB-Map pose.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build(args)
    print(f"[BuildRtabmapCameraTrajectory] jsonl: {summary['jsonl']}")
    print(f"[BuildRtabmapCameraTrajectory] mp4: {summary['mp4']}")
    print(f"[BuildRtabmapCameraTrajectory] summary: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
