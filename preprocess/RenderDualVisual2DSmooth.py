#!/usr/bin/env python3
"""Render left and right smoothed HaMeR 2D keypoints into one video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess.Timebase import relative_times_sec, repeat_counts
from preprocess.VisualizeGloveFkVsVisual import draw_skeleton, draw_text
from preprocess.VisualizeVisual2DSmooth import smooth_points


HAND_COLORS = {"left": (255, 120, 40), "right": (40, 140, 255)}


def read_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def row_key(row: Dict) -> tuple:
    stamp = row.get("rgb_stamp_ns")
    if stamp is not None:
        return ("stamp", int(stamp))
    return ("frame", str(row.get("frame")))


def row_smooth_points(row: Optional[Dict]) -> Optional[np.ndarray]:
    if row is None:
        return None
    payload = row.get("visual_2d_smooth") or {}
    if not payload.get("valid", False):
        return None
    pts = payload.get("kpts_2d")
    if pts is None:
        return None
    arr = np.asarray(pts, dtype=np.float64)
    if arr.shape != (21, 2) or not np.isfinite(arr).all():
        return None
    return arr


def _palm_normal(points: np.ndarray) -> Optional[np.ndarray]:
    normal = np.cross(points[5] - points[0], points[17] - points[0])
    norm = float(np.linalg.norm(normal))
    return None if norm <= 1e-9 else normal / norm


def _runs(values: np.ndarray) -> List[tuple[int, int, int]]:
    if len(values) == 0:
        return []
    result = []
    start = 0
    for i in range(1, len(values)):
        if values[i] != values[start]:
            result.append((start, i - 1, int(values[start])))
            start = i
    result.append((start, len(values) - 1, int(values[start])))
    return result


def detect_hamer_branch_outliers(
    visual_3d: List[Optional[np.ndarray]],
    jump_threshold_deg: float,
    bridge_gap_frames: int,
    max_reject_frames: int,
) -> tuple[np.ndarray, List[List[int]]]:
    normals = [_palm_normal(points) if points is not None else None for points in visual_3d]
    parity = np.zeros(len(normals), dtype=np.uint8)
    for i in range(1, len(normals)):
        parity[i] = parity[i - 1]
        if normals[i - 1] is None or normals[i] is None:
            continue
        angle_deg = float(np.degrees(np.arccos(np.clip(np.dot(normals[i - 1], normals[i]), -1.0, 1.0))))
        if angle_deg > jump_threshold_deg:
            parity[i] = 1 - parity[i]

    # Merge very short returns to the original branch between two flipped runs.
    for start, end, value in _runs(parity.copy()):
        if value != 0 or start == 0 or end == len(parity) - 1 or (end - start + 1) > bridge_gap_frames:
            continue
        if parity[start - 1] == 1 and parity[end + 1] == 1:
            parity[start:end + 1] = 1

    rejected = np.zeros(len(parity), dtype=bool)
    rejected_runs: List[List[int]] = []
    for start, end, value in _runs(parity):
        length = end - start + 1
        if value == 1 and start > 0 and end < len(parity) - 1 and length <= max_reject_frames:
            rejected[start:end + 1] = True
            rejected_runs.append([start, end])
    return rejected, rejected_runs


def interpolate_rejected_poses(raw: np.ndarray, valid: np.ndarray, rejected: np.ndarray, palm_indices: List[int]) -> None:
    for start, end, value in _runs(rejected.astype(np.uint8)):
        if value != 1:
            continue
        left = start - 1
        right = end + 1
        if left < 0 or right >= len(raw) or not valid[left].all() or not valid[right].all():
            valid[start:end + 1] = False
            continue
        left_center = np.mean(raw[left, palm_indices], axis=0)
        right_center = np.mean(raw[right, palm_indices], axis=0)
        left_scale = max(float(np.linalg.norm(raw[left, 9] - raw[left, 0])), 1e-6)
        right_scale = max(float(np.linalg.norm(raw[right, 9] - raw[right, 0])), 1e-6)
        left_pose = (raw[left] - left_center) / left_scale
        right_pose = (raw[right] - right_center) / right_scale
        for i in range(start, end + 1):
            weight = float(i - left) / float(right - left)
            if valid[i].all():
                center = np.mean(raw[i, palm_indices], axis=0)
                scale = max(float(np.linalg.norm(raw[i, 9] - raw[i, 0])), 1e-6)
            else:
                center = (1.0 - weight) * left_center + weight * right_center
                scale = (1.0 - weight) * left_scale + weight * right_scale
            pose = (1.0 - weight) * left_pose + weight * right_pose
            raw[i] = center + scale * pose
            valid[i] = True


def rows_from_trajectory(args: argparse.Namespace, trajectory_rows: List[Dict], side: str) -> tuple[List[Dict], Dict]:
    raw = np.full((len(trajectory_rows), 21, 2), np.nan, dtype=np.float64)
    valid = np.zeros((len(trajectory_rows), 21), dtype=bool)
    visual_3d: List[Optional[np.ndarray]] = []
    rgb_frames_dir = Path(args.rgb_frames_dir).expanduser().resolve() if args.rgb_frames_dir else None

    for i, row in enumerate(trajectory_rows):
        hand = ((row.get("hands") or {}).get(side) or {})
        visual_payload = hand.get("visual_prior") or {}
        visual = np.asarray(visual_payload.get("kpts_3d_camera_m"), dtype=np.float64)
        visual_3d.append(visual if visual.shape == (21, 3) and np.isfinite(visual).all() else None)
        fallback = np.asarray(visual_payload.get("kpts_2d"), dtype=np.float64)
        points_2d = fallback if fallback.shape == (21, 2) and np.isfinite(fallback).all() else None
        if points_2d is not None:
            raw[i] = points_2d
            valid[i] = True

    times_sec = np.asarray(relative_times_sec(trajectory_rows, fallback_fps=args.reference_fps), dtype=np.float64)
    palm_indices = [int(value) for value in args.palm_indices.split(",") if value.strip()]
    rejected, rejected_runs = detect_hamer_branch_outliers(
        visual_3d,
        args.branch_jump_threshold_deg,
        args.branch_bridge_gap_frames,
        args.branch_max_reject_frames,
    )
    interpolate_rejected_poses(raw, valid, rejected, palm_indices)
    smoothed = smooth_points(raw, valid, args.alpha, args.max_interp_gap, palm_indices, times_sec, args.reference_fps)
    rows = []
    for i, source in enumerate(trajectory_rows):
        frame = str(source.get("frame", f"{i:05d}"))
        rgb_path = (source.get("paths") or {}).get("rgb")
        if rgb_frames_dir is not None:
            candidate = rgb_frames_dir / f"{frame}.jpg"
            if candidate.is_file():
                rgb_path = str(candidate)
        rows.append(
            {
                "frame": frame,
                "idx": source.get("idx", i),
                "rgb_stamp_ns": source.get("timestamp", {}).get("rgb_stamp_ns") or source.get("rgb_stamp_ns"),
                "rgb_path": rgb_path,
                "visual_2d_smooth": {
                    "valid": bool(np.isfinite(smoothed[i]).all()),
                    "kpts_2d": smoothed[i].tolist(),
                },
            }
        )
    return rows, {"rejected_frames": int(rejected.sum()), "rejected_runs": rejected_runs}


def render(args: argparse.Namespace) -> Dict:
    branch_filter = {"left": {"rejected_frames": 0, "rejected_runs": []}, "right": {"rejected_frames": 0, "rejected_runs": []}}
    if args.trajectory_jsonl:
        trajectory_rows = read_jsonl(Path(args.trajectory_jsonl).expanduser().resolve())
        if args.max_frames is not None:
            trajectory_rows = trajectory_rows[: args.max_frames]
        left_rows, branch_filter["left"] = rows_from_trajectory(args, trajectory_rows, "left")
        right_rows, branch_filter["right"] = rows_from_trajectory(args, trajectory_rows, "right")
    else:
        if not args.left_jsonl or not args.right_jsonl:
            raise ValueError("Provide --trajectory_jsonl or both --left_jsonl and --right_jsonl")
        left_rows = read_jsonl(Path(args.left_jsonl).expanduser().resolve())
        right_rows = read_jsonl(Path(args.right_jsonl).expanduser().resolve())
    if args.max_frames is not None:
        left_rows = left_rows[: args.max_frames]
    if not left_rows:
        raise RuntimeError("No left smooth rows to render")

    right_by_key = {row_key(row): row for row in right_rows}
    repeats = repeat_counts(left_rows, output_fps=args.fps)
    out_path = Path(args.out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    valid = {"left": 0, "right": 0}
    missing_rgb = 0
    written = 0

    for i, left_row in enumerate(tqdm(left_rows, desc="Rendering dual-hand visual 2D smooth")):
        right_row = right_by_key.get(row_key(left_row))
        rgb_path = left_row.get("rgb_path") or (right_row or {}).get("rgb_path")
        image = cv2.imread(str(rgb_path)) if rgb_path else None
        if image is None:
            missing_rgb += 1
            continue

        overlay = image.copy()
        for side, row in (("left", left_row), ("right", right_row)):
            pts = row_smooth_points(row)
            if pts is None:
                continue
            draw_skeleton(overlay, pts, HAND_COLORS[side], radius=4, thickness=2, draw_indices=args.draw_indices)
            valid[side] += 1
        image = cv2.addWeighted(overlay, 0.9, image, 0.1, 0)
        frame = left_row.get("frame", f"{i:05d}")
        draw_text(image, f"{frame}  HaMeR 2D smooth: left=blue right=orange", (10, 24))

        if writer is None:
            height, width = image.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {out_path}")
        for _ in range(repeats[i]):
            writer.write(image)
            written += 1

    if writer is not None:
        writer.release()
    if written == 0:
        raise RuntimeError("No frames were written to the dual-hand smooth video")

    summary = {
        "left_jsonl": str(Path(args.left_jsonl).expanduser().resolve()) if args.left_jsonl else None,
        "right_jsonl": str(Path(args.right_jsonl).expanduser().resolve()) if args.right_jsonl else None,
        "trajectory_jsonl": str(Path(args.trajectory_jsonl).expanduser().resolve()) if args.trajectory_jsonl else None,
        "out_path": str(out_path),
        "source_frames": len(left_rows),
        "written": written,
        "valid_frames": valid,
        "missing_rgb": missing_rgb,
        "fps": args.fps,
        "timebase": "rgb_stamp_ns",
        "hamer_pose_branch_filter": branch_filter,
        "branch_jump_threshold_deg": args.branch_jump_threshold_deg,
        "branch_bridge_gap_frames": args.branch_bridge_gap_frames,
        "branch_max_reject_frames": args.branch_max_reject_frames,
    }
    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left_jsonl", default=None)
    parser.add_argument("--right_jsonl", default=None)
    parser.add_argument("--trajectory_jsonl", default=None)
    parser.add_argument("--rgb_frames_dir", default=None)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--reference_fps", type=float, default=30.0)
    parser.add_argument("--max_interp_gap", type=int, default=3)
    parser.add_argument("--palm_indices", default="0,5,9,13,17")
    parser.add_argument("--branch_jump_threshold_deg", type=float, default=75.0)
    parser.add_argument("--branch_bridge_gap_frames", type=int, default=3)
    parser.add_argument("--branch_max_reject_frames", type=int, default=60)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--draw_indices", action="store_true")
    args = parser.parse_args()
    summary = render(args)
    print(f"[RenderDualVisual2DSmooth] video: {summary['out_path']}")
    print(f"[RenderDualVisual2DSmooth] valid frames: {summary['valid_frames']}")


if __name__ == "__main__":
    main()
