#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render temporally smoothed visual 2D hand keypoints from fusion frames."""

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

from preprocess.VisualizeGloveFkVsVisual import draw_skeleton, draw_text
from preprocess.Timebase import effective_alpha, relative_times_sec, repeat_counts


SMOOTH_COLOR = (80, 240, 120)
RAW_COLOR = (40, 220, 255)


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


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def load_visual_2d(row: Dict) -> Optional[np.ndarray]:
    visual = row.get("visual") or row.get("visual_prior") or {}
    hand = visual.get("hand") or {}
    pts = hand.get("kpts_2d")
    if pts is None:
        return None
    arr = np.asarray(pts, dtype=np.float64)
    if arr.shape[0] < 21 or arr.shape[1] < 2:
        return None
    arr = arr[:21, :2]
    if not np.isfinite(arr).all():
        return None
    return arr


def interpolate_series(
    values: np.ndarray,
    valid: np.ndarray,
    max_gap: int,
    times_sec: np.ndarray,
    reference_fps: float,
) -> np.ndarray:
    out = values.copy()
    idx = np.arange(len(values))
    for joint in range(values.shape[1]):
        for coord in range(2):
            v = valid[:, joint] & np.isfinite(values[:, joint, coord])
            valid_idx = idx[v]
            if valid_idx.size == 0:
                out[:, joint, coord] = np.nan
                continue
            out[:, joint, coord] = np.nan
            out[valid_idx, joint, coord] = values[valid_idx, joint, coord]
            for left, right in zip(valid_idx[:-1], valid_idx[1:]):
                gap = int(right - left - 1)
                max_interval_sec = (max_gap + 1) / reference_fps
                if 0 < gap and float(times_sec[right] - times_sec[left]) <= max_interval_sec:
                    fill_idx = np.arange(left + 1, right)
                    out[fill_idx, joint, coord] = np.interp(
                        times_sec[fill_idx],
                        [times_sec[left], times_sec[right]],
                        [values[left, joint, coord], values[right, joint, coord]],
                    )
    return out


def ema_forward(values: np.ndarray, alpha: float, times_sec: np.ndarray, reference_fps: float) -> np.ndarray:
    out = values.copy()
    for i in range(1, len(values)):
        frame_alpha = effective_alpha(alpha, float(times_sec[i] - times_sec[i - 1]), reference_fps)
        out[i] = frame_alpha * values[i] + (1.0 - frame_alpha) * out[i - 1]
    return out


def zero_phase_ema(values: np.ndarray, alpha: float, times_sec: np.ndarray, reference_fps: float) -> np.ndarray:
    if len(values) <= 1:
        return values.copy()
    forward = ema_forward(values, alpha, times_sec, reference_fps)
    reverse_times = -times_sec[::-1]
    return ema_forward(forward[::-1], alpha, reverse_times, reference_fps)[::-1]


def finite_runs(mask: np.ndarray) -> List[np.ndarray]:
    runs: List[np.ndarray] = []
    start = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append(np.arange(start, i))
            start = None
    if start is not None:
        runs.append(np.arange(start, len(mask)))
    return runs


def smooth_vector_series(values: np.ndarray, alpha: float, times_sec: np.ndarray, reference_fps: float) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=np.float64)
    mask = np.isfinite(values).all(axis=1)
    for run in finite_runs(mask):
        if len(run) <= 1:
            out[run] = values[run]
        else:
            out[run] = zero_phase_ema(values[run], alpha, times_sec[run], reference_fps)
    return out


def smooth_points(
    raw: np.ndarray,
    valid: np.ndarray,
    alpha: float,
    max_gap: int,
    palm_indices: List[int],
    times_sec: np.ndarray,
    reference_fps: float,
) -> np.ndarray:
    filled = interpolate_series(raw, valid, max_gap, times_sec, reference_fps)
    frame_valid = np.isfinite(filled).all(axis=(1, 2))
    smoothed = np.full_like(filled, np.nan, dtype=np.float64)
    if not np.any(frame_valid):
        return smoothed

    palm = np.full((len(filled), 2), np.nan, dtype=np.float64)
    palm[frame_valid] = np.mean(filled[frame_valid][:, palm_indices, :], axis=1)
    palm_smooth = smooth_vector_series(palm, alpha, times_sec, reference_fps)

    rel = filled - palm[:, None, :]
    rel_smooth = np.full_like(rel, np.nan, dtype=np.float64)
    for joint in range(raw.shape[1]):
        rel_smooth[:, joint, :] = smooth_vector_series(rel[:, joint, :], alpha, times_sec, reference_fps)

    ok = np.isfinite(palm_smooth).all(axis=1) & np.isfinite(rel_smooth).all(axis=(1, 2))
    smoothed[ok] = rel_smooth[ok] + palm_smooth[ok, None, :]
    return smoothed


def valid_count(pts: Optional[np.ndarray], width: int, height: int) -> int:
    if pts is None:
        return 0
    ok = (
        np.isfinite(pts).all(axis=1)
        & (pts[:, 0] >= 0)
        & (pts[:, 0] < width)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] < height)
    )
    return int(ok.sum())


def build_smooth_rows(rows: List[Dict], smoothed: np.ndarray, valid: np.ndarray) -> List[Dict]:
    out = []
    for i, row in enumerate(rows):
        out.append(
            {
                "frame": row.get("frame"),
                "idx": row.get("idx", i),
                "rgb_stamp_ns": row.get("rgb_stamp_ns"),
                "rgb_bag_time_ns": row.get("rgb_bag_time_ns"),
                "rgb_path": row.get("rgb_path"),
                "visual_2d_smooth": {
                    "valid": bool(np.any(valid[i])),
                    "source": "visual.hand.kpts_2d",
                    "kpts_2d": smoothed[i].astype(float).tolist(),
                },
            }
        )
    return out


def render(args: argparse.Namespace) -> Dict:
    rows = read_jsonl(Path(args.input_jsonl).expanduser().resolve())
    if args.max_frames is not None:
        rows = rows[: args.max_frames]
    if not rows:
        raise RuntimeError("No rows to visualize")
    times_sec = np.asarray(relative_times_sec(rows, fallback_fps=args.reference_fps), dtype=np.float64)
    video_repeats = repeat_counts(rows, output_fps=args.fps)

    raw = np.full((len(rows), 21, 2), np.nan, dtype=np.float64)
    valid = np.zeros((len(rows), 21), dtype=bool)
    for i, row in enumerate(rows):
        pts = load_visual_2d(row)
        if pts is None:
            continue
        raw[i] = pts
        valid[i] = np.isfinite(pts).all(axis=1)

    palm_indices = [int(v) for v in args.palm_indices.split(",") if v.strip() != ""]
    smoothed = smooth_points(
        raw,
        valid,
        args.alpha,
        args.max_interp_gap,
        palm_indices,
        times_sec,
        args.reference_fps,
    )
    smooth_valid = np.isfinite(smoothed).all(axis=(1, 2))

    out_path = Path(args.out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    stats = {
        "frames": len(rows),
        "written": 0,
        "raw_present": int(np.any(valid, axis=1).sum()),
        "smooth_present": int(smooth_valid.sum()),
        "max_interp_gap": args.max_interp_gap,
        "missing_rgb": 0,
        "source_frames_written": 0,
    }
    raw_visible = []
    smooth_visible = []
    displacement = []

    for i, row in enumerate(tqdm(rows, desc="Rendering visual 2D smooth")):
        rgb_path = row.get("rgb_path")
        img = cv2.imread(str(rgb_path)) if rgb_path else None
        if img is None:
            stats["missing_rgb"] += 1
            continue

        h, w = img.shape[:2]
        raw_pts = raw[i] if np.any(valid[i]) else None
        smooth_pts = smoothed[i] if np.isfinite(smoothed[i]).all() else None

        overlay = img.copy()
        if args.draw_raw and raw_pts is not None:
            draw_skeleton(overlay, raw_pts, RAW_COLOR, radius=2, thickness=1, draw_indices=args.draw_indices)
        draw_skeleton(overlay, smooth_pts, SMOOTH_COLOR, radius=4, thickness=2, draw_indices=args.draw_indices)
        img = cv2.addWeighted(overlay, 0.9, img, 0.1, 0)

        raw_n = valid_count(raw_pts, w, h)
        smooth_n = valid_count(smooth_pts, w, h)
        raw_visible.append(raw_n)
        smooth_visible.append(smooth_n)
        if raw_pts is not None and smooth_pts is not None:
            mask = valid[i] & np.isfinite(smooth_pts).all(axis=1)
            if np.any(mask):
                displacement.append(float(np.mean(np.linalg.norm(raw_pts[mask] - smooth_pts[mask], axis=1))))

        frame = row.get("frame", f"{i:05d}")
        label = "visual2D smooth=green"
        if args.draw_raw:
            label += " raw=yellow"
        draw_text(img, f"{frame}  {label}", (10, 24))
        draw_text(img, f"visible raw={raw_n}/21 smooth={smooth_n}/21 alpha={args.alpha:.2f}", (10, 48))

        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {out_path}")
        for _ in range(video_repeats[i]):
            writer.write(img)
            stats["written"] += 1
        stats["source_frames_written"] += 1

    if writer is not None:
        writer.release()

    if args.output_jsonl:
        write_jsonl(Path(args.output_jsonl).expanduser().resolve(), build_smooth_rows(rows, smoothed, np.repeat(smooth_valid[:, None], 21, axis=1)))

    summary = {
        "input_jsonl": str(Path(args.input_jsonl).expanduser().resolve()),
        "out_path": str(out_path),
        "output_jsonl": str(Path(args.output_jsonl).expanduser().resolve()) if args.output_jsonl else None,
        "fps": args.fps,
        "alpha": args.alpha,
        "max_interp_gap": args.max_interp_gap,
        "reference_fps": args.reference_fps,
        "timebase": "rgb_stamp_ns",
        "duration_sec": float(times_sec[-1]) if len(times_sec) else 0.0,
        "palm_indices": palm_indices,
        "stats": stats,
    }
    if raw_visible:
        summary["raw_visible_points"] = {
            "min": int(np.min(raw_visible)),
            "median": float(np.median(raw_visible)),
            "max": int(np.max(raw_visible)),
        }
    if smooth_visible:
        summary["smooth_visible_points"] = {
            "min": int(np.min(smooth_visible)),
            "median": float(np.median(smooth_visible)),
            "max": int(np.max(smooth_visible)),
        }
    if displacement:
        arr = np.asarray(displacement, dtype=np.float64)
        summary["raw_to_smooth_mean_px"] = {
            "min": float(arr.min()),
            "median": float(np.median(arr)),
            "max": float(arr.max()),
            "p95": float(np.percentile(arr, 95)),
        }

    if args.summary_json:
        write_json(Path(args.summary_json).expanduser().resolve(), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render visual 21 keypoints after temporal 2D smoothing.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--reference_fps", type=float, default=30.0, help="FPS at which alpha and max_interp_gap retain their legacy per-frame meaning.")
    parser.add_argument("--max_interp_gap", type=int, default=3, help="Only fill missing visual detections across gaps up to this many frames; longer gaps remain invalid.")
    parser.add_argument("--palm_indices", default="0,5,9,13,17", help="Keypoints used for palm-center translation smoothing.")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--draw_raw", action="store_true")
    parser.add_argument("--draw_indices", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = render(args)
    print(f"[VisualizeVisual2DSmooth] video: {args.out_path}")
    print(f"[VisualizeVisual2DSmooth] summary: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
