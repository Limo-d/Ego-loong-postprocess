#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overlay HaMeR visual 2D keypoints and glove FK projected 21 points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
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


VISUAL_COLOR = (40, 220, 255)  # yellow/cyan in BGR
GLOVE_COLOR = (255, 80, 220)   # magenta in BGR
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


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


def valid_point(pt: np.ndarray, width: int, height: int) -> bool:
    x, y = float(pt[0]), float(pt[1])
    return np.isfinite(x) and np.isfinite(y) and 0 <= x < width and 0 <= y < height


def project_points(kpts_3d: List[List[float]], k: List[List[float]]) -> Optional[np.ndarray]:
    pts = np.asarray(kpts_3d, dtype=np.float64)
    K = np.asarray(k, dtype=np.float64)
    if pts.shape[0] < 21 or pts.shape[1] < 3 or K.shape != (3, 3):
        return None
    z = pts[:, 2]
    out = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    ok = np.isfinite(pts).all(axis=1) & (z > 1e-6)
    out[ok, 0] = K[0, 0] * pts[ok, 0] / z[ok] + K[0, 2]
    out[ok, 1] = K[1, 1] * pts[ok, 1] / z[ok] + K[1, 2]
    return out[:21]


def load_visual_2d(row: Dict) -> Optional[np.ndarray]:
    hand = (row.get("visual_prior") or {}).get("hand") or {}
    pts = hand.get("kpts_2d")
    if pts is None:
        return None
    arr = np.asarray(pts, dtype=np.float64)
    if arr.shape[0] < 21 or arr.shape[1] < 2:
        return None
    return arr[:21, :2]


def load_glove_2d(row: Dict) -> Optional[np.ndarray]:
    fk = row.get("glove_fk21") or {}
    if not fk.get("valid"):
        return None
    kpts_3d = fk.get("kpts_3d_camera_m")
    k = (row.get("camera") or {}).get("k")
    if kpts_3d is None or k is None:
        return None
    return project_points(kpts_3d, k)


def draw_text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float = 0.48) -> None:
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 1, cv2.LINE_AA)


def draw_skeleton(
    img: np.ndarray,
    pts: Optional[np.ndarray],
    color: Tuple[int, int, int],
    radius: int,
    thickness: int,
    draw_indices: bool,
) -> int:
    if pts is None:
        return 0
    h, w = img.shape[:2]
    visible = 0
    for a, b in MP_HAND_BONES:
        if valid_point(pts[a], w, h) and valid_point(pts[b], w, h):
            p1 = tuple(np.round(pts[a]).astype(int))
            p2 = tuple(np.round(pts[b]).astype(int))
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        if not valid_point(pt, w, h):
            continue
        visible += 1
        p = tuple(np.round(pt).astype(int))
        cv2.circle(img, p, radius + 2, BLACK, -1, cv2.LINE_AA)
        cv2.circle(img, p, radius, color, -1, cv2.LINE_AA)
        if draw_indices:
            cv2.putText(img, str(i), (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return visible


def mean_error(visual: Optional[np.ndarray], glove: Optional[np.ndarray], width: int, height: int) -> Tuple[Optional[float], Optional[float], int]:
    if visual is None or glove is None:
        return None, None, 0
    mask = np.array([valid_point(visual[i], width, height) and valid_point(glove[i], width, height) for i in range(21)])
    if not np.any(mask):
        return None, None, 0
    d = np.linalg.norm(visual[mask] - glove[mask], axis=1)
    wrist = None
    if mask[0]:
        wrist = float(np.linalg.norm(visual[0] - glove[0]))
    return float(np.mean(d)), wrist, int(mask.sum())


def visualize(args: argparse.Namespace) -> Dict:
    rows = read_jsonl(Path(args.input_jsonl).expanduser().resolve())
    if args.max_frames is not None:
        rows = rows[: args.max_frames]

    out_path = Path(args.out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    stats = {
        "frames": 0,
        "written": 0,
        "visual_present": 0,
        "glove_projected": 0,
        "missing_rgb": 0,
    }
    mean_errs = []
    wrist_errs = []
    visible_counts = []

    repeats = repeat_counts(rows, output_fps=args.fps)
    for row_i, row in enumerate(tqdm(rows, desc="Visualizing glove FK vs visual")):
        stats["frames"] += 1
        rgb_path = row.get("rgb_path")
        img = cv2.imread(str(rgb_path)) if rgb_path else None
        if img is None:
            stats["missing_rgb"] += 1
            continue

        visual = load_visual_2d(row)
        glove = load_glove_2d(row)
        if visual is not None:
            stats["visual_present"] += 1
        if glove is not None:
            stats["glove_projected"] += 1

        overlay = img.copy()
        draw_skeleton(overlay, visual, VISUAL_COLOR, radius=3, thickness=2, draw_indices=args.draw_indices)
        draw_skeleton(overlay, glove, GLOVE_COLOR, radius=4, thickness=2, draw_indices=args.draw_indices)
        img = cv2.addWeighted(overlay, 0.88, img, 0.12, 0)

        h, w = img.shape[:2]
        err, wrist_err, n_common = mean_error(visual, glove, w, h)
        if err is not None:
            mean_errs.append(err)
            visible_counts.append(n_common)
        if wrist_err is not None:
            wrist_errs.append(wrist_err)

        frame = row.get("frame", "")
        draw_text(img, f"{frame}  visual=yellow  gloveFK=magenta", (10, 24))
        if err is None:
            draw_text(img, "2D diff: n/a", (10, 48))
        else:
            wrist_text = "n/a" if wrist_err is None else f"{wrist_err:.1f}px"
            draw_text(img, f"2D diff mean={err:.1f}px wrist={wrist_text} common={n_common}/21", (10, 48))

        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {out_path}")
        for _ in range(repeats[row_i]):
            writer.write(img)
            stats["written"] += 1

    if writer is not None:
        writer.release()

    summary = {
        "input_jsonl": str(Path(args.input_jsonl).expanduser().resolve()),
        "out_path": str(out_path),
        "fps": args.fps,
        "stats": stats,
    }
    if mean_errs:
        arr = np.asarray(mean_errs, dtype=np.float64)
        summary["mean_2d_error_px"] = {
            "min": float(arr.min()),
            "median": float(np.median(arr)),
            "max": float(arr.max()),
            "p95": float(np.percentile(arr, 95)),
        }
        summary["common_points"] = {
            "min": int(np.min(visible_counts)),
            "median": float(np.median(visible_counts)),
            "max": int(np.max(visible_counts)),
        }
    if wrist_errs:
        arr = np.asarray(wrist_errs, dtype=np.float64)
        summary["wrist_2d_error_px"] = {
            "min": float(arr.min()),
            "median": float(np.median(arr)),
            "max": float(arr.max()),
            "p95": float(np.percentile(arr, 95)),
        }

    if args.summary_json:
        write_json(Path(args.summary_json).expanduser().resolve(), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overlay visual hand kpts and projected glove FK kpts.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--draw_indices", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = visualize(args)
    print(f"[VisualizeGloveFkVsVisual] video: {args.out_path}")
    print(f"[VisualizeGloveFkVsVisual] summary: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
