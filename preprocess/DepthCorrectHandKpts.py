#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Correct HaMeR hand kpts_3d root depth using aligned RGB depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def frame_dirs(session_path: Path, max_frames: Optional[int]) -> list[Path]:
    all_data = session_path / "preprocess" / "all_data"
    frames = [p for p in sorted(all_data.iterdir()) if p.is_dir() and p.name.isdigit()]
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames


def median_depth(depth: np.ndarray, x: float, y: float, radius: int) -> Tuple[Optional[float], int]:
    h, w = depth.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    x1, x2 = max(0, xi - radius), min(w, xi + radius + 1)
    y1, y2 = max(0, yi - radius), min(h, yi + radius + 1)
    patch = depth[y1:y2, x1:x2]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return None, 0
    units = 1000.0 if depth.dtype == np.uint16 else 1.0
    return float(np.median(valid)) / units, int(valid.size)


def backproject(k: np.ndarray, u: float, v: float, z: float) -> np.ndarray:
    return np.array([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], dtype=np.float64)


def parse_indices(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def candidate_points_2d(pts2: np.ndarray, args: argparse.Namespace) -> List[Dict]:
    candidates: List[Dict] = []
    root_idx = int(args.root_idx)
    candidates.append({"name": f"kpt_{root_idx}", "source": "keypoint", "idx": root_idx, "uv": pts2[root_idx, :2]})
    for idx in parse_indices(args.robust_indices):
        if idx == root_idx:
            continue
        candidates.append({"name": f"kpt_{idx}", "source": "keypoint", "idx": idx, "uv": pts2[idx, :2]})
    palm_indices = parse_indices(args.palm_indices)
    if palm_indices:
        uv = np.mean(pts2[np.asarray(palm_indices), :2], axis=0)
        candidates.append({"name": "palm_center", "source": "mean_keypoints", "indices": palm_indices, "uv": uv})
    return candidates


def robust_root_from_depth(pts2: np.ndarray, pts3: np.ndarray, k: np.ndarray, depth: np.ndarray, args: argparse.Namespace) -> Tuple[Optional[np.ndarray], Dict, str]:
    root_idx = int(args.root_idx)
    samples: List[Dict] = []
    root_candidates = []
    for cand in candidate_points_2d(pts2, args):
        uv = np.asarray(cand["uv"], dtype=np.float64)
        if not np.isfinite(uv).all():
            continue
        z, n = median_depth(depth, float(uv[0]), float(uv[1]), args.depth_radius)
        rec = dict(cand)
        rec["uv"] = [float(uv[0]), float(uv[1])]
        rec["valid_depth_pixels"] = n
        rec["depth_m"] = None if z is None else float(z)
        if z is None:
            rec["status"] = "no_depth"
            samples.append(rec)
            continue
        if not (args.min_depth_m <= z <= args.max_depth_m):
            rec["status"] = "depth_out_of_range"
            samples.append(rec)
            continue
        if rec.get("source") == "mean_keypoints":
            ref3 = np.mean(pts3[np.asarray(rec["indices"]), :3], axis=0)
        else:
            ref3 = pts3[int(rec["idx"]), :3]
        point_cam = backproject(k, float(uv[0]), float(uv[1]), float(z))
        root_cam = point_cam - (ref3 - pts3[root_idx, :3])
        rec["status"] = "candidate"
        rec["point_3d_from_depth"] = point_cam.astype(float).tolist()
        rec["root_3d_candidate"] = root_cam.astype(float).tolist()
        samples.append(rec)
        root_candidates.append(root_cam)

    if not root_candidates:
        detail = {"method": "robust_multi_keypoint_aligned_depth", "samples": samples, "candidate_count": 0}
        return None, detail, "no_depth"

    roots = np.asarray(root_candidates, dtype=np.float64)
    center = np.median(roots, axis=0)
    residuals = np.linalg.norm(roots - center[None, :], axis=1)
    inlier_mask = residuals <= float(args.robust_inlier_m)
    if int(np.count_nonzero(inlier_mask)) < int(args.min_depth_candidates):
        detail = {
            "method": "robust_multi_keypoint_aligned_depth",
            "samples": samples,
            "candidate_count": int(len(root_candidates)),
            "inlier_count": int(np.count_nonzero(inlier_mask)),
            "min_depth_candidates": int(args.min_depth_candidates),
            "residuals_m": [float(v) for v in residuals],
        }
        return None, detail, "too_few_depth_candidates"
    root_cam = np.median(roots[inlier_mask], axis=0)
    detail = {
        "method": "robust_multi_keypoint_aligned_depth",
        "root_idx": root_idx,
        "depth_radius_px": int(args.depth_radius),
        "robust_indices": parse_indices(args.robust_indices),
        "palm_indices": parse_indices(args.palm_indices),
        "candidate_count": int(len(root_candidates)),
        "inlier_count": int(np.count_nonzero(inlier_mask)),
        "robust_inlier_m": float(args.robust_inlier_m),
        "samples": samples,
        "residuals_m": [float(v) for v in residuals],
        "raw_root_3d": pts3[root_idx].astype(float).tolist(),
        "corrected_root_3d": root_cam.astype(float).tolist(),
        "depth_m": float(root_cam[2]),
    }
    return root_cam, detail, "applied"


def correct_hand(hand: Dict, k: np.ndarray, depth: np.ndarray, args: argparse.Namespace) -> Tuple[Optional[Dict], str]:
    pts2 = np.asarray(hand.get("kpts_2d"), dtype=np.float64)
    pts3 = np.asarray(hand.get("kpts_3d"), dtype=np.float64)
    if pts2.shape[0] < 21 or pts3.shape[0] < 21:
        return hand, "bad_shape"

    root_idx = int(args.root_idx)
    if args.method == "root":
        z, n = median_depth(depth, pts2[root_idx, 0], pts2[root_idx, 1], args.depth_radius)
        if z is None:
            return hand, "no_depth"
        if not (args.min_depth_m <= z <= args.max_depth_m):
            return hand, "depth_out_of_range"
        root_cam = backproject(k, pts2[root_idx, 0], pts2[root_idx, 1], z)
        correction = {
            "method": "aligned_depth_root_translation",
            "root_idx": root_idx,
            "root_kpt_2d": [float(pts2[root_idx, 0]), float(pts2[root_idx, 1])],
            "depth_m": float(z),
            "valid_depth_pixels": n,
            "depth_radius_px": args.depth_radius,
            "raw_root_3d": pts3[root_idx].astype(float).tolist(),
            "corrected_root_3d": root_cam.astype(float).tolist(),
        }
    else:
        root_cam, correction, status = robust_root_from_depth(pts2, pts3, k, depth, args)
        if root_cam is None:
            out = json.loads(json.dumps(hand))
            out["depth_root_correction"] = correction
            return out, status
        if args.anchor_root_xy:
            robust_root_cam = root_cam.copy()
            root_cam = backproject(k, pts2[root_idx, 0], pts2[root_idx, 1], float(root_cam[2]))
            correction["anchor_root_xy"] = True
            correction["robust_root_3d_before_xy_anchor"] = robust_root_cam.astype(float).tolist()
            correction["root_kpt_2d"] = [float(pts2[root_idx, 0]), float(pts2[root_idx, 1])]
            correction["corrected_root_3d"] = root_cam.astype(float).tolist()

    rel = pts3 - pts3[root_idx:root_idx + 1]
    corrected = root_cam[None, :] + rel
    out = json.loads(json.dumps(hand))
    out["kpts_3d_raw_hamer"] = pts3.astype(float).tolist()
    out["kpts_3d"] = corrected.astype(float).tolist()
    out["depth_root_correction"] = correction
    return out, "applied"


def process(args: argparse.Namespace) -> Dict:
    session = Path(args.session_path).expanduser().resolve()
    frames = frame_dirs(session, args.max_frames)
    stats = {"frames": 0, "hands": 0, "applied": 0, "no_depth": 0, "depth_out_of_range": 0, "bad_shape": 0, "missing_depth": 0, "too_few_depth_candidates": 0}

    for frame in tqdm(frames, desc="Depth correcting hand kpts"):
        data = load_json(frame / args.input_json_name)
        if not data:
            data = {"idx": int(frame.name), "ts": None, "hand_r": None, "hand_l": None}
        cam = load_json(frame / "aria_cam_rgb.json")
        k = np.asarray(cam.get("k"), dtype=np.float64)
        depth_path = frame / args.depth_name
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        stats["frames"] += 1
        if depth is None or k.shape != (3, 3):
            stats["missing_depth"] += 1
            save_json(frame / args.output_json_name, data)
            continue

        for key in ("hand_r", "hand_l"):
            hand = data.get(key)
            if not hand:
                continue
            stats["hands"] += 1
            corrected, status = correct_hand(hand, k, depth, args)
            stats[status] = stats.get(status, 0) + 1
            data[key] = corrected
        save_json(frame / args.output_json_name, data)

    if args.summary_json:
        save_json(Path(args.summary_json).expanduser().resolve(), stats)
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Use aligned depth to correct hand kpts_3d root translation.")
    parser.add_argument("--session_path", required=True)
    parser.add_argument("--input_json_name", required=True)
    parser.add_argument("--output_json_name", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--depth_name", default="depth_aligned.png")
    parser.add_argument("--root_idx", type=int, default=0)
    parser.add_argument("--depth_radius", type=int, default=6)
    parser.add_argument("--min_depth_m", type=float, default=0.12)
    parser.add_argument("--max_depth_m", type=float, default=1.2)
    parser.add_argument("--method", choices=["root", "robust"], default="robust")
    parser.add_argument("--robust_indices", default="0,5,9,13,17", help="Keypoint indices used as local depth samples. Defaults: wrist + four MCPs.")
    parser.add_argument("--palm_indices", default="0,5,9,13,17", help="Keypoint indices averaged to create a palm-center depth sample.")
    parser.add_argument("--min_depth_candidates", type=int, default=2)
    parser.add_argument("--robust_inlier_m", type=float, default=0.045)
    parser.add_argument("--anchor_root_xy", action=argparse.BooleanOptionalAction, default=True, help="Keep corrected root on the root 2D keypoint ray; robust samples estimate depth only.")
    parser.add_argument("--max_frames", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = process(args)
    print(f"[DepthCorrectHandKpts] output per-frame json: {args.output_json_name}")
    print(f"[DepthCorrectHandKpts] stats: {json.dumps(stats, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
