#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimate a fixed rotation from glove FK canonical frame to camera frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_INDICES = [1, 2, 5, 6, 9, 10, 13, 14, 17, 18]
IDENTITY3 = np.eye(3, dtype=np.float64)


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


def parse_indices(text: str) -> List[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def in_window(row: Dict, start: Optional[int], end: Optional[int]) -> bool:
    frame = row.get("frame")
    idx = int(row.get("idx", int(frame))) if frame is not None else int(row.get("idx", -1))
    if start is not None and idx < start:
        return False
    if end is not None and idx > end:
        return False
    return True


def get_full_rel(row: Dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    visual_hand = (row.get("visual_prior") or {}).get("hand") or {}
    visual = visual_hand.get("kpts_3d")
    glove = (row.get("glove_fk21") or {}).get("kpts_3d_wrist_relative_m")
    if visual is None or glove is None:
        return None
    visual = np.asarray(visual, dtype=np.float64)
    glove = np.asarray(glove, dtype=np.float64)
    if visual.shape[0] < 21 or visual.shape[1] < 3 or glove.shape[0] < 21 or glove.shape[1] < 3:
        return None
    visual_rel = visual[:, :3] - visual[0, :3]
    glove_rel = glove[:21, :3]
    if not np.isfinite(glove_rel).all() or not np.isfinite(visual_rel[:21, :3]).all():
        return None
    return glove_rel, visual_rel[:21, :3]


def get_pair(row: Dict, indices: Sequence[int]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    full = get_full_rel(row)
    if full is None:
        return None
    glove, visual_rel = full
    idx = np.asarray(indices, dtype=np.int64)
    g = glove[idx, :3]
    v = visual_rel[idx, :3]
    ok = np.isfinite(g).all(axis=1) & np.isfinite(v).all(axis=1)
    if ok.sum() < 3:
        return None
    return g[ok], v[ok]


def normalize(vec: np.ndarray, eps: float = 1e-9) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(vec))
    if not np.isfinite(n) or n < eps:
        return None
    return vec / n


def two_vector_frame(points: np.ndarray, middle_idx: int, thumb_idx: int) -> Optional[np.ndarray]:
    """Build a right-handed frame from wrist/back->middle and wrist/back->thumb constraints."""
    if points.shape[0] <= max(middle_idx, thumb_idx):
        return None
    x_axis = normalize(points[middle_idx, :3])
    thumb = normalize(points[thumb_idx, :3])
    if x_axis is None or thumb is None:
        return None
    y_raw = thumb - x_axis * float(np.dot(thumb, x_axis))
    y_axis = normalize(y_raw)
    if y_axis is None:
        return None
    z_axis = normalize(np.cross(x_axis, y_axis))
    if z_axis is None:
        return None
    y_axis = normalize(np.cross(z_axis, x_axis))
    if y_axis is None:
        return None
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def estimate_rotation(glove_vecs: np.ndarray, visual_vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    h = glove_vecs.T @ visual_vecs
    u, s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    rg = (r @ glove_vecs.T).T
    denom = float(np.sum(rg * rg))
    scale = 1.0 if denom <= 1e-12 else float(np.sum(visual_vecs * rg) / denom)
    return r, s, scale


def mat_to_quat_wxyz(r: np.ndarray) -> List[float]:
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(r)))
        if idx == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
    q = np.asarray([w, x, y, z], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    return [float(v) for v in q]


def rotation_angle_deg(r: np.ndarray) -> float:
    cos = (np.trace(r) - 1.0) * 0.5
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def estimate_constraint_prealign(rows: Sequence[Dict], args: argparse.Namespace) -> Tuple[np.ndarray, Dict]:
    all_g = []
    all_v = []
    frames = []
    for row in rows:
        if not in_window(row, args.frame_start, args.frame_end):
            continue
        full = get_full_rel(row)
        if full is None:
            continue
        glove, visual = full
        fg = two_vector_frame(glove, args.middle_idx, args.thumb_idx)
        fv = two_vector_frame(visual, args.middle_idx, args.thumb_idx)
        if fg is None or fv is None:
            continue
        # x: wrist/back -> middle proximal. y: thumb-side component orthogonal to x.
        # z is included with a lower-noise orthonormal frame so the result is a proper rotation.
        all_g.extend([fg[:, 0], fg[:, 1], fg[:, 2]])
        all_v.extend([fv[:, 0], fv[:, 1], fv[:, 2]])
        frames.append(row.get("frame"))

    if len(frames) < args.constraint_min_frames:
        return IDENTITY3.copy(), {
            "enabled": bool(args.constraint_prealign),
            "applied": False,
            "reason": f"not enough constraint frames: {len(frames)} < {args.constraint_min_frames}",
            "frame_count": len(frames),
            "frames": frames,
            "middle_idx": args.middle_idx,
            "thumb_idx": args.thumb_idx,
        }

    g = np.asarray(all_g, dtype=np.float64)
    v = np.asarray(all_v, dtype=np.float64)
    r, singular_values, _ = estimate_rotation(g, v)
    pred = (r @ g.T).T
    axis_err = np.linalg.norm(pred - v, axis=1)
    return r, {
        "enabled": True,
        "applied": True,
        "frame_count": len(frames),
        "frames": frames,
        "axis_pair_count": int(g.shape[0]),
        "middle_idx": args.middle_idx,
        "thumb_idx": args.thumb_idx,
        "constraints": [
            "glove wrist/back -> middle proximal axis aligns to visual wrist/back -> middle proximal axis",
            "glove wrist/back -> thumb proximal direction aligns to visual wrist/back -> thumb proximal direction after projection orthogonal to middle axis",
        ],
        "singular_values": [float(x) for x in singular_values],
        "rotation_angle_deg": rotation_angle_deg(r),
        "quat_wxyz": mat_to_quat_wxyz(r),
        "axis_fit_error": {
            "mean": float(np.mean(axis_err)),
            "median": float(np.median(axis_err)),
            "max": float(np.max(axis_err)),
        },
    }


def build(args: argparse.Namespace) -> Dict:
    rows = read_jsonl(Path(args.input_jsonl).expanduser().resolve())
    indices = parse_indices(args.indices)
    all_g = []
    all_v = []
    used_frames = []
    per_frame_err = []

    if args.constraint_prealign:
        r_prealign, prealign_info = estimate_constraint_prealign(rows, args)
    else:
        r_prealign = IDENTITY3.copy()
        prealign_info = {
            "enabled": False,
            "applied": False,
            "middle_idx": args.middle_idx,
            "thumb_idx": args.thumb_idx,
        }

    for row in rows:
        if not in_window(row, args.frame_start, args.frame_end):
            continue
        pair = get_pair(row, indices)
        if pair is None:
            continue
        g, v = pair
        g = (r_prealign @ g.T).T
        all_g.append(g)
        all_v.append(v)
        used_frames.append(row.get("frame"))

    if not all_g:
        raise RuntimeError("No valid visual/glove 3D point pairs found for calibration window.")

    glove_vecs = np.concatenate(all_g, axis=0)
    visual_vecs = np.concatenate(all_v, axis=0)
    r, singular_values, scale = estimate_rotation(glove_vecs, visual_vecs)
    use_scale = scale if args.estimate_scale else 1.0
    pred = (use_scale * (r @ glove_vecs.T).T)
    err = np.linalg.norm(pred - visual_vecs, axis=1)

    for row in rows:
        if not in_window(row, args.frame_start, args.frame_end):
            continue
        pair = get_pair(row, indices)
        if pair is None:
            continue
        g, v = pair
        g = (r_prealign @ g.T).T
        p = use_scale * (r @ g.T).T
        per_frame_err.append({
            "frame": row.get("frame"),
            "mean_err_m": float(np.mean(np.linalg.norm(p - v, axis=1))),
            "points": int(g.shape[0]),
        })

    r_total = r @ r_prealign
    summary = {
        "input_jsonl": str(Path(args.input_jsonl).expanduser().resolve()),
        "frame_start": args.frame_start,
        "frame_end": args.frame_end,
        "indices": indices,
        "index_note": "21-point order: wrist, thumb4, index4, middle4, ring4, little4. Defaults use palm/proximal joints, not fingertips.",
        "used_frame_count": len(used_frames),
        "used_frames": used_frames,
        "point_pair_count": int(glove_vecs.shape[0]),
        "R_glove_prealign": r_prealign.tolist(),
        "R_cam_glove": r.tolist(),
        "R_total_glove_to_visual": r_total.tolist(),
        "prealign": prealign_info,
        "prealign_quat_wxyz": mat_to_quat_wxyz(r_prealign),
        "total_quat_wxyz": mat_to_quat_wxyz(r_total),
        "scale": float(use_scale),
        "estimated_scale_raw": float(scale),
        "estimate_scale_enabled": bool(args.estimate_scale),
        "det_R": float(np.linalg.det(r)),
        "det_R_prealign": float(np.linalg.det(r_prealign)),
        "det_R_total": float(np.linalg.det(r_total)),
        "rotation_angle_deg": rotation_angle_deg(r),
        "prealign_rotation_angle_deg": rotation_angle_deg(r_prealign),
        "total_rotation_angle_deg": rotation_angle_deg(r_total),
        "singular_values": [float(v) for v in singular_values],
        "fit_error_m": {
            "mean": float(np.mean(err)),
            "median": float(np.median(err)),
            "max": float(np.max(err)),
            "p95": float(np.percentile(err, 95)),
        },
        "per_frame_error_m": per_frame_err,
        "method": "Optional two-vector glove prealignment, then Wahba/Kabsch residual fit: visual_rel ~= scale * R_cam_glove @ R_glove_prealign @ glove_rel",
    }
    write_json(Path(args.output_json).expanduser().resolve(), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate fixed R_cam_glove from glove FK to visual 3D hand.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--indices", default=",".join(str(i) for i in DEFAULT_INDICES))
    parser.add_argument(
        "--estimate_scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Estimate and apply glove-to-visual scale (default: enabled). Use --no-estimate_scale only for controlled comparisons.",
    )
    parser.add_argument("--constraint_prealign", action="store_true")
    parser.add_argument("--middle_idx", type=int, default=9)
    parser.add_argument("--thumb_idx", type=int, default=2)
    parser.add_argument("--constraint_min_frames", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build(args)
    print(f"[CalibrateGloveFkToCamera] output: {args.output_json}")
    fit = summary["fit_error_m"]
    print(
        "[CalibrateGloveFkToCamera] "
        f"frames={summary['used_frame_count']} points={summary['point_pair_count']} "
        f"scale={summary['scale']:.6f} median={fit['median']:.6f}m p95={fit['p95']:.6f}m"
    )


if __name__ == "__main__":
    main()
