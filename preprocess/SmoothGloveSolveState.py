#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal smoothing for /hand_frame solve_state before glove FK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from preprocess.Timebase import dt_sec, effective_alpha, row_stamp_ns
except ModuleNotFoundError:
    from Timebase import dt_sec, effective_alpha, row_stamp_ns

SENTINEL_ABS = 1e8


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
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def valid_state(values: Optional[List[float]]) -> bool:
    if values is None or len(values) != 27:
        return False
    arr = np.asarray(values, dtype=np.float64)
    return bool(np.isfinite(arr).all() and np.all(np.abs(arr) < SENTINEL_ABS))


def valid_scalar(v: float) -> bool:
    return np.isfinite(v) and abs(float(v)) < SENTINEL_ABS


def normalize_quat(q: np.ndarray) -> Optional[np.ndarray]:
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.any(np.abs(q) >= SENTINEL_ABS):
        return None
    n = np.linalg.norm(q)
    if n < 1e-9:
        return None
    return q / n


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    if q0 is None:
        return q1
    if q1 is None:
        return q0
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quat((1.0 - alpha) * q0 + alpha * q1)
    theta0 = np.arccos(dot)
    theta = theta0 * alpha
    sin_theta = np.sin(theta)
    sin_theta0 = np.sin(theta0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta0
    s1 = sin_theta / sin_theta0
    return normalize_quat(s0 * q0 + s1 * q1)


def smooth_rows(args: argparse.Namespace) -> Dict:
    rows = read_jsonl(Path(args.input_jsonl).expanduser().resolve())
    solve_key = f"solve_state_{args.glove_side}"
    alpha_angle = float(args.alpha_angle)
    alpha_quat = float(args.alpha_quat)
    reference_fps = float(args.reference_fps)

    prev_angles: Optional[np.ndarray] = None
    prev_cmc: Optional[np.ndarray] = None
    prev_palm: Optional[np.ndarray] = None
    prev_stamp_ns: Optional[int] = None
    out_rows = []
    stats = {
        "frames": 0,
        "state_present": 0,
        "state_smoothed": 0,
        "state_missing": 0,
        "quat_cmc_smoothed": 0,
        "quat_palm_smoothed": 0,
    }

    for row in rows:
        stats["frames"] += 1
        new_row = dict(row)
        hand_frame = row.get("hand_frame")
        if not hand_frame or not valid_state(hand_frame.get(solve_key)):
            stats["state_missing"] += 1
            out_rows.append(new_row)
            continue

        state = np.asarray(hand_frame[solve_key], dtype=np.float64)
        stamp_ns = row_stamp_ns(row)
        frame_dt_sec = dt_sec(prev_stamp_ns, stamp_ns, reference_fps)
        effective_alpha_angle = effective_alpha(alpha_angle, frame_dt_sec, reference_fps)
        effective_alpha_quat = effective_alpha(alpha_quat, frame_dt_sec, reference_fps)
        stats["state_present"] += 1
        smoothed = state.copy()

        angles = state[:19].copy()
        valid = np.array([valid_scalar(v) for v in angles], dtype=bool)
        if prev_angles is None:
            prev_angles = angles.copy()
        else:
            prev_angles[valid] = (
                (1.0 - effective_alpha_angle) * prev_angles[valid]
                + effective_alpha_angle * angles[valid]
            )
        smoothed[:19] = prev_angles

        cmc = normalize_quat(state[19:23])
        if cmc is not None:
            prev_cmc = cmc if prev_cmc is None else slerp(prev_cmc, cmc, effective_alpha_quat)
            smoothed[19:23] = prev_cmc
            stats["quat_cmc_smoothed"] += 1

        palm = normalize_quat(state[23:27])
        if palm is not None:
            prev_palm = palm if prev_palm is None else slerp(prev_palm, palm, effective_alpha_quat)
            smoothed[23:27] = prev_palm
            stats["quat_palm_smoothed"] += 1

        new_hand = dict(hand_frame)
        new_hand[f"{solve_key}_raw_before_smooth"] = hand_frame[solve_key]
        new_hand[solve_key] = [float(v) for v in smoothed]
        valid_dict = dict(new_hand.get("valid") or {})
        valid_dict[f"{args.glove_side}_solve_state_smoothed"] = True
        new_hand["valid"] = valid_dict
        new_row["hand_frame"] = new_hand
        new_row["solve_state_smoothing"] = {
            "glove_side": args.glove_side,
            "alpha_angle": alpha_angle,
            "alpha_quat": alpha_quat,
            "effective_alpha_angle": effective_alpha_angle,
            "effective_alpha_quat": effective_alpha_quat,
            "dt_sec": frame_dt_sec,
            "reference_fps": reference_fps,
            "method": "Timestamp-aware EMA for angle slots 0:19; timestamp-aware quaternion slerp EMA for slots 19:23 and 23:27.",
        }
        prev_stamp_ns = stamp_ns
        stats["state_smoothed"] += 1
        out_rows.append(new_row)

    summary = {
        "input_jsonl": str(Path(args.input_jsonl).expanduser().resolve()),
        "output_jsonl": str(Path(args.output_jsonl).expanduser().resolve()),
        "glove_side": args.glove_side,
        "alpha_angle": alpha_angle,
        "alpha_quat": alpha_quat,
        "reference_fps": reference_fps,
        "timebase": "rgb_stamp_ns",
        "stats": stats,
    }
    write_jsonl(Path(args.output_jsonl).expanduser().resolve(), out_rows)
    if args.summary_json:
        write_json(Path(args.summary_json).expanduser().resolve(), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smooth glove solve_state in fusion_frames.jsonl before FK.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--glove_side", choices=["left", "right"], default="left")
    parser.add_argument("--alpha_angle", type=float, default=0.45)
    parser.add_argument("--alpha_quat", type=float, default=0.45)
    parser.add_argument("--reference_fps", type=float, default=30.0, help="FPS at which alpha values retain their legacy per-frame meaning.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = smooth_rows(args)
    print(f"[SmoothGloveSolveState] output: {args.output_jsonl}")
    print(f"[SmoothGloveSolveState] summary: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
