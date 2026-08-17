#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pack calibrated visual/glove outputs into one per-frame trajectory JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def read_json(path: Optional[Path]) -> Optional[Dict]:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def valid_kpts(value, dims: int = 3) -> bool:
    if value is None:
        return False
    arr = np.asarray(value, dtype=np.float64)
    return bool(arr.shape[0] >= 21 and arr.shape[1] >= dims and np.isfinite(arr[:21, :dims]).all())


def build(args: argparse.Namespace) -> Dict:
    fk_rows = read_jsonl(Path(args.calibrated_fk_jsonl).expanduser().resolve())
    fusion_rows = {
        r["frame"]: r
        for r in read_jsonl(Path(args.smoothed_fusion_jsonl).expanduser().resolve())
    }

    calib = read_json(Path(args.calib_json).expanduser().resolve() if args.calib_json else None)
    hand_config = read_json(Path(args.hand_config_json).expanduser().resolve() if args.hand_config_json else None)
    hand_config_summary = read_json(Path(args.hand_config_summary_json).expanduser().resolve() if args.hand_config_summary_json else None)
    smoothing_summary = read_json(Path(args.smoothing_summary_json).expanduser().resolve() if args.smoothing_summary_json else None)

    out_rows = []
    stats = {
        "frames": 0,
        "matched_fusion": 0,
        "missing_fusion": 0,
        "glove_fk_valid": 0,
        "glove_camera_kpts_valid": 0,
        "glove_world_kpts_valid": 0,
        "visual_prior_valid": 0,
        "hand_frame_present": 0,
        "imu_left_present": 0,
        "solve_state_left_present": 0,
    }

    for fk_row in fk_rows:
        frame = fk_row.get("frame")
        fusion = fusion_rows.get(frame)
        stats["frames"] += 1
        if fusion is None:
            stats["missing_fusion"] += 1
        else:
            stats["matched_fusion"] += 1

        glove_fk = fk_row.get("glove_fk21") or {}
        visual = fk_row.get("visual_prior") or {}
        visual_hand = visual.get("hand") or {}
        hand_frame = (fusion or {}).get("hand_frame") or {}

        if glove_fk.get("valid"):
            stats["glove_fk_valid"] += 1
        if valid_kpts(glove_fk.get("kpts_3d_camera_m"), 3):
            stats["glove_camera_kpts_valid"] += 1
        if valid_kpts(glove_fk.get("kpts_3d_world_m"), 3):
            stats["glove_world_kpts_valid"] += 1
        if valid_kpts(visual_hand.get("kpts_3d"), 3):
            stats["visual_prior_valid"] += 1
        if hand_frame:
            stats["hand_frame_present"] += 1
            if hand_frame.get("imu_left"):
                stats["imu_left_present"] += 1
            if hand_frame.get("solve_state_left"):
                stats["solve_state_left_present"] += 1

        out_rows.append({
            "frame": frame,
            "idx": fk_row.get("idx"),
            "timestamp": {
                "rgb_stamp_ns": fk_row.get("rgb_stamp_ns"),
                "hand_frame_sync": fk_row.get("hand_frame_sync"),
            },
            "paths": {
                "rgb": fk_row.get("rgb_path"),
                "depth_aligned": fk_row.get("depth_aligned_path"),
            },
            "camera": fk_row.get("camera"),
            "head_pose": {
                "odom": fk_row.get("odom"),
                "c2w": (fk_row.get("camera") or {}).get("c2w"),
            },
            "visual_prior": {
                "json_name": visual.get("json_name"),
                "side": visual.get("side"),
                "confidence": visual_hand.get("confidence"),
                "kpts_2d": visual_hand.get("kpts_2d"),
                "kpts_3d_camera_m": visual_hand.get("kpts_3d"),
                "depth_root_correction": visual_hand.get("depth_root_correction"),
            },
            "glove": {
                "side": args.glove_side,
                "mapping": fk_row.get("fusion_mapping"),
                "kpt21_names": glove_fk.get("kpt21_names"),
                "kpts_3d_camera_m": glove_fk.get("kpts_3d_camera_m"),
                "kpts_3d_world_m": glove_fk.get("kpts_3d_world_m"),
                "kpts_3d_wrist_relative_m": glove_fk.get("kpts_3d_wrist_relative_m"),
                "finger_valid": glove_fk.get("finger_valid"),
                "pinch_distances_m": glove_fk.get("pinch_distances_m"),
                "solve_state": glove_fk.get("solve_state"),
                "calibration": glove_fk.get("calibration"),
            },
            "hand_frame": {
                "bag_time_ns": hand_frame.get("bag_time_ns"),
                "imu_stamp_left_ns": hand_frame.get("imu_stamp_left_ns"),
                "pressure_stamp_left_ns": hand_frame.get("pressure_stamp_left_ns"),
                "imu_left": hand_frame.get("imu_left"),
                "pressure_left": hand_frame.get("pressure_left"),
                "pressure_right": hand_frame.get("pressure_right"),
                "solve_state_left": hand_frame.get("solve_state_left"),
                "solve_state_left_raw_before_smooth": hand_frame.get("solve_state_left_raw_before_smooth"),
                "valid": hand_frame.get("valid"),
            },
            "processing": {
                "solve_state_smoothing": (fusion or {}).get("solve_state_smoothing"),
            },
        })

    summary = {
        "output_jsonl": str(Path(args.output_jsonl).expanduser().resolve()),
        "calibrated_fk_jsonl": str(Path(args.calibrated_fk_jsonl).expanduser().resolve()),
        "smoothed_fusion_jsonl": str(Path(args.smoothed_fusion_jsonl).expanduser().resolve()),
        "glove_side": args.glove_side,
        "stats": stats,
        "metadata": {
            "calibration": calib,
            "hand_config_json": str(Path(args.hand_config_json).expanduser().resolve()) if args.hand_config_json else None,
            "hand_config": hand_config,
            "hand_config_summary": hand_config_summary,
            "smoothing_summary": smoothing_summary,
        },
    }

    write_jsonl(Path(args.output_jsonl).expanduser().resolve(), out_rows)
    if args.summary_json:
        write_json(Path(args.summary_json).expanduser().resolve(), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build unified trajectory JSONL from calibrated glove FK and smoothed fusion rows.")
    parser.add_argument("--calibrated_fk_jsonl", required=True)
    parser.add_argument("--smoothed_fusion_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--glove_side", choices=["left", "right"], default="left")
    parser.add_argument("--calib_json", default=None)
    parser.add_argument("--hand_config_json", default=None)
    parser.add_argument("--hand_config_summary_json", default=None)
    parser.add_argument("--smoothing_summary_json", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build(args)
    print(f"[BuildUnifiedTrajectory] output: {args.output_jsonl}")
    print(f"[BuildUnifiedTrajectory] summary: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
