#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build per-RGB-frame fusion input from visual hand JSON and /hand_frame."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def nearest(rows: List[Dict], stamps: List[int], stamp: int, max_dt_ns: Optional[int]) -> Tuple[Optional[Dict], Optional[int]]:
    if not rows:
        return None, None
    pos = bisect.bisect_left(stamps, stamp)
    candidates = []
    if pos < len(rows):
        candidates.append(rows[pos])
    if pos > 0:
        candidates.append(rows[pos - 1])
    best = min(candidates, key=lambda r: abs(int(r["_sync_stamp_ns"]) - stamp))
    dt = int(best["_sync_stamp_ns"]) - stamp
    if max_dt_ns is not None and abs(dt) > max_dt_ns:
        return None, dt
    return best, dt


def finite_solve_state(values: List[float]) -> bool:
    if not values:
        return False
    arr = np.asarray(values, dtype=np.float64)
    return bool(np.isfinite(arr).all() and np.nanmax(np.abs(arr)) < 1e8)


def hand_valid_imu(samples: List[Dict]) -> bool:
    if not samples:
        return False
    arr = np.asarray([s.get("q_wxyz", [0, 0, 0, 0]) for s in samples], dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1)
    return bool(np.any(norms > 0.5))


def pick_visual_hand(data: Dict, visual_side: str) -> Tuple[Optional[Dict], str]:
    if visual_side in ("hand_r", "right"):
        return data.get("hand_r"), "hand_r"
    if visual_side in ("hand_l", "left"):
        return data.get("hand_l"), "hand_l"
    if data.get("hand_r"):
        return data.get("hand_r"), "hand_r"
    if data.get("hand_l"):
        return data.get("hand_l"), "hand_l"
    return None, "none"


def build(args: argparse.Namespace) -> Dict:
    session = Path(args.session_path).expanduser().resolve()
    rgbd_dir = session / args.rgbd_subdir
    all_data = rgbd_dir / "all_data"
    timestamps = read_jsonl(rgbd_dir / "timestamps.jsonl")
    odom_rows = {r["frame"]: r for r in read_jsonl(rgbd_dir / "odom.jsonl")}
    hand_rows = read_jsonl(rgbd_dir / "hand_frame.jsonl")

    hand_sync_key = args.hand_sync_key
    for row in hand_rows:
        stamp = int(row.get(hand_sync_key, 0) or 0)
        if stamp <= 0:
            stamp = int(row.get("bag_time_ns", 0))
        row["_sync_stamp_ns"] = stamp
    hand_rows = sorted(hand_rows, key=lambda r: int(r["_sync_stamp_ns"]))
    hand_stamps = [int(r["_sync_stamp_ns"]) for r in hand_rows]

    max_dt_ns = None if args.max_dt_ms < 0 else int(args.max_dt_ms * 1e6)
    out_rows = []
    dt_values = []
    stats = {
        "frames": 0,
        "matched_hand_frame": 0,
        "missing_hand_frame": 0,
        "visual_hand_present": 0,
        "left_imu_valid": 0,
        "right_imu_valid": 0,
        "left_solve_valid": 0,
        "right_solve_valid": 0,
    }

    for t in timestamps:
        frame = t["frame"]
        rgb_stamp = int(t["rgb_stamp_ns"])
        cam_json = load_json(all_data / frame / "aria_cam_rgb.json")
        rgb_bag_time_ns = int(cam_json.get("sync", {}).get("rgb_bag_time_ns") or t.get("rgb_bag_time_ns") or rgb_stamp)
        target_stamp = rgb_bag_time_ns if hand_sync_key == "bag_time_ns" else rgb_stamp
        hand_row, dt_ns = nearest(hand_rows, hand_stamps, target_stamp, max_dt_ns)
        visual_json = load_json(all_data / frame / args.visual_json_name)
        visual_hand, visual_side = pick_visual_hand(visual_json, args.visual_side)
        odom = odom_rows.get(frame)

        row = {
            "frame": frame,
            "idx": int(t.get("idx", int(frame))),
            "rgb_stamp_ns": rgb_stamp,
            "rgb_bag_time_ns": rgb_bag_time_ns,
            "rgb_path": t.get("rgb_path"),
            "depth_aligned_path": t.get("depth_aligned_path"),
            "camera": {
                "k": cam_json.get("k"),
                "c2w": cam_json.get("c2w"),
                "rgb_frame_id": cam_json.get("rgb_frame_id"),
                "depth_frame_id": cam_json.get("depth_frame_id"),
            },
            "odom": odom,
            "visual": {
                "json_name": args.visual_json_name,
                "side": visual_side,
                "hand": visual_hand,
            },
            "hand_frame_sync": {
                "sync_key": hand_sync_key,
                "target_stamp_ns": target_stamp,
                "stamp_ns": None if hand_row is None else int(hand_row["_sync_stamp_ns"]),
                "dt_ns": dt_ns,
                "dt_ms": None if dt_ns is None else dt_ns / 1e6,
            },
            "hand_frame": None,
            "fusion_mapping": {
                "visual_side": visual_side,
                "glove_side": args.glove_side,
                "note": "HaMeR handedness may be forced for MANO orientation; glove_side identifies /hand_frame physical glove stream.",
            },
        }

        stats["frames"] += 1
        if visual_hand:
            stats["visual_hand_present"] += 1
        if hand_row is None:
            stats["missing_hand_frame"] += 1
        else:
            stats["matched_hand_frame"] += 1
            if dt_ns is not None:
                dt_values.append(dt_ns / 1e6)
            left_valid = hand_valid_imu(hand_row.get("imu_left", []))
            right_valid = hand_valid_imu(hand_row.get("imu_right", []))
            left_solve = finite_solve_state(hand_row.get("solve_state_left", []))
            right_solve = finite_solve_state(hand_row.get("solve_state_right", []))
            stats["left_imu_valid"] += int(left_valid)
            stats["right_imu_valid"] += int(right_valid)
            stats["left_solve_valid"] += int(left_solve)
            stats["right_solve_valid"] += int(right_solve)
            row["hand_frame"] = {
                "bag_time_ns": hand_row.get("bag_time_ns"),
                "imu_stamp_left_ns": hand_row.get("imu_stamp_left_ns"),
                "imu_stamp_right_ns": hand_row.get("imu_stamp_right_ns"),
                "pressure_stamp_left_ns": hand_row.get("pressure_stamp_left_ns"),
                "pressure_stamp_right_ns": hand_row.get("pressure_stamp_right_ns"),
                "imu_left": hand_row.get("imu_left"),
                "imu_right": hand_row.get("imu_right"),
                "pressure_left": hand_row.get("pressure_left"),
                "pressure_right": hand_row.get("pressure_right"),
                "solve_state_left": hand_row.get("solve_state_left"),
                "solve_state_right": hand_row.get("solve_state_right"),
                "valid": {
                    "left_imu": left_valid,
                    "right_imu": right_valid,
                    "left_solve_state": left_solve,
                    "right_solve_state": right_solve,
                },
            }
        out_rows.append(row)

    summary = {
        "session_path": str(session),
        "rgbd_dir": str(rgbd_dir),
        "visual_json_name": args.visual_json_name,
        "hand_sync_key": hand_sync_key,
        "max_dt_ms": args.max_dt_ms,
        "stats": stats,
    }
    if dt_values:
        arr = np.asarray(dt_values, dtype=np.float64)
        summary["hand_frame_dt_ms"] = {
            "min": float(arr.min()),
            "median": float(np.median(arr)),
            "max": float(arr.max()),
            "p95_abs": float(np.percentile(np.abs(arr), 95)),
        }
    write_jsonl(Path(args.output_jsonl).expanduser().resolve(), out_rows)
    if args.summary_json:
        write_json(Path(args.summary_json).expanduser().resolve(), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build per-frame visual+hand_frame fusion input JSONL.")
    parser.add_argument("--session_path", required=True)
    parser.add_argument("--rgbd_subdir", default="rosbag_rgbd_handframe")
    parser.add_argument("--visual_json_name", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--hand_sync_key", default="bag_time_ns", choices=["imu_stamp_left_ns", "bag_time_ns", "pressure_stamp_left_ns"])
    parser.add_argument("--max_dt_ms", type=float, default=80.0)
    parser.add_argument("--visual_side", default="auto", choices=["auto", "left", "right", "hand_l", "hand_r"])
    parser.add_argument("--glove_side", default="left", choices=["left", "right"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build(args)
    print(f"[BuildHandFusionInput] output: {args.output_jsonl}")
    print(f"[BuildHandFusionInput] summary: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
