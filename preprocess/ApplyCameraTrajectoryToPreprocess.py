#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply an external per-frame camera trajectory to extracted preprocess frames."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict, atomic: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_path = path.with_name(f".{path.name}.{os.getpid()}.tmp") if atomic else path
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")
    if atomic:
        output_path.replace(path)


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build(args: argparse.Namespace) -> Dict:
    all_data_dir = Path(args.all_data_dir).expanduser().resolve()
    trajectory_jsonl = Path(args.trajectory_jsonl).expanduser().resolve()
    pose_rows = {str(row["frame"]): row for row in read_jsonl(trajectory_jsonl)}

    updated = 0
    missing_pose = []
    missing_frame_json = []
    pending = []
    frame_dirs = sorted(p for p in all_data_dir.iterdir() if p.is_dir())
    for frame_dir in frame_dirs:
        frame = frame_dir.name
        pose_row = pose_rows.get(frame)
        cam_path = frame_dir / args.camera_json_name
        if pose_row is None:
            missing_pose.append(frame)
            continue
        if not cam_path.exists():
            missing_frame_json.append(frame)
            continue

        cam = read_json(cam_path)
        camera = pose_row["camera"]
        if "c2w_before_rtabmap_pose" not in cam:
            cam["c2w_before_rtabmap_pose"] = cam.get("c2w")
        cam["c2w"] = camera["c2w"]
        if "rgb_frame_id_before_rtabmap_pose" not in cam:
            cam["rgb_frame_id_before_rtabmap_pose"] = cam.get("rgb_frame_id")
        cam["rgb_frame_id"] = camera.get("frame_id", "rtabmap_pose")
        old_pose_source = cam.get("pose_source")
        original_pose_source = cam.get("pose_source_before_rtabmap_pose", old_pose_source)
        if "pose_source_before_rtabmap_pose" not in cam:
            cam["pose_source_before_rtabmap_pose"] = original_pose_source
        cam["pose_source"] = {
            "method": "rtabmap_node_pose_interpolated_to_rgb_timestamp",
            "world_frame": camera.get("frame_id", "rtabmap_pose"),
            "rgb_stamp_ns": int(pose_row["rgb_stamp_ns"]),
            "stamp": float(pose_row["stamp"]),
            "trajectory_jsonl": str(trajectory_jsonl),
            "previous_pose_source": original_pose_source,
        }
        sync = dict(cam.get("sync") or {})
        sync["rtabmap_pose_stamp_ns"] = int(pose_row["rgb_stamp_ns"])
        sync["rtabmap_pose_stamp_sec"] = float(pose_row["stamp"])
        cam["sync"] = sync
        pending.append((cam_path, cam))

    total_frames = len(frame_dirs)
    summary = {
        "all_data_dir": str(all_data_dir),
        "trajectory_jsonl": str(trajectory_jsonl),
        "camera_json_name": args.camera_json_name,
        "trajectory_frames": len(pose_rows),
        "total_frame_dirs": total_frames,
        "updated_frames": 0,
        "coverage_ratio": float(len(pending) / total_frames) if total_frames else 0.0,
        "missing_pose_count": len(missing_pose),
        "missing_frame_json_count": len(missing_frame_json),
        "missing_pose_frames": missing_pose[:20],
        "missing_frame_json_frames": missing_frame_json[:20],
        "require_full_coverage": bool(args.require_full_coverage),
    }
    if args.require_full_coverage and (missing_pose or missing_frame_json or len(pending) != total_frames):
        summary.update({"status": "failed", "reason": "incomplete trajectory application coverage"})
        if args.summary_json:
            write_json(Path(args.summary_json).expanduser().resolve(), summary, atomic=True)
        raise RuntimeError(
            "Refusing partial RTAB-Map application: "
            f"ready={len(pending)}/{total_frames}, missing_pose={len(missing_pose)}, "
            f"missing_frame_json={len(missing_frame_json)}"
        )

    for cam_path, cam in pending:
        write_json(cam_path, cam, atomic=True)
        updated += 1

    summary.update({"status": "complete", "updated_frames": updated})
    if args.summary_json:
        write_json(Path(args.summary_json).expanduser().resolve(), summary, atomic=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply interpolated camera c2w poses to preprocess/all_data frame JSON files.")
    parser.add_argument("--all_data_dir", required=True)
    parser.add_argument("--trajectory_jsonl", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--camera_json_name", default="aria_cam_rgb.json")
    parser.add_argument(
        "--require_full_coverage",
        action="store_true",
        help="Validate all frame/pose inputs before writing and fail instead of applying a partial trajectory.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build(args)
    print(f"[ApplyCameraTrajectoryToPreprocess] summary: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
