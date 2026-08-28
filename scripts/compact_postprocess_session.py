#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compact a finished postprocess session for review/delivery.

This removes reproducible intermediate artifacts while keeping the self-contained
review web page and the most useful final diagnostics under outputs/.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List


TOP_LEVEL_REMOVE_PREFIXES = (
    "locateanything_",
    "hamer_from_",
    "depth_correct_",
    "fusion_input_",
    "visual_2d_smooth",
    "glove_fk21_",
    "hand_config_",
    "handeye_calibration_",
)
TOP_LEVEL_REMOVE_NAMES = {"preprocess", "rtabmap_pose", "calibration_handeye"}

KEEP_OUTPUT_VIDEOS = {
    "07_dual_trajectory_3d_world.mp4",
    "00_rgb_raw.mp4",
    "02_stable_bbox.mp4",
    "04_dual_visual_21kpts_2d_smooth.mp4",
    "04_left_visual_21kpts_2d_smooth.mp4",
    "04_right_visual_21kpts_2d_smooth.mp4",
    "06_left_glove_fk_overlay_wristroot_track.mp4",
    "06_right_glove_fk_overlay_wristroot_track.mp4",
    "07_left_trajectory_3d_world.mp4",
    "07_right_trajectory_3d_world.mp4",
    "07b_trajectory_3d_camera_frame.mp4",
}
REMOVE_OUTPUT_DATA = {
    "left_glove_fk21_wristroot_track_frames.jsonl",
    "right_glove_fk21_wristroot_track_frames.jsonl",
    "left_glove_fk21_calibrated_frames.jsonl",
    "right_glove_fk21_calibrated_frames.jsonl",
    "left_fusion_frames.jsonl",
    "right_fusion_frames.jsonl",
    "left_visual_2d_smooth.jsonl",
    "right_visual_2d_smooth.jsonl",
}
REMOVE_OUTPUT_SUMMARIES = {"hamer_aggregate.json"}


def path_size(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_file():
        return path.stat().st_size, 1
    total = 0
    files = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
            files += 1
    return total, files


def should_remove_top_level(path: Path) -> bool:
    name = path.name
    return name in TOP_LEVEL_REMOVE_NAMES or any(name.startswith(prefix) for prefix in TOP_LEVEL_REMOVE_PREFIXES)


def collect_targets(session: Path) -> List[Path]:
    targets: List[Path] = []
    for item in session.iterdir():
        if item.name == "outputs":
            continue
        if should_remove_top_level(item):
            targets.append(item)

    videos_dir = session / "outputs" / "videos"
    if videos_dir.exists():
        for item in videos_dir.iterdir():
            if item.is_file() and item.name not in KEEP_OUTPUT_VIDEOS:
                targets.append(item)

    data_dir = session / "outputs" / "data"
    if data_dir.exists():
        for name in REMOVE_OUTPUT_DATA:
            item = data_dir / name
            if item.exists():
                targets.append(item)

    summaries_dir = session / "outputs" / "summaries"
    if summaries_dir.exists():
        for name in REMOVE_OUTPUT_SUMMARIES:
            item = summaries_dir / name
            if item.exists():
                targets.append(item)

    web_dir = session / "outputs" / "web"
    if (web_dir / "tactile_hand.png").is_file():
        for name in ("traj_frames", "tactile_frames"):
            legacy = web_dir / name
            if legacy.exists():
                targets.append(legacy)

    return sorted(set(targets), key=lambda p: str(p))


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def write_summary(session: Path, rows: List[Dict[str, Any]], dry_run: bool) -> Path:
    out = session / "outputs" / "compact_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(int(row["bytes"]) for row in rows)
    total_files = sum(int(row["files"]) for row in rows)
    payload = {
        "dry_run": dry_run,
        "removed_bytes": 0 if dry_run else total_bytes,
        "removed_files": 0 if dry_run else total_files,
        "candidate_bytes": total_bytes,
        "candidate_files": total_files,
        "targets": rows,
        "kept": {
            "summary": "outputs/summary.json unified session result (written by the pipeline exit hook)",
            "web": "outputs/web/index.html plus web/rgb_frames, web/tactile_hand.png, and optional web/robot_simulation.mp4 (trajectory/tactile use Canvas)",
            "simulation": "outputs/simulation with source replay, final Mink trajectory, safety summary, and rendered MP4",
            "videos": sorted(KEEP_OUTPUT_VIDEOS),
            "data": ["trajectory_wristroot_track_cameraoptical.jsonl", "locate_bboxes.json", "stable_bboxes.json"],
        },
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact a finished postprocess session.")
    parser.add_argument("--session", required=True, help="postprocess_data session directory")
    parser.add_argument("--dry_run", action="store_true", help="Only report targets; do not delete anything.")
    args = parser.parse_args()

    session = Path(args.session).expanduser().resolve()
    outputs = session / "outputs"
    web = outputs / "web" / "index.html"
    rgb_frames = outputs / "web" / "rgb_frames"
    traj_frames = outputs / "web" / "traj_frames"
    tactile_frames = outputs / "web" / "tactile_frames"
    tactile_hand = outputs / "web" / "tactile_hand.png"
    if not session.exists():
        raise FileNotFoundError(session)
    canvas_web = tactile_hand.is_file()
    legacy_web = traj_frames.is_dir() and tactile_frames.is_dir()
    if not web.exists() or not rgb_frames.exists() or not (canvas_web or legacy_web):
        raise RuntimeError(
            "Review web is not self-contained yet. Run scripts/generate_review_web.py before compacting."
        )

    targets = collect_targets(session)
    rows: List[Dict[str, Any]] = []
    for target in targets:
        size, files = path_size(target)
        rows.append({
            "path": str(target.relative_to(session)),
            "bytes": size,
            "files": files,
        })

    if not args.dry_run:
        for target in targets:
            if target.exists():
                remove_path(target)

    summary = write_summary(session, rows, args.dry_run)
    print(json.dumps({
        "session": str(session),
        "dry_run": args.dry_run,
        "targets": len(rows),
        "candidate_mb": round(sum(row["bytes"] for row in rows) / 1024 / 1024, 2),
        "candidate_files": sum(row["files"] for row in rows),
        "summary": str(summary),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
