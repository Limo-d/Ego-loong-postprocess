#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect user-facing pipeline artifacts into one outputs directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def copy_if_exists(src: Path, dst: Path, copied: List[Dict], missing: List[str]) -> None:
    if not src.exists() or src.stat().st_size == 0:
        missing.append(str(src))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append({"src": str(src), "dst": str(dst), "bytes": dst.stat().st_size})


def artifact_specs(
    session: Path, prompt_tag: str, out_tag: str, active_glove_sides: set[str]
) -> Iterable[Tuple[str, Path]]:
    locate_dir = session / f"locateanything_{prompt_tag}"
    stable_dir = session / f"locateanything_{prompt_tag}_stable"
    hamer_dir = session / f"hamer_from_stable_locateanything_{prompt_tag}_force_right"
    depth_dir = session / "depth_correct_hamer_force_right"
    fusion_dirs = {side: session / f"fusion_input_{side}_depthroot" for side in ("left", "right")}
    visual_smooth_dirs = {side: session / f"visual_2d_smooth_{side}" for side in ("left", "right")}
    rtabmap_pose_dir = session / "rtabmap_pose"
    smooth_dirs = {side: session / f"fusion_input_{side}_depthroot_smooth_solve045" for side in ("left", "right")}
    fk_dirs = {side: session / f"glove_fk21_{out_tag}_{side}" for side in ("left", "right")}
    dual_fk_dir = session / f"glove_fk21_{out_tag}_dual"

    specs = [
        ("videos/01_locate_bboxes.mp4", locate_dir / "bboxes.mp4"),
        ("videos/02_stable_bbox.mp4", stable_dir / "bboxes_stable.mp4"),
        ("videos/03_visual_21kpts_raw.mp4", hamer_dir / "hamer_21kpts_stablebbox_force_right.mp4"),
        ("videos/04_left_visual_21kpts_2d_smooth.mp4", visual_smooth_dirs["left"] / "visual_21kpts_2d_smooth.mp4"),
        ("videos/04_right_visual_21kpts_2d_smooth.mp4", visual_smooth_dirs["right"] / "visual_21kpts_2d_smooth.mp4"),
        ("videos/05_left_glove_fk_overlay_rawroot.mp4", fk_dirs["left"] / f"glove_fk_vs_hamer_{out_tag}_overlay.mp4"),
        ("videos/05_right_glove_fk_overlay_rawroot.mp4", fk_dirs["right"] / f"glove_fk_vs_hamer_{out_tag}_overlay.mp4"),
        ("videos/06_left_glove_fk_overlay_wristroot_track.mp4", fk_dirs["left"] / f"glove_fk_vs_hamer_{out_tag}_wristroot_track_overlay.mp4"),
        ("videos/06_right_glove_fk_overlay_wristroot_track.mp4", fk_dirs["right"] / f"glove_fk_vs_hamer_{out_tag}_wristroot_track_overlay.mp4"),
        ("videos/07_left_trajectory_3d_world.mp4", fk_dirs["left"] / "trajectory_3d_world_wristroot_track_cameraoptical.mp4"),
        ("videos/07_right_trajectory_3d_world.mp4", fk_dirs["right"] / "trajectory_3d_world_wristroot_track_cameraoptical.mp4"),
        ("data/locate_bboxes.json", locate_dir / "bboxes.json"),
        ("data/stable_bboxes.json", stable_dir / "bboxes_stable.json"),
        ("data/left_visual_2d_smooth.jsonl", visual_smooth_dirs["left"] / "visual_2d_smooth.jsonl"),
        ("data/right_visual_2d_smooth.jsonl", visual_smooth_dirs["right"] / "visual_2d_smooth.jsonl"),
        ("data/left_fusion_frames.jsonl", fusion_dirs["left"] / "fusion_frames.jsonl"),
        ("data/right_fusion_frames.jsonl", fusion_dirs["right"] / "fusion_frames.jsonl"),
        ("data/left_glove_fk21_calibrated_frames.jsonl", fk_dirs["left"] / f"glove_fk21_{out_tag}_calibrated_frames.jsonl"),
        ("data/right_glove_fk21_calibrated_frames.jsonl", fk_dirs["right"] / f"glove_fk21_{out_tag}_calibrated_frames.jsonl"),
        ("data/left_glove_fk21_wristroot_track_frames.jsonl", fk_dirs["left"] / f"glove_fk21_{out_tag}_calibrated_wristroot_track_frames.jsonl"),
        ("data/right_glove_fk21_wristroot_track_frames.jsonl", fk_dirs["right"] / f"glove_fk21_{out_tag}_calibrated_wristroot_track_frames.jsonl"),
        ("data/trajectory_wristroot_track_cameraoptical.jsonl", dual_fk_dir / "trajectory_wristroot_track_cameraoptical.jsonl"),
        ("summaries/stable_bbox_tracking.json", stable_dir / "tracking_summary.json"),
        ("summaries/hamer_aggregate.json", hamer_dir / f"hamer_{prompt_tag}_stablebbox_force_right_aggregate.json"),
        ("summaries/depthroot_summary.json", depth_dir / "depthroot_summary.json"),
        ("summaries/rtabmap_depth_diagnostics.json", rtabmap_pose_dir / "rtabmap_depth_diagnostics.json"),
        ("summaries/rtabmap_trajectory_summary.json", rtabmap_pose_dir / "rtabmap_camera_pose_rgb30_interp_summary.json"),
        ("summaries/rtabmap_apply_summary.json", rtabmap_pose_dir / "apply_rtabmap_pose_to_preprocess_summary.json"),
        ("summaries/left_fusion_summary.json", fusion_dirs["left"] / "fusion_summary.json"),
        ("summaries/right_fusion_summary.json", fusion_dirs["right"] / "fusion_summary.json"),
        ("summaries/left_visual_2d_smooth_summary.json", visual_smooth_dirs["left"] / "visual_2d_smooth_summary.json"),
        ("summaries/right_visual_2d_smooth_summary.json", visual_smooth_dirs["right"] / "visual_2d_smooth_summary.json"),
        ("summaries/left_glove_smooth_summary.json", smooth_dirs["left"] / "smooth_solve045_summary.json"),
        ("summaries/right_glove_smooth_summary.json", smooth_dirs["right"] / "smooth_solve045_summary.json"),
        ("summaries/left_glove_fk_calibration.json", fk_dirs["left"] / "glove_fk_to_camera_calib_smooth_solve045.json"),
        ("summaries/right_glove_fk_calibration.json", fk_dirs["right"] / "glove_fk_to_camera_calib_smooth_solve045.json"),
        ("summaries/left_wristroot_track_summary.json", fk_dirs["left"] / "wristroot_track_summary.json"),
        ("summaries/right_wristroot_track_summary.json", fk_dirs["right"] / "wristroot_track_summary.json"),
        ("summaries/left_trajectory_summary.json", fk_dirs["left"] / "trajectory_wristroot_track_cameraoptical_summary.json"),
        ("summaries/right_trajectory_summary.json", fk_dirs["right"] / "trajectory_wristroot_track_cameraoptical_summary.json"),
        ("summaries/trajectory_summary.json", dual_fk_dir / "trajectory_wristroot_track_cameraoptical_summary.json"),
    ]
    inactive_prefixes = []
    for side in ("left", "right"):
        if side not in active_glove_sides:
            inactive_prefixes.extend(
                [
                    f"videos/05_{side}_",
                    f"videos/06_{side}_",
                    f"videos/07_{side}_",
                    f"data/{side}_glove_",
                    f"summaries/{side}_glove_",
                    f"summaries/{side}_wristroot_",
                    f"summaries/{side}_trajectory_",
                ]
            )
    return [spec for spec in specs if not any(spec[0].startswith(prefix) for prefix in inactive_prefixes)]


def collect(args: argparse.Namespace) -> Dict:
    session = Path(args.session_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else session / "outputs"
    copied: List[Dict] = []
    missing: List[str] = []

    active_glove_sides = {side for side in args.active_glove_sides.split(",") if side in {"left", "right"}}
    for rel, src in artifact_specs(session, args.prompt_tag, args.out_tag, active_glove_sides):
        copy_if_exists(src, output_dir / rel, copied, missing)

    manifest = {
        "session": str(session),
        "output_dir": str(output_dir),
        "prompt_tag": args.prompt_tag,
        "out_tag": args.out_tag,
        "active_glove_sides": sorted(active_glove_sides),
        "copied": copied,
        "missing": missing,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)
        f.write("\n")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect pipeline artifacts into one outputs directory.")
    parser.add_argument("--session_path", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--prompt_tag", default="white_glove_with_imu")
    parser.add_argument("--out_tag", default="visual_bones_smooth_solve045")
    parser.add_argument("--active_glove_sides", default="left,right")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = collect(args)
    print(f"[CollectPipelineOutputs] output_dir: {manifest['output_dir']}")
    print(f"[CollectPipelineOutputs] copied: {len(manifest['copied'])}, missing: {len(manifest['missing'])}")
    print(f"[CollectPipelineOutputs] manifest: {manifest['manifest_path']}")


if __name__ == "__main__":
    main()
