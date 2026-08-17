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


def artifact_specs(session: Path, prompt_tag: str, out_tag: str) -> Iterable[Tuple[str, Path]]:
    locate_dir = session / f"locateanything_{prompt_tag}"
    stable_dir = session / f"locateanything_{prompt_tag}_stable"
    hamer_dir = session / f"hamer_from_stable_locateanything_{prompt_tag}_force_right"
    depth_dir = session / "depth_correct_hamer_force_right"
    fusion_dir = session / "fusion_input_force_right_depthroot"
    visual_smooth_dir = session / "visual_2d_smooth"
    rtabmap_pose_dir = session / "rtabmap_pose"
    smooth_dir = session / "fusion_input_force_right_depthroot_smooth_solve045"
    fk_dir = session / f"glove_fk21_{out_tag}"

    return [
        ("videos/01_locate_bboxes.mp4", locate_dir / "bboxes.mp4"),
        ("videos/02_stable_bbox.mp4", stable_dir / "bboxes_stable.mp4"),
        ("videos/03_visual_21kpts_raw.mp4", hamer_dir / "hamer_21kpts_stablebbox_force_right.mp4"),
        ("videos/04_visual_21kpts_2d_smooth.mp4", visual_smooth_dir / "visual_21kpts_2d_smooth.mp4"),
        ("videos/05_glove_fk_overlay_rawroot.mp4", fk_dir / f"glove_fk_vs_hamer_{out_tag}_overlay.mp4"),
        ("videos/06_glove_fk_overlay_wristroot_track.mp4", fk_dir / f"glove_fk_vs_hamer_{out_tag}_wristroot_track_overlay.mp4"),
        ("videos/07_trajectory_3d_world.mp4", fk_dir / "trajectory_3d_world_wristroot_track_cameraoptical.mp4"),
        ("data/locate_bboxes.json", locate_dir / "bboxes.json"),
        ("data/stable_bboxes.json", stable_dir / "bboxes_stable.json"),
        ("data/visual_2d_smooth.jsonl", visual_smooth_dir / "visual_2d_smooth.jsonl"),
        ("data/fusion_frames.jsonl", fusion_dir / "fusion_frames.jsonl"),
        ("data/glove_fk21_calibrated_frames.jsonl", fk_dir / f"glove_fk21_{out_tag}_calibrated_frames.jsonl"),
        ("data/glove_fk21_wristroot_track_frames.jsonl", fk_dir / f"glove_fk21_{out_tag}_calibrated_wristroot_track_frames.jsonl"),
        ("data/trajectory_wristroot_track_cameraoptical.jsonl", fk_dir / "trajectory_wristroot_track_cameraoptical.jsonl"),
        ("summaries/stable_bbox_tracking.json", stable_dir / "tracking_summary.json"),
        ("summaries/hamer_aggregate.json", hamer_dir / f"hamer_{prompt_tag}_stablebbox_force_right_aggregate.json"),
        ("summaries/depthroot_summary.json", depth_dir / "depthroot_summary.json"),
        ("summaries/rtabmap_depth_diagnostics.json", rtabmap_pose_dir / "rtabmap_depth_diagnostics.json"),
        ("summaries/rtabmap_trajectory_summary.json", rtabmap_pose_dir / "rtabmap_camera_pose_rgb30_interp_summary.json"),
        ("summaries/rtabmap_apply_summary.json", rtabmap_pose_dir / "apply_rtabmap_pose_to_preprocess_summary.json"),
        ("summaries/fusion_summary.json", fusion_dir / "fusion_summary.json"),
        ("summaries/visual_2d_smooth_summary.json", visual_smooth_dir / "visual_2d_smooth_summary.json"),
        ("summaries/glove_smooth_summary.json", smooth_dir / "smooth_solve045_summary.json"),
        ("summaries/glove_fk_calibration.json", fk_dir / "glove_fk_to_camera_calib_smooth_solve045.json"),
        ("summaries/wristroot_track_summary.json", fk_dir / "wristroot_track_summary.json"),
        ("summaries/trajectory_summary.json", fk_dir / "trajectory_wristroot_track_cameraoptical_summary.json"),
    ]


def collect(args: argparse.Namespace) -> Dict:
    session = Path(args.session_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else session / "outputs"
    copied: List[Dict] = []
    missing: List[str] = []

    for rel, src in artifact_specs(session, args.prompt_tag, args.out_tag):
        copy_if_exists(src, output_dir / rel, copied, missing)

    manifest = {
        "session": str(session),
        "output_dir": str(output_dir),
        "prompt_tag": args.prompt_tag,
        "out_tag": args.out_tag,
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = collect(args)
    print(f"[CollectPipelineOutputs] output_dir: {manifest['output_dir']}")
    print(f"[CollectPipelineOutputs] copied: {len(manifest['copied'])}, missing: {len(manifest['missing'])}")
    print(f"[CollectPipelineOutputs] manifest: {manifest['manifest_path']}")


if __name__ == "__main__":
    main()
