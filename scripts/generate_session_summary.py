#!/usr/bin/env python3
"""Build one machine-readable summary for a postprocess session."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def get(data: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def relative(path: Path, session: Path) -> str:
    try:
        return path.relative_to(session).as_posix()
    except ValueError:
        return str(path)


def artifact(path: Path, session: Path) -> dict[str, Any]:
    exists = path.is_file() and path.stat().st_size > 0
    return {
        "path": relative(path, session),
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
    }


def stage_statuses(cache_dir: Path) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    if not cache_dir.is_dir():
        return stages
    for path in sorted(cache_dir.glob("*.json")):
        payload = load_json(path)
        if payload is None or not isinstance(payload.get("stage"), str):
            continue
        outputs = payload.get("outputs") if isinstance(payload.get("outputs"), list) else []
        stages[payload["stage"]] = {
            "status": payload.get("status", "unknown"),
            "completed_at": payload.get("completed_at"),
            "output_count": len(outputs),
        }
    return stages


def side_metrics(metrics: dict[str, Any], side: str) -> dict[str, Any]:
    return {
        "matched_hand_frame_ratio": metrics.get(f"{side}_matched_hand_frame_ratio"),
        "visual_hand_ratio": metrics.get(f"{side}_visual_hand_ratio"),
        "calibration_fit_median_m": metrics.get(f"{side}_calibration_fit_median_m"),
        "calibration_fit_p95_m": metrics.get(f"{side}_calibration_fit_p95_m"),
        "wrist_track_residual_p95_m": metrics.get(f"{side}_wrist_track_residual_p95_m"),
        "palm_rotation_step_max_deg": metrics.get(f"{side}_palm_frame_rotation_step_max_deg"),
        "hamer_rotation_step_max_deg": metrics.get(f"{side}_hamer_global_rotation_step_max_deg"),
        "wrist_translation_step_max_m": metrics.get(f"{side}_wrist_translation_step_max_m"),
        "fk_local_pose_rms_step_max_m": metrics.get(f"{side}_fk_local_pose_rms_step_max_m"),
        "hamer_branch_repaired_frames": metrics.get(f"{side}_hamer_global_branch_repaired_frames"),
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    session = Path(args.session).expanduser().resolve()
    outputs = session / "outputs"
    summaries = outputs / "summaries"
    quality_path = outputs / "quality_report.json"
    quality = load_json(quality_path)
    if quality is None:
        quality_path = summaries / "quality_report.json"
        quality = load_json(quality_path)
    quality_metrics = get(quality, "metrics", default={})
    if not isinstance(quality_metrics, dict):
        quality_metrics = {}

    trajectory = load_json(summaries / "trajectory_summary.json")
    motion = load_json(summaries / "motion_filter_summary.json")
    rtabmap = load_json(summaries / "rtabmap_trajectory_summary.json")
    rtabmap_apply = load_json(summaries / "rtabmap_apply_summary.json")
    manifest = load_json(outputs / "manifest.json")
    action_export = load_json(summaries / "camera_relative_actions_summary.json")
    hardware_dir = outputs / "hardware"
    hardware_trajectory_path = hardware_dir / "dual_ur5e_hardware_trajectory.json"
    hardware_npz_path = hardware_dir / "dual_ur5e_hardware_trajectory.npz"
    end_effector_trajectory_path = hardware_dir / "dual_ur5e_end_effector_trajectory.json"
    end_effector_npz_path = hardware_dir / "dual_ur5e_end_effector_trajectory.npz"
    hardware_preflight_path = hardware_dir / "preflight_report.json"
    hardware_trajectory = load_json(hardware_trajectory_path)
    end_effector_trajectory = load_json(end_effector_trajectory_path)
    hardware_preflight = load_json(hardware_preflight_path)

    simulation_dir = outputs / "simulation"
    simulation_summaries = sorted(simulation_dir.glob("*_mink_dual_ur5e_summary.json"))
    simulation_path = simulation_summaries[-1] if simulation_summaries else simulation_dir / "mink_summary.json"
    simulation = load_json(simulation_path)
    training_quality_paths = sorted(simulation_dir.glob("*_mink_training_quality.json"))
    training_quality_path = training_quality_paths[-1] if training_quality_paths else simulation_dir / "mink_training_quality.json"
    training_quality = load_json(training_quality_path)
    simulation_npz = simulation_dir / "missing.npz"
    if simulation is not None and simulation.get("npz"):
        candidate = Path(str(simulation["npz"])).expanduser()
        simulation_npz = candidate if candidate.is_absolute() else session / candidate
    if not simulation_npz.is_file():
        simulation_npzs = sorted(simulation_dir.glob("*_mink_dual_ur5e.npz"))
        simulation_npz = simulation_npzs[-1] if simulation_npzs else simulation_npz
    simulation_video = None
    if simulation is not None and simulation.get("video"):
        candidate = Path(str(simulation["video"])).expanduser()
        simulation_video = candidate if candidate.is_absolute() else session / candidate
    if simulation_video is None:
        videos = sorted(simulation_dir.glob("*_mink_dual_ur5e.mp4"))
        simulation_video = videos[-1] if videos else simulation_dir / "robot_simulation.mp4"

    stages = stage_statuses(session / ".pipeline_cache")
    pipeline_exit_code = args.pipeline_exit_code
    quality_available = quality is not None
    quality_passed = get(quality, "passed") if quality_available else None
    simulation_available = simulation is not None
    simulation_verdict = str(get(simulation, "verdict", default="NOT_AVAILABLE")).upper()
    trajectory_frames = quality_metrics.get("trajectory_frames", get(motion, "frame_count", default=get(trajectory, "frames")))

    categories: list[str] = []
    details: list[str] = []
    if pipeline_exit_code not in (None, 0):
        categories.append("pipeline_exit_nonzero")
        details.append(f"pipeline exited with code {pipeline_exit_code}")
    if quality_available and quality_passed is not True:
        categories.append("quality_failed")
        details.extend(str(item) for item in get(quality, "failures", default=[]) or [])
    elif args.quality_requested and not quality_available:
        categories.append("quality_not_available")
    if args.simulation_requested and not simulation_available:
        categories.append("simulation_not_available")
    elif simulation_available and simulation_verdict != "PASS":
        categories.append("simulation_failed")
        failed = get(simulation, "final_failed_frames", default=[]) or []
        details.append(f"simulation verdict={simulation_verdict}, failed_frames={len(failed)}")
    if not trajectory_frames:
        categories.append("trajectory_not_available")
    if not (outputs / "web" / "index.html").is_file():
        categories.append("web_not_available")
    categories = list(dict.fromkeys(categories))
    details = list(dict.fromkeys(details))

    postprocess_complete = bool(trajectory_frames) and (not args.quality_requested or quality_available)
    if pipeline_exit_code not in (None, 0) or categories:
        overall_verdict = "FAIL"
    elif postprocess_complete:
        overall_verdict = "PASS"
    else:
        overall_verdict = "INCOMPLETE"

    artifacts = {
        "manifest": artifact(outputs / "manifest.json", session),
        "trajectory": artifact(outputs / "data" / "trajectory_wristroot_track_cameraoptical.jsonl", session),
        "quality_report": artifact(quality_path, session),
        "camera_relative_actions": artifact(outputs / "data" / "camera_relative_actions.jsonl", session),
        "camera_relative_actions_summary": artifact(summaries / "camera_relative_actions_summary.json", session),
        "mink_training_quality": artifact(training_quality_path, session),
        "simulation_npz": artifact(simulation_npz, session),
        "simulation_summary": artifact(simulation_path, session),
        "simulation_video": artifact(simulation_video, session),
        "review_web": artifact(outputs / "web" / "index.html", session),
        "review_simulation_video": artifact(outputs / "web" / "robot_simulation.mp4", session),
        "hardware_trajectory_npz": artifact(hardware_npz_path, session),
        "hardware_trajectory_json": artifact(hardware_trajectory_path, session),
        "hardware_preflight": artifact(hardware_preflight_path, session),
        "end_effector_trajectory_npz": artifact(end_effector_npz_path, session),
        "end_effector_trajectory_json": artifact(end_effector_trajectory_path, session),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session": {
            "name": session.name,
            "path": str(session),
            "pipeline_exit_code": pipeline_exit_code,
        },
        "overall": {
            "verdict": overall_verdict,
            "postprocess_complete": postprocess_complete,
            "quality_passed": quality_passed,
            "simulation_verdict": simulation_verdict if simulation_available else "NOT_AVAILABLE",
            "failure_categories": categories,
            "failure_details": details,
        },
        "trajectory": {
            "frames": trajectory_frames,
            "duration_sec": quality_metrics.get("trajectory_duration_sec"),
            "nominal_fps": quality_metrics.get("trajectory_nominal_fps"),
            "active_sides": quality_metrics.get("active_glove_sides", get(trajectory, "active_sides", default=[])),
            "both_valid_frames_before_filter": get(trajectory, "both_valid"),
            "frame_valid_ratio": quality_metrics.get("motion_filter_frame_valid_ratio", get(motion, "frame_valid_ratio")),
            "static_candidate": quality_metrics.get("motion_filter_static_candidate", get(motion, "static", "static_candidate")),
            "terminal_trimmed_frames": quality_metrics.get("motion_filter_terminal_trimmed_frames", get(motion, "terminal_trim", "trimmed_frame_count")),
            "episode_pass": quality_metrics.get("motion_filter_episode_pass", get(motion, "episode_pass")),
            "episode_failures": quality_metrics.get("motion_filter_episode_failures", get(motion, "episode_failures", default=[])),
        },
        "perception": {
            "depth_applied_ratio": quality_metrics.get("depth_applied_ratio"),
            "left": side_metrics(quality_metrics, "left"),
            "right": side_metrics(quality_metrics, "right"),
        },
        "camera_tracking": {
            "available": rtabmap is not None or rtabmap_apply is not None,
            "status": get(rtabmap, "status", default=get(rtabmap_apply, "status")),
            "coverage_ratio": quality_metrics.get("rtabmap_coverage_ratio", get(rtabmap, "coverage_ratio")),
            "apply_coverage_ratio": quality_metrics.get("rtabmap_apply_coverage_ratio", get(rtabmap_apply, "coverage_ratio")),
            "max_interp_gap_sec": quality_metrics.get("rtabmap_max_interp_gap_sec", get(rtabmap, "max_bracketing_gap_sec_used")),
            "missing_pose_count": quality_metrics.get("rtabmap_missing_pose_count", get(rtabmap_apply, "missing_pose_count")),
        },
        "quality": {
            "available": quality_available,
            "passed": quality_passed,
            "thresholds": get(quality, "thresholds", default={}),
            "metrics": quality_metrics,
            "warnings": get(quality, "warnings", default=[]),
            "failures": get(quality, "failures", default=[]),
        },
        "simulation": {
            "requested": args.simulation_requested,
            "available": simulation_available,
            "verdict": simulation_verdict if simulation_available else "NOT_AVAILABLE",
            "frame_count": get(simulation, "frame_count"),
            "duration_sec": get(simulation, "time_scaling", "retimed_duration_sec"),
            "minimum_clearance_m": get(simulation, "minimum_clearance_m"),
            "minimum_clearance_frame": get(simulation, "minimum_clearance_frame"),
            "position_error_max_m": get(simulation, "position_error_max_m"),
            "orientation_error_max_deg": get(simulation, "orientation_error_max_deg"),
            "mink_failed_frames": get(simulation, "mink_failed_frames", default=[]),
            "final_failed_frames": get(simulation, "final_failed_frames", default=[]),
            "recovery_frames": get(simulation, "recovery_frames", default=[]),
            "motion_metrics": get(simulation, "motion_metrics", default={}),
            "safety_audit": get(simulation, "safety_audit", default={}),
            "training_quality_score": get(training_quality, "episode", "score"),
            "training_eligible": get(training_quality, "episode", "eligible"),
            "training_weight": get(training_quality, "episode", "training_weight"),
        },
        "training_actions": {
            "available": action_export is not None,
            "method": get(action_export, "method"),
            "horizon_frames": get(action_export, "horizon_frames"),
            "origin": get(action_export, "origin"),
            "required_sides": get(action_export, "required_sides", default=[]),
            "records": get(action_export, "records"),
            "eligible_records": get(action_export, "eligible_records"),
            "eligible_ratio": get(action_export, "eligible_ratio"),
            "coordinate_convention": get(action_export, "coordinate_convention", default={}),
        },
        "hardware_export": {
            "available": hardware_trajectory is not None and hardware_preflight is not None,
            "offline_preflight_pass": get(hardware_preflight, "offline_preflight_pass"),
            "hardware_execution_authorized": get(hardware_preflight, "hardware_execution_authorized", default=False),
            "verdict": get(hardware_preflight, "verdict", default="NOT_AVAILABLE"),
            "frame_count": get(hardware_trajectory, "frame_count"),
            "duration_sec": get(hardware_trajectory, "duration_sec"),
            "action_dimension": len(get(hardware_trajectory, "joint_names_14d", default=[]) or []),
            "layout": get(hardware_trajectory, "layout"),
            "arm_speed_max_rad_s": get(hardware_preflight, "metrics", "arm_speed_max_rad_s"),
            "arm_acceleration_max_rad_s2": get(hardware_preflight, "metrics", "arm_acceleration_max_rad_s2"),
            "minimum_simulation_clearance_m": get(hardware_preflight, "metrics", "minimum_simulation_clearance_m"),
            "hardware_blockers": get(hardware_preflight, "hardware_blockers", default=[]),
            "end_effector_available": end_effector_trajectory is not None,
            "end_effector_coordinate_frame": get(end_effector_trajectory, "coordinate_frame"),
            "end_effector_tcp_site_names": get(end_effector_trajectory, "tcp_site_names", default={}),
            "fk_to_ik_target_position_error_max_m": get(hardware_preflight, "metrics", "fk_to_ik_target_position_error_max_m"),
        },
        "web": {
            "available": artifacts["review_web"]["exists"],
            "rgb_frames": quality_metrics.get("web_rgb_frames"),
            "trajectory_renderer": quality_metrics.get("web_trajectory_renderer"),
            "tactile_renderer": quality_metrics.get("web_tactile_renderer"),
            "simulation_video_available": artifacts["review_simulation_video"]["exists"],
        },
        "collection": {
            "available": manifest is not None,
            "missing": get(manifest, "missing", default=[]),
        },
        "stages": stages,
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--pipeline_exit_code", type=int, default=None)
    parser.add_argument("--simulation_requested", action="store_true")
    parser.add_argument("--quality_requested", action="store_true")
    args = parser.parse_args()
    session = Path(args.session).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else session / "outputs" / "summary.json"
    payload = build_summary(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(output), "verdict": payload["overall"]["verdict"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
