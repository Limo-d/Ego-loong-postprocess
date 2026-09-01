#!/usr/bin/env python3
"""Export future end-effector deltas expressed in the current camera frame."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def so3(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def rotation_6d(rotation: np.ndarray) -> list[float]:
    return np.concatenate((rotation[:, 0], rotation[:, 1])).tolist()


def quaternion_wxyz(rotation: np.ndarray) -> list[float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (rotation[2, 1] - rotation[1, 2]) / s, (rotation[0, 2] - rotation[2, 0]) / s, (rotation[1, 0] - rotation[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(rotation)))
        if i == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            q = np.array([(rotation[2, 1] - rotation[1, 2]) / s, 0.25 * s, (rotation[0, 1] + rotation[1, 0]) / s, (rotation[0, 2] + rotation[2, 0]) / s])
        elif i == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            q = np.array([(rotation[0, 2] - rotation[2, 0]) / s, (rotation[0, 1] + rotation[1, 0]) / s, 0.25 * s, (rotation[1, 2] + rotation[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            q = np.array([(rotation[1, 0] - rotation[0, 1]) / s, (rotation[0, 2] + rotation[2, 0]) / s, (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s])
    q /= max(float(np.linalg.norm(q)), 1e-12)
    if q[0] < 0.0:
        q *= -1.0
    return q.tolist()


def rotation_vector(rotation: np.ndarray) -> list[float]:
    q = np.asarray(quaternion_wxyz(rotation), dtype=np.float64)
    vector_norm = float(np.linalg.norm(q[1:]))
    if vector_norm < 1e-12:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(vector_norm, float(q[0]))
    return (q[1:] * (angle / vector_norm)).tolist()


def pose(frame: dict[str, Any], side: str, origin: str, coordinate: str) -> tuple[np.ndarray, np.ndarray] | None:
    hand = frame.get("hands", {}).get(side, {})
    palm = hand.get("palm_frame") if isinstance(hand.get("palm_frame"), dict) else {}
    if palm.get("observed_valid") is not True:
        return None
    value = palm.get(f"{origin}_pose_{coordinate}")
    if not isinstance(value, dict):
        return None
    translation = np.asarray(value.get("translation_m"), dtype=np.float64)
    rotation = np.asarray(value.get("rotation_matrix"), dtype=np.float64)
    if translation.shape != (3,) or rotation.shape != (3, 3) or not np.all(np.isfinite(translation)) or not np.all(np.isfinite(rotation)):
        return None
    return translation, so3(rotation)


def serialize_pose(translation: np.ndarray, rotation: np.ndarray) -> dict[str, Any]:
    return {
        "translation_m": translation.tolist(),
        "rotation_matrix": rotation.tolist(),
        "rotation_6d": rotation_6d(rotation),
        "quaternion_wxyz": quaternion_wxyz(rotation),
    }


def relative_action(current: dict[str, Any], target: dict[str, Any], side: str, origin: str) -> dict[str, Any] | None:
    current_world = pose(current, side, origin, "world")
    target_world = pose(target, side, origin, "world")
    current_camera = pose(current, side, origin, "camera")
    camera = current.get("camera") if isinstance(current.get("camera"), dict) else {}
    c2w = np.asarray(camera.get("c2w"), dtype=np.float64)
    if current_world is None or target_world is None or current_camera is None or c2w.shape != (4, 4):
        return None
    camera_rotation_world = so3(c2w[:3, :3])
    current_position_world, current_rotation_world = current_world
    target_position_world, target_rotation_world = target_world
    delta_translation_camera = camera_rotation_world.T @ (target_position_world - current_position_world)
    delta_rotation_camera = so3(
        camera_rotation_world.T @ target_rotation_world @ current_rotation_world.T @ camera_rotation_world
    )
    target_position_current_camera = camera_rotation_world.T @ (target_position_world - c2w[:3, 3])
    target_rotation_current_camera = so3(camera_rotation_world.T @ target_rotation_world)
    delta_angle_deg = math.degrees(float(np.linalg.norm(rotation_vector(delta_rotation_camera))))
    return {
        "current_pose_camera": serialize_pose(*current_camera),
        "target_pose_in_current_camera": serialize_pose(target_position_current_camera, target_rotation_current_camera),
        "delta_camera": {
            "translation_m": delta_translation_camera.tolist(),
            "rotation_matrix": delta_rotation_camera.tolist(),
            "rotation_6d": rotation_6d(delta_rotation_camera),
            "quaternion_wxyz": quaternion_wxyz(delta_rotation_camera),
            "rotation_vector_rad": rotation_vector(delta_rotation_camera),
            "translation_norm_m": float(np.linalg.norm(delta_translation_camera)),
            "rotation_angle_deg": delta_angle_deg,
        },
    }


def relative_rgb_path(value: Any, session: Path) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    try:
        return path.resolve().relative_to(session).as_posix()
    except (OSError, ValueError):
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_output", required=True)
    parser.add_argument("--simulation_quality", default=None)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--origin", choices=("wrist", "palm"), default="wrist")
    parser.add_argument("--required_sides", choices=("auto", "left", "right", "both"), default="auto")
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be >= 1")

    trajectory_path = Path(args.trajectory).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    summary_output = Path(args.summary_output).expanduser().resolve()
    quality_path = Path(args.simulation_quality).expanduser().resolve() if args.simulation_quality else None
    simulation_quality = load_json(quality_path)
    frame_quality = simulation_quality.get("frames", []) if simulation_quality else []
    episode_quality = simulation_quality.get("episode", {}) if simulation_quality else {}
    frame_quality_by_index = {
        int(row["source_index"]): row for row in frame_quality if isinstance(row, dict) and "source_index" in row
    }

    frames = [json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    availability = {
        side: sum(pose(frame, side, args.origin, "world") is not None for frame in frames)
        for side in ("left", "right")
    }
    if args.required_sides == "both":
        required_sides = ["left", "right"]
    elif args.required_sides in ("left", "right"):
        required_sides = [args.required_sides]
    else:
        required_sides = [side for side in ("left", "right") if availability[side] >= max(1, len(frames) // 2)]
    if not required_sides:
        raise RuntimeError("No stable wrist/palm pose is available for action export")

    output.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    eligible_records = 0
    side_valid_counts = {side: 0 for side in ("left", "right")}
    translation_norms: list[float] = []
    rotation_angles: list[float] = []
    session = trajectory_path.parent.parent.parent
    with output.open("w", encoding="utf-8") as handle:
        for current_index in range(max(0, len(frames) - args.horizon)):
            target_index = current_index + args.horizon
            current = frames[current_index]
            target = frames[target_index]
            actions = {
                side: action
                for side in ("left", "right")
                if (action := relative_action(current, target, side, args.origin)) is not None
            }
            for side, action in actions.items():
                side_valid_counts[side] += 1
                translation_norms.append(float(action["delta_camera"]["translation_norm_m"]))
                rotation_angles.append(float(action["delta_camera"]["rotation_angle_deg"]))
            base_valid = bool(current.get("quality_filter", {}).get("frame_valid", True)) and bool(
                target.get("quality_filter", {}).get("frame_valid", True)
            )
            required_valid = all(side in actions for side in required_sides)
            sim_frame = frame_quality_by_index.get(current_index)
            target_sim_frame = frame_quality_by_index.get(target_index)
            simulation_available = bool(simulation_quality)
            simulation_valid = bool(
                simulation_available
                and episode_quality.get("eligible") is True
                and sim_frame
                and target_sim_frame
                and sim_frame.get("eligible") is True
                and target_sim_frame.get("eligible") is True
            )
            frame_score = min(float(sim_frame.get("score", 0.0)), float(target_sim_frame.get("score", 0.0))) if sim_frame and target_sim_frame else 0.0
            episode_score = float(episode_quality.get("score", 0.0)) if simulation_available else 0.0
            eligible = bool(base_valid and required_valid and simulation_valid)
            training_weight = min(frame_score, episode_score) if eligible else 0.0
            reasons = []
            if not base_valid:
                reasons.append("trajectory_quality_invalid")
            if not required_valid:
                reasons.append("required_hand_pose_missing")
            if not simulation_available:
                reasons.append("mink_quality_unavailable")
            elif not simulation_valid:
                reasons.append("mink_quality_ineligible")
            record = {
                "schema_version": SCHEMA_VERSION,
                "frame_index": current_index,
                "target_frame_index": target_index,
                "frame": current.get('frame'),
                "target_frame": target.get("frame"),
                "timestamp_ns": current.get("timestamp", {}).get("rgb_stamp_ns"),
                "target_timestamp_ns": target.get("timestamp", {}).get("rgb_stamp_ns"),
                "observation": {
                    "rgb": f"outputs/web/rgb_frames/{current.get('frame')}.jpg",
                    "source_rgb": relative_rgb_path(current.get("paths", {}).get("rgb"), session),
                    "camera_intrinsics": current.get("camera", {}).get("k"),
                },
                "action_definition": {
                    "horizon_frames": args.horizon,
                    "origin": args.origin,
                    "reference_frame": "current_head_camera_optical",
                    "camera_axes": "x_right_y_down_z_forward",
                    "translation": "R_world_camera(current)^T * (p_world_target - p_world_current)",
                    "rotation": "R_world_camera(current)^T * R_world_target * R_world_current^T * R_world_camera(current)",
                    "rotation_6d": "first_two_rotation_matrix_columns_column_major",
                },
                "actions": actions,
                "quality": {
                    "eligible": eligible,
                    "training_weight": training_weight,
                    "base_trajectory_valid": base_valid,
                    "required_sides": required_sides,
                    "simulation_available": simulation_available,
                    "mink_episode_score": episode_score if simulation_available else None,
                    "mink_frame_score": frame_score if simulation_available else None,
                    "reasons": reasons,
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            records += 1
            eligible_records += int(eligible)

    def percentile(values: list[float], percentile: float) -> float | None:
        return float(np.percentile(np.asarray(values), percentile)) if values else None

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "camera_relative_end_effector_action_v1",
        "trajectory": str(trajectory_path),
        "output": str(output),
        "simulation_quality": str(quality_path) if quality_path else None,
        "horizon_frames": args.horizon,
        "origin": args.origin,
        "required_sides": required_sides,
        "source_frames": len(frames),
        "records": records,
        "eligible_records": eligible_records,
        "eligible_ratio": eligible_records / records if records else 0.0,
        "side_valid_records": side_valid_counts,
        "delta_translation_norm_m": {"p50": percentile(translation_norms, 50), "p95": percentile(translation_norms, 95), "max": max(translation_norms, default=None)},
        "delta_rotation_angle_deg": {"p50": percentile(rotation_angles, 50), "p95": percentile(rotation_angles, 95), "max": max(rotation_angles, default=None)},
        "coordinate_convention": {
            "reference_frame": "current_head_camera_optical",
            "camera_axes": "x_right_y_down_z_forward",
            "rotation_6d": "first_two_rotation_matrix_columns_column_major",
        },
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "records": records, "eligible": eligible_records}, ensure_ascii=False))


if __name__ == "__main__":
    main()
