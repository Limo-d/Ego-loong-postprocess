#!/usr/bin/env python3
"""Export a controller-neutral 14D dual-UR5e trajectory from a Mink result.

This is an offline interchange exporter.  It intentionally does not connect to
ROS, a robot controller, or a gripper driver, and it never authorizes execution
on physical hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
LEFT_ARM_INDICES = np.arange(0, 6)
RIGHT_ARM_INDICES = np.arange(14, 20)
ARM_SUFFIXES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
JOINT_NAMES_14D = (
    *(f"left_{name}" for name in ARM_SUFFIXES),
    "left_gripper_normalized",
    *(f"right_{name}" for name in ARM_SUFFIXES),
    "right_gripper_normalized",
)
UNITS_14D = (*("rad" for _ in range(6)), "normalized_0_1", *("rad" for _ in range(6)), "normalized_0_1")
HARDWARE_BLOCKERS = (
    "actual_robot_joint_and_sign_mapping_not_verified",
    "robot_base_extrinsics_not_verified",
    "tcp_and_payload_not_verified",
    "gripper_driver_mapping_not_verified",
    "real_environment_and_collision_model_not_verified",
    "start_state_transition_not_planned",
    "dual_arm_runtime_safety_supervisor_not_connected",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def forward_kinematics(
    qpos: np.ndarray, config_path: Path
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Evaluate the final solved qpos in the same MuJoCo model used by Mink."""
    import mujoco

    repository = Path(__file__).resolve().parent.parent
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
    from simulation.dual_ur5e import replay_trajectory as replay

    config = replay.load_config(config_path)
    model, data = replay.build_model(config)
    if qpos.shape[1] != model.nq:
        raise ValueError(f"qpos width {qpos.shape[1]} does not match model nq={model.nq}")
    site_ids = {side: replay.end_effector_site_id(model, side) for side in ("left", "right")}
    site_names = {side: model.site(site_id).name for side, site_id in site_ids.items()}
    result: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        result[f"{side}_position_m"] = np.empty((len(qpos), 3), dtype=np.float64)
        result[f"{side}_rotation_matrix"] = np.empty((len(qpos), 3, 3), dtype=np.float64)
        result[f"{side}_quaternion_xyzw"] = np.empty((len(qpos), 4), dtype=np.float64)
    for frame, values in enumerate(qpos):
        data.qpos[:] = values
        mujoco.mj_forward(model, data)
        for side, site_id in site_ids.items():
            rotation = data.site_xmat[site_id].reshape(3, 3).copy()
            quaternion_wxyz = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quaternion_wxyz, rotation.reshape(9))
            quaternion_xyzw = quaternion_wxyz[[1, 2, 3, 0]]
            previous = result[f"{side}_quaternion_xyzw"]
            if frame and float(np.dot(previous[frame - 1], quaternion_xyzw)) < 0.0:
                quaternion_xyzw *= -1.0
            result[f"{side}_position_m"][frame] = data.site_xpos[site_id]
            result[f"{side}_rotation_matrix"][frame] = rotation
            result[f"{side}_quaternion_xyzw"][frame] = quaternion_xyzw
    return result, site_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, help="Final Mink trajectory NPZ")
    parser.add_argument("--summary", required=True, help="Matching Mink summary JSON")
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_ee_npz", required=True, help="End-effector FK trajectory NPZ")
    parser.add_argument("--output_ee_json", required=True, help="End-effector FK trajectory JSON")
    parser.add_argument("--preflight_output", required=True)
    parser.add_argument("--config", default=None, help="Simulation config; defaults to Mink summary.config")
    parser.add_argument(
        "--fail_on_preflight",
        action="store_true",
        help="Return exit code 2 when offline checks fail (outputs are still written)",
    )
    args = parser.parse_args()

    source_npz = Path(args.npz).expanduser().resolve()
    source_summary_path = Path(args.summary).expanduser().resolve()
    output_npz = Path(args.output_npz).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_ee_npz = Path(args.output_ee_npz).expanduser().resolve()
    output_ee_json = Path(args.output_ee_json).expanduser().resolve()
    preflight_output = Path(args.preflight_output).expanduser().resolve()
    summary = load_json(source_summary_path)
    raw_config = args.config or summary.get("config")
    if not raw_config:
        raise ValueError("simulation config is required via --config or Mink summary.config")
    config_path = Path(str(raw_config)).expanduser().resolve()

    with np.load(source_npz, allow_pickle=False) as data:
        required = {"times_sec", "qpos_rad", "left_gripper_position_command", "right_gripper_position_command"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"Mink NPZ is missing required arrays: {missing}")
        times = np.asarray(data["times_sec"], dtype=np.float64)
        qpos = np.asarray(data["qpos_rad"], dtype=np.float64)
        left_gripper = np.asarray(data["left_gripper_position_command"], dtype=np.float64)
        right_gripper = np.asarray(data["right_gripper_position_command"], dtype=np.float64)
        target_positions = {
            side: np.asarray(data[f"{side}_target_m"], dtype=np.float64)
            for side in ("left", "right")
            if f"{side}_target_m" in data.files
        }
        target_rotations = {
            side: np.asarray(data[f"{side}_target_rotation_matrix"], dtype=np.float64)
            for side in ("left", "right")
            if f"{side}_target_rotation_matrix" in data.files
        }

    if times.ndim != 1 or len(times) < 3:
        raise ValueError(f"times_sec must be a 1D array with at least 3 samples, got {times.shape}")
    if qpos.ndim != 2 or qpos.shape[0] != len(times) or qpos.shape[1] < 20:
        raise ValueError(f"qpos_rad must have shape (N, >=20), got {qpos.shape}")
    if left_gripper.shape != times.shape or right_gripper.shape != times.shape:
        raise ValueError("gripper position arrays must match times_sec")

    left_arm = qpos[:, LEFT_ARM_INDICES]
    right_arm = qpos[:, RIGHT_ARM_INDICES]
    positions = np.column_stack((left_arm, left_gripper, right_arm, right_gripper))
    finite = bool(np.isfinite(times).all() and np.isfinite(positions).all())
    dt = np.diff(times)
    strictly_increasing = bool(np.isfinite(dt).all() and np.all(dt > 0.0))
    if not finite:
        raise ValueError("trajectory contains NaN or infinity")
    if not strictly_increasing:
        raise ValueError("times_sec must be finite and strictly increasing")

    left_velocity = np.gradient(left_arm, times, axis=0, edge_order=2)
    right_velocity = np.gradient(right_arm, times, axis=0, edge_order=2)
    left_acceleration = np.gradient(left_velocity, times, axis=0, edge_order=2)
    right_acceleration = np.gradient(right_velocity, times, axis=0, edge_order=2)
    left_jerk = np.gradient(left_acceleration, times, axis=0, edge_order=2)
    right_jerk = np.gradient(right_acceleration, times, axis=0, edge_order=2)
    left_gripper_velocity = np.gradient(left_gripper, times, edge_order=2)
    right_gripper_velocity = np.gradient(right_gripper, times, edge_order=2)
    end_effector, tcp_site_names = forward_kinematics(qpos, config_path)
    for side in ("left", "right"):
        end_effector[f"{side}_linear_velocity_m_s"] = np.gradient(
            end_effector[f"{side}_position_m"], times, axis=0, edge_order=2
        )

    arm_speed_max = float(max(np.abs(left_velocity).max(), np.abs(right_velocity).max()))
    arm_acceleration_max = float(max(np.abs(left_acceleration).max(), np.abs(right_acceleration).max()))
    arm_jerk_max = float(max(np.abs(left_jerk).max(), np.abs(right_jerk).max()))
    endpoint_speed_max = float(
        max(
            np.abs(left_velocity[[0, -1]]).max(),
            np.abs(right_velocity[[0, -1]]).max(),
        )
    )
    speed_limit = float(nested(summary, "time_scaling", "limits", "joint_speed_rad_s", default=1.2))
    acceleration_limit = float(nested(summary, "time_scaling", "limits", "joint_acceleration_rad_s2", default=3.0))
    endpoint_limit = float(nested(summary, "safety_audit", "thresholds", "endpoint_joint_speed_rad_s", default=0.02))
    acceleration_tolerance = max(1e-3, acceleration_limit * 1e-3)
    gripper_min = float(min(left_gripper.min(), right_gripper.min()))
    gripper_max = float(max(left_gripper.max(), right_gripper.max()))
    fk_finite = bool(all(np.isfinite(values).all() for values in end_effector.values()))
    target_position_error_max = None
    if set(target_positions) == {"left", "right"}:
        target_position_error_max = float(
            max(
                np.linalg.norm(end_effector[f"{side}_position_m"] - target_positions[side], axis=1).max()
                for side in ("left", "right")
            )
        )

    check_specs = (
        ("shape_14d", positions.shape == (len(times), 14), list(positions.shape), [len(times), 14]),
        ("finite_values", finite, finite, True),
        ("strictly_increasing_time", strictly_increasing, float(dt.min()), "> 0 s"),
        ("mink_verdict", str(summary.get("verdict", "")).upper() == "PASS", summary.get("verdict"), "PASS"),
        ("mink_failed_frames_empty", not summary.get("mink_failed_frames"), len(summary.get("mink_failed_frames") or []), 0),
        ("final_failed_frames_empty", not summary.get("final_failed_frames"), len(summary.get("final_failed_frames") or []), 0),
        ("safety_audit", str(nested(summary, "safety_audit", "verdict", default="")).upper() == "PASS", nested(summary, "safety_audit", "verdict"), "PASS"),
        ("arm_speed_limit", arm_speed_max <= speed_limit * 1.001, arm_speed_max, speed_limit),
        ("arm_acceleration_limit", arm_acceleration_max <= acceleration_limit + acceleration_tolerance, arm_acceleration_max, acceleration_limit),
        ("endpoint_arm_speed", endpoint_speed_max <= endpoint_limit * 1.001, endpoint_speed_max, endpoint_limit),
        ("gripper_normalized_range", gripper_min >= -1e-9 and gripper_max <= 1.0 + 1e-9, [gripper_min, gripper_max], [0.0, 1.0]),
        ("end_effector_fk_finite", fk_finite, fk_finite, True),
    )
    checks = [
        {"name": name, "passed": bool(passed), "value": value, "limit_or_expected": limit}
        for name, passed, value, limit in check_specs
    ]
    offline_preflight_pass = all(item["passed"] for item in checks)

    for path in (output_npz, output_json, output_ee_npz, output_ee_json, preflight_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
        times_sec=times,
        positions_14d=positions,
        left_arm_positions_rad=left_arm,
        left_gripper_normalized=left_gripper,
        right_arm_positions_rad=right_arm,
        right_gripper_normalized=right_gripper,
        left_arm_velocities_rad_s=left_velocity,
        right_arm_velocities_rad_s=right_velocity,
        left_arm_accelerations_rad_s2=left_acceleration,
        right_arm_accelerations_rad_s2=right_acceleration,
        left_arm_jerks_rad_s3=left_jerk,
        right_arm_jerks_rad_s3=right_jerk,
        left_gripper_velocity_normalized_s=left_gripper_velocity,
        right_gripper_velocity_normalized_s=right_gripper_velocity,
        joint_names_14d=np.asarray(JOINT_NAMES_14D),
        units_14d=np.asarray(UNITS_14D),
        source_qpos_indices_12d=np.concatenate((LEFT_ARM_INDICES, RIGHT_ARM_INDICES)),
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    trajectory_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "type": "controller_neutral_offline_dual_arm_trajectory",
        "source": {"mink_npz": str(source_npz), "mink_summary": str(source_summary_path)},
        "layout": "left_arm_6_rad + left_gripper_normalized + right_arm_6_rad + right_gripper_normalized",
        "joint_names_14d": list(JOINT_NAMES_14D),
        "units_14d": list(UNITS_14D),
        "gripper_convention": {
            "range": [0.0, 1.0],
            "meaning": "0=fully open, 1=fully closed; finite-speed position copied from Mink gripper_position_command",
            "hardware_mapping_required": True,
        },
        "frame_count": len(times),
        "duration_sec": float(times[-1] - times[0]),
        "offline_preflight_pass": offline_preflight_pass,
        "hardware_execution_authorized": False,
        "points": [
            {"time_sec": float(time), "positions_14d": row.tolist()}
            for time, row in zip(times, positions)
        ],
    }
    output_json.write_text(json.dumps(trajectory_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    ee_npz_payload: dict[str, Any] = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "times_sec": times,
        "coordinate_frame": np.asarray("mujoco_world"),
        "quaternion_order": np.asarray("xyzw"),
        "left_tcp_site_name": np.asarray(tcp_site_names["left"]),
        "right_tcp_site_name": np.asarray(tcp_site_names["right"]),
        **end_effector,
    }
    for side, values in target_positions.items():
        ee_npz_payload[f"{side}_ik_target_position_m"] = values
    for side, values in target_rotations.items():
        ee_npz_payload[f"{side}_ik_target_rotation_matrix"] = values
    np.savez_compressed(output_ee_npz, **ee_npz_payload)
    ee_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "type": "dual_arm_end_effector_fk_trajectory",
        "source": {
            "joint_trajectory": str(output_npz),
            "mink_npz": str(source_npz),
            "simulation_config": str(config_path),
        },
        "coordinate_frame": "mujoco_world",
        "pose_convention": {
            "position": "meters",
            "quaternion_order": "xyzw",
            "meaning": "actual TCP pose from forward kinematics of final Mink qpos",
        },
        "tcp_site_names": tcp_site_names,
        "frame_count": len(times),
        "duration_sec": float(times[-1] - times[0]),
        "offline_preflight_pass": offline_preflight_pass,
        "hardware_execution_authorized": False,
        "points": [
            {
                "time_sec": float(times[frame]),
                "left": {
                    "position_m": end_effector["left_position_m"][frame].tolist(),
                    "quaternion_xyzw": end_effector["left_quaternion_xyzw"][frame].tolist(),
                },
                "right": {
                    "position_m": end_effector["right_position_m"][frame].tolist(),
                    "quaternion_xyzw": end_effector["right_quaternion_xyzw"][frame].tolist(),
                },
            }
            for frame in range(len(times))
        ],
    }
    output_ee_json.write_text(json.dumps(ee_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    preflight_payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "verdict": "PASS_OFFLINE_EXPORT" if offline_preflight_pass else "FAIL_OFFLINE_EXPORT",
        "offline_preflight_pass": offline_preflight_pass,
        "hardware_execution_authorized": False,
        "safe_to_send": False,
        "checks": checks,
        "metrics": {
            "frame_count": len(times),
            "duration_sec": float(times[-1] - times[0]),
            "sample_period_sec": {"min": float(dt.min()), "median": float(np.median(dt)), "max": float(dt.max())},
            "arm_speed_max_rad_s": arm_speed_max,
            "arm_acceleration_max_rad_s2": arm_acceleration_max,
            "arm_jerk_max_rad_s3": arm_jerk_max,
            "endpoint_arm_speed_max_rad_s": endpoint_speed_max,
            "gripper_normalized_min": gripper_min,
            "gripper_normalized_max": gripper_max,
            "minimum_simulation_clearance_m": summary.get("minimum_clearance_m"),
            "fk_to_ik_target_position_error_max_m": target_position_error_max,
        },
        "artifacts": {
            "trajectory_npz": {"path": str(output_npz), "sha256": sha256(output_npz)},
            "trajectory_json": {"path": str(output_json), "sha256": sha256(output_json)},
            "end_effector_trajectory_npz": {"path": str(output_ee_npz), "sha256": sha256(output_ee_npz)},
            "end_effector_trajectory_json": {"path": str(output_ee_json), "sha256": sha256(output_ee_json)},
            "source_mink_npz": {"path": str(source_npz), "sha256": sha256(source_npz)},
            "source_mink_summary": {"path": str(source_summary_path), "sha256": sha256(source_summary_path)},
        },
        "hardware_blockers": list(HARDWARE_BLOCKERS),
        "notice": "Offline validation only. Resolve every hardware blocker and use a robot-specific bridge plus runtime safety supervision before physical execution.",
    }
    preflight_output.write_text(json.dumps(preflight_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "trajectory_npz": str(output_npz),
        "trajectory_json": str(output_json),
        "end_effector_trajectory_npz": str(output_ee_npz),
        "end_effector_trajectory_json": str(output_ee_json),
        "preflight": str(preflight_output),
        "shape": list(positions.shape),
        "offline_preflight_pass": offline_preflight_pass,
        "hardware_execution_authorized": False,
    }, ensure_ascii=False))
    if args.fail_on_preflight and not offline_preflight_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
