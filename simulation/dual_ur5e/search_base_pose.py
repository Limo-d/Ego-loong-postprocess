#!/usr/bin/env python3
"""Search a manufacturable dual-UR5e base layout for fixed robot task paths.

The search follows the keyframe-screening idea used by base-pose search papers,
but adapts it to this project in three important ways:

1. Targets come from a previously generated replay NPZ and stay fixed while the
   robot bases move.  Re-anchoring every candidate to its own home TCP would
   cancel base translation and make the search meaningless.
2. Both UR5e bases share one rigid body, so candidates are symmetric paired
   layouts rather than two mechanically incompatible independent placements.
3. Keyframe IK feasibility is only a screen.  Top candidates are checked on a
   denser full path using the same collision, joint-margin and singularity
   audit as replay_trajectory.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

import replay_trajectory as replay


SIDES = replay.SIDES


@dataclass
class TaskPath:
    path: Path
    positions: dict[str, np.ndarray]
    rotations: dict[str, np.ndarray]
    times: np.ndarray
    keyframes: np.ndarray


@dataclass(frozen=True)
class LayoutCandidate:
    index: int
    spacing_m: float
    forward_offset_m: float
    height_offset_m: float
    mount_twist_deg: float


def parse_number_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    result = np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )
    return result / max(float(np.linalg.norm(result)), 1e-12)


def base_quaternion(config: dict[str, Any], side: str) -> np.ndarray:
    configured = (config.get("base_quaternion_wxyz") or {}).get(side)
    if configured is not None:
        value = np.asarray(configured, dtype=np.float64)
        return value / max(float(np.linalg.norm(value)), 1e-12)
    return replay.quaternion_from_rpy_deg(
        (config.get("base_rpy_deg") or {}).get(side, [0.0, 0.0, 0.0])
    )


def candidate_config(
    base: dict[str, Any], candidate: LayoutCandidate
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    original = base["base_positions_m"]
    center_y = 0.5 * (float(original["left"][1]) + float(original["right"][1]))
    center_z = 0.5 * (float(original["left"][2]) + float(original["right"][2]))
    y = center_y + candidate.forward_offset_m
    z = center_z + candidate.height_offset_m
    half_spacing = 0.5 * candidate.spacing_m
    config["base_positions_m"] = {
        "left": [-half_spacing, y, z],
        "right": [half_spacing, y, z],
    }

    # Twist is around each base's local mounting normal.  Mirrored signs retain
    # a symmetric arm layout on the shared shoulder beam.
    quaternions: dict[str, list[float]] = {}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        angle = math.radians(sign * candidate.mount_twist_deg) * 0.5
        local_twist = np.asarray([math.cos(angle), 0.0, 0.0, math.sin(angle)])
        quaternions[side] = quaternion_multiply(
            base_quaternion(base, side), local_twist
        ).tolist()
    config["base_quaternion_wxyz"] = quaternions

    # Keep the visual/collision body mechanically consistent with the shared
    # symmetric candidate rather than leaving the bases floating in space.
    torso = config.get("torso_geometry") or {}
    if bool(torso.get("enabled", False)):
        column_setback = float(torso.get("column_setback_m", 0.0))
        column_y = y - column_setback
        torso["beam_position_m"] = [0.0, y, z]
        torso["beam_half_size_m"] = [half_spacing, 0.0745, 0.0745]
        torso["beam_cylinder_size_m"] = [0.0745, half_spacing]
        torso["shoulder_position_m"] = [0.0, y, z]
        torso["shoulder_size_m"] = [0.0745, half_spacing]
        torso["column_position_m"] = [0.0, column_y, 0.5 * z]
        torso["column_half_size_m"] = [0.0745, 0.0745, 0.5 * z]
        pedestal_top = 0.14
        mast_top = max(pedestal_top + 0.02, z - 0.0745)
        torso["mast_position_m"] = [0.0, column_y, 0.5 * (pedestal_top + mast_top)]
        torso["mast_half_size_m"] = [
            0.0745,
            0.0745,
            0.5 * (mast_top - pedestal_top),
        ]
        pedestal_position = list(torso.get("pedestal_position_m", [0.0, 0.0, 0.07]))
        pedestal_position[1] = column_y
        torso["pedestal_position_m"] = pedestal_position
        config["torso_geometry"] = torso
    return config


def choose_array(payload: Any, keys: tuple[str, ...]) -> np.ndarray:
    for key in keys:
        if key in payload:
            return np.asarray(payload[key])
    raise KeyError(f"none of the required arrays exist: {keys}")


def keyframe_features(
    positions: dict[str, np.ndarray], rotations: dict[str, np.ndarray]
) -> np.ndarray:
    features: list[np.ndarray] = []
    for side in SIDES:
        relative_position = positions[side] - positions[side][0]
        relative_rotation = np.asarray(
            [
                replay.rotation_error_vector(rotation, rotations[side][0])
                for rotation in rotations[side]
            ]
        )
        features.extend([relative_position, 0.10 * relative_rotation])
    output = np.concatenate(features, axis=1)
    scale = np.ptp(output, axis=0)
    scale[scale < 1e-6] = 1.0
    return output / scale


def select_keyframes(
    positions: dict[str, np.ndarray],
    rotations: dict[str, np.ndarray],
    count: int,
) -> np.ndarray:
    frame_count = len(positions["left"])
    required = {0, frame_count - 1}
    for side in SIDES:
        for axis in range(3):
            required.add(int(np.argmin(positions[side][:, axis])))
            required.add(int(np.argmax(positions[side][:, axis])))
        angles = np.asarray(
            [
                np.linalg.norm(
                    replay.rotation_error_vector(rotation, rotations[side][0])
                )
                for rotation in rotations[side]
            ]
        )
        required.add(int(np.argmax(angles)))
        if frame_count > 1:
            motion = np.linalg.norm(np.diff(positions[side], axis=0), axis=1)
            required.add(int(np.argmax(motion)) + 1)
    features = keyframe_features(positions, rotations)
    selected = sorted(required)
    limit = min(max(count, len(selected)), frame_count)
    while len(selected) < limit:
        distance = np.full(frame_count, np.inf)
        for index in selected:
            distance = np.minimum(
                distance, np.linalg.norm(features - features[index], axis=1)
            )
        distance[selected] = -1.0
        selected.append(int(np.argmax(distance)))
    return np.asarray(sorted(selected), dtype=int)


def load_task(path: Path, keyframe_count: int) -> TaskPath:
    payload = np.load(path)
    positions = {
        side: choose_array(
            payload,
            (
                f"{side}_target_pre_retime_m",
                f"{side}_target_m",
                f"{side}_target_command_source_m",
            ),
        ).astype(np.float64)
        for side in SIDES
    }
    rotations = {
        side: choose_array(
            payload,
            (
                f"{side}_target_rotation_pre_retime_matrix",
                f"{side}_target_rotation_matrix",
            ),
        ).astype(np.float64)
        for side in SIDES
    }
    length = min(
        *(len(values) for values in positions.values()),
        *(len(values) for values in rotations.values()),
    )
    positions = {side: values[:length] for side, values in positions.items()}
    rotations = {side: values[:length] for side, values in rotations.items()}
    if "pre_retime_times_sec" in payload and len(payload["pre_retime_times_sec"]) >= length:
        times = np.asarray(payload["pre_retime_times_sec"][:length], dtype=np.float64)
    elif "times_sec" in payload and len(payload["times_sec"]) >= length:
        times = np.asarray(payload["times_sec"][:length], dtype=np.float64)
    else:
        times = np.arange(length, dtype=np.float64) / 60.0
    keyframes = select_keyframes(positions, rotations, keyframe_count)
    return TaskPath(path.resolve(), positions, rotations, times, keyframes)


def joint_metrics(
    model: mujoco.MjModel, data: mujoco.MjData, side: str
) -> tuple[float, float]:
    qpos_ids, dof_ids = replay.joint_addresses(model, side)
    site_id = replay.end_effector_site_id(model, side)
    jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(
        model, data, jacobian_position, jacobian_rotation, site_id
    )
    orientation_weight = 0.25
    singular_values = np.linalg.svd(
        np.vstack(
            [
                jacobian_position[:, dof_ids],
                orientation_weight * jacobian_rotation[:, dof_ids],
            ]
        ),
        compute_uv=False,
    )
    condition = float(singular_values[0] / max(singular_values[-1], 1e-12))
    margins: list[float] = []
    for joint_index, suffix in enumerate(replay.JOINT_NAMES):
        joint = model.joint(f"{side}_{suffix}")
        if int(joint.limited[0]):
            value = float(data.qpos[qpos_ids[joint_index]])
            margins.append(float(min(value - joint.range[0], joint.range[1] - value)))
    return condition, min(margins) if margins else float("inf")


def solve_samples(
    config: dict[str, Any],
    tasks: list[TaskPath],
    frame_sets: list[np.ndarray],
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    model, data = replay.build_model(config)
    seed = data.qpos.copy()
    samples: list[np.ndarray] = []
    sample_times: list[float] = []
    feasible = {side: 0 for side in SIDES}
    total = {side: 0 for side in SIDES}
    position_errors = {side: [] for side in SIDES}
    orientation_errors = {side: [] for side in SIDES}
    conditions = {side: [] for side in SIDES}
    margins = {side: [] for side in SIDES}
    horizontal_angles = {side: [] for side in SIDES}
    reaches = {side: [] for side in SIDES}
    position_limit = float(settings.get("position_tolerance_m", 0.005))
    orientation_limit = math.radians(
        float(settings.get("orientation_tolerance_deg", 5.0))
    )
    condition_limit = float(settings.get("max_jacobian_condition", 30.0))
    margin_limit = math.radians(float(settings.get("min_joint_margin_deg", 10.0)))
    max_reach = float(settings.get("kinematic_reach_m", 0.95))
    min_reach = float(settings.get("minimum_base_distance_m", 0.08))
    max_ratio = float(settings.get("maximum_reach_ratio", 0.90))
    sample_index = 0
    for task, indices in zip(tasks, frame_sets):
        for frame in indices:
            data.qpos[:] = seed
            for side in SIDES:
                position_error, orientation_error = replay.solve_pose_ik(
                    model,
                    data,
                    side,
                    task.positions[side][frame],
                    task.rotations[side][frame],
                    replay.home_q_for_side(config, side),
                    config,
                )
                mujoco.mj_forward(model, data)
                condition, margin = joint_metrics(model, data, side)
                site_id = replay.end_effector_site_id(model, side)
                tool_axis = data.site_xmat[site_id].reshape(3, 3)[:, 2]
                horizontal_angle = math.degrees(
                    math.asin(float(np.clip(abs(tool_axis[2]), 0.0, 1.0)))
                )
                base_position = np.asarray(config["base_positions_m"][side])
                reach = float(
                    np.linalg.norm(task.positions[side][frame] - base_position)
                )
                total[side] += 1
                is_feasible = (
                    position_error <= position_limit
                    and orientation_error <= orientation_limit
                    and condition <= condition_limit
                    and margin >= margin_limit
                    and reach >= min_reach
                    and reach <= max_ratio * max_reach
                )
                feasible[side] += int(is_feasible)
                position_errors[side].append(position_error)
                orientation_errors[side].append(orientation_error)
                conditions[side].append(condition)
                margins[side].append(margin)
                reaches[side].append(reach / max_reach)
                if sample_index == 0:
                    horizontal_angles[side].append(horizontal_angle)
            samples.append(data.qpos.copy())
            # Large spacing suppresses endpoint-speed artifacts in the static
            # keyframe audit; dynamics are checked later by normal replay.
            sample_times.append(float(sample_index) * 100.0)
            sample_index += 1
    qpos = np.asarray(samples)
    times = np.asarray(sample_times)
    audit = replay.execution_safety_audit(model, qpos, times, config, {})
    metrics = {
        "feasibility_rate": {
            side: feasible[side] / max(total[side], 1) for side in SIDES
        },
        "position_error_max_m": {
            side: float(max(position_errors[side], default=0.0)) for side in SIDES
        },
        "orientation_error_max_deg": {
            side: float(math.degrees(max(orientation_errors[side], default=0.0)))
            for side in SIDES
        },
        "jacobian_condition_max": {
            side: float(max(conditions[side], default=0.0)) for side in SIDES
        },
        "joint_margin_min_deg": {
            side: float(math.degrees(min(margins[side], default=float("inf"))))
            for side in SIDES
        },
        "initial_gripper_horizontal_angle_deg": {
            side: float(horizontal_angles[side][0])
            if horizontal_angles[side]
            else 0.0
            for side in SIDES
        },
        "mean_reach_ratio": {
            side: float(np.mean(reaches[side])) if reaches[side] else 0.0
            for side in SIDES
        },
        "safety_audit": audit,
    }
    return qpos, times, metrics


def score_metrics(metrics: dict[str, Any], settings: dict[str, Any]) -> float:
    feasibility = float(np.mean(list(metrics["feasibility_rate"].values())))
    reach_ratio = float(np.mean(list(metrics["mean_reach_ratio"].values())))
    target_ratio = float(settings.get("target_reach_ratio", 0.65))
    reach_weight = float(settings.get("reach_penalty_weight", 5.0))
    horizontal_deadband = float(settings.get("horizontal_deadband_deg", 5.0))
    horizontal_weight = float(settings.get("horizontal_penalty_weight", 0.20))
    horizontal = max(metrics["initial_gripper_horizontal_angle_deg"].values())
    horizontal_penalty = horizontal_weight * max(
        0.0, (horizontal - horizontal_deadband) / 45.0
    )
    violations = set(metrics["safety_audit"].get("violations", []))
    relevant = {
        "self_clearance",
        "mounting_clearance",
        "interarm_clearance",
        "structure_clearance",
        "environment_clearance",
        "joint_limit_margin",
        "jacobian_condition",
    }
    safety_penalty = float(settings.get("safety_violation_penalty", 2.0)) * len(
        violations & relevant
    )
    return (
        feasibility
        - reach_weight * abs(reach_ratio - target_ratio)
        - horizontal_penalty
        - safety_penalty
    )


def candidate_payload(
    candidate: LayoutCandidate,
    metrics: dict[str, Any],
    score: float,
    stage: str,
) -> dict[str, Any]:
    return {
        "candidate_index": candidate.index,
        "stage": stage,
        "spacing_m": candidate.spacing_m,
        "forward_offset_m": candidate.forward_offset_m,
        "height_offset_m": candidate.height_offset_m,
        "mount_twist_deg": candidate.mount_twist_deg,
        "score": float(score),
        "metrics": metrics,
    }


def solve_initial_home(
    config: dict[str, Any], task: TaskPath
) -> dict[str, list[float]]:
    """Return the candidate-base IK solution at the fixed task's first frame."""
    model, data = replay.build_model(config)
    for side in SIDES:
        replay.solve_pose_ik(
            model,
            data,
            side,
            task.positions[side][0],
            task.rotations[side][0],
            replay.home_q_for_side(config, side),
            config,
        )
    mujoco.mj_forward(model, data)
    return {
        side: data.qpos[replay.joint_addresses(model, side)[0]].tolist()
        for side in SIDES
    }


def recommended_override(
    source_config: Path,
    config: dict[str, Any],
    candidate: LayoutCandidate,
    task: TaskPath,
) -> dict[str, Any]:
    torso = config.get("torso_geometry") or {}
    return {
        "extends": str(source_config),
        "layout_name": "base_pose_search_recommended_symmetric_layout",
        "layout_status": (
            "kinematically searched prototype; requires table, payload and hardware validation"
        ),
        "base_positions_m": config["base_positions_m"],
        "base_quaternion_wxyz": config.get("base_quaternion_wxyz"),
        "home_q_rad_by_side": config["home_q_rad_by_side"],
        "task_anchor_positions_m": {
            side: task.positions[side][0].tolist() for side in SIDES
        },
        "task_anchor_rotation_matrix": {
            side: task.rotations[side][0].tolist() for side in SIDES
        },
        "torso_geometry": {
            key: torso[key]
            for key in (
                "column_position_m",
                "column_half_size_m",
                "column_setback_m",
                "beam_position_m",
                "beam_half_size_m",
                "beam_collision_shape",
                "beam_cylinder_size_m",
                "pedestal_position_m",
                "mast_position_m",
                "mast_half_size_m",
                "shoulder_position_m",
                "shoulder_visual_shape",
                "shoulder_size_m",
            )
            if key in torso
        },
        "base_pose_search_selection": {
            "spacing_m": candidate.spacing_m,
            "forward_offset_m": candidate.forward_offset_m,
            "height_offset_m": candidate.height_offset_m,
            "mount_twist_deg": candidate.mount_twist_deg,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay_npz", action="append", required=True)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config_shoulder_relative_human_orientation.json")),
    )
    parser.add_argument("--output_dir", default=str(Path(__file__).with_name("base_pose_search_outputs")))
    parser.add_argument("--name", default="base_pose_search")
    parser.add_argument("--keyframes", type=int, default=20)
    parser.add_argument("--full_stride", type=int, default=3)
    parser.add_argument("--top_k_full", type=int, default=5)
    parser.add_argument("--max_candidates", type=int, default=0)
    parser.add_argument("--spacing_m", default="0.48,0.50,0.52,0.54,0.56")
    parser.add_argument("--forward_offset_m", default="-0.05,0.0,0.05")
    parser.add_argument("--height_offset_m", default="-0.05,0.0,0.05")
    parser.add_argument("--mount_twist_deg", default="-10,0,10")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    base_config = replay.load_config(config_path)
    settings = {
        "position_tolerance_m": 0.005,
        "orientation_tolerance_deg": 5.0,
        "max_jacobian_condition": float(
            (base_config.get("safety_audit") or {}).get("max_jacobian_condition", 30.0)
        ),
        "min_joint_margin_deg": float(
            (base_config.get("safety_audit") or {}).get("min_joint_limit_margin_deg", 10.0)
        ),
        "kinematic_reach_m": 0.95,
        "minimum_base_distance_m": 0.08,
        "maximum_reach_ratio": 0.90,
        "target_reach_ratio": 0.65,
        "reach_penalty_weight": 5.0,
        "horizontal_deadband_deg": 5.0,
        "horizontal_penalty_weight": 0.20,
        "safety_violation_penalty": 2.0,
    }
    settings.update(base_config.get("base_pose_search") or {})
    tasks = [
        load_task(Path(value).expanduser().resolve(), args.keyframes)
        for value in args.replay_npz
    ]
    anchor_consistency: list[dict[str, Any]] = []
    position_anchor_limit = float(
        settings.get("batch_anchor_position_tolerance_m", 0.01)
    )
    rotation_anchor_limit = math.radians(
        float(settings.get("batch_anchor_orientation_tolerance_deg", 5.0))
    )
    reference_task = tasks[0]
    for task in tasks:
        item: dict[str, Any] = {"path": str(task.path), "sides": {}}
        for side in SIDES:
            position_delta = float(
                np.linalg.norm(
                    task.positions[side][0] - reference_task.positions[side][0]
                )
            )
            rotation_delta = float(
                np.linalg.norm(
                    replay.rotation_error_vector(
                        task.rotations[side][0], reference_task.rotations[side][0]
                    )
                )
            )
            item["sides"][side] = {
                "position_delta_m": position_delta,
                "orientation_delta_deg": math.degrees(rotation_delta),
            }
            if position_delta > position_anchor_limit or rotation_delta > rotation_anchor_limit:
                raise ValueError(
                    f"task anchor mismatch for {task.path} {side}: "
                    f"position={position_delta:.4f} m, "
                    f"orientation={math.degrees(rotation_delta):.2f} deg; "
                    "generate all replay NPZ files with the same reference config/task frame"
                )
        anchor_consistency.append(item)
    spacing_values = parse_number_list(args.spacing_m)
    forward_values = parse_number_list(args.forward_offset_m)
    height_values = parse_number_list(args.height_offset_m)
    twist_values = parse_number_list(args.mount_twist_deg)
    seed_spacing = abs(
        float(base_config["base_positions_m"]["right"][0])
        - float(base_config["base_positions_m"]["left"][0])
    )
    raw_candidates: list[LayoutCandidate] = []
    index = 0
    for spacing in spacing_values:
        for forward in forward_values:
            for height in height_values:
                for twist in twist_values:
                    raw_candidates.append(
                        LayoutCandidate(index, spacing, forward, height, twist)
                    )
                    index += 1
    raw_candidates.sort(
        key=lambda value: (
            abs(value.spacing_m - seed_spacing) / 0.02
            + abs(value.forward_offset_m) / 0.05
            + abs(value.height_offset_m) / 0.05
            + abs(value.mount_twist_deg) / 10.0,
            value.index,
        )
    )
    if args.max_candidates > 0:
        raw_candidates = raw_candidates[: args.max_candidates]

    screen_results: list[dict[str, Any]] = []
    keyframe_sets = [task.keyframes for task in tasks]
    for order, candidate in enumerate(raw_candidates, start=1):
        config = candidate_config(base_config, candidate)
        _, _, metrics = solve_samples(config, tasks, keyframe_sets, settings)
        score = score_metrics(metrics, settings)
        screen_results.append(candidate_payload(candidate, metrics, score, "keyframe"))
        print(
            f"[{order}/{len(raw_candidates)}] spacing={candidate.spacing_m:.3f} "
            f"forward={candidate.forward_offset_m:+.3f} height={candidate.height_offset_m:+.3f} "
            f"twist={candidate.mount_twist_deg:+.1f} score={score:.4f} "
            f"FR={np.mean(list(metrics['feasibility_rate'].values())):.3f} "
            f"audit={metrics['safety_audit']['verdict']}"
        )
    screen_results.sort(key=lambda value: value["score"], reverse=True)
    candidate_lookup = {candidate.index: candidate for candidate in raw_candidates}

    full_results: list[dict[str, Any]] = []
    full_sets = [
        np.arange(0, len(task.times), max(1, args.full_stride), dtype=int)
        for task in tasks
    ]
    for screened in screen_results[: max(1, args.top_k_full)]:
        candidate = candidate_lookup[int(screened["candidate_index"])]
        config = candidate_config(base_config, candidate)
        _, _, metrics = solve_samples(config, tasks, full_sets, settings)
        score = score_metrics(metrics, settings)
        full_results.append(candidate_payload(candidate, metrics, score, "full_path"))
    full_results.sort(
        key=lambda value: (
            value["metrics"]["safety_audit"]["verdict"] == "PASS",
            value["score"],
        ),
        reverse=True,
    )
    best = full_results[0]
    best_candidate = candidate_lookup[int(best["candidate_index"])]
    best_config = candidate_config(base_config, best_candidate)
    best_config["home_q_rad_by_side"] = solve_initial_home(best_config, tasks[0])

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.name}_report.json"
    config_output_path = output_dir / f"{args.name}_recommended_config.json"
    report = {
        "method": (
            "paper-inspired representative-keyframe base-pose search with fixed task targets, "
            "paired symmetric shared-body candidates, and dense full-path validation"
        ),
        "source_config": str(config_path),
        "tasks": [
            {
                "path": str(task.path),
                "frames": len(task.times),
                "keyframes": task.keyframes.tolist(),
            }
            for task in tasks
        ],
        "settings": settings,
        "batch_anchor_consistency": anchor_consistency,
        "candidate_count": len(raw_candidates),
        "screen_results": screen_results,
        "full_path_results": full_results,
        "recommended": best,
        "recommended_home_q_rad_by_side": best_config["home_q_rad_by_side"],
        "recommended_config": str(config_output_path),
        "selection_status": (
            "PASS"
            if best["metrics"]["safety_audit"]["verdict"] == "PASS"
            else "NO_FULL_PATH_PASS"
        ),
        "limitations": [
            "Targets must share one robot task frame across episodes.",
            "No table, payload, object contact or cable model is included yet.",
            "The recommended pose still requires hardware TCP/extrinsic validation.",
            "Normal replay must be run with the recommended config for dynamic retiming audit.",
        ],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    config_output_path.write_text(
        json.dumps(
            recommended_override(config_path, best_config, best_candidate, tasks[0]),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(report_path), "recommended_config": str(config_output_path), "recommended": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
