#!/usr/bin/env python3
"""Search grounded LingLong-H base placement, waist extension, and torso lean."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
REPLAY_PATH = HERE / "replay_trajectory.py"
SPEC = importlib.util.spec_from_file_location("linglong_h_replay_search", REPLAY_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load LingLong-H replay module: {REPLAY_PATH}")
linglong = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(linglong)
replay = linglong.replay
mujoco = linglong.mujoco

replay.JOINT_NAMES = linglong.ARM_JOINT_NAMES
replay.home_q_for_side = linglong.home_q_for_side
replay.end_effector_site_id = linglong.end_effector_site_id


def parse_numbers(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def choose_keyframes(left: np.ndarray, right: np.ndarray, count: int) -> np.ndarray:
    features = np.concatenate([left, right], axis=1)
    selected: set[int] = {0, len(features) - 1}
    for axis in range(features.shape[1]):
        selected.add(int(np.argmin(features[:, axis])))
        selected.add(int(np.argmax(features[:, axis])))
    scale = np.ptp(features, axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (features - np.mean(features, axis=0)) / scale
    while len(selected) < min(count, len(features)):
        indexes = np.asarray(sorted(selected), dtype=int)
        distances = np.linalg.norm(
            normalized[:, None, :] - normalized[indexes][None, :, :], axis=2
        )
        nearest = np.min(distances, axis=1)
        nearest[indexes] = -1.0
        selected.add(int(np.argmax(nearest)))
    return np.asarray(sorted(selected), dtype=int)


def grounded_base_z(config: dict[str, Any]) -> tuple[float, float]:
    model, data = linglong.build_model(config)
    floor = int(model.geom("floor").id)
    distances: list[float] = []
    for geom_id in range(model.ngeom):
        if model.body(int(model.geom_bodyid[geom_id])).name != "base_link":
            continue
        distances.append(
            float(mujoco.mj_geomDistance(model, data, floor, geom_id, 2.0, None))
        )
    if not distances:
        raise RuntimeError("LingLong-H base_link has no geometry")
    current_z = float(config.get("base_position_m", [0.0, 0.0, 0.0])[2])
    clearance = min(distances)
    return current_z - clearance, clearance


def waist_q(extension_rad: float, lean_deg: float) -> list[float]:
    lean_rad = float(np.deg2rad(lean_deg))
    return [-extension_rad, 2.0 * extension_rad, -extension_rad + lean_rad, 0.0]


def candidate_config(
    base: dict[str, Any], grounded_z: float, forward_m: float, extension: float, lean: float
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    base_x = float(config.get("base_position_m", [0.0, 0.0, grounded_z])[0])
    position = [base_x, forward_m, grounded_z]
    config["base_position_m"] = position
    config["base_positions_m"] = {"left": position, "right": position}
    config["waist_home_q_rad"] = waist_q(extension, lean)
    return config


def environment_clearance(model: Any, data: Any) -> tuple[float, list[str]]:
    robot_geoms = [
        geom
        for geom in range(model.ngeom)
        if int(model.geom_group[geom]) == 3
        and model.geom(geom).name.startswith("linglong_collision_")
    ]
    obstacles = [
        geom
        for geom in range(model.ngeom)
        if model.geom(geom).name == "floor"
        or model.geom(geom).name == "table_top"
        or model.geom(geom).name.startswith("table_leg_")
    ]
    closest = 0.30
    closest_names: list[str] = []
    for obstacle in obstacles:
        obstacle_name = model.geom(obstacle).name
        for robot_geom in robot_geoms:
            body_name = model.body(int(model.geom_bodyid[robot_geom])).name
            if obstacle_name == "floor" and body_name == "base_link":
                continue
            distance = float(
                mujoco.mj_geomDistance(model, data, obstacle, robot_geom, 0.30, None)
            )
            if distance < closest:
                closest = distance
                closest_names = [obstacle_name, body_name]
    return closest, closest_names


def cross_arm_clearance(model: Any, data: Any) -> tuple[float, list[str]]:
    """Return minimum collision-geometry distance between the two arms/grippers."""
    by_side: dict[str, list[int]] = {"left": [], "right": []}
    for geom_id in range(model.ngeom):
        if not (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])):
            continue
        body_name = model.body(int(model.geom_bodyid[geom_id])).name
        for side in replay.SIDES:
            if body_name.startswith(f"{side}_"):
                by_side[side].append(geom_id)
                break
    closest = 0.30
    closest_names: list[str] = []
    for left_geom in by_side["left"]:
        for right_geom in by_side["right"]:
            distance = float(
                mujoco.mj_geomDistance(model, data, left_geom, right_geom, 0.30, None)
            )
            if distance < closest:
                closest = distance
                closest_names = [
                    model.body(int(model.geom_bodyid[left_geom])).name,
                    model.body(int(model.geom_bodyid[right_geom])).name,
                ]
    return closest, closest_names


def initial_posture_metrics(model: Any, data: Any, config: dict[str, Any]) -> dict[str, Any]:
    table_height = float((config.get("table_geometry") or {}).get("top_height_m", 0.75))
    table_id = int(model.geom("table_top").id)
    arm_angles: dict[str, float] = {}
    upper_arm_angles: dict[str, float] = {}
    forearm_angles: dict[str, float] = {}
    tcp_forward_angles: dict[str, float] = {}
    jaw_axis_angles: dict[str, float] = {}
    tcp_heights: dict[str, float] = {}
    table_clearances: dict[str, float] = {}
    tcp_positions: dict[str, list[float]] = {}
    tcp_forward_vectors: dict[str, np.ndarray] = {}
    jaw_axis_vectors: dict[str, np.ndarray] = {}
    camera_up_vectors: dict[str, np.ndarray] = {}
    camera_up_angles: dict[str, float] = {}
    for side in replay.SIDES:
        site_id = replay.end_effector_site_id(model, side)
        tcp = data.site_xpos[site_id].copy()
        shoulder = data.xpos[model.body(f"{side}_shoulder_yaw_link").id].copy()
        elbow = data.xpos[model.body(f"{side}_elbow_link").id].copy()
        wrist = data.xpos[model.body(f"{side}_wrist_yaw_link").id].copy()
        arm_vector = tcp - shoulder
        arm_angles[side] = float(
            np.rad2deg(
                np.arcsin(
                    np.clip(abs(arm_vector[2]) / max(np.linalg.norm(arm_vector), 1e-12), 0.0, 1.0)
                )
            )
        )
        upper_vector = elbow - shoulder
        forearm_vector = wrist - elbow
        upper_arm_angles[side] = float(
            np.rad2deg(np.arcsin(np.clip(abs(upper_vector[2]) / max(np.linalg.norm(upper_vector), 1e-12), 0.0, 1.0)))
        )
        forearm_angles[side] = float(
            np.rad2deg(np.arcsin(np.clip(abs(forearm_vector[2]) / max(np.linalg.norm(forearm_vector), 1e-12), 0.0, 1.0)))
        )
        site_rotation = data.site_xmat[site_id].reshape(3, 3)
        # OmniPicker's TCP offset is along local +Z and its two fingers are
        # separated along local +Y.  Both axes must lie in the table plane for
        # the complete gripper, rather than an arbitrary site axis, to be
        # horizontal.
        tcp_forward_axis = site_rotation[:, 2]
        jaw_axis = site_rotation[:, 1]
        wrist_rotation = data.xmat[
            model.body(f"{side}_wrist_yaw_link").id
        ].reshape(3, 3)
        camera_up_axis = wrist_rotation[:, 2]
        tcp_forward_vectors[side] = tcp_forward_axis.copy()
        jaw_axis_vectors[side] = jaw_axis.copy()
        camera_up_vectors[side] = camera_up_axis.copy()
        tcp_forward_angles[side] = float(
            np.rad2deg(np.arcsin(np.clip(abs(tcp_forward_axis[2]), 0.0, 1.0)))
        )
        jaw_axis_angles[side] = float(
            np.rad2deg(np.arcsin(np.clip(abs(jaw_axis[2]), 0.0, 1.0)))
        )
        camera_up_angles[side] = float(
            np.rad2deg(np.arccos(np.clip(camera_up_axis[2], -1.0, 1.0)))
        )
        tcp_heights[side] = float(tcp[2] - table_height)
        tcp_positions[side] = tcp.tolist()
        distances = [
            float(mujoco.mj_geomDistance(model, data, table_id, geom, 0.50, None))
            for geom in range(model.ngeom)
            if int(model.geom_group[geom]) == 3
            and model.body(int(model.geom_bodyid[geom])).name.startswith(f"{side}_")
        ]
        table_clearances[side] = min(distances) if distances else 0.50
    parallel_angles = {
        "tcp_forward": float(
            np.rad2deg(
                np.arccos(
                    np.clip(
                        np.dot(tcp_forward_vectors["left"], tcp_forward_vectors["right"]),
                        -1.0,
                        1.0,
                    )
                )
            )
        ),
        # Finger separation is a geometrically undirected line.  Camera roll
        # is checked separately through the directed wrist +Z axis, so an
        # upside-down camera cannot pass merely because the jaws are parallel.
        "jaw_axis": float(
            np.rad2deg(
                np.arccos(
                    np.clip(
                        abs(np.dot(jaw_axis_vectors["left"], jaw_axis_vectors["right"])),
                        -1.0,
                        1.0,
                    )
                )
            )
        ),
    }
    return {
        "arm_horizontal_angle_deg": arm_angles,
        "upper_arm_horizontal_angle_deg": upper_arm_angles,
        "forearm_horizontal_angle_deg": forearm_angles,
        "tcp_forward_axis_horizontal_angle_deg": tcp_forward_angles,
        "jaw_axis_horizontal_angle_deg": jaw_axis_angles,
        "camera_up_axis_error_deg": camera_up_angles,
        "tcp_height_above_table_m": tcp_heights,
        "arm_table_clearance_m": table_clearances,
        "tcp_positions_m": tcp_positions,
        "gripper_axis_parallel_angle_deg": parallel_angles,
    }


def configured_initial_tcp_rotation(
    config: dict[str, Any], side: str, forward_yaw_deg: float | None = None
) -> np.ndarray | None:
    setting = config.get("initial_parallel_gripper_orientation") or {}
    if not bool(setting.get("enabled", False)):
        return None
    if forward_yaw_deg is None:
        forward = np.asarray(
            setting.get("tcp_forward_world", [0.0, 1.0, 0.0]), dtype=np.float64
        )
    else:
        yaw = np.deg2rad(forward_yaw_deg)
        forward = np.asarray([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
    jaw_by_side = setting.get("jaw_axis_world_by_side") or {}
    if forward_yaw_deg is None:
        jaw = np.asarray(
            jaw_by_side.get(side, setting.get("jaw_axis_world", [1.0, 0.0, 0.0])),
            dtype=np.float64,
        )
    else:
        jaw = np.asarray([np.sin(yaw), -np.cos(yaw), 0.0], dtype=np.float64)
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    jaw -= float(np.dot(jaw, forward)) * forward
    jaw /= max(float(np.linalg.norm(jaw)), 1e-12)
    local_x = np.cross(jaw, forward)
    return np.column_stack([local_x, jaw, forward])


def solve_initial_tcp_horizontal_pose(
    model: Any,
    data: Any,
    side: str,
    target: np.ndarray,
    seed: np.ndarray,
    fixed_wrist_yaw: float | None = None,
    target_rotation: np.ndarray | None = None,
) -> None:
    """Solve the first frame with the OmniPicker parallel to the table."""
    qpos_ids, _ = replay.joint_addresses(model, side)
    lower = np.full(len(qpos_ids), -np.pi, dtype=np.float64)
    upper = np.full(len(qpos_ids), np.pi, dtype=np.float64)
    for index, suffix in enumerate(linglong.ARM_JOINT_NAMES):
        joint = model.joint(f"{side}_{suffix}")
        if int(joint.limited[0]):
            lower[index], upper[index] = np.asarray(joint.range, dtype=np.float64)
    active = np.ones(len(qpos_ids), dtype=bool)
    fixed = seed.copy()
    if fixed_wrist_yaw is not None:
        active[-1] = False
        fixed[-1] = float(fixed_wrist_yaw)
    initial_full = np.clip(data.qpos[qpos_ids], lower + 1e-7, upper - 1e-7)
    initial = initial_full[active]

    def residual(q_active: np.ndarray) -> np.ndarray:
        q = fixed.copy()
        q[active] = q_active
        data.qpos[qpos_ids] = q
        mujoco.mj_forward(model, data)
        tcp = data.site_xpos[replay.end_effector_site_id(model, side)]
        site_rotation = data.site_xmat[
            replay.end_effector_site_id(model, side)
        ].reshape(3, 3)
        tcp_forward_axis = site_rotation[:, 2]
        jaw_axis = site_rotation[:, 1]
        orientation_residual = (
            10.0 * replay.rotation_error_vector(target_rotation, site_rotation)
            if target_rotation is not None
            else np.asarray([10.0 * tcp_forward_axis[2], 10.0 * jaw_axis[2]])
        )
        return np.concatenate(
            [
                (tcp - target) / 0.002,
                orientation_residual,
                0.005 * (q - seed),
            ]
        )

    result = least_squares(
        residual,
        initial,
        bounds=(lower[active], upper[active]),
        max_nfev=500,
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
    )
    solved = fixed.copy()
    solved[active] = result.x
    data.qpos[qpos_ids] = solved
    mujoco.mj_forward(model, data)


def frame_metrics(model: Any, data: Any, config: dict[str, Any]) -> tuple[float, float]:
    minimum_margin = float("inf")
    maximum_condition = 0.0
    for side in replay.SIDES:
        qpos_ids, dof_ids = replay.joint_addresses(model, side)
        site_id = replay.end_effector_site_id(model, side)
        jacobian = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacobian, None, site_id)
        singular = np.linalg.svd(jacobian[:, dof_ids], compute_uv=False)
        maximum_condition = max(
            maximum_condition, float(singular[0] / max(singular[-1], 1e-12))
        )
        for index, suffix in enumerate(linglong.ARM_JOINT_NAMES):
            joint = model.joint(f"{side}_{suffix}")
            if int(joint.limited[0]):
                value = float(data.qpos[qpos_ids[index]])
                minimum_margin = min(
                    minimum_margin,
                    value - float(joint.range[0]),
                    float(joint.range[1]) - value,
                )
    return float(np.rad2deg(minimum_margin)), maximum_condition


def evaluate_candidate(
    config: dict[str, Any],
    left_targets: np.ndarray,
    right_targets: np.ndarray,
    keyframes: np.ndarray,
    position_tolerance_m: float,
    max_initial_tcp_angle_deg: float,
    minimum_initial_tcp_above_table_m: float,
    minimum_initial_arm_table_clearance_m: float,
    horizontal_deadband_deg: float,
    horizontal_penalty_weight: float,
    max_initial_parallel_angle_deg: float,
    initial_gripper_forward_yaw_deg: float | None,
) -> dict[str, Any]:
    model, data = linglong.build_model(config)
    homes = {side: linglong.home_q_for_side(config, side) for side in replay.SIDES}
    errors: list[float] = []
    clearances: list[float] = []
    cross_arm_clearances: list[float] = []
    minimum_margin_deg = float("inf")
    maximum_condition = 0.0
    closest_pair: list[str] = []
    closest_cross_arm_pair: list[str] = []
    first_home: dict[str, list[float]] | None = None
    initial_posture: dict[str, Any] | None = None
    initial_target_rotations = {
        side: configured_initial_tcp_rotation(
            config, side, initial_gripper_forward_yaw_deg
        )
        for side in replay.SIDES
    }
    for keyframe_index, frame in enumerate(keyframes):
        targets = {"left": left_targets[frame], "right": right_targets[frame]}
        for side in replay.SIDES:
            if keyframe_index == 0:
                fixed_wrist_yaw = (
                    config.get("fixed_wrist_yaw_q_rad_by_side") or {}
                ).get(side)
                solve_initial_tcp_horizontal_pose(
                    model,
                    data,
                    side,
                    targets[side],
                    homes[side],
                    fixed_wrist_yaw,
                    initial_target_rotations[side],
                )
                homes[side] = data.qpos[replay.joint_addresses(model, side)[0]].copy()
            else:
                replay.solve_pose_ik(
                    model, data, side, targets[side], None, homes[side], config
                )
        mujoco.mj_forward(model, data)
        for side in replay.SIDES:
            site_id = replay.end_effector_site_id(model, side)
            errors.append(float(np.linalg.norm(targets[side] - data.site_xpos[site_id])))
        clearance, names = environment_clearance(model, data)
        clearances.append(clearance)
        if clearance == min(clearances):
            closest_pair = names
        cross_clearance, cross_names = cross_arm_clearance(model, data)
        cross_arm_clearances.append(cross_clearance)
        if cross_clearance == min(cross_arm_clearances):
            closest_cross_arm_pair = cross_names
        margin_deg, condition = frame_metrics(model, data, config)
        minimum_margin_deg = min(minimum_margin_deg, margin_deg)
        maximum_condition = max(maximum_condition, condition)
        if keyframe_index == 0:
            first_home = {
                side: data.qpos[replay.joint_addresses(model, side)[0]].tolist()
                for side in replay.SIDES
            }
            initial_posture = initial_posture_metrics(model, data, config)
    errors_array = np.asarray(errors, dtype=np.float64)
    clearance_min = float(min(clearances))
    cross_arm_clearance_min = float(min(cross_arm_clearances))
    feasible_ratio = float(np.mean(errors_array <= position_tolerance_m))
    collision_free = clearance_min >= -1e-5 and cross_arm_clearance_min >= -1e-5
    assert initial_posture is not None
    maximum_tcp_angle = max(
        max(initial_posture["tcp_forward_axis_horizontal_angle_deg"].values()),
        max(initial_posture["jaw_axis_horizontal_angle_deg"].values()),
        max(initial_posture["camera_up_axis_error_deg"].values()),
    )
    minimum_tcp_height = min(initial_posture["tcp_height_above_table_m"].values())
    minimum_arm_clearance = min(initial_posture["arm_table_clearance_m"].values())
    tcp_horizontal = maximum_tcp_angle <= max_initial_tcp_angle_deg
    maximum_parallel_angle = max(
        initial_posture["gripper_axis_parallel_angle_deg"].values()
    )
    grippers_parallel = maximum_parallel_angle <= max_initial_parallel_angle_deg
    initial_table_safe = (
        minimum_tcp_height >= minimum_initial_tcp_above_table_m
        and minimum_arm_clearance >= minimum_initial_arm_table_clearance_m
    )
    feasible = (
        feasible_ratio == 1.0
        and collision_free
        and minimum_margin_deg >= 1.0
        and tcp_horizontal
        and grippers_parallel
        and initial_table_safe
    )
    horizontal_penalty = horizontal_penalty_weight * max(
        0.0, maximum_tcp_angle - horizontal_deadband_deg
    )
    score = (
        1000.0 * feasible_ratio
        - 2000.0 * float(np.max(errors_array))
        + 10.0 * min(clearance_min, 0.05)
        + 10.0 * min(cross_arm_clearance_min, 0.05)
        + 0.02 * min(minimum_margin_deg, 30.0)
        - 0.002 * min(maximum_condition, 1000.0)
        - (100.0 if not collision_free else 0.0)
        - horizontal_penalty
        - (100.0 if not initial_table_safe else 0.0)
    )
    return {
        "feasible": feasible,
        "feasible_target_ratio": feasible_ratio,
        "position_error_mean_m": float(np.mean(errors_array)),
        "position_error_max_m": float(np.max(errors_array)),
        "environment_clearance_min_m": clearance_min,
        "closest_environment_pair": closest_pair,
        "cross_arm_clearance_min_m": cross_arm_clearance_min,
        "closest_cross_arm_pair": closest_cross_arm_pair,
        "joint_margin_min_deg": minimum_margin_deg,
        "jacobian_condition_max": maximum_condition,
        "initial_posture": initial_posture,
        "initial_tcp_horizontal": tcp_horizontal,
        "initial_grippers_parallel": grippers_parallel,
        "initial_table_safe": initial_table_safe,
        "score": score,
        "first_frame_home_q_rad_by_side": first_home,
        "initial_tcp_positions_m": initial_posture["tcp_positions_m"],
        "waist_yaw_position_m": data.xpos[model.body("waist_yaw_link").id].tolist(),
        "initial_gripper_forward_yaw_deg": initial_gripper_forward_yaw_deg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay_npz", required=True)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--output_dir", default=str(HERE / "base_pose_search_outputs"))
    parser.add_argument("--name", default="linglong_base_waist_search")
    parser.add_argument("--keyframes", type=int, default=24)
    parser.add_argument("--base_forward_m", default="-0.20,-0.15,-0.10,-0.05,0.0")
    parser.add_argument("--waist_extension_rad", default="0.05,0.15,0.25,0.35,0.45,0.55,0.65")
    parser.add_argument("--waist_lean_deg", default="-10,-5,0,5,10,15,20,25,30")
    parser.add_argument("--position_tolerance_m", type=float, default=0.005)
    parser.add_argument("--max_initial_tcp_angle_deg", type=float, default=10.0)
    parser.add_argument("--max_initial_parallel_angle_deg", type=float, default=5.0)
    parser.add_argument(
        "--initial_gripper_forward_yaw_deg",
        default=None,
        help="Comma-separated common horizontal TCP-forward yaw angles to search",
    )
    parser.add_argument("--initial_tcp_above_table_m", type=float, default=0.08)
    parser.add_argument("--minimum_initial_tcp_above_table_m", type=float, default=0.08)
    parser.add_argument("--initial_tcp_center_y_m", type=float, default=None)
    parser.add_argument("--initial_tcp_half_separation_m", type=float, default=None)
    parser.add_argument("--minimum_initial_arm_table_clearance_m", type=float, default=0.0)
    parser.add_argument("--horizontal_deadband_deg", type=float, default=5.0)
    parser.add_argument("--horizontal_penalty_weight", type=float, default=2.0)
    parser.add_argument("--max_candidates", type=int, default=0)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = replay.load_config(config_path)
    payload = np.load(Path(args.replay_npz).expanduser().resolve())
    left_targets = np.asarray(payload["left_target_m"], dtype=np.float64)
    right_targets = np.asarray(payload["right_target_m"], dtype=np.float64)
    if left_targets.shape != right_targets.shape or left_targets.ndim != 2:
        raise ValueError("replay NPZ target arrays must have matching Nx3 shapes")
    table_height = float((config.get("table_geometry") or {}).get("top_height_m", 0.75))
    target_initial_z = table_height + args.initial_tcp_above_table_m
    target_z_shifts = {
        "left": float(target_initial_z - left_targets[0, 2]),
        "right": float(target_initial_z - right_targets[0, 2]),
    }
    target_xy_shifts = {"left": [0.0, 0.0], "right": [0.0, 0.0]}
    if args.initial_tcp_center_y_m is not None:
        target_xy_shifts["left"][1] = float(
            args.initial_tcp_center_y_m - left_targets[0, 1]
        )
        target_xy_shifts["right"][1] = float(
            args.initial_tcp_center_y_m - right_targets[0, 1]
        )
    if args.initial_tcp_half_separation_m is not None:
        target_xy_shifts["left"][0] = float(
            -args.initial_tcp_half_separation_m - left_targets[0, 0]
        )
        target_xy_shifts["right"][0] = float(
            args.initial_tcp_half_separation_m - right_targets[0, 0]
        )
    left_targets = left_targets.copy()
    right_targets = right_targets.copy()
    left_targets[:, :2] += np.asarray(target_xy_shifts["left"])
    right_targets[:, :2] += np.asarray(target_xy_shifts["right"])
    left_targets[:, 2] += target_z_shifts["left"]
    right_targets[:, 2] += target_z_shifts["right"]
    minimum_tcp_z = table_height + args.minimum_initial_tcp_above_table_m
    tcp_height_clamp_counts = {
        "left": int(np.count_nonzero(left_targets[:, 2] < minimum_tcp_z)),
        "right": int(np.count_nonzero(right_targets[:, 2] < minimum_tcp_z)),
    }
    left_targets[:, 2] = np.maximum(left_targets[:, 2], minimum_tcp_z)
    right_targets[:, 2] = np.maximum(right_targets[:, 2], minimum_tcp_z)
    keyframes = choose_keyframes(left_targets, right_targets, args.keyframes)
    ground_z, previous_ground_clearance = grounded_base_z(config)
    yaw_values: list[float | None] = (
        parse_numbers(args.initial_gripper_forward_yaw_deg)
        if args.initial_gripper_forward_yaw_deg is not None
        else [None]
    )
    combinations = list(
        itertools.product(
            parse_numbers(args.base_forward_m),
            parse_numbers(args.waist_extension_rad),
            parse_numbers(args.waist_lean_deg),
            yaw_values,
        )
    )
    if args.max_candidates > 0:
        combinations = combinations[: args.max_candidates]

    candidates: list[dict[str, Any]] = []
    for index, (forward, extension, lean, gripper_yaw) in enumerate(combinations):
        candidate = candidate_config(config, ground_z, forward, extension, lean)
        metrics = evaluate_candidate(
            candidate,
            left_targets,
            right_targets,
            keyframes,
            args.position_tolerance_m,
            args.max_initial_tcp_angle_deg,
            args.minimum_initial_tcp_above_table_m,
            args.minimum_initial_arm_table_clearance_m,
            args.horizontal_deadband_deg,
            args.horizontal_penalty_weight,
            args.max_initial_parallel_angle_deg,
            gripper_yaw,
        )
        candidates.append(
            {
                "index": index,
                "base_forward_m": forward,
                "grounded_base_z_m": ground_z,
                "waist_extension_rad": extension,
                "waist_lean_deg": lean,
                "initial_gripper_forward_yaw_deg": gripper_yaw,
                "waist_home_q_rad": candidate["waist_home_q_rad"],
                **metrics,
            }
        )
        print(
            f"[{index + 1}/{len(combinations)}] y={forward:+.3f} "
            f"extension={extension:.3f} lean={lean:+.1f} yaw={gripper_yaw} "
            f"feasible={metrics['feasible']} score={metrics['score']:.3f} "
            f"error={metrics['position_error_max_m']:.4f} "
            f"clearance={metrics['environment_clearance_min_m']:.4f} "
            "orientation_angle="
            f"{max(max(metrics['initial_posture']['tcp_forward_axis_horizontal_angle_deg'].values()), max(metrics['initial_posture']['jaw_axis_horizontal_angle_deg'].values())):.1f}"
        )

    candidates.sort(
        key=lambda item: (
            bool(item["feasible"]),
            float(item["feasible_target_ratio"]),
            float(item["score"]),
        ),
        reverse=True,
    )
    best = candidates[0]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.name}_report.json"
    recommended_path = output_dir / f"{args.name}_recommended_config.json"
    full_trajectory_validated = len(keyframes) == len(left_targets)
    report = {
        "source_config": str(config_path),
        "source_replay_npz": str(Path(args.replay_npz).expanduser().resolve()),
        "grounding": {
            "original_base_z_m": float(config["base_position_m"][2]),
            "original_base_ground_clearance_m": previous_ground_clearance,
            "grounded_base_z_m": ground_z,
        },
        "keyframes": keyframes.tolist(),
        "constraints": {
            "position_tolerance_m": args.position_tolerance_m,
            "max_initial_tcp_angle_deg": args.max_initial_tcp_angle_deg,
            "max_initial_parallel_angle_deg": args.max_initial_parallel_angle_deg,
            "initial_tcp_above_table_m": args.initial_tcp_above_table_m,
            "initial_tcp_center_y_m": args.initial_tcp_center_y_m,
            "initial_tcp_half_separation_m": args.initial_tcp_half_separation_m,
            "minimum_initial_tcp_above_table_m": args.minimum_initial_tcp_above_table_m,
            "minimum_initial_arm_table_clearance_m": args.minimum_initial_arm_table_clearance_m,
            "horizontal_deadband_deg": args.horizontal_deadband_deg,
            "horizontal_penalty_weight": args.horizontal_penalty_weight,
        },
        "target_z_shifts_m": target_z_shifts,
        "target_xy_shifts_m": target_xy_shifts,
        "tcp_height_clamp_counts": tcp_height_clamp_counts,
        "candidate_count": len(candidates),
        "feasible_candidate_count": int(sum(bool(item["feasible"]) for item in candidates)),
        "best": best,
        "candidates": candidates,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    recommended = {
        "extends": "../config.json",
        "layout_name": f"{args.name}_recommended",
        "layout_status": (
            "base + waist full-trajectory validation result"
            if full_trajectory_validated
            else "base + waist keyframe search recommendation; full replay validation required"
        ),
        "base_position_m": [float(config["base_position_m"][0]), best["base_forward_m"], ground_z],
        "base_positions_m": {
            "left": [float(config["base_position_m"][0]), best["base_forward_m"], ground_z],
            "right": [float(config["base_position_m"][0]), best["base_forward_m"], ground_z],
        },
        "waist_home_q_rad": best["waist_home_q_rad"],
        "home_q_rad_by_side": best["first_frame_home_q_rad_by_side"],
        "task_anchor_positions_m": {
            "left": left_targets[0].tolist(),
            "right": right_targets[0].tolist(),
        },
        "base_waist_search": {
            "report": str(report_path),
            "feasible": best["feasible"],
            "validated_frame_count": len(keyframes),
            "full_trajectory_validated": full_trajectory_validated,
            "score": best["score"],
            "waist_extension_rad": best["waist_extension_rad"],
            "waist_lean_deg": best["waist_lean_deg"],
        },
    }
    recommended_path.write_text(
        json.dumps(recommended, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"report": str(report_path), "recommended": str(recommended_path), "best": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
