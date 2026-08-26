#!/usr/bin/env python3
"""Replay optimized human wrist translations on a dual-UR5e MuJoCo scene.

Version 1 deliberately maps translation only. Each robot end effector starts
from the configured home pose, then follows the corresponding human wrist
displacement relative to video frame 0.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mujoco
except ImportError as exc:
    raise SystemExit("MuJoCo is missing. Run simulation/dual_ur5e/setup.sh first.") from exc


ROOT = Path(__file__).resolve().parents[2]
SIDES = ("left", "right")
JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
OMNIPICKER_MIMIC_FROM_OUTER = {
    "inner_joint1": -1.0,
    "inner_joint3": -0.1,
    "inner_joint4": -0.25,
    "inner_joint0": 0.7,
    "outer_joint1": 1.0,
    "outer_joint3": -0.1,
    "outer_joint4": 0.25,
    "outer_joint0": -0.7,
}


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    model_path = Path(config["model_path"])
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    config["model_path"] = str(model_path.resolve())
    gripper = config.get("robot_gripper") or {}
    if gripper.get("model_path"):
        gripper_path = Path(gripper["model_path"])
        if not gripper_path.is_absolute():
            gripper_path = ROOT / gripper_path
        gripper["model_path"] = str(gripper_path.resolve())
        config["robot_gripper"] = gripper
    return config


def load_camera_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    required = ("lookat", "distance", "azimuth", "elevation")
    return state if all(key in state for key in required) else None


def apply_camera_state(camera: mujoco.MjvCamera, state: dict[str, Any]) -> None:
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray(state["lookat"], dtype=np.float64)
    camera.distance = float(state["distance"])
    camera.azimuth = float(state["azimuth"])
    camera.elevation = float(state["elevation"])


def save_camera_state(path: Path, camera: mujoco.MjvCamera) -> dict[str, Any]:
    state = {
        "lookat": [float(value) for value in camera.lookat],
        "distance": float(camera.distance),
        "azimuth": float(camera.azimuth),
        "elevation": float(camera.elevation),
    }
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return state


def build_model(config: dict[str, Any]) -> tuple[mujoco.MjModel, mujoco.MjData]:
    scene = mujoco.MjSpec.from_string(
        """<mujoco model="dual_ur5e">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.25 0.25 0.25" specular="0 0 0"/>
    <global offwidth="1920" offheight="1080"/>
  </visual>
  <asset>
    <texture type="2d" name="ground_tex" builtin="checker" rgb1=".22 .24 .28" rgb2=".12 .14 .18"
      mark="edge" markrgb=".65 .65 .65" width="256" height="256"/>
    <material name="ground_mat" texture="ground_tex" texuniform="true" texrepeat="4 4" reflectance=".08"/>
  </asset>
  <worldbody>
    <light pos="0 -1 2.5" dir="0 .35 -1" directional="true"/>
    <geom name="floor" type="plane" size="2 2 .05" material="ground_mat"/>
    <camera name="overview" pos="0 2.5 1.35" xyaxes="-1 0 0 0 -.30 .954"/>
    <body name="left_target" mocap="true"><geom type="sphere" size=".025" rgba=".1 .55 1 .8"
      contype="0" conaffinity="0"/></body>
    <body name="right_target" mocap="true"><geom type="sphere" size=".025" rgba="1 .45 .08 .8"
      contype="0" conaffinity="0"/></body>
  </worldbody>
</mujoco>"""
    )
    gripper = config.get("robot_gripper") or {}
    use_gripper = bool(gripper.get("enabled", False))
    for side in SIDES:
        child = mujoco.MjSpec.from_file(config["model_path"])
        if use_gripper:
            gripper_spec = mujoco.MjSpec.from_file(gripper["model_path"])
            mount_translation = np.asarray(
                gripper.get("ur5e_mount_translation_m", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            )
            if mount_translation.shape != (3,):
                raise ValueError("robot_gripper.ur5e_mount_translation_m must have 3 values")
            gripper_root = gripper_spec.body("base_link")
            gripper_root.pos = np.asarray(gripper_root.pos) + mount_translation
            child.attach(
                gripper_spec,
                prefix="omnipicker_",
                site=child.site("attachment_site"),
            )
        mount = scene.worldbody.add_frame(
            name=f"{side}_mount",
            pos=config["base_positions_m"][side],
        )
        scene.attach(child, prefix=f"{side}_", frame=mount)
    model = scene.compile()
    data = mujoco.MjData(model)
    home = np.asarray(config["home_q_rad"], dtype=np.float64)
    data.qpos[:] = 0.0
    for side in SIDES:
        qpos_ids, _ = joint_addresses(model, side)
        data.qpos[qpos_ids] = home
        if use_gripper:
            set_omnipicker_qpos(model, data.qpos, side, 0.0, config)
    set_controls_from_qpos(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def resolve_trajectory(args: argparse.Namespace) -> Path:
    if args.trajectory:
        path = Path(args.trajectory).expanduser().resolve()
    else:
        session = Path(args.session).expanduser().resolve()
        path = session / "outputs/data/trajectory_wristroot_track_cameraoptical.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"trajectory not found: {path}")
    return path


def row_stamp_ns(row: dict[str, Any]) -> int | None:
    for value in (
        row.get("rgb_stamp_ns"),
        (row.get("timestamp") or {}).get("rgb_stamp_ns"),
        (row.get("camera") or {}).get("rgb_stamp_ns"),
    ):
        if value is not None:
            return int(value)
    return None


def frame_times(rows: list[dict[str, Any]], fallback_fps: float) -> np.ndarray:
    stamps = [row_stamp_ns(row) for row in rows]
    if stamps and all(stamp is not None for stamp in stamps):
        values = np.asarray(stamps, dtype=np.float64)
        times = (values - values[0]) / 1e9
        if np.all(np.diff(times) > 0):
            return times
    return np.arange(len(rows), dtype=np.float64) / max(fallback_fps, 1e-6)


def wrist_point(row: dict[str, Any], side: str, coordinate: str) -> np.ndarray | None:
    hand = ((row.get("hands") or {}).get(side) or {})
    optimized = hand.get("optimized_trajectory") or {}
    glove = hand.get("glove") or {}
    direct = optimized.get(f"wrist_translation_{coordinate}_m")
    points = (
        optimized.get(f"kpts_3d_{coordinate}_m_optimized")
        or glove.get(f"kpts_3d_{coordinate}_m")
    )
    value = direct if direct is not None else (points[0] if points else None)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    return arr[:3] if arr.size >= 3 and np.isfinite(arr[:3]).all() else None


def interpolate_points(points: list[np.ndarray | None]) -> np.ndarray:
    values = np.full((len(points), 3), np.nan, dtype=np.float64)
    for index, point in enumerate(points):
        if point is not None:
            values[index] = point
    valid = np.flatnonzero(np.isfinite(values).all(axis=1))
    if not valid.size:
        raise ValueError("trajectory has no valid wrist points")
    indexes = np.arange(len(points))
    for axis in range(3):
        values[:, axis] = np.interp(indexes, valid, values[valid, axis])
    return values


def limit_target_speed(targets: np.ndarray, times: np.ndarray, max_speed: float) -> tuple[np.ndarray, int]:
    output = targets.copy()
    clipped = 0
    for index in range(1, len(output)):
        dt = max(1e-3, float(times[index] - times[index - 1]))
        delta = output[index] - output[index - 1]
        distance = float(np.linalg.norm(delta))
        allowed = max_speed * dt
        if distance > allowed > 0:
            output[index] = output[index - 1] + delta * (allowed / distance)
            clipped += 1
    return output, clipped


def build_targets(
    rows: list[dict[str, Any]],
    times: np.ndarray,
    initial_sites: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    coordinate = str(config.get("input_coordinate", "camera"))
    rotation = np.asarray(config["camera_optical_to_robot"], dtype=np.float64)
    scale = float(config.get("translation_scale", 1.0))
    lower = np.asarray(config["relative_workspace_min_m"], dtype=np.float64)
    upper = np.asarray(config["relative_workspace_max_m"], dtype=np.float64)
    targets: dict[str, np.ndarray] = {}
    speed_clips: dict[str, int] = {}
    for side in SIDES:
        human = interpolate_points([wrist_point(row, side, coordinate) for row in rows])
        relative = (rotation @ (human - human[0]).T).T * scale
        relative = np.clip(relative, lower, upper)
        target = initial_sites[side][None, :] + relative
        target, speed_clips[side] = limit_target_speed(
            target, times, float(config["max_target_speed_mps"])
        )
        targets[side] = target
    return targets, speed_clips


def trajectory_metrics(original: np.ndarray, conditioned: np.ndarray) -> dict[str, float]:
    deviation = np.linalg.norm(conditioned - original, axis=1)
    original_steps = np.linalg.norm(np.diff(original, axis=0), axis=1)
    conditioned_steps = np.linalg.norm(np.diff(conditioned, axis=0), axis=1)
    original_accel = np.linalg.norm(np.diff(original, n=2, axis=0), axis=1)
    conditioned_accel = np.linalg.norm(np.diff(conditioned, n=2, axis=0), axis=1)
    original_length = float(np.sum(original_steps))
    return {
        "deviation_p95_m": float(np.percentile(deviation, 95)),
        "deviation_max_m": float(np.max(deviation)),
        "endpoint_error_m": float(
            max(np.linalg.norm(conditioned[0] - original[0]), np.linalg.norm(conditioned[-1] - original[-1]))
        ),
        "path_length_ratio": float(np.sum(conditioned_steps) / max(original_length, 1e-12)),
        "step_p95_before_m": float(np.percentile(original_steps, 95)),
        "step_p95_after_m": float(np.percentile(conditioned_steps, 95)),
        "acceleration_p95_before_m_per_frame2": float(np.percentile(original_accel, 95)),
        "acceleration_p95_after_m_per_frame2": float(np.percentile(conditioned_accel, 95)),
    }


def condition_targets(
    targets: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    settings = config.get("action_conditioning") or {}
    if not bool(settings.get("enabled", False)):
        copied = {side: values.copy() for side, values in targets.items()}
        return copied, {side: trajectory_metrics(values, values) for side, values in targets.items()}
    frame_count = len(next(iter(targets.values())))
    if frame_count < 5:
        copied = {side: values.copy() for side, values in targets.items()}
        return copied, {side: trajectory_metrics(values, values) for side, values in targets.items()}

    identity = np.eye(frame_count, dtype=np.float64)
    acceleration = np.diff(identity, n=2, axis=0)
    jerk = np.diff(identity, n=3, axis=0)
    observation_weights = np.ones(frame_count, dtype=np.float64)
    observation_weights[[0, -1]] = float(settings.get("endpoint_weight", 1e6))
    system = np.diag(observation_weights)
    system += float(settings.get("acceleration_weight", 1.0)) * (acceleration.T @ acceleration)
    system += float(settings.get("jerk_weight", 1.0)) * (jerk.T @ jerk)

    conditioned: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    for side, original in targets.items():
        rhs = observation_weights[:, None] * original
        values = np.linalg.solve(system, rhs)
        values[0] = original[0]
        values[-1] = original[-1]
        conditioned[side] = values
        metrics[side] = trajectory_metrics(original, values)
    return conditioned, metrics


def contiguous_true_runs(mask: np.ndarray, minimum_length: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if bool(value) and start is None:
            start = index
        if start is not None and (not bool(value) or index == len(mask) - 1):
            end = index if bool(value) and index == len(mask) - 1 else index - 1
            if end - start + 1 >= minimum_length:
                runs.append((start, end))
            start = None
    return runs


def smootherstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


def expand_and_merge_runs(
    runs: list[tuple[int, int]], frame_count: int, expand_frames: int, merge_gap_frames: int
) -> list[tuple[int, int]]:
    if not runs:
        return []
    expanded = [
        (max(0, start - expand_frames), min(frame_count - 1, end + expand_frames))
        for start, end in runs
    ]
    merged: list[tuple[int, int]] = [expanded[0]]
    for start, end in expanded[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end - 1 <= merge_gap_frames:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def lock_static_targets(
    targets: dict[str, np.ndarray],
    times: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Lock only confidently static intervals, with C2 entry/exit blending."""
    settings = config.get("static_lock") or {}
    outputs = {side: values.copy() for side, values in targets.items()}
    candidates = {side: np.zeros(len(values), dtype=bool) for side, values in targets.items()}
    locked = {side: np.zeros(len(values), dtype=bool) for side, values in targets.items()}
    metrics: dict[str, Any] = {}
    if not bool(settings.get("enabled", False)) or len(times) < 3:
        return outputs, candidates, locked, metrics

    dt = float(np.median(np.diff(times)))
    requested_window_frames = int(settings.get("window_frames", 0))
    if requested_window_frames > 0:
        window_frames = max(3, requested_window_frames)
        if window_frames % 2 == 0:
            window_frames += 1
        half_window = window_frames // 2
    else:
        half_window = max(
            1,
            int(round(float(settings.get("window_sec", 0.5)) / max(dt, 1e-6) / 2.0)),
        )
        window_frames = 2 * half_window + 1
    max_net = float(settings.get("max_net_displacement_m", 0.003))
    max_radius = float(settings.get("max_radius_m", 0.0035))
    minimum_length = int(settings.get("min_candidate_frames", 6))
    requested_transition = int(settings.get("transition_frames", 4))
    expand_frames = max(0, int(settings.get("expand_frames", 0)))
    merge_gap_frames = max(0, int(settings.get("merge_gap_frames", 0)))
    transition_outside = bool(settings.get("transition_outside_candidate", False))
    candidate_merge_gap_frames = max(
        0,
        int(
            settings.get(
                "candidate_merge_gap_frames",
                merge_gap_frames + 2 * expand_frames,
            )
        ),
    )
    max_anchor_deviation = float(settings.get("max_anchor_deviation_m", np.inf))

    for side, original in targets.items():
        candidate = candidates[side]
        for center in range(half_window, len(original) - half_window):
            segment = original[center - half_window : center + half_window + 1]
            net = float(np.linalg.norm(segment[-1] - segment[0]))
            center_point = np.mean(segment, axis=0)
            radius = float(np.max(np.linalg.norm(segment - center_point, axis=1)))
            candidate[center] = net <= max_net and radius <= max_radius

        candidate_runs = contiguous_true_runs(candidate, minimum_length)
        if transition_outside:
            runs = expand_and_merge_runs(
                candidate_runs, len(original), 0, candidate_merge_gap_frames
            )
        else:
            runs = expand_and_merge_runs(
                candidate_runs, len(original), expand_frames, merge_gap_frames
            )
        output = outputs[side]
        run_records: list[dict[str, Any]] = []
        rejected_records: list[dict[str, Any]] = []
        for start, end in runs:
            length = end - start + 1
            transition = min(requested_transition, max(1, (length - 1) // 2))
            anchor = np.median(original[start : end + 1], axis=0)
            anchor_deviation = np.linalg.norm(original[start : end + 1] - anchor, axis=1)
            if float(np.max(anchor_deviation)) > max_anchor_deviation:
                rejected_records.append(
                    {
                        "start_frame": start,
                        "end_frame": end,
                        "max_anchor_deviation_m": float(np.max(anchor_deviation)),
                    }
                )
                continue
            before = output[start : end + 1].copy()
            if transition_outside:
                transition_start = max(0, start - requested_transition)
                transition_end = min(len(original) - 1, end + requested_transition)
                output[start : end + 1] = anchor
                left_span = max(1, start - transition_start)
                for frame_index in range(transition_start, start + 1):
                    alpha = smootherstep((frame_index - transition_start) / left_span)
                    output[frame_index] = (
                        (1.0 - alpha) * original[frame_index] + alpha * anchor
                    )
                right_span = max(1, transition_end - end)
                for frame_index in range(end, transition_end + 1):
                    alpha = smootherstep((transition_end - frame_index) / right_span)
                    output[frame_index] = (
                        (1.0 - alpha) * original[frame_index] + alpha * anchor
                    )
                core_start = start
                core_end = end
            else:
                output[start : end + 1] = anchor
                for offset in range(transition + 1):
                    alpha = smootherstep(offset / transition)
                    left_index = start + offset
                    right_index = end - offset
                    output[left_index] = (1.0 - alpha) * before[offset] + alpha * anchor
                    output[right_index] = (1.0 - alpha) * before[-offset - 1] + alpha * anchor
                core_start = start + transition
                core_end = end - transition
            locked[side][core_start : core_end + 1] = True
            run_records.append(
                {
                    "start_frame": start,
                    "end_frame": end,
                    "start_sec": float(times[start]),
                    "end_sec": float(times[end]),
                    "locked_core_start_frame": core_start,
                    "locked_core_end_frame": core_end,
                }
            )

        deviation = np.linalg.norm(output - original, axis=1)
        metrics[side] = {
            "candidate_frames": int(np.count_nonzero(candidate)),
            "locked_core_frames": int(np.count_nonzero(locked[side])),
            "segments": run_records,
            "rejected_segments": rejected_records,
            "deviation_p95_m": float(np.percentile(deviation, 95)),
            "deviation_max_m": float(np.max(deviation)),
        }
    metrics["parameters"] = {
        "window_sec": float(settings.get("window_sec", 0.5)),
        "window_frames": window_frames,
        "max_net_displacement_m": max_net,
        "max_radius_m": max_radius,
        "min_candidate_frames": minimum_length,
        "transition_frames": requested_transition,
        "expand_frames": expand_frames,
        "merge_gap_frames": merge_gap_frames,
        "transition_outside_candidate": transition_outside,
        "candidate_merge_gap_frames": candidate_merge_gap_frames,
        "max_anchor_deviation_m": max_anchor_deviation,
    }
    return outputs, candidates, locked, metrics


def resample_series(
    values: np.ndarray,
    old_times: np.ndarray,
    new_times: np.ndarray,
    zero_endpoint_velocity: bool = False,
) -> np.ndarray:
    """C1-continuous cubic-Hermite resampling for robot command targets.

    Linear interpolation makes velocity jump at every source frame. Those tiny
    corners are easy to miss in Cartesian plots but become visible joint jerk
    after IK. Central time derivatives preserve the measured samples while
    producing continuous command velocity between them.
    """
    flat = values.reshape(len(values), -1)
    if len(values) < 3:
        output = np.column_stack(
            [np.interp(new_times, old_times, flat[:, column]) for column in range(flat.shape[1])]
        )
        return output.reshape((len(new_times),) + values.shape[1:])

    slopes = np.gradient(flat, old_times, axis=0, edge_order=2)
    if zero_endpoint_velocity:
        slopes[0] = 0.0
        slopes[-1] = 0.0
    segment = np.searchsorted(old_times, new_times, side="right") - 1
    segment = np.clip(segment, 0, len(old_times) - 2)
    t0 = old_times[segment]
    t1 = old_times[segment + 1]
    duration = np.maximum(t1 - t0, 1e-9)
    phase = ((new_times - t0) / duration)[:, None]
    phase2 = phase * phase
    phase3 = phase2 * phase
    h00 = 2.0 * phase3 - 3.0 * phase2 + 1.0
    h10 = phase3 - 2.0 * phase2 + phase
    h01 = -2.0 * phase3 + 3.0 * phase2
    h11 = phase3 - phase2
    output = (
        h00 * flat[segment]
        + h10 * duration[:, None] * slopes[segment]
        + h01 * flat[segment + 1]
        + h11 * duration[:, None] * slopes[segment + 1]
    )
    output[0] = flat[0]
    output[-1] = flat[-1]
    return output.reshape((len(new_times),) + values.shape[1:])


def resample_targets(
    targets: dict[str, np.ndarray],
    times: np.ndarray,
    command_fps: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if len(times) < 2 or command_fps <= 0:
        return times.copy(), {side: values.copy() for side, values in targets.items()}
    frame_count = max(2, int(round(float(times[-1]) * command_fps)) + 1)
    command_times = np.linspace(0.0, float(times[-1]), frame_count)
    return command_times, {
        side: resample_series(values, times, command_times)
        for side, values in targets.items()
    }


def joint_addresses(model: mujoco.MjModel, side: str) -> tuple[np.ndarray, np.ndarray]:
    qpos, dofs = [], []
    for suffix in JOINT_NAMES:
        joint = model.joint(f"{side}_{suffix}")
        qpos.append(int(joint.qposadr[0]))
        dofs.append(int(joint.dofadr[0]))
    return np.asarray(qpos, dtype=int), np.asarray(dofs, dtype=int)


def end_effector_site_id(model: mujoco.MjModel, side: str) -> int:
    """Use the OmniPicker TCP when fitted, otherwise the bare UR5e flange site."""
    gripper_tcp = f"{side}_omnipicker_tcp"
    site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, gripper_tcp))
    if site_id >= 0:
        return site_id
    return int(model.site(f"{side}_attachment_site").id)


def set_omnipicker_qpos(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    side: str,
    canonical_command: float,
    config: dict[str, Any],
) -> None:
    """Map canonical 0=open, 1=closed onto the official mimic coordinates."""
    settings = config.get("robot_gripper") or {}
    open_angle = float(settings.get("open_joint_rad", np.pi / 4.0))
    command = float(np.clip(canonical_command, 0.0, 1.0))
    outer_joint = open_angle * (1.0 - command)
    for suffix, multiplier in OMNIPICKER_MIMIC_FROM_OUTER.items():
        joint = model.joint(f"{side}_omnipicker_{suffix}")
        qpos[int(joint.qposadr[0])] = multiplier * outer_joint


def apply_omnipicker_commands(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    gripper_artifacts: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any],
) -> None:
    settings = config.get("robot_gripper") or {}
    if not bool(settings.get("enabled", False)):
        return
    for side in SIDES:
        commands = np.asarray(gripper_artifacts[side]["command"], dtype=np.float64)
        if len(commands) != len(qpos):
            raise ValueError(
                f"{side} OmniPicker command length {len(commands)} != qpos length {len(qpos)}"
            )
        for frame, command in enumerate(commands):
            set_omnipicker_qpos(model, qpos[frame], side, float(command), config)


def set_controls_from_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Set joint-position actuators without assuming nu == nq."""
    data.ctrl[:] = 0.0
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            continue
        qpos_address = int(model.jnt_qposadr[joint_id])
        value = float(data.qpos[qpos_address])
        if int(model.actuator_ctrllimited[actuator_id]):
            value = float(
                np.clip(
                    value,
                    model.actuator_ctrlrange[actuator_id, 0],
                    model.actuator_ctrlrange[actuator_id, 1],
                )
            )
        data.ctrl[actuator_id] = value


def clamp_joint_ranges(model: mujoco.MjModel, qpos: np.ndarray, side: str) -> None:
    for suffix in JOINT_NAMES:
        joint = model.joint(f"{side}_{suffix}")
        if int(joint.limited[0]):
            address = int(joint.qposadr[0])
            qpos[address] = np.clip(qpos[address], joint.range[0], joint.range[1])


def rotation_error_vector(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking current orientation to target."""
    relative = target @ current.T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = np.asarray(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1e-7:
        return 0.5 * skew
    sine = float(np.sin(angle))
    if abs(sine) < 1e-7:
        # This path is not expected for warm-started replay, but keeps the
        # diagnostic finite if an initial pose is close to a pi rotation.
        eigenvalues, eigenvectors = np.linalg.eigh(relative)
        axis = eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))]
        return axis * angle
    return skew * (angle / (2.0 * sine))


def rotation_matrix_from_vector(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-10:
        return np.eye(3, dtype=np.float64)
    axis = vector / angle
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def interpolate_rotation(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    """Geodesic interpolation on SO(3), using world-frame left increments."""
    increment = rotation_error_vector(second, first)
    return rotation_matrix_from_vector(float(alpha) * increment) @ first


def repair_orientation_spikes(
    values: np.ndarray, settings: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Replace isolated SO(3) midpoint outliers by geodesic interpolation."""
    output = values.copy()
    mask = np.zeros(len(values), dtype=bool)
    if len(values) < 3 or not bool(settings.get("spike_filter_enabled", True)):
        return output, mask, {"repaired_frames": 0}

    rotations = np.asarray([rotation_matrix_from_vector(value) for value in values])
    residuals = np.zeros(len(values), dtype=np.float64)
    for index in range(1, len(values) - 1):
        midpoint = interpolate_rotation(rotations[index - 1], rotations[index + 1], 0.5)
        residuals[index] = np.linalg.norm(rotation_error_vector(rotations[index], midpoint))
    interior = residuals[1:-1]
    median = float(np.median(interior))
    mad = float(np.median(np.abs(interior - median)))
    robust_sigma = 1.4826 * mad
    floor = np.deg2rad(float(settings.get("spike_residual_floor_deg", 0.75)))
    multiplier = float(settings.get("spike_mad_multiplier", 6.0))
    threshold = max(floor, median + multiplier * robust_sigma)
    # A single bad sample also raises the midpoint residuals of its two
    # neighbours.  Non-maximum suppression keeps the actual impulse instead
    # of unnecessarily replacing all three samples.
    for index in range(1, len(values) - 1):
        mask[index] = (
            residuals[index] > threshold
            and residuals[index] >= residuals[index - 1]
            and residuals[index] >= residuals[index + 1]
        )

    for start, end in contiguous_true_runs(mask, 1):
        if start == 0 or end == len(values) - 1:
            continue
        span = end - start + 2
        for offset, index in enumerate(range(start, end + 1), start=1):
            repaired = interpolate_rotation(rotations[start - 1], rotations[end + 1], offset / span)
            output[index] = rotation_error_vector(repaired, np.eye(3))
    return output, mask, {
        "repaired_frames": int(np.count_nonzero(mask)),
        "midpoint_residual_p95_deg": float(np.rad2deg(np.percentile(interior, 95))),
        "midpoint_residual_max_deg": float(np.rad2deg(np.max(interior))),
        "threshold_deg": float(np.rad2deg(threshold)),
    }


def lock_static_orientations(
    detection_values: np.ndarray,
    command_values: np.ndarray,
    times: np.ndarray,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Lock only low-excursion, inefficient rotational motion (orientation jitter)."""
    output = command_values.copy()
    candidate = np.zeros(len(output), dtype=bool)
    locked = np.zeros(len(output), dtype=bool)
    if len(output) < 3 or not bool(settings.get("static_lock_enabled", True)):
        return output, candidate, locked, {}

    window_frames = max(3, int(settings.get("static_window_frames", 15)))
    if window_frames % 2 == 0:
        window_frames += 1
    half_window = window_frames // 2
    max_net = np.deg2rad(float(settings.get("static_max_net_angle_deg", 2.0)))
    max_radius = np.deg2rad(float(settings.get("static_max_radius_deg", 2.0)))
    strong_net = np.deg2rad(float(settings.get("static_strong_net_angle_deg", 0.75)))
    max_efficiency = float(settings.get("static_max_motion_efficiency", 0.60))
    rotations = np.asarray([rotation_matrix_from_vector(value) for value in detection_values])
    for center in range(half_window, len(output) - half_window):
        segment = rotations[center - half_window : center + half_window + 1]
        net = float(np.linalg.norm(rotation_error_vector(segment[-1], segment[0])))
        path = float(
            np.sum(
                [
                    np.linalg.norm(rotation_error_vector(segment[index], segment[index - 1]))
                    for index in range(1, len(segment))
                ]
            )
        )
        center_rotation = rotation_matrix_from_vector(
            np.median(detection_values[center - half_window : center + half_window + 1], axis=0)
        )
        radius = max(
            float(np.linalg.norm(rotation_error_vector(rotation, center_rotation)))
            for rotation in segment
        )
        efficiency = net / max(path, 1e-9)
        candidate[center] = (
            net <= max_net
            and radius <= max_radius
            and (net <= strong_net or efficiency <= max_efficiency)
        )

    minimum_length = int(settings.get("static_min_candidate_frames", 4))
    merge_gap = int(settings.get("static_candidate_merge_gap_frames", 9))
    runs = expand_and_merge_runs(
        contiguous_true_runs(candidate, minimum_length), len(output), 0, merge_gap
    )
    transition = max(0, int(settings.get("static_transition_frames", 6)))
    max_anchor_deviation = np.deg2rad(
        float(settings.get("static_max_anchor_deviation_deg", 5.0))
    )
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for start, end in runs:
        anchor_vector = np.median(command_values[start : end + 1], axis=0)
        anchor_rotation = rotation_matrix_from_vector(anchor_vector)
        deviations = np.asarray(
            [
                np.linalg.norm(
                    rotation_error_vector(rotation_matrix_from_vector(value), anchor_rotation)
                )
                for value in command_values[start : end + 1]
            ]
        )
        if float(np.max(deviations)) > max_anchor_deviation:
            rejected.append(
                {
                    "start_frame": start,
                    "end_frame": end,
                    "max_anchor_deviation_deg": float(np.rad2deg(np.max(deviations))),
                }
            )
            continue
        output[start : end + 1] = anchor_vector
        locked[start : end + 1] = True
        transition_start = max(0, start - transition)
        transition_end = min(len(output) - 1, end + transition)
        for index in range(transition_start, start):
            alpha = smootherstep((index - transition_start + 1) / max(1, start - transition_start + 1))
            blended = interpolate_rotation(
                rotation_matrix_from_vector(command_values[index]), anchor_rotation, alpha
            )
            output[index] = rotation_error_vector(blended, np.eye(3))
        for index in range(end + 1, transition_end + 1):
            alpha = smootherstep((transition_end - index + 1) / max(1, transition_end - end + 1))
            blended = interpolate_rotation(
                rotation_matrix_from_vector(command_values[index]), anchor_rotation, alpha
            )
            output[index] = rotation_error_vector(blended, np.eye(3))
        accepted.append(
            {
                "start_frame": start,
                "end_frame": end,
                "start_sec": float(times[start]),
                "end_sec": float(times[end]),
            }
        )

    correction = np.asarray(
        [
            np.linalg.norm(
                rotation_error_vector(
                    rotation_matrix_from_vector(output[index]),
                    rotation_matrix_from_vector(command_values[index]),
                )
            )
            for index in range(len(output))
        ]
    )
    return output, candidate, locked, {
        "candidate_frames": int(np.count_nonzero(candidate)),
        "locked_frames": int(np.count_nonzero(locked)),
        "segments": accepted,
        "rejected_segments": rejected,
        "correction_p95_deg": float(np.rad2deg(np.percentile(correction, 95))),
        "correction_max_deg": float(np.rad2deg(np.max(correction))),
        "parameters": {
            "window_frames": window_frames,
            "max_net_angle_deg": float(np.rad2deg(max_net)),
            "max_radius_deg": float(np.rad2deg(max_radius)),
            "strong_net_angle_deg": float(np.rad2deg(strong_net)),
            "max_motion_efficiency": max_efficiency,
            "max_anchor_deviation_deg": float(np.rad2deg(max_anchor_deviation)),
        },
    }


def regularize_vector_series(values: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    frame_count = len(values)
    if frame_count < 5 or not bool(settings.get("enabled", True)):
        return values.copy()
    identity = np.eye(frame_count, dtype=np.float64)
    acceleration = np.diff(identity, n=2, axis=0)
    jerk = np.diff(identity, n=3, axis=0)
    weights = np.ones(frame_count, dtype=np.float64)
    weights[[0, -1]] = float(settings.get("endpoint_weight", 1e6))
    system = np.diag(weights)
    system += float(settings.get("acceleration_weight", 4.0)) * (acceleration.T @ acceleration)
    system += float(settings.get("jerk_weight", 4.0)) * (jerk.T @ jerk)
    output = np.linalg.solve(system, weights[:, None] * values)
    output[0] = values[0]
    output[-1] = values[-1]
    return output


def build_relative_orientation_targets(
    rows: list[dict[str, Any]],
    source_times: np.ndarray,
    command_times: np.ndarray,
    initial_site_rotations: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, float]],
]:
    settings = config.get("human_orientation") or {}
    axis_map = np.asarray(config["camera_optical_to_robot"], dtype=np.float64)
    scale = float(settings.get("relative_rotation_scale", 0.5))
    raw_vectors: dict[str, np.ndarray] = {}
    repaired_vectors: dict[str, np.ndarray] = {}
    conditioned_vectors: dict[str, np.ndarray] = {}
    masks = {"spike": {}, "static_candidate": {}, "static_locked": {}}
    targets: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    for side in SIDES:
        rotations = np.asarray(
            [
                row["hands"][side]["optimized_trajectory"]["palm_rotation_camera"]
                for row in rows
            ],
            dtype=np.float64,
        )
        reference = rotations[0]
        human_relative_vectors = np.asarray(
            [rotation_error_vector(rotation @ reference.T, np.eye(3)) for rotation in rotations]
        )
        raw = (axis_map @ human_relative_vectors.T).T * scale
        repaired, spike_mask, spike_metrics = repair_orientation_spikes(raw, settings)
        smoothed = regularize_vector_series(repaired, settings)
        conditioned, static_candidate, static_locked, static_metrics = lock_static_orientations(
            repaired, smoothed, source_times, settings
        )
        command = resample_series(conditioned, source_times, command_times)
        target_matrices = np.asarray(
            [rotation_matrix_from_vector(vector) @ initial_site_rotations[side] for vector in command]
        )
        raw_vectors[side] = raw
        repaired_vectors[side] = repaired
        conditioned_vectors[side] = conditioned
        masks["spike"][side] = spike_mask
        masks["static_candidate"][side] = static_candidate
        masks["static_locked"][side] = static_locked
        targets[side] = target_matrices
        raw_step = np.linalg.norm(np.diff(raw, axis=0), axis=1)
        conditioned_step = np.linalg.norm(np.diff(conditioned, axis=0), axis=1)
        correction = np.linalg.norm(conditioned - raw, axis=1)
        metrics[side] = {
            "relative_rotation_scale": scale,
            "raw_relative_angle_max_deg": float(np.rad2deg(np.max(np.linalg.norm(raw, axis=1)))),
            "conditioned_relative_angle_max_deg": float(
                np.rad2deg(np.max(np.linalg.norm(conditioned, axis=1)))
            ),
            "raw_step_p95_deg": float(np.rad2deg(np.percentile(raw_step, 95))),
            "raw_step_max_deg": float(np.rad2deg(np.max(raw_step))),
            "conditioned_step_p95_deg": float(np.rad2deg(np.percentile(conditioned_step, 95))),
            "conditioned_step_max_deg": float(np.rad2deg(np.max(conditioned_step))),
            "conditioning_correction_p95_deg": float(np.rad2deg(np.percentile(correction, 95))),
            "conditioning_correction_max_deg": float(np.rad2deg(np.max(correction))),
            "spike_filter": spike_metrics,
            "static_lock": static_metrics,
        }
    return targets, raw_vectors, repaired_vectors, conditioned_vectors, masks, metrics


def solve_pose_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    target_position: np.ndarray,
    target_rotation: np.ndarray | None,
    home: np.ndarray,
    config: dict[str, Any],
) -> tuple[float, float]:
    site_id = end_effector_site_id(model, side)
    qpos_ids, dof_ids = joint_addresses(model, side)
    damping = float(config["ik_damping"])
    orientation_weight = float(config.get("ik_orientation_weight_m_per_rad", 0.25))
    position_tolerance = float(config["ik_tolerance_m"])
    orientation_tolerance = float(config.get("ik_orientation_tolerance_rad", 1e-3))
    for _ in range(int(config["ik_iterations"])):
        mujoco.mj_forward(model, data)
        position_error = target_position - data.site_xpos[site_id]
        rotation_error = (
            np.zeros(3, dtype=np.float64)
            if target_rotation is None
            else rotation_error_vector(
                target_rotation, data.site_xmat[site_id].reshape(3, 3)
            )
        )
        if (
            float(np.linalg.norm(position_error)) <= position_tolerance
            and float(np.linalg.norm(rotation_error)) <= orientation_tolerance
        ):
            break
        jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            model, data, jacobian_position, jacobian_rotation, site_id
        )
        if target_rotation is None:
            error = position_error
            jacobian = jacobian_position[:, dof_ids]
        else:
            error = np.concatenate(
                [position_error, orientation_weight * rotation_error]
            )
            jacobian = np.vstack(
                [
                    jacobian_position[:, dof_ids],
                    orientation_weight * jacobian_rotation[:, dof_ids],
                ]
            )
        inverse = np.linalg.solve(
            jacobian @ jacobian.T + np.eye(len(error)) * damping * damping,
            np.eye(len(error)),
        )
        pseudo_inverse = jacobian.T @ inverse
        delta = pseudo_inverse @ error
        nullspace = np.eye(len(dof_ids)) - pseudo_inverse @ jacobian
        delta += nullspace @ (
            float(config["ik_posture_gain"]) * (home - data.qpos[qpos_ids])
        )
        norm = float(np.linalg.norm(delta))
        limit = float(config["ik_step_limit_rad"])
        if norm > limit:
            delta *= limit / norm
        data.qpos[qpos_ids] += delta
        clamp_joint_ranges(model, data.qpos, side)
    mujoco.mj_forward(model, data)
    position_error_norm = float(
        np.linalg.norm(target_position - data.site_xpos[site_id])
    )
    orientation_error_norm = (
        0.0
        if target_rotation is None
        else float(
            np.linalg.norm(
                rotation_error_vector(
                    target_rotation, data.site_xmat[site_id].reshape(3, 3)
                )
            )
        )
    )
    return position_error_norm, orientation_error_norm


def condition_joint_trajectory(
    qpos: np.ndarray, model: mujoco.MjModel, config: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    settings = config.get("joint_conditioning") or {}
    passes = int(settings.get("binomial_passes", 0))
    if not bool(settings.get("enabled", False)) or passes <= 0 or len(qpos) < 5:
        return qpos.copy(), {"enabled": False, "binomial_passes": 0}
    output = qpos.copy()
    active_ids = np.concatenate([joint_addresses(model, side)[0] for side in SIDES])
    kernel = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float64) / 16.0
    for _ in range(passes):
        padded = np.pad(output, ((2, 2), (0, 0)), mode="edge")
        for joint in active_ids:
            output[:, joint] = np.convolve(padded[:, joint], kernel, mode="valid")
    if bool(settings.get("preserve_endpoints", True)):
        output[0] = qpos[0]
        output[-1] = qpos[-1]
    metrics: dict[str, Any] = {
        "enabled": True,
        "binomial_passes": passes,
        "preserve_endpoints": bool(settings.get("preserve_endpoints", True)),
    }
    for side in SIDES:
        side_ids, _ = joint_addresses(model, side)
        deviation = np.linalg.norm(output[:, side_ids] - qpos[:, side_ids], axis=1)
        metrics[side] = {
            "joint_deviation_p95_deg": float(np.rad2deg(np.percentile(deviation, 95))),
            "joint_deviation_max_deg": float(np.rad2deg(np.max(deviation))),
        }
    return output, metrics


def resample_linear(values: np.ndarray, old_times: np.ndarray, new_times: np.ndarray) -> np.ndarray:
    flat = values.reshape(len(values), -1)
    output = np.column_stack(
        [np.interp(new_times, old_times, flat[:, column]) for column in range(flat.shape[1])]
    )
    return output.reshape((len(new_times),) + values.shape[1:])


def resample_rotations(
    values: np.ndarray, old_times: np.ndarray, new_times: np.ndarray
) -> np.ndarray:
    segment = np.searchsorted(old_times, new_times, side="right") - 1
    segment = np.clip(segment, 0, len(old_times) - 2)
    duration = np.maximum(old_times[segment + 1] - old_times[segment], 1e-9)
    phase = (new_times - old_times[segment]) / duration
    output = np.asarray(
        [
            interpolate_rotation(values[index], values[index + 1], alpha)
            for index, alpha in zip(segment, phase)
        ]
    )
    output[0] = values[0]
    output[-1] = values[-1]
    return output


def motion_limit_metrics(
    qpos: np.ndarray,
    targets: dict[str, np.ndarray],
    rotations: dict[str, np.ndarray],
    times: np.ndarray,
    model: mujoco.MjModel,
) -> dict[str, float]:
    if len(times) < 2:
        return {}
    dt = np.maximum(np.diff(times), 1e-9)
    active_ids = np.concatenate([joint_addresses(model, side)[0] for side in SIDES])
    joint_velocity = np.diff(qpos[:, active_ids], axis=0) / dt[:, None]
    if len(joint_velocity) > 1:
        joint_acceleration = np.diff(joint_velocity, axis=0) / (
            0.5 * (dt[:-1] + dt[1:])
        )[:, None]
        acceleration_max = float(np.max(np.abs(joint_acceleration)))
    else:
        acceleration_max = 0.0
    translation_speeds = []
    angular_speeds = []
    for side in SIDES:
        translation_speeds.append(
            np.linalg.norm(np.diff(targets[side], axis=0), axis=1) / dt
        )
        angular_speeds.append(
            np.asarray(
                [
                    np.linalg.norm(
                        rotation_error_vector(
                            rotations[side][index + 1], rotations[side][index]
                        )
                    )
                    for index in range(len(dt))
                ]
            )
            / dt
        )
    return {
        "tcp_translation_speed_max_mps": float(np.max(translation_speeds)),
        "tcp_angular_speed_max_rad_s": float(np.max(angular_speeds)),
        "joint_speed_max_rad_s": float(np.max(np.abs(joint_velocity))),
        "joint_acceleration_max_rad_s2": acceleration_max,
    }


def apply_shared_time_scaling(
    qpos: np.ndarray,
    qpos_ik_raw: np.ndarray,
    targets: dict[str, np.ndarray],
    raw_targets: dict[str, np.ndarray],
    rotations: dict[str, np.ndarray],
    times: np.ndarray,
    model: mujoco.MjModel,
    config: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Retain the bimanual path while assigning a shared robot-safe time law."""
    settings = config.get("time_scaling") or {}
    before = motion_limit_metrics(qpos, targets, rotations, times, model)
    if not bool(settings.get("enabled", False)) or len(times) < 3:
        return qpos, qpos_ik_raw, targets, raw_targets, rotations, times, times.copy(), {
            "enabled": False,
            "before": before,
            "after": before,
        }

    original_dt = np.maximum(np.diff(times), 1e-6)
    duration = original_dt.copy()
    tcp_speed_limit = float(settings.get("max_tcp_translation_speed_mps", 0.25))
    angular_speed_limit = float(settings.get("max_tcp_angular_speed_rad_s", 0.8))
    joint_speed_limit = float(settings.get("max_joint_speed_rad_s", 1.2))
    joint_acceleration_limit = float(settings.get("max_joint_acceleration_rad_s2", 3.0))
    active_ids = np.concatenate([joint_addresses(model, side)[0] for side in SIDES])
    joint_delta = np.diff(qpos[:, active_ids], axis=0)

    required_translation = np.zeros_like(duration)
    required_rotation = np.zeros_like(duration)
    for side in SIDES:
        required_translation = np.maximum(
            required_translation,
            np.linalg.norm(np.diff(targets[side], axis=0), axis=1) / tcp_speed_limit,
        )
        required_rotation = np.maximum(
            required_rotation,
            np.asarray(
                [
                    np.linalg.norm(
                        rotation_error_vector(
                            rotations[side][index + 1], rotations[side][index]
                        )
                    )
                    for index in range(len(duration))
                ]
            )
            / angular_speed_limit,
        )
    required_joint = np.max(np.abs(joint_delta), axis=1) / joint_speed_limit
    duration = np.maximum.reduce(
        [duration, required_translation, required_rotation, required_joint]
    )

    max_iterations = int(settings.get("acceleration_iterations", 100))
    tolerance = float(settings.get("constraint_tolerance", 1e-3))
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        velocity = joint_delta / duration[:, None]
        acceleration = np.diff(velocity, axis=0) / (
            0.5 * (duration[:-1] + duration[1:])
        )[:, None]
        ratio = np.max(np.abs(acceleration), axis=1) / joint_acceleration_limit
        if float(np.max(ratio, initial=0.0)) <= 1.0 + tolerance:
            break
        interval_scale = np.ones_like(duration)
        for transition_index in np.flatnonzero(ratio > 1.0 + tolerance):
            scale = min(2.0, float(np.sqrt(ratio[transition_index])))
            interval_scale[transition_index] = max(interval_scale[transition_index], scale)
            interval_scale[transition_index + 1] = max(
                interval_scale[transition_index + 1], scale
            )
        duration *= interval_scale

    render_fps = float(config["render_fps"])
    resample_iterations = 0
    resample_iteration_limit = int(settings.get("resample_constraint_iterations", 20))
    global_safety_scale = 1.0
    while True:
        path_times = np.concatenate([[0.0], np.cumsum(duration)])
        frame_count = max(2, int(round(path_times[-1] * render_fps)) + 1)
        output_times = np.linspace(0.0, float(path_times[-1]), frame_count)
        # Joint commands need C1 interpolation: piecewise linear resampling
        # would reintroduce acceleration impulses at every retimed path knot.
        output_qpos = resample_series(
            qpos, path_times, output_times, zero_endpoint_velocity=True
        )
        output_dt = np.diff(output_times)
        output_velocity = np.diff(output_qpos[:, active_ids], axis=0) / output_dt[:, None]
        output_acceleration = np.diff(output_velocity, axis=0) / (
            0.5 * (output_dt[:-1] + output_dt[1:])
        )[:, None]
        output_ratio = (
            np.max(np.abs(output_acceleration), axis=1) / joint_acceleration_limit
        )
        start_ratio = float(
            np.max(np.abs(output_velocity[0]))
            / max(output_dt[0], 1e-9)
            / joint_acceleration_limit
        )
        end_ratio = float(
            np.max(np.abs(output_velocity[-1]))
            / max(output_dt[-1], 1e-9)
            / joint_acceleration_limit
        )
        maximum_output_ratio = max(
            float(np.max(output_ratio, initial=0.0)), start_ratio, end_ratio
        )
        if maximum_output_ratio <= 1.0 + tolerance:
            break
        if resample_iterations >= resample_iteration_limit:
            # Guaranteed conservative fallback. It is only reached if local
            # neighbourhood stretching cannot resolve a resampling corner.
            global_safety_scale = float(np.sqrt(maximum_output_ratio) * 1.002)
            duration *= global_safety_scale
            path_times = np.concatenate([[0.0], np.cumsum(duration)])
            frame_count = max(2, int(round(path_times[-1] * render_fps)) + 1)
            output_times = np.linspace(0.0, float(path_times[-1]), frame_count)
            output_qpos = resample_series(
                qpos, path_times, output_times, zero_endpoint_velocity=True
            )
            break
        interval_scale = np.ones_like(duration)
        if start_ratio > 1.0 + tolerance:
            scale = min(1.5, float(np.sqrt(start_ratio) * 1.002))
            interval_scale[: min(3, len(interval_scale))] = scale
        if end_ratio > 1.0 + tolerance:
            scale = min(1.5, float(np.sqrt(end_ratio) * 1.002))
            interval_scale[max(0, len(interval_scale) - 3) :] = np.maximum(
                interval_scale[max(0, len(interval_scale) - 3) :], scale
            )
        for output_index in np.flatnonzero(output_ratio > 1.0 + tolerance):
            event_time = output_times[output_index + 1]
            path_index = int(np.clip(np.searchsorted(path_times, event_time) - 1, 0, len(duration) - 1))
            scale = min(1.5, float(np.sqrt(output_ratio[output_index]) * 1.002))
            start = max(0, path_index - 2)
            end = min(len(duration), path_index + 3)
            interval_scale[start:end] = np.maximum(interval_scale[start:end], scale)
        duration *= interval_scale
        resample_iterations += 1

    output_qpos_ik_raw = resample_series(
        qpos_ik_raw, path_times, output_times, zero_endpoint_velocity=True
    )
    output_targets = {
        side: resample_linear(targets[side], path_times, output_times) for side in SIDES
    }
    output_raw_targets = {
        side: resample_linear(raw_targets[side], path_times, output_times) for side in SIDES
    }
    output_rotations = {
        side: resample_rotations(rotations[side], path_times, output_times) for side in SIDES
    }
    endpoint_hold_sec = max(0.0, float(settings.get("endpoint_hold_sec", 1.0)))
    hold_frames = int(round(endpoint_hold_sec * render_fps))
    if hold_frames > 0:
        output_qpos = np.concatenate(
            [
                np.repeat(output_qpos[:1], hold_frames, axis=0),
                output_qpos,
                np.repeat(output_qpos[-1:], hold_frames, axis=0),
            ]
        )
        output_qpos_ik_raw = np.concatenate(
            [
                np.repeat(output_qpos_ik_raw[:1], hold_frames, axis=0),
                output_qpos_ik_raw,
                np.repeat(output_qpos_ik_raw[-1:], hold_frames, axis=0),
            ]
        )
        for collection in (output_targets, output_raw_targets, output_rotations):
            for side in SIDES:
                collection[side] = np.concatenate(
                    [
                        np.repeat(collection[side][:1], hold_frames, axis=0),
                        collection[side],
                        np.repeat(collection[side][-1:], hold_frames, axis=0),
                    ]
                )
        motion_times = output_times.copy()
        pre_times = motion_times[0] - np.arange(hold_frames, 0, -1) / render_fps
        post_times = motion_times[-1] + np.arange(1, hold_frames + 1) / render_fps
        output_times = np.concatenate([pre_times, motion_times, post_times])
        output_times -= output_times[0]
        path_times = path_times + hold_frames / render_fps
    after = motion_limit_metrics(
        output_qpos, output_targets, output_rotations, output_times, model
    )
    slowdown = duration / original_dt
    metrics = {
        "enabled": True,
        "shared_bimanual_timeline": True,
        "original_duration_sec": float(times[-1]),
        "retimed_duration_sec": float(output_times[-1]),
        "duration_scale": float(output_times[-1] / max(float(times[-1]), 1e-9)),
        "segment_slowdown_p50": float(np.percentile(slowdown, 50)),
        "segment_slowdown_p95": float(np.percentile(slowdown, 95)),
        "segment_slowdown_max": float(np.max(slowdown)),
        "slowed_segment_count": int(np.count_nonzero(slowdown > 1.0 + tolerance)),
        "path_segment_count": int(len(duration)),
        "acceleration_iterations": iterations,
        "resample_constraint_iterations": resample_iterations,
        "global_safety_scale": global_safety_scale,
        "endpoint_hold_sec_each": endpoint_hold_sec,
        "limits": {
            "tcp_translation_speed_mps": tcp_speed_limit,
            "tcp_angular_speed_rad_s": angular_speed_limit,
            "joint_speed_rad_s": joint_speed_limit,
            "joint_acceleration_rad_s2": joint_acceleration_limit,
        },
        "initial_speed_limiter_segments": {
            "tcp_translation": int(np.count_nonzero(required_translation > original_dt)),
            "tcp_rotation": int(np.count_nonzero(required_rotation > original_dt)),
            "joint_speed": int(np.count_nonzero(required_joint > original_dt)),
        },
        "before": before,
        "after": after,
    }
    return (
        output_qpos,
        output_qpos_ik_raw,
        output_targets,
        output_raw_targets,
        output_rotations,
        output_times,
        path_times,
        metrics,
    )


def body_tree_distance(model: mujoco.MjModel, first: int, second: int) -> int:
    ancestors: dict[int, int] = {}
    depth = 0
    body = first
    while True:
        ancestors[body] = depth
        parent = int(model.body_parentid[body])
        if parent == body:
            break
        body = parent
        depth += 1
    depth = 0
    body = second
    while body not in ancestors:
        parent = int(model.body_parentid[body])
        if parent == body:
            return 10_000
        body = parent
        depth += 1
    return depth + ancestors[body]


def execution_safety_audit(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    times: np.ndarray,
    config: dict[str, Any],
    time_scaling_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Audit the generated command path; this does not modify the trajectory."""
    settings = config.get("safety_audit") or {}
    if not bool(settings.get("enabled", True)):
        return {"enabled": False, "verdict": "NOT_RUN"}
    data = mujoco.MjData(model)
    collision_geoms = [
        index for index in range(model.ngeom) if int(model.geom_group[index]) == 3
    ]

    def body_name(geom: int) -> str:
        return model.body(int(model.geom_bodyid[geom])).name

    def robot_side(geom: int) -> str:
        return body_name(geom).split("_", 1)[0]

    self_pairs: list[tuple[int, int]] = []
    mounting_pairs: list[tuple[int, int]] = []
    interarm_pairs: list[tuple[int, int]] = []
    for offset, first in enumerate(collision_geoms):
        for second in collision_geoms[offset + 1 :]:
            if robot_side(first) != robot_side(second):
                interarm_pairs.append((first, second))
                continue
            first_name = body_name(first)
            second_name = body_name(second)
            names = {first_name, second_name}
            if any(
                name.endswith("wrist_2_link") or name.endswith("wrist_3_link")
                for name in names
            ) and any(
                name.endswith("_omnipicker_camera_link")
                or name.endswith("_omnipicker_base_link")
                or name.endswith("_omnipicker_ur5e_adapter_link")
                for name in names
            ):
                # Flange, adapter and nearby camera shell are a deliberately
                # tight mounting assembly, so audit them under a small
                # penetration tolerance instead of the 20 mm link clearance.
                mounting_pairs.append((first, second))
                continue
            if "_omnipicker_" in first_name and "_omnipicker_" in second_name:
                # Opposing adaptive fingers intentionally approach/contact each
                # other.  Arm-to-gripper and gripper-to-world checks remain on.
                continue
            if body_tree_distance(
                model, int(model.geom_bodyid[first]), int(model.geom_bodyid[second])
            ) > 1:
                self_pairs.append((first, second))
    # Shoulder collision capsules are fixed mount geometry with an intentional
    # 23 mm floor clearance; moving links remain part of the environment audit.
    environment_geoms = [
        geom for geom in collision_geoms if not body_name(geom).endswith("shoulder_link")
    ]
    distance_query_max = float(settings.get("distance_query_max_m", 0.15))
    stride = max(1, int(settings.get("collision_sample_stride", 3)))
    sample_frames = list(range(0, len(qpos), stride))
    if sample_frames[-1] != len(qpos) - 1:
        sample_frames.append(len(qpos) - 1)
    closest: dict[str, dict[str, Any]] = {
        key: {"distance_m": distance_query_max, "frame": 0, "bodies": []}
        for key in ("self", "mounting", "interarm", "environment")
    }
    for frame in sample_frames:
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        for kind, pairs in (
            ("self", self_pairs),
            ("mounting", mounting_pairs),
            ("interarm", interarm_pairs),
        ):
            for first, second in pairs:
                distance = float(
                    mujoco.mj_geomDistance(
                        model, data, first, second, distance_query_max, None
                    )
                )
                if distance < closest[kind]["distance_m"]:
                    closest[kind] = {
                        "distance_m": distance,
                        "frame": frame,
                        "time_sec": float(times[frame]),
                        "bodies": [body_name(first), body_name(second)],
                    }
        for geom in environment_geoms:
            distance = float(
                mujoco.mj_geomDistance(
                    model, data, 0, geom, distance_query_max, None
                )
            )
            if distance < closest["environment"]["distance_m"]:
                closest["environment"] = {
                    "distance_m": distance,
                    "frame": frame,
                    "time_sec": float(times[frame]),
                    "bodies": ["floor", body_name(geom)],
                }

    jacobian_condition_max: dict[str, float] = {}
    jacobian_sigma_min: dict[str, float] = {}
    joint_margin_min_rad: dict[str, float] = {}
    orientation_weight = float(config.get("ik_orientation_weight_m_per_rad", 0.25))
    for side in SIDES:
        qpos_ids, dof_ids = joint_addresses(model, side)
        site_id = end_effector_site_id(model, side)
        conditions: list[float] = []
        minimum_sigmas: list[float] = []
        margins: list[float] = []
        for frame in range(len(qpos)):
            data.qpos[:] = qpos[frame]
            mujoco.mj_forward(model, data)
            jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
            jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
            mujoco.mj_jacSite(
                model, data, jacobian_position, jacobian_rotation, site_id
            )
            singular_values = np.linalg.svd(
                np.vstack(
                    [
                        jacobian_position[:, dof_ids],
                        orientation_weight * jacobian_rotation[:, dof_ids],
                    ]
                ),
                compute_uv=False,
            )
            minimum_sigmas.append(float(singular_values[-1]))
            conditions.append(float(singular_values[0] / max(singular_values[-1], 1e-12)))
            frame_margins: list[float] = []
            for joint_index, suffix in enumerate(JOINT_NAMES):
                joint = model.joint(f"{side}_{suffix}")
                if int(joint.limited[0]):
                    value = qpos[frame, qpos_ids[joint_index]]
                    frame_margins.append(
                        float(min(value - joint.range[0], joint.range[1] - value))
                    )
            if frame_margins:
                margins.append(min(frame_margins))
        jacobian_condition_max[side] = max(conditions)
        jacobian_sigma_min[side] = min(minimum_sigmas)
        joint_margin_min_rad[side] = min(margins) if margins else float("inf")

    if len(times) >= 2:
        dt = np.maximum(np.diff(times), 1e-9)
        active_ids = np.concatenate([joint_addresses(model, side)[0] for side in SIDES])
        velocity = np.diff(qpos[:, active_ids], axis=0) / dt[:, None]
        start_speed = float(np.max(np.abs(velocity[0])))
        end_speed = float(np.max(np.abs(velocity[-1])))
    else:
        start_speed = end_speed = 0.0

    thresholds = {
        "self_clearance_m": float(settings.get("min_self_clearance_m", 0.02)),
        "mounting_clearance_m": float(
            settings.get("min_mounting_clearance_m", -0.0005)
        ),
        "interarm_clearance_m": float(settings.get("min_interarm_clearance_m", 0.05)),
        "environment_clearance_m": float(
            settings.get("min_environment_clearance_m", 0.015)
        ),
        "joint_limit_margin_deg": float(settings.get("min_joint_limit_margin_deg", 10.0)),
        "jacobian_condition": float(settings.get("max_jacobian_condition", 30.0)),
        "endpoint_joint_speed_rad_s": float(
            settings.get("max_endpoint_joint_speed_rad_s", 0.02)
        ),
    }
    violations: list[str] = []
    for kind, threshold_key in (
        ("self", "self_clearance_m"),
        ("mounting", "mounting_clearance_m"),
        ("interarm", "interarm_clearance_m"),
        ("environment", "environment_clearance_m"),
    ):
        if closest[kind]["distance_m"] < thresholds[threshold_key]:
            violations.append(f"{kind}_clearance")
    if min(np.rad2deg(list(joint_margin_min_rad.values()))) < thresholds["joint_limit_margin_deg"]:
        violations.append("joint_limit_margin")
    if max(jacobian_condition_max.values()) > thresholds["jacobian_condition"]:
        violations.append("jacobian_condition")
    if max(start_speed, end_speed) > thresholds["endpoint_joint_speed_rad_s"]:
        violations.append("endpoint_joint_speed")
    limits = (time_scaling_metrics.get("limits") or {})
    measured = (time_scaling_metrics.get("after") or {})
    tolerance = 1.0 + float((config.get("time_scaling") or {}).get("constraint_tolerance", 1e-3))
    for metric, limit in (
        ("tcp_translation_speed_max_mps", "tcp_translation_speed_mps"),
        ("tcp_angular_speed_max_rad_s", "tcp_angular_speed_rad_s"),
        ("joint_speed_max_rad_s", "joint_speed_rad_s"),
        ("joint_acceleration_max_rad_s2", "joint_acceleration_rad_s2"),
    ):
        if metric in measured and limit in limits and measured[metric] > limits[limit] * tolerance:
            violations.append(metric)
    return {
        "enabled": True,
        "verdict": "PASS" if not violations else "FAIL",
        "violations": violations,
        "model_scope": (
            "dual UR5e + AgiBot OmniPicker collision meshes and floor; "
            "internal same-gripper finger contacts excluded; no table, payload, or external obstacles"
            if any("_omnipicker_" in body_name(geom) for geom in collision_geoms)
            else "dual UR5e collision capsules and floor; no gripper, table, payload, or external obstacles"
        ),
        "collision_sample_stride": stride,
        "collision_sample_rate_hz_approx": float(config["render_fps"]) / stride,
        "closest_clearance": closest,
        "joint_limit_margin_min_deg": {
            side: float(np.rad2deg(value)) for side, value in joint_margin_min_rad.items()
        },
        "jacobian_condition_max": jacobian_condition_max,
        "jacobian_sigma_min": jacobian_sigma_min,
        "endpoint_joint_speed_max_rad_s": {"start": start_speed, "end": end_speed},
        "thresholds": thresholds,
    }


GRIPPER_STATES = ("OPEN", "CLOSING", "GRASPED", "OPENING")


def median_filter_1d(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.asarray(
        [np.median(padded[index : index + window]) for index in range(len(values))]
    )


def tactile_contact_confidence(
    rows: list[dict[str, Any]], side: str, settings: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    sensor_count = int(settings.get("tactile_sensor_count", 68))
    raw = np.asarray(
        [
            (row.get("hand_frame") or {}).get(
                f"pressure_{side}", [0.0] * sensor_count
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    if raw.shape != (len(rows), sensor_count):
        return np.zeros(len(rows)), {"available": False, "reason": "invalid_shape"}
    baseline_frames = min(len(raw), int(settings.get("tactile_baseline_frames", 24)))
    baseline = np.nanmedian(raw[:baseline_frames], axis=0)
    baseline = np.nan_to_num(baseline, nan=0.0, posinf=0.0, neginf=0.0)
    noise_gate = float(settings.get("tactile_noise_gate", 1.0))
    rise = float(settings.get("tactile_ema_rise", 0.58))
    fall = float(settings.get("tactile_ema_fall", 0.22))
    threshold = float(settings.get("tactile_contact_threshold", 2.0))
    minimum_value = float(settings.get("tactile_min_filtered_value", 0.35))
    filtered = np.zeros(sensor_count, dtype=np.float64)
    values = np.zeros_like(raw)
    for frame in range(baseline_frames, len(raw)):
        sample = np.where(np.isfinite(raw[frame]), raw[frame], baseline)
        current = np.maximum(0.0, sample - baseline)
        quiet = current <= noise_gate
        current[quiet] = 0.0
        baseline[quiet] = baseline[quiet] * 0.999 + sample[quiet] * 0.001
        alpha = np.where(current >= filtered, rise, fall)
        filtered = filtered * (1.0 - alpha) + current * alpha
        filtered[filtered < minimum_value] = 0.0
        values[frame] = filtered
    contact_count = np.count_nonzero(values > threshold, axis=1)
    strength = np.percentile(values, 90, axis=1)
    confidence = np.clip(
        np.maximum(
            contact_count / max(float(settings.get("tactile_full_contact_sensors", 8)), 1.0),
            strength / max(float(settings.get("tactile_full_contact_strength", 4.0)), 1e-6),
        ),
        0.0,
        1.0,
    )
    return confidence, {
        "available": bool(np.any(np.abs(raw) > 1e-9)),
        "baseline_frames": baseline_frames,
        "contact_frames": int(np.count_nonzero(contact_count > 0)),
        "max_contact_sensor_count": int(np.max(contact_count)),
        "confidence_max": float(np.max(confidence)),
    }


def gripper_state_machine(
    command: np.ndarray, contact: np.ndarray, settings: dict[str, Any]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    close_threshold = float(settings.get("close_threshold", 0.55))
    open_threshold = float(settings.get("open_threshold", 0.30))
    confirmation = max(1, int(settings.get("state_confirmation_frames", 5)))
    contact_confirmation = max(1, int(settings.get("contact_confirmation_frames", 3)))
    states = np.zeros(len(command), dtype=np.int8)
    state = 0
    close_count = open_count = contact_count = 0
    release_count = reclose_count = 0
    grasp_peak = float(command[0]) if len(command) else 0.0
    opening_valley = grasp_peak
    relative_release = float(settings.get("relative_release_delta", 0.25))
    relative_reclose = float(settings.get("relative_reclose_delta", 0.20))
    transitions: list[dict[str, Any]] = []
    for frame, value in enumerate(command):
        close_count = close_count + 1 if value >= close_threshold else 0
        open_count = open_count + 1 if value <= open_threshold else 0
        contact_count = contact_count + 1 if contact[frame] >= 0.20 else 0
        previous = state
        if state == 0 and close_count >= confirmation:
            state = 1
        elif state == 1:
            if open_count >= confirmation:
                state = 3
            elif contact_count >= contact_confirmation or (
                value >= 0.80 and close_count >= confirmation
            ):
                state = 2
                grasp_peak = float(value)
        elif state == 2:
            grasp_peak = max(grasp_peak, float(value))
            relative_open = value <= grasp_peak - relative_release
            release_count = release_count + 1 if relative_open else 0
            if open_count >= confirmation or release_count >= confirmation:
                state = 3
                opening_valley = float(value)
        elif state == 3:
            opening_valley = min(opening_valley, float(value))
            relative_close = value >= opening_valley + relative_reclose
            reclose_count = reclose_count + 1 if relative_close else 0
            if open_count >= confirmation:
                state = 0
            elif reclose_count >= confirmation:
                state = 1
        if state != previous:
            transitions.append(
                {
                    "frame": frame,
                    "from": GRIPPER_STATES[previous],
                    "to": GRIPPER_STATES[state],
                }
            )
            close_count = open_count = contact_count = 0
            release_count = reclose_count = 0
        states[frame] = state
    return states, transitions


def build_gripper_commands(
    rows: list[dict[str, Any]],
    source_times: np.ndarray,
    pre_retime_times: np.ndarray,
    retimed_path_times: np.ndarray,
    output_times: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    settings = config.get("gripper_mapping") or {}
    artifacts: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, Any] = {"enabled": bool(settings.get("enabled", True))}
    if not metrics["enabled"]:
        return artifacts, metrics
    flex_weight = float(settings.get("flex_weight", 0.65))
    pinch_weight = float(settings.get("pinch_weight", 0.25))
    contact_weight = float(settings.get("contact_weight", 0.10))
    total_weight = max(flex_weight + pinch_weight + contact_weight, 1e-9)
    minimum_pinch_span = float(settings.get("minimum_pinch_span_m", 0.008))
    minimum_score_span = float(settings.get("minimum_score_span", 0.15))
    source_dt = float(np.median(np.diff(source_times))) if len(source_times) > 1 else 1 / 30
    maximum_step = float(settings.get("max_normalized_speed_per_sec", 1.5)) * source_dt
    for side in SIDES:
        flexion: list[float] = []
        pinch_by_finger: dict[str, list[float]] = {
            finger: [] for finger in ("index", "middle", "ring", "little")
        }
        for row in rows:
            glove = row["hands"][side]["glove"]
            fingers = glove["solve_state"]["fingers_deg"]
            flexion.append(
                float(
                    np.median(
                        [
                            sum(
                                float(values.get(key, 0.0))
                                for key in ("mcp_flex_deg", "pip_flex_deg", "dip_flex_deg")
                            )
                            for values in fingers.values()
                        ]
                    )
                )
            )
            distances = glove["pinch_distances_m"]
            for finger in pinch_by_finger:
                pinch_by_finger[finger].append(float(distances[finger]))
        flexion_values = np.asarray(flexion)
        flex_low, flex_high = np.percentile(flexion_values, [10, 90])
        flex_score = np.clip(
            (flexion_values - flex_low) / max(float(flex_high - flex_low), 1e-9),
            0.0,
            1.0,
        )
        pinch_scores: list[np.ndarray] = []
        pinch_calibration: dict[str, Any] = {}
        for finger, values in pinch_by_finger.items():
            array = np.asarray(values)
            closed, opened = np.percentile(array, [10, 90])
            span = float(opened - closed)
            valid = span >= minimum_pinch_span
            pinch_calibration[finger] = {
                "closed_p10_m": float(closed),
                "open_p90_m": float(opened),
                "span_m": span,
                "used": valid,
            }
            if valid:
                pinch_scores.append(np.clip((opened - array) / span, 0.0, 1.0))
        pinch_score = (
            np.median(np.stack(pinch_scores), axis=0)
            if pinch_scores
            else np.zeros(len(rows), dtype=np.float64)
        )
        contact_score, tactile_metrics = tactile_contact_confidence(rows, side, settings)
        raw_score = (
            flex_weight * flex_score
            + pinch_weight * pinch_score
            + contact_weight * contact_score
        ) / total_weight
        score_low, score_high = np.percentile(raw_score, [10, 90])
        score_span = float(score_high - score_low)
        active = score_span >= minimum_score_span
        normalized = (
            np.clip((raw_score - score_low) / score_span, 0.0, 1.0)
            if active
            else np.zeros_like(raw_score)
        )
        filtered = median_filter_1d(
            normalized, int(settings.get("median_window_frames", 5))
        )
        alpha = float(settings.get("ema_alpha", 0.25))
        for frame in range(1, len(filtered)):
            filtered[frame] = alpha * filtered[frame] + (1.0 - alpha) * filtered[frame - 1]
        command_source = np.zeros_like(filtered)
        command_source[0] = filtered[0]
        deadband = float(settings.get("command_deadband", 0.01))
        for frame in range(1, len(filtered)):
            delta = filtered[frame] - command_source[frame - 1]
            if abs(delta) <= deadband:
                command_source[frame] = command_source[frame - 1]
            else:
                command_source[frame] = command_source[frame - 1] + np.clip(
                    delta, -maximum_step, maximum_step
                )
        state_source, transitions = gripper_state_machine(
            command_source, contact_score, settings
        )
        command_pre_retime = np.interp(pre_retime_times, source_times, command_source)
        contact_pre_retime = np.interp(pre_retime_times, source_times, contact_score)
        state_pre_retime = np.rint(
            np.interp(pre_retime_times, source_times, state_source)
        ).astype(np.int8)
        command_output = np.interp(
            output_times, retimed_path_times, command_pre_retime
        )
        contact_output = np.interp(
            output_times, retimed_path_times, contact_pre_retime
        )
        nearest_path = np.clip(
            np.searchsorted(retimed_path_times, output_times, side="left"),
            0,
            len(retimed_path_times) - 1,
        )
        state_output = state_pre_retime[nearest_path]
        artifacts[side] = {
            "flex_source": flex_score,
            "pinch_source": pinch_score,
            "contact_source": contact_score,
            "raw_source": raw_score,
            "filtered_source": filtered,
            "command_source": command_source,
            "state_source": state_source,
            "command_pre_retime": command_pre_retime,
            "contact_pre_retime": contact_pre_retime,
            "state_pre_retime": state_pre_retime,
            "command": command_output,
            "contact": contact_output,
            "state": state_output,
        }
        final_speed = (
            np.abs(np.diff(command_output)) / np.maximum(np.diff(output_times), 1e-9)
        )
        metrics[side] = {
            "active": active,
            "score_p10": float(score_low),
            "score_p90": float(score_high),
            "score_span": score_span,
            "flex_calibration_deg": {
                "open_p10": float(flex_low),
                "closed_p90": float(flex_high),
            },
            "pinch_calibration": pinch_calibration,
            "tactile": tactile_metrics,
            "command_min": float(np.min(command_output)),
            "command_max": float(np.max(command_output)),
            "command_speed_max_per_sec": float(np.max(final_speed, initial=0.0)),
            "state_frame_counts_source": {
                name: int(np.count_nonzero(state_source == index))
                for index, name in enumerate(GRIPPER_STATES)
            },
            "transitions_source": [
                {
                    **transition,
                    "time_sec": float(source_times[transition["frame"]]),
                }
                for transition in transitions
            ],
        }
    metrics["parameters"] = {
        "canonical_convention": "0=open, 1=closed",
        "weights": {
            "multi_finger_flex": flex_weight,
            "multi_finger_pinch": pinch_weight,
            "tactile_contact": contact_weight,
        },
        "specific_robot_gripper_mapping": False,
        "state_ids": {name: index for index, name in enumerate(GRIPPER_STATES)},
    }
    return artifacts, metrics


def write_gripper_svg(
    output: Path,
    times: np.ndarray,
    gripper: dict[str, dict[str, np.ndarray]],
) -> None:
    width, height = 1200, 520
    left_margin, right_margin, top, bottom = 70, 30, 55, 55
    plot_width = width - left_margin - right_margin
    panel_height = (height - top - bottom - 35) / 2
    colors = {"left": "#2563eb", "right": "#ea580c"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="70" y="31" font-family="sans-serif" font-size="20" font-weight="700" fill="#0f172a">Canonical gripper command · 0 open / 1 closed</text>',
    ]
    duration = max(float(times[-1]), 1e-9)
    for panel, side in enumerate(SIDES):
        y0 = top + panel * (panel_height + 35)
        parts.append(
            f'<rect x="{left_margin}" y="{y0}" width="{plot_width}" height="{panel_height}" fill="#ffffff" stroke="#cbd5e1"/>'
        )
        for value in (0.0, 0.3, 0.55, 1.0):
            y = y0 + panel_height * (1.0 - value)
            parts.append(
                f'<line x1="{left_margin}" y1="{y:.2f}" x2="{left_margin + plot_width}" y2="{y:.2f}" stroke="#e2e8f0" stroke-width="1"/>'
            )
            parts.append(
                f'<text x="{left_margin - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#64748b">{value:.2f}</text>'
            )
        values = gripper[side]["command"]
        contact = gripper[side]["contact"]
        states = gripper[side]["state"]
        state_colors = ("#94a3b8", "#fbbf24", "#22c55e", "#a78bfa")
        run_start = 0
        for index in range(1, len(states) + 1):
            if index == len(states) or states[index] != states[run_start]:
                x = left_margin + plot_width * float(times[run_start]) / duration
                x2 = left_margin + plot_width * float(times[index - 1]) / duration
                parts.append(
                    f'<rect x="{x:.2f}" y="{y0 + panel_height - 7:.2f}" width="{max(1.0, x2 - x):.2f}" height="6" fill="{state_colors[int(states[run_start])]}" opacity="0.65"/>'
                )
                run_start = index
        contact_points = " ".join(
            f"{left_margin + plot_width * float(t) / duration:.2f},{y0 + panel_height * (1.0 - float(v)):.2f}"
            for t, v in zip(times, contact)
        )
        parts.append(
            f'<polyline points="{contact_points}" fill="none" stroke="#64748b" stroke-width="1.2" stroke-dasharray="4 4" opacity="0.7"/>'
        )
        points = " ".join(
            f"{left_margin + plot_width * float(t) / duration:.2f},{y0 + panel_height * (1.0 - float(v)):.2f}"
            for t, v in zip(times, values)
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[side]}" stroke-width="2.5" stroke-linejoin="round"/>'
        )
        parts.append(
            f'<text x="{left_margin + 12}" y="{y0 + 20}" font-family="sans-serif" font-size="14" font-weight="700" fill="{colors[side]}">{side.upper()}</text>'
        )
    for value in np.linspace(0.0, duration, 6):
        x = left_margin + plot_width * value / duration
        parts.append(
            f'<text x="{x:.2f}" y="{height - 19}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#64748b">{value:.1f}s</text>'
        )
    parts.append('</svg>')
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def set_target_markers(
    model: mujoco.MjModel, data: mujoco.MjData, targets: dict[str, np.ndarray], frame: int
) -> None:
    for side in SIDES:
        mocap_id = int(model.body(f"{side}_target").mocapid[0])
        data.mocap_pos[mocap_id] = targets[side][frame]


def render_video(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    targets: dict[str, np.ndarray],
    output: Path,
    config: dict[str, Any],
    camera_state: dict[str, Any] | None,
) -> None:
    import imageio.v2 as imageio

    output.parent.mkdir(parents=True, exist_ok=True)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(
        model,
        height=int(config["render_height"]),
        width=int(config["render_width"]),
    )
    render_camera: str | mujoco.MjvCamera = "overview"
    if camera_state is not None:
        free_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, free_camera)
        apply_camera_state(free_camera, camera_state)
        render_camera = free_camera
    with imageio.get_writer(
        output,
        fps=float(config["render_fps"]),
        codec="libx264",
        quality=8,
        macro_block_size=None,
    ) as writer:
        for frame in range(len(qpos)):
            data.qpos[:] = qpos[frame]
            set_controls_from_qpos(model, data)
            set_target_markers(model, data, targets, frame)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=render_camera)
            writer.append_data(renderer.render())
    renderer.close()


def play_viewer(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    targets: dict[str, np.ndarray],
    times: np.ndarray,
    camera_state: dict[str, Any] | None,
    camera_path: Path,
    save_camera: bool,
) -> None:
    import mujoco.viewer

    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        if camera_state is not None:
            apply_camera_state(viewer.cam, camera_state)
        try:
            wall_start = time.monotonic()
            for frame in range(len(qpos)):
                deadline = wall_start + float(times[frame])
                while time.monotonic() < deadline:
                    time.sleep(0.001)
                data.qpos[:] = qpos[frame]
                set_controls_from_qpos(model, data)
                set_target_markers(model, data, targets, frame)
                mujoco.mj_forward(model, data)
                viewer.sync()
                if not viewer.is_running():
                    break
        finally:
            if save_camera:
                saved = save_camera_state(camera_path, viewer.cam)
                print("Saved viewer camera:", json.dumps(saved, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--session", help="Postprocess session directory")
    source.add_argument("--trajectory", help="Published trajectory JSONL")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument(
        "--camera_config",
        default=str(Path(__file__).with_name("viewer_camera.json")),
        help="Free-camera JSON loaded by viewer/video and updated when viewer closes",
    )
    parser.add_argument("--output_dir", default=str(Path(__file__).with_name("outputs")))
    parser.add_argument("--name", default=None, help="Output stem; defaults to session name")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--video", action="store_true", help="Render an MP4 (use MUJOCO_GL=egl headlessly)")
    parser.add_argument("--viewer", action="store_true", help="Open the interactive MuJoCo viewer")
    parser.add_argument(
        "--save_viewer_camera",
        action="store_true",
        help="Persist mouse-adjusted viewer camera; off by default to preserve the reference view",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config).expanduser().resolve())
    camera_path = Path(args.camera_config).expanduser().resolve()
    camera_state = load_camera_state(camera_path)
    trajectory_path = resolve_trajectory(args)
    rows = [json.loads(line) for line in trajectory_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_frames > 0:
        rows = rows[: args.max_frames]
    if not rows:
        raise ValueError("empty trajectory")
    source_times = frame_times(rows, float(config["render_fps"]))
    model, data = build_model(config)
    home = np.asarray(config["home_q_rad"], dtype=np.float64)
    initial_sites = {
        side: data.site_xpos[end_effector_site_id(model, side)].copy()
        for side in SIDES
    }
    initial_site_rotations = {
        side: data.site_xmat[end_effector_site_id(model, side)]
        .reshape(3, 3)
        .copy()
        for side in SIDES
    }
    raw_targets, speed_clips = build_targets(rows, source_times, initial_sites, config)
    conditioned_targets, conditioning_metrics = condition_targets(raw_targets, config)
    command_source_targets, static_candidates, static_locked, static_lock_metrics = lock_static_targets(
        conditioned_targets, source_times, config
    )
    command_fps = float((config.get("action_conditioning") or {}).get("command_fps", config["render_fps"]))
    times, targets = resample_targets(command_source_targets, source_times, command_fps)
    _, raw_targets_resampled = resample_targets(raw_targets, source_times, command_fps)
    qpos = np.zeros((len(times), model.nq), dtype=np.float64)
    position_errors = np.zeros((len(times), 2), dtype=np.float64)
    orientation_errors = np.zeros((len(times), 2), dtype=np.float64)
    orientation_mode = str(config.get("ik_orientation_mode", "fixed_initial"))
    if orientation_mode not in {"fixed_initial", "relative_human", "position_only"}:
        raise ValueError(f"unsupported ik_orientation_mode: {orientation_mode}")
    if orientation_mode == "relative_human":
        (
            target_rotations,
            human_rotation_raw,
            human_rotation_repaired,
            human_rotation_conditioned,
            human_rotation_masks,
            orientation_metrics,
        ) = (
            build_relative_orientation_targets(
                rows, source_times, times, initial_site_rotations, config
            )
        )
    else:
        target_rotations = {
            side: np.repeat(initial_site_rotations[side][None, :, :], len(times), axis=0)
            for side in SIDES
        }
        human_rotation_raw = {side: np.zeros((len(source_times), 3)) for side in SIDES}
        human_rotation_conditioned = {
            side: np.zeros((len(source_times), 3)) for side in SIDES
        }
        human_rotation_repaired = {
            side: np.zeros((len(source_times), 3)) for side in SIDES
        }
        human_rotation_masks = {
            key: {side: np.zeros(len(source_times), dtype=bool) for side in SIDES}
            for key in ("spike", "static_candidate", "static_locked")
        }
        orientation_metrics = {}
    velocity_clips = {side: 0 for side in SIDES}
    previous = data.qpos.copy()

    for frame in range(len(times)):
        data.qpos[:] = previous
        for side_index, side in enumerate(SIDES):
            target_rotation = (
                target_rotations[side][frame]
                if orientation_mode != "position_only"
                else None
            )
            position_errors[frame, side_index], orientation_errors[frame, side_index] = solve_pose_ik(
                model,
                data,
                side,
                targets[side][frame],
                target_rotation,
                home,
                config,
            )
        dt = 1.0 / float(config["render_fps"]) if frame == 0 else max(1e-3, float(times[frame] - times[frame - 1]))
        max_delta = float(config["max_joint_speed_rad_s"]) * dt
        for side in SIDES:
            ids, _ = joint_addresses(model, side)
            delta = data.qpos[ids] - previous[ids]
            clipped = np.clip(delta, -max_delta, max_delta)
            if not np.allclose(delta, clipped):
                velocity_clips[side] += 1
            data.qpos[ids] = previous[ids] + clipped
        mujoco.mj_forward(model, data)
        for side_index, side in enumerate(SIDES):
            site_id = end_effector_site_id(model, side)
            position_errors[frame, side_index] = np.linalg.norm(
                targets[side][frame] - data.site_xpos[site_id]
            )
            orientation_errors[frame, side_index] = (
                0.0
                if orientation_mode == "position_only"
                else np.linalg.norm(
                    rotation_error_vector(
                        target_rotations[side][frame],
                        data.site_xmat[site_id].reshape(3, 3),
                    )
                )
            )
        qpos[frame] = data.qpos
        previous = data.qpos.copy()

    qpos_ik_raw = qpos.copy()
    qpos, joint_conditioning_metrics = condition_joint_trajectory(qpos_ik_raw, model, config)
    pre_retime_times = times.copy()
    qpos_pre_retime = qpos.copy()
    qpos_ik_raw_pre_retime = qpos_ik_raw.copy()
    targets_pre_retime = {side: targets[side].copy() for side in SIDES}
    target_rotations_pre_retime = {
        side: target_rotations[side].copy() for side in SIDES
    }
    (
        qpos,
        qpos_ik_raw,
        targets,
        raw_targets_resampled,
        target_rotations,
        times,
        retimed_path_times,
        time_scaling_metrics,
    ) = apply_shared_time_scaling(
        qpos,
        qpos_ik_raw,
        targets,
        raw_targets_resampled,
        target_rotations,
        times,
        model,
        config,
    )
    position_errors = np.zeros((len(times), 2), dtype=np.float64)
    orientation_errors = np.zeros((len(times), 2), dtype=np.float64)
    for frame in range(len(times)):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        for side_index, side in enumerate(SIDES):
            site_id = end_effector_site_id(model, side)
            position_errors[frame, side_index] = np.linalg.norm(
                targets[side][frame] - data.site_xpos[site_id]
            )
            orientation_errors[frame, side_index] = (
                0.0
                if orientation_mode == "position_only"
                else np.linalg.norm(
                    rotation_error_vector(
                        target_rotations[side][frame],
                        data.site_xmat[site_id].reshape(3, 3),
                    )
                )
            )

    gripper_artifacts, gripper_metrics = build_gripper_commands(
        rows,
        source_times,
        pre_retime_times,
        retimed_path_times,
        times,
        config,
    )
    if gripper_artifacts:
        apply_omnipicker_commands(model, qpos, gripper_artifacts, config)
        apply_omnipicker_commands(model, qpos_ik_raw, gripper_artifacts, config)
        robot_gripper = config.get("robot_gripper") or {}
        if bool(robot_gripper.get("enabled", False)):
            gripper_metrics["specific_robot_gripper_mapping"] = True
            (gripper_metrics.get("parameters") or {})[
                "specific_robot_gripper_mapping"
            ] = True
            gripper_metrics["robot_gripper"] = str(
                robot_gripper.get("type", "agibot_omnipicker")
            )
            gripper_metrics["physical_mapping"] = (
                "outer_joint1_rad = open_joint_rad * (1 - canonical_command)"
            )
            gripper_metrics["open_joint_rad"] = float(
                robot_gripper.get("open_joint_rad", np.pi / 4.0)
            )
    safety_audit = execution_safety_audit(
        model, qpos, times, config, time_scaling_metrics
    )
    name = args.name or trajectory_path.parents[2].name
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{name}_dual_ur5e.npz"
    summary_path = output_dir / f"{name}_dual_ur5e_summary.json"
    video_path = output_dir / f"{name}_dual_ur5e.mp4"
    gripper_svg_path = output_dir / f"{name}_gripper_commands.svg"
    if gripper_artifacts:
        write_gripper_svg(gripper_svg_path, times, gripper_artifacts)
    omnipicker_outer_joint: dict[str, np.ndarray] = {}
    for side in SIDES:
        joint_id = int(
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{side}_omnipicker_outer_joint1",
            )
        )
        omnipicker_outer_joint[side] = (
            qpos[:, int(model.jnt_qposadr[joint_id])]
            if joint_id >= 0
            else np.full(len(qpos), np.nan, dtype=np.float64)
        )
    np.savez_compressed(
        npz_path,
        times_sec=times,
        source_times_sec=source_times,
        qpos_rad=qpos,
        qpos_ik_raw_rad=qpos_ik_raw,
        pre_retime_times_sec=pre_retime_times,
        retimed_path_times_sec=retimed_path_times,
        qpos_pre_retime_rad=qpos_pre_retime,
        qpos_ik_raw_pre_retime_rad=qpos_ik_raw_pre_retime,
        left_target_raw_source_m=raw_targets["left"],
        right_target_raw_source_m=raw_targets["right"],
        left_target_conditioned_source_m=conditioned_targets["left"],
        right_target_conditioned_source_m=conditioned_targets["right"],
        left_target_command_source_m=command_source_targets["left"],
        right_target_command_source_m=command_source_targets["right"],
        left_static_candidate_source=static_candidates["left"],
        right_static_candidate_source=static_candidates["right"],
        left_static_locked_source=static_locked["left"],
        right_static_locked_source=static_locked["right"],
        left_target_raw_m=raw_targets_resampled["left"],
        right_target_raw_m=raw_targets_resampled["right"],
        left_target_m=targets["left"],
        right_target_m=targets["right"],
        left_target_pre_retime_m=targets_pre_retime["left"],
        right_target_pre_retime_m=targets_pre_retime["right"],
        ik_error_m=position_errors,
        ik_orientation_error_rad=orientation_errors,
        left_fixed_target_rotation_matrix=initial_site_rotations["left"],
        right_fixed_target_rotation_matrix=initial_site_rotations["right"],
        left_human_relative_rotation_raw_source_rotvec=human_rotation_raw["left"],
        right_human_relative_rotation_raw_source_rotvec=human_rotation_raw["right"],
        left_human_relative_rotation_repaired_source_rotvec=human_rotation_repaired["left"],
        right_human_relative_rotation_repaired_source_rotvec=human_rotation_repaired["right"],
        left_human_relative_rotation_conditioned_source_rotvec=human_rotation_conditioned["left"],
        right_human_relative_rotation_conditioned_source_rotvec=human_rotation_conditioned["right"],
        left_orientation_spike_source=human_rotation_masks["spike"]["left"],
        right_orientation_spike_source=human_rotation_masks["spike"]["right"],
        left_orientation_static_candidate_source=human_rotation_masks["static_candidate"]["left"],
        right_orientation_static_candidate_source=human_rotation_masks["static_candidate"]["right"],
        left_orientation_static_locked_source=human_rotation_masks["static_locked"]["left"],
        right_orientation_static_locked_source=human_rotation_masks["static_locked"]["right"],
        left_target_rotation_matrix=target_rotations["left"],
        right_target_rotation_matrix=target_rotations["right"],
        left_target_rotation_pre_retime_matrix=target_rotations_pre_retime["left"],
        right_target_rotation_pre_retime_matrix=target_rotations_pre_retime["right"],
        left_gripper_flex_source=gripper_artifacts["left"]["flex_source"],
        right_gripper_flex_source=gripper_artifacts["right"]["flex_source"],
        left_gripper_pinch_source=gripper_artifacts["left"]["pinch_source"],
        right_gripper_pinch_source=gripper_artifacts["right"]["pinch_source"],
        left_gripper_contact_source=gripper_artifacts["left"]["contact_source"],
        right_gripper_contact_source=gripper_artifacts["right"]["contact_source"],
        left_gripper_raw_source=gripper_artifacts["left"]["raw_source"],
        right_gripper_raw_source=gripper_artifacts["right"]["raw_source"],
        left_gripper_command_source=gripper_artifacts["left"]["command_source"],
        right_gripper_command_source=gripper_artifacts["right"]["command_source"],
        left_gripper_state_source=gripper_artifacts["left"]["state_source"],
        right_gripper_state_source=gripper_artifacts["right"]["state_source"],
        left_gripper_command=gripper_artifacts["left"]["command"],
        right_gripper_command=gripper_artifacts["right"]["command"],
        left_gripper_contact=gripper_artifacts["left"]["contact"],
        right_gripper_contact=gripper_artifacts["right"]["contact"],
        left_gripper_state=gripper_artifacts["left"]["state"],
        right_gripper_state=gripper_artifacts["right"]["state"],
        left_omnipicker_outer_joint_rad=omnipicker_outer_joint["left"],
        right_omnipicker_outer_joint_rad=omnipicker_outer_joint["right"],
    )
    summary = {
        "trajectory": str(trajectory_path),
        "source_frames": len(rows),
        "frames": len(times),
        "duration_sec": float(times[-1]) if len(times) else 0.0,
        "mapping": (
            "frame0-anchored conditioned wrist translation with scaled relative human wrist orientation"
            if orientation_mode == "relative_human"
            else "frame0-anchored conditioned wrist translation with fixed initial end-effector orientation"
            if orientation_mode == "fixed_initial"
            else "frame0-anchored conditioned wrist translation with position-only IK"
        ),
        "input_coordinate": config["input_coordinate"],
        "translation_scale": config["translation_scale"],
        "command_fps": command_fps,
        "ik_orientation_mode": orientation_mode,
        "human_orientation": orientation_metrics,
        "action_conditioning": conditioning_metrics,
        "static_lock": static_lock_metrics,
        "joint_conditioning": joint_conditioning_metrics,
        "time_scaling": time_scaling_metrics,
        "safety_audit": safety_audit,
        "gripper_mapping": gripper_metrics,
        "robot_gripper": config.get("robot_gripper") or {"enabled": False},
        "target_speed_clipped_frames": speed_clips,
        "joint_speed_clipped_frames": velocity_clips,
        "ik_error_mean_m": dict(zip(SIDES, np.mean(position_errors, axis=0).tolist())),
        "ik_error_p95_m": dict(zip(SIDES, np.percentile(position_errors, 95, axis=0).tolist())),
        "ik_error_max_m": dict(zip(SIDES, np.max(position_errors, axis=0).tolist())),
        "ik_orientation_error_mean_deg": dict(
            zip(SIDES, np.rad2deg(np.mean(orientation_errors, axis=0)).tolist())
        ),
        "ik_orientation_error_p95_deg": dict(
            zip(SIDES, np.rad2deg(np.percentile(orientation_errors, 95, axis=0)).tolist())
        ),
        "ik_orientation_error_max_deg": dict(
            zip(SIDES, np.rad2deg(np.max(orientation_errors, axis=0)).tolist())
        ),
        "npz": str(npz_path),
        "video": str(video_path) if args.video else None,
        "gripper_svg": str(gripper_svg_path) if gripper_artifacts else None,
        "camera_config": str(camera_path),
    }
    if args.video:
        render_video(model, qpos, targets, video_path, config, camera_state)
    if args.viewer:
        play_viewer(
            model,
            qpos,
            targets,
            times,
            camera_state,
            camera_path,
            args.save_viewer_camera,
        )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
