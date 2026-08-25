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


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    model_path = Path(config["model_path"])
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    config["model_path"] = str(model_path.resolve())
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
    for side in SIDES:
        child = mujoco.MjSpec.from_file(config["model_path"])
        mount = scene.worldbody.add_frame(
            name=f"{side}_mount",
            pos=config["base_positions_m"][side],
        )
        scene.attach(child, prefix=f"{side}_", frame=mount)
    model = scene.compile()
    data = mujoco.MjData(model)
    home = np.asarray(config["home_q_rad"], dtype=np.float64)
    data.qpos[:] = np.tile(home, 2)
    data.ctrl[:] = data.qpos
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


def resample_series(values: np.ndarray, old_times: np.ndarray, new_times: np.ndarray) -> np.ndarray:
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


def clamp_joint_ranges(model: mujoco.MjModel, qpos: np.ndarray, side: str) -> None:
    for suffix in JOINT_NAMES:
        joint = model.joint(f"{side}_{suffix}")
        if int(joint.limited[0]):
            address = int(joint.qposadr[0])
            qpos[address] = np.clip(qpos[address], joint.range[0], joint.range[1])


def solve_position_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    target: np.ndarray,
    home: np.ndarray,
    config: dict[str, Any],
) -> float:
    site_id = int(model.site(f"{side}_attachment_site").id)
    qpos_ids, dof_ids = joint_addresses(model, side)
    damping = float(config["ik_damping"])
    for _ in range(int(config["ik_iterations"])):
        mujoco.mj_forward(model, data)
        error = target - data.site_xpos[site_id]
        if float(np.linalg.norm(error)) <= float(config["ik_tolerance_m"]):
            break
        jacobian = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacobian, None, site_id)
        jacobian = jacobian[:, dof_ids]
        inverse = np.linalg.solve(
            jacobian @ jacobian.T + np.eye(3) * damping * damping,
            np.eye(3),
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
    return float(np.linalg.norm(target - data.site_xpos[site_id]))


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
            data.ctrl[:] = qpos[frame]
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
                data.ctrl[:] = qpos[frame]
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
        side: data.site_xpos[int(model.site(f"{side}_attachment_site").id)].copy()
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
    errors = np.zeros((len(times), 2), dtype=np.float64)
    velocity_clips = {side: 0 for side in SIDES}
    previous = data.qpos.copy()

    for frame in range(len(times)):
        data.qpos[:] = previous
        for side_index, side in enumerate(SIDES):
            errors[frame, side_index] = solve_position_ik(
                model, data, side, targets[side][frame], home, config
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
            site_id = int(model.site(f"{side}_attachment_site").id)
            errors[frame, side_index] = np.linalg.norm(targets[side][frame] - data.site_xpos[site_id])
        qpos[frame] = data.qpos
        previous = data.qpos.copy()

    name = args.name or trajectory_path.parents[2].name
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{name}_dual_ur5e.npz"
    summary_path = output_dir / f"{name}_dual_ur5e_summary.json"
    video_path = output_dir / f"{name}_dual_ur5e.mp4"
    np.savez_compressed(
        npz_path,
        times_sec=times,
        source_times_sec=source_times,
        qpos_rad=qpos,
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
        ik_error_m=errors,
    )
    summary = {
        "trajectory": str(trajectory_path),
        "source_frames": len(rows),
        "frames": len(times),
        "duration_sec": float(times[-1]) if len(times) else 0.0,
        "mapping": "frame0-anchored, conditioned and static-locked wrist translation only",
        "input_coordinate": config["input_coordinate"],
        "translation_scale": config["translation_scale"],
        "command_fps": command_fps,
        "action_conditioning": conditioning_metrics,
        "static_lock": static_lock_metrics,
        "target_speed_clipped_frames": speed_clips,
        "joint_speed_clipped_frames": velocity_clips,
        "ik_error_mean_m": dict(zip(SIDES, np.mean(errors, axis=0).tolist())),
        "ik_error_p95_m": dict(zip(SIDES, np.percentile(errors, 95, axis=0).tolist())),
        "ik_error_max_m": dict(zip(SIDES, np.max(errors, axis=0).tolist())),
        "npz": str(npz_path),
        "video": str(video_path) if args.video else None,
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
