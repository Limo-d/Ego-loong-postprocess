#!/usr/bin/env python3
"""Solve a continuous dual-UR5e trajectory with collision-aware Mink IK."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mink
import mujoco
import numpy as np

import replay_trajectory as replay
import validate_mink_multistart as multistart
import validate_mink_single_frame as single


SIDES = replay.SIDES


def set_targets(
    tasks: dict[str, mink.FrameTask], payload: dict[str, np.ndarray], frame: int
) -> None:
    for side in SIDES:
        tasks[side].set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(payload[f"{side}_target_rotation_matrix"][frame]),
                payload[f"{side}_target_m"][frame],
            )
        )


def errors(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    payload: dict[str, np.ndarray],
    frame: int,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for side in SIDES:
        site = replay.end_effector_site_id(model, side)
        output[side] = {
            "position_m": float(
                np.linalg.norm(payload[f"{side}_target_m"][frame] - data.site_xpos[site])
            ),
            "orientation_deg": float(
                np.rad2deg(
                    np.linalg.norm(
                        replay.rotation_error_vector(
                            payload[f"{side}_target_rotation_matrix"][frame],
                            data.site_xmat[site].reshape(3, 3),
                        )
                    )
                )
            ),
        }
    return output


def frame_is_valid(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    payload: dict[str, np.ndarray],
    frame: int,
    minimum_distance: float,
    position_tolerance: float,
    orientation_tolerance_deg: float,
) -> tuple[bool, float, dict[str, dict[str, float]]]:
    frame_errors = errors(model, data, payload, frame)
    distances = single.distance_report(model, data)
    minimum = min(float(value["distance_m"]) for value in distances.values())
    valid = (
        minimum >= minimum_distance - 1e-5
        and max(value["position_m"] for value in frame_errors.values())
        <= position_tolerance
        and max(value["orientation_deg"] for value in frame_errors.values())
        <= orientation_tolerance_deg
    )
    return valid, minimum, frame_errors


def resample_payload_arrays(
    payload: dict[str, np.ndarray],
    path_times: np.ndarray,
    output_times: np.ndarray,
) -> dict[str, np.ndarray]:
    """Resample arrays aligned to the input command timeline."""
    output = dict(payload)
    input_length = len(path_times)
    nearest = np.clip(
        np.searchsorted(path_times, output_times, side="left"), 0, input_length - 1
    )
    previous = np.maximum(nearest - 1, 0)
    choose_previous = np.abs(output_times - path_times[previous]) <= np.abs(
        path_times[nearest] - output_times
    )
    nearest = np.where(choose_previous, previous, nearest)
    for key, values in payload.items():
        if values.ndim == 0 or len(values) != input_length:
            continue
        if values.dtype.kind in {"b", "i", "u", "S", "U", "O"}:
            output[key] = values[nearest]
        else:
            output[key] = replay.resample_linear(values, path_times, output_times)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--output_npz", required=True)
    parser.add_argument("--output_summary", required=True)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--iterations_per_frame", type=int, default=30)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--minimum_distance_m", type=float, default=0.02)
    parser.add_argument("--detection_distance_m", type=float, default=0.12)
    parser.add_argument("--position_tolerance_m", type=float, default=0.005)
    parser.add_argument("--orientation_tolerance_deg", type=float, default=5.0)
    parser.add_argument("--early_position_tolerance_m", type=float, default=0.0002)
    parser.add_argument("--early_orientation_tolerance_deg", type=float, default=0.2)
    parser.add_argument("--recovery_starts", type=int, default=8)
    parser.add_argument("--random_seed", type=int, default=17)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    source_path = Path(args.npz).expanduser().resolve()
    config = replay.load_config(config_path)
    with np.load(source_path) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    length = len(payload["qpos_rad"])
    if args.max_frames > 0:
        length = min(length, args.max_frames)

    model, data = replay.build_model(config)
    source_qpos = payload["qpos_rad"][:length]
    if source_qpos.shape[1] != model.nq:
        raise ValueError(f"source nq={source_qpos.shape[1]} does not match model nq={model.nq}")
    configuration = mink.Configuration(model, q=data.qpos.copy())
    tasks = {
        side: mink.FrameTask(
            frame_name=model.site(replay.end_effector_site_id(model, side)).name,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.25,
            gain=0.85,
            lm_damping=1e-4,
        )
        for side in SIDES
    }
    left = single.robot_envelope_geoms(model, "left")
    right = single.robot_envelope_geoms(model, "right")
    table = single.table_obstacle_geoms(model)
    structure = single.torso_geoms(model)
    limits = [
        mink.ConfigurationLimit(model),
        mink.CollisionAvoidanceLimit(
            model=model,
            geom_pairs=[
                (left, table),
                (right, table),
                (left, right),
                (left, structure),
                (right, structure),
            ],
            minimum_distance_from_collisions=args.minimum_distance_m,
            collision_detection_distance=args.detection_distance_m,
            gain=0.85,
            broadphase=True,
        ),
    ]
    active_qpos, active_dofs = multistart.active_dofs(model)
    active_qpos_set = set(active_qpos.tolist())
    active_dof_set = set(active_dofs.tolist())
    frozen_qpos = [index for index in range(model.nq) if index not in active_qpos_set]
    constraints = [
        mink.DofFreezingTask(
            model=model,
            dof_indices=[index for index in range(model.nv) if index not in active_dof_set],
        )
    ]

    qpos = np.empty((length, model.nq), dtype=np.float64)
    position_errors = np.empty((length, len(SIDES)), dtype=np.float64)
    orientation_errors_deg = np.empty((length, len(SIDES)), dtype=np.float64)
    clearances = np.empty(length, dtype=np.float64)
    iteration_counts = np.zeros(length, dtype=np.int32)
    recovery_frames: list[int] = []
    failed_frames: list[int] = []
    rng = np.random.default_rng(args.random_seed)

    for frame in range(length):
        working = configuration.q.copy()
        working[frozen_qpos] = source_qpos[frame, frozen_qpos]
        configuration.update(q=working)
        set_targets(tasks, payload, frame)
        solver_error = None
        for iteration in range(args.iterations_per_frame):
            try:
                velocity = mink.solve_ik(
                    configuration=configuration,
                    tasks=list(tasks.values()),
                    dt=args.dt,
                    solver="daqp",
                    damping=1e-8,
                    safety_break=True,
                    limits=limits,
                    constraints=constraints,
                )
            except Exception as exc:
                solver_error = f"{type(exc).__name__}: {exc}"
                break
            configuration.integrate_inplace(velocity, args.dt)
            iteration_counts[frame] = iteration + 1
            if iteration >= 2:
                converged, _, _ = frame_is_valid(
                    model,
                    configuration.data,
                    payload,
                    frame,
                    args.minimum_distance_m,
                    args.early_position_tolerance_m,
                    args.early_orientation_tolerance_deg,
                )
                if converged:
                    break

        valid, clearance, frame_errors = frame_is_valid(
            model,
            configuration.data,
            payload,
            frame,
            args.minimum_distance_m,
            args.position_tolerance_m,
            args.orientation_tolerance_deg,
        )
        if not valid or solver_error is not None:
            previous = qpos[frame - 1].copy() if frame > 0 else configuration.q.copy()
            candidates: list[tuple[float, np.ndarray]] = []
            for start in range(args.recovery_starts):
                seed = source_qpos[frame].copy()
                if start == 0:
                    seed[active_qpos] = previous[active_qpos]
                elif start == 1:
                    for side in SIDES:
                        ids = replay.joint_addresses(model, side)[0]
                        seed[ids] = replay.home_q_for_side(config, side)
                else:
                    for side in SIDES:
                        ids = replay.joint_addresses(model, side)[0]
                        seed[ids] = replay.home_q_for_side(config, side) + rng.normal(
                            0.0, 0.55, len(ids)
                        )
                result = multistart.solve_attempt(
                    config,
                    payload,  # type: ignore[arg-type]
                    frame,
                    seed,
                    250,
                    args.dt,
                    args.minimum_distance_m,
                    args.detection_distance_m,
                )
                if result["verdict"] == "PASS":
                    candidate = np.asarray(result["qpos_rad"], dtype=np.float64)
                    jump = float(np.linalg.norm(candidate[active_qpos] - previous[active_qpos]))
                    candidates.append((jump, candidate))
            if candidates:
                candidates.sort(key=lambda item: item[0])
                configuration.update(q=candidates[0][1])
                valid, clearance, frame_errors = frame_is_valid(
                    model,
                    configuration.data,
                    payload,
                    frame,
                    args.minimum_distance_m,
                    args.position_tolerance_m,
                    args.orientation_tolerance_deg,
                )
                recovery_frames.append(frame)
            if not valid:
                failed_frames.append(frame)

        qpos[frame] = configuration.q
        clearances[frame] = clearance
        for side_index, side in enumerate(SIDES):
            position_errors[frame, side_index] = frame_errors[side]["position_m"]
            orientation_errors_deg[frame, side_index] = frame_errors[side]["orientation_deg"]
        if frame % 100 == 0 or frame == length - 1:
            print(
                f"frame={frame}/{length - 1} clearance={clearance:.4f} "
                f"recoveries={len(recovery_frames)} failures={len(failed_frames)}"
            )

    input_times = payload["times_sec"][:length]
    mink_qpos = qpos.copy()
    mink_clearances = clearances.copy()
    qpos, conditioning_metrics = replay.condition_joint_trajectory(
        mink_qpos, model, config
    )
    targets = {
        side: payload[f"{side}_target_m"][:length].copy() for side in SIDES
    }
    raw_targets = {
        side: (
            payload[f"{side}_target_raw_m"][:length].copy()
            if f"{side}_target_raw_m" in payload
            and len(payload[f"{side}_target_raw_m"]) >= length
            else targets[side].copy()
        )
        for side in SIDES
    }
    rotations = {
        side: payload[f"{side}_target_rotation_matrix"][:length].copy()
        for side in SIDES
    }
    retime_config = copy.deepcopy(config)
    retime_config.setdefault("time_scaling", {})["endpoint_hold_sec"] = 0.0
    (
        qpos,
        mink_qpos_retimed,
        targets,
        raw_targets,
        rotations,
        times,
        retimed_path_times,
        time_scaling_metrics,
    ) = replay.apply_shared_time_scaling(
        qpos,
        mink_qpos,
        targets,
        raw_targets,
        rotations,
        input_times,
        model,
        retime_config,
    )

    final_payload = {
        **payload,
        **{f"{side}_target_m": targets[side] for side in SIDES},
        **{f"{side}_target_rotation_matrix": rotations[side] for side in SIDES},
    }
    final_clearances = np.empty(len(qpos), dtype=np.float64)
    final_position_errors = np.empty((len(qpos), len(SIDES)), dtype=np.float64)
    final_orientation_errors_deg = np.empty((len(qpos), len(SIDES)), dtype=np.float64)
    final_failed_frames: list[int] = []
    for frame in range(len(qpos)):
        data.qpos[:] = qpos[frame]
        mujoco.mj_forward(model, data)
        valid, clearance, frame_errors = frame_is_valid(
            model,
            data,
            final_payload,
            frame,
            args.minimum_distance_m,
            args.position_tolerance_m,
            args.orientation_tolerance_deg,
        )
        final_clearances[frame] = clearance
        if not valid:
            final_failed_frames.append(frame)
        for side_index, side in enumerate(SIDES):
            final_position_errors[frame, side_index] = frame_errors[side]["position_m"]
            final_orientation_errors_deg[frame, side_index] = frame_errors[side][
                "orientation_deg"
            ]

    audit = replay.execution_safety_audit(
        model, qpos, times, config, time_scaling_metrics
    )
    if len(qpos) > 1:
        joint_steps = np.linalg.norm(np.diff(qpos[:, active_qpos], axis=0), axis=1)
        max_joint_step = float(np.max(joint_steps))
        dt = np.maximum(np.diff(times), 1e-9)
        velocity = np.diff(qpos[:, active_qpos], axis=0) / dt[:, None]
        acceleration = np.diff(velocity, axis=0) / (
            0.5 * (dt[:-1] + dt[1:])
        )[:, None]
        jerk = np.diff(acceleration, axis=0) / dt[1:-1, None]
    else:
        max_joint_step = 0.0
        velocity = np.empty((0, len(active_qpos)))
        acceleration = np.empty((0, len(active_qpos)))
        jerk = np.empty((0, len(active_qpos)))
    motion_metrics = {
        "joint_speed_max_rad_s": float(np.max(np.abs(velocity), initial=0.0)),
        "joint_acceleration_max_rad_s2": float(
            np.max(np.abs(acceleration), initial=0.0)
        ),
        "joint_jerk_p95_rad_s3": float(
            np.percentile(np.abs(jerk), 95) if jerk.size else 0.0
        ),
        "joint_jerk_max_rad_s3": float(np.max(np.abs(jerk), initial=0.0)),
    }
    summary = {
        "config": str(config_path),
        "source_npz": str(source_path),
        "input_frame_count": length,
        "frame_count": len(qpos),
        "verdict": (
            "PASS"
            if not failed_frames
            and not final_failed_frames
            and audit["verdict"] == "PASS"
            else "FAIL"
        ),
        "mink_failed_frames": failed_frames,
        "final_failed_frames": final_failed_frames,
        "recovery_frames": recovery_frames,
        "mink_minimum_clearance_m": float(np.min(mink_clearances)),
        "minimum_clearance_m": float(np.min(final_clearances)),
        "minimum_clearance_frame": int(np.argmin(final_clearances)),
        "position_error_max_m": float(np.max(final_position_errors)),
        "orientation_error_max_deg": float(np.max(final_orientation_errors_deg)),
        "max_active_joint_step_rad": max_joint_step,
        "iterations_mean": float(np.mean(iteration_counts)),
        "joint_conditioning": conditioning_metrics,
        "time_scaling": time_scaling_metrics,
        "motion_metrics": motion_metrics,
        "safety_audit": audit,
    }
    source_command_length = len(payload["times_sec"])
    aligned_payload = {
        key: (
            values[:length]
            if values.ndim > 0 and len(values) == source_command_length
            else values
        )
        for key, values in payload.items()
    }
    output_arrays = resample_payload_arrays(
        aligned_payload, retimed_path_times, times
    )
    output_arrays.update(
        qpos_rad=qpos,
        qpos_ik_raw_rad=mink_qpos_retimed,
        times_sec=times,
        retimed_path_times_sec=retimed_path_times,
        left_target_raw_m=raw_targets["left"],
        right_target_raw_m=raw_targets["right"],
        left_target_m=targets["left"],
        right_target_m=targets["right"],
        left_target_rotation_matrix=rotations["left"],
        right_target_rotation_matrix=rotations["right"],
        ik_error_m=final_position_errors,
        ik_orientation_error_rad=np.deg2rad(final_orientation_errors_deg),
        mink_clearance_m=final_clearances,
        mink_clearance_pre_conditioning_m=mink_clearances,
        mink_qpos_pre_conditioning_rad=mink_qpos,
    )
    output_npz = Path(args.output_npz).expanduser().resolve()
    output_summary = Path(args.output_summary).expanduser().resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **output_arrays)
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "npz": str(output_npz),
                "summary": str(output_summary),
                **{
                    key: summary[key]
                    for key in (
                        "verdict",
                        "frame_count",
                        "minimum_clearance_m",
                        "minimum_clearance_frame",
                        "recovery_frames",
                        "final_failed_frames",
                        "motion_metrics",
                    )
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
