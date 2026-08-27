#!/usr/bin/env python3
"""Test difficult bimanual keyframes with deterministic multi-start Mink IK."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mink
import mujoco
import numpy as np

import replay_trajectory as replay
import validate_mink_single_frame as single


SIDES = replay.SIDES


def parse_frames(value: str, length: int) -> list[int]:
    output = {int(np.clip(int(item), 0, length - 1)) for item in value.split(",") if item.strip()}
    return sorted(output)


def active_dofs(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.concatenate([replay.joint_addresses(model, side)[0] for side in SIDES])
    dofs = np.concatenate([replay.joint_addresses(model, side)[1] for side in SIDES])
    return qpos, dofs


def target_errors(
    model: mujoco.MjModel, data: mujoco.MjData, payload: np.lib.npyio.NpzFile, frame: int
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for side in SIDES:
        site = replay.end_effector_site_id(model, side)
        position = np.asarray(payload[f"{side}_target_m"][frame])
        rotation = np.asarray(payload[f"{side}_target_rotation_matrix"][frame])
        output[side] = {
            "position_m": float(np.linalg.norm(position - data.site_xpos[site])),
            "orientation_deg": float(
                np.rad2deg(
                    np.linalg.norm(
                        replay.rotation_error_vector(
                            rotation, data.site_xmat[site].reshape(3, 3)
                        )
                    )
                )
            ),
        }
    return output


def make_seed(
    model: mujoco.MjModel,
    config: dict,
    payload: np.lib.npyio.NpzFile,
    frame: int,
    start: int,
    rng: np.random.Generator,
    noise_rad: float,
) -> np.ndarray:
    seed = np.asarray(payload["qpos_rad"][frame], dtype=np.float64).copy()
    if start == 0:
        for side in SIDES:
            seed[replay.joint_addresses(model, side)[0]] = replay.home_q_for_side(config, side)
    elif start >= 2:
        for side in SIDES:
            ids = replay.joint_addresses(model, side)[0]
            seed[ids] = replay.home_q_for_side(config, side) + rng.normal(0.0, noise_rad, len(ids))
    return seed


def solve_attempt(
    config: dict,
    payload: np.lib.npyio.NpzFile,
    frame: int,
    seed: np.ndarray,
    iterations: int,
    dt: float,
    minimum_distance: float,
    detection_distance: float,
) -> dict:
    model, data = replay.build_model(config)
    data.qpos[:] = seed
    mujoco.mj_forward(model, data)

    # Use the start itself as the DLS regularization reference so different
    # starts can reach different elbow/wrist branches before Mink refinement.
    proposal_config = copy.deepcopy(config)
    proposal_config["home_q_rad_by_side"] = {
        side: data.qpos[replay.joint_addresses(model, side)[0]].tolist() for side in SIDES
    }
    for side in SIDES:
        replay.solve_pose_ik(
            model,
            data,
            side,
            np.asarray(payload[f"{side}_target_m"][frame]),
            np.asarray(payload[f"{side}_target_rotation_matrix"][frame]),
            replay.home_q_for_side(proposal_config, side),
            proposal_config,
        )
    mujoco.mj_forward(model, data)
    proposal = {
        "distances": single.distance_report(model, data),
        "target_errors": target_errors(model, data, payload, frame),
    }

    configuration = mink.Configuration(model, q=data.qpos.copy())
    tasks: list[mink.FrameTask] = []
    for side in SIDES:
        task = mink.FrameTask(
            frame_name=model.site(replay.end_effector_site_id(model, side)).name,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.25,
            gain=0.85,
            lm_damping=1e-4,
        )
        task.set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(np.asarray(payload[f"{side}_target_rotation_matrix"][frame])),
                np.asarray(payload[f"{side}_target_m"][frame]),
            )
        )
        tasks.append(task)

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
            minimum_distance_from_collisions=minimum_distance,
            collision_detection_distance=detection_distance,
            gain=0.85,
            broadphase=True,
        ),
    ]
    _, movable_dofs = active_dofs(model)
    movable = set(movable_dofs.tolist())
    constraints = [
        mink.DofFreezingTask(
            model=model,
            dof_indices=[index for index in range(model.nv) if index not in movable],
        )
    ]
    solver_error = None
    completed = 0
    for iteration in range(iterations):
        try:
            velocity = mink.solve_ik(
                configuration=configuration,
                tasks=tasks,
                dt=dt,
                solver="daqp",
                damping=1e-8,
                safety_break=True,
                limits=limits,
                constraints=constraints,
            )
        except Exception as exc:
            solver_error = f"{type(exc).__name__}: {exc}"
            break
        configuration.integrate_inplace(velocity, dt)
        completed = iteration + 1

    after = single.distance_report(model, configuration.data)
    errors = target_errors(model, configuration.data, payload, frame)
    audit = replay.execution_safety_audit(
        model,
        np.stack([configuration.q, configuration.q]),
        np.asarray([0.0, 100.0]),
        config,
        {},
    )
    minimum_after = min(float(value["distance_m"]) for value in after.values())
    passed = (
        solver_error is None
        and minimum_after >= minimum_distance - 1e-5
        and max(value["position_m"] for value in errors.values()) <= 0.005
        and max(value["orientation_deg"] for value in errors.values()) <= 5.0
        and audit["verdict"] == "PASS"
    )
    return {
        "verdict": "PASS" if passed else "FAIL",
        "iterations_completed": completed,
        "solver_error": solver_error,
        "proposal": proposal,
        "after": after,
        "minimum_after_m": minimum_after,
        "target_errors": errors,
        "audit_verdict": audit["verdict"],
        "audit_violations": audit.get("violations", []),
        "qpos_rad": configuration.q.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--frames", required=True, help="Comma-separated retimed frame indices")
    parser.add_argument("--starts", type=int, default=12)
    parser.add_argument("--random_seed", type=int, default=7)
    parser.add_argument("--noise_rad", type=float, default=0.55)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--minimum_distance_m", type=float, default=0.02)
    parser.add_argument("--detection_distance_m", type=float, default=0.12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    npz_path = Path(args.npz).expanduser().resolve()
    config = replay.load_config(config_path)
    payload = np.load(npz_path)
    model, _ = replay.build_model(config)
    frames = parse_frames(args.frames, len(payload["qpos_rad"]))
    rng = np.random.default_rng(args.random_seed)
    results: dict[str, dict] = {}
    for frame in frames:
        attempts: list[dict] = []
        for start in range(args.starts):
            seed = make_seed(model, config, payload, frame, start, rng, args.noise_rad)
            result = solve_attempt(
                config,
                payload,
                frame,
                seed,
                args.iterations,
                args.dt,
                args.minimum_distance_m,
                args.detection_distance_m,
            )
            result["start"] = start
            attempts.append(result)
            print(
                f"frame={frame} start={start:02d} {result['verdict']} "
                f"clearance={result['minimum_after_m']:.4f}"
            )
        attempts.sort(
            key=lambda value: (
                value["verdict"] == "PASS",
                value["minimum_after_m"],
                -max(item["position_m"] for item in value["target_errors"].values()),
            ),
            reverse=True,
        )
        results[str(frame)] = {
            "pass_count": sum(item["verdict"] == "PASS" for item in attempts),
            "attempt_count": len(attempts),
            "best": attempts[0],
            "attempts": attempts,
        }

    report = {
        "config": str(config_path),
        "npz": str(npz_path),
        "frames": frames,
        "starts_per_frame": args.starts,
        "minimum_distance_m": args.minimum_distance_m,
        "all_frames_feasible": all(value["pass_count"] > 0 for value in results.values()),
        "results": results,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output), "all_frames_feasible": report["all_frames_feasible"], "pass_counts": {key: value["pass_count"] for key, value in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
