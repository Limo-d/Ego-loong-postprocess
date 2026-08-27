#!/usr/bin/env python3
"""Validate one bimanual target with Mink collision-aware IK.

This is intentionally separate from the replay pipeline.  It proves that
table/torso/inter-arm distances participate in the IK QP before Mink is used
for trajectory generation or base-pose search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mink
import mujoco
import numpy as np

import replay_trajectory as replay


SIDES = ("left", "right")


def robot_envelope_geoms(model: mujoco.MjModel, side: str) -> list[int]:
    output: list[int] = []
    for geom in range(model.ngeom):
        body = model.body(int(model.geom_bodyid[geom])).name
        if (
            body.startswith(f"{side}_")
            and int(model.geom_group[geom]) in {2, 3}
            and body != f"{side}_base"
            and not body.endswith("shoulder_link")
        ):
            output.append(geom)
    return output


def torso_geoms(model: mujoco.MjModel) -> list[int]:
    return [
        geom
        for geom in range(model.ngeom)
        if model.body(int(model.geom_bodyid[geom])).name == "torso_structure"
        and (int(model.geom_contype[geom]) != 0 or int(model.geom_conaffinity[geom]) != 0)
    ]


def table_obstacle_geoms(model: mujoco.MjModel) -> list[int]:
    output: list[int] = []
    for geom_id in range(model.ngeom):
        name = model.geom(geom_id).name
        if name in {"table_top", "table_safety_plane"} or name.startswith(
            "table_leg_"
        ):
            output.append(geom_id)
    return output


def minimum_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    first: list[int],
    second: list[int],
    query_max: float = 0.5,
) -> tuple[float, list[str]]:
    best = query_max
    names: list[str] = []
    for geom_a in first:
        for geom_b in second:
            distance = float(
                mujoco.mj_geomDistance(model, data, geom_a, geom_b, query_max, None)
            )
            if distance < best:
                best = distance
                names = [
                    model.body(int(model.geom_bodyid[geom_a])).name,
                    model.body(int(model.geom_bodyid[geom_b])).name,
                ]
    return best, names


def distance_report(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, object]:
    left = robot_envelope_geoms(model, "left")
    right = robot_envelope_geoms(model, "right")
    structure = torso_geoms(model)
    table = table_obstacle_geoms(model)
    pairs = {
        "left_table": (left, table),
        "right_table": (right, table),
        "interarm": (left, right),
        "left_structure": (left, structure),
        "right_structure": (right, structure),
    }
    output: dict[str, object] = {}
    for name, (first, second) in pairs.items():
        distance, bodies = minimum_distance(model, data, first, second)
        output[name] = {"distance_m": distance, "bodies": bodies}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--minimum_distance_m", type=float, default=0.02)
    parser.add_argument("--detection_distance_m", type=float, default=0.12)
    args = parser.parse_args()

    config = replay.load_config(Path(args.config))
    model, _ = replay.build_model(config)
    payload = np.load(Path(args.npz))
    frame = int(np.clip(args.frame, 0, len(payload["qpos_rad"]) - 1))
    seed = np.asarray(payload["qpos_rad"][frame], dtype=np.float64)
    if seed.shape != (model.nq,):
        raise ValueError(f"seed shape {seed.shape} does not match model nq={model.nq}")

    configuration = mink.Configuration(model, q=seed)
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
        rotation = np.asarray(payload[f"{side}_target_rotation_matrix"][frame])
        position = np.asarray(payload[f"{side}_target_m"][frame])
        task.set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(rotation), position
            )
        )
        tasks.append(task)

    left = robot_envelope_geoms(model, "left")
    right = robot_envelope_geoms(model, "right")
    structure = torso_geoms(model)
    table = table_obstacle_geoms(model)
    collision_pairs = [
        (left, table),
        (right, table),
        (left, right),
        (left, structure),
        (right, structure),
    ]
    limits = [
        mink.ConfigurationLimit(model),
        mink.CollisionAvoidanceLimit(
            model=model,
            geom_pairs=collision_pairs,
            minimum_distance_from_collisions=args.minimum_distance_m,
            collision_detection_distance=args.detection_distance_m,
            gain=0.85,
            broadphase=True,
        ),
    ]

    active_dofs = set()
    for side in SIDES:
        active_dofs.update(replay.joint_addresses(model, side)[1].tolist())
    frozen_dofs = [dof for dof in range(model.nv) if dof not in active_dofs]
    constraints = [mink.DofFreezingTask(model=model, dof_indices=frozen_dofs)]

    before = distance_report(model, configuration.data)
    solver_error: str | None = None
    completed = 0
    for iteration in range(args.iterations):
        try:
            velocity = mink.solve_ik(
                configuration=configuration,
                tasks=tasks,
                dt=args.dt,
                solver="daqp",
                damping=1e-8,
                safety_break=True,
                limits=limits,
                constraints=constraints,
            )
        except Exception as exc:  # Diagnostic tool: preserve exact solver failure.
            solver_error = f"{type(exc).__name__}: {exc}"
            break
        configuration.integrate_inplace(velocity, args.dt)
        completed = iteration + 1

    after = distance_report(model, configuration.data)
    target_errors: dict[str, dict[str, float]] = {}
    for side in SIDES:
        site = replay.end_effector_site_id(model, side)
        position = np.asarray(payload[f"{side}_target_m"][frame])
        rotation = np.asarray(payload[f"{side}_target_rotation_matrix"][frame])
        target_errors[side] = {
            "position_m": float(np.linalg.norm(position - configuration.data.site_xpos[site])),
            "orientation_deg": float(
                np.rad2deg(
                    np.linalg.norm(
                        replay.rotation_error_vector(
                            rotation, configuration.data.site_xmat[site].reshape(3, 3)
                        )
                    )
                )
            ),
        }

    minimum_after = min(
        float(value["distance_m"]) for value in after.values()  # type: ignore[index]
    )
    result = {
        "config": str(Path(args.config).resolve()),
        "npz": str(Path(args.npz).resolve()),
        "frame": frame,
        "iterations_completed": completed,
        "solver_error": solver_error,
        "required_minimum_distance_m": args.minimum_distance_m,
        "before": before,
        "after": after,
        "minimum_after_m": minimum_after,
        "target_errors": target_errors,
        "verdict": (
            "PASS"
            if solver_error is None
            and minimum_after >= args.minimum_distance_m - 1e-5
            and max(value["position_m"] for value in target_errors.values()) <= 0.005
            and max(value["orientation_deg"] for value in target_errors.values()) <= 5.0
            else "FAIL"
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
