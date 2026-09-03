#!/usr/bin/env python3
"""Re-anchor an existing LingLong replay and resolve it from a new home branch."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import replay_trajectory as linglong


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_npz", required=True)
    parser.add_argument("--output_npz", required=True)
    args = parser.parse_args()

    replay = linglong.replay
    replay.JOINT_NAMES = linglong.ARM_JOINT_NAMES
    replay.home_q_for_side = linglong.home_q_for_side
    replay.end_effector_site_id = linglong.end_effector_site_id
    config = replay.load_config(Path(args.config).expanduser().resolve())
    model, data = linglong.build_model(config)

    source = np.load(Path(args.input_npz).expanduser().resolve())
    arrays = {key: np.array(source[key], copy=True) for key in source.files}
    for side in replay.SIDES:
        anchor = np.asarray(config["task_anchor_positions_m"][side], dtype=np.float64)
        shift = anchor - np.asarray(source[f"{side}_target_m"])[0]
        for key, values in tuple(arrays.items()):
            if (
                key.startswith(f"{side}_target")
                and key.endswith("_m")
                and values.ndim >= 2
                and values.shape[-1] == 3
            ):
                arrays[key] = values + shift

    homes = {
        side: linglong.home_q_for_side(config, side) for side in replay.SIDES
    }
    frame_count = len(arrays["times_sec"])
    qpos = np.empty((frame_count, model.nq), dtype=np.float64)
    errors = np.empty((frame_count, len(replay.SIDES)), dtype=np.float64)
    rotations = {
        side: np.empty((frame_count, 3, 3), dtype=np.float64)
        for side in replay.SIDES
    }
    for frame in range(frame_count):
        for side_index, side in enumerate(replay.SIDES):
            target = arrays[f"{side}_target_m"][frame]
            replay.solve_pose_ik(model, data, side, target, None, homes[side], config)
            replay.set_omnipicker_qpos(
                model,
                data.qpos,
                side,
                float(arrays[f"{side}_gripper_position_command"][frame]),
                config,
            )
            linglong.mujoco.mj_forward(model, data)
            site_id = replay.end_effector_site_id(model, side)
            errors[frame, side_index] = np.linalg.norm(
                target - data.site_xpos[site_id]
            )
            rotations[side][frame] = data.site_xmat[site_id].reshape(3, 3)
        qpos[frame] = data.qpos
        if frame % 500 == 0 or frame + 1 == frame_count:
            print(f"solved {frame + 1}/{frame_count}", flush=True)

    arrays["qpos_rad"] = qpos
    arrays["qpos_ik_raw_rad"] = qpos.copy()
    arrays["ik_error_m"] = errors
    arrays["ik_orientation_error_rad"] = np.zeros_like(errors)
    for side in replay.SIDES:
        arrays[f"{side}_target_rotation_matrix"] = rotations[side]
        arrays[f"{side}_fixed_target_rotation_matrix"] = rotations[side][0]
    output = Path(args.output_npz).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(f"output={output}")
    print(f"maximum_ik_error_m={float(np.max(errors)):.9f}")


if __name__ == "__main__":
    main()
