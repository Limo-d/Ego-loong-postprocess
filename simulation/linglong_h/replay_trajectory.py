#!/usr/bin/env python3
"""Replay optimized bimanual wrist trajectories on the LingLong-H robot."""

from __future__ import annotations

import importlib.util
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
DUAL_UR5E_REPLAY = HERE.parent / "dual_ur5e" / "replay_trajectory.py"
REPLAY_SPEC = importlib.util.spec_from_file_location("dual_ur5e_replay", DUAL_UR5E_REPLAY)
if REPLAY_SPEC is None or REPLAY_SPEC.loader is None:
    raise ImportError(f"cannot load shared replay module: {DUAL_UR5E_REPLAY}")
replay = importlib.util.module_from_spec(REPLAY_SPEC)
REPLAY_SPEC.loader.exec_module(replay)

mujoco = replay.mujoco

ARM_JOINT_NAMES = (
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_joint",
    "wrist_roll_joint",
    "wrist_pitch_joint",
    "wrist_yaw_joint",
)
WAIST_JOINT_NAMES = (
    "waist_1_joint",
    "waist_2_joint",
    "waist_3_joint",
    "waist_yaw_joint",
)


def home_q_for_side(config: dict[str, Any], side: str) -> np.ndarray:
    values = (config.get("home_q_rad_by_side") or {}).get(side)
    if values is None:
        raise ValueError(f"home_q_rad_by_side.{side} is required")
    home = np.asarray(values, dtype=np.float64)
    if home.shape != (len(ARM_JOINT_NAMES),):
        raise ValueError(
            f"home_q_rad_by_side.{side} must have {len(ARM_JOINT_NAMES)} values"
        )
    return home


def _add_table(scene: Any, config: dict[str, Any]) -> None:
    table = config.get("table_geometry") or {}
    if not bool(table.get("enabled", False)):
        return
    length = float(table.get("length_m", 1.20))
    width = float(table.get("width_m", 0.60))
    thickness = float(table.get("thickness_m", 0.03))
    top_height = float(table.get("top_height_m", 0.75))
    center_xy = np.asarray(table.get("center_xy_m", [0.0, 0.55]), dtype=np.float64)
    if center_xy.shape != (2,) or min(length, width, thickness, top_height) <= 0.0:
        raise ValueError("invalid table_geometry dimensions")
    table_body = scene.worldbody.add_body(name="table_structure")
    table_body.add_geom(
        name="table_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[float(center_xy[0]), float(center_xy[1]), top_height - thickness / 2.0],
        size=[length / 2.0, width / 2.0, thickness / 2.0],
        rgba=table.get("rgba", [0.58, 0.61, 0.65, 1.0]),
        contype=1,
        conaffinity=1,
    )
    if not bool(table.get("legs_enabled", True)):
        return
    leg_size = np.asarray(table.get("leg_size_m", [0.05, 0.05]), dtype=np.float64)
    leg_inset = np.asarray(table.get("leg_inset_m", [0.08, 0.08]), dtype=np.float64)
    underside = top_height - thickness
    leg_half_height = underside / 2.0
    leg_x = length / 2.0 - leg_inset[0] - leg_size[0] / 2.0
    leg_y = width / 2.0 - leg_inset[1] - leg_size[1] / 2.0
    for index, (sign_x, sign_y) in enumerate(
        ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0))
    ):
        table_body.add_geom(
            name=f"table_leg_{index}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[
                float(center_xy[0] + sign_x * leg_x),
                float(center_xy[1] + sign_y * leg_y),
                leg_half_height,
            ],
            size=[leg_size[0] / 2.0, leg_size[1] / 2.0, leg_half_height],
            rgba=table.get("leg_rgba", [0.78, 0.80, 0.83, 1.0]),
            contype=1,
            conaffinity=1,
        )


def build_model(config: dict[str, Any]) -> tuple[Any, Any]:
    scene = mujoco.MjSpec.from_string(
        """<mujoco model="linglong_h_replay">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.25 0.25 0.25" specular="0 0 0"/>
    <global offwidth="1920" offheight="1080"/>
  </visual>
  <asset>
    <texture type="2d" name="ground_tex" builtin="checker" rgb1=".22 .24 .28"
      rgb2=".12 .14 .18" mark="edge" markrgb=".65 .65 .65" width="256" height="256"/>
    <material name="ground_mat" texture="ground_tex" texuniform="true" texrepeat="4 4"
      reflectance=".08"/>
  </asset>
  <worldbody>
    <light pos="0 -1 2.5" dir="0 .35 -1" directional="true"/>
    <geom name="floor" type="plane" size="2 2 .05" material="ground_mat"/>
    <camera name="overview" pos="1.8 2.3 1.4" xyaxes="-.79 .61 0 -.24 -.31 .92"/>
    <body name="left_target" mocap="true"><geom type="sphere" size=".025"
      rgba=".1 .55 1 .8" contype="0" conaffinity="0"/></body>
    <body name="right_target" mocap="true"><geom type="sphere" size=".025"
      rgba="1 .45 .08 .8" contype="0" conaffinity="0"/></body>
  </worldbody>
</mujoco>"""
    )
    _add_table(scene, config)

    model_path = Path(config["model_path"]).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"LingLong-H model not found: {model_path}")
    robot = mujoco.MjSpec.from_file(str(model_path))
    gripper = config.get("robot_gripper") or {}
    use_gripper = bool(gripper.get("enabled", False))
    for side in replay.SIDES:
        wrist = robot.body(f"{side}_wrist_yaw_link")
        if use_gripper:
            gripper_path = Path(gripper["model_path"]).expanduser().resolve()
            if not gripper_path.is_file():
                raise FileNotFoundError(f"OmniPicker model not found: {gripper_path}")
            mount_positions_by_side = gripper.get("linglong_mount_position_m_by_side") or {}
            mount_quaternions_by_side = (
                gripper.get("linglong_mount_quaternion_wxyz_by_side") or {}
            )
            mount_position = np.asarray(
                mount_positions_by_side.get(
                    side,
                    gripper.get("linglong_mount_position_m", [0.116, 0.0, 0.0]),
                ),
                dtype=np.float64,
            )
            mount_quaternion = np.asarray(
                mount_quaternions_by_side.get(
                    side,
                    gripper.get(
                        "linglong_mount_quaternion_wxyz",
                        [0.7071067811865476, 0.0, 0.7071067811865475, 0.0],
                    ),
                ),
                dtype=np.float64,
            )
            if mount_position.shape != (3,) or mount_quaternion.shape != (4,):
                raise ValueError("LingLong OmniPicker mount position/quaternion shape is invalid")
            mount_quaternion /= max(float(np.linalg.norm(mount_quaternion)), 1e-12)
            attachment = wrist.add_site(
                name=f"{side}_gripper_attachment_site",
                pos=mount_position,
                quat=mount_quaternion,
                size=[0.006],
                rgba=[0.2, 0.8, 0.3, 0.5],
            )
            robot.attach(
                mujoco.MjSpec.from_file(str(gripper_path)),
                prefix=f"{side}_omnipicker_",
                site=attachment,
            )
        else:
            tcp_offset = np.asarray(
                config.get("tcp_offset_m", [0.0, 0.0, 0.0]), dtype=np.float64
            )
            if tcp_offset.shape != (3,):
                raise ValueError("tcp_offset_m must have 3 values")
            wrist.add_site(
                name=f"{side}_tcp",
                pos=tcp_offset,
                size=[0.012],
                rgba=[0.1, 0.55, 1.0, 0.7]
                if side == "left"
                else [1.0, 0.45, 0.08, 0.7],
            )
    for index, geom in enumerate(robot.geoms):
        is_collision = int(geom.contype) != 0 or int(geom.conaffinity) != 0
        geom.name = f"linglong_{'collision' if is_collision else 'visual'}_{index}"
        geom.group = 3 if is_collision else 2

    base_position = np.asarray(config.get("base_position_m", [0.0, 0.0, 0.0]))
    base_rpy = config.get("base_rpy_deg", [0.0, 0.0, 0.0])
    mount = scene.worldbody.add_frame(
        name="linglong_mount",
        pos=base_position,
        quat=replay.quaternion_from_rpy_deg(base_rpy),
    )
    scene.attach(robot, prefix="", frame=mount)
    model = scene.compile()
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    waist_home = np.asarray(config.get("waist_home_q_rad", [0.0, 0.0, 0.0, 0.0]))
    if waist_home.shape != (len(WAIST_JOINT_NAMES),):
        raise ValueError(f"waist_home_q_rad must have {len(WAIST_JOINT_NAMES)} values")
    for joint_name, value in zip(WAIST_JOINT_NAMES, waist_home):
        data.qpos[int(model.joint(joint_name).qposadr[0])] = value
    for side in replay.SIDES:
        ids, _ = replay.joint_addresses(model, side)
        data.qpos[ids] = home_q_for_side(config, side)
        if use_gripper:
            replay.set_omnipicker_qpos(model, data.qpos, side, 0.0, config)
    replay.set_controls_from_qpos(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def end_effector_site_id(model: Any, side: str) -> int:
    omnipicker_tcp = f"{side}_omnipicker_tcp"
    site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, omnipicker_tcp))
    if site_id >= 0:
        return site_id
    return int(model.site(f"{side}_tcp").id)


def main() -> None:
    replay.__doc__ = __doc__
    replay.JOINT_NAMES = ARM_JOINT_NAMES
    replay.home_q_for_side = home_q_for_side
    replay.build_model = build_model
    replay.end_effector_site_id = end_effector_site_id

    arguments = sys.argv[1:]
    hide_grippers = "--hide_grippers" in arguments
    if hide_grippers:
        arguments.remove("--hide_grippers")
        original_load_config = replay.load_config

        def load_camera_alignment_config(path: str | Path) -> dict[str, Any]:
            config = copy.deepcopy(original_load_config(path))
            config.setdefault("robot_gripper", {})["enabled"] = False
            return config

        replay.load_config = load_camera_alignment_config
    if "--config" not in arguments:
        arguments += ["--config", str(HERE / "config.json")]
    if "--camera_config" not in arguments:
        arguments += ["--camera_config", str(HERE / "viewer_camera.json")]
    if "--output_dir" not in arguments:
        arguments += ["--output_dir", str(HERE / "outputs")]
    sys.argv = [sys.argv[0], *arguments]
    replay.main()


if __name__ == "__main__":
    main()
