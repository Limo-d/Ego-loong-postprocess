# Dual UR5e MuJoCo wrist replay

This is the first, Retarget-independent simulation bridge. It maps the optimized
left/right human wrist translations to two UR5e end-effectors.

The mapping is intentionally conservative:

1. Read optimized camera-frame wrist positions.
2. Subtract each hand's video-frame-0 position.
3. Convert camera optical axes `(x right, y down, z forward)` to robot-world
   axes `(x right, y forward, z up)`.
4. Add each displacement to that robot's configured home end-effector position.
5. Apply workspace, Cartesian-speed and joint-speed limits.
6. Apply endpoint-preserving acceleration/jerk regularization.
7. Detect only sustained near-static windows, lock their core to a constant
   target, and use C2-continuous entry/exit blending.
8. Resample the Cartesian command trajectory to 60 Hz with C1-continuous
   cubic-Hermite interpolation (continuous command velocity).
9. Solve 6D damped IK with the initial robot end-effector orientation held
   fixed; human/FK absolute orientation is not used.
10. Apply one endpoint-preserving zero-phase joint-conditioning pass and retain
    the unsmoothed 6D IK joints for audit.
11. Assign one shared bimanual time law after IK. The path is only slowed down,
    never sped up, until TCP translation/rotation and joint velocity/
    acceleration limits are satisfied; then resample commands to 60 Hz with
    continuous joint velocity.

Absolute human wrist position is never sent directly to the robot, so a mismatch
between the human starting pose and the robot initial pose is expected and safe.

## Setup

```bash
simulation/dual_ur5e/setup.sh
```

The setup creates `.venv-mujoco` and sparsely downloads Google DeepMind's
BSD-3-Clause UR5e model from:
https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e

## Headless replay and MP4

```bash
MUJOCO_GL=egl .venv-mujoco/bin/python simulation/dual_ur5e/replay_trajectory.py \
  --session postprocess_data/2026-08-22T0722.35_1_20260822T073040 \
  --video
```

Outputs are written under `simulation/dual_ur5e/outputs/`:

- `*_dual_ur5e.mp4`: MuJoCo replay.
- `*_dual_ur5e.npz`: times, 12 joint angles, targets and IK errors.
- `*_dual_ur5e_summary.json`: clipping and tracking-quality summary.

For an interactive local window, omit `MUJOCO_GL=egl` and use `--viewer`.
The viewer and video renderer load `viewer_camera.json`. Mouse adjustments are
temporary by default, preserving the reference-video view. Add
`--save_viewer_camera` only when a new view should replace the default.

## Current boundary

This version replays human wrist translation while holding each robot's initial
end-effector orientation fixed. It does not yet command human wrist rotation,
grippers, collisions, or a real robot. Those layers should be added only after
the coordinate mapping and left/right behavior are visually verified.

`config_relative_human_orientation.json` is an opt-in comparison profile. It
maps the glove-FK palm rotation relative to frame 0 into robot axes, applies a
0.5 rotation scale, repairs isolated SO(3) midpoint spikes by geodesic
interpolation, and applies rotation-vector acceleration/jerk regularization.
An independent 15-frame orientation lock removes low-amplitude back-and-forth
jitter while a motion-efficiency gate preserves sustained in-place wrist
rotation. The resulting relative rotation is composed with the robot's initial
end-effector orientation. The fixed mode remains the conservative fallback.

The orientation NPZ audit fields retain raw, spike-repaired and final
conditioned source-rate rotation vectors, plus spike/candidate/locked boolean
masks. Detection statistics and every accepted/rejected lock segment are also
written to the summary JSON.

The NPZ retains the source-rate raw and conditioned arrays as
`*_target_raw_source_m` and `*_target_conditioned_source_m`, plus their 60 Hz
command-rate forms. Conditioning preserves the first/last Cartesian targets
exactly and reports deviation, path-length retention and acceleration reduction
in the summary JSON. Static candidates and exact locked cores are stored as
separate boolean masks; slow motion is not classified from a single-frame speed.
The raw arrays remain the audit ground truth; derived robot commands never
overwrite them. This conditioned Cartesian layer is intended to be shared by
simulation and the future real-robot command path.

Time scaling keeps `pre_retime_times_sec`, `retimed_path_times_sec`, the final
60 Hz `times_sec`, and both pre/post-retime joint paths in the NPZ. Both arms
always use the same retimed path knots, so their coordination is unchanged.
The summary records original/final duration, local slowdown distribution,
active limits, limiting segment counts, and measured motion peaks before and
after retiming.

The command path starts and ends with zero joint velocity and one second of
hold. A final safety gate reports `PASS`/`FAIL` for joint soft-limit margin,
scaled 6D Jacobian condition, endpoint velocity, time-scaling limits, and
sampled signed clearances for non-adjacent self links, the other arm, and the
floor. Its current collision scope is the dual-UR5e capsule model only; a real
deployment must add the actual grippers, table, payload and surrounding
obstacles before treating a `PASS` as hardware authorization.

The relative-orientation profile also derives a robot-independent canonical
gripper command (`0=open`, `1=closed`) from calibrated multi-finger flexion,
usable thumb-to-fingertip spans, and baseline-corrected tactile confidence.
The command has median/EMA filtering, deadband, normalized speed limiting and
a hysteretic OPEN/CLOSING/GRASPED/OPENING state machine. It is resampled through
the same bimanual time law as the arms. This is not yet a physical gripper
stroke or device command; the NPZ and SVG retain its signals for review first.
