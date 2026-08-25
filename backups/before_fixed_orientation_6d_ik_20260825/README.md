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
9. Solve position-only IK with MuJoCo Jacobians.

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

This version replays wrist translation only. It does not yet command wrist
orientation, grippers, collisions, or a real robot. Those layers should be
added only after the coordinate mapping and left/right behavior are visually
verified.

The NPZ retains the source-rate raw and conditioned arrays as
`*_target_raw_source_m` and `*_target_conditioned_source_m`, plus their 60 Hz
command-rate forms. Conditioning preserves the first/last Cartesian targets
exactly and reports deviation, path-length retention and acceleration reduction
in the summary JSON. Static candidates and exact locked cores are stored as
separate boolean masks; slow motion is not classified from a single-frame speed.
The raw arrays remain the audit ground truth; derived robot commands never
overwrite them. This conditioned Cartesian layer is intended to be shared by
simulation and the future real-robot command path.
