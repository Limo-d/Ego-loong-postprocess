# Dual UR5e MuJoCo wrist replay

This is the Retarget-independent simulation bridge. It maps optimized human
wrist trajectories and canonical grasp commands to dual UR5e arms fitted with
AgiBot OmniPicker grippers.

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
12. Map canonical grasp `0=open, 1=closed` to the official OmniPicker master
    coordinate `outer_joint1 = 0.785398 * (1-command)` and write all eight
    official mimic-joint coordinates per hand.

Absolute human wrist position is never sent directly to the robot, so a mismatch
between the human starting pose and the robot initial pose is expected and safe.

## Setup

```bash
simulation/dual_ur5e/setup.sh
```

The setup creates `.venv-mujoco` and sparsely downloads Google DeepMind's
BSD-3-Clause UR5e model from:
https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e

The OmniPicker link geometry and kinematics come from AgiBot's official Genie
Sim repository. `omnipicker.xml` transcribes its link transforms, limits,
masses and mimic ratios. The included `assets/omnipicker/ur5e_adapter.stl` is
generated from the user-supplied `连接件.STEP` (63 x 63 x 16 mm).

## Headless replay and MP4

```bash
MUJOCO_GL=egl .venv-mujoco/bin/python simulation/dual_ur5e/replay_trajectory.py \
  --session postprocess_data/2026-08-22T0722.35_1_20260822T073040 \
  --video
```

Outputs are written under `simulation/dual_ur5e/outputs/`:

- `*_dual_ur5e.mp4`: MuJoCo replay.
- `*_dual_ur5e.npz`: times, 12 arm joints, 16 OmniPicker joints, physical
  master-joint commands, targets and IK errors.
- `*_dual_ur5e_summary.json`: clipping and tracking-quality summary.

For an interactive local window, omit `MUJOCO_GL=egl` and use `--viewer`.
The viewer and video renderer load `viewer_camera.json`. Mouse adjustments are
temporary by default, preserving the reference-video view. Add
`--save_viewer_camera` only when a new view should replace the default.

## Current boundary

`config.json` remains the conservative fixed-orientation profile.
`config_relative_human_orientation.json` additionally commands filtered
relative human wrist rotation and enables the complete OmniPicker replay.
Neither path talks to a real robot yet. Object contact dynamics, grasp-force
control, table/payload geometry and the real controller protocol remain future
hardware-enablement work.

`config_relative_human_orientation.json` maps
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
floor. The scope includes the dual-UR5e capsules, official OmniPicker collision
meshes, and the supplied STEP adapter. Tight wrist/flange mounting pairs are
reported separately with a 0.5 mm numerical penetration tolerance. A real
deployment must still add the table, payload and surrounding obstacles before
treating a `PASS` as hardware authorization.

The relative-orientation profile also derives a robot-independent canonical
gripper command (`0=open`, `1=closed`) from calibrated multi-finger flexion,
usable thumb-to-fingertip spans, and baseline-corrected tactile confidence.
The command has median/EMA filtering, deadband, normalized speed limiting and
a hysteretic OPEN/CLOSING/GRASPED/OPENING state machine. It is resampled through
the same bimanual time law as the arms and converted to the physical OmniPicker
master-joint angle. The NPZ retains both canonical and physical commands. This
is a kinematic replay command, not yet a force/current command for hardware.

To regenerate the adapter STL after updating the STEP file:

```bash
.venv-step/bin/python simulation/dual_ur5e/convert_step_adapter.py \
  /path/to/连接件.STEP \
  simulation/dual_ur5e/assets/omnipicker/ur5e_adapter.stl
```
