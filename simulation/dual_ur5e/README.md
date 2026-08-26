# Dual UR5e MuJoCo wrist replay

This is the Retarget-independent simulation bridge. It maps optimized human
wrist trajectories and canonical grasp commands to dual UR5e arms fitted with
AgiBot OmniPicker grippers.

The mapping is intentionally conservative:

1. Read optimized wrist positions in the world frame already rebased to the
   first camera optical pose.
2. Subtract each hand's video-frame-0 position.
3. Use the recorded first-camera pose and gravity to level the camera frame:
   keep its horizontal viewing direction as robot forward and use gravity-up
   as robot up. This separates forward motion from lifting for a downward-tilted
   head camera.
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
Press Space in the viewer window to pause or resume trajectory playback.
The viewer and video renderer load `viewer_camera.json`. Mouse adjustments are
temporary by default, preserving the reference-video view. Add
`--save_viewer_camera` only when a new view should replace the default.

The optional symmetric shoulder-mount prototype places the two base centers
0.52 m apart at a height of 0.95 m, rotates both base axes inward, and uses
mirrored home joint configurations. It also adds a collision-enabled torso
column and shoulder beam to the safety audit:

```bash
MUJOCO_GL=egl .venv-mujoco/bin/python simulation/dual_ur5e/replay_trajectory.py \
  --session postprocess_data/2026-08-22T0722.35_1_20260822T073040 \
  --config simulation/dual_ur5e/config_shoulder_relative_human_orientation.json \
  --camera_config simulation/dual_ur5e/viewer_camera_shoulder.json \
  --name shoulder_layout \
  --video
```

These dimensions are a simulation prototype rather than an installation
drawing. Final base spacing, height and home posture must be revalidated over
the batch trajectory distribution and against the real stand, table and tools.

## Base pose search

`search_base_pose.py` implements a paper-inspired two-stage search for the
shared dual-UR5e body. It first selects representative position/orientation
extrema and farthest-point keyframes, evaluates symmetric paired base
candidates with 6D IK, reach utilization, joint margin, Jacobian condition,
collision clearance and a soft initial-gripper-horizontal preference, then
reruns the top candidates on a denser full path.

The search consumes replay NPZ files instead of raw human coordinates. This is
intentional: the robot task targets remain fixed while candidate bases move.
Re-anchoring every candidate to its own home TCP would cancel base translation
and produce a meaningless search. Generate every batch NPZ with the same
reference config so their task anchors match; the search rejects inconsistent
batch anchors.

Quick single-session search:

```bash
.venv-mujoco/bin/python simulation/dual_ur5e/search_base_pose.py \
  --replay_npz simulation/dual_ur5e/outputs/<replay>_dual_ur5e.npz \
  --config simulation/dual_ur5e/config_shoulder_relative_human_orientation.json \
  --name base_search_single
```

Add one `--replay_npz` argument per episode for batch selection. The default
grid searches 48--56 cm spacing, +/-5 cm forward/height offsets and mirrored
mount-normal twist. `--max_candidates` can limit a smoke test; omit it for the
complete grid. Outputs include a ranked JSON report and a directly loadable
recommended config with explicit `task_anchor_positions_m` and
`task_anchor_rotation_matrix`.

Run the normal replay once more with that recommended config. This final pass
performs command conditioning, shared bimanual time scaling and dynamic safety
checks that are deliberately outside the kinematic search:

```bash
MUJOCO_GL=egl .venv-mujoco/bin/python simulation/dual_ur5e/replay_trajectory.py \
  --session postprocess_data/<session> \
  --config simulation/dual_ur5e/base_pose_search_outputs/<name>_recommended_config.json \
  --name <name>_validated \
  --video
```

The search is not hardware authorization. Table, payload, object contacts,
cables and measured camera-to-body/TCP extrinsics still need to be added before
real-robot playback.

Gripper output is strictly binary: `0` opens and `1` closes. Continuous glove,
pinch, and tactile evidence feeds a hysteresis state machine for debouncing;
any stable thumb-to-finger pinch can request closure, including a thumb-index
precision grasp;
after confirmed tactile contact, sustained contact loss plus an opening trend
can request release before every finger is fully extended;
only its discrete state is sent to the OmniPicker, using zero-order hold during
trajectory retiming so interpolation cannot create partial-close commands.
MuJoCo visualizes that binary intent with a finite-speed open/close motion
(0.30 seconds per full stroke by default). The physical position is continuous
only while moving and always settles at exactly open or closed.

## Current boundary

`config.json` remains the conservative fixed-orientation profile.
`config_relative_human_orientation.json` additionally commands filtered
relative human wrist rotation and enables the complete OmniPicker replay.
Neither path talks to a real robot yet. Object contact dynamics, grasp-force
control, table/payload geometry and the real controller protocol remain future
hardware-enablement work.

`config_relative_human_orientation.json` maps the glove-FK palm rotation
relative to frame 0 into robot axes, applies a
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
floor. When enabled by a layout config, it also audits the torso column and
shoulder beam. The scope includes the dual-UR5e capsules, official OmniPicker collision
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
