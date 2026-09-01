# LingLong-H MuJoCo session notes (2026-09-01)

## User goal

Add LingLong-H to the same MuJoCo table scene used by the dual UR5e replay,
keeping all required robot assets inside this postprocess repository. The latest
requested initial layout is:

- the robot base is grounded;
- the OmniPicker TCP forward axis is parallel to the table;
- both TCPs are constrained to at least 8 cm above the 0.75 m table
  (`minimum_tcp_height_m = 0.83`);
- the initial TCPs should move farther forward, as close as practical to the
  middle of the table;
- both LingLong wrists use the same AgiBot OmniPicker model and canonical
  command mapping as the UR5e scene.

“TCP horizontal” refers to the OmniPicker TCP forward axis, not to the upper-arm
or forearm links.

## Repository additions and changes

- `simulation/linglong_h/replay_trajectory.py`
  - imports the LingLong URDF into the dual-UR5e table/replay backend;
  - grounds and mounts the whole robot;
  - mounts one OmniPicker on each `*_wrist_yaw_link`;
  - uses `left_omnipicker_tcp` / `right_omnipicker_tcp` when the grippers are
    enabled;
  - initializes and replays the same eight-joint OmniPicker mimic mapping used
    by UR5e.
- `simulation/linglong_h/search_base_waist_pose.py`
  - measures the base mesh against the floor instead of guessing base height;
  - searches base forward position, waist extension and torso lean;
  - solves the initial pose with a horizontal TCP-axis residual;
  - supports `--initial_tcp_above_table_m`,
    `--minimum_initial_tcp_above_table_m`, `--initial_tcp_center_y_m` and
    `--initial_tcp_half_separation_m`;
  - shifts the task path to the requested initial anchor and clamps the complete
    target path to the minimum TCP height.
- OmniPicker assets copied locally:
  - `simulation/linglong_h/omnipicker.xml`
  - `simulation/linglong_h/assets/omnipicker/`
- LingLong source URDF and meshes remain under
  `simulation/linglong_h/assets/urdf/` and `assets/meshes/`.

## OmniPicker mount

Measured LingLong wrist geometry extends to approximately local `x=0.116 m`.
The current mount is:

```json
"linglong_mount_position_m": [0.116, 0.0, 0.0],
"linglong_mount_quaternion_wxyz":
  [0.7071067811865476, 0.0, 0.7071067811865475, 0.0]
```

This aligns OmniPicker local `+Z` with LingLong wrist local `+X`. The model
compiles with 34 joints, and the expected TCP sites and
`*_omnipicker_outer_joint1` joints are present. The transform is suitable for
simulation but still requires hardware-adapter verification.

## Previous validated bare-wrist result (superseded)

Before adding OmniPicker, the 5 cm TCP result used base `y=-0.40 m`, base
`z=0.3399999214 m`, waist extension `0`, lean `35 deg`, and passed all 606
source frames. It used a temporary 14 cm wrist TCP and must not be reused for
the OmniPicker layout.

## Current 8 cm OmniPicker status

The first exact-center smoke target was:

```text
left  TCP = [-0.30, 0.55, 0.83] m
right TCP = [ 0.30, 0.55, 0.83] m
```

It reached the initial anchors with TCP-axis error below `0.00001 deg`, but was
not a valid trajectory candidate: maximum keyframe position error was about
`6.8 mm`, minimum joint margin about `4.5 deg`, and an open OmniPicker finger
penetrated the table by about `2.9 mm`.

The follow-up search therefore moved slightly back/outward to the middle-table
region:

```text
left  TCP initial target = [-0.35, 0.50, 0.83] m
right TCP initial target = [ 0.35, 0.50, 0.83] m
```

and searches:

```text
base y:         -0.55, -0.50, -0.45, -0.40, -0.35, -0.30 m
waist extension: 0.00, 0.05, 0.10 rad
torso lean:      15, 20, 25, 30, 35, 40 deg
keyframes:       24
```

The exact command is:

```bash
.venv-mujoco/bin/python simulation/linglong_h/search_base_waist_pose.py \
  --replay_npz simulation/linglong_h/outputs/linglong_search_source_linglong_h.npz \
  --name omnipicker_midtable_8cm_refined \
  --keyframes 24 \
  --base_forward_m=-0.55,-0.50,-0.45,-0.40,-0.35,-0.30 \
  --waist_extension_rad=0.0,0.05,0.10 \
  --waist_lean_deg=15,20,25,30,35,40 \
  --initial_tcp_center_y_m=0.50 \
  --initial_tcp_half_separation_m=0.35
```

At the time these notes were written, that command was still running as PID
`188446` after the tool wait was interrupted. Check whether it completed and
look for:

```text
simulation/linglong_h/base_pose_search_outputs/
  omnipicker_midtable_8cm_refined_report.json
  omnipicker_midtable_8cm_refined_recommended_config.json
```

## Important intermediate-state warning (resolved)

At the time of the original notes, `simulation/linglong_h/config.json` enabled
OmniPicker but still contained the earlier bare-wrist base/waist/home values.
This was resolved by the completed 8.5 cm write-back described below.

## Resume checklist

1. Check PID `188446` and the `omnipicker_midtable_8cm_refined` report.
2. If a feasible candidate exists, rerun that single candidate with
   `--keyframes 606`.
3. Confirm open-gripper collision clearance, TCP-axis angle, joint margin and
   maximum TCP error.
4. Write the winning base/waist/home joints and `z=0.83 m` mid-table anchors to
   `simulation/linglong_h/config.json`.
5. Render and visually inspect a preview with both OmniPickers.
6. Update `simulation/linglong_h/README.md` and the root `README.md` with the
   final numbers.

## Resume outcome (completed)

The 108-candidate 8 cm search completed with no strictly collision-free
candidate. Its best pose tracked every sampled target with `0.190 mm` maximum
error and `31.20 deg` minimum joint margin, but the open right OmniPicker inner
finger link penetrated the table by `1.81 mm`.

Because the requirement is at least 8 cm rather than exactly 8 cm, the same
best base/waist pose was revalidated at 8.5 cm over all 606 source frames. It
passed every feasibility check:

```text
base position:             [0.0, -0.30, 0.3399999214] m
waist extension / lean:    0.05 rad / 35 deg
left initial TCP:          [-0.35, 0.50, 0.835] m
right initial TCP:         [ 0.35, 0.50, 0.835] m
maximum TCP error:         0.193 mm
minimum environment gap:  3.76 mm
minimum joint margin:      29.96 deg
maximum initial TCP angle: 0.000047 deg
```

The winning values and `minimum_tcp_height_m = 0.835` are now in `config.json`.
The final report is
`base_pose_search_outputs/omnipicker_midtable_8p5cm_all_606_frames_report.json`.
A 120-source-frame preview was rendered as
`outputs/linglong_omnipicker_midtable_8p5cm_preview_linglong_h.mp4`; inspection
of its beginning, middle and end frames showed a grounded base, both
OmniPickers correctly attached, and no visible table penetration. Both READMEs
have been updated with the validated layout and metrics.

## Visual-orientation correction (supersedes the result above)

Close-up inspection showed that the fingers were still vertical. The search
had incorrectly treated MuJoCo site local `+X` as the OmniPicker forward axis;
the model's TCP offset is actually along local `+Z`, while finger separation is
along local `+Y`. The initial solver now constrains both `+Z` and `+Y` to the
table plane. A new 108-candidate search selected base `y=-0.30 m`, waist
extension `0.0 rad`, and lean `40 deg`. Full 606-frame validation passed with
`0.117 mm` maximum TCP error, `40.40 mm` minimum environment clearance,
`41.94 deg` minimum joint margin, and both relevant initial orientation angles
below `0.000021 deg`. The authoritative report is now
`base_pose_search_outputs/omnipicker_jaws_horizontal_midtable_8p5cm_all_606_frames_report.json`.

## Wrist-camera roll correction

The earlier connected-component mesh edit was rejected after close-up
inspection showed duplicated/protruding flange geometry. It is superseded;
the runtime URDF references the original wrist meshes.

The final workflow hides the grippers, rotates the mirrored
`wrist_yaw_joint` coordinates by `left=-90 deg` and `right=+90 deg` relative
to the previous pose, and fixes them at `left=-1.0819593317 rad` and
`right=+1.0820688607 rad`. Each OmniPicker is then mounted horizontally at
the original flange position. The remaining six arm joints compensate the
shifted flange position.

The final base is at `y=-0.20 m`; the waist uses `extension=0.15 rad` and
`lean=32.5 deg`; initial TCP height is `0.13 m` above the table. All 606
frames pass with `0.117 mm` maximum TCP error, `10.61 mm` minimum environment
clearance, and `18.00 deg` minimum joint-limit margin. The authoritative
report is
`base_pose_search_outputs/wrist_camera_parallel_gripper_horizontal_fixed_yaw_all_606_report.json`.

## Forward-parallel initial OmniPicker pose (final current result)

The previous 45 cm pose kept both grippers horizontal but made their
longitudinal TCP-forward axes face each other. The search now constrains both
TCP-forward axes to the same horizontal direction and treats the mirrored
finger-separation axes as parallel undirected lines. The accepted common
forward yaw is `80 deg`, within the requested table-`+Y +/- 10 deg` tolerance.

```text
base position:                  [0.0, 0.0, 0.3399999214] m
waist extension / lean:         0.55 rad / 20 deg
left initial TCP:               [-0.225, 0.50, 0.88] m
right initial TCP:              [ 0.225, 0.50, 0.88] m
initial TCP separation:         0.45 m
initial height above table:     0.13 m
maximum horizontal error:       5.8381 deg
maximum parallel-axis error:    7.3314 deg
606-keyframe maximum TCP error: 2.4776 mm
606-keyframe environment gap:   30.5299 mm
606-keyframe cross-arm gap:     215.2191 mm
```

The fixed terminal wrist-camera yaw joints are unchanged. One remaining
active joint is exactly at its valid limit in the initial pose; no joint
exceeds its URDF range. The geometric report is
`base_pose_search_outputs/parallel_forward80_tcp45cm_13cm_ext055_all_606_report.json`.

The rebuilt 606-source-frame trajectory contains 3243 retimed control frames.
Mink checks the arm/OmniPicker meshes against the table, floor, torso/head,
and opposite arm. A 20 mm run found no penetration but bottomed at 18.693 mm;
multistart recovery could not increase that distance under the fixed-wrist and
5 mm TCP-error constraints. The retained threshold is therefore 18 mm. All
3243 frames pass with zero recovery and zero failed frames:

```text
minimum Mink clearance:       18.692997 mm (frame 909)
maximum joint speed:          0.477516 rad/s
maximum joint acceleration:   2.500985 rad/s^2
verdict:                      PASS
```

Final outputs:

```text
outputs/linglong_parallel_forward_tcp45cm_13cm_mink_source_linglong_h.npz
outputs/linglong_parallel_forward_tcp45cm_13cm_mink18mm_linglong_h.npz
outputs/linglong_parallel_forward_tcp45cm_13cm_mink18mm_linglong_h_summary.json
outputs/linglong_parallel_forward_tcp45cm_13cm_preview_linglong_h.mp4
outputs/linglong_parallel_forward_tcp45cm_13cm_closeup_linglong_h.mp4
```

Interactive viewer command:

```bash
MUJOCO_GL=egl .venv-mujoco/bin/python \
  simulation/linglong_h/replay_trajectory.py \
  --session postprocess_data/2026-08-22T0722.35_1_20260822T073040 \
  --config simulation/linglong_h/config.json \
  --name linglong_parallel_forward_tcp45cm_13cm_current \
  --viewer
```

## Directed roll correction after front-view inspection

Front-view inspection showed that treating the finger-separation axes as
undirected parallel lines allowed one complete OmniPicker to be upside-down.
The posture metric now compares the jaw axes as directed vectors. Both native
connection positions remain `[0.063, 0, 0] m`; only the right OmniPicker mount
receives an additional `180 deg` roll about that same connection axis:

```text
left mount quaternion wxyz:   [ 0.5, 0.5,  0.5, 0.5]
right mount quaternion wxyz:  [-0.5, 0.5, -0.5, 0.5]
directed TCP-forward error:    7.331421 deg
directed jaw-axis error:       7.041793 deg
```

The wrist cameras and fixed terminal wrist-yaw coordinates are unchanged.
The corrected 606-keyframe geometric report has `2.4777 mm` maximum TCP
error, `30.5300 mm` environment clearance, and `196.4085 mm` cross-arm
clearance. Its report is
`base_pose_search_outputs/directed_same_roll_mount_tcp45cm_13cm_all_606_report.json`.

The rebuilt trajectory has 3237 control frames. All pass the 18 mm Mink audit
with zero failures and zero recovery frames; minimum clearance is
`18.690581 mm` at frame 909, maximum joint speed is `0.476128 rad/s`, and
maximum joint acceleration is `2.501745 rad/s^2`. The final summary is
`outputs/linglong_directed_same_roll_tcp45cm_13cm_mink18mm_linglong_h_summary.json`.

## Camera-up branch correction (supersedes directed jaw-roll correction)

The front-view complaint referred to the white wrist cameras, not to the
black OmniPicker bodies. The previous right-only `180 deg` OmniPicker roll did
not correct the wrist camera and has been removed. Both OmniPickers now use
the identical native mount quaternion `[0.5, 0.5, 0.5, 0.5]` at the unchanged
`[0.063, 0, 0] m` connection.

The camera housing protrudes along `wrist_yaw_link +Z`. The old right-arm IK
branch had that directed axis at world `-Z`; a mirrored multistart seed found
the alternate legal branch with the camera axis at world `+Z`. The terminal
wrist-yaw coordinates are now fixed at `left=-1.0819593317 rad` and
`right=-0.6065221754 rad`. Initial camera-up errors are `7.731374 deg` left
and `0.000018 deg` right. TCP separation remains `45.01 cm`, with both TCPs
about `13 cm` above the table.

Per user guidance, later camera orientation is not rigidly tracked; the
position-only IK only requires it to remain continuous. Across all 2685
retimed frames, maximum camera rotation per frame is `0.554464 deg` left and
`0.659266 deg` right. Maximum angular speed is `33.262991 deg/s` left and
`39.550232 deg/s` right. Neither camera crosses into an inverted branch.

The 606-source-frame replay has maximum TCP errors of `2.5902 mm` left and
`0.8675 mm` right. The final position-only Mink 18 mm audit passes all 2685
frames with zero recovery and zero failure; minimum clearance is
`19.342814 mm` at frame 2392, maximum joint speed is `0.489044 rad/s`, and
maximum joint acceleration is `2.500885 rad/s^2`. Final outputs:

```text
outputs/linglong_camera_both_up_tcp45cm_13cm_source_linglong_h.npz
outputs/linglong_camera_both_up_tcp45cm_13cm_mink18mm_linglong_h.npz
outputs/linglong_camera_both_up_tcp45cm_13cm_mink18mm_linglong_h_summary.json
outputs/linglong_camera_both_up_check_linglong_h.mp4
```

## Source gripper event correction

The shared raw-glove gripper mapper previously rejected any per-finger pinch
channel whose calibrated p10--p90 travel was below `8 mm`.  The left
thumb--index precision-pinch channel in this recording spans `4.240760 mm`, so
that threshold incorrectly dropped the first grasp and delayed the first
LingLong close transition to source frame 96.  The robot-independent shared
default is now `minimum_pinch_span_m=0.004`; LingLong also records the same
value explicitly in `config.json` for reproducibility.

Reprocessing the original 606-frame JSONL now gives the correct first left
event sequence: `OPEN -> CLOSING` at frame 62 (`2.066873088 s`),
`CLOSING -> GRASPED` at frame 66 (`2.200206080 s`), and
`GRASPED -> OPENING` at frame 114 (`3.866871296 s`).  Both complete source
state arrays exactly match the result obtained by applying the same mapper to
the UR5E run; no UR5E NPZ values are copied into LingLong outputs.

The original palm-frame convention uses `+x` from wrist toward the MCPs.  Its
mapped left-hand `+x` elevation is about `76 deg` at frame 62 and `78 deg` at
frame 66, remaining roughly `58--78 deg` through frame 114.  Therefore the
first source grasp itself is upward-oriented.  LingLong remains in
`position_only` IK mode and does not rigidly track that palm orientation.

The official source and final Mink NPZ files above were regenerated.  The
18 mm audit still passes all `2685/2685` frames with no recovery or failures;
minimum clearance remains `19.342814 mm` at frame 2392.

## SDK-limit re-solve and real-robot CSV export

The controller ranges were inspected directly from
`user@192.168.3.27:/home/user/sdk/linglong_h_sdk/joint_limits.py`.  They are
now built into `replay_trajectory.py` and intersected with the local URDF
ranges before IK, base/waist search, and Mink optimization.  The selected
configuration uses a `0.5 deg` solver margin and is recorded in
`config_sdk_limits_candidate70.json`.

The new full 606-source-frame solution keeps the initial TCP separation at
`45 cm`, TCP height at `13 cm` above the table, and common forward yaw near
`70 deg`.  Maximum source TCP error is `3.202 mm`.  The retimed 2518-frame
18 mm Mink audit passes without failures or recovery frames:

```text
minimum Mink clearance:       28.473374 mm
maximum joint speed:           0.413864 rad/s
maximum joint acceleration:    2.502116 rad/s^2
verdict:                       PASS
```

Validated trajectory:

```text
outputs/linglong_sdk_safe_tcp45cm_13cm_yaw70_mink18mm_linglong_h.npz
```

`scripts/export_linglong_h_sdk_trajectories.py` resamples that trajectory at
the SDK's 50 Hz rate and refuses to export any SDK joint-limit violation.  It
also converts the gripper convention from simulation `0=open, 1=closed` to
SDK `1=open, 0=closed`.  The 2099-row joint and EEF tasks are under:

```text
sdk_exports/linglong_sdk_safe_yaw70/joint/
sdk_exports/linglong_sdk_safe_yaw70/eef/
sdk_exports/linglong_sdk_safe_yaw70/preflight_report.json
```

The smallest raw SDK-limit margin is `0.496904 deg` at the left elbow.  The
joint task preserves the collision-validated branch and is the preferred
hardware path.  The EEF task uses controller wrist frames in base coordinates
and SDK RPY convention `Rz(yaw) Ry(pitch) Rx(roll)`; its RPY branches are
unwrapped, but controller-side IK can select another joint branch, so the EEF
task remains a comparison/commissioning export rather than collision-certified
hardware output.  The first 5 s move from live robot state to trajectory frame
zero is also not yet collision-validated and must be checked from the actual
measured startup state.
