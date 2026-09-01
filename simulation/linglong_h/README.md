# LingLong-H MuJoCo wrist replay

This is an additional simulation backend beside `simulation/dual_ur5e`. It
uses the same published bimanual wrist trajectory, table dimensions, target
conditioning, shared time scaling, rendering, and output conventions.

The robot model and the 21 meshes needed by that URDF are vendored under:

```text
simulation/linglong_h/assets/urdf/linglong-h.urdf
simulation/linglong_h/assets/meshes/*.STL
```

The directory layout intentionally preserves the URDF's `../meshes/*.STL`
references. See `assets/SOURCE.md` for provenance and the current licensing
boundary.

Run a headless replay:

```bash
MUJOCO_GL=egl .venv-mujoco/bin/python simulation/linglong_h/replay_trajectory.py \
  --session postprocess_data/SESSION \
  --video
```

Outputs are written to `simulation/linglong_h/outputs/` as
`*_linglong_h.npz`, `*_linglong_h_summary.json`, and optionally
`*_linglong_h.mp4`.

## Grounded base and waist-pose search

The default config is the result of a search over base forward placement,
waist extension, and torso lean. The base height is not guessed: the search
measures the imported `base_link` mesh against the MuJoCo floor and shifts the
model until its minimum signed distance is zero. The initial-pose solve makes
both the OmniPicker TCP forward axis (local `+Z`) and the finger-opening axis
(local `+Y`) parallel to the table, so the complete gripper is horizontal
rather than merely one arbitrary site axis. Both TCPs remain at least 8 cm
above the table; the validated default uses 8.5 cm. The complete target paths
are clamped to the same minimum TCP height.

Generate a fixed task-path NPZ, then search it:

```bash
.venv-mujoco/bin/python simulation/linglong_h/replay_trajectory.py \
  --session postprocess_data/SESSION \
  --name linglong_search_source

.venv-mujoco/bin/python simulation/linglong_h/search_base_waist_pose.py \
  --replay_npz simulation/linglong_h/outputs/linglong_search_source_linglong_h.npz \
  --name full_session_base_waist
```

The waist is parameterized as:

```text
waist_1 = -extension
waist_2 =  2 * extension
waist_3 = -extension + lean
```

This lets `extension` raise the torso while approximately preserving its
upright orientation, then adds a signed forward/backward lean. The current
full-session result grounds the base at `z=0.3399999 m`, places it at
`y=0.0 m`, uses `extension=0.55 rad`, and leans forward by `20 degrees`.
The initial TCP anchors are `[-0.225, 0.50, 0.88] m` and
`[0.225, 0.50, 0.88] m`: `45 cm` apart and `13 cm` above the table. Both
OmniPickers point approximately toward table `+Y` (common yaw `80 degrees`)
instead of facing each other. The maximum initial horizontal error is
`5.84 degrees`; the TCP-forward difference is `7.33 degrees`, and the
undirected jaw-line difference is `7.04 degrees`. Both wrist-camera local
`+Z` axes start upward: the left/right errors from world `+Z` are `7.73` and
`0.00002 degrees`. This directed camera-up check, rather than the sign of the
geometrically undirected jaw axis, prevents an upside-down camera branch.
The 606-keyframe geometric audit has `2.478 mm` maximum TCP error,
`30.53 mm` minimum environment clearance, and `215.22 mm` minimum cross-arm
clearance. The terminal wrist-camera joints remain fixed; one active joint is
exactly at its valid limit in the initial pose. The authoritative report is
`base_pose_search_outputs/directed_same_roll_mount_tcp45cm_13cm_all_606_report.json`.

The current backend uses position-only dual-arm IK. Each wrist carries the
vendored AgiBot OmniPicker model with the same canonical `0=open, 1=closed`
mapping as the dual-UR5e backend. The native gripper is first removed from the
fused wrist STL, then the OmniPicker local `+Z` is mounted along the native
wrist-local `+X` connection at `[0.063, 0, 0] m`. Both mount quaternions are
identical: `[0.5, 0.5, 0.5, 0.5]`. Camera uprightness comes from selecting the
appropriate arm IK branch rather than rolling either gripper independently.
The adapter transform still requires verification against physical hardware.

The runtime URDF uses deterministic `*_wrist_yaw_link_without_gripper.STL`
meshes that retain the wrist and camera. The mirrored `wrist_yaw_joint`
coordinates are fixed at `left=-1.08195933 rad` and `right=-0.60652218 rad`;
position-only IK uses the other six arm joints. Camera orientation is allowed
to evolve with the motion but remains continuous: maximum per-frame rotation
is `0.55/0.66 degrees` and maximum angular speed is `33.3/39.6 degrees/s` for
left/right, with no flips.

`solve_mink_trajectory.py` adapts the shared Mink QP solver to LingLong-H. It
checks moving arm and OmniPicker collision meshes against the table, floor,
central torso/head geometry, and the opposite arm, with a required `18 mm`
distance. The solver uses the configured position-only task semantics while
the terminal camera-yaw coordinates remain fixed. All `2685` control frames
derived from the `606` source frames pass, with `19.343 mm` minimum Mink
clearance and no recovery or failed frames. Maximum joint speed is
`0.489 rad/s`, and maximum joint acceleration is `2.501 rad/s^2`. The summary
is `outputs/linglong_camera_both_up_tcp45cm_13cm_mink18mm_linglong_h_summary.json`.
