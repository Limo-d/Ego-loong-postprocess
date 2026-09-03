#!/usr/bin/env python3
"""Preflight and replay an exported Ego-Loong trajectory on LingLong-H."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_SDK_ROOT = Path("/home/user/sdk")
CONFIRM_TEXT = "RUN_LINGLONG_TRAJECTORY"

SDK_LIMITS = {
    "waist": np.array(
        [[-1.003564319, -0.087266463], [0.087266463, 3.010692959],
         [-2.574360646, 1.003564319], [-2.312561258, 2.312561258]], dtype=float
    ),
    "left": np.array(
        [[-2.879793265, 2.879793265], [0.0, math.pi],
         [-2.879793265, 2.879793265], [-0.698131700, 1.483529864],
         [-2.879793265, 2.879793265], [-1.308996939, 1.308996939],
         [-1.308996939, 1.308996939]], dtype=float
    ),
    "right": np.array(
        [[-2.879793265, 2.879793265], [-math.pi, 0.0],
         [-2.879793265, 2.879793265], [-0.698131700, 1.483529864],
         [-2.879793265, 2.879793265], [-1.308996939, 1.308996939],
         [-1.308996939, 1.308996939]], dtype=float
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an exported CSV task; connect to the robot only with --execute."
    )
    parser.add_argument("--mode", choices=("joint", "eef"), default="joint")
    parser.add_argument("--sdk-root", type=Path, default=DEFAULT_SDK_ROOT)
    parser.add_argument("--robot-ip", default="192.168.1.28")
    parser.add_argument("--speed", type=float, default=0.2,
                        help="Playback multiplier; 0.2 is five times slower than nominal.")
    parser.add_argument("--execute", action="store_true",
                        help="Actually connect and send commands. Without this flag: preflight only.")
    parser.add_argument("--initial-only", action="store_true",
                        help="Send only CSV row zero, moving to the trajectory initial pose.")
    parser.add_argument("--confirm", default="",
                        help=f"Required execution token: {CONFIRM_TEXT}")
    parser.add_argument("--allow-unvalidated-approach", action="store_true",
                        help="Acknowledge that live-state to frame-zero collision has not been simulated.")
    parser.add_argument("--allow-eef", action="store_true",
                        help="Acknowledge controller-side EEF IK is not collision-certified.")
    parser.add_argument("--enable-up", action="store_true")
    parser.add_argument("--autonomous", action="store_true")
    parser.add_argument("--shutdown-after", action="store_true")
    parser.add_argument("--countdown", type=float, default=5.0)
    parser.add_argument("--max-start-delta-deg", type=float, default=170.0)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the final interactive confirmation (token is still required).")
    args = parser.parse_args()
    if not 0.05 <= args.speed <= 1.0:
        parser.error("--speed must be in [0.05, 1.0]")
    if args.countdown < 0.0 or args.max_start_delta_deg <= 0.0:
        parser.error("countdown and start-delta limit must be valid")
    return args


def add_sdk_import_path(sdk_root: Path) -> None:
    root = sdk_root.expanduser().resolve()
    if not (root / "linglong_h_sdk").is_dir():
        raise FileNotFoundError(f"LingLong SDK package not found under {root}")
    sys.path.insert(0, str(root))


def validate_joint_rows(rows) -> dict[str, float]:
    minimum_margin = float("inf")
    duration = 0.0
    for frame, row in enumerate(rows):
        groups = {
            "waist": np.asarray(row.waist_q, dtype=float),
            "left": np.asarray(row.arm_q_l, dtype=float),
            "right": np.asarray(row.arm_q_r, dtype=float),
        }
        for group, values in groups.items():
            limits = SDK_LIMITS[group]
            if values.shape != (limits.shape[0],) or not np.isfinite(values).all():
                raise ValueError(f"frame {frame}: invalid {group} joint vector")
            margins = np.minimum(values - limits[:, 0], limits[:, 1] - values)
            if float(np.min(margins)) < -1e-7:
                joint = int(np.argmin(margins))
                raise ValueError(
                    f"frame {frame}: {group}[{joint}] violates SDK limit by "
                    f"{-math.degrees(float(margins[joint])):.6f} deg"
                )
            minimum_margin = min(minimum_margin, float(np.min(margins)))
        caps = np.array([row.cap_l, row.cap_r], dtype=float)
        if not np.isfinite(caps).all() or np.any(caps < -1e-7) or np.any(caps > 1.0 + 1e-7):
            raise ValueError(f"frame {frame}: cap command outside [0,1]")
        if not math.isfinite(row.send_time_s) or row.send_time_s <= 0.0:
            raise ValueError(f"frame {frame}: invalid segment time")
        duration += float(row.send_time_s)
    return {"duration_sec": duration, "minimum_margin_deg": math.degrees(minimum_margin)}


def validate_eef_rows(rows) -> dict[str, float]:
    duration = 0.0
    maximum_rpy_step = 0.0
    previous = None
    for frame, row in enumerate(rows):
        values = np.concatenate((row.ee_l, row.ee_r, row.ee_waist)).astype(float)
        if values.shape != (18,) or not np.isfinite(values).all():
            raise ValueError(f"frame {frame}: invalid EEF target")
        if previous is not None:
            maximum_rpy_step = max(
                maximum_rpy_step,
                *(float(np.max(np.abs(values[offset + 3:offset + 6] - previous[offset + 3:offset + 6])))
                  for offset in (0, 6, 12)),
            )
        previous = values
        if row.has_cap and not (0.0 <= row.cap_l <= 1.0 and 0.0 <= row.cap_r <= 1.0):
            raise ValueError(f"frame {frame}: cap command outside [0,1]")
        if not math.isfinite(row.send_time_s) or row.send_time_s <= 0.0:
            raise ValueError(f"frame {frame}: invalid segment time")
        duration += float(row.send_time_s)
    return {"duration_sec": duration, "maximum_rpy_step_deg": math.degrees(maximum_rpy_step)}


def main() -> int:
    args = parse_args()
    add_sdk_import_path(args.sdk_root)
    from linglong_h_sdk import (  # pylint: disable=import-outside-toplevel
        DEFAULT_CMD_PORT, DEFAULT_MODE_PORT, DEFAULT_STATE_PORT,
        LinglongHSdkClass, ManiInterpStartSource, ManiInterpTimeLaw,
        RobotModeManager, traj_replan,
    )

    task_name = f"ego_{args.mode}"
    config_root = HERE / "config"
    ok, bundle, error = traj_replan.load_trajectory_task_from_config_directory(
        str(config_root), task_name
    )
    if not ok:
        raise RuntimeError(f"cannot load task {task_name}: {error}")
    if len(bundle.actions) != 1:
        raise ValueError("expected exactly one action")
    action = bundle.actions[0]
    rows = action.joint_interp_rows if args.mode == "joint" else action.eef_interp_rows
    if not rows:
        raise ValueError(f"task has no {args.mode} rows")
    selected_rows = rows[:1] if args.initial_only else rows
    metrics = (validate_joint_rows(selected_rows) if args.mode == "joint"
               else validate_eef_rows(selected_rows))
    print(f"[preflight] mode={args.mode} frames={len(selected_rows)} speed={args.speed}")
    print(f"[preflight] nominal CSV duration={metrics['duration_sec']:.6f}s; "
          f"execution duration≈{metrics['duration_sec'] / args.speed:.3f}s")
    print(f"[preflight] metrics={metrics}")
    if not args.execute:
        print("[preflight] PASS; no robot connection and no command sent.")
        return 0
    if args.confirm != CONFIRM_TEXT:
        raise ValueError(f"execution requires --confirm {CONFIRM_TEXT}")
    if not args.allow_unvalidated_approach:
        raise ValueError("execution requires --allow-unvalidated-approach")
    if args.mode == "eef" and not args.allow_eef:
        raise ValueError("EEF execution requires --allow-eef")

    manager = RobotModeManager(args.robot_ip, DEFAULT_MODE_PORT)
    sdk = LinglongHSdkClass(
        args.robot_ip, DEFAULT_CMD_PORT, state_port=DEFAULT_STATE_PORT,
        chassis_tcp_on_send=False, auto_state_thread=True,
        state_poll_timeout=0.02, debug=False, object_udp_listen_port=0,
        interp_time_law=ManiInterpTimeLaw.kLinear, enable_camera=False,
    )
    try:
        print("[robot] waiting for a valid state packet...")
        if not sdk.wait_for_first_state_udp(total_timeout_s=3.0, poll_timeout_s=0.05):
            raise RuntimeError("no valid robot state; nothing was sent")
        if args.mode == "joint":
            first = rows[0]
            current_l, current_r, current_w, _head, _cap_l, _cap_r = sdk._current_joint_start()
            target = np.concatenate((first.waist_q, first.arm_q_l, first.arm_q_r)).astype(float)
            current = np.concatenate((current_w, current_l, current_r)).astype(float)
            delta_deg = np.degrees(np.abs(target - current))
            print(f"[robot] live-to-first maximum joint delta={float(np.max(delta_deg)):.3f} deg")
            if float(np.max(delta_deg)) > args.max_start_delta_deg:
                raise RuntimeError(
                    f"start delta exceeds --max-start-delta-deg={args.max_start_delta_deg}"
                )
        if not args.yes:
            typed = input(f"Type {CONFIRM_TEXT} once more to begin motion: ").strip()
            if typed != CONFIRM_TEXT:
                print("[robot] confirmation mismatch; nothing was sent.")
                return 2
        if args.countdown:
            print(f"[robot] motion starts in {args.countdown:.1f}s; Ctrl+C to abort.", flush=True)
            time.sleep(args.countdown)
        if args.enable_up:
            manager.robot_enable_up()
            time.sleep(15.0)
        if args.autonomous:
            manager.robot_autonomous_mode()
            time.sleep(2.0)

        execution_rows = selected_rows
        if args.initial_only:
            print("[robot] initial-only mode: exactly one CSV row will be sent.")
        for frame, row in enumerate(execution_rows):
            start = ManiInterpStartSource.kStatus if frame == 0 else ManiInterpStartSource.kCtrl
            segment_time = max(0.001, float(row.send_time_s) / args.speed)
            if args.mode == "joint":
                sent = sdk.send_joint_interpolation(
                    row.arm_q_l.astype(float).tolist(), row.arm_q_r.astype(float).tolist(),
                    segment_time, row.waist_q.astype(float).tolist(),
                    row.head_q.astype(float).tolist(), float(row.cap_l), float(row.cap_r),
                    start, ManiInterpTimeLaw.kLinear,
                )
            else:
                sent = sdk.send_eef_interpolation(
                    row.ee_l.astype(float).tolist(), row.ee_r.astype(float).tolist(),
                    segment_time, row.ee_waist[:3].astype(float).tolist(),
                    row.ee_waist[3:6].astype(float).tolist(),
                    row.head_att.astype(float).tolist() if row.has_head_att else None,
                    float(row.cap_l) if row.has_cap else None,
                    float(row.cap_r) if row.has_cap else None,
                    start, ManiInterpTimeLaw.kLinear,
                )
            if not sent:
                raise RuntimeError(f"SDK interrupted or rejected frame {frame}")
            if frame % 100 == 0 or frame + 1 == len(execution_rows):
                print(f"[robot] frame {frame + 1}/{len(execution_rows)}", flush=True)
        print("[robot] initial pose reached." if args.initial_only else "[robot] trajectory completed.")
    except KeyboardInterrupt:
        print("\n[robot] Ctrl+C: requesting hold at measured state.", file=sys.stderr)
        sdk.pause_mani_send(fetch_state=True, timeout=0.1)
        return 130
    except Exception:
        try:
            sdk.pause_mani_send(fetch_state=True, timeout=0.1)
        except Exception:
            pass
        raise
    finally:
        if args.shutdown_after:
            manager.robot_operation_mode()
            time.sleep(1.0)
            manager.robot_enable_down()
            time.sleep(1.0)
        sdk.close()
        manager.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
