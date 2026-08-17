#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Track glove wrist/root translation with depth-mode hysteresis.

The visual/depth wrist root can jump between nearby depth surfaces. A simple EMA
still follows repeated wrong measurements, so this tracker only switches to a
far root candidate after it persists for several consecutive frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from preprocess.Timebase import dt_sec, effective_alpha, row_stamp_ns
except ModuleNotFoundError:
    from Timebase import dt_sec, effective_alpha, row_stamp_ns


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def valid_points(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape[0] < 21 or arr.shape[1] < 3 or not np.isfinite(arr[:21, :3]).all():
        return None
    return arr[:21, :3]


def transform_points(mat4, pts: np.ndarray) -> Optional[np.ndarray]:
    if mat4 is None:
        return None
    t = np.asarray(mat4, dtype=np.float64)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        return None
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return (t @ homo.T).T[:, :3]


def cap_delta(prev: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    delta = target - prev
    dist = float(np.linalg.norm(delta))
    if dist <= max_step or dist < 1e-9:
        return target
    return prev + delta * (max_step / dist)


def get_visual_wrist_uv(row: Dict) -> Optional[np.ndarray]:
    hand = ((row.get("visual_prior") or {}).get("hand") or {})
    kpts_2d = hand.get("kpts_2d")
    if kpts_2d is None:
        return None
    arr = np.asarray(kpts_2d, dtype=np.float64)
    if arr.shape[0] < 1 or arr.shape[1] < 2 or not np.isfinite(arr[0, :2]).all():
        return None
    return arr[0, :2].copy()


def get_camera_k(row: Dict) -> Optional[np.ndarray]:
    k = (row.get("camera") or {}).get("k")
    if k is None:
        return None
    arr = np.asarray(k, dtype=np.float64)
    if arr.shape != (3, 3) or not np.isfinite(arr).all():
        return None
    return arr


def backproject_root(k: np.ndarray, uv: np.ndarray, z: float) -> np.ndarray:
    return np.array([
        (float(uv[0]) - k[0, 2]) * z / k[0, 0],
        (float(uv[1]) - k[1, 2]) * z / k[1, 1],
        z,
    ], dtype=np.float64)


def cap_scalar_delta(prev: float, target: float, max_step: float) -> float:
    delta = float(target - prev)
    if abs(delta) <= max_step:
        return float(target)
    return float(prev + np.sign(delta) * max_step)


def summarize_steps(steps: List[float]) -> Dict:
    if not steps:
        return {}
    arr = np.asarray(steps, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def track(args: argparse.Namespace) -> Dict:
    rows = read_jsonl(Path(args.input_jsonl).expanduser().resolve())
    out_rows = []

    filt: Optional[np.ndarray] = None
    filt_z: Optional[float] = None
    vel = np.zeros(3, dtype=np.float64)
    vel_z = 0.0
    pending_center: Optional[np.ndarray] = None
    pending_count = 0
    pending_elapsed_sec = 0.0
    prev_stamp_ns: Optional[int] = None
    reference_fps = float(args.reference_fps)
    confirm_duration_sec = float(max(0, args.confirm_frames - 1)) / reference_fps

    raw_steps = []
    tracked_steps = []
    residuals = []
    stats = {
        "frames": 0,
        "valid_glove": 0,
        "tracked": 0,
        "accepted": 0,
        "rejected": 0,
        "pending_resets": 0,
        "confirmed_switches": 0,
        "missing": 0,
    }

    for row in rows:
        stats["frames"] += 1
        new_row = dict(row)
        glove = dict(row.get("glove_fk21") or row.get("glove") or {})
        pts_cam = valid_points(glove.get("kpts_3d_camera_m"))
        if pts_cam is None:
            stats["missing"] += 1
            out_rows.append(new_row)
            continue

        stats["valid_glove"] += 1
        raw_root = pts_cam[0].copy()
        stamp_ns = row_stamp_ns(row)
        frame_dt_sec = dt_sec(prev_stamp_ns, stamp_ns, reference_fps)
        dt_scale = frame_dt_sec * reference_fps
        alpha = effective_alpha(args.alpha, frame_dt_sec, reference_fps)
        accept_step_m = args.accept_step_m * dt_scale
        max_step_m = args.max_step_m * dt_scale
        mode = "init"
        accepted_measurement = True

        if args.snap_to_input_root:
            new_filt = raw_root.copy()
            mode = "snap_to_input_root"
            if filt is not None:
                raw_steps.append(float(np.linalg.norm(raw_root - filt)))
                tracked_steps.append(float(np.linalg.norm(raw_root - filt)))
                residuals.append(0.0)
            vel = np.zeros(3, dtype=np.float64)
            vel_z = 0.0
        elif args.filter_root_depth_only:
            uv = get_visual_wrist_uv(row)
            k = get_camera_k(row)
            if uv is None or k is None or abs(k[0, 0]) < 1e-9 or abs(k[1, 1]) < 1e-9:
                new_filt = raw_root.copy()
                mode = "depth_only_missing_anchor"
                residuals.append(0.0)
                vel = np.zeros(3, dtype=np.float64)
                vel_z = 0.0
            elif filt_z is None or filt is None:
                new_z = float(raw_root[2])
                new_filt = backproject_root(k, uv, new_z)
                mode = "depth_only_init"
                residuals.append(float(abs(raw_root[2] - new_z)))
            else:
                pred_z = float(filt_z + vel_z * frame_dt_sec * args.prediction_gain)
                raw_steps.append(float(abs(raw_root[2] - filt_z)))
                pred_dist_z = float(abs(raw_root[2] - pred_z))

                if pred_dist_z <= accept_step_m:
                    mode = "depth_only_accepted"
                    pending_center = None
                    pending_count = 0
                    pending_elapsed_sec = 0.0
                    target_z = float(raw_root[2])
                    stats["accepted"] += 1
                    accepted_measurement = True
                else:
                    accepted_measurement = False
                    stats["rejected"] += 1
                    raw_center = np.array([0.0, 0.0, float(raw_root[2])], dtype=np.float64)
                    if pending_center is None or float(abs(raw_root[2] - pending_center[2])) > args.pending_radius_m:
                        pending_center = raw_center
                        pending_count = 1
                        pending_elapsed_sec = 0.0
                        stats["pending_resets"] += 1
                    else:
                        pending_count += 1
                        pending_elapsed_sec += frame_dt_sec
                        pending_center[2] = (pending_center[2] * (pending_count - 1) + raw_root[2]) / pending_count

                    if pending_elapsed_sec >= confirm_duration_sec:
                        mode = "depth_only_confirmed_switch"
                        target_z = float(raw_root[2])
                        pending_center = None
                        pending_count = 0
                        pending_elapsed_sec = 0.0
                        accepted_measurement = True
                        stats["confirmed_switches"] += 1
                    else:
                        mode = "depth_only_held"
                        target_z = pred_z if args.use_prediction_when_rejected else float(filt_z)

                blended_z = (1.0 - alpha) * float(filt_z) + alpha * target_z if accepted_measurement else target_z
                new_z = cap_scalar_delta(float(filt_z), float(blended_z), max_step_m)
                new_filt = backproject_root(k, uv, new_z)
                vel_z = ((new_z - float(filt_z)) / frame_dt_sec) * (1.0 if accepted_measurement else args.rejected_velocity_decay)
                vel = (new_filt - filt) / frame_dt_sec
                tracked_steps.append(float(abs(new_z - float(filt_z))))
                residuals.append(float(abs(raw_root[2] - new_z)))
        elif filt is None:
            new_filt = raw_root
        else:
            pred = filt + vel * frame_dt_sec * args.prediction_gain
            raw_steps.append(float(np.linalg.norm(raw_root - filt)))
            pred_dist = float(np.linalg.norm(raw_root - pred))

            if pred_dist <= accept_step_m:
                mode = "accepted"
                pending_center = None
                pending_count = 0
                pending_elapsed_sec = 0.0
                target = raw_root
                stats["accepted"] += 1
            else:
                accepted_measurement = False
                stats["rejected"] += 1
                if pending_center is None or float(np.linalg.norm(raw_root - pending_center)) > args.pending_radius_m:
                    pending_center = raw_root.copy()
                    pending_count = 1
                    pending_elapsed_sec = 0.0
                    stats["pending_resets"] += 1
                else:
                    pending_count += 1
                    pending_elapsed_sec += frame_dt_sec
                    pending_center = (pending_center * (pending_count - 1) + raw_root) / pending_count

                if pending_elapsed_sec >= confirm_duration_sec:
                    mode = "confirmed_switch"
                    target = raw_root
                    pending_center = None
                    pending_count = 0
                    pending_elapsed_sec = 0.0
                    accepted_measurement = True
                    stats["confirmed_switches"] += 1
                else:
                    mode = "held"
                    target = pred if args.use_prediction_when_rejected else filt

            blended = (1.0 - alpha) * filt + alpha * target if accepted_measurement else target
            new_filt = cap_delta(filt, blended, max_step_m)
            vel = ((new_filt - filt) / frame_dt_sec) * (1.0 if accepted_measurement else args.rejected_velocity_decay)
            tracked_steps.append(float(np.linalg.norm(new_filt - filt)))
            residuals.append(float(np.linalg.norm(raw_root - new_filt)))

        shift = new_filt - raw_root
        pts_cam_tracked = pts_cam + shift[None, :]
        pts_world_tracked = transform_points((row.get("camera") or {}).get("c2w"), pts_cam_tracked)

        glove["kpts_3d_camera_m_before_wrist_track"] = glove.get("kpts_3d_camera_m")
        glove["kpts_3d_world_m_before_wrist_track"] = glove.get("kpts_3d_world_m")
        glove["kpts_3d_camera_m"] = pts_cam_tracked.tolist()
        if pts_world_tracked is not None:
            glove["kpts_3d_world_m"] = pts_world_tracked.tolist()
        glove["wrist_root_tracking"] = {
            "method": "Wrist/root tracker. In depth-only mode, only camera z is filtered and x/y are re-anchored to the visual wrist pixel each frame.",
            "mode": mode,
            "alpha": args.alpha,
            "effective_alpha": alpha,
            "dt_sec": frame_dt_sec,
            "reference_fps": reference_fps,
            "accept_step_m": args.accept_step_m,
            "effective_accept_step_m": accept_step_m,
            "pending_radius_m": args.pending_radius_m,
            "confirm_frames": args.confirm_frames,
            "confirm_duration_sec": confirm_duration_sec,
            "max_step_m": args.max_step_m,
            "effective_max_step_m": max_step_m,
            "filter_root_depth_only": bool(args.filter_root_depth_only),
            "raw_root_camera_m": raw_root.tolist(),
            "tracked_root_camera_m": new_filt.tolist(),
            "shift_camera_m": shift.tolist(),
            "pending_count": pending_count,
            "pending_elapsed_sec": pending_elapsed_sec,
        }

        filt = new_filt.copy()
        filt_z = float(new_filt[2])
        prev_stamp_ns = stamp_ns
        new_row["glove_fk21"] = glove
        if "glove" in new_row:
            new_row["glove"] = glove
        stats["tracked"] += 1
        out_rows.append(new_row)

    summary = {
        "input_jsonl": str(Path(args.input_jsonl).expanduser().resolve()),
        "output_jsonl": str(Path(args.output_jsonl).expanduser().resolve()),
        "params": {
            "alpha": args.alpha,
            "accept_step_m": args.accept_step_m,
            "pending_radius_m": args.pending_radius_m,
            "confirm_frames": args.confirm_frames,
            "max_step_m": args.max_step_m,
            "prediction_gain": args.prediction_gain,
            "use_prediction_when_rejected": args.use_prediction_when_rejected,
            "rejected_velocity_decay": args.rejected_velocity_decay,
            "snap_to_input_root": args.snap_to_input_root,
            "filter_root_depth_only": args.filter_root_depth_only,
            "reference_fps": reference_fps,
            "confirm_duration_sec": confirm_duration_sec,
            "timebase": "rgb_stamp_ns",
        },
        "stats": stats,
        "raw_to_previous_tracked_step_m": summarize_steps(raw_steps),
        "tracked_step_m": summarize_steps(tracked_steps),
        "raw_to_tracked_residual_m": summarize_steps(residuals),
    }

    write_jsonl(Path(args.output_jsonl).expanduser().resolve(), out_rows)
    if args.summary_json:
        write_json(Path(args.summary_json).expanduser().resolve(), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track calibrated glove wrist/root translation with hysteresis.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--accept_step_m", type=float, default=0.06)
    parser.add_argument("--pending_radius_m", type=float, default=0.06)
    parser.add_argument("--confirm_frames", type=int, default=8)
    parser.add_argument("--max_step_m", type=float, default=0.025)
    parser.add_argument("--prediction_gain", type=float, default=0.0)
    parser.add_argument("--use_prediction_when_rejected", action="store_true")
    parser.add_argument("--rejected_velocity_decay", type=float, default=0.35)
    parser.add_argument("--snap_to_input_root", action="store_true", help="Do not smooth/hold root translation; keep FK wrist exactly on the input visual root.")
    parser.add_argument("--filter_root_depth_only", action="store_true", help="Filter only wrist/root camera depth z, then reproject x/y from the visual wrist pixel to keep 2D alignment.")
    parser.add_argument("--reference_fps", type=float, default=30.0, help="FPS at which legacy per-frame alpha, step and confirmation parameters retain their meaning.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = track(args)
    print(f"[TrackGloveWristRoot] output: {args.output_jsonl}")
    print(f"[TrackGloveWristRoot] summary: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
