#!/usr/bin/env python3
"""Attach frame/episode motion-quality labels without deleting trajectory data."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


FINGERTIPS = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
SIGNAL_CONFIG = {
    "camera_translation_m": {"hard": 0.20, "sigma_floor": 0.0030},
    "camera_rotation_deg": {"hard": 28.0, "sigma_floor": 0.50},
    "left_raw_wrist_translation_m": {"hard": 0.30, "sigma_floor": 0.0020, "diagnostic_only": True},
    "right_raw_wrist_translation_m": {"hard": 0.30, "sigma_floor": 0.0020, "diagnostic_only": True},
    "left_wrist_translation_m": {"hard": 0.30, "sigma_floor": 0.0020},
    "right_wrist_translation_m": {"hard": 0.30, "sigma_floor": 0.0020},
    "left_wrist_rotation_deg": {"hard": 41.0, "sigma_floor": 1.00},
    "right_wrist_rotation_deg": {"hard": 41.0, "sigma_floor": 1.00},
    "left_finger_rms_m": {"hard": 0.30, "sigma_floor": 0.0010},
    "right_finger_rms_m": {"hard": 0.30, "sigma_floor": 0.0010},
    "left_finger_max_m": {"hard": 0.30, "sigma_floor": 0.0020},
    "right_finger_max_m": {"hard": 0.30, "sigma_floor": 0.0020},
}


def valid_array(value: Any, shape: tuple[int, ...]) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        return None
    return array


def rotation_step_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = left.T @ right
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return int(best)


def edge_invalid_runs(mask: np.ndarray) -> tuple[int, int]:
    leading = 0
    for value in mask:
        if value:
            break
        leading += 1
    trailing = 0
    for value in mask[::-1]:
        if value:
            break
        trailing += 1
    return int(leading), int(trailing)


def stats(values: np.ndarray) -> Dict[str, Optional[float]]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(np.max(values)),
    }


def robust_threshold(values: np.ndarray, multiplier: float, sigma_floor: float) -> Dict[str, Optional[float]]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"median": None, "mad": None, "robust_sigma": None, "threshold": None}
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    sigma = max(1.4826 * mad, float(sigma_floor))
    return {
        "median": median,
        "mad": mad,
        "robust_sigma": sigma,
        "threshold": median + float(multiplier) * sigma,
    }


def read_rows(path: Path) -> list[Dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def raw_wrist_world(row: Dict[str, Any], side: str) -> Optional[np.ndarray]:
    hand = ((row.get("hands") or {}).get(side) or {})
    glove = hand.get("glove") or {}
    points = valid_array(
        glove.get("kpts_3d_camera_m_before_palm_level") or glove.get("kpts_3d_camera_m"),
        (21, 3),
    )
    c2w = valid_array((row.get("camera") or {}).get("c2w"), (4, 4))
    if points is None or c2w is None:
        return None
    return c2w[:3, :3] @ points[0] + c2w[:3, 3]


def optimized_hand_state(row: Dict[str, Any], side: str) -> Optional[Dict[str, np.ndarray]]:
    optimized = (((row.get("hands") or {}).get(side) or {}).get("optimized_trajectory") or {})
    root = valid_array(optimized.get("wrist_translation_world_m"), (3,))
    rotation = valid_array(optimized.get("palm_rotation_world"), (3, 3))
    points = valid_array(optimized.get("kpts_3d_world_m_optimized"), (21, 3))
    if root is None or rotation is None or points is None:
        return None
    local_tips = (rotation.T @ (points[FINGERTIPS] - root).T).T
    return {"root": root, "rotation": rotation, "local_tips": local_tips}


def quaternion_quality(row: Dict[str, Any], side: str, tolerance: float) -> tuple[bool, Optional[float]]:
    palm = (((row.get("hands") or {}).get(side) or {}).get("palm_frame") or {})
    pose = palm.get("wrist_pose_world") or {}
    quaternion = valid_array(pose.get("quaternion_wxyz"), (4,))
    if quaternion is None:
        return False, None
    error = abs(float(np.linalg.norm(quaternion)) - 1.0)
    return error <= tolerance, error


def run(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()
    rows = read_rows(input_path)
    count = len(rows)
    signals = {name: np.full(count, np.nan, dtype=np.float64) for name in SIGNAL_CONFIG}
    spike_basis = {name: np.full(count, np.nan, dtype=np.float64) for name in SIGNAL_CONFIG}
    side_valid = {side: np.zeros(count, dtype=bool) for side in ("left", "right")}
    quaternion_valid = {side: np.zeros(count, dtype=bool) for side in ("left", "right")}
    quaternion_error = {side: np.full(count, np.nan) for side in ("left", "right")}
    static_energy = {side: np.full(count, np.nan) for side in ("left", "right")}
    states = {side: [optimized_hand_state(row, side) for row in rows] for side in ("left", "right")}
    raw_roots = {side: [raw_wrist_world(row, side) for row in rows] for side in ("left", "right")}
    camera_poses = [valid_array((row.get("camera") or {}).get("c2w"), (4, 4)) for row in rows]

    for i, row in enumerate(rows):
        c2w = valid_array((row.get("camera") or {}).get("c2w"), (4, 4))
        if i > 0:
            previous_c2w = valid_array((rows[i - 1].get("camera") or {}).get("c2w"), (4, 4))
            if c2w is not None and previous_c2w is not None:
                signals["camera_translation_m"][i] = np.linalg.norm(c2w[:3, 3] - previous_c2w[:3, 3])
                signals["camera_rotation_deg"][i] = rotation_step_deg(previous_c2w[:3, :3], c2w[:3, :3])
        for side in ("left", "right"):
            state = states[side][i]
            side_valid[side][i] = state is not None
            quaternion_valid[side][i], quaternion_error[side][i] = quaternion_quality(
                row, side, float(args.quaternion_tolerance)
            )
            if i == 0:
                continue
            previous = states[side][i - 1]
            raw = raw_roots[side][i]
            previous_raw = raw_roots[side][i - 1]
            if raw is not None and previous_raw is not None:
                signals[f"{side}_raw_wrist_translation_m"][i] = np.linalg.norm(raw - previous_raw)
            if state is None or previous is None:
                continue
            translation = float(np.linalg.norm(state["root"] - previous["root"]))
            rotation = rotation_step_deg(previous["rotation"], state["rotation"])
            finger_delta = np.linalg.norm(state["local_tips"] - previous["local_tips"], axis=1)
            finger_rms = float(np.sqrt(np.mean(finger_delta ** 2)))
            signals[f"{side}_wrist_translation_m"][i] = translation
            signals[f"{side}_wrist_rotation_deg"][i] = rotation
            signals[f"{side}_finger_rms_m"][i] = finger_rms
            signals[f"{side}_finger_max_m"][i] = float(np.max(finger_delta))
            rotational_equivalent = float(args.static_hand_radius_m) * math.radians(rotation)
            static_energy[side][i] = math.sqrt(
                translation * translation + rotational_equivalent * rotational_equivalent
                + finger_rms * finger_rms
            )

    # Robust spikes are evaluated on acceleration/constant-velocity residuals,
    # not on motion magnitude. A sustained fast reach is valid; an abrupt
    # one-frame change in velocity is the artifact we want to identify.
    for i in range(2, count):
        c0, c1, c2 = camera_poses[i - 2], camera_poses[i - 1], camera_poses[i]
        if c0 is not None and c1 is not None and c2 is not None:
            spike_basis["camera_translation_m"][i] = np.linalg.norm(c2[:3, 3] - 2.0 * c1[:3, 3] + c0[:3, 3])
        if np.isfinite(signals["camera_rotation_deg"][i - 1:i + 1]).all():
            spike_basis["camera_rotation_deg"][i] = abs(
                signals["camera_rotation_deg"][i] - signals["camera_rotation_deg"][i - 1]
            )
        for side in ("left", "right"):
            r0, r1, r2 = raw_roots[side][i - 2], raw_roots[side][i - 1], raw_roots[side][i]
            if r0 is not None and r1 is not None and r2 is not None:
                spike_basis[f"{side}_raw_wrist_translation_m"][i] = np.linalg.norm(r2 - 2.0 * r1 + r0)
            s0, s1, s2 = states[side][i - 2], states[side][i - 1], states[side][i]
            if s0 is None or s1 is None or s2 is None:
                continue
            spike_basis[f"{side}_wrist_translation_m"][i] = np.linalg.norm(
                s2["root"] - 2.0 * s1["root"] + s0["root"]
            )
            rotation_name = f"{side}_wrist_rotation_deg"
            if np.isfinite(signals[rotation_name][i - 1:i + 1]).all():
                spike_basis[rotation_name][i] = abs(signals[rotation_name][i] - signals[rotation_name][i - 1])
            tip_second = np.linalg.norm(
                s2["local_tips"] - 2.0 * s1["local_tips"] + s0["local_tips"], axis=1
            )
            spike_basis[f"{side}_finger_rms_m"][i] = float(np.sqrt(np.mean(tip_second ** 2)))
            spike_basis[f"{side}_finger_max_m"][i] = float(np.max(tip_second))

    original_count = count
    trim_candidates = []
    if args.trim_terminal_bad_tail:
        for side in ("left", "right"):
            _leading, trailing = edge_invalid_runs(side_valid[side])
            if trailing <= int(args.max_terminal_invalid_frames):
                continue
            invalid_start = count - trailing
            lookback_start = max(1, invalid_start - int(args.terminal_trim_lookback_frames))
            translation = signals[f"{side}_wrist_translation_m"]
            rotation = signals[f"{side}_wrist_rotation_deg"]
            rapid = np.zeros(count, dtype=bool)
            rapid |= np.isfinite(translation) & (translation > float(args.terminal_fast_translation_m))
            rapid |= np.isfinite(rotation) & (rotation > float(args.terminal_fast_rotation_deg))
            rapid_before_loss = np.flatnonzero(rapid[lookback_start:invalid_start]) + lookback_start
            rapid_detected = bool(rapid_before_loss.size)
            onset = int(rapid_before_loss[0]) if rapid_detected else int(invalid_start)
            trim_candidate = max(0, onset - int(args.terminal_trim_pre_roll_frames)) if rapid_detected else onset
            trim_candidates.append({
                "side": side,
                "rapid_motion_detected": rapid_detected,
                "rapid_motion_onset_index": onset,
                "trim_pre_roll_frames": int(args.terminal_trim_pre_roll_frames) if rapid_detected else 0,
                "trim_start_index": int(trim_candidate),
                "invalid_start_index": int(invalid_start),
                "terminal_invalid_frames": int(trailing),
            })
    trim_start = min((item["trim_start_index"] for item in trim_candidates), default=count)
    trim_info = {
        "enabled": bool(args.trim_terminal_bad_tail),
        "pre_roll_frames": int(args.terminal_trim_pre_roll_frames),
        "applied": bool(trim_start < count),
        "original_frame_count": int(original_count),
        "output_frame_count": int(trim_start),
        "trimmed_frame_count": int(original_count - trim_start),
        "trim_start_index": int(trim_start) if trim_start < count else None,
        "trim_start_frame": rows[trim_start].get("frame") if trim_start < count else None,
        "kept_last_frame": rows[trim_start - 1].get("frame") if 0 < trim_start < count else None,
        "reasons": trim_candidates,
        "raw_data_deleted": False,
    }
    if trim_start < count:
        rows = rows[:trim_start]
        signals = {name: values[:trim_start] for name, values in signals.items()}
        spike_basis = {name: values[:trim_start] for name, values in spike_basis.items()}
        side_valid = {side: values[:trim_start] for side, values in side_valid.items()}
        quaternion_valid = {side: values[:trim_start] for side, values in quaternion_valid.items()}
        quaternion_error = {side: values[:trim_start] for side, values in quaternion_error.items()}
        static_energy = {side: values[:trim_start] for side, values in static_energy.items()}
        count = trim_start

    robust = {}
    hard_masks = {}
    spike_masks = {}
    for name, config in SIGNAL_CONFIG.items():
        robust[name] = robust_threshold(spike_basis[name], args.spike_sigma_multiplier, config["sigma_floor"])
        hard_masks[name] = np.isfinite(signals[name]) & (signals[name] > float(config["hard"]))
        threshold = robust[name]["threshold"]
        spike_masks[name] = (
            np.isfinite(spike_basis[name]) & (spike_basis[name] > float(threshold))
            if threshold is not None else np.zeros(count, dtype=bool)
        )

    frame_valid = np.ones(count, dtype=bool)
    frame_reasons: list[list[str]] = [[] for _ in rows]
    for i in range(count):
        for side in ("left", "right"):
            if not side_valid[side][i]:
                frame_reasons[i].append(f"{side}_optimized_wrist_missing")
            elif not quaternion_valid[side][i]:
                frame_reasons[i].append(f"{side}_quaternion_invalid")
        for name in SIGNAL_CONFIG:
            if hard_masks[name][i]:
                frame_reasons[i].append(f"{name}_hard_limit")
            elif spike_masks[name][i] and not bool(SIGNAL_CONFIG[name].get("diagnostic_only")):
                frame_reasons[i].append(f"{name}_robust_spike")
        frame_valid[i] = not frame_reasons[i]
        rows[i]["quality_filter"] = {
            "frame_valid": bool(frame_valid[i]),
            "reasons": frame_reasons[i],
            "motion": {
                name: (float(values[i]) if np.isfinite(values[i]) else None)
                for name, values in signals.items()
            },
            "spike_metric": {
                name: (float(values[i]) if np.isfinite(values[i]) else None)
                for name, values in spike_basis.items()
            },
            "quaternion_norm_error": {
                side: (float(quaternion_error[side][i]) if np.isfinite(quaternion_error[side][i]) else None)
                for side in ("left", "right")
            },
            "static_equivalent_motion_m": {
                side: (float(static_energy[side][i]) if np.isfinite(static_energy[side][i]) else None)
                for side in ("left", "right")
            },
        }

    signal_summary = {}
    hard_violation_count = 0
    episode_spike_failures = []
    denominator = max(1, count - 1)
    for name in SIGNAL_CONFIG:
        hard_count = int(hard_masks[name].sum())
        spike_count = int(spike_masks[name].sum())
        spike_fraction = float(spike_count / denominator)
        hard_violation_count += hard_count
        diagnostic_only = bool(SIGNAL_CONFIG[name].get("diagnostic_only"))
        if spike_fraction > float(args.max_spike_frame_fraction) and not diagnostic_only:
            episode_spike_failures.append(name)
        signal_summary[name] = {
            "motion": stats(signals[name]),
            "spike_metric": stats(spike_basis[name]),
            "spike_metric_definition": "translation/finger second difference; rotation step difference",
            "robust": robust[name],
            "hard_threshold": float(SIGNAL_CONFIG[name]["hard"]),
            "hard_violation_frames": hard_count,
            "spike_frames": spike_count,
            "spike_fraction": spike_fraction,
            "diagnostic_only": diagnostic_only,
        }

    track_failures = []
    track_summary = {}
    for side in ("left", "right"):
        longest = longest_run(side_valid[side])
        leading_invalid, trailing_invalid = edge_invalid_runs(side_valid[side])
        ratio = float(side_valid[side].mean()) if count else 0.0
        track_summary[side] = {
            "valid_frames": int(side_valid[side].sum()), "valid_ratio": ratio,
            "longest_run": longest, "leading_invalid_frames": leading_invalid,
            "trailing_invalid_frames": trailing_invalid,
        }
        if longest < int(args.min_track_length):
            track_failures.append(f"{side}_track_too_short")
        if ratio < float(args.min_hand_valid_ratio):
            track_failures.append(f"{side}_valid_ratio_low")
        if trailing_invalid > int(args.max_terminal_invalid_frames):
            track_failures.append(f"{side}_terminal_track_lost")

    both_available = np.isfinite(static_energy["left"]) & np.isfinite(static_energy["right"])
    both_static = both_available.copy()
    both_static &= static_energy["left"] <= float(args.static_energy_threshold_m)
    both_static &= static_energy["right"] <= float(args.static_energy_threshold_m)
    static_fraction = float(both_static.sum() / max(1, both_available.sum()))
    static_candidate = static_fraction >= float(args.static_episode_fraction)
    quaternion_violation_count = int(sum((side_valid[s] & ~quaternion_valid[s]).sum() for s in ("left", "right")))
    episode_failures = list(track_failures)
    if hard_violation_count:
        episode_failures.append("physical_hard_limit_violation")
    if quaternion_violation_count:
        episode_failures.append("quaternion_tolerance_violation")
    if episode_spike_failures:
        episode_failures.append("spike_fraction_exceeded")
    if args.fail_static and static_candidate:
        episode_failures.append("static_episode")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, output_path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise

    summary = {
        "input_jsonl": str(input_path), "output_jsonl": str(output_path),
        "filter_version": "episode_frame_motion_v1",
        "duration_filter_enabled": False, "chunk_filter_enabled": False,
        "terminal_trim": trim_info,
        "episode_pass": not episode_failures, "episode_failures": episode_failures,
        "frame_count": count, "frame_valid_count": int(frame_valid.sum()),
        "frame_invalid_count": int((~frame_valid).sum()),
        "frame_valid_ratio": float(frame_valid.mean()) if count else None,
        "track": track_summary,
        "signals": signal_summary,
        "quaternion_tolerance": float(args.quaternion_tolerance),
        "quaternion_violation_frames": quaternion_violation_count,
        "spike_sigma_multiplier": float(args.spike_sigma_multiplier),
        "max_spike_frame_fraction": float(args.max_spike_frame_fraction),
        "static": {
            "definition": "both hands: sqrt(wrist_translation^2 + (hand_radius*wrist_rotation_rad)^2 + fingertip_local_rms^2)",
            "hand_radius_m": float(args.static_hand_radius_m),
            "per_frame_threshold_m": float(args.static_energy_threshold_m),
            "episode_fraction_threshold": float(args.static_episode_fraction),
            "both_hands_static_fraction": static_fraction,
            "static_candidate": bool(static_candidate),
            "causes_episode_failure": bool(args.fail_static),
            "left": stats(static_energy["left"]), "right": stats(static_energy["right"]),
        },
        "hard_violation_frames": hard_violation_count,
        "spike_fraction_failure_signals": episode_spike_failures,
    }
    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json")
    parser.add_argument("--min_track_length", type=int, default=15)
    parser.add_argument("--min_hand_valid_ratio", type=float, default=0.90)
    parser.add_argument("--max_terminal_invalid_frames", type=int, default=5)
    parser.add_argument("--terminal_trim_lookback_frames", type=int, default=30)
    parser.add_argument("--terminal_trim_pre_roll_frames", type=int, default=15)
    parser.add_argument("--terminal_fast_translation_m", type=float, default=0.012)
    parser.add_argument("--terminal_fast_rotation_deg", type=float, default=5.0)
    parser.add_argument("--no_trim_terminal_bad_tail", dest="trim_terminal_bad_tail", action="store_false")
    parser.set_defaults(trim_terminal_bad_tail=True)
    parser.add_argument("--quaternion_tolerance", type=float, default=1e-3)
    parser.add_argument("--spike_sigma_multiplier", type=float, default=3.0)
    parser.add_argument("--max_spike_frame_fraction", type=float, default=0.05)
    parser.add_argument("--static_hand_radius_m", type=float, default=0.10)
    parser.add_argument("--static_energy_threshold_m", type=float, default=0.002)
    parser.add_argument("--static_episode_fraction", type=float, default=0.90)
    parser.add_argument("--fail_static", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
