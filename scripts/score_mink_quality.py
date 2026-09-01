#!/usr/bin/env python3
"""Convert a Mink trajectory audit into episode and source-frame training scores."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def finite(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def nearest_indices(query_times: np.ndarray, solved_times: np.ndarray) -> np.ndarray:
    right = np.searchsorted(solved_times, query_times, side="left")
    right = np.clip(right, 0, len(solved_times) - 1)
    left = np.clip(right - 1, 0, len(solved_times) - 1)
    choose_left = np.abs(query_times - solved_times[left]) <= np.abs(solved_times[right] - query_times)
    return np.where(choose_left, left, right).astype(np.int64)


def source_to_solved_times(
    source_times: np.ndarray,
    solved_times: np.ndarray,
    pre_retime_times: np.ndarray | None,
    retimed_path_times: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    if (
        pre_retime_times is not None
        and retimed_path_times is not None
        and len(pre_retime_times) == len(retimed_path_times)
        and len(pre_retime_times) >= 2
        and np.all(np.diff(pre_retime_times) >= 0.0)
        and np.all(np.diff(retimed_path_times) >= 0.0)
    ):
        return np.interp(source_times, pre_retime_times, retimed_path_times), "retimed_path"
    source_span = float(source_times[-1] - source_times[0]) if len(source_times) >= 2 else 0.0
    solved_span = float(solved_times[-1] - solved_times[0]) if len(solved_times) >= 2 else 0.0
    if source_span <= 0.0:
        return np.full_like(source_times, solved_times[0]), "duration_scaled"
    progress = (source_times - source_times[0]) / source_span
    return solved_times[0] + progress * solved_span, "duration_scaled"


def normalized_limit_score(value: float, limit: float) -> float:
    if value <= limit:
        return 1.0
    return float(math.exp(-(value / max(limit, 1e-9) - 1.0) * 4.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Mink *_summary.json")
    parser.add_argument("--npz", required=True, help="Mink solved trajectory NPZ")
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum_score", type=float, default=0.60)
    parser.add_argument("--clearance_soft_scale_m", type=float, default=0.005)
    parser.add_argument("--position_error_scale_m", type=float, default=0.005)
    parser.add_argument("--orientation_error_scale_deg", type=float, default=2.0)
    args = parser.parse_args()

    summary_path = Path(args.summary).expanduser().resolve()
    npz_path = Path(args.npz).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as solved:
        solved_times = np.asarray(solved["times_sec"], dtype=np.float64)
        source_times = np.asarray(solved["source_times_sec"], dtype=np.float64)
        pre_retime_times = np.asarray(solved["pre_retime_times_sec"], dtype=np.float64) if "pre_retime_times_sec" in solved else None
        retimed_path_times = np.asarray(solved["retimed_path_times_sec"], dtype=np.float64) if "retimed_path_times_sec" in solved else None
        mapped_times, sync_mode = source_to_solved_times(
            source_times, solved_times, pre_retime_times, retimed_path_times
        )
        mapped = nearest_indices(mapped_times, solved_times)
        clearances = np.asarray(solved["mink_clearance_m"], dtype=np.float64)[mapped]
        pos_errors = np.max(np.asarray(solved["ik_error_m"], dtype=np.float64)[mapped], axis=1)
        ori_errors_deg = np.degrees(
            np.max(np.asarray(solved["ik_orientation_error_rad"], dtype=np.float64)[mapped], axis=1)
        )

    audit = summary.get("safety_audit") if isinstance(summary.get("safety_audit"), dict) else {}
    thresholds = audit.get("thresholds") if isinstance(audit.get("thresholds"), dict) else {}
    clearance_threshold = finite(thresholds.get("environment_clearance_m"), 0.02)
    frame_count = max(1, int(summary.get("frame_count") or len(solved_times)))
    failed = {int(index) for index in summary.get("final_failed_frames", [])}
    mink_failed = {int(index) for index in summary.get("mink_failed_frames", [])}
    recovered = {int(index) for index in summary.get("recovery_frames", [])}

    frame_rows: list[dict[str, Any]] = []
    for source_index, solved_index in enumerate(mapped.tolist()):
        clearance_score = sigmoid(
            (float(clearances[source_index]) - clearance_threshold) / max(args.clearance_soft_scale_m, 1e-9)
        )
        tracking_score = math.sqrt(
            math.exp(-float(pos_errors[source_index]) / max(args.position_error_scale_m, 1e-9))
            * math.exp(-float(ori_errors_deg[source_index]) / max(args.orientation_error_scale_deg, 1e-9))
        )
        initially_failed = solved_index in mink_failed
        was_recovered = solved_index in recovered
        solver_ok = solved_index not in failed and (not initially_failed or was_recovered)
        recovery_factor = 0.75 if was_recovered else 1.0
        score = clamp01((0.55 * clearance_score + 0.45 * tracking_score) * recovery_factor)
        if not solver_ok:
            score = 0.0
        frame_rows.append(
            {
                "source_index": source_index,
                "source_time_sec": float(source_times[source_index]),
                "solved_index": solved_index,
                "mapped_solved_time_sec": float(mapped_times[source_index]),
                "solved_time_sec": float(solved_times[solved_index]),
                "score": score,
                "eligible": bool(solver_ok and score >= args.minimum_score),
                "solver_ok": solver_ok,
                "initially_failed": initially_failed,
                "recovered": was_recovered,
                "clearance_m": float(clearances[source_index]),
                "position_error_max_m": float(pos_errors[source_index]),
                "orientation_error_max_deg": float(ori_errors_deg[source_index]),
                "components": {
                    "clearance": clearance_score,
                    "tracking": tracking_score,
                    "recovery_factor": recovery_factor,
                },
            }
        )

    closest = audit.get("closest_clearance") if isinstance(audit.get("closest_clearance"), dict) else {}
    clearance_component_scores: dict[str, float] = {}
    threshold_names = {
        "self": "self_clearance_m",
        "mounting": "mounting_clearance_m",
        "interarm": "interarm_clearance_m",
        "structure": "structure_clearance_m",
        "environment": "environment_clearance_m",
    }
    for category, threshold_name in threshold_names.items():
        detail = closest.get(category) if isinstance(closest.get(category), dict) else {}
        distance = finite(detail.get("distance_m"), math.inf)
        threshold = finite(thresholds.get(threshold_name), 0.0)
        if math.isfinite(distance):
            scale = max(args.clearance_soft_scale_m, abs(threshold) * 0.25)
            clearance_component_scores[category] = sigmoid((distance - threshold) / scale)
    clearance_episode = min(clearance_component_scores.values(), default=0.0)

    solver_success = clamp01(1.0 - len(failed) / frame_count)
    recovery_score = float(math.exp(-(len(recovered) / frame_count) / 0.02))
    tracking_score = math.sqrt(
        math.exp(-finite(summary.get("position_error_max_m"), math.inf) / max(args.position_error_scale_m, 1e-9))
        * math.exp(-finite(summary.get("orientation_error_max_deg"), math.inf) / max(args.orientation_error_scale_deg, 1e-9))
    )
    time_scaling = summary.get("time_scaling") if isinstance(summary.get("time_scaling"), dict) else {}
    limits = time_scaling.get("limits") if isinstance(time_scaling.get("limits"), dict) else {}
    after = time_scaling.get("after") if isinstance(time_scaling.get("after"), dict) else {}
    dynamics_values = [
        normalized_limit_score(
            finite(after.get("tcp_translation_speed_max_mps"), math.inf),
            finite(limits.get("tcp_translation_speed_mps"), 0.25),
        ),
        normalized_limit_score(
            finite(after.get("tcp_angular_speed_max_rad_s"), math.inf),
            finite(limits.get("tcp_angular_speed_rad_s"), 0.8),
        ),
        normalized_limit_score(
            finite(after.get("joint_speed_max_rad_s"), math.inf),
            finite(limits.get("joint_speed_rad_s"), 1.2),
        ),
        normalized_limit_score(
            finite(after.get("joint_acceleration_max_rad_s2"), math.inf),
            finite(limits.get("joint_acceleration_rad_s2"), 3.0),
        ),
    ]
    dynamics_score = min(dynamics_values)
    episode_score = clamp01(
        0.35 * solver_success
        + 0.25 * clearance_episode
        + 0.20 * tracking_score
        + 0.10 * recovery_score
        + 0.10 * dynamics_score
    )
    verdict_pass = str(summary.get("verdict", "")).upper() == "PASS" and str(audit.get("verdict", "PASS")).upper() == "PASS"
    episode_eligible = bool(verdict_pass and episode_score >= args.minimum_score)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "mink_training_quality_v1",
        "sync_mode": sync_mode,
        "inputs": {"summary": str(summary_path), "npz": str(npz_path)},
        "thresholds": {
            "minimum_score": args.minimum_score,
            "clearance_soft_scale_m": args.clearance_soft_scale_m,
            "position_error_scale_m": args.position_error_scale_m,
            "orientation_error_scale_deg": args.orientation_error_scale_deg,
            "environment_clearance_m": clearance_threshold,
        },
        "episode": {
            "verdict": str(summary.get("verdict", "NOT_AVAILABLE")).upper(),
            "eligible": episode_eligible,
            "score": episode_score,
            "training_weight": episode_score if episode_eligible else 0.0,
            "components": {
                "solver_success": solver_success,
                "clearance": clearance_episode,
                "tracking": tracking_score,
                "recovery": recovery_score,
                "dynamics": dynamics_score,
            },
            "clearance_components": clearance_component_scores,
            "source_frame_count": len(frame_rows),
            "eligible_source_frames": sum(bool(row["eligible"]) for row in frame_rows),
        },
        "frames": frame_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "score": episode_score, "eligible": episode_eligible}, ensure_ascii=False))


if __name__ == "__main__":
    main()
