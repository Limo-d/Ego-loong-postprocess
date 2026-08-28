#!/usr/bin/env python3
"""Aggregate per-session postprocess summaries into one batch JSON report."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def get(data: dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = data
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def distribution(values: Iterable[Any]) -> dict[str, Any]:
    numbers = sorted(value for raw in values if (value := finite_number(raw)) is not None)
    if not numbers:
        return {"count": 0, "min": None, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(numbers),
        "min": numbers[0],
        "mean": sum(numbers) / len(numbers),
        "p50": percentile(numbers, 0.50),
        "p95": percentile(numbers, 0.95),
        "max": numbers[-1],
    }


def discover(args: argparse.Namespace) -> list[Path]:
    candidates: set[Path] = set()
    for raw in args.summary:
        candidates.add(Path(raw).expanduser().resolve())
    for raw in args.session:
        session = Path(raw).expanduser().resolve()
        candidates.add(session / "outputs" / "summary.json")
    for raw in args.root:
        root = Path(raw).expanduser().resolve()
        if (root / "outputs" / "summary.json").is_file():
            candidates.add(root / "outputs" / "summary.json")
        candidates.update(root.glob("*/outputs/summary.json"))
        if args.recursive:
            candidates.update(root.glob("**/outputs/summary.json"))
    return sorted(candidates)


def count_bool(rows: list[dict[str, Any]], dotted: str) -> dict[str, int]:
    counts = Counter(str(get(row, dotted)).lower() if get(row, dotted) is not None else "not_available" for row in rows)
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", default=[], help="Root containing session directories")
    parser.add_argument("--session", action="append", default=[], help="Explicit postprocess session directory")
    parser.add_argument("--summary", action="append", default=[], help="Explicit per-session summary.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    paths = discover(args)
    rows: list[dict[str, Any]] = []
    invalid: list[str] = []
    for path in paths:
        payload = load_json(path)
        if payload is None or not isinstance(payload.get("session"), dict):
            invalid.append(str(path))
            continue
        payload["_summary_path"] = str(path)
        rows.append(payload)

    verdicts = Counter(str(get(row, "overall.verdict", "INCOMPLETE")) for row in rows)
    simulation_verdicts = Counter(str(get(row, "simulation.verdict", "NOT_AVAILABLE")) for row in rows)
    failure_categories = Counter(
        str(category)
        for row in rows
        for category in (get(row, "overall.failure_categories", []) or [])
    )
    stage_completion = Counter(
        name
        for row in rows
        for name, stage in (row.get("stages") or {}).items()
        if isinstance(stage, dict) and stage.get("status") == "complete"
    )

    metric_paths = {
        "trajectory_frames": "trajectory.frames",
        "trajectory_duration_sec": "trajectory.duration_sec",
        "trajectory_frame_valid_ratio": "trajectory.frame_valid_ratio",
        "depth_applied_ratio": "perception.depth_applied_ratio",
        "left_visual_hand_ratio": "perception.left.visual_hand_ratio",
        "right_visual_hand_ratio": "perception.right.visual_hand_ratio",
        "left_calibration_fit_p95_m": "perception.left.calibration_fit_p95_m",
        "right_calibration_fit_p95_m": "perception.right.calibration_fit_p95_m",
        "left_wrist_track_residual_p95_m": "perception.left.wrist_track_residual_p95_m",
        "right_wrist_track_residual_p95_m": "perception.right.wrist_track_residual_p95_m",
        "rtabmap_coverage_ratio": "camera_tracking.coverage_ratio",
        "rtabmap_max_interp_gap_sec": "camera_tracking.max_interp_gap_sec",
        "simulation_minimum_clearance_m": "simulation.minimum_clearance_m",
        "simulation_position_error_max_m": "simulation.position_error_max_m",
        "simulation_orientation_error_max_deg": "simulation.orientation_error_max_deg",
    }
    distributions = {
        name: distribution(get(row, path) for row in rows)
        for name, path in metric_paths.items()
    }

    session_rows = []
    for row in rows:
        session_rows.append(
            {
                "name": get(row, "session.name"),
                "path": get(row, "session.path"),
                "summary": row["_summary_path"],
                "pipeline_exit_code": get(row, "session.pipeline_exit_code"),
                "verdict": get(row, "overall.verdict"),
                "quality_passed": get(row, "quality.passed"),
                "simulation_verdict": get(row, "simulation.verdict"),
                "failure_categories": get(row, "overall.failure_categories", []),
                "failure_details": get(row, "overall.failure_details", []),
                "trajectory_frames": get(row, "trajectory.frames"),
                "trajectory_duration_sec": get(row, "trajectory.duration_sec"),
                "frame_valid_ratio": get(row, "trajectory.frame_valid_ratio"),
                "depth_applied_ratio": get(row, "perception.depth_applied_ratio"),
                "rtabmap_coverage_ratio": get(row, "camera_tracking.coverage_ratio"),
                "simulation_minimum_clearance_m": get(row, "simulation.minimum_clearance_m"),
                "web_available": get(row, "web.available"),
            }
        )

    total = len(rows)
    pass_count = verdicts.get("PASS", 0)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "sessions": total,
            "pass": pass_count,
            "fail": verdicts.get("FAIL", 0),
            "incomplete": verdicts.get("INCOMPLETE", 0),
            "pass_rate": pass_count / total if total else None,
            "overall_verdict_counts": dict(sorted(verdicts.items())),
            "quality_passed_counts": count_bool(rows, "quality.passed"),
            "simulation_verdict_counts": dict(sorted(simulation_verdicts.items())),
            "web_available_counts": count_bool(rows, "web.available"),
            "failure_category_counts": dict(sorted(failure_categories.items())),
            "stage_complete_session_counts": dict(sorted(stage_completion.items())),
            "total_trajectory_frames": sum(int(value) for row in rows if (value := finite_number(get(row, "trajectory.frames"))) is not None),
            "total_trajectory_duration_sec": sum(value for row in rows if (value := finite_number(get(row, "trajectory.duration_sec"))) is not None),
        },
        "metric_distributions": distributions,
        "sessions": session_rows,
        "inputs": {
            "discovered_summary_files": len(paths),
            "valid_summary_files": total,
            "invalid_summary_files": invalid,
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sessions": total, "pass_rate": payload["summary"]["pass_rate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
