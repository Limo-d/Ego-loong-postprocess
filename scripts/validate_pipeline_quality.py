#!/usr/bin/env python3
"""Validate final pipeline outputs before destructive compaction."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess.Timebase import nominal_fps, relative_times_sec, row_stamp_ns


OPTIONAL_MISSING_ARTIFACTS = {
    "rtabmap_depth_diagnostics.json",
    "rtabmap_trajectory_summary.json",
    "rtabmap_apply_summary.json",
    # Collector records missing source paths, whose basenames differ from the
    # normalized names under outputs/summaries.
    "apply_rtabmap_pose_to_preprocess_summary.json",
}


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_count(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def ratio(numerator: Any, denominator: Any) -> Optional[float]:
    try:
        den = float(denominator)
        return None if den <= 0 else float(numerator) / den
    except (TypeError, ValueError):
        return None


def validate(args: argparse.Namespace) -> Dict[str, Any]:
    session = Path(args.session).expanduser().resolve()
    outputs = session / "outputs"
    summaries = outputs / "summaries"
    failures: List[str] = []
    warnings: List[str] = []
    metrics: Dict[str, Any] = {}

    required = [
        outputs / "manifest.json",
        outputs / "data" / "trajectory_wristroot_track_cameraoptical.jsonl",
        outputs / "web" / "index.html",
    ]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"required output missing or empty: {path}")

    trajectory_path = required[1]
    trajectory_frames = jsonl_count(trajectory_path)
    metrics["trajectory_frames"] = trajectory_frames
    if trajectory_frames < args.min_frames:
        failures.append(f"trajectory frames {trajectory_frames} < {args.min_frames}")
    trajectory_rows = [
        json.loads(line)
        for line in trajectory_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if trajectory_path.is_file() else []
    trajectory_stamps = [row_stamp_ns(row) for row in trajectory_rows]
    missing_timestamps = sum(stamp is None for stamp in trajectory_stamps)
    nonmonotonic_timestamps = sum(
        right is None or left is None or right <= left
        for left, right in zip(trajectory_stamps[:-1], trajectory_stamps[1:])
    )
    metrics["trajectory_missing_rgb_timestamps"] = missing_timestamps
    metrics["trajectory_nonmonotonic_rgb_timestamps"] = nonmonotonic_timestamps
    if missing_timestamps:
        failures.append(f"trajectory has {missing_timestamps} frames without rgb_stamp_ns")
    if nonmonotonic_timestamps:
        failures.append(f"trajectory has {nonmonotonic_timestamps} non-monotonic rgb timestamp steps")
    if trajectory_rows and not missing_timestamps and not nonmonotonic_timestamps:
        frame_times = relative_times_sec(trajectory_rows)
        metrics["trajectory_duration_sec"] = frame_times[-1] if frame_times else 0.0
        metrics["trajectory_nominal_fps"] = nominal_fps(trajectory_rows)

    web_dir = outputs / "web"
    rgb_frame_dir = web_dir / "rgb_frames"
    rgb_frame_count = len(list(rgb_frame_dir.glob("*.jpg"))) if rgb_frame_dir.is_dir() else 0
    metrics["web_rgb_frames"] = rgb_frame_count
    if rgb_frame_count < trajectory_frames:
        failures.append(f"web rgb_frames has {rgb_frame_count} frames, expected at least {trajectory_frames}")

    # The current review page renders trajectory and tactile data in browser-side
    # Canvas elements. Older pages used one JPEG per frame, so accept either
    # representation while still verifying that the Canvas payload is complete.
    index_path = web_dir / "index.html"
    index_text = index_path.read_text(encoding="utf-8") if index_path.is_file() else ""
    frame_count_match = re.search(r"const\s+FRAME_COUNT\s*=\s*(\d+)\s*;", index_text)
    canvas_frame_count = int(frame_count_match.group(1)) if frame_count_match else None
    metrics["web_canvas_frame_count"] = canvas_frame_count

    canvas_specs = {
        "trajectory": {
            "legacy_dir": "traj_frames",
            "markers": ('id="trajCanvas"', "const TRAJ="),
        },
        "tactile": {
            "legacy_dir": "tactile_frames",
            "markers": ('id="tactileCanvas"', "const TACTILE="),
        },
    }
    for renderer_name, spec in canvas_specs.items():
        legacy_dir = web_dir / spec["legacy_dir"]
        legacy_count = len(list(legacy_dir.glob("*.jpg"))) if legacy_dir.is_dir() else 0
        metrics[f"web_{spec['legacy_dir']}"] = legacy_count
        if legacy_count >= trajectory_frames:
            metrics[f"web_{renderer_name}_renderer"] = "jpeg_frames"
            continue

        markers_present = all(marker in index_text for marker in spec["markers"])
        frame_count_ok = canvas_frame_count == trajectory_frames
        tactile_asset_ok = renderer_name != "tactile" or (
            (web_dir / "tactile_hand.png").is_file()
            and (web_dir / "tactile_hand.png").stat().st_size > 0
        )
        if markers_present and frame_count_ok and tactile_asset_ok:
            metrics[f"web_{renderer_name}_renderer"] = "canvas"
            continue

        failures.append(
            f"web {renderer_name} renderer incomplete: "
            f"legacy_frames={legacy_count}, canvas_markers={markers_present}, "
            f"canvas_frame_count={canvas_frame_count}, expected={trajectory_frames}, "
            f"asset_ok={tactile_asset_ok}"
        )

    try:
        collected = read_json(outputs / "manifest.json")
        missing = [Path(path).name for path in collected.get("missing", [])]
        required_missing = [name for name in missing if name not in OPTIONAL_MISSING_ARTIFACTS]
        metrics["collected_missing"] = missing
        if required_missing:
            failures.append(f"required collected artifacts missing: {required_missing}")
        if missing and not required_missing:
            warnings.append(f"optional collected artifacts missing: {missing}")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        failures.append(f"cannot read output manifest: {exc}")

    active_glove_sides = ["left", "right"]
    try:
        trajectory_summary = read_json(summaries / "trajectory_summary.json")
        declared_sides = trajectory_summary.get("active_sides")
        if isinstance(declared_sides, list):
            active_glove_sides = [side for side in declared_sides if side in {"left", "right"}]
        if not active_glove_sides:
            failures.append("trajectory summary declares no active glove side")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        failures.append(f"cannot read trajectory summary: {exc}")
    metrics["active_glove_sides"] = active_glove_sides

    for side in ("left", "right"):
        try:
            fusion = read_json(summaries / f"{side}_fusion_summary.json")
            stats = fusion.get("stats", {})
            matched_ratio = ratio(stats.get("matched_hand_frame"), stats.get("frames"))
            visual_ratio = ratio(stats.get("visual_hand_present"), stats.get("frames"))
            metrics[f"{side}_matched_hand_frame_ratio"] = matched_ratio
            metrics[f"{side}_visual_hand_ratio"] = visual_ratio
            if matched_ratio is None or matched_ratio < args.min_hand_match_ratio:
                failures.append(f"{side} hand match ratio {matched_ratio} < {args.min_hand_match_ratio}")
            if visual_ratio is None or visual_ratio < args.min_visual_ratio:
                failures.append(f"{side} visual hand ratio {visual_ratio} < {args.min_visual_ratio}")
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            failures.append(f"cannot read {side} fusion summary: {exc}")

    try:
        depth = read_json(summaries / "depthroot_summary.json")
        applied_ratio = ratio(depth.get("applied"), depth.get("hands"))
        metrics["depth_applied_ratio"] = applied_ratio
        if applied_ratio is None or applied_ratio < args.min_depth_applied_ratio:
            failures.append(f"depth applied ratio {applied_ratio} < {args.min_depth_applied_ratio}")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        failures.append(f"cannot read depth summary: {exc}")

    rtab_trajectory_path = summaries / "rtabmap_trajectory_summary.json"
    rtab_apply_path = summaries / "rtabmap_apply_summary.json"
    rtab_exists = [rtab_trajectory_path.is_file(), rtab_apply_path.is_file()]
    if any(rtab_exists) and not all(rtab_exists):
        message = "RTAB-Map quality summaries are incomplete"
        if args.require_rtabmap:
            failures.append(message)
        else:
            warnings.append(f"{message}; RTAB-Map quality checks unavailable")
    elif all(rtab_exists):
        try:
            rtab_trajectory = read_json(rtab_trajectory_path)
            rtab_apply = read_json(rtab_apply_path)
            coverage = ratio(
                rtab_trajectory.get("rgb_frames_interpolated"),
                rtab_trajectory.get("rgb_frames_total"),
            )
            applied_coverage = rtab_apply.get("coverage_ratio")
            max_gap = rtab_trajectory.get("max_bracketing_gap_sec_used")
            missing_pose_count = int(rtab_apply.get("missing_pose_count", -1))
            missing_frame_json_count = int(rtab_apply.get("missing_frame_json_count", -1))
            metrics["rtabmap_coverage_ratio"] = coverage
            metrics["rtabmap_apply_coverage_ratio"] = applied_coverage
            metrics["rtabmap_max_interp_gap_sec"] = max_gap
            metrics["rtabmap_missing_pose_count"] = missing_pose_count
            metrics["rtabmap_missing_frame_json_count"] = missing_frame_json_count
            if rtab_trajectory.get("status") != "complete" or rtab_apply.get("status") != "complete":
                failures.append("RTAB-Map trajectory or application did not complete")
            if coverage is None or coverage < args.min_rtabmap_coverage_ratio:
                failures.append(f"RTAB-Map coverage ratio {coverage} < {args.min_rtabmap_coverage_ratio}")
            if applied_coverage is None or float(applied_coverage) < args.min_rtabmap_coverage_ratio:
                failures.append(
                    f"RTAB-Map apply coverage ratio {applied_coverage} < {args.min_rtabmap_coverage_ratio}"
                )
            if max_gap is None or float(max_gap) > args.max_rtabmap_interp_gap_sec:
                failures.append(
                    f"RTAB-Map interpolation gap {max_gap} > {args.max_rtabmap_interp_gap_sec} sec"
                )
            if missing_pose_count != 0 or missing_frame_json_count != 0:
                failures.append(
                    "RTAB-Map application has missing frames: "
                    f"pose={missing_pose_count}, frame_json={missing_frame_json_count}"
                )
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError) as exc:
            failures.append(f"cannot read RTAB-Map quality summaries: {exc}")
    else:
        message = "RTAB-Map summaries absent; RTAB-Map quality checks unavailable"
        if args.require_rtabmap:
            failures.append(message)
        else:
            warnings.append(message)

    for side in active_glove_sides:
        try:
            calibration = read_json(summaries / f"{side}_glove_fk_calibration.json")
            fit = calibration.get("fit_error_m", {})
            median = fit.get("median")
            p95 = fit.get("p95")
            metrics[f"{side}_calibration_fit_median_m"] = median
            metrics[f"{side}_calibration_fit_p95_m"] = p95
            if median is None or float(median) > args.max_calibration_median_m:
                failures.append(f"{side} calibration median error {median} > {args.max_calibration_median_m} m")
            if p95 is None or float(p95) > args.max_calibration_p95_m:
                failures.append(f"{side} calibration p95 error {p95} > {args.max_calibration_p95_m} m")
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            failures.append(f"cannot read {side} calibration summary: {exc}")

        try:
            wrist = read_json(summaries / f"{side}_wristroot_track_summary.json")
            residual_p95 = (wrist.get("raw_to_tracked_residual_m") or {}).get("p95")
            metrics[f"{side}_wrist_track_residual_p95_m"] = residual_p95
            if residual_p95 is None or float(residual_p95) > args.max_wrist_residual_p95_m:
                failures.append(f"{side} wrist residual p95 {residual_p95} > {args.max_wrist_residual_p95_m} m")
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            failures.append(f"cannot read {side} wrist tracking summary: {exc}")

    report = {
        "session": str(session),
        "passed": not failures,
        "thresholds": {
            "min_frames": args.min_frames,
            "min_hand_match_ratio": args.min_hand_match_ratio,
            "min_visual_ratio": args.min_visual_ratio,
            "min_depth_applied_ratio": args.min_depth_applied_ratio,
            "max_calibration_median_m": args.max_calibration_median_m,
            "max_calibration_p95_m": args.max_calibration_p95_m,
            "max_wrist_residual_p95_m": args.max_wrist_residual_p95_m,
            "min_rtabmap_coverage_ratio": args.min_rtabmap_coverage_ratio,
            "max_rtabmap_interp_gap_sec": args.max_rtabmap_interp_gap_sec,
            "require_rtabmap": args.require_rtabmap,
        },
        "metrics": metrics,
        "warnings": warnings,
        "failures": failures,
    }
    report_path = Path(args.report).expanduser().resolve() if args.report else outputs / "quality_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--min_frames", type=int, default=1)
    parser.add_argument("--min_hand_match_ratio", type=float, default=0.95)
    parser.add_argument("--min_visual_ratio", type=float, default=0.90)
    parser.add_argument("--min_depth_applied_ratio", type=float, default=0.85)
    parser.add_argument("--max_calibration_median_m", type=float, default=0.030)
    parser.add_argument("--max_calibration_p95_m", type=float, default=0.060)
    parser.add_argument("--max_wrist_residual_p95_m", type=float, default=0.070)
    parser.add_argument("--min_rtabmap_coverage_ratio", type=float, default=1.0)
    parser.add_argument("--max_rtabmap_interp_gap_sec", type=float, default=0.25)
    parser.add_argument("--require_rtabmap", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = validate(args)
    print(json.dumps(report, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
