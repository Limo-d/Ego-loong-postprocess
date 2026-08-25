#!/usr/bin/env python3
"""Track two glove boxes with configurable image-to-physical hand identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    from preprocess.Timebase import effective_alpha, relative_times_sec, repeat_counts, row_stamp_ns
    from preprocess.TrackSingleHandBboxes import (
        bbox_area,
        bbox_array,
        bbox_center,
        bbox_iou,
        box_from_state,
        clip_box,
        get_rgb_size,
        load_json,
        save_json,
        state_from_box,
        zero_phase_ema,
    )
except ModuleNotFoundError:
    from Timebase import effective_alpha, relative_times_sec, repeat_counts, row_stamp_ns
    from TrackSingleHandBboxes import (
        bbox_area,
        bbox_array,
        bbox_center,
        bbox_iou,
        box_from_state,
        clip_box,
        get_rgb_size,
        load_json,
        save_json,
        state_from_box,
        zero_phase_ema,
    )


SIDES = ("right", "left")


def _initial_side(box: np.ndarray, width: int, image_left_side: str) -> str:
    # The acquisition image may be mirrored or mounted with either physical
    # hand on image-left. This is only a bootstrap; temporal association then
    # preserves the configured identities.
    if float(bbox_center(box)[0]) < width * 0.5:
        return image_left_side
    return "right" if image_left_side == "left" else "left"


def _candidate_cost(box: np.ndarray, prev_box: np.ndarray, score: float) -> float:
    return float(np.linalg.norm(bbox_center(box) - bbox_center(prev_box))) - 80.0 * bbox_iou(box, prev_box) - 10.0 * score


def _allowed(box: np.ndarray, prev_box: np.ndarray, score: float, lost: int, elapsed_frames: float, args: argparse.Namespace) -> bool:
    del score
    center_dist = float(np.linalg.norm(bbox_center(box) - bbox_center(prev_box)))
    ratio = bbox_area(box) / bbox_area(prev_box)
    ratio_sym = max(ratio, 1.0 / max(ratio, 1e-6))
    if center_dist > args.max_center_jump_px * elapsed_frames + lost * args.lost_jump_px:
        return False
    if ratio_sym > args.max_area_ratio:
        return False
    return not (bbox_iou(box, prev_box) < args.min_iou and center_dist > args.min_iou_center_px)


def _smooth_and_interpolate(states: List[Optional[np.ndarray]], times: np.ndarray, args: argparse.Namespace) -> Tuple[List[Optional[np.ndarray]], np.ndarray]:
    valid = np.asarray([state is not None for state in states], dtype=bool)
    interpolated = np.zeros(len(states), dtype=bool)
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) >= 2 and args.max_gap > 0:
        for a, b in zip(valid_idx[:-1], valid_idx[1:]):
            gap = b - a - 1
            max_interval_sec = (args.max_gap + 1) / args.reference_fps
            if 0 < gap and float(times[b] - times[a]) <= max_interval_sec:
                denom = max(float(times[b] - times[a]), 1e-9)
                for idx in range(a + 1, b):
                    t = float(times[idx] - times[a]) / denom
                    states[idx] = (1.0 - t) * states[a] + t * states[b]
                    valid[idx] = True
                    interpolated[idx] = True
    if valid.any() and args.smooth_alpha > 0:
        idxs = np.flatnonzero(valid)
        smoothed = zero_phase_ema(np.stack([states[i] for i in idxs]), args.smooth_alpha, times[idxs], args.reference_fps)
        for idx, state in zip(idxs, smoothed):
            states[idx] = state
    return states, interpolated


def track_boxes(data: Dict, width: int, height: int, args: argparse.Namespace) -> Tuple[Dict, Dict]:
    frames = data.get("frames", [])
    times = np.asarray(relative_times_sec(frames, fallback_fps=args.reference_fps), dtype=np.float64)
    states: Dict[str, List[Optional[np.ndarray]]] = {side: [None] * len(frames) for side in SIDES}
    raw_boxes: Dict[str, List[Optional[np.ndarray]]] = {side: [None] * len(frames) for side in SIDES}
    raw_dets: Dict[str, List[Optional[Dict]]] = {side: [None] * len(frames) for side in SIDES}
    prev_box: Dict[str, Optional[np.ndarray]] = {side: None for side in SIDES}
    prev_state: Dict[str, Optional[np.ndarray]] = {side: None for side in SIDES}
    previous_time: Dict[str, Optional[float]] = {side: None for side in SIDES}
    lost = {side: 0 for side in SIDES}

    for i, frame in enumerate(frames):
        detections = []
        for det in frame.get("detections", []):
            if "bbox" not in det:
                continue
            box = clip_box(bbox_array(det), width, height)
            detections.append((box, float(det.get("score", 1.0)), det))

        assignments: Dict[str, int] = {}
        used = set()
        candidates = []
        for side in SIDES:
            if prev_box[side] is None:
                continue
            elapsed = 1.0 / args.reference_fps if previous_time[side] is None else max(1.0 / args.reference_fps, float(times[i] - previous_time[side]))
            elapsed_frames = elapsed * args.reference_fps
            for det_idx, (box, score, _) in enumerate(detections):
                if _allowed(box, prev_box[side], score, lost[side], elapsed_frames, args):
                    candidates.append((_candidate_cost(box, prev_box[side], score), side, det_idx, elapsed))
        elapsed_by_side: Dict[str, float] = {}
        for _, side, det_idx, elapsed in sorted(candidates):
            if side in assignments or det_idx in used:
                continue
            assignments[side] = det_idx
            elapsed_by_side[side] = elapsed
            used.add(det_idx)

        # Bootstrap an untracked identity from the configured image side.
        remaining = [idx for idx in range(len(detections)) if idx not in used]
        both_uninitialized = prev_box["right"] is None and prev_box["left"] is None
        if both_uninitialized:
            # A single startup detection is ambiguous: LocateAnything can
            # return one large box covering both hands. Do not bind that box
            # to a physical identity, because the stale track would then gate
            # out the two correct per-hand boxes that appear a few frames
            # later. Bootstrap identities only from a simultaneous pair.
            if len(remaining) >= 2:
                ordered = sorted(remaining, key=lambda idx: float(bbox_center(detections[idx][0])[0]))
                image_right_side = "right" if args.image_left_side == "left" else "left"
                assignments[args.image_left_side], assignments[image_right_side] = ordered[0], ordered[-1]
                used.update((ordered[0], ordered[-1]))
        else:
            for det_idx in remaining:
                preferred = _initial_side(detections[det_idx][0], width, args.image_left_side)
                other = "left" if preferred == "right" else "right"
                side = preferred if preferred not in assignments and prev_box[preferred] is None else other
                if side not in assignments and prev_box[side] is None:
                    assignments[side] = det_idx
                    used.add(det_idx)

        for side in SIDES:
            det_idx = assignments.get(side)
            if det_idx is None:
                lost[side] += 1
                continue
            box, _, det = detections[det_idx]
            state = state_from_box(box)
            elapsed = elapsed_by_side.get(side, 1.0 / args.reference_fps)
            if prev_state[side] is not None:
                alpha = effective_alpha(args.raw_alpha, elapsed, args.reference_fps)
                state = alpha * state + (1.0 - alpha) * prev_state[side]
            states[side][i] = state
            raw_boxes[side][i] = box
            raw_dets[side][i] = det
            prev_state[side] = state
            prev_box[side] = box_from_state(state, width, height, 1.0)
            previous_time[side] = float(times[i])
            lost[side] = 0

    interpolated = {}
    for side in SIDES:
        states[side], interpolated[side] = _smooth_and_interpolate(states[side], times, args)

    output = json.loads(json.dumps(data))
    output.setdefault("postprocess", {})["dual_hand_bbox_tracking"] = {
        "method": "two_target_gated_assignment_with_gap_interpolation",
        "identity_rule": f"image_left_is_physical_{args.image_left_side}_then_temporal_association",
        "image_left_side": args.image_left_side,
        "reference_fps": args.reference_fps,
        "max_gap": args.max_gap,
        "scale": args.scale,
    }
    for i, frame in enumerate(output.get("frames", [])):
        frame_dets = []
        for side in SIDES:
            if states[side][i] is None:
                continue
            source = json.loads(json.dumps(raw_dets[side][i] or {}))
            source["bbox"] = [float(v) for v in box_from_state(states[side][i], width, height, args.scale).tolist()]
            source["raw_bbox"] = None if raw_boxes[side][i] is None else [float(v) for v in raw_boxes[side][i].tolist()]
            source["track_id"] = f"hand_{side[0]}"
            source["side"] = side
            source["score"] = float(source.get("score", 1.0))
            source["tracking"] = {"valid": True, "interpolated": bool(interpolated[side][i])}
            frame_dets.append(source)
        frame["detections"] = frame_dets

    stats = {
        "frames": len(frames),
        "input_detections": sum(len(frame.get("detections", [])) for frame in frames),
        "right_output_frames": int(sum(state is not None for state in states["right"])),
        "left_output_frames": int(sum(state is not None for state in states["left"])),
        "right_interpolated": int(interpolated["right"].sum()),
        "left_interpolated": int(interpolated["left"].sum()),
        "both_output_frames": int(sum(states["right"][i] is not None and states["left"][i] is not None for i in range(len(frames)))),
    }
    return output, stats


def draw_video(session: Path, data: Dict, out_video: Path, fps: float) -> None:
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    frames = data.get("frames", [])
    repeats = repeat_counts(frames, output_fps=fps)
    colors = {"right": (60, 150, 255), "left": (60, 230, 60)}
    for frame_i, frame in enumerate(tqdm(frames, desc="Drawing dual tracked bboxes")):
        idx = int(str(frame.get("frame", "0")))
        image = cv2.imread(str(session / "preprocess" / "all_data" / f"{idx:05d}" / "rgb.png"))
        if image is None:
            continue
        for det in frame.get("detections", []):
            side = str(det.get("side", "unknown"))
            color = colors.get(side, (0, 190, 255))
            x1, y1, x2, y2 = map(int, map(round, det["bbox"]))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
            suffix = " interp" if det.get("tracking", {}).get("interpolated") else ""
            cv2.putText(image, f"hand_{side}{suffix}", (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        if writer is None:
            h, w = image.shape[:2]
            writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {out_video}")
        for _ in range(repeats[frame_i]):
            writer.write(image)
    if writer is not None:
        writer.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track/stabilize left and right glove bboxes.")
    parser.add_argument("--session_path", required=True)
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--out_video", default=None)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--max_center_jump_px", type=float, default=130.0)
    parser.add_argument("--lost_jump_px", type=float, default=25.0)
    parser.add_argument("--max_area_ratio", type=float, default=2.8)
    parser.add_argument("--min_iou", type=float, default=0.02)
    parser.add_argument("--min_iou_center_px", type=float, default=80.0)
    parser.add_argument("--max_gap", type=int, default=10)
    parser.add_argument("--raw_alpha", type=float, default=0.75)
    parser.add_argument("--smooth_alpha", type=float, default=0.35)
    parser.add_argument("--scale", type=float, default=1.04)
    parser.add_argument("--reference_fps", type=float, default=30.0)
    parser.add_argument(
        "--image_left_side",
        choices=["left", "right"],
        default="left",
        help="Physical hand identity assigned to detections on image-left.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    session = Path(args.session_path).expanduser().resolve()
    data = load_json(Path(args.input_json).expanduser().resolve())
    timestamp_path = session / "preprocess" / "timestamps.jsonl"
    if timestamp_path.is_file():
        rows = [json.loads(line) for line in timestamp_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        stamps = {str(row.get("frame")): row_stamp_ns(row) for row in rows}
        for frame in data.get("frames", []):
            if str(frame.get("frame")) in stamps:
                frame["rgb_stamp_ns"] = stamps[str(frame.get("frame"))]
    width, height = get_rgb_size(session, data)
    output, stats = track_boxes(data, width, height, args)
    save_json(Path(args.output_json).expanduser().resolve(), output)
    if args.summary_json:
        save_json(Path(args.summary_json).expanduser().resolve(), stats)
    if args.out_video:
        draw_video(session, output, Path(args.out_video).expanduser().resolve(), args.fps)
    print(f"[TrackDualHandBboxes] output: {args.output_json}")
    print(f"[TrackDualHandBboxes] stats: {json.dumps(stats, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
