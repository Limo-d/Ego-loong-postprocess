#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Track and stabilize a single hand bbox sequence from detector JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    from preprocess.Timebase import effective_alpha, relative_times_sec, repeat_counts, row_stamp_ns
except ModuleNotFoundError:
    from Timebase import effective_alpha, relative_times_sec, repeat_counts, row_stamp_ns


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def frame_index(frame: Dict) -> int:
    value = str(frame.get("frame", "0"))
    return int(value) if value.isdigit() else int(Path(value).stem)


def bbox_array(det: Dict) -> np.ndarray:
    return np.asarray(det["bbox"], dtype=np.float64)[:4]


def bbox_center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float64)


def bbox_area(box: np.ndarray) -> float:
    return max(1.0, float(box[2] - box[0])) * max(1.0, float(box[3] - box[1]))


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = bbox_area(a) + bbox_area(b) - inter
    return 0.0 if union <= 0 else inter / union


def clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    out = box.astype(np.float64).copy()
    out[[0, 2]] = np.clip(out[[0, 2]], 0.0, float(width - 1))
    out[[1, 3]] = np.clip(out[[1, 3]], 0.0, float(height - 1))
    if out[2] <= out[0] + 1:
        out[2] = min(float(width - 1), out[0] + 2.0)
    if out[3] <= out[1] + 1:
        out[3] = min(float(height - 1), out[1] + 2.0)
    return out


def state_from_box(box: np.ndarray) -> np.ndarray:
    c = bbox_center(box)
    return np.array([c[0], c[1], math.log(max(1.0, box[2] - box[0])), math.log(max(1.0, box[3] - box[1]))])


def box_from_state(state: np.ndarray, width: int, height: int, scale: float) -> np.ndarray:
    w = math.exp(float(state[2])) * scale
    h = math.exp(float(state[3])) * scale
    cx, cy = float(state[0]), float(state[1])
    return clip_box(np.array([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]), width, height)


def zero_phase_ema(states: np.ndarray, alpha: float, times_sec: np.ndarray, reference_fps: float) -> np.ndarray:
    if len(states) <= 1 or alpha <= 0 or alpha >= 1:
        return states.copy()
    fwd = states.copy()
    for i in range(1, len(states)):
        a = effective_alpha(alpha, float(times_sec[i] - times_sec[i - 1]), reference_fps)
        fwd[i] = a * states[i] + (1.0 - a) * fwd[i - 1]
    bwd = fwd.copy()
    for i in range(len(states) - 2, -1, -1):
        a = effective_alpha(alpha, float(times_sec[i + 1] - times_sec[i]), reference_fps)
        bwd[i] = a * fwd[i] + (1.0 - a) * bwd[i + 1]
    return bwd


def get_rgb_size(session_path: Path, data: Dict) -> Tuple[int, int]:
    for frame in data.get("frames", []):
        rgb_path = frame.get("rgb_path")
        candidates = []
        if rgb_path:
            candidates.append(Path(rgb_path))
        candidates.append(session_path / "preprocess" / "all_data" / f"{frame_index(frame):05d}" / "rgb.png")
        for candidate in candidates:
            img = cv2.imread(str(candidate))
            if img is not None:
                h, w = img.shape[:2]
                return w, h
    raise FileNotFoundError("Could not find a readable rgb.png for image size")


def select_seed(frames: List[Dict], prefer_x: Optional[float]) -> Optional[np.ndarray]:
    for frame in frames:
        dets = [d for d in frame.get("detections", []) if "bbox" in d]
        if not dets:
            continue
        if prefer_x is None:
            det = max(dets, key=lambda d: float(d.get("score", 1.0)) * bbox_area(bbox_array(d)))
        else:
            det = min(dets, key=lambda d: abs(float(bbox_center(bbox_array(d))[0]) - prefer_x))
        return bbox_array(det)
    return None


def track_boxes(data: Dict, width: int, height: int, args: argparse.Namespace) -> Tuple[Dict, Dict]:
    frames = data.get("frames", [])
    seed = select_seed(frames, args.prefer_x)
    states: List[Optional[np.ndarray]] = [None] * len(frames)
    kept_raw: List[Optional[np.ndarray]] = [None] * len(frames)
    reject_reasons: Dict[str, int] = {}
    accepted = 0
    rejected = 0
    missing = 0
    times_sec = np.asarray(relative_times_sec(frames, fallback_fps=args.reference_fps), dtype=np.float64)

    prev_box = None if seed is None else seed
    prev_state = None if seed is None else state_from_box(seed)
    lost = 0
    previous_accept_time_sec: Optional[float] = None

    for i, frame in enumerate(frames):
        dets = [d for d in frame.get("detections", []) if "bbox" in d]
        if not dets:
            missing += 1
            lost += 1
            continue

        candidates = []
        elapsed_sec = 1.0 / args.reference_fps if previous_accept_time_sec is None else max(
            1.0 / args.reference_fps,
            float(times_sec[i] - previous_accept_time_sec),
        )
        elapsed_frames_at_reference = elapsed_sec * args.reference_fps
        for det in dets:
            box = clip_box(bbox_array(det), width, height)
            score = float(det.get("score", 1.0))
            if prev_box is None:
                cost = -score * bbox_area(box)
                reason = "seed"
            else:
                center_dist = float(np.linalg.norm(bbox_center(box) - bbox_center(prev_box)))
                area_ratio = bbox_area(box) / bbox_area(prev_box)
                area_ratio_sym = max(area_ratio, 1.0 / max(area_ratio, 1e-6))
                iou = bbox_iou(box, prev_box)
                cost = center_dist - 80.0 * iou - 10.0 * score
                reason = "ok"
                if center_dist > args.max_center_jump_px * elapsed_frames_at_reference + lost * args.lost_jump_px:
                    reason = "center_jump"
                elif area_ratio_sym > args.max_area_ratio:
                    reason = "area_ratio"
                elif iou < args.min_iou and center_dist > args.min_iou_center_px:
                    reason = "low_iou"
            candidates.append((reason, cost, box, det))

        ok = [c for c in candidates if c[0] in ("ok", "seed")]
        if ok:
            _, _, box, _ = min(ok, key=lambda item: item[1])
            state = state_from_box(box)
            if prev_state is not None:
                frame_alpha = effective_alpha(args.raw_alpha, elapsed_sec, args.reference_fps)
                state = frame_alpha * state + (1.0 - frame_alpha) * prev_state
            states[i] = state
            kept_raw[i] = box
            prev_state = state
            prev_box = box_from_state(state, width, height, scale=1.0)
            accepted += 1
            lost = 0
            previous_accept_time_sec = float(times_sec[i])
        else:
            reason = min(candidates, key=lambda item: item[1])[0]
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            rejected += 1
            lost += 1

    valid = np.array([s is not None for s in states], dtype=bool)
    interp = np.zeros(len(states), dtype=bool)
    valid_idx = np.flatnonzero(valid)
    if len(valid_idx) >= 2 and args.max_gap > 0:
        for left, right in zip(valid_idx[:-1], valid_idx[1:]):
            gap = right - left - 1
            max_interval_sec = (args.max_gap + 1) / args.reference_fps
            if 0 < gap and float(times_sec[right] - times_sec[left]) <= max_interval_sec:
                for off, idx in enumerate(range(left + 1, right), start=1):
                    t = float((times_sec[idx] - times_sec[left]) / (times_sec[right] - times_sec[left]))
                    states[idx] = (1.0 - t) * states[left] + t * states[right]
                    valid[idx] = True
                    interp[idx] = True

    if valid.sum() > 0 and args.smooth_alpha > 0:
        idxs = np.flatnonzero(valid)
        smooth_states = zero_phase_ema(
            np.stack([states[i] for i in idxs]),
            args.smooth_alpha,
            times_sec[idxs],
            args.reference_fps,
        )
        for idx, state in zip(idxs, smooth_states):
            states[idx] = state

    output = json.loads(json.dumps(data))
    output.setdefault("postprocess", {})
    output["postprocess"]["single_hand_bbox_tracking"] = {
        "method": "single_target_gated_center_area_iou_with_gap_interpolation",
        "max_center_jump_px": args.max_center_jump_px,
        "lost_jump_px": args.lost_jump_px,
        "max_area_ratio": args.max_area_ratio,
        "min_iou": args.min_iou,
        "min_iou_center_px": args.min_iou_center_px,
        "max_gap": args.max_gap,
        "raw_alpha": args.raw_alpha,
        "smooth_alpha": args.smooth_alpha,
        "scale": args.scale,
        "reference_fps": args.reference_fps,
        "timebase": "rgb_stamp_ns",
    }

    for i, frame in enumerate(output.get("frames", [])):
        if states[i] is None:
            frame["detections"] = []
            continue
        box = box_from_state(states[i], width, height, args.scale)
        det = frame.get("detections", [{}])[0] if frame.get("detections") else {}
        det = json.loads(json.dumps(det))
        det["bbox"] = [float(v) for v in box.tolist()]
        det["raw_bbox"] = None if kept_raw[i] is None else [float(v) for v in kept_raw[i].tolist()]
        det["track_id"] = "target_hand"
        det["score"] = float(det.get("score", 1.0))
        det["tracking"] = {"interpolated": bool(interp[i]), "valid": True}
        frame["detections"] = [det]

    stats = {
        "frames": len(frames),
        "input_detections": sum(len(f.get("detections", [])) for f in frames),
        "accepted_raw": accepted,
        "missing_detector": missing,
        "rejected_raw": rejected,
        "reject_reasons": reject_reasons,
        "output_detections": int(sum(1 for s in states if s is not None)),
        "interpolated": int(interp.sum()),
    }
    return output, stats


def draw_video(session_path: Path, data: Dict, out_video: Path, fps: float) -> None:
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    frames = data.get("frames", [])
    repeats = repeat_counts(frames, output_fps=fps)
    for frame_i, frame in enumerate(tqdm(frames, desc="Drawing tracked bbox")):
        idx = frame_index(frame)
        img = cv2.imread(str(session_path / "preprocess" / "all_data" / f"{idx:05d}" / "rgb.png"))
        if img is None:
            continue
        for det in frame.get("detections", []):
            x1, y1, x2, y2 = map(int, map(round, det["bbox"]))
            color = (60, 230, 60) if not det.get("tracking", {}).get("interpolated") else (0, 190, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
            label = "target_hand interp" if det.get("tracking", {}).get("interpolated") else "target_hand"
            cv2.putText(img, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        cv2.putText(img, f"{idx:05d}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {out_video}")
        for _ in range(repeats[frame_i]):
            writer.write(img)
    if writer is not None:
        writer.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track/stabilize single hand bbox from detector JSON.")
    parser.add_argument("--session_path", required=True)
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--out_video", default=None)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--prefer_x", type=float, default=None)
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    session_path = Path(args.session_path).expanduser().resolve()
    data = load_json(Path(args.input_json).expanduser().resolve())
    timestamp_path = session_path / "preprocess" / "timestamps.jsonl"
    if timestamp_path.is_file():
        timestamp_rows = [json.loads(line) for line in timestamp_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        stamps_by_frame = {str(row.get("frame")): row_stamp_ns(row) for row in timestamp_rows}
        for frame in data.get("frames", []):
            stamp_ns = stamps_by_frame.get(str(frame.get("frame")))
            if stamp_ns is not None:
                frame["rgb_stamp_ns"] = stamp_ns
    width, height = get_rgb_size(session_path, data)
    output, stats = track_boxes(data, width, height, args)
    save_json(Path(args.output_json).expanduser().resolve(), output)
    if args.summary_json:
        save_json(Path(args.summary_json).expanduser().resolve(), stats)
    if args.out_video:
        draw_video(session_path, output, Path(args.out_video).expanduser().resolve(), args.fps)
    print(f"[TrackSingleHandBboxes] output: {args.output_json}")
    print(f"[TrackSingleHandBboxes] stats: {json.dumps(stats, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
