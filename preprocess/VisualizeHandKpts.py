# -*- coding: utf-8 -*-
"""Visualize saved 21-point hand keypoints from HaMeR/WiLoR/MediaPipe JSON files."""

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    from preprocess.Timebase import repeat_counts
except ModuleNotFoundError:
    from Timebase import repeat_counts


METHOD_TO_JSON = {
    "hamer": "hamer_hands.json",
    "wilor": "wilor_hands.json",
    "mediapipe": "mediapipe_hands.json",
    "aria": "aria_hands.json",
}

# Hands-compatible order used by this repo:
# 0 thumb tip, 1 index tip, 2 middle tip, 3 ring tip, 4 pinky tip,
# 5 wrist, 6-7 thumb, 8-10 index, 11-13 middle, 14-16 ring, 17-19 pinky,
# 20 palm center.
ARIA_HAND_BONES = [
    (5, 6), (6, 7), (7, 0),
    (5, 8), (8, 9), (9, 10), (10, 1),
    (5, 11), (11, 12), (12, 13), (13, 2),
    (5, 14), (14, 15), (15, 16), (16, 3),
    (5, 17), (17, 18), (18, 19), (19, 4),
    (6, 8), (8, 11), (11, 14), (14, 17),
    (5, 20),
]

MP_HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

ARIA_FINGER_COLOR = {
    0: (80, 120, 255),    # thumb
    1: (80, 220, 80),     # index
    2: (255, 180, 80),    # middle
    3: (80, 220, 220),    # ring
    4: (220, 100, 220),   # pinky
    5: (255, 255, 255),   # wrist
    20: (30, 30, 30),     # palm center
}

ARIA_POINT_TO_FINGER = {
    0: 0, 6: 0, 7: 0,
    1: 1, 8: 1, 9: 1, 10: 1,
    2: 2, 11: 2, 12: 2, 13: 2,
    3: 3, 14: 3, 15: 3, 16: 3,
    4: 4, 17: 4, 18: 4, 19: 4,
}

MP_FINGER_COLOR = {
    0: (255, 255, 255),   # wrist
    1: (80, 120, 255),    # thumb
    2: (80, 220, 80),     # index
    3: (255, 180, 80),    # middle
    4: (80, 220, 220),    # ring
    5: (220, 100, 220),   # pinky
}

MP_POINT_TO_FINGER = {
    0: 0,
    1: 1, 2: 1, 3: 1, 4: 1,
    5: 2, 6: 2, 7: 2, 8: 2,
    9: 3, 10: 3, 11: 3, 12: 3,
    13: 4, 14: 4, 15: 4, 16: 4,
    17: 5, 18: 5, 19: 5, 20: 5,
}


def _iter_frame_dirs(session_path: str, max_frames: Optional[int]) -> List[str]:
    all_data_dir = os.path.join(session_path, "preprocess", "all_data")
    if not os.path.isdir(all_data_dir):
        raise FileNotFoundError(f"Missing frame directory: {all_data_dir}")

    frame_names = [
        name
        for name in sorted(os.listdir(all_data_dir))
        if name.isdigit() and os.path.isdir(os.path.join(all_data_dir, name))
    ]
    if frame_names:
        max_width = max(len(name) for name in frame_names)
        frame_names = [name for name in frame_names if len(name) == max_width]
    frame_dirs = [os.path.join(all_data_dir, name) for name in frame_names]
    if max_frames is not None:
        frame_dirs = frame_dirs[:max_frames]
    return frame_dirs


def _load_points(hand_entry: Optional[Dict]) -> Optional[np.ndarray]:
    if not hand_entry:
        return None
    pts = hand_entry.get("kpts_2d")
    if pts is None:
        return None
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[0] < 21 or pts.shape[1] < 2:
        return None
    return pts[:21, :2]


def _valid_pt(pt: np.ndarray, w: int, h: int) -> bool:
    x, y = float(pt[0]), float(pt[1])
    return np.isfinite(x) and np.isfinite(y) and 0 <= x < w and 0 <= y < h


def _draw_label(img: np.ndarray, text: str, origin: Tuple[int, int]) -> None:
    x, y = origin
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hand(
    img: np.ndarray,
    pts: np.ndarray,
    side: str,
    confidence: Optional[float],
    draw_indices: bool,
    bones: List[Tuple[int, int]],
    point_to_finger: Dict[int, int],
    finger_color: Dict[int, Tuple[int, int, int]],
) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = img.copy()

    for i, j in bones:
        if _valid_pt(pts[i], w, h) and _valid_pt(pts[j], w, h):
            p1 = tuple(np.round(pts[i]).astype(int))
            p2 = tuple(np.round(pts[j]).astype(int))
            cv2.line(overlay, p1, p2, (220, 220, 220), 2, cv2.LINE_AA)

    for idx, pt in enumerate(pts):
        if not _valid_pt(pt, w, h):
            continue
        p = tuple(np.round(pt).astype(int))
        finger_id = point_to_finger.get(idx, idx)
        color = finger_color.get(finger_id, (200, 200, 200))
        cv2.circle(overlay, p, 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(overlay, p, 3, color, -1, cv2.LINE_AA)
        if draw_indices:
            cv2.putText(overlay, str(idx), (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    label = side
    if confidence is not None:
        label += f" conf={confidence:.2f}"
    valid = pts[np.array([_valid_pt(p, w, h) for p in pts])]
    if len(valid) > 0:
        x, y = np.min(valid, axis=0).astype(int)
        _draw_label(overlay, label, (max(4, x), max(18, y - 8)))

    return overlay


def visualize_session(
    session_path: str,
    json_name: str,
    side: str,
    out_path: str,
    fps: float,
    draw_indices: bool,
    save_frames: bool,
    max_frames: Optional[int],
    out_frames_dir: Optional[str] = None,
    rgb_dir: Optional[str] = None,
) -> Dict[str, int]:
    frame_dirs = _iter_frame_dirs(session_path, max_frames)
    if not frame_dirs:
        raise FileNotFoundError(f"No numeric frame directories under {session_path}/preprocess/all_data")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frames_out_dir = out_frames_dir or (os.path.splitext(out_path)[0] + "_frames")
    if save_frames:
        os.makedirs(frames_out_dir, exist_ok=True)

    use_mp_order = os.path.basename(json_name) != "aria_hands.json"
    bones = MP_HAND_BONES if use_mp_order else ARIA_HAND_BONES
    point_to_finger = MP_POINT_TO_FINGER if use_mp_order else ARIA_POINT_TO_FINGER
    finger_color = MP_FINGER_COLOR if use_mp_order else ARIA_FINGER_COLOR

    writer = None
    stats = {"frames": 0, "written": 0, "right": 0, "left": 0, "missing_json": 0, "missing_rgb": 0}
    timestamp_path = os.path.join(session_path, "preprocess", "timestamps.jsonl")
    timestamp_by_frame = {}
    if os.path.isfile(timestamp_path):
        with open(timestamp_path, "r", encoding="utf-8") as handle:
            timestamp_by_frame = {
                str(row.get("frame")): row.get("rgb_stamp_ns")
                for row in (json.loads(line) for line in handle if line.strip())
            }
    repeats = repeat_counts(
        [{"rgb_stamp_ns": timestamp_by_frame.get(os.path.basename(frame_dir))} for frame_dir in frame_dirs],
        output_fps=fps,
    )

    sides: Iterable[Tuple[str, str]]
    if side == "both":
        sides = (("right", "hand_r"), ("left", "hand_l"))
    elif side == "right":
        sides = (("right", "hand_r"),)
    else:
        sides = (("left", "hand_l"),)

    for frame_i, frame_dir in enumerate(tqdm(frame_dirs, desc="Visualizing hands")):
        stats["frames"] += 1
        frame_name = os.path.basename(frame_dir)
        rgb_path = os.path.join(rgb_dir, f"{frame_name}.png") if rgb_dir else os.path.join(frame_dir, "rgb.png")
        json_path = os.path.join(frame_dir, json_name)

        img = cv2.imread(rgb_path)
        if img is None:
            stats["missing_rgb"] += 1
            continue
        if not os.path.exists(json_path):
            stats["missing_json"] += 1
            data = {}
        else:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        for side_label, key in sides:
            hand = data.get(key)
            pts = _load_points(hand)
            if pts is None:
                continue
            stats[side_label] += 1
            img = draw_hand(
                img=img,
                pts=pts,
                side=side_label,
                confidence=hand.get("confidence"),
                draw_indices=draw_indices,
                bones=bones,
                point_to_finger=point_to_finger,
                finger_color=finger_color,
            )

        _draw_label(img, f"{frame_name}  {json_name}", (10, 24))

        if writer is None:
            h, w = img.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {out_path}")

        for _ in range(repeats[frame_i]):
            writer.write(img)
            stats["written"] += 1

        if save_frames:
            cv2.imwrite(os.path.join(frames_out_dir, f"{frame_name}.png"), img)

    if writer is not None:
        writer.release()

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize saved 21-point hand keypoints.")
    parser.add_argument("--session_path", "--mps_path", dest="session_path", required=True)
    parser.add_argument("--method", choices=sorted(METHOD_TO_JSON.keys()), default="wilor")
    parser.add_argument("--json_name", default=None, help="Override JSON filename, e.g. hamer_hands.json")
    parser.add_argument("--side", choices=["both", "right", "left"], default="both")
    parser.add_argument("--out_path", default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--draw_indices", action="store_true")
    parser.add_argument("--save_frames", action="store_true")
    parser.add_argument("--out_frames_dir", default=None)
    parser.add_argument("--rgb_dir", default=None, help="Optional image directory for video background, e.g. vis_output_1")
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    json_name = args.json_name or METHOD_TO_JSON[args.method]
    out_path = args.out_path or os.path.join(
        args.session_path, "preprocess", "vis", f"{args.method}_21kpts_vis.mp4"
    )

    stats = visualize_session(
        session_path=args.session_path,
        json_name=json_name,
        side=args.side,
        out_path=out_path,
        fps=args.fps,
        draw_indices=args.draw_indices,
        save_frames=args.save_frames,
        max_frames=args.max_frames,
        out_frames_dir=args.out_frames_dir,
        rgb_dir=args.rgb_dir,
    )
    print(f"[VisualizeHandKpts] Saved: {out_path}")
    print(f"[VisualizeHandKpts] Stats: {stats}")


if __name__ == "__main__":
    main()
