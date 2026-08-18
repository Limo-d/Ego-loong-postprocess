#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run HaMeR 21-keypoint estimation from LocateAnything/GroundingDINO bbox JSON.

Default paths are set for the current white_glove_high/preprocess directory.
The script writes:
  1. per-frame JSONs: all_data/<frame>/locate_hamer_hands.json
  2. aggregate JSON: locate_hamer_21kpts.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
for _proxy_key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    _proxy_val = os.environ.get(_proxy_key)
    if _proxy_val and _proxy_val.startswith("socks://"):
        os.environ[_proxy_key] = "socks5://" + _proxy_val[len("socks://"):]

import cv2
import numpy as np
from tqdm import tqdm

def find_project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "preprocess" / "HaMeRHands.py").is_file():
            return path
    fallback = Path("/home/rx01285/HumanEgo")
    if (fallback / "preprocess" / "HaMeRHands.py").is_file():
        return fallback
    raise FileNotFoundError("Cannot find HumanEgo project root with preprocess/HaMeRHands.py")


PROJECT_ROOT = find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess.HandsTypes import HandData, HandsJointAngles
from preprocess.HaMeRHands import HaMeRModel, remap_mp_to_aria
from preprocess.MediaPipeHands import _build_aria_cam_from_disk
from preprocess.VisualizeHandKpts import visualize_session


MP_KEYPOINT_NAMES = [
    "Wrist",
    "ThumbCMC",
    "ThumbMCP",
    "ThumbIP",
    "ThumbTip",
    "IndexMCP",
    "IndexPIP",
    "IndexDIP",
    "IndexTip",
    "MiddleMCP",
    "MiddlePIP",
    "MiddleDIP",
    "MiddleTip",
    "RingMCP",
    "RingPIP",
    "RingDIP",
    "RingTip",
    "PinkyMCP",
    "PinkyPIP",
    "PinkyDIP",
    "PinkyTip",
]

DEFAULT_WRIST_TO_MIDDLE_MCP_M = 0.085


def infer_frame_digits(session_path: Path, bbox_by_frame: Dict[str, List[Dict[str, Any]]]) -> int:
    widths = [len(key) for key in bbox_by_frame if key.isdigit()]
    all_data_dir = session_path / "preprocess" / "all_data"
    if all_data_dir.is_dir():
        widths.extend(
            len(path.name)
            for path in all_data_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        )
    return max(widths) if widths else 5


def safe_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def recover_absolute_3d_from_hamer(
    kpts_3d_hamer: np.ndarray,
    kpts_2d_hamer: np.ndarray,
    k: np.ndarray,
    wrist_to_middle_mcp_m: float,
    min_depth_m: float,
    max_depth_m: float,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Estimate camera-space depth from a physical palm-length constraint."""
    kpts_3d_hamer = np.asarray(kpts_3d_hamer, dtype=np.float32)
    kpts_2d_hamer = np.asarray(kpts_2d_hamer, dtype=np.float32)
    k = np.asarray(k, dtype=np.float64)
    details: Dict[str, Any] = {
        "applied": False,
        "method": "pinhole_wrist_middle_mcp",
        "assumed_wrist_to_middle_mcp_m": float(wrist_to_middle_mcp_m),
    }

    if kpts_3d_hamer.shape != (21, 3) or kpts_2d_hamer.shape != (21, 2) or k.shape != (3, 3):
        details["reason"] = "invalid_shape"
        return None, details
    if not np.isfinite(kpts_3d_hamer).all() or not np.isfinite(kpts_2d_hamer).all() or not np.isfinite(k).all():
        details["reason"] = "non_finite_value"
        return None, details

    wrist_2d = kpts_2d_hamer[0]
    middle_mcp_2d = kpts_2d_hamer[9]
    pixel_dist = float(np.linalg.norm(middle_mcp_2d - wrist_2d))
    details["wrist_to_middle_mcp_px"] = pixel_dist
    if pixel_dist < 5.0:
        details["reason"] = "palm_span_too_small"
        return None, details

    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        details["reason"] = "invalid_focal_length"
        return None, details

    focal = 0.5 * (fx + fy)
    z_wrist = focal * wrist_to_middle_mcp_m / pixel_dist
    details["estimated_wrist_depth_m"] = float(z_wrist)
    if not min_depth_m <= z_wrist <= max_depth_m:
        details["reason"] = "estimated_depth_out_of_range"
        return None, details

    relative_offsets = kpts_3d_hamer - kpts_3d_hamer[0:1]
    relative_palm_span = float(np.linalg.norm(relative_offsets[9]))
    details["hamer_relative_wrist_to_middle_mcp_m"] = relative_palm_span
    if relative_palm_span < 0.01:
        details["reason"] = "invalid_hamer_relative_palm_span"
        return None, details

    relative_scale = wrist_to_middle_mcp_m / relative_palm_span
    details["hamer_relative_scale"] = float(relative_scale)
    wrist_cam = np.array(
        [
            (wrist_2d[0] - cx) * z_wrist / fx,
            (wrist_2d[1] - cy) * z_wrist / fy,
            z_wrist,
        ],
        dtype=np.float32,
    )
    kpts_cam = wrist_cam[np.newaxis, :] + relative_offsets * relative_scale
    if np.any(kpts_cam[:, 2] <= 0.01):
        details["reason"] = "corrected_keypoint_behind_camera"
        return None, details

    details["applied"] = True
    return kpts_cam.astype(np.float32), details


def load_bbox_json(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"Unsupported bbox json format: {path}")

    by_frame: Dict[str, List[Dict[str, Any]]] = {}
    for item in frames:
        frame = str(item.get("frame", "")).zfill(5)
        detections = item.get("detections", [])
        if isinstance(detections, list):
            by_frame[frame] = detections
    return by_frame


def select_detections(
    detections: Iterable[Dict[str, Any]],
    max_boxes: int,
    score_thresh: float,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for det in detections:
        if "bbox" not in det:
            continue
        bbox = np.asarray(det["bbox"], dtype=np.float32)
        if bbox.shape[0] < 4:
            continue
        x1, y1, x2, y2 = bbox[:4].tolist()
        if x2 <= x1 or y2 <= y1:
            continue
        score = float(det.get("score", det.get("confidence", 1.0)))
        if score < score_thresh:
            continue
        selected.append(
            {
                "bbox": bbox[:4],
                "score": score,
                "label": det.get("label", "hand"),
                "side": det.get("side"),
                "track_id": det.get("track_id"),
            }
        )

    selected.sort(key=lambda item: item["score"], reverse=True)
    return selected[:max_boxes]


def bbox_center(bbox: np.ndarray) -> np.ndarray:
    return np.asarray([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float32)


def infer_is_right(bbox: np.ndarray, image_width: int, handedness: str) -> Optional[bool]:
    cx = float((bbox[0] + bbox[2]) * 0.5)
    if handedness == "image_left_is_right":
        return cx < image_width * 0.5
    if handedness == "image_right_is_right":
        return cx >= image_width * 0.5
    if handedness == "all_right":
        return True
    if handedness == "all_left":
        return False
    return None


def bootstrap_handedness_by_x(detections: List[Dict[str, Any]]) -> List[Optional[bool]]:
    sides: List[Optional[bool]] = [None] * len(detections)
    if len(detections) == 1:
        return sides
    x_order = sorted(range(len(detections)), key=lambda idx: float(bbox_center(detections[idx]["bbox"])[0]))
    split = (len(x_order) + 1) // 2
    for rank, det_idx in enumerate(x_order):
        sides[det_idx] = rank < split
    return sides


def assign_handedness(
    detections: List[Dict[str, Any]],
    image_width: int,
    handedness: str,
    prev_anchors: Dict[str, Optional[np.ndarray]],
    track_max_jump: float,
) -> List[Optional[bool]]:
    explicit: List[Optional[bool]] = []
    has_explicit = False
    for det in detections:
        side = str(det.get("side") or "").lower()
        if side in ("right", "hand_r"):
            explicit.append(True)
            has_explicit = True
        elif side in ("left", "hand_l"):
            explicit.append(False)
            has_explicit = True
        else:
            explicit.append(None)
    if has_explicit and all(side is not None for side in explicit):
        return explicit
    if handedness != "track":
        return [infer_is_right(det["bbox"], image_width, handedness) for det in detections]

    sides: List[Optional[bool]] = [None] * len(detections)
    if not detections:
        return sides

    candidates = []
    for side_name, is_right in (("right", True), ("left", False)):
        anchor = prev_anchors.get(side_name)
        if anchor is None:
            continue
        for det_idx, det in enumerate(detections):
            dist = float(np.linalg.norm(bbox_center(det["bbox"]) - anchor))
            if dist <= track_max_jump:
                candidates.append((dist, side_name, is_right, det_idx))

    used_sides = set()
    used_dets = set()
    for _, side_name, is_right, det_idx in sorted(candidates, key=lambda item: item[0]):
        if side_name in used_sides or det_idx in used_dets:
            continue
        sides[det_idx] = is_right
        used_sides.add(side_name)
        used_dets.add(det_idx)

    # Bootstrap only when there is no previous identity at all. This avoids the
    # old behavior where every frame re-sorted hands by x and flipped identities.
    if not used_sides and prev_anchors.get("right") is None and prev_anchors.get("left") is None:
        return bootstrap_handedness_by_x(detections)
    return sides


def hand_anchor_2d(hand: Optional[HandData]) -> Optional[np.ndarray]:
    if hand is None or hand.hand_keypoints_2d is None:
        return None
    pts = np.asarray(hand.hand_keypoints_2d, dtype=np.float32)
    for idx in (20, 5):
        if pts.shape[0] > idx and np.all(np.isfinite(pts[idx])):
            return pts[idx, :2].copy()
    valid = pts[np.isfinite(pts).all(axis=1)]
    if len(valid) == 0:
        return None
    return valid[:, :2].mean(axis=0)


def build_wrist_pose_from_native_kpts(kpts_cam: np.ndarray) -> np.ndarray:
    """Build an approximate wrist pose from native HaMeR/Mediapipe 21-point order."""
    pts = np.asarray(kpts_cam, dtype=np.float64)
    wrist_pos = pts[0]
    wrist_pose = np.eye(4, dtype=np.float64)
    wrist_pose[:3, 3] = wrist_pos

    if pts.shape[0] < 10 or not np.isfinite(pts[:10]).all():
        return wrist_pose

    # Native order: 0 wrist, 5 index MCP, 9 middle MCP, 17 pinky MCP.
    palm_center = (pts[0] + pts[5] + pts[9] + pts[17]) / 4.0 if pts.shape[0] > 17 else (pts[0] + pts[5] + pts[9]) / 3.0
    y_axis = palm_center - wrist_pos
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-6:
        return wrist_pose
    y_axis /= y_norm

    lateral = pts[5] - pts[17] if pts.shape[0] > 17 else pts[5] - pts[9]
    lat_norm = np.linalg.norm(lateral)
    if lat_norm < 1e-6:
        return wrist_pose
    lateral /= lat_norm

    z_axis = np.cross(lateral, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        return wrist_pose
    z_axis /= z_norm

    x_axis = np.cross(y_axis, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        return wrist_pose
    x_axis /= x_norm
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis) + 1e-8

    wrist_pose[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    return wrist_pose


def make_hand_data(
    kpts_cam_aria: np.ndarray,
    kpts_2d_aria: np.ndarray,
    confidence: float,
    c2w: np.ndarray,
    is_right: bool,
    mano_vertices_3d: Optional[np.ndarray] = None,
) -> HandData:
    d2c = np.eye(4, dtype=np.float64)
    wrist_pose = build_wrist_pose_from_native_kpts(kpts_cam_aria)

    thumb_tip = kpts_cam_aria[4]
    index_tip = kpts_cam_aria[8]
    wrist = kpts_cam_aria[0]
    middle_mcp = kpts_cam_aria[9]
    distance = float(np.linalg.norm(thumb_tip - index_tip))
    palm_size = float(np.linalg.norm(middle_mcp - wrist))
    grasp_state = 1 if palm_size > 0.01 and distance / palm_size < 1.0 else 0

    hand_data = HandData(
        d2c=d2c,
        c2w=c2w,
        is_right=is_right,
        confidence=confidence,
        wrist_pose=wrist_pose,
        palm_pose=wrist_pose,
        hand_keypoints_3d=kpts_cam_aria,
        hand_keypoints_2d=kpts_2d_aria,
        grasp_state=grasp_state,
        joint_angles=HandsJointAngles(data={}),
    )
    if mano_vertices_3d is not None:
        hand_data.mano_vertices_3d = np.asarray(mano_vertices_3d, dtype=np.float32)
    return hand_data


def pack_hand(hand: Optional[HandData]) -> Optional[Dict[str, Any]]:
    if hand is None:
        return None
    joint_angles = hand.joint_angles.data if hand.joint_angles else {}
    return {
        "d2c": safe_list(hand.d2c),
        "c2w": safe_list(hand.c2w),
        "confidence": safe_list(hand.confidence),
        "grasp_state": safe_list(hand.grasp_state),
        "wrist_pose": safe_list(hand.wrist_pose),
        "palm_pose": safe_list(hand.palm_pose),
        "kpts_3d": safe_list(hand.hand_keypoints_3d),
        "kpts_2d": safe_list(hand.hand_keypoints_2d),
        "mano_vertices_3d": safe_list(getattr(hand, "mano_vertices_3d", None)),
        "joint_angles": {k: safe_list(v) for k, v in joint_angles.items()},
        "wrist_pose_raw_world": None,
        "wrist_pose_opt_world": None,
        "wrist_lin_vel_raw_world": [0.0, 0.0, 0.0],
        "wrist_ang_vel_raw_world": [0.0, 0.0, 0.0],
        "wrist_lin_vel_opt_world": [0.0, 0.0, 0.0],
        "wrist_ang_vel_opt_world": [0.0, 0.0, 0.0],
        "index_translation_raw_world": None,
        "index_translation_opt_world": None,
        "thumb_translation_raw_world": None,
        "thumb_translation_opt_world": None,
        "midpoint_pose_raw_world": None,
        "midpoint_pose_opt_world": None,
        "midpoint_translation_raw_world": None,
        "midpoint_orientation_raw_world": None,
        "midpoint_translation_opt_world": None,
        "midpoint_orientation_opt_world": None,
        "midpoint_lin_vel_raw_world": [0.0, 0.0, 0.0],
        "midpoint_ang_vel_raw_world": [0.0, 0.0, 0.0],
        "midpoint_lin_vel_opt_world": [0.0, 0.0, 0.0],
        "midpoint_ang_vel_opt_world": [0.0, 0.0, 0.0],
        "distance_midpoint2wrist_raw_world": None,
        "distance_midpoint2wrist_opt_world": None,
    }


def write_frame_json(
    output_dir: Path,
    frame_idx: int,
    frame_key: str,
    ts: int,
    hand_r: Optional[HandData],
    hand_l: Optional[HandData],
    filename: str,
) -> None:
    frame_dir = output_dir / frame_key
    frame_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "idx": frame_idx,
        "ts": safe_list(ts),
        "hand_r": pack_hand(hand_r),
        "hand_l": pack_hand(hand_l),
    }
    with (frame_dir / filename).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    preprocess_dir = Path(__file__).resolve().parent
    session_path = Path(args.session_path).expanduser().resolve() if args.session_path else preprocess_dir.parent
    bbox_json = Path(args.bbox_json).expanduser().resolve()
    fallback_bbox_json = Path(args.fallback_bbox_json).expanduser().resolve() if args.fallback_bbox_json else None
    aggregate_json = Path(args.aggregate_json).expanduser().resolve()
    out_video = Path(args.out_video).expanduser().resolve() if args.out_video else (
        session_path / "preprocess" / "vis" / "locate_hamer_21kpts_vis.mp4"
    )
    per_frame_output_dir = (
        Path(args.per_frame_output_dir).expanduser().resolve()
        if args.per_frame_output_dir
        else session_path / "preprocess" / "all_data"
    )

    bbox_by_frame: Dict[str, List[Dict[str, Any]]] = {}
    bbox_source_by_frame: Dict[str, str] = {}
    bbox_paths = []
    if fallback_bbox_json is not None:
        bbox_paths.append(fallback_bbox_json)
    bbox_paths.append(bbox_json)
    for path in bbox_paths:
        loaded = load_bbox_json(path)
        for frame_key, detections in loaded.items():
            bbox_by_frame[frame_key] = detections
            bbox_source_by_frame[frame_key] = str(path)
    frame_digits = infer_frame_digits(session_path, bbox_by_frame)

    aria_cam = _build_aria_cam_from_disk(str(session_path), args.camera_json_dir)

    cam_frames = aria_cam.cam
    if args.frame_start is not None:
        cam_frames = [frame for frame in cam_frames if frame.idx >= args.frame_start]
    if args.frame_end is not None:
        cam_frames = [frame for frame in cam_frames if frame.idx <= args.frame_end]
    if args.max_frames is not None:
        cam_frames = cam_frames[: args.max_frames]

    model = HaMeRModel(device=args.device)
    if not model.is_available:
        raise RuntimeError("HaMeR model is not available. Check the hamer environment and checkpoint files.")

    stats = {
        "frames_total": len(cam_frames),
        "frames_with_bbox": 0,
        "bbox_used": 0,
        "hamer_ok": 0,
        "depth_correction_applied": 0,
        "depth_correction_failed": 0,
        "hand_r": 0,
        "hand_l": 0,
        "failed_frames": [],
    }
    aggregate_frames: List[Dict[str, Any]] = []
    prev_anchors: Dict[str, Optional[np.ndarray]] = {"right": None, "left": None}

    for cam_data in tqdm(cam_frames, desc="HaMeR from LocateAnything bboxes"):
        frame_key = str(cam_data.idx).zfill(frame_digits)
        raw_detections = bbox_by_frame.get(frame_key, [])
        detections = select_detections(
            raw_detections,
            max_boxes=args.max_boxes,
            score_thresh=args.score_thresh,
        )
        if detections:
            stats["frames_with_bbox"] += 1

        img_bgr = cam_data.img
        if img_bgr is None:
            img_path = session_path / "preprocess" / "all_data" / frame_key / "rgb.png"
            img_bgr = cv2.imread(str(img_path))

        frame_results: List[Dict[str, Any]] = []
        hand_r: Optional[HandData] = None
        hand_l: Optional[HandData] = None

        if img_bgr is None:
            stats["failed_frames"].append({"frame": frame_key, "reason": "missing rgb.png"})
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            image_h, image_w = img_bgr.shape[:2]
            focal = float((cam_data.k[0, 0] + cam_data.k[1, 1]) * 0.5)
            inferred_sides = assign_handedness(
                detections=detections,
                image_width=image_w,
                handedness=args.handedness,
                prev_anchors=prev_anchors,
                track_max_jump=args.track_max_jump,
            )

            for det_idx, det in enumerate(detections):
                bbox = det["bbox"]
                is_right = inferred_sides[det_idx]
                if is_right is None:
                    # HaMeR needs a side flag for mirroring. Use right-hand mode for
                    # inference, but keep side unknown in the aggregate output.
                    hamer_side_flag = 1
                else:
                    hamer_side_flag = 1 if is_right else 0

                hamer_result = model.predict_from_crop(
                    img_rgb=img_rgb,
                    bbox=bbox,
                    is_right=hamer_side_flag,
                    focal_length=focal,
                )
                stats["bbox_used"] += 1
                if hamer_result is None:
                    frame_results.append(
                        {
                            "det_idx": det_idx,
                            "label": det["label"],
                            "bbox": safe_list(bbox),
                            "bbox_score": det["score"],
                            "side": "unknown" if is_right is None else ("right" if is_right else "left"),
                            "status": "hamer_failed",
                        }
                    )
                    continue

                kpts_3d_raw_hamer = np.asarray(hamer_result["joints_3d"], dtype=np.float32)
                kpts_2d_hamer = np.asarray(hamer_result["joints_2d"], dtype=np.float32)
                vertices_3d_raw_hamer = None
                vertices_3d_hamer = None
                if hamer_result.get("vertices_3d") is not None:
                    vertices_3d_raw_hamer = np.asarray(hamer_result["vertices_3d"], dtype=np.float32)
                if args.no_depth_correction:
                    kpts_3d_hamer = kpts_3d_raw_hamer
                    vertices_3d_hamer = vertices_3d_raw_hamer
                    depth_correction = {
                        "applied": False,
                        "method": "pinhole_wrist_middle_mcp",
                        "reason": "disabled",
                    }
                else:
                    corrected, depth_correction = recover_absolute_3d_from_hamer(
                        kpts_3d_hamer=kpts_3d_raw_hamer,
                        kpts_2d_hamer=kpts_2d_hamer,
                        k=cam_data.k,
                        wrist_to_middle_mcp_m=args.wrist_to_middle_mcp_m,
                        min_depth_m=args.min_depth_m,
                        max_depth_m=args.max_depth_m,
                    )
                    if corrected is None:
                        stats["depth_correction_failed"] += 1
                        kpts_3d_hamer = kpts_3d_raw_hamer
                        vertices_3d_hamer = vertices_3d_raw_hamer
                        depth_correction["fallback"] = "raw_hamer_camera_translation"
                    else:
                        stats["depth_correction_applied"] += 1
                        kpts_3d_hamer = corrected
                        if vertices_3d_raw_hamer is not None:
                            raw_span = float(np.linalg.norm(kpts_3d_raw_hamer[9] - kpts_3d_raw_hamer[0]))
                            corrected_span = float(np.linalg.norm(kpts_3d_hamer[9] - kpts_3d_hamer[0]))
                            scale = corrected_span / raw_span if raw_span > 1e-6 else 1.0
                            vertices_3d_hamer = (
                                kpts_3d_hamer[0:1]
                                + (vertices_3d_raw_hamer - kpts_3d_raw_hamer[0:1]) * scale
                            ).astype(np.float32)

                kpts_3d_aria = remap_mp_to_aria(kpts_3d_hamer)
                kpts_3d_raw_aria = remap_mp_to_aria(kpts_3d_raw_hamer)
                kpts_2d_aria = remap_mp_to_aria(
                    np.column_stack([kpts_2d_hamer, np.zeros(21, dtype=np.float32)])
                )[:, :2]
                confidence = float(det["score"]) * float(hamer_result.get("confidence", 1.0))
                side_name = "unknown" if is_right is None else ("right" if is_right else "left")

                frame_results.append(
                    {
                        "det_idx": det_idx,
                        "label": det["label"],
                        "bbox": safe_list(bbox),
                        "bbox_score": det["score"],
                        "side": side_name,
                        "confidence": confidence,
                        "kpts_2d_aria": safe_list(kpts_2d_aria),
                        "kpts_3d_camera_aria": safe_list(kpts_3d_aria),
                        "kpts_3d_camera_aria_raw_hamer": safe_list(kpts_3d_raw_aria),
                        "mano_vertices_3d_camera": safe_list(vertices_3d_hamer),
                        "depth_correction": depth_correction,
                    }
                )
                stats["hamer_ok"] += 1

                if is_right is None:
                    continue
                hand_data = make_hand_data(
                    kpts_cam_aria=kpts_3d_aria,
                    kpts_2d_aria=kpts_2d_aria,
                    confidence=confidence,
                    c2w=cam_data.c2w,
                    is_right=is_right,
                    mano_vertices_3d=vertices_3d_hamer,
                )
                if is_right:
                    if hand_r is None or confidence > float(hand_r.confidence):
                        hand_r = hand_data
                else:
                    if hand_l is None or confidence > float(hand_l.confidence):
                        hand_l = hand_data

        if hand_r is not None:
            stats["hand_r"] += 1
            prev_anchors["right"] = hand_anchor_2d(hand_r)
        if hand_l is not None:
            stats["hand_l"] += 1
            prev_anchors["left"] = hand_anchor_2d(hand_l)

        if not args.no_per_frame:
            write_frame_json(
                output_dir=per_frame_output_dir,
                frame_idx=cam_data.idx,
                frame_key=frame_key,
                ts=cam_data.ts,
                hand_r=hand_r,
                hand_l=hand_l,
                filename=args.out_json_name,
            )

        aggregate_frames.append(
            {
                "frame": frame_key,
                "rgb_path": str(session_path / "preprocess" / "all_data" / frame_key / "rgb.png"),
                "ts": safe_list(cam_data.ts),
                "bbox_source": bbox_source_by_frame.get(frame_key),
                "detections": frame_results,
            }
        )

    aggregate = {
        "session_path": str(session_path),
        "bbox_json": str(bbox_json),
        "fallback_bbox_json": str(fallback_bbox_json) if fallback_bbox_json is not None else None,
        "per_frame_json_name": None if args.no_per_frame else args.out_json_name,
        "per_frame_output_dir": None if args.no_per_frame else str(per_frame_output_dir),
        "frame_digits": frame_digits,
        "handedness": args.handedness,
        "track_max_jump": args.track_max_jump,
        "depth_correction": {
            "enabled": not args.no_depth_correction,
            "method": "pinhole_wrist_middle_mcp",
            "wrist_to_middle_mcp_m": args.wrist_to_middle_mcp_m,
            "min_depth_m": args.min_depth_m,
            "max_depth_m": args.max_depth_m,
        },
        "keypoint_order": "mediapipe_mano_native",
        "keypoint_names": MP_KEYPOINT_NAMES,
        "hamer_model": {
            "checkpoint_path": getattr(model, "checkpoint_path", None),
            "cache_dir": getattr(model, "cache_dir", None),
            "mano_path": getattr(model, "mano_path", None),
            "mano_mean_params_path": getattr(model, "mano_mean_params_path", None),
            "device": str(getattr(model, "device", "")),
        },
        "mano_faces": safe_list(model.faces),
        "stats": stats,
        "frames": aggregate_frames,
    }
    aggregate_json.parent.mkdir(parents=True, exist_ok=True)
    with aggregate_json.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=4)

    if not args.no_video:
        if args.no_per_frame:
            print("[locate-hamer] skip video because --no_per_frame was set")
        else:
            vis_stats = visualize_session(
                session_path=str(session_path),
                json_name=args.out_json_name,
                side="both",
                out_path=str(out_video),
                fps=args.fps,
                draw_indices=args.draw_indices,
                save_frames=args.save_video_frames,
                max_frames=args.max_frames,
                out_frames_dir=args.out_frames_dir,
                rgb_dir=args.vis_rgb_dir,
            )
            aggregate["video_path"] = str(out_video)
            aggregate["vis_rgb_dir"] = args.vis_rgb_dir
            aggregate["out_frames_dir"] = args.out_frames_dir
            aggregate["vis_stats"] = vis_stats
            with aggregate_json.open("w", encoding="utf-8") as f:
                json.dump(aggregate, f, indent=4)

    if args.out_mano_video:
        if args.no_per_frame:
            print("[locate-hamer] skip MANO video because --no_per_frame was set")
        elif model.faces is None:
            print("[locate-hamer] skip MANO video because MANO faces are unavailable")
        else:
            from preprocess.VisualizeMANOMesh import visualize_mano_session

            mano_stats = visualize_mano_session(
                session_path=str(session_path),
                json_name=args.out_json_name,
                out_path=str(Path(args.out_mano_video).expanduser().resolve()),
                faces=model.faces,
                side="r" if args.handedness != "all_left" else "l",
                fps=args.fps,
                max_frames=args.max_frames,
                alpha=args.mano_alpha,
            )
            aggregate["mano_video_path"] = str(Path(args.out_mano_video).expanduser().resolve())
            aggregate["mano_vis_stats"] = mano_stats
            with aggregate_json.open("w", encoding="utf-8") as f:
                json.dump(aggregate, f, indent=4)

    return aggregate


def build_parser() -> argparse.ArgumentParser:
    preprocess_dir = Path(__file__).resolve().parent
    data_preprocess_dir = PROJECT_ROOT / "data" / "hand_kpts" / "white_glove_high" / "preprocess"
    default_session = data_preprocess_dir.parent
    parser = argparse.ArgumentParser(
        description="Estimate HaMeR 21 hand keypoints from LocateAnything/GroundingDINO bboxes."
    )
    parser.add_argument("--session_path", default=str(default_session), help="Session root, e.g. .../white_glove_high")
    parser.add_argument("--camera_json_dir", default=None, help="Optional per-frame camera metadata root, e.g. RTAB-Map pose output.")
    parser.add_argument(
        "--bbox_json",
        default=str(data_preprocess_dir / "vis_output" / "locateanything_gray_glove_bboxes.json"),
        help="Primary bbox JSON. This overrides fallback frames when both contain the same frame.",
    )
    parser.add_argument(
        "--fallback_bbox_json",
        default=str(data_preprocess_dir / "groundingdino_white_glove_bboxes_locate.json"),
        help="Fallback bbox JSON for frames missing from --bbox_json, e.g. frames before 00513.",
    )
    parser.add_argument("--aggregate_json", default=str(data_preprocess_dir / "locate_gray_glove_hamer_21kpts.json"))
    parser.add_argument("--out_json_name", default="locate_gray_glove_hamer_hands.json")
    parser.add_argument(
        "--per_frame_output_dir",
        default=None,
        help="Directory containing one subdirectory per frame for derived JSON output. Defaults to preprocess/all_data for backward compatibility.",
    )
    parser.add_argument(
        "--out_video",
        default=str(data_preprocess_dir / "vis" / "locate_gray_glove_hamer_21kpts_vis.mp4"),
        help="Output mp4 with the 21 keypoints drawn on RGB frames",
    )
    parser.add_argument("--out_mano_video", default=None, help="Optional mp4 with MANO mesh projected on RGB frames")
    parser.add_argument("--mano_alpha", type=float, default=0.42, help="Transparency for MANO mesh overlay")
    parser.add_argument("--device", default="cuda", help="cuda or cpu; falls back to cpu if CUDA is unavailable")
    parser.add_argument("--max_boxes", type=int, default=2, help="Use top-N boxes per frame after score filtering")
    parser.add_argument("--score_thresh", type=float, default=0.0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument(
        "--handedness",
        default="track",
        choices=["track", "image_left_is_right", "image_right_is_right", "all_right", "all_left", "unknown"],
        help="How to assign bbox detections to hands. Default track uses previous-frame position instead of per-frame x sorting.",
    )
    parser.add_argument("--track_max_jump", type=float, default=220.0, help="Max pixel jump for track-based hand identity matching")
    parser.add_argument(
        "--wrist_to_middle_mcp_m",
        type=float,
        default=DEFAULT_WRIST_TO_MIDDLE_MCP_M,
        help="Assumed physical wrist-to-middle-MCP distance used for monocular depth correction",
    )
    parser.add_argument("--min_depth_m", type=float, default=0.05, help="Minimum accepted corrected wrist depth in meters")
    parser.add_argument("--max_depth_m", type=float, default=3.0, help="Maximum accepted corrected wrist depth in meters")
    parser.add_argument("--no_depth_correction", action="store_true", help="Keep HaMeR camera translation without intrinsics-based depth correction")
    parser.add_argument("--no_per_frame", action="store_true", help="Only write the aggregate JSON")
    parser.add_argument("--no_video", action="store_true", help="Do not render the 21-keypoint mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--vis_rgb_dir", default=None, help="Optional image directory for video background, e.g. .../vis_output_1")
    parser.add_argument("--out_frames_dir", default=None, help="Directory for rendered frames when --save_video_frames is set")
    parser.add_argument("--draw_indices", action="store_true", help="Draw keypoint indices 0..20 on the video")
    parser.add_argument("--save_video_frames", action="store_true", help="Also save rendered frames next to the mp4 or --out_frames_dir")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run(args)
    stats = result["stats"]
    print(f"[locate-hamer] aggregate: {args.aggregate_json}")
    if not args.no_per_frame:
        print(f"[locate-hamer] per-frame filename: {args.out_json_name}")
    if not args.no_video:
        print(f"[locate-hamer] video: {args.out_video}")
    if args.out_mano_video:
        print(f"[locate-hamer] mano video: {args.out_mano_video}")
    print(f"[locate-hamer] stats: {stats}")


if __name__ == "__main__":
    main()
