#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the review web page for one postprocess session.

Outputs:
  outputs/web/index.html
  outputs/web/rgb_frames/*.jpg
  outputs/web/tactile_hand.png

RGB uses frame-based playback rather than browser video decoding. Trajectory
and tactile panels are rendered directly from embedded data with Canvas.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess.Timebase import nominal_fps, relative_times_sec


BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

FINGER_COLORS = [
    (196, 64, 255),
    (77, 154, 255),
    (79, 214, 132),
    (255, 169, 65),
    (196, 113, 255),
]
C_HEAD = (210, 210, 80)
C_WRIST = (255, 205, 85)
C_TEXT = (42, 52, 68)
C_AXIS = [(64, 84, 239), (86, 200, 92), (235, 146, 62)]

TACTILE_SENSOR_COUNT = 68
TACTILE_FRAME_W = 676
TACTILE_FRAME_H = 328
TRAJECTORY_FRAME_W = 676
TRAJECTORY_FRAME_H = 328
TACTILE_BASELINE_FRAMES = 24
TACTILE_MIN_SCALE = 0.02
TACTILE_DISPLAY_SCALE_MULTIPLIER = 1.60
TACTILE_SENSOR_DEADZONE_PERCENTILE = 60.0
TACTILE_SENSOR_DEADZONE_MIN = 0.08
TACTILE_BG_TOP_BGR = (250, 245, 237)  # Live UI #edf5fa, stored as BGR.
TACTILE_BG_BOTTOM_BGR = (246, 238, 228)
TACTILE_CANVAS_BG = "#e4eef6"
TRAJECTORY_CANVAS_BG = "#e4eef6"
TRAJECTORY_BG_TOP_BGR = (252, 248, 242)
TRAJECTORY_BG_BOTTOM_BGR = (246, 238, 228)
TRAJECTORY_MIRROR_LEFT_HAND = False
TACTILE_HAND_ASSET = ROOT / "tactile" / "assets" / "hand_live.png"

TACTILE_FINGER_POINTS = {
    "A": [(0.876, 0.518), (0.909, 0.502), (0.815, 0.607), (0.850, 0.586)],
    "B": [(0.614, 0.183), (0.643, 0.183), (0.599, 0.344), (0.632, 0.344)],
    "C": [(0.453, 0.115), (0.482, 0.115), (0.453, 0.288), (0.482, 0.288)],
    "D": [(0.294, 0.176), (0.323, 0.176), (0.309, 0.344), (0.342, 0.344)],
    "E": [(0.155, 0.318), (0.184, 0.318), (0.198, 0.483), (0.231, 0.483)],
}
TACTILE_PALM_ROWS = [
    (0.663, 0.247, 0.637),
    (0.702, 0.238, 0.663),
    (0.740, 0.231, 0.692),
    (0.778, 0.243, 0.677),
    (0.817, 0.265, 0.625),
    (0.855, 0.289, 0.570),
]
TACTILE_COLOR_STOPS = [
    # Exact Live UI RGB ramp converted to OpenCV BGR.
    (0.00, (220, 179, 112)),
    (0.45, (200, 191, 71)),
    (0.70, (91, 196, 240)),
    (0.86, (67, 144, 238)),
    (1.00, (83, 83, 211)),
]

# Visual slot -> incoming sensor index. This is the same left-glove correction
# used by Ego-Loong-Live: the two physical wrist rows arrive exchanged.
TACTILE_LEFT_VISUAL_SOURCE = np.arange(TACTILE_SENSOR_COUNT, dtype=np.int64)
TACTILE_LEFT_VISUAL_SOURCE[36:44] = np.arange(60, 68)
TACTILE_LEFT_VISUAL_SOURCE[60:68] = np.arange(36, 44)


def read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def checked_imwrite(path: Path, image: np.ndarray, params: Optional[List[int]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, params or [])
    if not ok:
        raise OSError(f"Failed to write image: {path}")


def load_rows(traj_path: Path) -> tuple[List[Dict[str, Any]], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    rows: List[Dict[str, Any]] = []
    all_points: List[np.ndarray] = []
    wrists: List[np.ndarray] = []
    hand_offsets: List[np.ndarray] = []
    camera_axes: List[np.ndarray] = []
    for line in traj_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        mat = (row.get("head_pose") or {}).get("c2w") or (row.get("camera") or {}).get("c2w")
        item: Dict[str, Any] = {"hand": None, "hand_l": None, "hand_r": None, "head": None, "axes": None}
        hand_sources = row.get("hands") or {}
        if not hand_sources:
            legacy_side = str((row.get("glove") or {}).get("side") or "left")
            hand_sources = {legacy_side: {"glove": row.get("glove")}}
        for side, key in (("left", "hand_l"), ("right", "hand_r")):
            pts = (((hand_sources.get(side) or {}).get("glove") or {}).get("kpts_3d_world_m"))
            if pts is None:
                continue
            arr = np.asarray(pts, dtype=np.float64)
            if arr.shape[0] >= 21 and np.isfinite(arr[:21, :3]).all():
                item[key] = arr[:21, :3]
                if item["hand"] is None:
                    item["hand"] = arr[:21, :3]
                all_points.append(arr[:21, :3])
                wrists.append(arr[0, :3])
                hand_offsets.append(arr[:21, :3] - arr[0, :3][None, :])
        if mat is not None:
            m = np.asarray(mat, dtype=np.float64)
            if m.shape == (4, 4) and np.isfinite(m).all():
                item["head"] = m[:3, 3]
                item["axes"] = m[:3, :3]
                all_points.append(m[:3, 3][None, :])
                camera_axes.append(m[:3, :3])
        rows.append(item)
    if not rows or not all_points:
        raise RuntimeError(f"No valid trajectory rows in {traj_path}")
    return rows, all_points, wrists, hand_offsets, camera_axes


def finger_color(joint_idx: int) -> tuple[int, int, int]:
    if joint_idx <= 4:
        return FINGER_COLORS[0]
    if joint_idx <= 8:
        return FINGER_COLORS[1]
    if joint_idx <= 12:
        return FINGER_COLORS[2]
    if joint_idx <= 16:
        return FINGER_COLORS[3]
    return FINGER_COLORS[4]


class WorldProjector:
    def __init__(self, points: List[np.ndarray], hand_offsets: Optional[List[np.ndarray]] = None, camera_axes: Optional[List[np.ndarray]] = None, wrists: Optional[List[np.ndarray]] = None, width: int = TRAJECTORY_FRAME_W, height: int = TRAJECTORY_FRAME_H, scene_scale: float = 1.0) -> None:
        flat = np.concatenate(points, axis=0)
        lo = np.percentile(flat, 1, axis=0)
        hi = np.percentile(flat, 99, axis=0)
        self.center = (lo + hi) / 2.0
        self.width = width
        self.height = height

        if camera_axes:
            mean_axes = np.mean(np.stack(camera_axes, axis=0), axis=0)
            u, _s, vt = np.linalg.svd(mean_axes, full_matrices=False)
            cam_axes = u @ vt
            # Keep the horizontal direction consistent with the RGB image, but
            # use a shallow lower-front oblique view for the review panel.
            self.view_u = -cam_axes[:, 0].astype(np.float64)
            self.view_u[2] = 0.0
            if np.linalg.norm(self.view_u) < 1e-6:
                self.view_u = cam_axes[:, 0].astype(np.float64)
            self.view_u /= max(np.linalg.norm(self.view_u), 1e-12)
            top = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            oblique_dir = -top * 0.45 - cam_axes[:, 2].astype(np.float64) * 0.18
            oblique_dir -= self.view_u * float(oblique_dir @ self.view_u)
            oblique_dir /= max(np.linalg.norm(oblique_dir), 1e-12)
            self.view_v = np.cross(oblique_dir, self.view_u)
            self.view_v /= max(np.linalg.norm(self.view_v), 1e-12)
        else:
            # Face the hand plane instead of the world axes. This keeps the 21 keypoints
            # visually spread out even when the wrist trajectory is viewed from the side.
            if hand_offsets:
                rel_hand = np.concatenate(hand_offsets, axis=0)
                rel_hand = rel_hand[np.linalg.norm(rel_hand, axis=1) > 1e-6]
            else:
                rel_hand = np.empty((0, 3), dtype=np.float64)
            if rel_hand.shape[0] >= 8:
                _, _s, vt = np.linalg.svd(rel_hand - rel_hand.mean(axis=0, keepdims=True), full_matrices=False)
                self.view_u = vt[0].astype(np.float64)
                palm_v = vt[1].astype(np.float64)
                palm_n = vt[2].astype(np.float64)
                self.view_v = palm_v * 0.92 + palm_n * 0.30
                self.view_v /= max(np.linalg.norm(self.view_v), 1e-12)
            else:
                self.view_u = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
                self.view_v = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)

            # Keep screen X aligned with world X so trajectory movement direction is not mirrored.
            if self.view_u[0] < 0:
                self.view_u *= -1.0

        head_med = np.median(flat, axis=0)
        # Keep head/camera above the hand only for the PCA fallback. For the
        # camera-axis review view, flipping this would invert the requested view side.
        if not camera_axes and hand_offsets and len(points) > 0:
            all_wrist_like = np.asarray([arr[0] for arr in points if len(arr) >= 21], dtype=np.float64)
            if all_wrist_like.size:
                head_delta = head_med - np.median(all_wrist_like, axis=0)
                if float(head_delta @ self.view_v) < 0:
                    self.view_v *= -1.0
        self.view_d = np.cross(self.view_u, self.view_v)
        self.view_d /= max(np.linalg.norm(self.view_d), 1e-12)
        self.view_v = np.cross(self.view_d, self.view_u)
        self.view_v /= max(np.linalg.norm(self.view_v), 1e-12)

        # Rotate only within the current view plane so the wrist trajectory's
        # dominant motion runs left-right on screen, without changing the view side.
        if wrists and len(wrists) >= 5:
            wrist_arr = np.asarray(wrists, dtype=np.float64)
            wrist_arr = wrist_arr[np.isfinite(wrist_arr).all(axis=1)]
            if wrist_arr.shape[0] >= 5:
                centered = wrist_arr - wrist_arr.mean(axis=0, keepdims=True)
                plane = centered - (centered @ self.view_d)[:, None] * self.view_d[None, :]
                if float(np.linalg.norm(plane)) > 1e-6:
                    _u, _s, vt = np.linalg.svd(plane, full_matrices=False)
                    motion_u = vt[0].astype(np.float64)
                    motion_u -= self.view_d * float(motion_u @ self.view_d)
                    motion_u /= max(np.linalg.norm(motion_u), 1e-12)
                    if float(motion_u @ self.view_u) > 0:
                        motion_u *= -1.0
                    self.view_u = motion_u
                    self.view_v = np.cross(self.view_d, self.view_u)
                    self.view_v /= max(np.linalg.norm(self.view_v), 1e-12)

        rel = flat - self.center[None, :]
        u_vals = rel @ self.view_u
        v_vals = rel @ self.view_v
        u_span = float(np.percentile(u_vals, 99) - np.percentile(u_vals, 1))
        v_span = float(np.percentile(v_vals, 99) - np.percentile(v_vals, 1))
        self.base_scale = 0.74 * min(
            width / max(u_span + 0.26, 0.50),
            height / max(v_span + 0.20, 0.42),
        )
        self.scene_scale = float(scene_scale)
        self.scale = self.base_scale * self.scene_scale

    def project(self, point: np.ndarray | List[float]) -> tuple[int, int]:
        rel = np.asarray(point, dtype=np.float64) - self.center
        u = float(rel @ self.view_u)
        v = float(rel @ self.view_v)
        sx = self.width * 0.51 + u * self.scale
        sy = self.height * 0.56 + v * self.scale
        return int(round(sx)), int(round(sy))


def draw_line(img: np.ndarray, a: tuple[int, int], b: tuple[int, int], color, thickness=1, alpha=1.0) -> None:
    if alpha >= 1.0:
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
        return
    overlay = img.copy()
    cv2.line(overlay, a, b, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def draw_back_wall(img: np.ndarray, prj: WorldProjector) -> None:
    # Screen-space back wall grid. It stays behind the trajectory and fills the
    # entire review panel, without adding a floor plane.
    h, w = img.shape[:2]
    minor = (236, 230, 222)
    major = (219, 211, 202)
    minor_step = 28
    major_step = minor_step * 4
    for x in range(-major_step, w + major_step, minor_step):
        is_major = (x // minor_step) % 4 == 0
        color = major if is_major else minor
        alpha = 0.78 if is_major else 0.50
        draw_line(img, (x, 0), (x, h), color, 1, alpha)
    for y in range(-major_step, h + major_step, minor_step):
        is_major = (y // minor_step) % 4 == 0
        color = major if is_major else minor
        alpha = 0.78 if is_major else 0.50
        draw_line(img, (0, y), (w, y), color, 1, alpha)


def rotate_hand_about_wrist_for_display(
    hand: np.ndarray,
    degrees: float,
    scale: float = 1.0,
    view_u: Optional[np.ndarray] = None,
    view_v: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Rotate non-wrist joints around wrist in the current rendered view plane.

    This is visualization-only. Rotating in the projector plane keeps the review
    panel top-down; the older fixed X/Z display rotation would flatten the hand
    once the panel switched to an X/Y top-down view.
    """
    if abs(degrees) < 1e-9 and abs(scale - 1.0) < 1e-9:
        return hand
    out = hand.copy()
    theta = np.deg2rad(degrees)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    root = out[0].copy()
    rel0 = out[1:] - root[None, :]
    if view_u is None or view_v is None:
        rel = rel0 * float(scale)
        x = rel[:, 0].copy()
        z = rel[:, 2].copy()
        rel[:, 0] = c * x - s * z
        rel[:, 2] = s * x + c * z
    else:
        u_axis = np.asarray(view_u, dtype=np.float64)
        v_axis = np.asarray(view_v, dtype=np.float64)
        u_axis /= max(np.linalg.norm(u_axis), 1e-12)
        v_axis -= u_axis * float(v_axis @ u_axis)
        v_axis /= max(np.linalg.norm(v_axis), 1e-12)
        u = rel0 @ u_axis
        v = rel0 @ v_axis
        residual = rel0 - u[:, None] * u_axis[None, :] - v[:, None] * v_axis[None, :]
        u *= float(scale)
        v *= float(scale)
        u_rot = c * u - s * v
        v_rot = s * u + c * v
        rel = u_rot[:, None] * u_axis[None, :] + v_rot[:, None] * v_axis[None, :] + residual
    out[1:] = root[None, :] + rel
    return out


def align_middle_vertical_for_display(hand: np.ndarray, view_u: np.ndarray, view_v: np.ndarray, middle_idx: int = 9) -> np.ndarray:
    """Rotate displayed hand around wrist so wrist->middle_mcp is vertical on screen."""
    if hand.shape[0] <= middle_idx:
        return hand
    root = hand[0]
    vec = hand[middle_idx] - root
    u_axis = np.asarray(view_u, dtype=np.float64)
    v_axis = np.asarray(view_v, dtype=np.float64)
    u_axis /= max(np.linalg.norm(u_axis), 1e-12)
    v_axis -= u_axis * float(v_axis @ u_axis)
    v_axis /= max(np.linalg.norm(v_axis), 1e-12)
    u = float(vec @ u_axis)
    v = float(vec @ v_axis)
    if not np.isfinite([u, v]).all() or (u * u + v * v) < 1e-12:
        return hand
    # Projection uses screen_y = center_y + v * scale. Add pi so the middle
    # direction is vertical and points upward on screen.
    theta = np.arctan2(u, v) + np.pi
    return rotate_hand_about_wrist_for_display(hand, np.degrees(theta), scale=1.0, view_u=u_axis, view_v=v_axis)

def render_trajectory_frames(rows: List[Dict[str, Any]], projector: WorldProjector, out_dir: Path, hand_display_rotate_deg: float = 0.0, hand_display_scale: float = 2.0, align_middle_vertical: bool = True) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = projector.width, projector.height
    wrist_paths: Dict[str, List[np.ndarray]] = {"hand_l": [], "hand_r": []}
    head_path: List[np.ndarray] = []
    for idx, row in enumerate(rows):
        img = np.empty((height, width, 3), dtype=np.uint8)
        top = np.asarray(TRAJECTORY_BG_TOP_BGR, dtype=np.float32)
        bottom = np.asarray(TRAJECTORY_BG_BOTTOM_BGR, dtype=np.float32)
        for yy in range(height):
            t = yy / max(1, height - 1)
            img[yy, :] = np.clip(top * (1.0 - t) + bottom * t, 0, 255).astype(np.uint8)
        draw_back_wall(img, projector)

        hands: Dict[str, Optional[np.ndarray]] = {key: row.get(key) for key in ("hand_l", "hand_r")}
        for key, hand in list(hands.items()):
            if hand is None:
                continue
            if align_middle_vertical:
                hand = rotate_hand_about_wrist_for_display(hand, 0.0, scale=hand_display_scale, view_u=projector.view_u, view_v=projector.view_v)
                hand = align_middle_vertical_for_display(hand, projector.view_u, projector.view_v, middle_idx=9)
            else:
                hand = rotate_hand_about_wrist_for_display(hand, hand_display_rotate_deg, scale=hand_display_scale, view_u=projector.view_u, view_v=projector.view_v)
            hands[key] = hand
        head: Optional[np.ndarray] = row.get("head")
        axes: Optional[np.ndarray] = row.get("axes")
        for key, hand in hands.items():
            if hand is not None:
                wrist_paths[key].append(hand[0].copy())
        if head is not None:
            head_path.append(head.copy())

        paths = ((head_path[-120:], C_HEAD, 2), (wrist_paths["hand_l"][-120:], (255, 205, 85), 2), (wrist_paths["hand_r"][-120:], (85, 190, 255), 2))
        for path, color, thickness in paths:
            for k in range(1, len(path)):
                alpha = 0.16 + 0.60 * k / max(1, len(path) - 1)
                draw_line(img, projector.project(path[k - 1]), projector.project(path[k]), color, thickness, alpha)

        if head is not None and axes is not None:
            origin = projector.project(head)
            for axis_idx, color in enumerate(C_AXIS):
                end = projector.project(head + axes[:, axis_idx] * 0.045)
                draw_line(img, origin, end, color, 3, 0.95)
            cv2.circle(img, origin, 5, (230, 238, 247), -1, cv2.LINE_AA)
            cv2.circle(img, origin, 6, (4, 9, 18), 1, cv2.LINE_AA)

        for hand in hands.values():
            if hand is None:
                continue
            for a, b in BONES:
                draw_line(img, projector.project(hand[a]), projector.project(hand[b]), finger_color(b), 3, 0.96)
            for joint_idx, point in enumerate(hand):
                q = projector.project(point)
                color = (245, 248, 252) if joint_idx == 0 else finger_color(joint_idx)
                cv2.circle(img, q, 4 if joint_idx else 5, color, -1, cv2.LINE_AA)
                cv2.circle(img, q, 5 if joint_idx else 6, (3, 7, 14), 1, cv2.LINE_AA)

        x0, y0 = width - 300, 25
        cv2.line(img, (x0, y0), (x0 + 34, y0), C_WRIST, 2, cv2.LINE_AA)
        cv2.putText(img, "hand_l", (x0 + 42, y0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.43, C_TEXT, 1, cv2.LINE_AA)
        cv2.line(img, (x0 + 118, y0), (x0 + 152, y0), C_HEAD, 2, cv2.LINE_AA)
        cv2.putText(img, "head", (x0 + 160, y0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.43, C_TEXT, 1, cv2.LINE_AA)
        checked_imwrite(out_dir / f"{idx:05d}.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 91])


def render_rgb_frames(
    session: Path,
    out_dir: Path,
    frame_count: int,
    max_width: int = 960,
    jpeg_quality: int = 88,
    workers: int = 8,
) -> int:
    src_root = session / "preprocess" / "all_data"
    if not src_root.exists():
        existing = sorted(out_dir.glob("*.jpg")) if out_dir.exists() else []
        if existing:
            return len(existing)
        fallback = session / "outputs" / "web" / "rgb_frames"
        if fallback.exists() and fallback.resolve() != out_dir.resolve():
            out_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            for src in sorted(fallback.glob("*.jpg"))[:frame_count]:
                shutil.copy2(src, out_dir / src.name)
                copied += 1
            if copied:
                return copied
        raise FileNotFoundError(f"Missing RGB frame source: {src_root}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.jpg"):
        try:
            idx = int(stale.stem)
        except ValueError:
            continue
        if idx >= frame_count:
            stale.unlink()

    def export_one(idx: int) -> bool:
        src = src_root / f"{idx:05d}" / "rgb.png"
        if not src.exists():
            return False
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            return False
        h, w = img.shape[:2]
        if max_width > 0 and w > max_width:
            scale = max_width / float(w)
            img = cv2.resize(img, (max_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        checked_imwrite(out_dir / f"{idx:05d}.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        return True

    worker_count = max(1, int(workers))
    if worker_count == 1:
        written = sum(export_one(idx) for idx in range(frame_count))
    else:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="review-rgb") as pool:
            written = sum(pool.map(export_one, range(frame_count)))
    if written == 0:
        raise RuntimeError(f"No RGB frames exported from {src_root}")
    return written




def _live_tactile_points() -> np.ndarray:
    points: List[List[float]] = []
    for key in "ABCDE":
        points.extend([[x, y] for x, y in TACTILE_FINGER_POINTS[key]])
    for y, x0, x1 in TACTILE_PALM_ROWS:
        for col in range(8):
            x = x1 - (x1 - x0) * col / 7.0
            points.append([x, y])
    arr = np.asarray(points, dtype=np.float64)
    if arr.shape != (TACTILE_SENSOR_COUNT, 2):
        raise RuntimeError(f"Expected {TACTILE_SENSOR_COUNT} tactile points, got {arr.shape[0]}")
    return arr


def _load_live_tactile_hand() -> np.ndarray:
    candidates = [
        TACTILE_HAND_ASSET,
        Path(__file__).resolve().parents[1] / "tactile" / "assets" / "sensor_layout.png",
    ]
    for asset_path in candidates:
        if not asset_path.exists():
            continue
        img = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
        if img is not None and img.size:
            return img
    return np.full((1040, 920, 4), (32, 32, 32, 255), dtype=np.uint8)


def _tactile_ramp(value: float) -> tuple[float, float, float]:
    v = float(np.clip(value, 0.0, 1.0))
    hi = 1
    while hi < len(TACTILE_COLOR_STOPS) - 1 and v > TACTILE_COLOR_STOPS[hi][0]:
        hi += 1
    p0, c0 = TACTILE_COLOR_STOPS[hi - 1]
    p1, c1 = TACTILE_COLOR_STOPS[hi]
    t = 0.0 if p1 <= p0 else (v - p0) / (p1 - p0)
    return tuple(c0[k] + (c1[k] - c0[k]) * t for k in range(3))


def _blend_circle(img: np.ndarray, center: tuple[float, float], radius: float, color: tuple[float, float, float], alpha: float) -> None:
    x, y = center
    r = max(1, int(np.ceil(radius)))
    x0 = max(0, int(np.floor(x)) - r)
    x1 = min(img.shape[1], int(np.floor(x)) + r + 1)
    y0 = max(0, int(np.floor(y)) - r)
    y1 = min(img.shape[0], int(np.floor(y)) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    mask = dist <= radius
    if not np.any(mask):
        return
    a = np.zeros_like(dist, dtype=np.float32)
    a[mask] = float(alpha)
    patch = img[y0:y1, x0:x1].astype(np.float32)
    patch = patch * (1.0 - a[..., None]) + np.asarray(color, dtype=np.float32)[None, None, :] * a[..., None]
    img[y0:y1, x0:x1] = patch


def _blend_radial(img: np.ndarray, center: tuple[float, float], radius: float, color: tuple[float, float, float]) -> None:
    x, y = center
    r = max(1, int(np.ceil(radius)))
    x0 = max(0, int(np.floor(x)) - r)
    x1 = min(img.shape[1], int(np.floor(x)) + r + 1)
    y0 = max(0, int(np.floor(y)) - r)
    y1 = min(img.shape[0], int(np.floor(y)) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    t = dist / max(radius, 1e-6)
    alpha = np.zeros_like(t, dtype=np.float32)
    inner = t <= 0.40
    outer = (t > 0.40) & (t <= 1.0)
    alpha[inner] = 0.95 + (0.42 - 0.95) * (t[inner] / 0.40)
    alpha[outer] = 0.42 * (1.0 - (t[outer] - 0.40) / 0.60)
    patch = img[y0:y1, x0:x1].astype(np.float32)
    patch = patch * (1.0 - alpha[..., None]) + np.asarray(color, dtype=np.float32)[None, None, :] * alpha[..., None]
    img[y0:y1, x0:x1] = patch


def _hand_rgba_without_gray_background(src: np.ndarray, target_h: int) -> np.ndarray:
    if src.ndim == 2:
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGRA)
    elif src.shape[2] == 3:
        alpha = np.full(src.shape[:2] + (1,), 255, dtype=src.dtype)
        src = np.concatenate([src, alpha], axis=2)
    scale = target_h / float(src.shape[0])
    target_w = max(1, int(round(src.shape[1] * scale)))
    rgba = cv2.resize(src, (target_w, target_h), interpolation=cv2.INTER_AREA).astype(np.float32)
    bgr = rgba[:, :, :3]
    alpha = rgba[:, :, 3] / 255.0
    gray = cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    # hand_live.png has an opaque gray artwork background. Treat low-saturation gray
    # pixels as transparent while keeping the blue hand texture and colored heatmap.
    gray_bg = (hsv[:, :, 1] < 42.0) & (gray > 28.0) & (gray < 92.0)
    alpha = np.where(gray_bg, 0.0, alpha)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.8)
    out = np.dstack([bgr, np.clip(alpha * 255.0, 0, 255)])
    return out.astype(np.float32)


def _overlay_rgba(dst: np.ndarray, src_rgba: np.ndarray, x0: int, y0: int) -> None:
    h, w = src_rgba.shape[:2]
    dx0 = max(0, x0)
    dy0 = max(0, y0)
    dx1 = min(dst.shape[1], x0 + w)
    dy1 = min(dst.shape[0], y0 + h)
    if dx0 >= dx1 or dy0 >= dy1:
        return
    sx0 = dx0 - x0
    sy0 = dy0 - y0
    patch = src_rgba[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    alpha = patch[:, :, 3:4] / 255.0
    dst[dy0:dy1, dx0:dx1] = dst[dy0:dy1, dx0:dx1] * (1.0 - alpha) + patch[:, :, :3] * alpha


def _draw_live_tactile_hand(canvas: np.ndarray, values: np.ndarray, points: np.ndarray, hand_asset: np.ndarray, vmax: float, rect: tuple[int, int, int, int], mirror: bool = False, remap_left: bool = False) -> tuple[float, float, int, int]:
    x, y, rect_w, rect_h = rect
    hand = _hand_rgba_without_gray_background(hand_asset, rect_h).copy()
    if mirror:
        hand = hand[:, ::-1].copy()
    hand_alpha = hand[:, :, 3:4].copy()
    display = np.clip(values.astype(np.float64) / max(vmax, 1e-9) * 100.0, 0.0, 100.0)
    if remap_left:
        display = display[TACTILE_LEFT_VISUAL_SOURCE]
    h, w = hand.shape[:2]
    heat_bgr = hand[:, :, :3]
    for (px, py), value in zip(points, display):
        if not np.isfinite(value) or value <= 0.0:
            continue
        visual = float(min(1.0, value / 100.0))
        sx = 1.0 - float(px) if mirror else float(px)
        cx = sx * w
        cy = float(py) * h
        size = min(w, h)
        _blend_radial(heat_bgr, (cx, cy), 4.0 + visual * size * 0.045, _tactile_ramp(visual))
        _blend_circle(heat_bgr, (cx, cy), 2.0 + visual * 2.5, _tactile_ramp(min(1.0, visual + 0.15)), 1.0)
    hand[:, :, 3:4] = hand_alpha
    x0 = x + (rect_w - hand.shape[1]) // 2
    y0 = y + (rect_h - hand.shape[0]) // 2
    _overlay_rgba(canvas, hand, x0, y0)
    finite = display[np.isfinite(display)]
    if not finite.size:
        return 0.0, 0.0, 0, 0
    return float(np.max(finite)), float(np.mean(finite)), int(np.sum(finite > 0.0)), int(np.sum(finite >= 70.0))


def _draw_tactile_stats(canvas: np.ndarray, stats: tuple[float, float, int, int], x: int, y: int, width: int) -> None:
    labels = ("MAX", "AVG", "CONTACT", "HIGH")
    values = (f"{stats[0]:.1f}", f"{stats[1]:.1f}", str(stats[2]), str(stats[3]))
    cell_w = width / 4.0
    for i, (label, value) in enumerate(zip(labels, values)):
        cx = int(round(x + (i + 0.5) * cell_w))
        cv2.putText(canvas, label, (cx - 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.27, (167, 149, 128), 1, cv2.LINE_AA)
        cv2.putText(canvas, value, (cx - 14, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (116, 90, 53), 1, cv2.LINE_AA)


def _render_live_tactile_frame(left_values: np.ndarray, right_values: np.ndarray, points: np.ndarray, hand_asset: np.ndarray, left_vmax: float, right_vmax: float) -> np.ndarray:
    canvas = np.empty((TACTILE_FRAME_H, TACTILE_FRAME_W, 3), dtype=np.float32)
    top = np.asarray(TACTILE_BG_TOP_BGR, dtype=np.float32)
    bottom = np.asarray(TACTILE_BG_BOTTOM_BGR, dtype=np.float32)
    for yy in range(TACTILE_FRAME_H):
        t = yy / max(1, TACTILE_FRAME_H - 1)
        canvas[yy, :] = top * (1.0 - t) + bottom * t
    # Match the Live dashboard's paired tactile cards inside the review panel.
    for x in (10, 346):
        cv2.rectangle(canvas, (x, 8), (x + 320, 319), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x, 8), (x + 320, 319), (243, 235, 223), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x + 8, 36), (x + 312, 278), (250, 245, 237), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x + 8, 36), (x + 312, 278), (243, 235, 223), 1, cv2.LINE_AA)
    cv2.putText(canvas, "LEFT TACTILE", (24, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (116, 90, 53), 1, cv2.LINE_AA)
    cv2.putText(canvas, "RIGHT TACTILE", (360, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (116, 90, 53), 1, cv2.LINE_AA)
    left_stats = _draw_live_tactile_hand(canvas, left_values, points, hand_asset, left_vmax, (30, 38, 280, 236), mirror=False, remap_left=True)
    right_stats = _draw_live_tactile_hand(canvas, right_values, points, hand_asset, right_vmax, (366, 38, 280, 236), mirror=True)
    _draw_tactile_stats(canvas, left_stats, 20, 294, 300)
    _draw_tactile_stats(canvas, right_stats, 356, 294, 300)
    return np.clip(canvas, 0.0, 255.0).astype(np.uint8)


def _tactile_delta(values: np.ndarray) -> tuple[np.ndarray, float]:
    if values.size == 0:
        return values, 1.0
    n_base = min(TACTILE_BASELINE_FRAMES, values.shape[0])
    baseline = np.nanmedian(values[:n_base], axis=0) if n_base > 0 else np.zeros(TACTILE_SENSOR_COUNT)
    delta = np.maximum(0.0, values - baseline[None, :])
    finite = delta[np.isfinite(delta)]
    if finite.size == 0:
        return np.zeros_like(values), 1.0
    raw_vmax = float(np.percentile(finite, 99.0))
    if not np.isfinite(raw_vmax) or raw_vmax <= 1e-9:
        raw_vmax = float(np.nanmax(finite)) if finite.size else 1.0
    sensor_floor = np.nanpercentile(delta, TACTILE_SENSOR_DEADZONE_PERCENTILE, axis=0)
    sensor_floor = np.nan_to_num(sensor_floor, nan=0.0, posinf=0.0, neginf=0.0)
    sensor_floor = np.maximum(sensor_floor, TACTILE_SENSOR_DEADZONE_MIN)
    delta = np.maximum(0.0, delta - sensor_floor[None, :])
    vmax = max(raw_vmax, TACTILE_MIN_SCALE) * TACTILE_DISPLAY_SCALE_MULTIPLIER
    return delta, vmax


def load_tactile_rows(traj_path: Path, frame_count: int) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    sides: Dict[str, List[List[float]]] = {"left": [], "right": []}
    has_data = {"left": False, "right": False}
    for line in traj_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        hand_frame = row.get("hand_frame") or {}
        for side in ("left", "right"):
            raw = hand_frame.get(f"pressure_{side}")
            ok = isinstance(raw, list) and len(raw) == TACTILE_SENSOR_COUNT
            if ok:
                arr = [float(v) for v in raw]
                finite = [v for v in arr if np.isfinite(v)]
                if finite and any(abs(v) > 1e-9 for v in finite):
                    has_data[side] = True
            else:
                arr = [0.0] * TACTILE_SENSOR_COUNT
            sides[side].append(arr)
    for side in ("left", "right"):
        while len(sides[side]) < frame_count:
            sides[side].append([0.0] * TACTILE_SENSOR_COUNT)
    left = np.asarray(sides["left"][:frame_count], dtype=np.float64)
    right = np.asarray(sides["right"][:frame_count], dtype=np.float64)
    return left, right, has_data["left"], has_data["right"]

def render_tactile_frames(traj_path: Path, out_dir: Path, frame_count: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    points = _live_tactile_points()
    hand_asset = _load_live_tactile_hand()
    left, right, has_left, has_right = load_tactile_rows(traj_path, frame_count)
    left_delta, left_vmax = _tactile_delta(left)
    right_delta, right_vmax = _tactile_delta(right)
    left_vmax = max(left_vmax if has_left else 0.0, TACTILE_MIN_SCALE)
    right_vmax = max(right_vmax if has_right else 0.0, TACTILE_MIN_SCALE)
    written = 0
    for i in range(frame_count):
        canvas = _render_live_tactile_frame(left_delta[i], right_delta[i], points, hand_asset, left_vmax, right_vmax)
        checked_imwrite(out_dir / f"{i:05d}.jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        written += 1
    return written


def build_tactile_web_data(traj_path: Path, frame_count: int) -> Dict[str, Any]:
    """Prepare compact normalized sensor values for browser-side Canvas rendering."""
    points = _live_tactile_points()
    left, right, has_left, has_right = load_tactile_rows(traj_path, frame_count)
    left_delta, left_vmax = _tactile_delta(left)
    right_delta, right_vmax = _tactile_delta(right)
    left_vmax = max(left_vmax if has_left else 0.0, TACTILE_MIN_SCALE)
    right_vmax = max(right_vmax if has_right else 0.0, TACTILE_MIN_SCALE)
    left_display = np.rint(np.clip(left_delta / left_vmax * 100.0, 0.0, 100.0)).astype(np.uint8)
    right_display = np.rint(np.clip(right_delta / right_vmax * 100.0, 0.0, 100.0)).astype(np.uint8)
    left_display = left_display[:, TACTILE_LEFT_VISUAL_SOURCE]
    return {
        "points": np.round(points, 5).tolist(),
        "left": left_display.tolist(),
        "right": right_display.tolist(),
        "has_left": bool(has_left),
        "has_right": bool(has_right),
    }


def export_tactile_hand_asset(out_path: Path) -> Path:
    """Write one transparent hand texture reused by every tactile Canvas frame."""
    hand = _hand_rgba_without_gray_background(_load_live_tactile_hand(), 236)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    checked_imwrite(out_path, np.clip(hand, 0.0, 255.0).astype(np.uint8))
    return out_path


def remove_legacy_web_frame_dirs(web_dir: Path) -> None:
    """Remove reproducible frame sets replaced by browser-side Canvas rendering."""
    for name in ("traj_frames", "tactile_frames"):
        path = web_dir / name
        if path.is_dir():
            shutil.rmtree(path)


def load_frame_times(traj_path: Path, fallback_fps: float = 30.0) -> tuple[List[float], float]:
    rows = [json.loads(line) for line in traj_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    fps = nominal_fps(rows, fallback=fallback_fps)
    return relative_times_sec(rows, fallback_fps=fps), fps


def build_chart_data(traj_path: Path, frame_times: List[float]) -> List[Dict[str, Any]]:
    """Build per-frame wrist and head-camera positions in the world frame."""
    out: List[Dict[str, Any]] = []
    lines = [line for line in traj_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row_index, line in enumerate(lines):
        row = json.loads(line)
        item: Dict[str, Any] = {
            "i": int(row.get("idx", row_index)),
            "t": frame_times[row_index],
            "lx": None, "ly": None, "lz": None,
            "rx": None, "ry": None, "rz": None,
            "hx": None, "hy": None, "hz": None,
        }
        hands = row.get("hands") or {}
        for side, prefix in (("left", "l"), ("right", "r")):
            pts = (((hands.get(side) or {}).get("glove") or {}).get("kpts_3d_world_m"))
            if pts is None:
                continue
            arr = np.asarray(pts, dtype=np.float64)
            if arr.ndim >= 2 and arr.shape[0] > 0 and arr.shape[1] >= 3 and np.isfinite(arr[0, :3]).all():
                wrist = arr[0, :3]
                item[f"{prefix}x"] = float(wrist[0])
                item[f"{prefix}y"] = float(wrist[1])
                item[f"{prefix}z"] = float(wrist[2])
        c2w = (row.get("head_pose") or {}).get("c2w") or (row.get("camera") or {}).get("c2w")
        if c2w is not None:
            mat = np.asarray(c2w, dtype=np.float64)
            if mat.shape == (4, 4) and np.isfinite(mat[:3, 3]).all():
                head = mat[:3, 3]
                item["hx"] = float(head[0])
                item["hy"] = float(head[1])
                item["hz"] = float(head[2])
        out.append(item)
    return out


def build_trajectory_web_data(rows: List[Dict[str, Any]], all_points: List[np.ndarray], hand_display_scale: float = 3.0) -> Dict[str, Any]:
    finite_sets = []
    for pts in all_points:
        arr = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        arr = arr[np.isfinite(arr).all(axis=1)]
        if arr.size:
            finite_sets.append(arr)
    if finite_sets:
        cloud = np.concatenate(finite_sets, axis=0)
        center = np.nanmedian(cloud, axis=0)
        radius = float(np.nanpercentile(np.linalg.norm(cloud - center[None, :], axis=1), 98.0))
    else:
        center = np.zeros(3, dtype=np.float64)
        radius = 1.0
    if not np.isfinite(radius) or radius < 1e-6:
        radius = 1.0

    frames: List[Dict[str, Any]] = []
    for row in rows:
        item: Dict[str, Any] = {}
        for key in ("hand_l", "hand_r"):
            hand = row.get(key)
            if hand is None:
                continue
            hand_arr = np.asarray(hand, dtype=np.float64).copy()
            if hand_arr.shape[0] > 1:
                root = hand_arr[0].copy()
                hand_arr[1:] = root[None, :] + (hand_arr[1:] - root[None, :]) * float(hand_display_scale)
                if TRAJECTORY_MIRROR_LEFT_HAND and hand_arr.shape[0] > 9:
                    axis = hand_arr[9] - root
                    axis_norm = float(np.linalg.norm(axis))
                    if np.isfinite(axis_norm) and axis_norm > 1e-9:
                        axis = axis / axis_norm
                        rel = hand_arr - root[None, :]
                        hand_arr = root[None, :] + 2.0 * (rel @ axis)[:, None] * axis[None, :] - rel
            item[key] = np.round(hand_arr, 5).tolist()
        head = row.get("head")
        if head is not None:
            item["head"] = np.round(np.asarray(head, dtype=np.float64), 5).tolist()
        axes = row.get("axes")
        if axes is not None:
            item["axes"] = np.round(np.asarray(axes, dtype=np.float64), 5).tolist()
        frames.append(item)
    initial_directions = []
    if frames:
        for key in ("hand_l", "hand_r"):
            hand = frames[0].get(key)
            if hand is None or len(hand) <= 9:
                continue
            direction = np.asarray(hand[9], dtype=np.float64)[:2] - np.asarray(hand[0], dtype=np.float64)[:2]
            norm = float(np.linalg.norm(direction))
            if norm > 1e-9:
                initial_directions.append(direction / norm)
    default_yaw = -2.3517
    if initial_directions:
        mean_direction = np.sum(initial_directions, axis=0)
        if float(np.linalg.norm(mean_direction)) > 1e-9:
            mean_direction /= np.linalg.norm(mean_direction)
            default_yaw = float(-np.pi / 2.0 - np.arctan2(mean_direction[1], mean_direction[0]))
    return {
        "frames": frames,
        "center": np.round(center, 5).tolist(),
        "radius": round(radius, 5),
        "bones": BONES,
        "default_yaw": default_yaw,
    }


def write_html(
    session: Path,
    chart_rows: List[Dict[str, Any]],
    trajectory_data: Dict[str, Any],
    tactile_data: Dict[str, Any],
    frame_times: List[float],
    fps: float = 30.0,
    output_subdir: str = "web",
) -> Path:
    outputs = session / "outputs"
    summary = read_json(outputs / "summaries" / "trajectory_3d_camera_frame.json", {})
    wrist_summary = read_json(outputs / "summaries" / "wristroot_track_summary.json", {})
    frames = len(frame_times)
    left_valid = sum(1 for row in chart_rows if all(row.get(k) is not None for k in ("lx", "ly", "lz")))
    right_valid = sum(1 for row in chart_rows if all(row.get(k) is not None for k in ("rx", "ry", "rz")))
    fps = float(fps) if float(fps) > 0 else 30.0
    duration = frame_times[-1] if frame_times else 0.0
    if duration <= 0.0 and frames:
        duration = 1.0 / fps
    step = summary.get("wrist_camera_step_m", {})
    params = wrist_summary.get("params", {})
    chart_json = json.dumps(chart_rows, separators=(",", ":"))
    traj_json = json.dumps(trajectory_data, separators=(",", ":"))
    tactile_json = json.dumps(tactile_data, separators=(",", ":"))
    frame_times_json = json.dumps(frame_times, separators=(",", ":"))
    html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>第一视角采集</title>
<style>
:root{{--bg:#eaf1f8;--panel:#fff;--ink:#172033;--muted:#64748b;--line:#d9e4ef;--blue:#2563eb;--green:#16a34a;--orange:#ea580c;--shadow:0 10px 24px rgba(25,42,70,.10);--scale:1}}
*{{box-sizing:border-box}} body{{margin:0;width:100vw;height:100vh;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;overflow:hidden}}
.shell{{position:absolute;left:50%;top:50%;width:1280px;height:720px;transform:translate(-50%,-50%) scale(var(--scale));transform-origin:center center;display:grid;grid-template-rows:42px 554px 44px;gap:10px;padding:18px 48px 16px;background:var(--bg)}}
.topbar{{display:grid;grid-template-columns:160px 1fr 160px;align-items:center;height:42px}}.topbar:before{{content:"退出采集";justify-self:start;width:88px;height:28px;line-height:26px;text-align:center;border:1px solid #ef4444;border-radius:4px;color:#ef4444;background:#fff;font-size:13px;font-weight:700}}.topbar>div:first-child{{grid-column:2;text-align:center}}.topbar h1{{margin:0;font-size:22px;line-height:1.1;color:#1d5ecb;font-weight:800;letter-spacing:0}}.sub{{display:none}}.status{{grid-column:3;justify-self:end;display:flex;align-items:center;gap:8px;padding:6px 10px;background:rgba(255,255,255,.9);border:1px solid var(--line);border-radius:6px;color:var(--muted);font-size:12px}}.dot{{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 0 3px rgba(22,163,74,.12)}}
.main{{display:grid;grid-template-columns:676px 474px;gap:28px;height:554px;min-height:0;align-items:start}}.stage{{display:grid;grid-template-rows:380px 164px;gap:10px;width:676px;height:554px;min-height:0}}.lowerStage{{display:grid;grid-template-columns:1fr 1fr;gap:10px;min-height:0}}.panel{{background:var(--panel);border:1px solid var(--line);border-radius:6px;box-shadow:var(--shadow);overflow:hidden}}.videoPanel{{position:relative;background:#06101d;height:100%;min-height:0}}.videoPanel canvas{{width:100%;height:100%;display:block;background:#06101d}}.lowerStage .videoPanel,.lowerStage .videoPanel canvas{{background:#e4eef6}}.lowerStage .tactilePanel,.lowerStage .tactilePanel canvas{{background:#e4eef6}}.trajPanel canvas{{cursor:grab}}.trajPanel.dragging canvas{{cursor:grabbing}}.badge{{position:absolute;left:5px;top:5px;z-index:2;padding:3px 7px;border-radius:4px;background:rgba(44,64,86,.78);border:1px solid rgba(255,255,255,.30);color:#f5f9fc;font-size:10px;line-height:1.1;font-weight:620;letter-spacing:0;backdrop-filter:blur(8px)}}
.side{{width:474px;height:554px;padding:12px;display:grid;grid-template-rows:92px 1fr 100px;gap:8px;min-height:0}}.section-title{{display:flex;align-items:center;justify-content:center;position:relative;font-weight:800;font-size:15px;margin-bottom:9px;color:#334155}}.section-title:before,.section-title:after{{content:"";height:1px;background:#e2e8f0;flex:1;margin:0 18px}}.frameTag{{position:absolute;right:0;color:var(--muted);font-size:11px;font-weight:600}}.grid3{{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}}.metric{{height:58px;padding:9px;border:1px solid #e7eef6;border-radius:6px;background:#f8fbfe;text-align:center}}.label{{color:var(--muted);font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.value{{margin-top:5px;color:var(--blue);font-size:17px;font-weight:800}}.unit{{color:var(--muted);font-size:11px;margin-left:2px}}.chartBox{{height:100%;min-height:0;display:grid;grid-template-rows:42px 1fr;padding:8px;border:1px solid #e4edf6;border-radius:6px;background:#fbfdff}}.legend{{display:grid;grid-template-columns:repeat(3,1fr);row-gap:5px;color:var(--muted);font-size:11px;padding:1px 4px 5px}}.legend span:before{{content:"";display:inline-block;width:16px;height:3px;border-radius:9px;margin-right:5px;vertical-align:middle}}.llx:before{{background:#2563eb}}.lly:before{{background:#ea580c}}.llz:before{{background:#16a34a}}.rrx:before{{background:#7c3aed}}.rry:before{{background:#db2777}}.rrz:before{{background:#0891b2}}#chartCanvas{{width:100%;height:100%;min-height:0}}.current{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px 8px}}.readout{{border:1px solid #e7eef6;border-radius:5px;padding:6px 9px;background:#fff;height:47px}}.readout b{{display:block;color:var(--muted);font-size:10px;margin-bottom:3px}}.readout span{{font-size:13px;font-weight:760}}.timeline{{display:grid;grid-template-columns:122px 1fr 116px;align-items:center;gap:10px;height:44px;padding:7px 10px;background:var(--panel);border:1px solid var(--line);border-radius:6px;box-shadow:var(--shadow)}}.controls{{display:flex;gap:7px}}button{{height:30px;min-width:56px;border:0;border-radius:6px;background:var(--blue);color:white;padding:0 12px;font-size:12px;font-weight:750;cursor:pointer}}button.secondary{{min-width:50px;background:#e2e8f0;color:#1e293b}}input[type=range]{{width:100%;accent-color:var(--blue)}}.time{{min-width:108px;text-align:right;color:var(--muted);font-size:12px}}
.side{{grid-template-rows:92px 1fr 153px}}
.chartBox{{grid-template-rows:59px 1fr}}
.hhx:before{{background:#0f766e}}.hhy:before{{background:#64748b}}.hhz:before{{background:#a16207}}
</style></head>
<body><div class="shell"><header class="topbar"><div><h1>第一人称数据采集</h1><div class="sub">review</div></div><div class="status"><span class="dot"></span><span>审核进度 · 69/100</span></div></header>
<main class="main"><section class="stage"><div class="panel videoPanel"><div class="badge">RGB</div><canvas id="rgbCanvas"></canvas></div><div class="lowerStage"><div class="panel videoPanel trajPanel"><div class="badge">Trajectory</div><canvas id="trajCanvas"></canvas></div><div class="panel videoPanel tactilePanel"><div class="badge">Tactile</div><canvas id="tactileCanvas"></canvas></div></div></section>
<aside class="panel side"><section><div class="section-title"><span>采集信息</span><span class="frameTag" id="frameTag">0 / {frames}</span></div><div class="grid3"><div class="metric"><div class="label">采集时长</div><div class="value">{duration:.2f}<span class="unit">s</span></div></div><div class="metric"><div class="label">实时帧率</div><div class="value">{fps:.1f}<span class="unit">fps</span></div></div><div class="metric"><div class="label">有效帧（左/右）</div><div class="value">{left_valid}/{right_valid}</div></div></div></section><section class="chartBox"><div class="legend"><span class="llx">左手 x</span><span class="lly">左手 y</span><span class="llz">左手 z</span><span class="rrx">右手 x</span><span class="rry">右手 y</span><span class="rrz">右手 z</span></div><canvas id="chartCanvas"></canvas></section><section class="current"><div class="readout"><b>左手 wrist x · world</b><span id="lxNow">--</span></div><div class="readout"><b>左手 wrist y · world</b><span id="lyNow">--</span></div><div class="readout"><b>左手 wrist z · world</b><span id="lzNow">--</span></div><div class="readout"><b>右手 wrist x · world</b><span id="rxNow">--</span></div><div class="readout"><b>右手 wrist y · world</b><span id="ryNow">--</span></div><div class="readout"><b>右手 wrist z · world</b><span id="rzNow">--</span></div></section></aside></main>
<footer class="timeline"><div class="controls"><button id="playBtn">播放</button><button class="secondary" id="resetBtn">重置</button></div><input id="scrub" type="range" min="0" max="{duration:.6f}" step="0.01" value="0"><div class="time"><span id="timeNow">0.00</span>s / {duration:.2f}s</div></footer></div>
<script>
function fitShell(){{const scale=Math.min(window.innerWidth/1280,window.innerHeight/720);document.documentElement.style.setProperty('--scale',String(scale));}}
window.addEventListener('resize',fitShell);fitShell();
const DATA={chart_json}; const TRAJ={traj_json}; const TACTILE={tactile_json}; const FRAME_TIMES={frame_times_json}; const DURATION={duration:.6f}; const FPS={fps:.6f}; const FRAME_COUNT={frames}; const TACTILE_CANVAS_BG="{TACTILE_CANVAS_BG}"; const TRAJECTORY_CANVAS_BG="{TRAJECTORY_CANVAS_BG}";
const rgbFrames=Array.from({{length:FRAME_COUNT}},(_,i)=>`rgb_frames/${{String(i).padStart(5,'0')}}.jpg`);
const rgbCanvas=document.getElementById('rgbCanvas'),trajCanvas=document.getElementById('trajCanvas'),tactileCanvas=document.getElementById('tactileCanvas'),rgbCtx=rgbCanvas.getContext('2d'),trajCtx=trajCanvas.getContext('2d'),tactileCtx=tactileCanvas.getContext('2d'),playBtn=document.getElementById('playBtn'),resetBtn=document.getElementById('resetBtn'),scrub=document.getElementById('scrub'),timeNow=document.getElementById('timeNow'),frameTag=document.getElementById('frameTag'),lxNow=document.getElementById('lxNow'),lyNow=document.getElementById('lyNow'),lzNow=document.getElementById('lzNow'),rxNow=document.getElementById('rxNow'),ryNow=document.getElementById('ryNow'),rzNow=document.getElementById('rzNow'),canvas=document.getElementById('chartCanvas'),ctx=canvas.getContext('2d'); let frame=0,playing=false,lastTs=0,playTime=0; const imgCache=new Map();
function nearest(t){{let b=null,bd=1e9;for(const p of DATA){{const d=Math.abs(p.t-t);if(d<bd){{bd=d;b=p}}}}return b}} function fmt(v){{return Number.isFinite(v)?v.toFixed(3)+' m':'--'}} function loadImage(src){{if(imgCache.has(src))return imgCache.get(src);const im=new Image();im.src=src;imgCache.set(src,im);return im}} function preload(i){{for(let k=-2;k<=8;k++){{const j=Math.max(0,Math.min(FRAME_COUNT-1,i+k));loadImage(rgbFrames[j]);}}loadImage('tactile_hand.png');}}

const DEFAULT_TRAJ_VIEW={{yaw:Number.isFinite(TRAJ.default_yaw)?TRAJ.default_yaw:-2.3517,pitch:0.0,roll:3.1415926536,zoom:1}};
const trajView={{...DEFAULT_TRAJ_VIEW,drag:false,x:0,y:0}};
const FINGER_RGB=['#ff40c4','#f59e0b','#22c55e','#0ea5e9','#8b5cf6'];
function clamp(v,a,b){{return Math.max(a,Math.min(b,v))}}
function resetTrajView(){{trajView.yaw=DEFAULT_TRAJ_VIEW.yaw;trajView.pitch=DEFAULT_TRAJ_VIEW.pitch;trajView.roll=DEFAULT_TRAJ_VIEW.roll;trajView.zoom=DEFAULT_TRAJ_VIEW.zoom;trajView.drag=false;}}
function projectTraj(pt){{const c=TRAJ.center||[0,0,0],r=Math.max(1e-6,TRAJ.radius||1),w=trajCanvas.width,h=trajCanvas.height;let x=pt[0]-c[0],y=pt[1]-c[1],z=pt[2]-c[2];const cy=Math.cos(trajView.yaw),sy=Math.sin(trajView.yaw),cp=Math.cos(trajView.pitch),sp=Math.sin(trajView.pitch),cr=Math.cos(trajView.roll||0),sr=Math.sin(trajView.roll||0);const x1=cy*x-sy*y,y1=sy*x+cy*y,z1=z;const y2=cp*y1-sp*z1,z2=sp*y1+cp*z1;const sx=cr*x1-sr*(-y2),sy2=sr*x1+cr*(-y2);const sc=Math.min(w,h)*0.43/r*trajView.zoom;return [w*.5+sx*sc,h*.52+sy2*sc,z2];}}
function projectTrajHand(pt,root){{const q=projectTraj(pt),r=projectTraj(root);q[0]=2*r[0]-q[0];return q;}}
function line3(a,b,color,width=2,alpha=1,mirrorRoot=null){{const p=mirrorRoot?projectTrajHand(a,mirrorRoot):projectTraj(a),q=mirrorRoot?projectTrajHand(b,mirrorRoot):projectTraj(b);trajCtx.globalAlpha=alpha;trajCtx.strokeStyle=color;trajCtx.lineWidth=width;trajCtx.lineCap='round';trajCtx.beginPath();trajCtx.moveTo(p[0],p[1]);trajCtx.lineTo(q[0],q[1]);trajCtx.stroke();trajCtx.globalAlpha=1;}}
function dot3(p,r,color,stroke='#ffffff',mirrorRoot=null){{const q=mirrorRoot?projectTrajHand(p,mirrorRoot):projectTraj(p);trajCtx.fillStyle=color;trajCtx.beginPath();trajCtx.arc(q[0],q[1],r,0,Math.PI*2);trajCtx.fill();trajCtx.strokeStyle=stroke;trajCtx.lineWidth=1;trajCtx.stroke();}}
function drawTrajGrid(w,h){{const g=trajCtx.createLinearGradient(0,0,0,h);g.addColorStop(0,'#f2f8fc');g.addColorStop(1,'#e4eef6');trajCtx.fillStyle=g;trajCtx.fillRect(0,0,w,h);for(let x=0;x<=w;x+=28){{const major=Math.round(x/28)%4===0;trajCtx.strokeStyle=major?'rgba(124,141,160,.32)':'rgba(154,171,190,.18)';trajCtx.lineWidth=1;trajCtx.beginPath();trajCtx.moveTo(x,0);trajCtx.lineTo(x,h);trajCtx.stroke();}}for(let y=0;y<=h;y+=28){{const major=Math.round(y/28)%4===0;trajCtx.strokeStyle=major?'rgba(124,141,160,.32)':'rgba(154,171,190,.18)';trajCtx.lineWidth=1;trajCtx.beginPath();trajCtx.moveTo(0,y);trajCtx.lineTo(w,y);trajCtx.stroke();}}}}
function drawHand3(hand){{if(!hand)return;for(const [a,b] of TRAJ.bones)line3(hand[a],hand[b],FINGER_RGB[Math.max(0,Math.min(4,Math.floor((b-1)/4)))],3,.96);for(let i=0;i<hand.length;i++)dot3(hand[i],i===0?5:4,i===0?'#f8fafc':FINGER_RGB[Math.max(0,Math.min(4,Math.floor((i-1)/4)))],'#1f2937');}}
function drawTrajectory(){{const w=trajCanvas.width,h=trajCanvas.height;trajCtx.setTransform(1,0,0,1,0,0);drawTrajGrid(w,h);const frames=TRAJ.frames||[],f=frames[Math.max(0,Math.min(frames.length-1,frame))]||{{}},start=Math.max(1,frame-120);for(let i=start;i<=frame&&i<frames.length;i++){{const a=frames[i-1],b=frames[i],t=(i-start)/Math.max(1,frame-start);if(a&&b&&a.head&&b.head)line3(a.head,b.head,'#38a9a2',2,.18+.54*t);if(a&&b&&a.hand_l&&b.hand_l)line3(a.hand_l[0],b.hand_l[0],'#35aee9',2,.18+.58*t);if(a&&b&&a.hand_r&&b.hand_r)line3(a.hand_r[0],b.hand_r[0],'#f59e0b',2,.18+.58*t);}}if(f.head&&f.axes){{const axisColors=['#ef5b4f','#2ea65a','#dc8a32'];for(let k=0;k<3;k++){{const e=[f.head[0]+f.axes[0][k]*.045,f.head[1]+f.axes[1][k]*.045,f.head[2]+f.axes[2][k]*.045];line3(f.head,e,axisColors[k],3,.95);}}dot3(f.head,5,'#f8fafc','#334155');}}drawHand3(f.hand_l);drawHand3(f.hand_r);const lx=w-250,ly=18;trajCtx.font='12px system-ui';trajCtx.fillStyle='#334155';for(const [off,color,label] of [[0,'#35aee9','hand_l'],[82,'#f59e0b','hand_r'],[164,'#38a9a2','head']]){{trajCtx.strokeStyle=color;trajCtx.lineWidth=2;trajCtx.beginPath();trajCtx.moveTo(lx+off,ly);trajCtx.lineTo(lx+off+26,ly);trajCtx.stroke();trajCtx.fillText(label,lx+off+31,ly+4);}}}}

const TACTILE_STOPS=[[0,[112,179,220]],[.45,[71,191,200]],[.70,[240,196,91]],[.86,[238,144,67]],[1,[211,83,83]]];
const tactileScratch=[document.createElement('canvas'),document.createElement('canvas')];
function tactileColor(value,alpha=1){{const v=clamp(value,0,1);let hi=1;while(hi<TACTILE_STOPS.length-1&&v>TACTILE_STOPS[hi][0])hi++;const [p0,c0]=TACTILE_STOPS[hi-1],[p1,c1]=TACTILE_STOPS[hi],t=p1<=p0?0:(v-p0)/(p1-p0),c=c0.map((x,k)=>Math.round(x+(c1[k]-x)*t));return `rgba(${{c[0]}},${{c[1]}},${{c[2]}},${{alpha}})`;}}
function tactileStats(values){{if(!values||!values.length)return [0,0,0,0];let mx=0,sum=0,contact=0,high=0;for(const v0 of values){{const v=Number.isFinite(v0)?v0:0;mx=Math.max(mx,v);sum+=v;if(v>0)contact++;if(v>=70)high++;}}return [mx,sum/values.length,contact,high];}}
function drawTactileStats(stats,x,y,width){{const labels=['MAX','AVG','CONTACT','HIGH'],values=[stats[0].toFixed(1),stats[1].toFixed(1),String(stats[2]),String(stats[3])],cw=width/4;tactileCtx.textAlign='center';for(let i=0;i<4;i++){{const cx=x+(i+.5)*cw;tactileCtx.fillStyle='#806f61';tactileCtx.font='9px system-ui';tactileCtx.fillText(labels[i],cx,y);tactileCtx.fillStyle='#745a35';tactileCtx.font='11px system-ui';tactileCtx.fillText(values[i],cx,y+14);}}tactileCtx.textAlign='start';}}
function drawTactileHand(values,rect,mirror,scratchIndex){{const asset=loadImage('tactile_hand.png');if(!asset.complete||!asset.naturalWidth){{asset.onload=()=>drawAll();return tactileStats(values)}}const ow=asset.naturalWidth,oh=asset.naturalHeight,off=tactileScratch[scratchIndex];if(off.width!==ow||off.height!==oh){{off.width=ow;off.height=oh}}const oc=off.getContext('2d');oc.setTransform(1,0,0,1,0,0);oc.clearRect(0,0,ow,oh);if(mirror){{oc.save();oc.translate(ow,0);oc.scale(-1,1);oc.drawImage(asset,0,0);oc.restore()}}else oc.drawImage(asset,0,0);oc.globalCompositeOperation='source-atop';const pts=TACTILE.points||[];for(let i=0;i<Math.min(pts.length,values.length);i++){{const visual=clamp((values[i]||0)/100,0,1);if(visual<=0)continue;const px=mirror?1-pts[i][0]:pts[i][0],cx=px*ow,cy=pts[i][1]*oh,r=4+visual*Math.min(ow,oh)*.045,g=oc.createRadialGradient(cx,cy,0,cx,cy,r);g.addColorStop(0,tactileColor(visual,.95));g.addColorStop(.4,tactileColor(visual,.42));g.addColorStop(1,tactileColor(visual,0));oc.fillStyle=g;oc.beginPath();oc.arc(cx,cy,r,0,Math.PI*2);oc.fill();oc.fillStyle=tactileColor(Math.min(1,visual+.15),1);oc.beginPath();oc.arc(cx,cy,2+visual*2.5,0,Math.PI*2);oc.fill();}}oc.globalCompositeOperation='source-over';const [x,y,rw,rh]=rect,scale=Math.min(rw/ow,rh/oh),dw=ow*scale,dh=oh*scale;tactileCtx.drawImage(off,x+(rw-dw)/2,y+(rh-dh)/2,dw,dh);return tactileStats(values);}}
function drawTactile(){{const w=tactileCanvas.width,h=tactileCanvas.height;tactileCtx.setTransform(w/676,0,0,h/328,0,0);const bg=tactileCtx.createLinearGradient(0,0,0,328);bg.addColorStop(0,'#edf5fa');bg.addColorStop(1,'#e4eef6');tactileCtx.fillStyle=bg;tactileCtx.fillRect(0,0,676,328);for(const x of [10,346]){{tactileCtx.fillStyle='#fff';tactileCtx.strokeStyle='#dfeaf3';tactileCtx.lineWidth=1;tactileCtx.fillRect(x,8,320,311);tactileCtx.strokeRect(x+.5,8.5,319,310);tactileCtx.fillStyle='#edf5fa';tactileCtx.fillRect(x+8,36,304,242);tactileCtx.strokeRect(x+8.5,36.5,303,241)}}tactileCtx.fillStyle='#745a35';tactileCtx.font='13px system-ui';tactileCtx.fillText('LEFT TACTILE',24,31);tactileCtx.fillText('RIGHT TACTILE',360,31);const i=Math.max(0,Math.min(FRAME_COUNT-1,frame)),left=(TACTILE.left||[])[i]||[],right=(TACTILE.right||[])[i]||[],ls=drawTactileHand(left,[30,38,280,236],false,0),rs=drawTactileHand(right,[366,38,280,236],true,1);drawTactileStats(ls,20,294,300);drawTactileStats(rs,356,294,300);}}
trajCanvas.addEventListener('pointerdown',e=>{{if(e.button!==0)return;trajView.drag=true;trajView.x=e.clientX;trajView.y=e.clientY;trajCanvas.setPointerCapture(e.pointerId);trajCanvas.parentElement.classList.add('dragging');pause();}});
trajCanvas.addEventListener('pointermove',e=>{{if(!trajView.drag)return;const dx=e.clientX-trajView.x,dy=e.clientY-trajView.y;trajView.x=e.clientX;trajView.y=e.clientY;trajView.yaw+=dx*0.01;trajView.pitch=clamp(trajView.pitch+dy*0.01,-3.05,3.05);drawAll();}});
trajCanvas.addEventListener('pointerup',e=>{{trajView.drag=false;trajCanvas.parentElement.classList.remove('dragging');}});
trajCanvas.addEventListener('pointercancel',e=>{{trajView.drag=false;trajCanvas.parentElement.classList.remove('dragging');}});
trajCanvas.addEventListener('wheel',e=>{{e.preventDefault();trajView.zoom=clamp(trajView.zoom*(e.deltaY<0?1.08:0.92),0.45,3.0);drawAll();}},{{passive:false}});

function resizeOne(c){{const r=c.getBoundingClientRect(),d=window.devicePixelRatio||1;c.width=Math.max(160,Math.floor(r.width*d));c.height=Math.max(120,Math.floor(r.height*d));}} function drawImageFit(ctx,c,im,bg='#06101d'){{const w=c.width,h=c.height;ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle=bg;ctx.fillRect(0,0,w,h);if(!im.complete||!im.naturalWidth){{im.onload=()=>drawAll();return}}const s=Math.min(w/im.naturalWidth,h/im.naturalHeight),iw=im.naturalWidth*s,ih=im.naturalHeight*s;ctx.drawImage(im,(w-iw)/2,(h-ih)/2,iw,ih)}} function drawFrames(){{preload(frame);drawImageFit(rgbCtx,rgbCanvas,loadImage(rgbFrames[frame]));drawTrajectory();drawTactile();}}
function frameForTime(t){{let lo=0,hi=FRAME_COUNT-1;while(lo<hi){{const mid=Math.ceil((lo+hi)/2);if(FRAME_TIMES[mid]<=t)lo=mid;else hi=mid-1}}return lo}} function updateReadout(t){{t=Math.max(0,Math.min(DURATION,t||0));scrub.value=String(t);timeNow.textContent=t.toFixed(2);frameTag.textContent=frame+' / '+(FRAME_COUNT-1);const p=nearest(t);if(p){{lxNow.textContent=fmt(p.lx);lyNow.textContent=fmt(p.ly);lzNow.textContent=fmt(p.lz);rxNow.textContent=fmt(p.rx);ryNow.textContent=fmt(p.ry);rzNow.textContent=fmt(p.rz)}}drawChart(t)}} function seekTime(t){{playTime=Math.max(0,Math.min(DURATION,t||0));frame=frameForTime(playTime);drawFrames();updateReadout(playTime)}} function setFrame(i){{frame=Math.max(0,Math.min(FRAME_COUNT-1,Math.round(i)));playTime=FRAME_TIMES[frame]||0;drawFrames();updateReadout(playTime)}} function drawAll(){{drawFrames();updateReadout(playTime)}} function tick(ts){{if(!lastTs)lastTs=ts;const dt=(ts-lastTs)/1000;lastTs=ts;if(playing){{playTime+=dt;if(playTime>DURATION)playTime=0;frame=frameForTime(playTime);drawAll()}}requestAnimationFrame(tick)}} function play(){{playing=true;playBtn.textContent='暂停';lastTs=0}} function pause(){{playing=false;playBtn.textContent='播放'}} playBtn.onclick=()=>playing?pause():play(); resetBtn.onclick=()=>{{pause();resetTrajView();setFrame(0)}}; scrub.oninput=()=>{{pause();seekTime(Number(scrub.value))}};
function resize(){{fitShell();resizeOne(rgbCanvas);resizeOne(trajCanvas);resizeOne(tactileCanvas);const r=canvas.getBoundingClientRect(),d=window.devicePixelRatio||1;canvas.width=Math.max(320,Math.floor(r.width*d));canvas.height=Math.max(180,Math.floor(r.height*d));drawAll()}} function drawChart(t){{const dpr=window.devicePixelRatio||1,w=canvas.width,h=canvas.height,p={{l:54*dpr,r:18*dpr,t:16*dpr,b:34*dpr}};ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,w,h);const keys=['lx','ly','lz','rx','ry','rz'],vals=[];DATA.forEach(d=>keys.forEach(k=>{{if(Number.isFinite(d[k]))vals.push(d[k])}}));let mn=vals.length?Math.min(...vals):-1,mx=vals.length?Math.max(...vals):1,sp=Math.max(.001,mx-mn);mn-=sp*.08;mx+=sp*.08;const sx=v=>p.l+(w-p.l-p.r)*v/DURATION,sy=v=>p.t+(h-p.t-p.b)*(1-(v-mn)/(mx-mn));ctx.strokeStyle='#e2e8f0';ctx.lineWidth=1*dpr;ctx.fillStyle='#64748b';ctx.font=`${{12*dpr}}px system-ui`;for(let i=0;i<=5;i++){{const y=p.t+(h-p.t-p.b)*i/5;ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(w-p.r,y);ctx.stroke();ctx.fillText((mx-(mx-mn)*i/5).toFixed(2),8*dpr,y+4*dpr)}}for(let i=0;i<=4;i++){{const x=p.l+(w-p.l-p.r)*i/4;ctx.beginPath();ctx.moveTo(x,p.t);ctx.lineTo(x,h-p.b);ctx.stroke();ctx.fillText((DURATION*i/4).toFixed(1)+'s',x-12*dpr,h-10*dpr)}}function line(k,c){{ctx.beginPath();let drawing=false;DATA.forEach(pt=>{{if(!Number.isFinite(pt[k])){{drawing=false;return}}const x=sx(pt.t),y=sy(pt[k]);if(!drawing){{ctx.moveTo(x,y);drawing=true}}else ctx.lineTo(x,y)}});ctx.strokeStyle=c;ctx.lineWidth=2.1*dpr;ctx.stroke()}}line('lx','#2563eb');line('ly','#ea580c');line('lz','#16a34a');line('rx','#7c3aed');line('ry','#db2777');line('rz','#0891b2');const x=sx(t);ctx.strokeStyle='#0f172a';ctx.lineWidth=1.6*dpr;ctx.beginPath();ctx.moveTo(x,p.t);ctx.lineTo(x,h-p.b);ctx.stroke()}}
document.querySelector('.legend').insertAdjacentHTML('beforeend','<span class="hhx">头部 x</span><span class="hhy">头部 y</span><span class="hhz">头部 z</span>');
document.querySelector('.current').insertAdjacentHTML('beforeend','<div class="readout"><b>头部 camera x · world</b><span id="hxNow">--</span></div><div class="readout"><b>头部 camera y · world</b><span id="hyNow">--</span></div><div class="readout"><b>头部 camera z · world</b><span id="hzNow">--</span></div>');
const hxNow=document.getElementById('hxNow'),hyNow=document.getElementById('hyNow'),hzNow=document.getElementById('hzNow');
const updateReadoutBase=updateReadout;
updateReadout=function(t){{updateReadoutBase(t);const p=nearest(Math.max(0,Math.min(DURATION,t||0)));if(p){{hxNow.textContent=fmt(p.hx);hyNow.textContent=fmt(p.hy);hzNow.textContent=fmt(p.hz)}}}};
drawChart=function(t){{const dpr=window.devicePixelRatio||1,w=canvas.width,h=canvas.height,p={{l:54*dpr,r:18*dpr,t:16*dpr,b:34*dpr}};ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,w,h);const series=[['lx','#2563eb'],['ly','#ea580c'],['lz','#16a34a'],['rx','#7c3aed'],['ry','#db2777'],['rz','#0891b2'],['hx','#0f766e'],['hy','#64748b'],['hz','#a16207']],vals=[];DATA.forEach(d=>series.forEach(s=>{{if(Number.isFinite(d[s[0]]))vals.push(d[s[0]])}}));let mn=vals.length?Math.min(...vals):-1,mx=vals.length?Math.max(...vals):1,sp=Math.max(.001,mx-mn);mn-=sp*.08;mx+=sp*.08;const sx=v=>p.l+(w-p.l-p.r)*v/DURATION,sy=v=>p.t+(h-p.t-p.b)*(1-(v-mn)/(mx-mn));ctx.strokeStyle='#e2e8f0';ctx.lineWidth=1*dpr;ctx.fillStyle='#64748b';ctx.font=`${{12*dpr}}px system-ui`;for(let i=0;i<=5;i++){{const y=p.t+(h-p.t-p.b)*i/5;ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(w-p.r,y);ctx.stroke();ctx.fillText((mx-(mx-mn)*i/5).toFixed(2),8*dpr,y+4*dpr)}}for(let i=0;i<=4;i++){{const x=p.l+(w-p.l-p.r)*i/4;ctx.beginPath();ctx.moveTo(x,p.t);ctx.lineTo(x,h-p.b);ctx.stroke();ctx.fillText((DURATION*i/4).toFixed(1)+'s',x-12*dpr,h-10*dpr)}}function line(k,c){{ctx.beginPath();let drawing=false;DATA.forEach(pt=>{{if(!Number.isFinite(pt[k])){{drawing=false;return}}const x=sx(pt.t),y=sy(pt[k]);if(!drawing){{ctx.moveTo(x,y);drawing=true}}else ctx.lineTo(x,y)}});ctx.strokeStyle=c;ctx.lineWidth=2.1*dpr;ctx.stroke()}}series.forEach(s=>line(s[0],s[1]));const x=sx(t);ctx.strokeStyle='#0f172a';ctx.lineWidth=1.6*dpr;ctx.beginPath();ctx.moveTo(x,p.t);ctx.lineTo(x,h-p.b);ctx.stroke()}};
window.addEventListener('resize',resize); resize(); requestAnimationFrame(tick);
</script></body></html>'''
    out = outputs / output_subdir / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate review web page for a postprocess session.")
    parser.add_argument("--session", required=True, help="Session directory under postprocess_data")
    parser.add_argument("--hand_display_rotate_deg", type=float, default=45.0, help="Visualization-only rotation of non-wrist hand joints around the wrist in the rendered X/Z plane.")
    parser.add_argument("--hand_display_scale", type=float, default=1.0, help="Visualization-only scale for non-wrist hand joints around the wrist in the 3D review panel.")
    parser.add_argument("--scene_display_scale", type=float, default=1.0, help="Visualization-only scale for the rendered 3D scene and wrist trajectory length.")
    parser.add_argument("--no_align_middle_vertical", dest="align_middle_vertical", action="store_false", help="Disable per-frame display alignment that keeps wrist->middle_mcp vertical.")
    parser.set_defaults(align_middle_vertical=True)
    parser.add_argument("--rgb_max_width", type=int, default=960, help="Max width for self-contained RGB review frames; <=0 keeps original width.")
    parser.add_argument("--rgb_jpeg_quality", type=int, default=88, help="JPEG quality for self-contained RGB review frames.")
    parser.add_argument("--rgb_workers", type=int, default=8, help="Parallel RGB PNG-to-JPEG workers.")
    parser.add_argument("--fps", type=float, default=30.0, help="Fallback/nominal FPS only; playback timing comes from trajectory rgb_stamp_ns.")
    parser.add_argument("--output_subdir", default="web", help="Subdirectory under outputs for the generated web page, e.g. web_collect to avoid overwriting outputs/web.")
    args = parser.parse_args()
    session = Path(args.session).expanduser().resolve()
    outputs = session / "outputs"
    traj_path = outputs / "data" / "trajectory_wristroot_track_cameraoptical.jsonl"
    rows, all_points, _wrists, hand_offsets, camera_axes = load_rows(traj_path)
    web_dir = outputs / args.output_subdir
    remove_legacy_web_frame_dirs(web_dir)
    rgb_written = render_rgb_frames(
        session,
        web_dir / "rgb_frames",
        len(rows),
        max_width=args.rgb_max_width,
        jpeg_quality=args.rgb_jpeg_quality,
        workers=args.rgb_workers,
    )
    tactile_data = build_tactile_web_data(traj_path, len(rows))
    tactile_asset = export_tactile_hand_asset(web_dir / "tactile_hand.png")
    frame_times, actual_fps = load_frame_times(traj_path, fallback_fps=args.fps)
    chart_rows = build_chart_data(traj_path, frame_times)
    web_hand_display_scale = float(args.hand_display_scale)
    trajectory_data = build_trajectory_web_data(rows, all_points, hand_display_scale=web_hand_display_scale)
    html_path = write_html(session, chart_rows, trajectory_data, tactile_data, frame_times, fps=actual_fps, output_subdir=args.output_subdir)
    print(json.dumps({"html": str(html_path), "frames": len(rows), "rgb_frames": rgb_written, "trajectory_renderer": "canvas", "tactile_renderer": "canvas", "tactile_asset": str(tactile_asset), "hand_display_rotate_deg": args.hand_display_rotate_deg, "hand_display_scale": args.hand_display_scale, "web_hand_display_scale": web_hand_display_scale, "scene_display_scale": args.scene_display_scale, "fps": actual_fps, "duration_sec": frame_times[-1] if frame_times else 0.0, "timebase": "rgb_stamp_ns", "output_subdir": args.output_subdir, "align_middle_vertical": args.align_middle_vertical}, ensure_ascii=False))


if __name__ == "__main__":
    main()
