#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the review web page for one postprocess session.

Outputs:
  outputs/web/index.html
  outputs/web/traj_frames/*.jpg

The page uses frame-based playback rather than browser video decoding, so it
works even when MP4/H.264 support is unavailable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
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
    (56, 181, 255),
    (77, 154, 255),
    (79, 214, 132),
    (255, 169, 65),
    (196, 113, 255),
]
C_HEAD = (210, 210, 80)
C_WRIST = (255, 205, 85)
C_TEXT = (218, 228, 240)
C_AXIS = [(64, 84, 239), (86, 200, 92), (235, 146, 62)]

TACTILE_SENSOR_COUNT = 68
TACTILE_FRAME_W = 676
TACTILE_FRAME_H = 328
TACTILE_BASELINE_FRAMES = 10
TACTILE_MIN_SCALE = 0.02
TACTILE_DISPLAY_SCALE_MULTIPLIER = 1.25
TACTILE_HEAT_BLUR_NORMALIZER = 0.72


def read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


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
        pts = (row.get("glove") or {}).get("kpts_3d_world_m")
        mat = (row.get("head_pose") or {}).get("c2w") or (row.get("camera") or {}).get("c2w")
        item: Dict[str, Any] = {"hand": None, "head": None, "axes": None}
        if pts is not None:
            arr = np.asarray(pts, dtype=np.float64)
            if arr.shape[0] >= 21 and np.isfinite(arr[:21, :3]).all():
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
    def __init__(self, points: List[np.ndarray], hand_offsets: Optional[List[np.ndarray]] = None, camera_axes: Optional[List[np.ndarray]] = None, wrists: Optional[List[np.ndarray]] = None, width: int = 676, height: int = 466, scene_scale: float = 1.0) -> None:
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
            # use a camera-rear, slightly-above oblique view for the review panel.
            self.view_u = -cam_axes[:, 0].astype(np.float64)
            self.view_u[2] = 0.0
            if np.linalg.norm(self.view_u) < 1e-6:
                self.view_u = cam_axes[:, 0].astype(np.float64)
            self.view_u /= max(np.linalg.norm(self.view_u), 1e-12)
            top = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            oblique_dir = top * 0.55 - cam_axes[:, 2].astype(np.float64) * 0.35
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
    minor = (26, 38, 55)
    major = (50, 63, 84)
    minor_step = 28
    major_step = minor_step * 4
    for x in range(-major_step, w + major_step, minor_step):
        is_major = (x // minor_step) % 4 == 0
        color = major if is_major else minor
        alpha = 0.58 if is_major else 0.34
        draw_line(img, (x, 0), (x, h), color, 1, alpha)
    for y in range(-major_step, h + major_step, minor_step):
        is_major = (y // minor_step) % 4 == 0
        color = major if is_major else minor
        alpha = 0.58 if is_major else 0.34
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
    wrist_path: List[np.ndarray] = []
    head_path: List[np.ndarray] = []
    for idx, row in enumerate(rows):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for yy in range(height):
            v = int(8 + 18 * (1 - yy / height))
            img[yy, :] = (v, int(v * 0.92), int(v * 0.82))
        draw_back_wall(img, projector)

        hand: Optional[np.ndarray] = row.get("hand")
        if hand is not None:
            if align_middle_vertical:
                hand = rotate_hand_about_wrist_for_display(hand, 0.0, scale=hand_display_scale, view_u=projector.view_u, view_v=projector.view_v)
                hand = align_middle_vertical_for_display(hand, projector.view_u, projector.view_v, middle_idx=9)
            else:
                hand = rotate_hand_about_wrist_for_display(hand, hand_display_rotate_deg, scale=hand_display_scale, view_u=projector.view_u, view_v=projector.view_v)
        head: Optional[np.ndarray] = row.get("head")
        axes: Optional[np.ndarray] = row.get("axes")
        if hand is not None:
            wrist_path.append(hand[0].copy())
        if head is not None:
            head_path.append(head.copy())

        for path, color, thickness in ((head_path[-120:], C_HEAD, 2), (wrist_path[-120:], C_WRIST, 2)):
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

        if hand is not None:
            for a, b in BONES:
                draw_line(img, projector.project(hand[a]), projector.project(hand[b]), finger_color(b), 3, 0.96)
            for joint_idx, point in enumerate(hand):
                q = projector.project(point)
                color = (245, 248, 252) if joint_idx == 0 else finger_color(joint_idx)
                cv2.circle(img, q, 4 if joint_idx else 5, color, -1, cv2.LINE_AA)
                cv2.circle(img, q, 5 if joint_idx else 6, (3, 7, 14), 1, cv2.LINE_AA)

        cv2.rectangle(img, (12, 12), (134, 42), (10, 24, 42), -1, cv2.LINE_AA)
        cv2.putText(img, "Head + hand 3D", (25, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_TEXT, 1, cv2.LINE_AA)
        x0, y0 = width - 300, 25
        cv2.line(img, (x0, y0), (x0 + 34, y0), C_WRIST, 2, cv2.LINE_AA)
        cv2.putText(img, "hand_l", (x0 + 42, y0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.43, C_TEXT, 1, cv2.LINE_AA)
        cv2.line(img, (x0 + 118, y0), (x0 + 152, y0), C_HEAD, 2, cv2.LINE_AA)
        cv2.putText(img, "head", (x0 + 160, y0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.43, C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(img, f"frame {idx:03d}/{len(rows) - 1}", (width - 120, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (178, 190, 205), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{idx:05d}.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), 91])


def render_rgb_frames(session: Path, out_dir: Path, frame_count: int, max_width: int = 960, jpeg_quality: int = 88) -> int:
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
    written = 0
    for idx in range(frame_count):
        src = src_root / f"{idx:05d}" / "rgb.png"
        if not src.exists():
            continue
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        if max_width > 0 and w > max_width:
            scale = max_width / float(w)
            img = cv2.resize(img, (max_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"{idx:05d}.jpg"), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        written += 1
    if written == 0:
        raise RuntimeError(f"No RGB frames exported from {src_root}")
    return written




def _load_tactile_layout() -> tuple[np.ndarray, Optional[np.ndarray]]:
    assets = Path(__file__).resolve().parents[1] / "tactile" / "assets"
    coords_path = assets / "sensor_layout_coords.npy"
    coords: Optional[np.ndarray] = None
    if coords_path.exists():
        try:
            loaded = np.load(str(coords_path)).astype(np.float64)
            if loaded.shape == (TACTILE_SENSOR_COUNT, 2) and np.isfinite(loaded).all():
                coords = loaded
        except Exception:
            coords = None
    if coords is None:
        xs = np.linspace(120, 620, 8)
        ys = np.linspace(100, 650, 10)
        coords = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)[:TACTILE_SENSOR_COUNT]

    base_path = assets / "sensor_layout.png"
    base = cv2.imread(str(base_path), cv2.IMREAD_COLOR) if base_path.exists() else None
    return coords, base


def _tactile_delta(values: np.ndarray) -> tuple[np.ndarray, float]:
    if values.size == 0:
        return values, 1.0
    n_base = min(TACTILE_BASELINE_FRAMES, values.shape[0])
    baseline = np.nanmedian(values[:n_base], axis=0) if n_base > 0 else np.zeros(TACTILE_SENSOR_COUNT)
    delta = np.clip(values - baseline[None, :], 0.0, None)
    finite = delta[np.isfinite(delta)]
    if finite.size == 0:
        return np.zeros_like(values), 1.0
    vmax = float(np.percentile(finite, 99.0))
    if not np.isfinite(vmax) or vmax <= 1e-9:
        vmax = float(np.nanmax(finite)) if finite.size else 1.0
    vmax = max(vmax, TACTILE_MIN_SCALE) * TACTILE_DISPLAY_SCALE_MULTIPLIER
    return delta, vmax


def load_tactile_rows(traj_path: Path, frame_count: int) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    left: List[List[float]] = []
    right: List[List[float]] = []
    has_left = False
    has_right = False
    for line in traj_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        hand_frame = row.get("hand_frame") or {}
        pair = []
        for key in ("pressure_left", "pressure_right"):
            raw = hand_frame.get(key)
            ok = isinstance(raw, list) and len(raw) == TACTILE_SENSOR_COUNT
            if ok:
                arr = [float(v) for v in raw]
                if any(np.isfinite(arr)) and any(abs(v) > 1e-9 for v in arr if np.isfinite(v)):
                    if key.endswith("left"):
                        has_left = True
                    else:
                        has_right = True
            else:
                arr = [0.0] * TACTILE_SENSOR_COUNT
            pair.append(arr)
        left.append(pair[0])
        right.append(pair[1])
    while len(left) < frame_count:
        left.append([0.0] * TACTILE_SENSOR_COUNT)
        right.append([0.0] * TACTILE_SENSOR_COUNT)
    return (
        np.asarray(left[:frame_count], dtype=np.float64),
        np.asarray(right[:frame_count], dtype=np.float64),
        has_left,
        has_right,
    )


def _make_tactile_base(base: Optional[np.ndarray], coords: np.ndarray) -> np.ndarray:
    if base is None or base.size == 0:
        x_max = int(max(720, np.ceil(coords[:, 0].max() + 120)))
        y_max = int(max(760, np.ceil(coords[:, 1].max() + 120)))
        return np.full((y_max, x_max, 3), (24, 25, 28), dtype=np.uint8)
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    out = np.full(base.shape[:2] + (3,), (23, 24, 27), dtype=np.uint8)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    if float(np.mean(gray)) > 127.0:
        ink = np.clip((248.0 - gray.astype(np.float32)) / 190.0, 0.0, 1.0)
    else:
        ink = np.clip(gray.astype(np.float32) / 220.0, 0.0, 1.0)
    ink = cv2.GaussianBlur(ink, (0, 0), 0.7)
    fg = np.full_like(out, (205, 209, 213))
    alpha = (0.12 + 0.52 * ink)[..., None]
    out = (out.astype(np.float32) * (1.0 - alpha) + fg.astype(np.float32) * alpha).astype(np.uint8)
    return out


def _render_tactile_hand(values: np.ndarray, coords: np.ndarray, base: Optional[np.ndarray], vmax: float, mirror: bool = False) -> np.ndarray:
    hand = _make_tactile_base(base, coords).astype(np.float32)
    h, w = hand.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    norm_values = np.clip(values.astype(np.float64) / max(vmax, 1e-9), 0.0, 1.0)
    for (x, y), val in zip(coords, norm_values):
        if not np.isfinite(val) or val <= 0.001:
            continue
        cv2.circle(heat, (int(round(x)), int(round(y))), 34, float(val), -1, lineType=cv2.LINE_AA)
    heat = cv2.GaussianBlur(heat, (0, 0), 18.0)
    heat = np.clip(heat / TACTILE_HEAT_BLUR_NORMALIZER, 0.0, 1.0)
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_TURBO).astype(np.float32)
    alpha = (heat ** 0.78 * 0.82)[..., None]
    hand = hand * (1.0 - alpha) + color * alpha

    for (x, y), val in zip(coords, norm_values):
        c = (128, 134, 142) if val < 0.02 else (245, 245, 245)
        r = 2 if val < 0.02 else 3
        cv2.circle(hand, (int(round(x)), int(round(y))), r, c, -1, lineType=cv2.LINE_AA)

    margin = 76
    x0 = max(0, int(np.floor(coords[:, 0].min())) - margin)
    x1 = min(w, int(np.ceil(coords[:, 0].max())) + margin)
    y0 = max(0, int(np.floor(coords[:, 1].min())) - margin)
    y1 = min(h, int(np.ceil(coords[:, 1].max())) + margin)
    crop = hand[y0:y1, x0:x1].astype(np.uint8)
    if mirror:
        crop = cv2.flip(crop, 1)
    return crop


def _paste_fit(dst: np.ndarray, src: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    x, y, w, h = rect
    if src.size == 0 or w <= 0 or h <= 0:
        return
    scale = min(w / src.shape[1], h / src.shape[0])
    nw = max(1, int(round(src.shape[1] * scale)))
    nh = max(1, int(round(src.shape[0] * scale)))
    resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
    px = x + (w - nw) // 2
    py = y + (h - nh) // 2
    dst[py:py + nh, px:px + nw] = resized


def _draw_tactile_label(img: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    x, y = origin
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150, 157, 168), 1, cv2.LINE_AA)


def render_tactile_frames(traj_path: Path, out_dir: Path, frame_count: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    coords, base = _load_tactile_layout()
    left, right, has_left, has_right = load_tactile_rows(traj_path, frame_count)
    left_delta, left_vmax = _tactile_delta(left)
    right_delta, right_vmax = _tactile_delta(right)
    vmax = max(left_vmax if has_left else 0.0, right_vmax if has_right else 0.0, TACTILE_MIN_SCALE)

    if has_left and has_right:
        layout = "dual"
    else:
        layout = "single"
    written = 0
    for i in range(frame_count):
        canvas = np.full((TACTILE_FRAME_H, TACTILE_FRAME_W, 3), (23, 24, 27), dtype=np.uint8)
        if layout == "dual":
            left_img = _render_tactile_hand(left_delta[i], coords, base, vmax, mirror=False)
            right_img = _render_tactile_hand(right_delta[i], coords, base, vmax, mirror=True)
            _paste_fit(canvas, left_img, (18, 16, 304, 284))
            _paste_fit(canvas, right_img, (354, 16, 304, 284))
            _draw_tactile_label(canvas, "hand_left", (104, 311))
            _draw_tactile_label(canvas, "hand_right", (438, 311))
        else:
            values = left_delta[i] if has_left or not has_right else right_delta[i]
            single_img = _render_tactile_hand(values, coords, base, vmax, mirror=(has_right and not has_left))
            _paste_fit(canvas, single_img, (186, 14, 304, 292))
            _draw_tactile_label(canvas, "hand_left" if has_left or not has_right else "hand_right", (286, 313))
        peak = float(np.nanmax(left_delta[i] if has_left or not has_right else right_delta[i])) if frame_count else 0.0
        cv2.putText(canvas, f"peak dP {peak:.3f}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (184, 190, 200), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{i:05d}.jpg"), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        written += 1
    return written


def load_frame_times(traj_path: Path, fallback_fps: float = 30.0) -> tuple[List[float], float]:
    rows = [json.loads(line) for line in traj_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    fps = nominal_fps(rows, fallback=fallback_fps)
    return relative_times_sec(rows, fallback_fps=fps), fps


def build_chart_data(traj_path: Path, frame_times: List[float]) -> List[Dict[str, float]]:
    out = []
    lines = [line for line in traj_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row_index, line in enumerate(lines):
        row = json.loads(line)
        idx = int(row.get("idx", len(out)))
        pts = (row.get("glove") or {}).get("kpts_3d_world_m")
        if pts is None:
            continue
        wrist = np.asarray(pts, dtype=np.float64)[0]
        if np.isfinite(wrist).all():
            out.append({"i": idx, "t": frame_times[row_index], "x": float(wrist[0]), "y": float(wrist[1]), "z": float(wrist[2])})
    return out


def write_html(session: Path, chart_rows: List[Dict[str, float]], frame_times: List[float], fps: float = 30.0, output_subdir: str = "web_rear_top") -> Path:
    outputs = session / "outputs"
    summary = read_json(outputs / "summaries" / "trajectory_3d_camera_frame.json", {})
    wrist_summary = read_json(outputs / "summaries" / "wristroot_track_summary.json", {})
    frames = len(frame_times)
    valid = int(summary.get("valid_hand_camera", len(chart_rows)))
    fps = float(fps) if float(fps) > 0 else 30.0
    duration = frame_times[-1] if frame_times else 0.0
    if duration <= 0.0 and frames:
        duration = 1.0 / fps
    step = summary.get("wrist_camera_step_m", {})
    params = wrist_summary.get("params", {})
    chart_json = json.dumps(chart_rows, separators=(",", ":"))
    frame_times_json = json.dumps(frame_times, separators=(",", ":"))
    html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>第一视角采集</title>
<style>
:root{{--bg:#eaf1f8;--panel:#fff;--ink:#172033;--muted:#64748b;--line:#d9e4ef;--blue:#2563eb;--green:#16a34a;--orange:#ea580c;--shadow:0 10px 24px rgba(25,42,70,.10);--scale:1}}
*{{box-sizing:border-box}} body{{margin:0;width:100vw;height:100vh;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;overflow:hidden}}
.shell{{position:absolute;left:50%;top:50%;width:1280px;height:760px;transform:translate(-50%,-50%) scale(var(--scale));transform-origin:center center;display:grid;grid-template-rows:42px 620px 44px;gap:10px;padding:18px 48px 16px;background:var(--bg)}}
.topbar{{display:grid;grid-template-columns:160px 1fr 160px;align-items:center;height:42px}}.topbar:before{{content:"退出采集";justify-self:start;width:88px;height:28px;line-height:26px;text-align:center;border:1px solid #ef4444;border-radius:4px;color:#ef4444;background:#fff;font-size:13px;font-weight:700}}.topbar>div:first-child{{grid-column:2;text-align:center}}.topbar h1{{margin:0;font-size:22px;line-height:1.1;color:#1d5ecb;font-weight:800;letter-spacing:0}}.sub{{display:none}}.status{{grid-column:3;justify-self:end;display:flex;align-items:center;gap:8px;padding:6px 10px;background:rgba(255,255,255,.9);border:1px solid var(--line);border-radius:6px;color:var(--muted);font-size:12px}}.dot{{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 0 3px rgba(22,163,74,.12)}}
.main{{display:grid;grid-template-columns:676px 474px;gap:28px;height:620px;min-height:0;align-items:start}}.stage{{display:grid;grid-template-rows:380px 230px;gap:10px;width:676px;height:620px;min-height:0}}.lowerStage{{display:grid;grid-template-columns:1fr 1fr;gap:10px;min-height:0}}.panel{{background:var(--panel);border:1px solid var(--line);border-radius:6px;box-shadow:var(--shadow);overflow:hidden}}.videoPanel{{position:relative;background:#06101d;height:100%;min-height:0}}.videoPanel canvas{{width:100%;height:100%;display:block;background:#06101d}}.badge{{position:absolute;left:10px;top:10px;z-index:2;padding:5px 9px;border-radius:3px;background:rgba(5,20,38,.82);color:#e5edf6;font-size:12px;backdrop-filter:blur(8px)}}
.side{{width:474px;height:620px;padding:12px;display:grid;grid-template-rows:92px 1fr 46px;gap:8px;min-height:0}}.section-title{{display:flex;align-items:center;justify-content:center;position:relative;font-weight:800;font-size:15px;margin-bottom:9px;color:#334155}}.section-title:before,.section-title:after{{content:"";height:1px;background:#e2e8f0;flex:1;margin:0 18px}}.frameTag{{position:absolute;right:0;color:var(--muted);font-size:11px;font-weight:600}}.grid3{{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}}.metric{{height:58px;padding:9px;border:1px solid #e7eef6;border-radius:6px;background:#f8fbfe;text-align:center}}.label{{color:var(--muted);font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.value{{margin-top:5px;color:var(--blue);font-size:17px;font-weight:800}}.unit{{color:var(--muted);font-size:11px;margin-left:2px}}.chartBox{{height:100%;min-height:0;display:grid;grid-template-rows:24px 1fr;padding:8px;border:1px solid #e4edf6;border-radius:6px;background:#fbfdff}}.legend{{display:flex;gap:12px;color:var(--muted);font-size:11px;padding:1px 4px 5px}}.legend span:before{{content:"";display:inline-block;width:16px;height:3px;border-radius:9px;margin-right:5px;vertical-align:middle}}.lx:before{{background:var(--blue)}}.ly:before{{background:var(--orange)}}.lz:before{{background:var(--green)}}#chartCanvas{{width:100%;height:100%;min-height:0}}.current{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}.readout{{border:1px solid #e7eef6;border-radius:5px;padding:8px 9px;background:#fff;height:46px}}.readout b{{display:block;color:var(--muted);font-size:11px;margin-bottom:4px}}.readout span{{font-size:14px;font-weight:760}}.timeline{{display:grid;grid-template-columns:122px 1fr 116px;align-items:center;gap:10px;height:44px;padding:7px 10px;background:var(--panel);border:1px solid var(--line);border-radius:6px;box-shadow:var(--shadow)}}.controls{{display:flex;gap:7px}}button{{height:30px;min-width:56px;border:0;border-radius:6px;background:var(--blue);color:white;padding:0 12px;font-size:12px;font-weight:750;cursor:pointer}}button.secondary{{min-width:50px;background:#e2e8f0;color:#1e293b}}input[type=range]{{width:100%;accent-color:var(--blue)}}.time{{min-width:108px;text-align:right;color:var(--muted);font-size:12px}}
</style></head>
<body><div class="shell"><header class="topbar"><div><h1>第一人称数据采集</h1><div class="sub">review</div></div><div class="status"><span class="dot"></span><span>审核进度 · 69/100</span></div></header>
<main class="main"><section class="stage"><div class="panel videoPanel"><div class="badge">RGB</div><canvas id="rgbCanvas"></canvas></div><div class="lowerStage"><div class="panel videoPanel"><div class="badge">Head + hand 3D</div><canvas id="trajCanvas"></canvas></div><div class="panel videoPanel"><div class="badge">Tactile heatmap</div><canvas id="tactileCanvas"></canvas></div></div></section>
<aside class="panel side"><section><div class="section-title"><span>采集信息</span><span class="frameTag" id="frameTag">0 / {frames}</span></div><div class="grid3"><div class="metric"><div class="label">采集时长</div><div class="value">{duration:.2f}<span class="unit">s</span></div></div><div class="metric"><div class="label">实时帧率</div><div class="value">{fps:.1f}<span class="unit">fps</span></div></div><div class="metric"><div class="label">有效手帧</div><div class="value">{valid}<span class="unit">/{frames}</span></div></div></div></section><section class="chartBox"><div class="legend"><span class="lx">x</span><span class="ly">y</span><span class="lz">z</span></div><canvas id="chartCanvas"></canvas></section><section class="current"><div class="readout"><b>world x</b><span id="xNow">--</span></div><div class="readout"><b>world y</b><span id="yNow">--</span></div><div class="readout"><b>world z</b><span id="zNow">--</span></div></section></aside></main>
<footer class="timeline"><div class="controls"><button id="playBtn">播放</button><button class="secondary" id="resetBtn">重置</button></div><input id="scrub" type="range" min="0" max="{duration:.6f}" step="0.01" value="0"><div class="time"><span id="timeNow">0.00</span>s / {duration:.2f}s</div></footer></div>
<script>
function fitShell(){{const scale=Math.min(window.innerWidth/1280,window.innerHeight/760);document.documentElement.style.setProperty('--scale',String(scale));}}
window.addEventListener('resize',fitShell);fitShell();
const DATA={chart_json}; const FRAME_TIMES={frame_times_json}; const DURATION={duration:.6f}; const FPS={fps:.6f}; const FRAME_COUNT={frames};
const rgbFrames=Array.from({{length:FRAME_COUNT}},(_,i)=>`rgb_frames/${{String(i).padStart(5,'0')}}.jpg`); const trajFrames=Array.from({{length:FRAME_COUNT}},(_,i)=>`traj_frames/${{String(i).padStart(5,'0')}}.jpg`); const tactileFrames=Array.from({{length:FRAME_COUNT}},(_,i)=>`tactile_frames/${{String(i).padStart(5,'0')}}.jpg`);
const TACTILE_LIVE_URL='http://127.0.0.1:8790'; const TACTILE_POINTS=[[0.876,0.518],[0.909,0.502],[0.815,0.607],[0.850,0.586],[0.614,0.183],[0.643,0.183],[0.599,0.344],[0.632,0.344],[0.453,0.115],[0.482,0.115],[0.453,0.288],[0.482,0.288],[0.294,0.176],[0.323,0.176],[0.309,0.344],[0.342,0.344],[0.155,0.318],[0.184,0.318],[0.198,0.483],[0.231,0.483],[0.637,0.663],[0.581,0.663],[0.526,0.663],[0.470,0.663],[0.414,0.663],[0.358,0.663],[0.303,0.663],[0.247,0.663],[0.663,0.702],[0.602,0.702],[0.542,0.702],[0.481,0.702],[0.420,0.702],[0.359,0.702],[0.299,0.702],[0.238,0.702],[0.692,0.740],[0.626,0.740],[0.560,0.740],[0.494,0.740],[0.429,0.740],[0.363,0.740],[0.297,0.740],[0.231,0.740],[0.677,0.778],[0.615,0.778],[0.553,0.778],[0.491,0.778],[0.429,0.778],[0.367,0.778],[0.305,0.778],[0.243,0.778],[0.625,0.817],[0.574,0.817],[0.522,0.817],[0.471,0.817],[0.419,0.817],[0.368,0.817],[0.316,0.817],[0.265,0.817],[0.570,0.855],[0.530,0.855],[0.490,0.855],[0.450,0.855],[0.409,0.855],[0.369,0.855],[0.329,0.855],[0.289,0.855]];
const tactileHand=new Image(); tactileHand.crossOrigin='anonymous'; tactileHand.src=TACTILE_LIVE_URL+'/assets/hand.png'; let tactileLive=null,tactileLiveOk=false;

const rgbCanvas=document.getElementById('rgbCanvas'),trajCanvas=document.getElementById('trajCanvas'),tactileCanvas=document.getElementById('tactileCanvas'),rgbCtx=rgbCanvas.getContext('2d'),trajCtx=trajCanvas.getContext('2d'),tactileCtx=tactileCanvas.getContext('2d'),playBtn=document.getElementById('playBtn'),resetBtn=document.getElementById('resetBtn'),scrub=document.getElementById('scrub'),timeNow=document.getElementById('timeNow'),frameTag=document.getElementById('frameTag'),xNow=document.getElementById('xNow'),yNow=document.getElementById('yNow'),zNow=document.getElementById('zNow'),canvas=document.getElementById('chartCanvas'),ctx=canvas.getContext('2d'); let frame=0,playing=false,lastTs=0,playTime=0; const imgCache=new Map();
function nearest(t){{let b=null,bd=1e9;for(const p of DATA){{const d=Math.abs(p.t-t);if(d<bd){{bd=d;b=p}}}}return b}} function fmt(v){{return Number.isFinite(v)?v.toFixed(3)+' m':'--'}} function loadImage(src){{if(imgCache.has(src))return imgCache.get(src);const im=new Image();im.src=src;imgCache.set(src,im);return im}} function preload(i){{for(let k=-2;k<=8;k++){{const j=Math.max(0,Math.min(FRAME_COUNT-1,i+k));loadImage(rgbFrames[j]);loadImage(trajFrames[j]);loadImage(tactileFrames[j]);}}}}
function resizeOne(c){{const r=c.getBoundingClientRect(),d=window.devicePixelRatio||1;c.width=Math.max(160,Math.floor(r.width*d));c.height=Math.max(120,Math.floor(r.height*d));}}
function drawImageFit(ctx,c,im,mode='contain'){{const w=c.width,h=c.height;ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle='#06101d';ctx.fillRect(0,0,w,h);if(!im.complete||!im.naturalWidth){{im.onload=()=>drawAll();return}}let s=Math.min(w/im.naturalWidth,h/im.naturalHeight);if(mode==='cover')s=Math.max(w/im.naturalWidth,h/im.naturalHeight);else if(mode==='zoom')s=Math.min(Math.max(w/im.naturalWidth,h/im.naturalHeight),s*1.18);const iw=im.naturalWidth*s,ih=im.naturalHeight*s;ctx.drawImage(im,(w-iw)/2,(h-ih)/2,iw,ih)}}
function tactileColor(v,a){{const stops=[[0,35,108,170],[.45,44,210,220],[.7,65,230,205],[.86,255,178,64],[1,255,76,76]];let hi=1;while(hi<stops.length-1&&v>stops[hi][0])hi++;const p=stops[hi-1],q=stops[hi],t=(v-p[0])/(q[0]-p[0]);return 'rgba('+(p[1]+(q[1]-p[1])*t)+','+(p[2]+(q[2]-p[2])*t)+','+(p[3]+(q[3]-p[3])*t)+','+a+')'}}
function drawTactileLive(){{const w=tactileCanvas.width,h=tactileCanvas.height;tactileCtx.setTransform(1,0,0,1,0,0);tactileCtx.fillStyle='#06101d';tactileCtx.fillRect(0,0,w,h);if(tactileHand.complete&&tactileHand.naturalWidth){{const s=Math.min(w/tactileHand.naturalWidth,h/tactileHand.naturalHeight)*1.04,iw=tactileHand.naturalWidth*s,ih=tactileHand.naturalHeight*s;tactileCtx.globalAlpha=.9;tactileCtx.drawImage(tactileHand,(w-iw)/2,(h-ih)/2,iw,ih);tactileCtx.globalAlpha=1;}}
  const d=tactileLive,vals=d&&d.display?d.display:null;if(!vals){{tactileCtx.fillStyle='rgba(229,237,246,.72)';tactileCtx.font='12px system-ui';tactileCtx.fillText('waiting live tactile @ 127.0.0.1:8790',14,h-16);return}}
  const s=Math.min(w/920,h/1040)*1.04,ox=(w-920*s)/2,oy=(h-1040*s)/2;for(let i=0;i<TACTILE_POINTS.length;i++){{const p=TACTILE_POINTS[i],raw=Math.max(0,Math.min(100,vals[i]||0)),v=raw<=0?0:Math.pow(Math.min(1,raw/100),.62),x=ox+p[0]*920*s,y=oy+p[1]*1040*s;if(v>.01){{const r=2+v*12,g=tactileCtx.createRadialGradient(x,y,0,x,y,r);g.addColorStop(0,tactileColor(v,.9));g.addColorStop(.36,tactileColor(v,.34));g.addColorStop(1,tactileColor(v,0));tactileCtx.fillStyle=g;tactileCtx.beginPath();tactileCtx.arc(x,y,r,0,Math.PI*2);tactileCtx.fill();}}tactileCtx.fillStyle=tactileColor(Math.min(1,v+.12),v>.01?.92:.38);tactileCtx.beginPath();tactileCtx.arc(x,y,v>.01?2.0:1.1,0,Math.PI*2);tactileCtx.fill();}}
  tactileCtx.fillStyle='rgba(229,237,246,.76)';tactileCtx.font='11px system-ui';tactileCtx.fillText(tactileLiveOk?'live tactile · '+(d.peak_name||'--')+' · '+(d.peak||0).toFixed(1)+' ΔAD':'live tactile disconnected',12,h-12);}}
function startTactileLive(){{try{{const es=new EventSource(TACTILE_LIVE_URL+'/events');es.onmessage=e=>{{tactileLive=JSON.parse(e.data);tactileLiveOk=true;drawTactileLive();}};es.onerror=()=>{{tactileLiveOk=false;drawTactileLive();}};}}catch(e){{tactileLiveOk=false;}}}}
function drawFrames(){{preload(frame);drawImageFit(rgbCtx,rgbCanvas,loadImage(rgbFrames[frame]));drawImageFit(trajCtx,trajCanvas,loadImage(trajFrames[frame]));drawTactileLive();}} startTactileLive();
function frameForTime(t){{let lo=0,hi=FRAME_COUNT-1;while(lo<hi){{const mid=Math.ceil((lo+hi)/2);if(FRAME_TIMES[mid]<=t)lo=mid;else hi=mid-1}}return lo}} function updateReadout(t){{t=Math.max(0,Math.min(DURATION,t||0));scrub.value=String(t);timeNow.textContent=t.toFixed(2);frameTag.textContent=frame+' / '+(FRAME_COUNT-1);const p=nearest(t);if(p){{xNow.textContent=fmt(p.x);yNow.textContent=fmt(p.y);zNow.textContent=fmt(p.z)}}drawChart(t)}} function seekTime(t){{playTime=Math.max(0,Math.min(DURATION,t||0));frame=frameForTime(playTime);drawFrames();updateReadout(playTime)}} function setFrame(i){{frame=Math.max(0,Math.min(FRAME_COUNT-1,Math.round(i)));playTime=FRAME_TIMES[frame]||0;drawFrames();updateReadout(playTime)}} function drawAll(){{drawFrames();updateReadout(playTime)}} function tick(ts){{if(!lastTs)lastTs=ts;const dt=(ts-lastTs)/1000;lastTs=ts;if(playing){{playTime+=dt;if(playTime>DURATION)playTime=0;frame=frameForTime(playTime);drawAll()}}requestAnimationFrame(tick)}} function play(){{playing=true;playBtn.textContent='暂停';lastTs=0}} function pause(){{playing=false;playBtn.textContent='播放'}} playBtn.onclick=()=>playing?pause():play(); resetBtn.onclick=()=>{{pause();setFrame(0)}}; scrub.oninput=()=>{{pause();seekTime(Number(scrub.value))}};
function resize(){{fitShell();resizeOne(rgbCanvas);resizeOne(trajCanvas);resizeOne(tactileCanvas);const r=canvas.getBoundingClientRect(),d=window.devicePixelRatio||1;canvas.width=Math.max(320,Math.floor(r.width*d));canvas.height=Math.max(220,Math.floor(r.height*d));drawAll()}} function drawChart(t){{const dpr=window.devicePixelRatio||1,w=canvas.width,h=canvas.height,p={{l:54*dpr,r:18*dpr,t:16*dpr,b:34*dpr}};ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,w,h);const vals=[];DATA.forEach(d=>vals.push(d.x,d.y,d.z));let mn=Math.min(...vals),mx=Math.max(...vals),sp=Math.max(.001,mx-mn);mn-=sp*.08;mx+=sp*.08;const sx=v=>p.l+(w-p.l-p.r)*v/DURATION,sy=v=>p.t+(h-p.t-p.b)*(1-(v-mn)/(mx-mn));ctx.strokeStyle='#e2e8f0';ctx.lineWidth=1*dpr;ctx.fillStyle='#64748b';ctx.font=`${{12*dpr}}px system-ui`;for(let i=0;i<=5;i++){{const y=p.t+(h-p.t-p.b)*i/5;ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(w-p.r,y);ctx.stroke();ctx.fillText((mx-(mx-mn)*i/5).toFixed(2),8*dpr,y+4*dpr)}}for(let i=0;i<=4;i++){{const x=p.l+(w-p.l-p.r)*i/4;ctx.beginPath();ctx.moveTo(x,p.t);ctx.lineTo(x,h-p.b);ctx.stroke();ctx.fillText((DURATION*i/4).toFixed(1)+'s',x-12*dpr,h-10*dpr)}}function line(k,c){{ctx.beginPath();DATA.forEach((pt,i)=>{{const x=sx(pt.t),y=sy(pt[k]);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)}});ctx.strokeStyle=c;ctx.lineWidth=2.4*dpr;ctx.stroke()}}line('x','#2563eb');line('y','#ea580c');line('z','#16a34a');const x=sx(t);ctx.strokeStyle='#0f172a';ctx.lineWidth=1.6*dpr;ctx.beginPath();ctx.moveTo(x,p.t);ctx.lineTo(x,h-p.b);ctx.stroke()}}
window.addEventListener('resize',resize); resize(); requestAnimationFrame(tick);
</script></body></html>'''
    out = outputs / output_subdir / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rear-top camera-angle review web page for a postprocess session.")
    parser.add_argument("--session", required=True, help="Session directory under postprocess_data")
    parser.add_argument("--hand_display_rotate_deg", type=float, default=45.0, help="Visualization-only rotation of non-wrist hand joints around the wrist in the rendered X/Z plane.")
    parser.add_argument("--hand_display_scale", type=float, default=1.50, help="Visualization-only scale for non-wrist hand joints around the wrist in the 3D review panel.")
    parser.add_argument("--scene_display_scale", type=float, default=1.0, help="Visualization-only scale for the rendered 3D scene and wrist trajectory length.")
    parser.add_argument("--no_align_middle_vertical", dest="align_middle_vertical", action="store_false", help="Disable per-frame display alignment that keeps wrist->middle_mcp vertical.")
    parser.set_defaults(align_middle_vertical=True)
    parser.add_argument("--rgb_max_width", type=int, default=960, help="Max width for self-contained RGB review frames; <=0 keeps original width.")
    parser.add_argument("--rgb_jpeg_quality", type=int, default=88, help="JPEG quality for self-contained RGB review frames.")
    parser.add_argument("--fps", type=float, default=30.0, help="Fallback/nominal FPS only; playback timing comes from trajectory rgb_stamp_ns.")
    parser.add_argument("--output_subdir", default="web_rear_top", help="Subdirectory under outputs for this rear-top camera-angle web page.")
    args = parser.parse_args()
    session = Path(args.session).expanduser().resolve()
    outputs = session / "outputs"
    traj_path = outputs / "data" / "trajectory_wristroot_track_cameraoptical.jsonl"
    rows, all_points, _wrists, hand_offsets, camera_axes = load_rows(traj_path)
    projector = WorldProjector(all_points, hand_offsets=hand_offsets, camera_axes=camera_axes, wrists=_wrists, scene_scale=args.scene_display_scale)
    render_trajectory_frames(rows, projector, outputs / args.output_subdir / "traj_frames", hand_display_rotate_deg=args.hand_display_rotate_deg, hand_display_scale=args.hand_display_scale, align_middle_vertical=args.align_middle_vertical)
    rgb_written = render_rgb_frames(session, outputs / args.output_subdir / "rgb_frames", len(rows), max_width=args.rgb_max_width, jpeg_quality=args.rgb_jpeg_quality)
    tactile_written = render_tactile_frames(traj_path, outputs / args.output_subdir / "tactile_frames", len(rows))
    frame_times, actual_fps = load_frame_times(traj_path, fallback_fps=args.fps)
    chart_rows = build_chart_data(traj_path, frame_times)
    html_path = write_html(session, chart_rows, frame_times, fps=actual_fps, output_subdir=args.output_subdir)
    print(json.dumps({"html": str(html_path), "frames": len(rows), "rgb_frames": rgb_written, "traj_frames": str(outputs / args.output_subdir / "traj_frames"), "tactile_frames": tactile_written, "hand_display_rotate_deg": args.hand_display_rotate_deg, "hand_display_scale": args.hand_display_scale, "scene_display_scale": args.scene_display_scale, "fps": actual_fps, "duration_sec": frame_times[-1] if frame_times else 0.0, "timebase": "rgb_stamp_ns", "output_subdir": args.output_subdir, "align_middle_vertical": args.align_middle_vertical}, ensure_ascii=False))


if __name__ == "__main__":
    main()
