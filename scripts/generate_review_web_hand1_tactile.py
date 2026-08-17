#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate review web page with tactile heatmap rendered on hand_1.png.

This script reuses generate_review_web.py for the page, RGB export, trajectory
rendering, and data loading. Only the tactile renderer is replaced.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import generate_review_web as base


HAND1_ASSET = Path("/home/lenovo/Downloads/ourhost/assets/hand_1.png")
HAND1_MASK_GRAY_THRESHOLD = 248
HAND1_MIN_COMPONENT_AREA = 50000
HAND1_POINT_X_CENTER = 0.50
HAND1_POINT_X_SCALE = 0.72
HAND1_POINT_X_OFFSET = 0.035
HAND1_POINT_Y_SCALE = 0.86
HAND1_POINT_Y_OFFSET = 0.075
HAND1_GLOW_COLOR_STOPS = [
    (0.00, (255, 250, 232)),
    (0.45, (255, 255, 255)),
    (0.72, (255, 244, 190)),
    (0.88, (255, 214, 95)),
    (1.00, (255, 160, 64)),
]


def _foreground_mask(src_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray < HAND1_MASK_GRAY_THRESHOLD).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    return mask


def _crop_components(src_bgr: np.ndarray) -> list[np.ndarray]:
    mask = _foreground_mask(src_bgr)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes: list[tuple[int, int, int, int, int]] = []
    for idx in range(1, num):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area >= HAND1_MIN_COMPONENT_AREA:
            boxes.append((x, y, w, h, area))
    if len(boxes) < 2:
        raise RuntimeError(f"Expected two hand components in {HAND1_ASSET}, found {len(boxes)}")
    boxes = sorted(boxes, key=lambda b: b[0])[:2]
    crops: list[np.ndarray] = []
    for x, y, w, h, _ in boxes:
        pad = 10
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(src_bgr.shape[1], x + w + pad)
        y1 = min(src_bgr.shape[0], y + h + pad)
        bgr = src_bgr[y0:y1, x0:x1].astype(np.float32)
        alpha = mask[y0:y1, x0:x1].astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)
        rgba = np.dstack([bgr, np.clip(alpha * 255.0, 0, 255)])
        crops.append(rgba.astype(np.float32))
    return crops


def _load_live_tactile_hand() -> dict[str, np.ndarray]:
    img = cv2.imread(str(HAND1_ASSET), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise RuntimeError(f"Could not read tactile hand asset: {HAND1_ASSET}")
    left, right = _crop_components(img)
    return {"left": left, "right": right}


def _fit_rgba_to_rect(src_rgba: np.ndarray, rect_w: int, rect_h: int) -> np.ndarray:
    h, w = src_rgba.shape[:2]
    scale = min(rect_w / float(w), rect_h / float(h))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    return cv2.resize(src_rgba, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(np.float32)


def _alpha_bbox(rgba: np.ndarray) -> tuple[int, int, int, int]:
    alpha = rgba[:, :, 3] > 8
    ys, xs = np.where(alpha)
    if xs.size == 0 or ys.size == 0:
        return (0, 0, rgba.shape[1] - 1, rgba.shape[0] - 1)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _hand1_ramp(value: float) -> tuple[float, float, float]:
    v = float(np.clip(value, 0.0, 1.0))
    hi = 1
    while hi < len(HAND1_GLOW_COLOR_STOPS) - 1 and v > HAND1_GLOW_COLOR_STOPS[hi][0]:
        hi += 1
    p0, c0 = HAND1_GLOW_COLOR_STOPS[hi - 1]
    p1, c1 = HAND1_GLOW_COLOR_STOPS[hi]
    t = 0.0 if p1 <= p0 else (v - p0) / (p1 - p0)
    return tuple(c0[k] + (c1[k] - c0[k]) * t for k in range(3))


def _calibrate_point(px: float, py: float, mirror_points: bool) -> tuple[float, float]:
    sx = 1.0 - float(px) if mirror_points else float(px)
    sx = HAND1_POINT_X_CENTER + (sx - HAND1_POINT_X_CENTER) * HAND1_POINT_X_SCALE + HAND1_POINT_X_OFFSET
    sy = float(py) * HAND1_POINT_Y_SCALE + HAND1_POINT_Y_OFFSET
    return float(np.clip(sx, 0.02, 0.98)), float(np.clip(sy, 0.02, 0.98))


def _draw_hand1_tactile(
    canvas: np.ndarray,
    values: np.ndarray,
    points: np.ndarray,
    hand_rgba: np.ndarray,
    vmax: float,
    rect: tuple[int, int, int, int],
    mirror_points: bool = False,
) -> None:
    x, y, rect_w, rect_h = rect
    hand = _fit_rgba_to_rect(hand_rgba, rect_w, rect_h)
    hand_alpha = hand[:, :, 3:4].copy()
    display = np.clip(values.astype(np.float64) / max(vmax, 1e-9) * 100.0, 0.0, 100.0)
    bx0, by0, bx1, by1 = _alpha_bbox(hand)
    bw = max(1.0, float(bx1 - bx0))
    bh = max(1.0, float(by1 - by0))
    heat_bgr = hand[:, :, :3]
    for (px, py), value in zip(points, display):
        if not np.isfinite(value) or value <= 0.0:
            continue
        visual = float(np.power(min(1.0, value / 100.0), base.TACTILE_RESPONSE_GAMMA))
        sx, sy = _calibrate_point(float(px), float(py), mirror_points)
        cx = bx0 + sx * bw
        cy = by0 + sy * bh
        color = _hand1_ramp(visual)
        base._blend_radial(heat_bgr, (cx, cy), 7.0 + visual * 20.0, color)
        base._blend_circle(heat_bgr, (cx, cy), 2.0 + visual * 1.6, _hand1_ramp(min(1.0, visual + 0.22)), 0.88)
    hand[:, :, 3:4] = hand_alpha
    x0 = x + (rect_w - hand.shape[1]) // 2
    y0 = y + (rect_h - hand.shape[0]) // 2
    base._overlay_rgba(canvas, hand, x0, y0)


def _render_live_tactile_frame(
    left_values: np.ndarray,
    right_values: np.ndarray,
    points: np.ndarray,
    hand_asset: dict[str, np.ndarray],
    vmax: float,
) -> np.ndarray:
    canvas = np.empty((base.TACTILE_FRAME_H, base.TACTILE_FRAME_W, 3), dtype=np.float32)
    top = np.asarray(base.TACTILE_BG_TOP_BGR, dtype=np.float32)
    bottom = np.asarray(base.TACTILE_BG_BOTTOM_BGR, dtype=np.float32)
    for yy in range(base.TACTILE_FRAME_H):
        t = yy / max(1, base.TACTILE_FRAME_H - 1)
        canvas[yy, :] = top * (1.0 - t) + bottom * t
    _draw_hand1_tactile(canvas, left_values, points, hand_asset["left"], vmax, (30, 20, 294, 288), mirror_points=False)
    _draw_hand1_tactile(canvas, right_values, points, hand_asset["right"], vmax, (352, 20, 294, 288), mirror_points=True)
    return np.clip(canvas, 0.0, 255.0).astype(np.uint8)


base.TACTILE_HAND_ASSET = HAND1_ASSET
base._load_live_tactile_hand = _load_live_tactile_hand
base._render_live_tactile_frame = _render_live_tactile_frame


def main() -> None:
    base.main()


if __name__ == "__main__":
    main()
