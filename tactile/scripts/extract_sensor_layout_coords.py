#!/usr/bin/env python3
"""从 assets/sensor_layout.png 提取 68 路传感点像素坐标（一次性标定脚本）。"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.signal import correlate2d

ROOT = Path(__file__).resolve().parents[1]
LAYOUT_IMAGE = ROOT / "assets" / "sensor_layout.png"
OUT_NPY = ROOT / "assets" / "sensor_layout_coords.npy"
OUT_PREVIEW = ROOT / "assets" / "sensor_layout_coords_preview.png"

FINGER_LABELS = (
    "A0", "A1", "A2", "A3",
    "B0", "B1", "B2", "B3",
    "C0", "C1", "C2", "C3",
    "D0", "D1", "D2", "D3",
    "E0", "E1", "E2", "E3",
)
PALM_ROWS, PALM_COLS = 6, 8
PALM_ROI = (300, 440, 670, 740)

# (tip_roi, base_roi) — 每指远端 2 点 + 近端 2 点，与示意图 B–E 一致
_FINGER_TIP_BASE_ROIS: Tuple[
    Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]], ...
] = (
    # index B
    ((298, 108, 368, 178), (305, 228, 395, 302)),
    # middle C — base 仅 y>142，避免把 C1/C3 误吸到指尖行
    ((418, 72, 512, 118), (418, 142, 508, 218)),
    # ring D — base 限 x<610、y>212，避开 D1 远侧误检
    ((528, 108, 638, 138), (518, 212, 598, 248)),
    # little E — tip/base 分开 ROI，避免 E1 吸到掌缘外误检
    ((608, 168, 668, 218), (598, 268, 648, 302)),
)
# 拇指 ROI（排除掌缘角点 x<88 的误检）
_THUMB_ROI = (88, 340, 188, 478)

# 自动提取后的人工微调（像素坐标，与示意图黄字/红圈对齐）
_MANUAL_SENSOR_REFINEMENTS: dict[str, tuple[float, float]] = {
    "A0": (85.0, 391.0),
    "A1": (106.0, 373.0),
    "A2": (135.0, 455.0),
    "A3": (143.0, 456.0),
    "C0": (444.0, 96.0),
    "C1": (467.0, 97.0),
    "D2": (518.0, 231.0),
    "D3": (540.0, 235.0),
    "E0": (621.0, 206.0),
    "E1": (635.0, 206.0),
    "E2": (613.0, 284.0),
    "E3": (627.0, 284.0),
}


def _ring_response(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = img[..., 0].astype(float)
    g = img[..., 1].astype(float)
    b = img[..., 2].astype(float)
    score = np.clip(r - np.maximum(g, b), 0, 255)
    dark = (r + g + b) / 3 < 40

    size = 15
    cy, cx = size // 2, size // 2
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    ring = ((dist >= 3.5) & (dist <= 6.5)).astype(float)
    ring /= max(float(ring.sum()), 1.0)
    ann = correlate2d(score, ring, mode="same")
    return ann, score, dark


def _find_peaks_in_roi(
    ann: np.ndarray,
    dark: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    min_resp: float,
    min_dist: float,
    max_n: int,
) -> list[tuple[float, float, float]]:
    sub = ann[y0:y1, x0:x1].copy()
    sub[~dark[y0:y1, x0:x1]] = 0
    peaks: list[tuple[float, float, float]] = []
    for _ in range(max_n):
        masked = sub.copy()
        for px, py, _ in peaks:
            yy, xx = np.ogrid[: sub.shape[0], : sub.shape[1]]
            masked[(xx - px) ** 2 + (yy - py) ** 2 < min_dist**2] = 0
        flat_idx = int(np.argmax(masked))
        val = float(masked.flat[flat_idx])
        if val < min_resp:
            break
        py, px = divmod(flat_idx, sub.shape[1])
        peaks.append((float(px), float(py), val))
        yy, xx = np.ogrid[: sub.shape[0], : sub.shape[1]]
        sub[(xx - px) ** 2 + (yy - py) ** 2 < min_dist**2] = 0
    return [(x0 + px, y0 + py, v) for px, py, v in peaks]


def _pair_left_right(
    peaks: Sequence[tuple[float, float, float]],
    *,
    name: str,
    y_tol: float = 28.0,
) -> np.ndarray:
    """取 ROI 内最强且 Y 接近的一对，按 X 左→右。"""
    if len(peaks) < 2:
        raise RuntimeError(f"{name} 仅检测到 {len(peaks)} 个红圈，期望 2")
    ranked = sorted(peaks, key=lambda p: -p[2])
    best_pair: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    best_score = -1.0
    for i, p in enumerate(ranked):
        for q in ranked[i + 1 :]:
            if abs(p[1] - q[1]) > y_tol:
                continue
            score = p[2] + q[2]
            if score > best_score:
                best_score = score
                best_pair = (p, q)
    if best_pair is None:
        top2 = ranked[:2]
        best_pair = (top2[0], top2[1])
    arr = np.array(
        [(best_pair[0][0], best_pair[0][1]), (best_pair[1][0], best_pair[1][1])],
        dtype=np.float64,
    )
    return arr[np.argsort(arr[:, 0])]


def _extract_finger_row_major(
    ann: np.ndarray,
    dark: np.ndarray,
    tip_roi: Tuple[int, int, int, int],
    base_roi: Tuple[int, int, int, int],
    *,
    name: str,
) -> np.ndarray:
    """食指–小指：上排 B0/B1，下排 B2/B3（与示意图黄字一致）。"""
    tip = _find_peaks_in_roi(
        ann, dark, *tip_roi, min_resp=32, min_dist=14, max_n=6
    )
    base = _find_peaks_in_roi(
        ann, dark, *base_roi, min_resp=32, min_dist=14, max_n=6
    )
    top = _pair_left_right(tip, name=f"{name} tip")
    bot = _pair_left_right(base, name=f"{name} base")
    return np.vstack([top, bot])


def _extract_thumb(
    ann: np.ndarray,
    dark: np.ndarray,
) -> np.ndarray:
    """拇指 A0–A3：示意图上排 A1(左) A0(右)，下排 A3(左) A2(右)。"""
    peaks = _find_peaks_in_roi(
        ann,
        dark,
        *_THUMB_ROI,
        min_resp=40,
        min_dist=20,
        max_n=6,
    )
    if len(peaks) < 4:
        raise RuntimeError(f"拇指仅检测到 {len(peaks)} 个红圈，期望 4")
    best4 = sorted(peaks, key=lambda p: -p[2])[:4]
    top2 = sorted(best4, key=lambda p: p[1])[:2]
    bot2 = sorted(best4, key=lambda p: p[1])[2:4]
    top2 = sorted(top2, key=lambda p: p[0])
    bot2 = sorted(bot2, key=lambda p: p[0])
    a1 = np.array(top2[0][:2], dtype=np.float64)
    a0 = np.array(top2[1][:2], dtype=np.float64)
    a3 = np.array(bot2[0][:2], dtype=np.float64)
    a2 = np.array(bot2[1][:2], dtype=np.float64)
    return np.vstack([a0, a1, a2, a3])


def _snap_to_peak(
    ann: np.ndarray,
    dark: np.ndarray,
    x: float,
    y: float,
    *,
    radius: int = 16,
) -> tuple[float, float]:
    x0, y0 = max(0, int(x - radius)), max(0, int(y - radius))
    x1, y1 = min(ann.shape[1], int(x + radius)), min(ann.shape[0], int(y + radius))
    sub = ann[y0:y1, x0:x1].copy()
    sub[~dark[y0:y1, x0:x1]] = 0
    flat_idx = int(np.argmax(sub))
    py, px = divmod(flat_idx, sub.shape[1])
    return float(x0 + px), float(y0 + py)


def _extract_palm_grid(
    ann: np.ndarray,
    dark: np.ndarray,
    cands: list[tuple[float, float, float]],
) -> np.ndarray:
    """取响应最强的 48 个掌区红圈，按 Y 分 6 行、每行 X 排序得 F0–F47。"""
    if len(cands) < PALM_ROWS * PALM_COLS:
        raise RuntimeError(
            f"掌区候选点 {len(cands)} < {PALM_ROWS * PALM_COLS}，请检查 PALM_ROI 或阈值"
        )
    cands = sorted(cands, key=lambda p: -p[2])[: PALM_ROWS * PALM_COLS]
    pts = np.array([(p[0], p[1]) for p in cands], dtype=np.float64)
    pts = pts[np.argsort(pts[:, 1])]
    grid_rows: List[np.ndarray] = []
    for ri in range(PALM_ROWS):
        row = pts[ri * PALM_COLS : (ri + 1) * PALM_COLS]
        row = row[np.argsort(row[:, 0])]
        snapped = np.array(
            [_snap_to_peak(ann, dark, float(x), float(y)) for x, y in row],
            dtype=np.float64,
        )
        grid_rows.append(snapped)
    out = np.vstack(grid_rows)
    if out.shape != (PALM_ROWS * PALM_COLS, 2):
        raise RuntimeError(f"掌区点数 {out.shape[0]} != {PALM_ROWS * PALM_COLS}")
    return out


def _apply_manual_refinements(
    xy: np.ndarray,
    labels: Sequence[str],
    ann: np.ndarray,
    dark: np.ndarray,
) -> np.ndarray:
    out = xy.copy()
    for i, label in enumerate(labels):
        if label not in _MANUAL_SENSOR_REFINEMENTS:
            continue
        rx, ry = _MANUAL_SENSOR_REFINEMENTS[label]
        out[i] = (rx, ry)
    return out


def extract_sensor_layout_xy() -> np.ndarray:
    img = np.array(Image.open(LAYOUT_IMAGE).convert("RGB"))
    ann, _, dark = _ring_response(img)

    finger_parts: List[np.ndarray] = [ _extract_thumb(ann, dark) ]
    finger_names = ("index", "middle", "ring", "little")
    for name, (tip_roi, base_roi) in zip(finger_names, _FINGER_TIP_BASE_ROIS):
        finger_parts.append(
            _extract_finger_row_major(ann, dark, tip_roi, base_roi, name=name)
        )

    palm_cands = _find_peaks_in_roi(
        ann,
        dark,
        *PALM_ROI,
        min_resp=30,
        min_dist=18,
        max_n=80,
    )
    palm_xy = _extract_palm_grid(ann, dark, palm_cands)

    xy = np.vstack(finger_parts + [palm_xy])
    if xy.shape != (68, 2):
        raise RuntimeError(f"总点数 {xy.shape[0]} != 68")
    labels = list(FINGER_LABELS) + [f"F{i}" for i in range(PALM_ROWS * PALM_COLS)]
    return _apply_manual_refinements(xy, labels, ann, dark)


def main() -> None:
    xy = extract_sensor_layout_xy()
    np.save(OUT_NPY, xy)

    labels = list(FINGER_LABELS) + [f"F{i}" for i in range(PALM_ROWS * PALM_COLS)]
    img = np.array(Image.open(LAYOUT_IMAGE).convert("RGB"))
    vis = Image.fromarray(img)
    draw = ImageDraw.Draw(vis)
    for (x, y), label in zip(xy, labels):
        xi, yi = int(round(x)), int(round(y))
        draw.ellipse((xi - 4, yi - 4, xi + 4, yi + 4), outline="lime", width=2)
        draw.text((xi + 5, yi - 5), label, fill="lime")
    vis.save(OUT_PREVIEW)
    print(f"saved {OUT_NPY} shape={xy.shape}")
    print(f"preview {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
