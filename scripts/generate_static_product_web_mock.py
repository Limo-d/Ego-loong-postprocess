#!/usr/bin/env python3
"""Generate one static product-style review web mockup image.

This script is intentionally independent from generate_review_web.py. It renders a
single PNG for presentation using the current session assets plus product photos.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION = ROOT / "postprocess_data" / "20260707T094101"
DEFAULT_HEADSET = Path("/tmp/codex-clipboard-a9lG8n.png")
DEFAULT_GLOVES = Path("/tmp/codex-clipboard-2ClJIp.png")
DEFAULT_RGB = Path("/home/lenovo/Downloads/rgb.jpg")
DEFAULT_OUTPUT = DEFAULT_SESSION / "outputs" / "static_product_web_preview.png"

FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
FONT_MONO = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS
ROTATE_CCW = getattr(getattr(Image, "Transpose", Image), "ROTATE_90")

PAGE_W = 1280
PAGE_H = 720


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    candidates = []
    if mono:
        candidates.append(FONT_MONO)
    candidates.append(FONT_BOLD if bold else FONT_REGULAR)
    candidates.append(FONT_REGULAR)
    candidates.append(Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except Exception:
                pass
    return ImageFont.load_default()


def read_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def fit_cover(img: Image.Image, size: tuple[int, int], anchor: tuple[float, float] = (0.5, 0.5)) -> Image.Image:
    src_w, src_h = img.size
    dst_w, dst_h = size
    scale = max(dst_w / src_w, dst_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    resized = img.resize((new_w, new_h), RESAMPLE)
    ax = max(0.0, min(1.0, anchor[0]))
    ay = max(0.0, min(1.0, anchor[1]))
    left = int(round((new_w - dst_w) * ax))
    top = int(round((new_h - dst_h) * ay))
    return resized.crop((left, top, left + dst_w, top + dst_h))


def fit_contain(img: Image.Image, size: tuple[int, int], bg: tuple[int, int, int]) -> Image.Image:
    src_w, src_h = img.size
    dst_w, dst_h = size
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = img.resize((new_w, new_h), RESAMPLE)
    out = Image.new("RGB", size, bg)
    out.paste(resized, ((dst_w - new_w) // 2, (dst_h - new_h) // 2))
    return out



def replace_edge_dark_background(img: Image.Image, target: tuple[int, int, int] = (248, 251, 254)) -> Image.Image:
    """Replace dark background connected to image edges with a light surface."""
    arr = np.asarray(img.convert("RGB")).copy()
    h, w = arr.shape[:2]
    rgb = arr.astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    bright = rgb.max(axis=2)
    dark_range = bright - rgb.min(axis=2)
    neutral_dark = (bright < 78) & (dark_range < 55)
    warm_dark = (bright < 92) & (r > g + 5) & (g >= b - 4) & (b < 58)
    very_dark = bright < 28
    candidate = neutral_dark | warm_dark | very_dark

    connected = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for x in range(w):
        if candidate[0, x]:
            connected[0, x] = True
            q.append((0, x))
        if candidate[h - 1, x]:
            connected[h - 1, x] = True
            q.append((h - 1, x))
    for y in range(h):
        if candidate[y, 0] and not connected[y, 0]:
            connected[y, 0] = True
            q.append((y, 0))
        if candidate[y, w - 1] and not connected[y, w - 1]:
            connected[y, w - 1] = True
            q.append((y, w - 1))
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and candidate[ny, nx] and not connected[ny, nx]:
                connected[ny, nx] = True
                q.append((ny, nx))

    arr[connected] = np.array(target, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def flatten_light_background(img: Image.Image, target: tuple[int, int, int]) -> Image.Image:
    """Blend a pale, low-saturation image background into the panel surface."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    low = arr.min(axis=2)
    chroma = arr.max(axis=2) - low
    brightness_weight = np.clip((low - 165.0) / 55.0, 0.0, 1.0)
    neutral_weight = np.clip((52.0 - chroma) / 28.0, 0.0, 1.0)
    weight = (brightness_weight * neutral_weight)[..., None]
    result = arr * (1.0 - weight) + np.asarray(target, dtype=np.float32) * weight
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), "RGB")


def rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255)
    return mask


def paste_round(base: Image.Image, img: Image.Image, box: tuple[int, int, int, int], radius: int) -> None:
    x0, y0, x1, y1 = box
    patch = img.resize((x1 - x0, y1 - y0), RESAMPLE).convert("RGBA")
    mask = rounded_mask((x1 - x0, y1 - y0), radius)
    base.paste(patch, (x0, y0), mask)


def shadow_rect(
    base: Image.Image,
    box: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, int, int, int],
    shadow: tuple[int, int, int, int] = (36, 54, 77, 38),
    blur: int = 18,
    offset: tuple[int, int] = (0, 10),
    outline: tuple[int, int, int, int] | None = None,
) -> None:
    x0, y0, x1, y1 = box
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    sx, sy = offset
    d.rounded_rectangle((x0 + sx, y0 + sy, x1 + sx, y1 + sy), radius=radius, fill=shadow)
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    base.alpha_composite(layer)
    d = ImageDraw.Draw(base)
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int, color, bold: bool = False, mono: bool = False) -> None:
    draw.text(xy, value, fill=color, font=font(size, bold=bold, mono=mono))


def draw_label(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str, accent: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill=(242, 247, 252, 238), outline=(214, 226, 238, 255))
    draw.ellipse((x0 + 12, y0 + 12, x0 + 22, y0 + 22), fill=accent)
    text(draw, (x0 + 30, y0 + 8), label, 17, (25, 39, 55, 255), bold=True)


def draw_panel_image(
    page: Image.Image,
    box: tuple[int, int, int, int],
    title: str,
    img: Image.Image,
    mode: str = "cover",
    anchor: tuple[float, float] = (0.5, 0.5),
    bg: tuple[int, int, int] = (14, 22, 31),
    accent: tuple[int, int, int] = (59, 180, 218),
    panel_bg: tuple[int, int, int, int] = (255, 255, 255, 244),
    full_bleed: bool = False,
) -> None:
    x0, y0, x1, y1 = box
    shadow_rect(page, box, radius=10, fill=panel_bg, shadow=(37, 54, 72, 24), blur=18, offset=(0, 8), outline=(213, 225, 238, 255))
    content_box = (x0 + 1, y0 + 1, x1 - 1, y1 - 1) if full_bleed else (x0 + 12, y0 + 46, x1 - 12, y1 - 12)
    cw, ch = content_box[2] - content_box[0], content_box[3] - content_box[1]
    framed = fit_cover(img, (cw, ch), anchor) if mode == "cover" else fit_contain(img, (cw, ch), bg)
    paste_round(page, framed, content_box, 8)
    d = ImageDraw.Draw(page)
    label_font = font(17, bold=True)
    label_w = max(104, min(x1 - x0 - 36, d.textbbox((0, 0), title, font=label_font)[2] + 46))
    label_x = x0 if full_bleed else x0 + 18
    label_y = y0 if full_bleed else y0 + 14
    draw_label(d, (label_x, label_y, label_x + label_w, label_y + 34), title, accent)


def draw_product_card(page: Image.Image, box: tuple[int, int, int, int], title: str, subtitle: str, img: Image.Image, anchor=(0.5, 0.5), mode: str = "cover") -> None:
    x0, y0, x1, y1 = box
    shadow_rect(page, box, radius=12, fill=(255, 255, 255, 244), shadow=(37, 54, 72, 28), blur=18, offset=(0, 8), outline=(213, 225, 238, 255))
    d = ImageDraw.Draw(page)
    text(d, (x0 + 16, y0 + 14), title, 18, (29, 43, 59, 255), bold=True)
    text(d, (x0 + 16, y0 + 40), subtitle, 12, (94, 112, 130, 255))
    image_box = (x0 + 14, y0 + 66, x1 - 14, y1 - 14)
    target_size = (image_box[2] - image_box[0], image_box[3] - image_box[1])
    fitted = fit_cover(img, target_size, anchor) if mode == "cover" else fit_contain(img, target_size, (248, 250, 252))
    # Blend the product photo into the light card with a very soft inner border.
    paste_round(page, fitted, image_box, 10)
    d.rounded_rectangle(image_box, radius=10, outline=(226, 234, 242, 220), width=1)


def load_trajectory_points(path: Path, max_points: int = 360) -> list[tuple[float, float, float]]:
    if not path.exists():
        return []
    points: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            glove = rec.get("glove") or {}
            val = glove.get("wristroot_3d_camera_m") or glove.get("wristroot_3d_world_m")
            if val is None:
                kpts = glove.get("kpts_3d_camera_m") or glove.get("kpts_3d_world_m") or []
                val = kpts[0] if kpts else None
            if isinstance(val, Sequence) and len(val) >= 3:
                try:
                    points.append((float(val[0]), float(val[1]), float(val[2])))
                except Exception:
                    pass
    if len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = [points[i] for i in idx]
    return points


def draw_mini_chart(page: Image.Image, box: tuple[int, int, int, int], points: Iterable[tuple[float, float, float]]) -> None:
    x0, y0, x1, y1 = box
    shadow_rect(page, box, radius=10, fill=(255, 255, 255, 238), shadow=(37, 54, 72, 24), blur=18, offset=(0, 8), outline=(213, 225, 238, 255))
    d = ImageDraw.Draw(page)
    text(d, (x0 + 16, y0 + 14), "轨迹信号", 18, (29, 43, 59, 255), bold=True)
    text(d, (x0 + 16, y0 + 40), "wristroot camera optical", 12, (94, 112, 130, 255), mono=True)
    chart = (x0 + 16, y0 + 76, x1 - 16, y1 - 58)
    d.rounded_rectangle(chart, radius=8, fill=(244, 248, 252, 255), outline=(222, 231, 240, 255))
    for i in range(1, 4):
        yy = chart[1] + i * (chart[3] - chart[1]) / 4
        d.line((chart[0] + 8, yy, chart[2] - 8, yy), fill=(224, 232, 240, 255), width=1)
    pts = list(points)
    if pts:
        arr = np.array(pts, dtype=np.float32)
        labels = ["x", "y", "z"]
        colors = [(31, 125, 185), (230, 142, 45), (43, 160, 98)]
        for dim, color in enumerate(colors):
            vals = arr[:, dim]
            lo, hi = float(np.min(vals)), float(np.max(vals))
            if hi - lo < 1e-6:
                hi = lo + 1.0
            coords = []
            for i, v in enumerate(vals):
                px = chart[0] + 12 + i * (chart[2] - chart[0] - 24) / max(1, len(vals) - 1)
                py = chart[3] - 12 - (float(v) - lo) * (chart[3] - chart[1] - 24) / (hi - lo)
                coords.append((px, py))
            if len(coords) >= 2:
                d.line(coords, fill=color + (255,), width=2, joint="curve")
            lx = x0 + 18 + dim * 58
            ly = y1 - 34
            d.rounded_rectangle((lx, ly, lx + 12, ly + 12), radius=3, fill=color + (255,))
            text(d, (lx + 18, ly - 4), labels[dim], 13, (60, 76, 92, 255), mono=True)
    else:
        text(d, (chart[0] + 22, chart[1] + 36), "trajectory data unavailable", 14, (120, 134, 150, 255), mono=True)


def draw_status_card(page: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    shadow_rect(page, box, radius=10, fill=(9, 22, 34, 255), shadow=(15, 32, 50, 42), blur=18, offset=(0, 8), outline=(28, 48, 65, 255))
    d = ImageDraw.Draw(page)
    text(d, (x0 + 18, y0 + 16), "采集概览", 18, (234, 244, 255, 255), bold=True)
    metrics = [("Frames", "581"), ("Tactile", "68 ch"), ("Pressure L", "active"), ("Pressure R", "zero")]
    y = y0 + 58
    for label, value in metrics:
        d.rounded_rectangle((x0 + 16, y, x1 - 16, y + 44), radius=8, fill=(15, 32, 48, 255), outline=(35, 58, 77, 255))
        text(d, (x0 + 30, y + 12), label, 13, (144, 164, 184, 255), mono=True)
        text(d, (x1 - 112, y + 9), value, 16, (237, 246, 255, 255), bold=True, mono=True)
        y += 54
    d.rounded_rectangle((x0 + 16, y + 4, x1 - 16, y + 96), radius=8, fill=(18, 39, 55, 255), outline=(38, 66, 84, 255))
    text(d, (x0 + 30, y + 22), "RGB image", 13, (144, 164, 184, 255), mono=True)
    text(d, (x0 + 30, y + 48), "rgb.jpg", 22, (255, 255, 255, 255), bold=True)
    text(d, (x0 + 30, y + 76), "product layout preview", 12, (102, 216, 235, 255))



def draw_collection_info_panel(page: Image.Image, box: tuple[int, int, int, int], points: Iterable[tuple[float, float, float]], frame_idx: int, total_frames: int = 581) -> None:
    x0, y0, x1, y1 = box
    shadow_rect(page, box, radius=12, fill=(250, 253, 255, 244), shadow=(37, 54, 72, 30), blur=18, offset=(0, 8), outline=(211, 225, 238, 255))
    d = ImageDraw.Draw(page)
    cx = (x0 + x1) // 2
    d.line((x0 + 20, y0 + 28, cx - 52, y0 + 28), fill=(220, 230, 240, 255), width=1)
    d.line((cx + 52, y0 + 28, x1 - 20, y0 + 28), fill=(220, 230, 240, 255), width=1)
    text(d, (cx - 42, y0 + 15), "采集信息", 20, (22, 36, 52, 255), bold=True)
    text(d, (x1 - 76, y0 + 17), f"{frame_idx}/{total_frames - 1}", 13, (88, 108, 132, 255), bold=True, mono=True)

    metrics = [("采集时长", "19.37", "s"), ("实时帧率", "30.0", "fps"), ("有效手帧", "581", "/581")]
    pad = 16
    gap = 10
    metric_y = y0 + 52
    metric_h = 74
    metric_w = (x1 - x0 - pad * 2 - gap * 2) // 3
    for i, (label, value, unit) in enumerate(metrics):
        mx = x0 + pad + i * (metric_w + gap)
        d.rounded_rectangle((mx, metric_y, mx + metric_w, metric_y + metric_h), radius=7, fill=(245, 249, 253, 255), outline=(221, 232, 242, 255))
        tw = d.textbbox((0, 0), label, font=font(13))[2]
        text(d, (mx + (metric_w - tw) // 2, metric_y + 14), label, 13, (88, 108, 132, 255))
        val_font = font(20, bold=True, mono=True)
        unit_font = font(12, mono=True)
        val_w = d.textbbox((0, 0), value, font=val_font)[2]
        unit_w = d.textbbox((0, 0), unit, font=unit_font)[2]
        start = mx + (metric_w - val_w - unit_w - 4) // 2
        d.text((start, metric_y + 38), value, fill=(37, 105, 224, 255), font=val_font)
        d.text((start + val_w + 4, metric_y + 45), unit, fill=(37, 105, 224, 255), font=unit_font)

    chart_outer = (x0 + 16, y0 + 142, x1 - 16, y1 - 88)
    d.rounded_rectangle(chart_outer, radius=7, fill=(248, 251, 254, 255), outline=(219, 231, 242, 255))
    legend_y = chart_outer[1] + 18
    colors = [(35, 119, 221), (239, 104, 45), (67, 159, 91)]
    labels = ["x", "y", "z"]
    for i, (label, color) in enumerate(zip(labels, colors)):
        lx = chart_outer[0] + 20 + i * 58
        d.line((lx, legend_y + 5, lx + 18, legend_y + 5), fill=color + (255,), width=4)
        text(d, (lx + 25, legend_y - 5), label, 13, (86, 104, 124, 255), mono=True)

    plot = (chart_outer[0] + 58, chart_outer[1] + 58, chart_outer[2] - 24, chart_outer[3] - 45)
    d.rectangle(plot, fill=(248, 251, 254, 255), outline=(225, 235, 244, 255))
    y_ticks = [0.54, 0.42, 0.30, 0.18, 0.06, -0.06]
    ymin, ymax = -0.06, 0.54
    for tick in y_ticks:
        py = plot[3] - (tick - ymin) * (plot[3] - plot[1]) / (ymax - ymin)
        d.line((plot[0], py, plot[2], py), fill=(226, 235, 244, 255), width=1)
        text(d, (chart_outer[0] + 18, int(py) - 8), f"{tick:.2f}", 10, (139, 155, 174, 255), mono=True)
    for i in range(5):
        px = plot[0] + i * (plot[2] - plot[0]) / 4
        d.line((px, plot[1], px, plot[3]), fill=(226, 235, 244, 255), width=1)
        text(d, (int(px) - 12, plot[3] + 14), ["0.0s", "4.8s", "9.7s", "14.5s", "19.4s"][i], 10, (116, 135, 156, 255), mono=True)

    pts = list(points)
    current_values = [0.0, 0.0, 0.0]
    if pts:
        arr = np.array(pts, dtype=np.float32)
        use_idx = min(len(arr) - 1, max(0, int(round(frame_idx / max(1, total_frames - 1) * (len(arr) - 1)))))
        current_values = [float(v) for v in arr[use_idx, :3]]
        for dim, color in enumerate(colors):
            vals = arr[:, dim]
            coords = []
            for i, v in enumerate(vals):
                px = plot[0] + i * (plot[2] - plot[0]) / max(1, len(vals) - 1)
                py = plot[3] - (float(v) - ymin) * (plot[3] - plot[1]) / (ymax - ymin)
                py = max(plot[1], min(plot[3], py))
                coords.append((px, py))
            if len(coords) >= 2:
                d.line(coords, fill=color + (255,), width=2, joint="curve")
        cur_x = plot[0] + frame_idx / max(1, total_frames - 1) * (plot[2] - plot[0])
        d.line((cur_x, plot[1], cur_x, plot[3]), fill=(34, 45, 59, 190), width=2)

    bottom_y = y1 - 70
    bottom_h = 52
    bottom_w = (x1 - x0 - pad * 2 - gap * 2) // 3
    for i, (label, value) in enumerate(zip(["wrist x", "wrist y", "wrist z"], current_values)):
        bx = x0 + pad + i * (bottom_w + gap)
        d.rounded_rectangle((bx, bottom_y, bx + bottom_w, bottom_y + bottom_h), radius=7, fill=(255, 255, 255, 255), outline=(221, 232, 242, 255))
        text(d, (bx + 12, bottom_y + 10), label, 11, (88, 108, 132, 255), bold=True, mono=True)
        text(d, (bx + 12, bottom_y + 28), f"{value:.3f} m", 14, (3, 20, 38, 255), bold=True, mono=True)

def gradient_background(size: tuple[int, int]) -> Image.Image:
    w, h = size
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    top = np.array([232, 240, 248], dtype=np.float32)
    bottom = np.array([217, 229, 241], dtype=np.float32)
    base = top * (1 - y[..., None]) + bottom * y[..., None]
    base = np.broadcast_to(base, (h, w, 3)).copy()
    glow1 = np.exp(-(((x - 0.16) / 0.25) ** 2 + ((y - 0.24) / 0.32) ** 2))
    glow2 = np.exp(-(((x - 0.78) / 0.32) ** 2 + ((y - 0.14) / 0.22) ** 2))
    base += glow1[..., None] * np.array([18, 34, 48], dtype=np.float32)
    base += glow2[..., None] * np.array([22, 36, 46], dtype=np.float32)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


def build(args: argparse.Namespace) -> Path:
    session = Path(args.session)
    outputs = session / "outputs"
    headset = read_image(Path(args.headset_image))
    gloves = read_image(Path(args.glove_image)).transpose(ROTATE_CCW).rotate(180)
    rgb = read_image(Path(args.rgb_image)).transpose(ROTATE_CCW)

    frame = int(args.frame_idx)
    total_frames = 581
    traj_frame_path = (
        Path(args.trajectory_image)
        if args.trajectory_image
        else outputs / "web" / "traj_frames" / f"{frame:05d}.jpg"
    )
    tactile_frame_path = (
        Path(args.tactile_image)
        if args.tactile_image
        else outputs / "web" / "tactile_frames" / f"{frame:05d}.jpg"
    )
    traj_img = read_image(traj_frame_path) if traj_frame_path.exists() else Image.new("RGB", (676, 328), (248, 251, 254))
    tactile_img = read_image(tactile_frame_path) if tactile_frame_path.exists() else Image.new("RGB", (676, 328), (248, 251, 254))
    traj_img = replace_edge_dark_background(traj_img)
    tactile_img = replace_edge_dark_background(tactile_img)
    tactile_img = flatten_light_background(tactile_img, (235, 243, 250))

    page = gradient_background((PAGE_W, PAGE_H))
    d = ImageDraw.Draw(page)

    # Header
    shadow_rect(page, (32, 22, PAGE_W - 32, 78), radius=14, fill=(255, 255, 255, 214), shadow=(45, 62, 80, 26), blur=16, offset=(0, 8), outline=(218, 229, 240, 255))
    text(d, (56, 39), "Ego-Loong 数据审核", 23, (26, 39, 54, 255), bold=True)
    # Left product rail
    shadow_rect(page, (34, 96, 286, 684), radius=16, fill=(246, 250, 254, 232), shadow=(31, 47, 65, 30), blur=20, offset=(0, 10), outline=(216, 228, 239, 255))
    text(d, (56, 118), "产品图", 24, (25, 39, 55, 255), bold=True)
    text(d, (56, 151), "当前硬件外观参考", 13, (92, 110, 128, 255))
    draw_product_card(page, (54, 184, 266, 364), "头部相机", "head camera", headset, anchor=(0.50, 0.55), mode="cover")
    draw_product_card(page, (54, 384, 266, 662), "触觉手套", "rotated RGB reference", gloves, anchor=(0.50, 0.50), mode="contain")

    # Main RGB and lower previews
    draw_panel_image(page, (306, 96, 850, 446), "RGB camera", rgb, mode="cover", anchor=(0.50, 0.50), bg=(248, 251, 254), accent=(83, 195, 226), full_bleed=True)
    preview_bg = (235, 243, 250, 255)
    draw_panel_image(page, (306, 466, 572, 684), "Trajectory", traj_img, mode="cover", anchor=(0.5, 0.5), bg=preview_bg[:3], accent=(246, 159, 71), panel_bg=preview_bg, full_bleed=True)
    draw_panel_image(page, (590, 466, 850, 684), "Tactile", tactile_img, mode="contain", anchor=(0.5, 0.5), bg=preview_bg[:3], accent=(72, 214, 181), panel_bg=preview_bg, full_bleed=True)

    # Right side collection info, matching the original review panel style.
    trajectory_path = outputs / "data" / "trajectory_wristroot_track_cameraoptical.jsonl"
    points = load_trajectory_points(trajectory_path, max_points=total_frames)
    draw_collection_info_panel(page, (872, 96, 1246, 684), points, frame, total_frames=total_frames)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    page.convert("RGB").save(out, quality=96)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default=str(DEFAULT_SESSION), help="postprocess session directory")
    parser.add_argument("--headset-image", default=str(DEFAULT_HEADSET), help="headset product image")
    parser.add_argument("--glove-image", default=str(DEFAULT_GLOVES), help="glove product image")
    parser.add_argument("--rgb-image", default=str(DEFAULT_RGB), help="RGB image to place in the main preview")
    parser.add_argument("--frame-idx", type=int, default=296, help="frame index for 3D/tactile preview images")
    parser.add_argument("--trajectory-image", help="explicit trajectory preview image")
    parser.add_argument("--tactile-image", help="explicit tactile preview image")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="output PNG path")
    return parser.parse_args()


def main() -> None:
    out = build(parse_args())
    print(out)


if __name__ == "__main__":
    main()
