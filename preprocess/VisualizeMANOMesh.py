#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render saved MANO camera-space vertices as an RGB overlay video."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm


def iter_frame_dirs(session_path: Path, max_frames: Optional[int]) -> List[Path]:
    all_data = session_path / "preprocess" / "all_data"
    if not all_data.is_dir():
        raise FileNotFoundError(f"Missing frame directory: {all_data}")
    frames = [p for p in sorted(all_data.iterdir()) if p.is_dir() and p.name.isdigit()]
    if frames:
        max_width = max(len(p.name) for p in frames)
        frames = [p for p in frames if len(p.name) == max_width]
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def project_vertices(vertices: np.ndarray, k: np.ndarray) -> np.ndarray:
    z = vertices[:, 2:3]
    uv = np.full((len(vertices), 2), np.nan, dtype=np.float32)
    valid = z[:, 0] > 1e-6
    uv[valid, 0] = vertices[valid, 0] / z[valid, 0] * k[0, 0] + k[0, 2]
    uv[valid, 1] = vertices[valid, 1] / z[valid, 0] * k[1, 1] + k[1, 2]
    return uv


def draw_mesh(
    image: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    k: np.ndarray,
    fill_color=(65, 190, 255),
    edge_color=(20, 80, 120),
    alpha: float = 0.42,
) -> np.ndarray:
    uv = project_vertices(vertices, k)
    h, w = image.shape[:2]
    overlay = image.copy()

    face_depth = vertices[faces, 2].mean(axis=1)
    for face_idx in np.argsort(face_depth)[::-1]:
        tri = faces[face_idx]
        pts = uv[tri]
        if not np.isfinite(pts).all():
            continue
        if (pts[:, 0].max() < 0 or pts[:, 0].min() >= w or pts[:, 1].max() < 0 or pts[:, 1].min() >= h):
            continue
        poly = np.round(pts).astype(np.int32)
        cv2.fillConvexPoly(overlay, poly, fill_color, lineType=cv2.LINE_AA)

    out = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)

    # Draw sparse edges so the mesh shape remains readable without over-darkening.
    for face_idx in range(0, len(faces), 2):
        tri = faces[face_idx]
        pts = uv[tri]
        if not np.isfinite(pts).all():
            continue
        poly = np.round(pts).astype(np.int32)
        cv2.polylines(out, [poly], True, edge_color, 1, cv2.LINE_AA)
    return out


def visualize_mano_session(
    session_path: str,
    json_name: str,
    out_path: str,
    faces: np.ndarray,
    side: str = "r",
    fps: float = 30.0,
    max_frames: Optional[int] = None,
    alpha: float = 0.42,
) -> Dict[str, int]:
    session = Path(session_path).expanduser().resolve()
    frames = iter_frame_dirs(session, max_frames)
    faces = np.asarray(faces, dtype=np.int32)
    side_key = f"hand_{side}"

    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    stats = {"frames": len(frames), "written": 0, "mesh": 0, "missing_json": 0, "missing_rgb": 0}

    for frame_dir in tqdm(frames, desc="Rendering MANO mesh"):
        rgb_path = frame_dir / "rgb.png"
        image = cv2.imread(str(rgb_path))
        if image is None:
            stats["missing_rgb"] += 1
            continue

        cam = load_json(frame_dir / "aria_cam_rgb.json")
        k = np.asarray(cam.get("k", np.eye(3)), dtype=np.float32)
        data = load_json(frame_dir / json_name)
        if not data:
            stats["missing_json"] += 1
        hand = data.get(side_key) if data else None
        vertices = None
        if hand and hand.get("mano_vertices_3d") is not None:
            vertices = np.asarray(hand["mano_vertices_3d"], dtype=np.float32)
            if vertices.ndim != 2 or vertices.shape[1] < 3:
                vertices = None

        if vertices is not None:
            image = draw_mesh(image, vertices[:, :3], faces, k, alpha=alpha)
            stats["mesh"] += 1

        cv2.putText(
            image,
            f"{json_name} | {side_key} | frame {frame_dir.name}",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"{json_name} | {side_key} | frame {frame_dir.name}",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if writer is None:
            h, w = image.shape[:2]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {out}")
        writer.write(image)
        stats["written"] += 1

    if writer is not None:
        writer.release()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Render saved MANO vertices as an RGB overlay video.")
    parser.add_argument("--session_path", required=True)
    parser.add_argument("--json_name", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--side", choices=["r", "l"], default="r")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.42)
    args = parser.parse_args()

    from preprocess.HaMeRHands import HaMeRModel

    model = HaMeRModel(device="cpu")
    if model.faces is None:
        raise RuntimeError("Could not load MANO faces from HaMeR")
    stats = visualize_mano_session(
        session_path=args.session_path,
        json_name=args.json_name,
        out_path=args.out_path,
        faces=model.faces,
        side=args.side,
        fps=args.fps,
        max_frames=args.max_frames,
        alpha=args.alpha,
    )
    print(f"[VisualizeMANOMesh] video: {args.out_path}")
    print(f"[VisualizeMANOMesh] stats: {stats}")


if __name__ == "__main__":
    main()
