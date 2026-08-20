#!/usr/bin/env python3
"""Level each hand's initial palm plane in the final dual-hand world trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PALM_INDICES = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)


def read_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


def valid_points(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    points = np.asarray(value, dtype=np.float64)
    if points.shape[0] < 21 or points.shape[1] < 3 or not np.isfinite(points[:21, :3]).all():
        return None
    return points[:21, :3]


def palm_normal(points: np.ndarray) -> Optional[np.ndarray]:
    palm = points[PALM_INDICES]
    centered = palm - palm.mean(axis=0, keepdims=True)
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if singular.shape[0] < 2 or singular[1] < 1e-8:
        return None
    normal = vt[-1]
    norm = float(np.linalg.norm(normal))
    return None if norm < 1e-9 else normal / norm


def rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if sine < 1e-9:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        helper = np.asarray([1.0, 0.0, 0.0]) if abs(source[0]) < 0.9 else np.asarray([0.0, 1.0, 0.0])
        axis = np.cross(source, helper)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    axis = cross / sine
    skew = np.asarray([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + skew * sine + (skew @ skew) * (1.0 - cosine)


def estimate_level_rotation(rows: List[Dict], side: str, frame_count: int) -> Dict:
    normals = []
    for row in rows[:frame_count]:
        glove = (((row.get("hands") or {}).get(side) or {}).get("glove") or {})
        points = valid_points(glove.get("kpts_3d_world_m"))
        normal = None if points is None else palm_normal(points)
        if normal is None:
            continue
        if normals and float(normal @ normals[0]) < 0.0:
            normal = -normal
        normals.append(normal)
    if not normals:
        raise RuntimeError(f"No valid {side} palm normals in first {frame_count} frames")
    mean = np.sum(normals, axis=0)
    mean /= np.linalg.norm(mean)
    target = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if float(mean @ target) < 0.0:
        target = -target
    deviations = np.degrees(np.arccos(np.clip(np.abs(np.asarray(normals) @ mean), 0.0, 1.0)))
    tilt_deg = float(np.degrees(np.arccos(np.clip(mean @ target, -1.0, 1.0))))
    return {
        "rotation": rotation_between(mean, target),
        "source_normal": mean,
        "target_normal": target,
        "samples": len(normals),
        "tilt_deg": tilt_deg,
        "normal_deviation_max_deg": float(deviations.max()),
    }


def align_initial_yaw(rows: List[Dict], levels: Dict, frame_count: int) -> None:
    """Align both initial wrist->middle-MCP directions in the world XY plane."""
    directions = {}
    for side, level in levels.items():
        samples = []
        for row in rows[:frame_count]:
            glove = (((row.get("hands") or {}).get(side) or {}).get("glove") or {})
            points = valid_points(glove.get("kpts_3d_world_m"))
            if points is None:
                continue
            direction = level["rotation"] @ (points[9] - points[0])
            direction = direction[:2]
            norm = float(np.linalg.norm(direction))
            if norm > 1e-9:
                samples.append(direction / norm)
        if not samples:
            raise RuntimeError(f"No valid {side} palm directions in first {frame_count} frames")
        mean = np.sum(samples, axis=0)
        mean /= np.linalg.norm(mean)
        directions[side] = mean

    target = np.sum(list(directions.values()), axis=0)
    if float(np.linalg.norm(target)) < 1e-9:
        target = directions["left"].copy()
    target /= np.linalg.norm(target)
    for side, level in levels.items():
        source = directions[side]
        cosine = float(np.clip(source @ target, -1.0, 1.0))
        sine = float(source[0] * target[1] - source[1] * target[0])
        yaw = float(np.arctan2(sine, cosine))
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        yaw_rotation = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        level["palm_level_rotation"] = level["rotation"]
        level["yaw_rotation"] = yaw_rotation
        level["rotation"] = yaw_rotation @ level["rotation"]
        level["source_direction_xy"] = source
        level["target_direction_xy"] = target
        level["yaw_correction_deg"] = float(np.degrees(yaw))


def palm_z_range_mm(points: np.ndarray) -> float:
    return float(np.ptp(points[PALM_INDICES, 2]) * 1000.0)


def process(args: argparse.Namespace) -> Dict:
    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()
    rows = read_jsonl(input_path)
    levels = {side: estimate_level_rotation(rows, side, args.level_frames) for side in ("left", "right")}
    align_initial_yaw(rows, levels, args.level_frames)
    stats = {side: {"corrected": 0, "missing": 0} for side in levels}
    first_ranges = {side: {} for side in levels}

    for row_index, row in enumerate(rows):
        hands = row.get("hands") or {}
        camera = np.asarray((row.get("camera") or {}).get("c2w"), dtype=np.float64)
        camera_valid = camera.shape == (4, 4) and np.isfinite(camera).all()
        for side, level in levels.items():
            hand = hands.get(side) or {}
            glove = hand.get("glove") or {}
            world = valid_points(glove.get("kpts_3d_world_m"))
            if world is None or not camera_valid:
                stats[side]["missing"] += 1
                continue
            root = world[0].copy()
            corrected_world = root[None, :] + (level["rotation"] @ (world - root[None, :]).T).T
            corrected_camera = (camera[:3, :3].T @ (corrected_world - camera[:3, 3][None, :]).T).T
            if row_index == 0:
                first_ranges[side] = {
                    "before_mm": palm_z_range_mm(world),
                    "after_mm": palm_z_range_mm(corrected_world),
                }
            glove.setdefault("kpts_3d_world_m_before_palm_level", glove.get("kpts_3d_world_m"))
            glove.setdefault("kpts_3d_camera_m_before_palm_level", glove.get("kpts_3d_camera_m"))
            glove["kpts_3d_world_m"] = corrected_world.tolist()
            glove["kpts_3d_camera_m"] = corrected_camera.tolist()
            glove["palm_plane_level"] = {
                "level_frames": int(args.level_frames),
                "rotation_world": level["rotation"].tolist(),
                "source_normal_world": level["source_normal"].tolist(),
                "target_normal_world": level["target_normal"].tolist(),
                "tilt_deg": level["tilt_deg"],
                "yaw_correction_deg": level["yaw_correction_deg"],
                "source_direction_xy": level["source_direction_xy"].tolist(),
                "target_direction_xy": level["target_direction_xy"].tolist(),
            }
            hand["glove"] = glove
            hands[side] = hand
            stats[side]["corrected"] += 1
        row["hands"] = hands
        legacy_side = str((row.get("glove") or {}).get("side") or "")
        if legacy_side in hands and (hands[legacy_side] or {}).get("glove"):
            row["glove"] = hands[legacy_side]["glove"]

    summary = {
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "level_frames": int(args.level_frames),
        "levels": {
            side: {
                "samples": level["samples"],
                "tilt_deg": level["tilt_deg"],
                "normal_deviation_max_deg": level["normal_deviation_max_deg"],
                "source_normal_world": level["source_normal"].tolist(),
                "target_normal_world": level["target_normal"].tolist(),
                "rotation_world": level["rotation"].tolist(),
                "yaw_correction_deg": level["yaw_correction_deg"],
                "source_direction_xy": level["source_direction_xy"].tolist(),
                "target_direction_xy": level["target_direction_xy"].tolist(),
                "first_frame_palm_z_range": first_ranges[side],
                "stats": stats[side],
            }
            for side, level in levels.items()
        },
    }
    write_jsonl(output_path, rows)
    if args.summary_json:
        write_json(Path(args.summary_json).expanduser().resolve(), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--level_frames", type=int, default=30)
    args = parser.parse_args()
    if args.level_frames <= 0:
        raise ValueError("--level_frames must be positive")
    summary = process(args)
    print(f"[LevelDualHandPalmPlane] summary: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
