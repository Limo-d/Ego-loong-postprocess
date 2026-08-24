#!/usr/bin/env python3
"""Rebase a trajectory world frame to the first valid camera optical pose."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


def valid_transform(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        return None
    return matrix


def first_camera_pose(path: Path) -> tuple[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            matrix = valid_transform((row.get("camera") or {}).get("c2w"))
            if matrix is None:
                matrix = valid_transform((row.get("head_pose") or {}).get("c2w"))
            if matrix is not None:
                return str(row.get("frame", row.get("idx", "unknown"))), matrix
    raise RuntimeError(f"No valid camera.c2w or head_pose.c2w in {path}")


def transform_points(value: Any, rebase: np.ndarray) -> Any:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3 or not np.isfinite(points[:, :3]).all():
        return value
    transformed = points.copy()
    transformed[:, :3] = (rebase[:3, :3] @ points[:, :3].T).T + rebase[:3, 3]
    return transformed.tolist()


def transform_glove_world_fields(node: Any, rebase: np.ndarray) -> int:
    count = 0
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if key.startswith("kpts_3d_world_m") and value is not None:
                node[key] = transform_points(value, rebase)
                count += 1
            else:
                count += transform_glove_world_fields(value, rebase)
    elif isinstance(node, list):
        for value in node:
            count += transform_glove_world_fields(value, rebase)
    return count


def transform_pose(container: Dict[str, Any], key: str, rebase: np.ndarray) -> bool:
    matrix = valid_transform(container.get(key))
    if matrix is None:
        return False
    container[key] = (rebase @ matrix).tolist()
    return True


def transform_palm_metadata(node: Any, rotation: np.ndarray) -> int:
    count = 0
    if isinstance(node, dict):
        metadata = node.get("palm_plane_level")
        if isinstance(metadata, dict):
            level_rotation = np.asarray(metadata.get("rotation_world"), dtype=np.float64)
            if level_rotation.shape == (3, 3):
                metadata["rotation_world"] = (rotation @ level_rotation @ rotation.T).tolist()
                count += 1
            for key in ("source_normal_world", "target_normal_world"):
                vector = np.asarray(metadata.get(key), dtype=np.float64)
                if vector.shape == (3,):
                    metadata[key] = (rotation @ vector).tolist()
            metadata["coordinate_frame"] = "first_camera_optical"
        for value in node.values():
            count += transform_palm_metadata(value, rotation)
    elif isinstance(node, list):
        for value in node:
            count += transform_palm_metadata(value, rotation)
    return count


def process_row(row: Dict[str, Any], rebase: np.ndarray, reference_frame: str) -> Dict[str, int]:
    stats = {"camera_poses": 0, "head_poses": 0, "world_point_arrays": 0, "palm_metadata": 0}

    camera = row.get("camera") or {}
    for key in ("c2w", "c2w_before_camera_optical_fix"):
        stats["camera_poses"] += int(transform_pose(camera, key, rebase))
    if camera:
        camera["world_frame"] = "first_camera_optical"
        row["camera"] = camera

    head = row.get("head_pose") or {}
    for key in ("c2w", "c2w_before_camera_optical_fix"):
        stats["head_poses"] += int(transform_pose(head, key, rebase))
    odom = head.get("odom") or {}
    # odom.c2w is the composed map/world pose. odom_base_c2w and raw TF
    # records remain untouched as source diagnostics.
    stats["head_poses"] += int(transform_pose(odom, "c2w", rebase))
    if head:
        row["head_pose"] = head

    stats["world_point_arrays"] = transform_glove_world_fields(row, rebase)
    stats["palm_metadata"] = transform_palm_metadata(row, rebase[:3, :3])
    row["world_rebase"] = {
        "method": "left_multiply_inverse_first_camera_c2w",
        "world_frame": "first_camera_optical",
        "reference_frame": reference_frame,
    }
    return stats


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()
    reference_frame, first_c2w = first_camera_pose(input_path)
    rebase = np.linalg.inv(first_c2w)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    totals = {"frames": 0, "camera_poses": 0, "head_poses": 0, "world_point_arrays": 0, "palm_metadata": 0}
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in iter_rows(input_path):
                if row.get("world_rebase"):
                    raise RuntimeError("Input trajectory is already world-rebased")
                stats = process_row(row, rebase, reference_frame)
                totals["frames"] += 1
                for key, value in stats.items():
                    totals[key] += value
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary_name, output_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    summary = {
        "method": "left_multiply_inverse_first_camera_c2w",
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "reference_frame": reference_frame,
        "world_frame": "first_camera_optical",
        "original_first_camera_c2w": first_c2w.tolist(),
        "original_world_to_first_camera": rebase.tolist(),
        "stats": totals,
    }
    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", default=None)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
