#!/usr/bin/env python3
"""Publish the globally optimized glove/FK wrist and palm action frames."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
from scipy.spatial.transform import Rotation


PALM_INDICES = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)
CONVENTION = {
    "name": "ego_loong_glove_fk_palm_v5",
    "handedness": "right_handed_so3",
    "origin_wrist": "optimized_trajectory_wrist_root",
    "origin_palm": "mean_of_optimized_wrist_and_four_mcp",
    "x_axis": "wrist_to_mean_mcp_from_glove_fk",
    "z_axis": "dorsal_palm_normal_from_glove_fk",
    "y_axis": "z_cross_x",
    "rotation_6d": "first_two_rotation_matrix_columns_column_major",
    "camera_frame": "head_camera_optical_x_right_y_down_z_forward",
    "geometry_source": "globally_optimized_glove_fk_trajectory",
    "orientation_filter": "inherited_from_optimized_trajectory",
    "wrist_translation": "inherited_from_optimized_trajectory",
    "hamer_orientation_dependency": False,
    "fk_dependency": True,
}


def valid_matrix(value: Any, shape: tuple[int, ...]) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        return None
    return array


def project_so3(matrix: np.ndarray) -> np.ndarray:
    u, _s, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    quat = Rotation.from_matrix(matrix).as_quat()
    return np.asarray([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


def angular_distance_deg(left: Optional[np.ndarray], right: Optional[np.ndarray]) -> Optional[float]:
    if left is None or right is None:
        return None
    relative = left.T @ right
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def summarize(values: list[float]) -> Dict[str, Optional[float]]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(array.size), "mean": float(np.mean(array)),
        "median": float(np.median(array)), "p95": float(np.percentile(array, 95.0)),
        "p99": float(np.percentile(array, 99.0)), "max": float(np.max(array)),
    }


def pose_payload(translation: np.ndarray, rotation: np.ndarray) -> Dict[str, Any]:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return {
        "translation_m": translation.tolist(),
        "rotation_matrix": rotation.tolist(),
        "quaternion_wxyz": matrix_to_quat_wxyz(rotation).tolist(),
        "rotation_6d": np.concatenate([rotation[:, 0], rotation[:, 1]]).tolist(),
        "transform": transform.tolist(),
    }


def iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    states = {
        side: {"rotation_world": None, "translation_world": None}
        for side in ("left", "right")
    }
    stats = {
        side: {
            "frames": 0, "valid": 0, "invalid": 0, "rotation_steps_deg": [],
            "translation_steps_m": [], "determinant_errors": [],
            "orthogonality_errors": [], "camera_world_rotation_consistency_deg": [],
        }
        for side in ("left", "right")
    }

    fd, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            for row in iter_rows(input_path):
                hands = row.get("hands") or {}
                camera_c2w = valid_matrix((row.get("camera") or {}).get("c2w"), (4, 4))
                for side in ("left", "right"):
                    hand = hands.get(side)
                    if not isinstance(hand, dict):
                        continue
                    if hand.get("palm_frame") is not None and not args.replace_existing:
                        raise RuntimeError("Input trajectory already contains palm_frame")
                    hand.pop("palm_frame", None)
                    side_stats = stats[side]
                    side_stats["frames"] += 1
                    optimized = hand.get("optimized_trajectory") or {}
                    wrist_camera = valid_matrix(optimized.get("wrist_translation_camera_m"), (3,))
                    wrist_world = valid_matrix(optimized.get("wrist_translation_world_m"), (3,))
                    rotation_camera = valid_matrix(optimized.get("palm_rotation_camera"), (3, 3))
                    rotation_world = valid_matrix(optimized.get("palm_rotation_world"), (3, 3))
                    points_camera = valid_matrix(optimized.get("kpts_3d_camera_m_optimized"), (21, 3))
                    points_world = valid_matrix(optimized.get("kpts_3d_world_m_optimized"), (21, 3))
                    required = (wrist_camera, wrist_world, rotation_camera, rotation_world, points_camera, points_world)
                    if camera_c2w is None or any(value is None for value in required):
                        side_stats["invalid"] += 1
                        hand["palm_frame"] = {
                            "convention": CONVENTION, "observed_valid": False,
                            "reason": "missing_optimized_glove_fk_pose",
                            "source": "hands.<side>.optimized_trajectory",
                        }
                        hands[side] = hand
                        continue

                    rotation_camera = project_so3(rotation_camera)
                    rotation_world = project_so3(rotation_world)
                    palm_camera = np.mean(points_camera[PALM_INDICES], axis=0)
                    palm_world = np.mean(points_world[PALM_INDICES], axis=0)
                    state = states[side]
                    rotation_step = angular_distance_deg(state["rotation_world"], rotation_world)
                    translation_step = (
                        float(np.linalg.norm(wrist_world - state["translation_world"]))
                        if state["translation_world"] is not None else None
                    )
                    expected_world = project_so3(camera_c2w[:3, :3] @ rotation_camera)
                    consistency = angular_distance_deg(expected_world, rotation_world)
                    determinant_error = abs(float(np.linalg.det(rotation_world)) - 1.0)
                    orthogonality_error = float(np.linalg.norm(rotation_world.T @ rotation_world - np.eye(3), ord="fro"))
                    if rotation_step is not None:
                        side_stats["rotation_steps_deg"].append(rotation_step)
                    if translation_step is not None:
                        side_stats["translation_steps_m"].append(translation_step)
                    if consistency is not None:
                        side_stats["camera_world_rotation_consistency_deg"].append(consistency)
                    side_stats["determinant_errors"].append(determinant_error)
                    side_stats["orthogonality_errors"].append(orthogonality_error)
                    side_stats["valid"] += 1

                    hand["palm_frame"] = {
                        "convention": CONVENTION,
                        "observed_valid": bool(optimized.get("observed_valid", True)),
                        "source": "hands.<side>.optimized_trajectory",
                        "wrist_translation_source": "optimized_trajectory",
                        "orientation_source": "optimized_trajectory_glove_fk",
                        "hamer_orientation_used": False,
                        "rotation_step_world_deg": rotation_step,
                        "translation_step_world_m": translation_step,
                        "wrist_pose_camera": pose_payload(wrist_camera, rotation_camera),
                        "palm_pose_camera": pose_payload(palm_camera, rotation_camera),
                        "wrist_pose_world": pose_payload(wrist_world, rotation_world),
                        "palm_pose_world": pose_payload(palm_world, rotation_world),
                        "quality": {
                            "determinant": float(np.linalg.det(rotation_world)),
                            "orthogonality_error": orthogonality_error,
                            "camera_world_rotation_consistency_deg": consistency,
                        },
                    }
                    state["rotation_world"] = rotation_world
                    state["translation_world"] = wrist_world
                    hands[side] = hand
                row["hands"] = hands
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary_name, output_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    summary_sides = {}
    for side, value in stats.items():
        summary_sides[side] = {
            "frames": value["frames"], "observed_valid": value["valid"],
            "invalid": value["invalid"],
            "valid_ratio": float(value["valid"]) / value["frames"] if value["frames"] else None,
            "wrist_translation_step_m": summarize(value["translation_steps_m"]),
            "palm_rotation_step_deg": summarize(value["rotation_steps_deg"]),
            "rotation_determinant_abs_error": summarize(value["determinant_errors"]),
            "rotation_orthogonality_fro_error": summarize(value["orthogonality_errors"]),
            "camera_world_rotation_consistency_deg": summarize(value["camera_world_rotation_consistency_deg"]),
            "optimized_wrist_translation_frames": value["valid"],
            "visual_wrist_translation_fallback_frames": 0,
            "hamer_orientation_frames": 0,
        }
    summary = {
        "input_jsonl": str(input_path), "output_jsonl": str(output_path),
        "convention": CONVENTION,
        "params": {
            "wrist_translation_source": "optimized_trajectory",
            "orientation_source": "optimized_trajectory_glove_fk",
            "hamer_orientation_used": False, "fk_dependency": True,
            "legacy_visual_smooth_inputs_ignored": True,
        },
        "sides": summary_sides,
    }
    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--left_visual_2d_smooth_jsonl", default=None)
    parser.add_argument("--right_visual_2d_smooth_jsonl", default=None)
    parser.add_argument("--replace_existing", action="store_true")
    parser.add_argument("--orientation_alpha", type=float, default=0.25)
    parser.add_argument("--max_orientation_step_deg", type=float, default=10.0)
    parser.add_argument("--reference_fps", type=float, default=30.0)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
