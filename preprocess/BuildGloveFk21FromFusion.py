#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a 21-point glove FK skeleton from fusion_frames.jsonl.

This is intentionally a bridge, not the final optimizer:
Retarget provides state27 -> canonical hand FK. We preserve that canonical
output and also create an initial camera/world placement using the visual wrist
as pose prior.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


RETARGET_ROOT_DEFAULT = "/home/lenovo/Retarget/retarget"

KPT21_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "little_mcp", "little_pip", "little_dip", "little_tip",
]

FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def finite_state(values: Optional[List[float]]) -> bool:
    if not values or len(values) != 27:
        return False
    arr = np.asarray(values, dtype=np.float64)
    return bool(np.isfinite(arr).all() and np.nanmax(np.abs(arr)) < 1e8)


def transform_points(mat4: List[List[float]], pts: np.ndarray) -> np.ndarray:
    T = np.asarray(mat4, dtype=np.float64)
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return (T @ homo.T).T[:, :3]


def quat_to_matrix_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def solve_state_struct(values: List[float]) -> Dict:
    fingers = {}
    labels = ("mcp_flex_deg", "mcp_abd_deg", "pip_flex_deg", "dip_flex_deg")
    for i, name in enumerate(("index", "middle", "ring", "little")):
        seg = values[4 * i:4 * i + 4]
        fingers[name] = {k: float(v) for k, v in zip(labels, seg)}
    return {
        "raw": [float(v) for v in values],
        "fingers_deg": fingers,
        "thumb_deg": {
            "mcp_flex_deg": float(values[16]),
            "mcp_abd_deg": float(values[17]),
            "ip_flex_deg": float(values[18]),
        },
        "thumb_cmc_quat_wxyz": [float(v) for v in values[19:23]],
        "palm_quat_wxyz": [float(v) for v in values[23:27]],
    }


def human_to_kpts21(human) -> np.ndarray:
    pts = [np.asarray(human.wrist, dtype=np.float64)]
    for name in FINGER_ORDER:
        pts.extend(np.asarray(human.points[name], dtype=np.float64))
    return np.asarray(pts, dtype=np.float64)


def load_retarget_fk(retarget_root: Path):
    """Load only Retarget's pure FK modules without importing mujoco retargeter."""
    pkg_name = "_retarget_fk_only"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(retarget_root / "hand_retarget")]  # type: ignore[attr-defined]
    sys.modules[pkg_name] = pkg

    loaded = {}
    for mod in ("config", "layout", "human_fk"):
        name = f"{pkg_name}.{mod}"
        path = retarget_root / "hand_retarget" / f"{mod}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load Retarget module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded[mod] = module
    return loaded["config"].HandConfig, loaded["layout"].parse_state27, loaded["human_fk"].reconstruct


def visual_root_and_rot(row: Dict, alignment: str) -> Tuple[Optional[np.ndarray], np.ndarray, Dict]:
    hand = (row.get("visual") or {}).get("hand") or {}
    kpts_3d = hand.get("kpts_3d") or []
    root = None
    if len(kpts_3d) >= 1:
        root = np.asarray(kpts_3d[0], dtype=np.float64)

    R = np.eye(3, dtype=np.float64)
    source = "identity"
    if alignment == "visual_wrist_rotation":
        wrist_pose = hand.get("wrist_pose")
        if wrist_pose:
            W = np.asarray(wrist_pose, dtype=np.float64)
            if W.shape == (4, 4) and np.isfinite(W[:3, :3]).all():
                R = W[:3, :3]
                source = "visual_wrist_pose_rotation"

    return root, R, {"translation": "visual_kpts_3d[0]", "rotation": source}


def build(args: argparse.Namespace) -> Dict:
    retarget_root = Path(args.retarget_root).expanduser().resolve()
    HandConfig, parse_state27, reconstruct = load_retarget_fk(retarget_root)

    cfg = HandConfig.load(args.hand_config)
    rows = read_jsonl(Path(args.input_jsonl).expanduser().resolve())
    out_rows = []
    stats = {
        "frames": 0,
        "glove_fk_valid": 0,
        "missing_glove_state": 0,
        "visual_root_valid": 0,
        "camera_aligned": 0,
        "world_aligned": 0,
    }
    root_z = []

    solve_key = f"solve_state_{args.glove_side}"

    for row in rows:
        stats["frames"] += 1
        hand_frame = row.get("hand_frame") or {}
        solve_state = hand_frame.get(solve_key)
        visual_root, R_cam, alignment_info = visual_root_and_rot(row, args.alignment)

        glove_fk = None
        if finite_state(solve_state):
            state = parse_state27(solve_state)
            human = reconstruct(state, cfg, mirror_to_right=args.mirror_to_right)
            kpts_model = human_to_kpts21(human)
            if args.apply_palm_quat and state.palm_valid:
                wrist0 = kpts_model[0].copy()
                r_palm = quat_to_matrix_wxyz(state.palm_quat)
                kpts_model = wrist0[None, :] + (r_palm @ (kpts_model - wrist0[None, :]).T).T
            kpts_rel = kpts_model - kpts_model[0]
            stats["glove_fk_valid"] += 1

            kpts_camera = None
            kpts_world = None
            if visual_root is not None and np.isfinite(visual_root).all():
                stats["visual_root_valid"] += 1
                root_z.append(float(visual_root[2]))
                kpts_camera = visual_root[None, :] + (R_cam @ kpts_rel.T).T
                stats["camera_aligned"] += 1
                c2w = (row.get("camera") or {}).get("c2w")
                if c2w:
                    kpts_world = transform_points(c2w, kpts_camera)
                    stats["world_aligned"] += 1

            glove_fk = {
                "valid": True,
                "source": {
                    "retarget_root": str(retarget_root),
                    "hand_config": str(Path(args.hand_config).expanduser().resolve()),
                    "glove_side": args.glove_side,
                    "mirror_to_right": bool(args.mirror_to_right),
                    "alignment": args.alignment,
                    "alignment_info": alignment_info,
                    "apply_palm_quat": bool(args.apply_palm_quat),
                    "note": "Canonical FK is from Retarget human_fk; optional palm_quat applies whole-hand IMU orientation around wrist before camera/world placement.",
                },
                "kpt21_names": KPT21_NAMES,
                "kpts_3d_model_m": kpts_model.tolist(),
                "kpts_3d_wrist_relative_m": kpts_rel.tolist(),
                "kpts_3d_camera_m": None if kpts_camera is None else kpts_camera.tolist(),
                "kpts_3d_world_m": None if kpts_world is None else kpts_world.tolist(),
                "finger_valid": {k: bool(v) for k, v in human.valid.items()},
                "pinch_distances_m": human.pinch_distances(),
                "solve_state": solve_state_struct(solve_state),
            }
        else:
            stats["missing_glove_state"] += 1
            glove_fk = {
                "valid": False,
                "source": {"glove_side": args.glove_side},
                "reason": f"missing_or_invalid_{solve_key}",
            }

        out_rows.append({
            "frame": row.get("frame"),
            "idx": row.get("idx"),
            "rgb_stamp_ns": row.get("rgb_stamp_ns"),
            "rgb_path": row.get("rgb_path"),
            "depth_aligned_path": row.get("depth_aligned_path"),
            "camera": row.get("camera"),
            "odom": row.get("odom"),
            "visual_prior": row.get("visual"),
            "hand_frame_sync": row.get("hand_frame_sync"),
            "fusion_mapping": row.get("fusion_mapping"),
            "glove_fk21": glove_fk,
        })

    summary = {
        "input_jsonl": str(Path(args.input_jsonl).expanduser().resolve()),
        "output_jsonl": str(Path(args.output_jsonl).expanduser().resolve()),
        "summary_json": None if not args.summary_json else str(Path(args.summary_json).expanduser().resolve()),
        "retarget_root": str(retarget_root),
        "hand_config": str(Path(args.hand_config).expanduser().resolve()),
        "glove_side": args.glove_side,
        "alignment": args.alignment,
        "apply_palm_quat": bool(args.apply_palm_quat),
        "kpt21_names": KPT21_NAMES,
        "stats": stats,
    }
    if root_z:
        arr = np.asarray(root_z, dtype=np.float64)
        summary["visual_root_camera_z_m"] = {
            "min": float(arr.min()),
            "median": float(np.median(arr)),
            "max": float(arr.max()),
        }

    write_jsonl(Path(args.output_jsonl).expanduser().resolve(), out_rows)
    if args.summary_json:
        write_json(Path(args.summary_json).expanduser().resolve(), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build 21-point glove FK skeletons from fusion_frames.jsonl.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--retarget_root", default=RETARGET_ROOT_DEFAULT)
    parser.add_argument("--hand_config", default="/home/lenovo/Retarget/host/hand_config.json")
    parser.add_argument("--glove_side", choices=["left", "right"], default="left")
    parser.add_argument("--alignment", choices=["translation_only", "visual_wrist_rotation"], default="visual_wrist_rotation")
    parser.add_argument("--apply_palm_quat", action="store_true", help="Apply state27 palm_quat whole-hand IMU orientation around the wrist before camera calibration.")
    parser.add_argument("--no_mirror_to_right", dest="mirror_to_right", action="store_false")
    parser.set_defaults(mirror_to_right=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build(args)
    print(f"[BuildGloveFk21FromFusion] output: {args.output_jsonl}")
    print(f"[BuildGloveFk21FromFusion] summary: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
