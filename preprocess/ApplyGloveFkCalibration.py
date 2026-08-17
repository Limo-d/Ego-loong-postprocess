#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply calibrated R_cam_glove to glove FK 21-point sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def visual_root(row: Dict) -> Optional[np.ndarray]:
    hand = (row.get("visual_prior") or {}).get("hand") or {}
    kpts = hand.get("kpts_3d")
    if not kpts:
        return None
    arr = np.asarray(kpts, dtype=np.float64)
    if arr.shape[0] < 1 or arr.shape[1] < 3 or not np.isfinite(arr[0, :3]).all():
        return None
    return arr[0, :3]


def transform_points(mat4: List[List[float]], pts: np.ndarray) -> np.ndarray:
    t = np.asarray(mat4, dtype=np.float64)
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return (t @ homo.T).T[:, :3]


def apply(args: argparse.Namespace) -> Dict:
    rows = read_jsonl(Path(args.input_jsonl).expanduser().resolve())
    calib = read_json(Path(args.calib_json).expanduser().resolve())
    r = np.asarray(calib["R_cam_glove"], dtype=np.float64)
    r_prealign = np.asarray(calib.get("R_glove_prealign", np.eye(3).tolist()), dtype=np.float64)
    scale = float(calib.get("scale", 1.0))
    if r.shape != (3, 3):
        raise ValueError("R_cam_glove must be 3x3")
    if r_prealign.shape != (3, 3):
        raise ValueError("R_glove_prealign must be 3x3")

    out_rows = []
    stats = {
        "frames": 0,
        "applied": 0,
        "missing_visual_root": 0,
        "missing_glove_rel": 0,
        "world_aligned": 0,
    }
    root_z = []

    for row in rows:
        stats["frames"] += 1
        fk = row.get("glove_fk21") or {}
        root = visual_root(row)
        rel = fk.get("kpts_3d_wrist_relative_m")
        new_row = row
        if not fk.get("valid") or rel is None:
            stats["missing_glove_rel"] += 1
            out_rows.append(new_row)
            continue
        if root is None:
            stats["missing_visual_root"] += 1
            out_rows.append(new_row)
            continue

        rel_arr = np.asarray(rel, dtype=np.float64)
        if rel_arr.shape[0] < 21 or rel_arr.shape[1] < 3:
            stats["missing_glove_rel"] += 1
            out_rows.append(new_row)
            continue

        local_rel = rel_arr[:21, :3]
        local_rel = (r_prealign @ local_rel.T).T
        if args.flip_palm_normal:
            # Legacy fallback: proper 180 deg rotation around the local finger-forward axis.
            # Keep disabled when R_glove_prealign is estimated from visual constraints.
            local_rel = (np.diag([1.0, -1.0, -1.0]) @ local_rel.T).T
        cam = root[None, :] + scale * (r @ local_rel.T).T
        world = None
        c2w = (row.get("camera") or {}).get("c2w")
        if c2w:
            world = transform_points(c2w, cam)
            stats["world_aligned"] += 1

        new_fk = dict(fk)
        new_fk["kpts_3d_camera_m_uncalibrated"] = fk.get("kpts_3d_camera_m")
        new_fk["kpts_3d_world_m_uncalibrated"] = fk.get("kpts_3d_world_m")
        new_fk["kpts_3d_camera_m"] = cam.tolist()
        new_fk["kpts_3d_world_m"] = None if world is None else world.tolist()
        new_fk["calibration"] = {
            "calib_json": str(Path(args.calib_json).expanduser().resolve()),
            "R_glove_prealign": r_prealign.tolist(),
            "R_cam_glove": r.tolist(),
            "R_total_glove_to_visual": calib.get("R_total_glove_to_visual"),
            "prealign_quat_wxyz": calib.get("prealign_quat_wxyz"),
            "total_quat_wxyz": calib.get("total_quat_wxyz"),
            "scale": scale,
            "method": calib.get("method"),
            "frame_start": calib.get("frame_start"),
            "frame_end": calib.get("frame_end"),
            "indices": calib.get("indices"),
            "placement": "visual wrist/root translation plus calibrated fixed rotation",
            "flip_palm_normal": bool(args.flip_palm_normal),
        }

        new_row = dict(row)
        new_row["glove_fk21"] = new_fk
        out_rows.append(new_row)
        stats["applied"] += 1
        root_z.append(float(root[2]))

    summary = {
        "input_jsonl": str(Path(args.input_jsonl).expanduser().resolve()),
        "output_jsonl": str(Path(args.output_jsonl).expanduser().resolve()),
        "calib_json": str(Path(args.calib_json).expanduser().resolve()),
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
    parser = argparse.ArgumentParser(description="Apply calibrated R_cam_glove to glove FK sequence.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--calib_json", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument(
        "--flip_palm_normal",
        action="store_true",
        help="Apply a 180 degree local rotation around the finger-forward axis to flip palm/dorsal orientation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = apply(args)
    print(f"[ApplyGloveFkCalibration] output: {args.output_jsonl}")
    print(f"[ApplyGloveFkCalibration] summary: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
