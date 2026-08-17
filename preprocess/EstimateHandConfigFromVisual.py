#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimate Retarget hand_config geometry from visual 21-point 3D keypoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


FINGERS = ("thumb", "index", "middle", "ring", "little")
FINGER_IDXS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "little": [17, 18, 19, 20],
}


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write("\n")


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def row_idx(row: Dict) -> int:
    if row.get("idx") is not None:
        return int(row["idx"])
    return int(row["frame"])


def in_window(row: Dict, start: Optional[int], end: Optional[int]) -> bool:
    idx = row_idx(row)
    if start is not None and idx < start:
        return False
    if end is not None and idx > end:
        return False
    return True


def visual_kpts(row: Dict) -> Optional[np.ndarray]:
    visual_block = row.get("visual_prior") or row.get("visual") or {}
    hand = visual_block.get("hand") or {}
    pts = hand.get("kpts_3d")
    if pts is None:
        return None
    arr = np.asarray(pts, dtype=np.float64)
    if arr.shape[0] < 21 or arr.shape[1] < 3:
        return None
    arr = arr[:21, :3]
    if not np.isfinite(arr).all():
        return None
    return arr


def robust_median(values: List[float], min_v: float, max_v: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr >= min_v) & (arr <= max_v)]
    if arr.size == 0:
        raise RuntimeError("No valid values for robust median.")
    return float(np.median(arr))


def estimate(args: argparse.Namespace) -> Dict:
    rows = read_jsonl(Path(args.input_jsonl).expanduser().resolve())
    base_cfg = read_json(Path(args.base_config).expanduser().resolve())

    lengths: Dict[str, List[List[float]]] = {f: [[], [], []] for f in FINGERS}
    offsets: Dict[str, List[np.ndarray]] = {f: [] for f in FINGERS}
    used_frames = []

    for row in rows:
        if not in_window(row, args.frame_start, args.frame_end):
            continue
        pts = visual_kpts(row)
        if pts is None:
            continue
        wrist = pts[0]
        rel = pts - wrist[None, :]
        if not np.isfinite(rel).all():
            continue
        used_frames.append(row.get("frame"))

        for finger in FINGERS:
            ids = FINGER_IDXS[finger]
            chain = pts[ids, :]
            for j in range(3):
                lengths[finger][j].append(float(np.linalg.norm(chain[j + 1] - chain[j])))
            offsets[finger].append(rel[ids[0], :])

    if not used_frames:
        raise RuntimeError("No valid visual 3D frames in selected window.")

    bones_mm = {}
    offsets_mm = {}
    stats = {}
    for finger in FINGERS:
        bones_mm[finger] = [
            robust_median(lengths[finger][j], args.min_bone_m, args.max_bone_m) * 1000.0
            for j in range(3)
        ]
        off_arr = np.asarray(offsets[finger], dtype=np.float64)
        med = np.median(off_arr, axis=0) * 1000.0
        offsets_mm[finger] = [float(med[0]), float(med[1]), float(med[2])]
        stats[finger] = {
            "bones_mm": bones_mm[finger],
            "mcp_or_cmc_offset_mm": offsets_mm[finger],
            "samples": len(offsets[finger]),
        }

    out_cfg = dict(base_cfg)
    out_cfg["bones_mm"] = bones_mm
    if args.estimate_offsets:
        out_cfg["mcp_offsets_mm"] = offsets_mm
    out_cfg["hand"] = args.hand
    out_cfg["_visual_estimation"] = {
        "source_jsonl": str(Path(args.input_jsonl).expanduser().resolve()),
        "base_config": str(Path(args.base_config).expanduser().resolve()),
        "frame_start": args.frame_start,
        "frame_end": args.frame_end,
        "used_frame_count": len(used_frames),
        "used_frames": used_frames,
        "method": "Median visual 3D MANO/HaMeR per-finger bone lengths. MCP/CMC offsets are preserved from base_config unless --estimate_offsets is set.",
        "estimate_offsets": bool(args.estimate_offsets),
        "note": "Temporary visual geometry initialization; replace with measured geometry or stronger multi-pose calibration later.",
    }

    summary = {
        "output_config": str(Path(args.output_config).expanduser().resolve()),
        "used_frame_count": len(used_frames),
        "used_frames": used_frames,
        "stats": stats,
    }
    write_json(Path(args.output_config).expanduser().resolve(), out_cfg)
    if args.summary_json:
        write_json(Path(args.summary_json).expanduser().resolve(), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate Retarget hand_config geometry from visual 3D hand keypoints.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--base_config", default="/home/lenovo/Retarget/host/hand_config.json")
    parser.add_argument("--output_config", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--frame_start", type=int, default=None)
    parser.add_argument("--frame_end", type=int, default=None)
    parser.add_argument("--hand", choices=["left", "right"], default="left")
    parser.add_argument("--estimate_offsets", action="store_true", help="Also replace mcp_offsets_mm with visual wrist-relative offsets. Usually leave off unless a canonical-frame alignment has been handled.")
    parser.add_argument("--min_bone_m", type=float, default=0.005)
    parser.add_argument("--max_bone_m", type=float, default=0.12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = estimate(args)
    print(f"[EstimateHandConfigFromVisual] output: {args.output_config}")
    print(f"[EstimateHandConfigFromVisual] summary: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
