#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose RTAB-Map depth coverage against OAK aligned hand depth samples."""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
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


def stats(values: List[float]) -> Dict:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def load_rtabmap_summary(db_path: Path) -> Dict:
    con = sqlite3.connect(str(db_path))
    try:
        node_rows = con.execute("select id, stamp from Node order by stamp").fetchall()
        data_summary = con.execute(
            "select count(*), sum(image is not null), sum(depth is not null), sum(calibration is not null), "
            "min(length(depth)), max(length(depth)) from Data"
        ).fetchone()
        depth_rows = con.execute(
            "select Node.id, Node.stamp, length(Data.image), length(Data.depth), length(Data.calibration) "
            "from Node join Data on Node.id=Data.id where Data.depth is not null order by Node.stamp"
        ).fetchall()
        feature_summary = con.execute(
            "select count(*), sum(depth_z is not null), min(depth_z), max(depth_z) from Feature"
        ).fetchone()
        per_node_features = con.execute(
            "select node_id, count(*), sum(depth_z is not null), min(depth_z), max(depth_z), avg(depth_z) "
            "from Feature group by node_id"
        ).fetchall()
    finally:
        con.close()
    per_node = {
        int(node_id): {
            "feature_count": int(count),
            "feature_depth_count": int(depth_count or 0),
            "feature_depth_z_min": None if zmin is None else float(zmin),
            "feature_depth_z_max": None if zmax is None else float(zmax),
            "feature_depth_z_mean": None if zmean is None else float(zmean),
        }
        for node_id, count, depth_count, zmin, zmax, zmean in per_node_features
    }
    return {
        "node_rows": [{"id": int(i), "stamp": float(s)} for i, s in node_rows],
        "data": {
            "rows": int(data_summary[0]),
            "image_nonnull": int(data_summary[1] or 0),
            "depth_nonnull": int(data_summary[2] or 0),
            "calibration_nonnull": int(data_summary[3] or 0),
            "depth_blob_len_min": None if data_summary[4] is None else int(data_summary[4]),
            "depth_blob_len_max": None if data_summary[5] is None else int(data_summary[5]),
        },
        "depth_nodes": [
            {
                "id": int(i),
                "stamp": float(stamp),
                "image_blob_len": None if ilen is None else int(ilen),
                "depth_blob_len": None if dlen is None else int(dlen),
                "calibration_blob_len": None if clen is None else int(clen),
            }
            for i, stamp, ilen, dlen, clen in depth_rows
        ],
        "features": {
            "rows": int(feature_summary[0]),
            "depth_z_nonnull": int(feature_summary[1] or 0),
            "depth_z_min": None if feature_summary[2] is None else float(feature_summary[2]),
            "depth_z_max": None if feature_summary[3] is None else float(feature_summary[3]),
        },
        "per_node_features": per_node,
    }


def nearest_frame(timestamps: List[Dict], stamps_ns: List[int], target_sec: float) -> Tuple[Optional[Dict], Optional[float]]:
    target_ns = int(round(target_sec * 1e9))
    pos = bisect.bisect_left(stamps_ns, target_ns)
    candidates = []
    if pos < len(timestamps):
        candidates.append(timestamps[pos])
    if pos > 0:
        candidates.append(timestamps[pos - 1])
    if not candidates:
        return None, None
    best = min(candidates, key=lambda r: abs(int(r["rgb_stamp_ns"]) - target_ns))
    return best, (int(best["rgb_stamp_ns"]) - target_ns) / 1e6


def hand_depth_values(hand_json: Dict) -> List[float]:
    values = []
    for key in ("hand_r", "hand_l"):
        hand = hand_json.get(key)
        if not hand:
            continue
        corr = hand.get("depth_root_correction") or {}
        samples = corr.get("samples") or []
        if samples:
            for sample in samples:
                z = sample.get("depth_m")
                if z is not None:
                    values.append(float(z))
        elif corr.get("depth_m") is not None:
            values.append(float(corr["depth_m"]))
    return values


def diagnose(args: argparse.Namespace) -> Dict:
    session = Path(args.session_path).expanduser().resolve()
    db_path = Path(args.rtabmap_db).expanduser().resolve()
    rtab = load_rtabmap_summary(db_path)
    timestamps = read_jsonl(session / "preprocess" / "timestamps.jsonl")
    stamps_ns = [int(r["rgb_stamp_ns"]) for r in timestamps]

    comparisons = []
    oak_depth_all = []
    feature_z_all = []
    for node in rtab["depth_nodes"]:
        frame_row, dt_ms = nearest_frame(timestamps, stamps_ns, node["stamp"])
        if frame_row is None:
            continue
        frame = str(frame_row["frame"])
        hand_json = read_json(session / "preprocess" / "all_data" / frame / args.visual_json_name)
        oak_values = hand_depth_values(hand_json)
        oak_depth_all.extend(oak_values)
        fstats = rtab["per_node_features"].get(int(node["id"]), {})
        if fstats.get("feature_depth_z_mean") is not None:
            feature_z_all.append(float(fstats["feature_depth_z_mean"]))
        comparisons.append({
            "rtab_node_id": node["id"],
            "rtab_stamp": node["stamp"],
            "nearest_frame": frame,
            "dt_ms": dt_ms,
            "rtab_depth_blob_len": node["depth_blob_len"],
            "oak_hand_depth_samples_m": oak_values,
            "oak_hand_depth_stats_m": stats(oak_values),
            "feature_depth": fstats,
        })

    summary = {
        "session_path": str(session),
        "rtabmap_db": str(db_path),
        "visual_json_name": args.visual_json_name,
        "rtabmap": {
            "nodes": len(rtab["node_rows"]),
            "data": rtab["data"],
            "features": rtab["features"],
            "depth_node_count": len(rtab["depth_nodes"]),
        },
        "comparison": {
            "matched_depth_nodes": len(comparisons),
            "oak_hand_depth_stats_m": stats(oak_depth_all),
            "rtab_feature_depth_z_mean_stats": stats(feature_z_all),
            "note": "Data.depth blobs are present only on sparse RTAB-Map nodes and are not decoded here; Feature.depth_z is sparse map/feature depth, not a per-pixel hand depth map.",
        },
        "depth_node_comparisons": comparisons,
    }
    if args.output_json:
        write_json(Path(args.output_json).expanduser().resolve(), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose RTAB-Map depth coverage versus OAK aligned hand depth samples.")
    parser.add_argument("--session_path", required=True)
    parser.add_argument("--rtabmap_db", required=True)
    parser.add_argument("--visual_json_name", required=True)
    parser.add_argument("--output_json", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = diagnose(args)
    print(f"[DiagnoseRtabmapDepth] summary: {json.dumps(summary['rtabmap'], ensure_ascii=False)}")
    if args.output_json:
        print(f"[DiagnoseRtabmapDepth] output: {args.output_json}")


if __name__ == "__main__":
    main()
