#!/usr/bin/env python3
"""Merge independently calibrated left/right trajectories by frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def valid_hand(row: Dict) -> bool:
    return bool((row.get("glove") or {}).get("kpts_3d_camera_m"))


def build(args: argparse.Namespace) -> Dict:
    raw_paths = {"left": args.left_jsonl, "right": args.right_jsonl}
    paths = {
        side: Path(value).expanduser().resolve() if value else None
        for side, value in raw_paths.items()
    }
    by_side = {
        side: {str(row.get("frame")): row for row in read_jsonl(path)} if path else {}
        for side, path in paths.items()
    }
    active_sides = [side for side in ("left", "right") if paths[side] is not None]
    frame_order = []
    seen = set()
    for side in ("left", "right"):
        for frame in by_side[side]:
            if frame not in seen:
                seen.add(frame)
                frame_order.append(frame)
    frame_order.sort(key=lambda value: (0, int(value)) if value.isdigit() else (1, value))

    out_rows = []
    counts = {"left_valid": 0, "right_valid": 0, "both_valid": 0}
    for frame in frame_order:
        left = by_side["left"].get(frame)
        right = by_side["right"].get(frame)
        base = json.loads(json.dumps(left or right or {"frame": frame}))
        hands = {}
        for side, source in (("left", left), ("right", right)):
            if source is None:
                hands[side] = None
                continue
            hands[side] = {
                "visual_prior": source.get("visual_prior"),
                "glove": source.get("glove"),
                "processing": source.get("processing"),
            }
        base["hands"] = hands
        # Preserve the established single-hand keys as a compatibility alias.
        # New consumers must read `hands.left/right`; old tools continue on the
        # left hand (or the right hand when left is unavailable).
        alias = left or right
        base["visual_prior"] = (alias or {}).get("visual_prior")
        base["glove"] = (alias or {}).get("glove")
        base["processing"] = {
            "mode": "dual_hand",
            "left": (left or {}).get("processing"),
            "right": (right or {}).get("processing"),
        }
        left_ok = left is not None and valid_hand(left)
        right_ok = right is not None and valid_hand(right)
        counts["left_valid"] += int(left_ok)
        counts["right_valid"] += int(right_ok)
        counts["both_valid"] += int(left_ok and right_ok)
        out_rows.append(base)

    output = Path(args.output_jsonl).expanduser().resolve()
    write_jsonl(output, out_rows)
    summary = {
        "mode": "dual_hand",
        "output_jsonl": str(output),
        "inputs": {side: str(path) if path else None for side, path in paths.items()},
        "active_sides": active_sides,
        "frames": len(out_rows),
        **counts,
    }
    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left_jsonl", default=None)
    parser.add_argument("--right_jsonl", default=None)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_json", default=None)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
