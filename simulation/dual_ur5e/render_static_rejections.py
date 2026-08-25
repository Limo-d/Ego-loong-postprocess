#!/usr/bin/env python3
"""Render source-RGB clips for every static-lock segment rejected by a run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--rgb_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--context_frames", type=int, default=15)
    args = parser.parse_args()

    summary_path = Path(args.summary).expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rgb_dir = Path(args.rgb_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = np.load(Path(summary["npz"]))
    source_times = trajectory["source_times_sec"]
    fps = 1.0 / float(np.median(np.diff(source_times)))

    outputs: list[dict[str, object]] = []
    for side in ("left", "right"):
        side_data = (summary.get("static_lock") or {}).get(side) or {}
        for rejected in side_data.get("rejected_segments") or []:
            rejected_start = int(rejected["start_frame"])
            rejected_end = int(rejected["end_frame"])
            clip_start = max(0, rejected_start - args.context_frames)
            clip_end = min(len(source_times) - 1, rejected_end + args.context_frames)
            name = f"rejected_{side}_{rejected_start:04d}_{rejected_end:04d}_rgb.mp4"
            output = output_dir / name
            writer: cv2.VideoWriter | None = None
            try:
                for frame_index in range(clip_start, clip_end + 1):
                    image_path = rgb_dir / f"{frame_index:05d}.jpg"
                    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise FileNotFoundError(image_path)
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(output),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (frame.shape[1], frame.shape[0]),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"cannot open MP4 writer: {output}")
                    active = rejected_start <= frame_index <= rejected_end
                    color = (30, 30, 230) if active else (160, 160, 160)
                    status = "REJECTED" if active else "CONTEXT"
                    cv2.rectangle(frame, (3, 3), (frame.shape[1] - 4, frame.shape[0] - 4), color, 6)
                    cv2.putText(
                        frame,
                        f"{status} {side.upper()}  frame {frame_index:04d}",
                        (22, 42),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"range {rejected_start}-{rejected_end}  max anchor deviation "
                        f"{float(rejected['max_anchor_deviation_m']) * 1000:.2f} mm",
                        (22, 78),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                    writer.write(frame)
            finally:
                if writer is not None:
                    writer.release()
            outputs.append(
                {
                    "side": side,
                    "rejected_start_frame": rejected_start,
                    "rejected_end_frame": rejected_end,
                    "clip_start_frame": clip_start,
                    "clip_end_frame": clip_end,
                    "output": str(output),
                }
            )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
