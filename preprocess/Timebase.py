#!/usr/bin/env python3
"""Shared RGB timestamp helpers for time-aware filtering and playback."""

from __future__ import annotations

from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_REFERENCE_FPS = 30.0


def row_stamp_ns(row: Dict[str, Any]) -> Optional[int]:
    candidates = [
        row.get("rgb_stamp_ns"),
        (row.get("timestamp") or {}).get("rgb_stamp_ns"),
        (row.get("camera") or {}).get("rgb_stamp_ns"),
        ((row.get("camera") or {}).get("sync") or {}).get("rgb_stamp_ns"),
    ]
    for value in candidates:
        try:
            if value is not None and int(value) > 0:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def positive_deltas_sec(rows: Sequence[Dict[str, Any]]) -> List[float]:
    stamps = [row_stamp_ns(row) for row in rows]
    return [
        (right - left) / 1e9
        for left, right in zip(stamps[:-1], stamps[1:])
        if left is not None and right is not None and right > left
    ]


def nominal_fps(rows: Sequence[Dict[str, Any]], fallback: float = DEFAULT_REFERENCE_FPS) -> float:
    deltas = positive_deltas_sec(rows)
    return float(1.0 / median(deltas)) if deltas else float(fallback)


def relative_times_sec(
    rows: Sequence[Dict[str, Any]],
    fallback_fps: float = DEFAULT_REFERENCE_FPS,
) -> List[float]:
    if not rows:
        return []
    fallback_dt = 1.0 / max(float(fallback_fps), 1e-9)
    stamps = [row_stamp_ns(row) for row in rows]
    first = next((stamp for stamp in stamps if stamp is not None), None)
    out: List[float] = []
    for i, stamp in enumerate(stamps):
        if stamp is not None and first is not None:
            value = (stamp - first) / 1e9
        else:
            value = out[-1] + fallback_dt if out else 0.0
        if out and value <= out[-1]:
            value = out[-1] + fallback_dt
        out.append(float(max(0.0, value)))
    return out


def effective_alpha(alpha_at_reference_fps: float, dt_sec: float, reference_fps: float) -> float:
    alpha = min(1.0, max(0.0, float(alpha_at_reference_fps)))
    if alpha in (0.0, 1.0):
        return alpha
    steps = max(0.0, float(dt_sec)) * max(float(reference_fps), 1e-9)
    return float(1.0 - (1.0 - alpha) ** steps)


def dt_sec(
    previous_stamp_ns: Optional[int],
    current_stamp_ns: Optional[int],
    reference_fps: float = DEFAULT_REFERENCE_FPS,
) -> float:
    fallback = 1.0 / max(float(reference_fps), 1e-9)
    if previous_stamp_ns is None or current_stamp_ns is None or current_stamp_ns <= previous_stamp_ns:
        return fallback
    return float((current_stamp_ns - previous_stamp_ns) / 1e9)


def repeat_counts(rows: Sequence[Dict[str, Any]], output_fps: Optional[float] = None) -> List[int]:
    """CFR repeat counts that preserve timestamp intervals, including capture pauses."""
    if not rows:
        return []
    fps = nominal_fps(rows) if output_fps is None else float(output_fps)
    times = relative_times_sec(rows, fallback_fps=fps)
    counts = [max(1, int(round((right - left) * fps))) for left, right in zip(times[:-1], times[1:])]
    counts.append(1)
    return counts

