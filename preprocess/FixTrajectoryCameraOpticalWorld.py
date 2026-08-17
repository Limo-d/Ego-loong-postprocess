#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute trajectory world points using base_link -> RGB optical camera TF.

Older sampler outputs stored /odom pose (child base_link) directly as camera.c2w,
while hand keypoints are in oak_rgb_optical_frame. This script composes the
static TF chain and rewrites camera/head c2w plus glove world keypoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        f.write('\n')


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write('\n')


def quat_wxyz_to_mat(q: List[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = float(np.sqrt(w*w + x*x + y*y + z*z))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def tf_to_mat(tf: Dict) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = quat_wxyz_to_mat(tf['rotation_wxyz'])
    mat[:3, 3] = np.asarray(tf['translation'], dtype=np.float64)
    return mat


def invert(mat: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = mat[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ mat[:3, 3]
    return out


def load_static_edges(path: Path) -> Dict[Tuple[str, str], np.ndarray]:
    edges: Dict[Tuple[str, str], np.ndarray] = {}
    if not path.exists():
        return edges
    for row in read_jsonl(path):
        records = row.get('transforms') if isinstance(row.get('transforms'), list) else [row]
        for tr in records:
            parent = tr.get('parent')
            child = tr.get('child')
            if parent and child and tr.get('translation') is not None and tr.get('rotation_wxyz') is not None:
                mat = tf_to_mat(tr)
                edges[(parent, child)] = mat
                edges[(child, parent)] = invert(mat)
    return edges


def find_chain(edges: Dict[Tuple[str, str], np.ndarray], source: str, target: str) -> Optional[np.ndarray]:
    if source == target:
        return np.eye(4, dtype=np.float64)
    frontier = [(source, np.eye(4, dtype=np.float64))]
    seen = {source}
    while frontier:
        node, acc = frontier.pop(0)
        for (a, b), mat in edges.items():
            if a != node or b in seen:
                continue
            nxt = acc @ mat
            if b == target:
                return nxt
            seen.add(b)
            frontier.append((b, nxt))
    return None


def default_base_to_rgb_optical() -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    # From current tf_static: oak_rgb_frame -> oak_rgb_optical_frame.
    mat[:3, :3] = quat_wxyz_to_mat([0.5, -0.5, 0.5, -0.5])
    return mat


def valid_points(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape[0] < 21 or arr.shape[1] < 3 or not np.isfinite(arr[:21, :3]).all():
        return None
    return arr[:21, :3]


def transform_points(mat4: np.ndarray, pts: np.ndarray) -> np.ndarray:
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return (mat4 @ homo.T).T[:, :3]


def process(args: argparse.Namespace) -> Dict:
    rows = read_jsonl(Path(args.input_jsonl).expanduser().resolve())
    edges = load_static_edges(Path(args.tf_static_jsonl).expanduser().resolve()) if args.tf_static_jsonl else {}
    t_base_cam = find_chain(edges, args.base_frame, args.camera_frame) if edges else None
    if t_base_cam is None:
        if args.allow_default_oak_rgb_optical:
            t_base_cam = default_base_to_rgb_optical()
            tf_source = 'default_oak_rgb_optical_rotation'
        else:
            raise RuntimeError(f'Cannot find static TF chain {args.base_frame} -> {args.camera_frame}')
    else:
        tf_source = str(Path(args.tf_static_jsonl).expanduser().resolve())

    out_rows = []
    stats = {'frames': 0, 'camera_pose_fixed': 0, 'glove_world_recomputed': 0, 'missing_camera': 0, 'missing_glove_camera_points': 0}
    cam_z_old = []
    wrist_z_old = []
    wrist_z_new = []

    for row in rows:
        stats['frames'] += 1
        new = json.loads(json.dumps(row))
        camera = dict(new.get('camera') or {})
        head_pose = dict(new.get('head_pose') or {})
        base_c2w_raw = (row.get('camera') or {}).get('c2w') or (row.get('head_pose') or {}).get('c2w')
        base_c2w = np.asarray(base_c2w_raw, dtype=np.float64) if base_c2w_raw is not None else None
        if base_c2w is None or base_c2w.shape != (4, 4) or not np.isfinite(base_c2w).all():
            stats['missing_camera'] += 1
            out_rows.append(new)
            continue

        cam_c2w = base_c2w @ t_base_cam
        camera['c2w_before_camera_optical_fix'] = base_c2w.tolist()
        camera['c2w'] = cam_c2w.tolist()
        camera['frame_id'] = args.camera_frame
        camera['camera_optical_fix'] = {
            'method': 'compose odom->base_link with static base_link->camera optical transform',
            'base_frame': args.base_frame,
            'camera_frame': args.camera_frame,
            'tf_source': tf_source,
            't_base_camera': t_base_cam.tolist(),
        }
        head_pose['c2w_before_camera_optical_fix'] = base_c2w.tolist()
        head_pose['c2w'] = cam_c2w.tolist()
        new['camera'] = camera
        new['head_pose'] = head_pose
        stats['camera_pose_fixed'] += 1
        cam_z_old.append(float(base_c2w[2, 3]))

        glove = dict(new.get('glove') or {})
        pts_cam = valid_points(glove.get('kpts_3d_camera_m'))
        if pts_cam is None:
            stats['missing_glove_camera_points'] += 1
        else:
            old_world = valid_points(glove.get('kpts_3d_world_m'))
            if old_world is not None:
                wrist_z_old.append(float(old_world[0, 2]))
            pts_world = transform_points(cam_c2w, pts_cam)
            glove['kpts_3d_world_m_before_camera_optical_fix'] = glove.get('kpts_3d_world_m')
            glove['kpts_3d_world_m'] = pts_world.tolist()
            new['glove'] = glove
            wrist_z_new.append(float(pts_world[0, 2]))
            stats['glove_world_recomputed'] += 1

        out_rows.append(new)

    summary = {
        'input_jsonl': str(Path(args.input_jsonl).expanduser().resolve()),
        'output_jsonl': str(Path(args.output_jsonl).expanduser().resolve()),
        'base_frame': args.base_frame,
        'camera_frame': args.camera_frame,
        't_base_camera': t_base_cam.tolist(),
        'tf_source': tf_source,
        'stats': stats,
    }
    if cam_z_old and wrist_z_old and wrist_z_new:
        summary['z_medians'] = {
            'old_camera_base_z': float(np.median(cam_z_old)),
            'old_wrist_world_z': float(np.median(wrist_z_old)),
            'new_wrist_world_z': float(np.median(wrist_z_new)),
            'new_wrist_minus_camera_z': float(np.median(np.asarray(wrist_z_new) - np.asarray(cam_z_old[:len(wrist_z_new)]))),
        }

    write_jsonl(Path(args.output_jsonl).expanduser().resolve(), out_rows)
    if args.summary_json:
        write_json(Path(args.summary_json).expanduser().resolve(), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Fix trajectory world coordinates by composing base_link->camera optical static TF.')
    parser.add_argument('--input_jsonl', required=True)
    parser.add_argument('--output_jsonl', required=True)
    parser.add_argument('--summary_json', default=None)
    parser.add_argument('--tf_static_jsonl', default=None)
    parser.add_argument('--base_frame', default='base_link')
    parser.add_argument('--camera_frame', default='oak_rgb_optical_frame')
    parser.add_argument('--allow_default_oak_rgb_optical', action='store_true')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = process(args)
    print(f"[FixTrajectoryCameraOpticalWorld] output: {args.output_jsonl}")
    print(f"[FixTrajectoryCameraOpticalWorld] summary: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == '__main__':
    main()
