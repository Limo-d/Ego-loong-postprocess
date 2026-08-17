#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MANO 3D 手模 + 68 路触觉热力图（SynchroTactile 风格）。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
if not (ROOT / "mano_v1_2").is_dir() and (ROOT.parent / "mano_v1_2").is_dir():
    ROOT = ROOT.parent
MANO_MODELS_DIR = ROOT / "mano_v1_2" / "models"
_ASSETS_DIR = ROOT / "assets"
_MESH_LEFT_NPZ = _ASSETS_DIR / "mano_left_mesh.npz"
_MESH_RIGHT_NPZ = _ASSETS_DIR / "mano_right_mesh.npz"
_ANCHORS_LEFT_NPY = _ASSETS_DIR / "mano_sensor_anchors_left.npy"
_ANCHORS_RIGHT_NPY = _ASSETS_DIR / "mano_sensor_anchors_right.npy"

_MANO_VIZ_BG = "#2b2b2b"
_MANO_BASE_GRAY = np.array([0.72, 0.72, 0.72], dtype=np.float64)
_MANO_CMAP = "jet"
_MANO_HEAT_SIGMA_FINGER = 0.012
_MANO_HEAT_SIGMA_PALM = 0.018
_MANO_HAND_SPACING = 0.28
_ANCHOR_CALIB_VERSION = 38
_PALM_GRID_ROWS = 6
_PALM_GRID_COLS = 8
_NUM_HEAT_REGIONS = 6  # 拇/食/中/无名/小指/掌


def _sensor_region_ids() -> np.ndarray:
    """68 路传感器 → 区域 id（0–4 指，5 掌）。"""
    from tactile_serial_reader import FINGER_POINTS, NUM_FINGERS

    n_finger = NUM_FINGERS * FINGER_POINTS
    regions = np.empty(n_finger + _PALM_GRID_ROWS * _PALM_GRID_COLS, dtype=np.int32)
    for f in range(NUM_FINGERS):
        regions[f * FINGER_POINTS : (f + 1) * FINGER_POINTS] = f
    regions[n_finger:] = NUM_FINGERS
    return regions


def _vertex_region_labels(
    vertices: np.ndarray,
    anchors: np.ndarray,
    palm_face: np.ndarray,
) -> np.ndarray:
    """掌面顶点按最近锚点所属区域划分，热力各区域互不影响。"""
    sensor_regions = _sensor_region_ids()
    labels = np.full(len(vertices), -1, dtype=np.int32)
    pf_idx = np.where(palm_face)[0]
    if pf_idx.size == 0:
        return labels
    pf_v = vertices[pf_idx]
    d2 = np.sum((pf_v[:, None, :] - anchors[None, :, :]) ** 2, axis=2)
    labels[pf_idx] = sensor_regions[np.argmin(d2, axis=1)]
    return labels


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("3D 可视化需要 matplotlib: pip install matplotlib") from e
    return plt


def _mano_pkl_path(is_right: bool) -> Path:
    name = "MANO_RIGHT.pkl" if is_right else "MANO_LEFT.pkl"
    return MANO_MODELS_DIR / name


def _mesh_npz_path(is_right: bool) -> Path:
    return _MESH_RIGHT_NPZ if is_right else _MESH_LEFT_NPZ


def _anchors_npy_path(is_right: bool) -> Path:
    return _ANCHORS_RIGHT_NPY if is_right else _ANCHORS_LEFT_NPY


def mano_models_available() -> bool:
    return _mano_pkl_path(False).is_file() or _MESH_LEFT_NPZ.is_file()


def export_mano_meshes(
    *,
    flat_hand: bool = True,
    global_orient: Optional[np.ndarray] = None,
) -> Tuple[Path, Path]:
    """从 MANO pkl 导出左右手网格缓存（需 smplx + torch + MANO_*.pkl）。"""
    import torch

    try:
        import smplx
    except ImportError as e:
        raise SystemExit("导出 MANO 网格需要 smplx: pip install smplx") from e

    if global_orient is None:
        global_orient = np.array([np.pi, 0.0, 0.0], dtype=np.float32)

    _ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for is_right in (False, True):
        pkl = _mano_pkl_path(is_right)
        if not pkl.is_file():
            raise FileNotFoundError(
                f"未找到 {pkl.name}，请从 https://mano.is.tue.mpg.de 下载并放入 {MANO_MODELS_DIR}"
            )
        layer = smplx.create(
            str(pkl),
            model_type="mano",
            is_rhand=is_right,
            use_pca=True,
            num_pca_comps=6,
            flat_hand_mean=flat_hand,
        )
        with torch.no_grad():
            out = layer(
                global_orient=torch.as_tensor(global_orient.reshape(1, 3), dtype=torch.float32),
                hand_pose=torch.zeros(1, 6),
                betas=torch.zeros(1, layer.num_betas),
                return_verts=True,
            )
        verts = out.vertices[0].cpu().numpy().astype(np.float64)
        faces = layer.faces.astype(np.int32)
        out_path = _mesh_npz_path(is_right)
        np.savez_compressed(out_path, vertices=verts, faces=faces)
        paths.append(out_path)
    return paths[0], paths[1]


def load_mano_mesh(is_right: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """加载 MANO 网格 (vertices, faces)，优先读缓存 npz。"""
    npz_path = _mesh_npz_path(is_right)
    if npz_path.is_file():
        data = np.load(npz_path)
        return data["vertices"].astype(np.float64), data["faces"].astype(np.int32)

    pkl = _mano_pkl_path(is_right)
    if pkl.is_file():
        export_mano_meshes()
        data = np.load(npz_path)
        return data["vertices"].astype(np.float64), data["faces"].astype(np.int32)

    side = "MANO_RIGHT.pkl / MANO_LEFT.pkl"
    raise FileNotFoundError(
        f"未找到 MANO 网格。请将 {side} 放入 {MANO_MODELS_DIR}，"
        f"然后运行: python scripts/export_mano_meshes.py"
    )


def _display_transform(vertices: np.ndarray, is_right: bool) -> np.ndarray:
    """掌心朝屏幕、手指向上；右手沿屏幕水平方向（X）镜像。"""
    from scipy.spatial.transform import Rotation

    v = vertices.copy()
    v -= v.mean(axis=0, keepdims=True)
    rot = Rotation.from_euler("y", -90, degrees=True).as_matrix()
    v = v @ rot.T
    if is_right:
        v[:, 0] *= -1.0
    return v


def _project_display_xy(vertices: np.ndarray) -> np.ndarray:
    """与 elev=0, azim=90 掌面视角一致的屏幕平面投影 (X, Z)。"""
    return np.column_stack([vertices[:, 0], vertices[:, 2]])


def _mirror_sensor_xy(sensor_xy: np.ndarray, *, is_right: bool) -> np.ndarray:
    sens = np.asarray(sensor_xy, dtype=np.float64).copy()
    if is_right:
        sens[:, 0] = sens[:, 0].max() + sens[:, 0].min() - sens[:, 0]
    return sens


def _palm_face_mask(v_disp: np.ndarray, *, xz_radius: float = 0.011) -> np.ndarray:
    """掌心朝相机一侧 + 指尖高 Z 区域（指尖 Y 较浅但仍可见）。"""
    xz = _project_display_xy(v_disp)
    y = v_disp[:, 1]
    cand = y >= float(np.percentile(y, 56))
    cand_idx = np.where(cand)[0]
    if cand_idx.size == 0:
        keep = cand
    else:
        keep = np.zeros(len(v_disp), dtype=bool)
        r2 = xz_radius * xz_radius
        for vi in cand_idx:
            d2 = np.sum((xz - xz[vi]) ** 2, axis=1)
            local = cand & (d2 <= r2)
            if y[vi] >= float(y[local].max()) - 1e-8:
                keep[vi] = True
        if keep.sum() < 80:
            keep = cand

    # 指腹/指尖：Z 较高但 Y 偏浅，原掌面判定会漏掉（C0/C1 等）
    tip_extra = xz[:, 1] >= float(np.percentile(xz[:, 1], 58))
    return keep | tip_extra


def _mesh_plane_bounds(
    plane_xy: np.ndarray,
    mask: np.ndarray,
    pad_frac: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    pts = plane_xy[mask]
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    pad = np.maximum((hi - lo) * pad_frac, 0.003)
    return lo - pad, hi + pad


def _layout_to_display_xz(
    sens: np.ndarray,
    mesh_lo: np.ndarray,
    mesh_hi: np.ndarray,
) -> np.ndarray:
    """布局图像素坐标 → 屏幕平面 (X, Z)，与 2D sensor_layout_coords 同构。"""
    sx = sens[:, 0]
    sy = sens[:, 1]
    x0, x1 = float(sx.min()), float(sx.max())
    y0, y1 = float(sy.min()), float(sy.max())
    tx = (x1 - sx) / max(x1 - x0, 1e-9)
    ty = (sy - y0) / max(y1 - y0, 1e-9)
    lo = np.asarray(mesh_lo, dtype=np.float64)
    hi = np.asarray(mesh_hi, dtype=np.float64)
    return np.column_stack(
        [
            lo[0] + tx * (hi[0] - lo[0]),
            hi[1] - ty * (hi[1] - lo[1]),
        ]
    )


def _anchor_vertex_ids(v_disp: np.ndarray, anchors: np.ndarray) -> set:
    """锚点 → 最近 mesh 顶点索引。"""
    return {
        int(np.argmin(np.sum((v_disp - a) ** 2, axis=1))) for a in anchors
    }


def _palm_grid_region_mask(v_disp: np.ndarray, palm_face: np.ndarray) -> np.ndarray:
    """掌心 6×8 可用区域：掌面、排除指尖远端。"""
    xz = _project_display_xy(v_disp)
    y = v_disp[:, 1]
    not_tip = xz[:, 1] <= 0.018
    not_back = y >= float(np.percentile(y[palm_face], 34))
    return palm_face & not_tip & not_back


def _palm_grid_serpentine_order() -> List[int]:
    """蛇形遍历 6×8，顺序分配时减少局部挤压。"""
    order: List[int] = []
    for ri in range(_PALM_GRID_ROWS):
        cols = (
            range(_PALM_GRID_COLS)
            if ri % 2 == 0
            else range(_PALM_GRID_COLS - 1, -1, -1)
        )
        for ci in cols:
            order.append(ri * _PALM_GRID_COLS + ci)
    return order


def _assign_palm_grid_anchors(
    v_disp: np.ndarray,
    sens_palm: np.ndarray,
    palm_face: np.ndarray,
    *,
    reserved_verts: set,
) -> np.ndarray:
    """F0–F47：在掌心区域均匀映射，不占用手指已保留顶点。"""
    xz = _project_display_xy(v_disp)
    palm_mask = _palm_grid_region_mask(v_disp, palm_face)
    pg_xz = xz[np.where(palm_mask)[0]]
    lo_p, hi_p = _mesh_plane_bounds(pg_xz, np.ones(len(pg_xz), dtype=bool), pad_frac=0.03)
    palm_targets = _layout_to_display_xz(sens_palm, lo_p, hi_p)

    reg_idx = np.where(palm_mask)[0]
    used = set(reserved_verts)
    order = _palm_grid_serpentine_order()
    anchors = np.zeros((len(sens_palm), 3), dtype=np.float64)

    for si in order:
        tgt = palm_targets[si]
        cand = np.array([int(vi) for vi in reg_idx if int(vi) not in used], dtype=np.int64)
        if cand.size == 0:
            cand = np.array([int(vi) for vi in reg_idx], dtype=np.int64)
        d2 = np.sum((xz[cand] - tgt) ** 2, axis=1)
        vi = int(cand[int(np.argmin(d2))])
        anchors[si] = v_disp[vi]
        used.add(vi)
    return anchors


def _assign_sensors_to_palm_vertices(
    v_disp: np.ndarray,
    sens: np.ndarray,
    palm_face: np.ndarray,
) -> np.ndarray:
    """全局 68 点布局 → 掌面锚点（屏幕 XZ 最近邻 + 唯一性）。"""
    xz = _project_display_xy(v_disp)
    pf_idx = np.where(palm_face)[0]
    pf_xz = xz[pf_idx]
    lo, hi = _mesh_plane_bounds(pf_xz, np.ones(len(pf_xz), dtype=bool), pad_frac=0.01)
    targets = _layout_to_display_xz(sens, lo, hi)

    d2 = np.sum((pf_xz[None, :, :] - targets[:, None, :]) ** 2, axis=2)
    pairs: List[Tuple[float, int, int]] = []
    for si in range(len(sens)):
        order = np.argsort(d2[si])
        for rank, k in enumerate(order[:48]):
            pairs.append((d2[si, k], si, int(pf_idx[k])))
    pairs.sort(key=lambda item: item[0])

    anchors = np.zeros((len(sens), 3), dtype=np.float64)
    assigned_sens: set = set()
    used_verts: set = set()
    for _, si, vi in pairs:
        if si in assigned_sens or vi in used_verts:
            continue
        anchors[si] = v_disp[vi]
        assigned_sens.add(si)
        used_verts.add(vi)

    for si in range(len(sens)):
        if si in assigned_sens:
            continue
        k = int(np.argmin(d2[si]))
        anchors[si] = v_disp[int(pf_idx[k])]
    return anchors


def _refine_thumb_inner_anchors(
    v_disp: np.ndarray,
    sens: np.ndarray,
    palm_face: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    """A2/A3：往拇指内侧（掌心侧，较低 Z、略靠掌中心 X）。"""
    out = anchors.copy()
    xz = _project_display_xy(v_disp)
    a0, a1, a2, a3 = 0, 1, 2, 3

    anchor_verts = [
        int(np.argmin(np.sum((v_disp - out[i]) ** 2, axis=1)))
        for i in range(len(out))
    ]
    used = {anchor_verts[i] for i in range(len(out)) if i not in (a2, a3)}

    a_xz = xz[anchor_verts[a0 : a3 + 1]]
    a_sens = sens[a0 : a3 + 1]
    sx = a_sens[:, 0]
    sx0, sx1 = float(sx.min()), float(sx.max())
    sx_span = max(sx1 - sx0, 1e-9)

    tip_x = float(np.mean(a_xz[0:2, 0]))
    if tip_x >= 0.0:
        x_band = (tip_x - 0.042, tip_x + 0.010)
    else:
        x_band = (tip_x - 0.010, tip_x + 0.042)

    edge_mask = (
        palm_face
        & (xz[:, 0] >= min(x_band))
        & (xz[:, 0] <= max(x_band))
        & (xz[:, 1] < -0.024)
        & (xz[:, 1] > -0.068)
    )
    edge_xz = xz[edge_mask]
    if edge_xz.shape[0] >= 4:
        z_lo = float(np.percentile(edge_xz[:, 1], 10))
        z_hi = float(np.percentile(edge_xz[:, 1], 48))
        if tip_x >= 0.0:
            x_inner = float(np.percentile(edge_xz[:, 0], 18))
            x_mid = float(np.percentile(edge_xz[:, 0], 52))
        else:
            x_inner = float(np.percentile(edge_xz[:, 0], 48))
            x_mid = float(np.percentile(edge_xz[:, 0], 82))
    else:
        z_lo, z_hi = -0.058, -0.038
        if tip_x >= 0.0:
            x_inner, x_mid = tip_x - 0.028, tip_x - 0.012
        else:
            x_inner, x_mid = tip_x + 0.012, tip_x + 0.028

    targets = np.zeros((2, 2), dtype=np.float64)
    for k, idx in enumerate((a2, a3)):
        tx = (float(a_sens[idx, 0]) - sx0) / sx_span
        tx = float(np.clip(tx, 0.0, 1.0))
        targets[k, 0] = x_inner + tx * (x_mid - x_inner)
        targets[k, 1] = z_lo + 0.16 * (z_hi - z_lo)

    region = edge_mask & (
        (xz[:, 0] >= float(targets[:, 0].min()) - 0.010)
        & (xz[:, 0] <= float(targets[:, 0].max()) + 0.010)
    )
    base_cand = np.array([i for i in np.where(region)[0] if i not in used], dtype=np.int64)
    if base_cand.size < 2:
        base_cand = np.array([i for i in np.where(edge_mask)[0] if i not in used], dtype=np.int64)

    picked: List[int] = []
    for si in range(2):
        cand = np.array([i for i in base_cand if int(i) not in picked], dtype=np.int64)
        if cand.size == 0:
            continue
        cand_xz = xz[cand]
        dx = (cand_xz[:, 0] - targets[si, 0]) * 2.5
        dz = (cand_xz[:, 1] - targets[si, 1]) * 1.2
        d2 = dx * dx + dz * dz
        for k in np.argsort(d2):
            vi = int(cand[k])
            if vi in picked:
                continue
            out[a2 + si] = v_disp[vi].copy()
            picked.append(vi)
            break
    return out


def _refine_pinky_palm_edge_anchors(
    v_disp: np.ndarray,
    sens: np.ndarray,
    palm_face: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    """E2/E3：贴掌缘（小指侧掌心交界，低 Z 区域）。"""
    out = anchors.copy()
    xz = _project_display_xy(v_disp)
    e0, e1, e2, e3 = 16, 17, 18, 19

    anchor_verts = [
        int(np.argmin(np.sum((v_disp - out[i]) ** 2, axis=1)))
        for i in range(len(out))
    ]
    used = {anchor_verts[i] for i in range(len(out)) if i not in (e2, e3)}

    e_xz = xz[anchor_verts[e0 : e3 + 1]]
    e_sens = sens[e0 : e3 + 1]
    x_left, x_right = float(e_xz[0, 0]), float(e_xz[1, 0])
    sx_left, sx_right = float(e_sens[2, 0]), float(e_sens[3, 0])
    sx_span = max(sx_right - sx_left, 1e-9)

    edge_mask = (
        palm_face
        & (xz[:, 0] < -0.054)
        & (xz[:, 1] < 0.014)
        & (xz[:, 1] > -0.012)
    )
    edge_xz = xz[edge_mask]
    if edge_xz.shape[0] >= 4:
        z_lo = float(np.percentile(edge_xz[:, 1], 18))
        z_hi = float(np.percentile(edge_xz[:, 1], 62))
    else:
        z_lo, z_hi = 0.0, 0.012

    targets = np.zeros((2, 2), dtype=np.float64)
    for k, idx in enumerate((e2, e3)):
        tx = (float(e_sens[idx - e0, 0]) - sx_left) / sx_span
        tx = float(np.clip(tx, 0.0, 1.0))
        targets[k, 0] = x_left + tx * (x_right - x_left)
        targets[k, 1] = z_lo + 0.35 * (z_hi - z_lo)

    region = edge_mask & (
        (xz[:, 0] >= float(targets[:, 0].min()) - 0.012)
        & (xz[:, 0] <= float(targets[:, 0].max()) + 0.012)
    )
    cand = np.array([i for i in np.where(region)[0] if i not in used], dtype=np.int64)
    if cand.size < 2:
        cand = np.array([i for i in np.where(edge_mask)[0] if i not in used], dtype=np.int64)

    cand_xz = xz[cand]
    picked: List[int] = []
    for si in range(2):
        dx = (cand_xz[:, 0] - targets[si, 0]) * 3.0
        dz = cand_xz[:, 1] - targets[si, 1]
        d2 = dx * dx + dz * dz
        for k in np.argsort(d2):
            vi = int(cand[k])
            if vi in picked:
                continue
            out[e2 + si] = v_disp[vi].copy()
            picked.append(vi)
            break
        else:
            vi = int(cand[int(np.argmin(d2))])
            out[e2 + si] = v_disp[vi].copy()
            if vi not in picked:
                picked.append(vi)
    return out


def calibrate_sensor_anchors(
    vertices: np.ndarray,
    sensor_xy_2d: np.ndarray,
    *,
    is_right: bool = False,
) -> np.ndarray:
    """68 路布局 → MANO 掌心面 3D 锚点（屏幕 XZ 与 2D 布局同构）。"""
    from tactile_serial_reader import FINGER_POINTS, NUM_FINGERS

    n_expected = NUM_FINGERS * FINGER_POINTS + _PALM_GRID_ROWS * _PALM_GRID_COLS
    sens = _mirror_sensor_xy(sensor_xy_2d, is_right=is_right)
    if sens.shape[0] != n_expected:
        raise ValueError(f"期望 68 路坐标，得到 {sens.shape[0]}")

    v_disp = _display_transform(vertices, is_right)
    palm_face = _palm_face_mask(v_disp)
    n_finger = NUM_FINGERS * FINGER_POINTS

    # 手指：沿用 v33 全局贪心 + E2/E3 掌缘 refine，映射保持不变
    anchors = _assign_sensors_to_palm_vertices(v_disp, sens, palm_face)
    anchors = _refine_pinky_palm_edge_anchors(v_disp, sens, palm_face, anchors)
    anchors = _refine_thumb_inner_anchors(v_disp, sens, palm_face, anchors)
    finger_anchors = anchors[:n_finger].copy()
    reserved = _anchor_vertex_ids(v_disp, finger_anchors)

    # 掌心：独立重标，仅使用未被手指占用的顶点
    palm_anchors = _assign_palm_grid_anchors(
        v_disp,
        sens[n_finger:],
        palm_face,
        reserved_verts=reserved,
    )
    return np.vstack([finger_anchors, palm_anchors])


def _anchors_version_path(is_right: bool) -> Path:
    return _anchors_npy_path(is_right).with_suffix(".version")


def _sensor_sigmas_from_anchors(anchors: np.ndarray) -> np.ndarray:
    """按锚点局部间距估计 RBF 半径（带上限，避免热力铺满整手）。"""
    from tactile_serial_reader import FINGER_POINTS, NUM_FINGERS

    n_finger = NUM_FINGERS * FINGER_POINTS
    sig = np.full(len(anchors), _MANO_HEAT_SIGMA_PALM, dtype=np.float64)

    for f in range(NUM_FINGERS):
        pts = anchors[f * FINGER_POINTS : (f + 1) * FINGER_POINTS]
        dists = [
            float(np.linalg.norm(pts[i] - pts[j]))
            for i in range(FINGER_POINTS)
            for j in range(i + 1, FINGER_POINTS)
        ]
        s = float(np.median(dists)) * 0.55 if dists else _MANO_HEAT_SIGMA_FINGER
        s = float(np.clip(s, _MANO_HEAT_SIGMA_FINGER * 0.75, 0.016))
        sig[f * FINGER_POINTS : (f + 1) * FINGER_POINTS] = s

    palm = anchors[n_finger:].reshape(_PALM_GRID_ROWS, _PALM_GRID_COLS, 3)
    for ri in range(_PALM_GRID_ROWS):
        for ci in range(_PALM_GRID_COLS):
            idx = n_finger + ri * _PALM_GRID_COLS + ci
            dists: List[float] = []
            if ci > 0:
                dists.append(float(np.linalg.norm(palm[ri, ci] - palm[ri, ci - 1])))
            if ci + 1 < _PALM_GRID_COLS:
                dists.append(float(np.linalg.norm(palm[ri, ci] - palm[ri, ci + 1])))
            if ri > 0:
                dists.append(float(np.linalg.norm(palm[ri, ci] - palm[ri - 1, ci])))
            if ri + 1 < _PALM_GRID_ROWS:
                dists.append(float(np.linalg.norm(palm[ri, ci] - palm[ri + 1, ci])))
            if dists:
                s = float(np.median(dists)) * 0.75
                sig[idx] = float(np.clip(s, _MANO_HEAT_SIGMA_PALM * 0.8, 0.014))
    return sig


def load_sensor_anchors(
    vertices: np.ndarray,
    sensor_xy_2d: np.ndarray,
    *,
    is_right: bool = False,
) -> np.ndarray:
    path = _anchors_npy_path(is_right)
    ver_path = _anchors_version_path(is_right)
    if path.is_file() and ver_path.is_file():
        try:
            if int(ver_path.read_text(encoding="utf-8").strip()) == _ANCHOR_CALIB_VERSION:
                anchors = np.load(path)
                if anchors.shape == (len(sensor_xy_2d), 3):
                    return anchors.astype(np.float64)
        except ValueError:
            pass
    anchors = calibrate_sensor_anchors(vertices, sensor_xy_2d, is_right=is_right)
    _ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, anchors)
    ver_path.write_text(str(_ANCHOR_CALIB_VERSION), encoding="utf-8")
    return anchors


def _sensor_sigmas(anchors: Optional[np.ndarray] = None) -> np.ndarray:
    if anchors is not None:
        return _sensor_sigmas_from_anchors(anchors)
    from tactile_serial_reader import FINGER_POINTS, NUM_FINGERS

    n_finger = NUM_FINGERS * FINGER_POINTS
    sig = np.full(n_finger + _PALM_GRID_ROWS * _PALM_GRID_COLS, _MANO_HEAT_SIGMA_PALM)
    sig[:n_finger] = _MANO_HEAT_SIGMA_FINGER
    return sig.astype(np.float64)


def vertex_heat(
    vertices: np.ndarray,
    anchors: np.ndarray,
    values: np.ndarray,
    sigmas: np.ndarray,
    *,
    vmin: float = 0.0,
    vmax: float = 200.0,
    vertex_regions: Optional[np.ndarray] = None,
    sensor_regions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """RBF 插值：传感 AD → 逐顶点热强度 [0, 1]（固定色标 vmin–vmax）。"""
    vals = np.clip(np.asarray(values, dtype=np.float64).reshape(-1), 0.0, None)
    span = max(float(vmax - vmin), 1e-9)
    vals = np.clip((vals - vmin) / span, 0.0, 1.0)
    heat = np.zeros(len(vertices), dtype=np.float64)
    if sensor_regions is None:
        sensor_regions = _sensor_region_ids()
    for si, (anchor, val, sigma) in enumerate(zip(anchors, vals, sigmas)):
        if val <= 1e-6:
            continue
        d2 = np.sum((vertices - anchor) ** 2, axis=1)
        contrib = val * np.exp(-d2 / (2.0 * sigma * sigma))
        if vertex_regions is not None:
            contrib = np.where(vertex_regions == sensor_regions[si], contrib, 0.0)
        heat += contrib
    return np.clip(heat, 0.0, 1.0)


def _vertex_rgba(base_gray: np.ndarray, heat: np.ndarray, cmap) -> np.ndarray:
    rgb = cmap(heat)[:, :3]
    alpha = np.clip(heat * 1.15, 0.0, 1.0)[:, None]
    out = base_gray[None, :] * (1.0 - alpha) + rgb * alpha
    return np.clip(out, 0.0, 1.0)


def _face_colors(vertex_rgba: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return vertex_rgba[faces].mean(axis=1)


class ManoHandTactileViewer:
    """双手 MANO 网格 + 触觉热力图（左：实时数据，右：镜像布局无数据）。"""

    def __init__(
        self,
        sensor_xy_2d: np.ndarray,
        *,
        vmin: float = 0.0,
        vmax: float = 200.0,
        value_unit: str = "AD",
        subtract_baseline: bool = False,
        baseline_frames: int = 10,
        show_right_hand: bool = True,
        show_right_heat: bool = False,
        title: str = "SynchroTactile",
    ) -> None:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.colors import Normalize

        self._Poly3DCollection = Poly3DCollection
        self._value_unit = value_unit
        self._clim = (vmin, vmax)
        self._subtract_baseline = subtract_baseline
        self._baseline_frames = max(1, baseline_frames)
        self._baseline: Optional[np.ndarray] = None
        self._baseline_buf: List[np.ndarray] = []
        self._show_right = show_right_hand
        self._show_right_heat = show_right_heat

        self._v_left, self._f_left = load_mano_mesh(is_right=False)
        hand_sep = _MANO_HAND_SPACING * 0.55
        self._anchors_left = load_sensor_anchors(
            self._v_left, sensor_xy_2d, is_right=False
        ).copy()
        self._sigmas_left = _sensor_sigmas(self._anchors_left)
        self._v_left_disp = _display_transform(self._v_left, is_right=False)
        # elev=0, azim=90 时视线沿 Y，双手沿 X 分开（负 X 在屏幕左侧）
        self._v_left_disp[:, 0] += hand_sep
        self._anchors_left[:, 0] += hand_sep
        self._pf_left = _palm_face_mask(self._v_left_disp)
        self._region_labels_left = _vertex_region_labels(
            self._v_left_disp, self._anchors_left, self._pf_left
        )
        self._sensor_regions = _sensor_region_ids()

        if show_right_hand:
            # 与左手同拓扑，镜像显示右手（保证掌心同样朝向屏幕）
            self._v_right = self._v_left
            self._f_right = self._f_left
            self._anchors_right = load_sensor_anchors(
                self._v_left, sensor_xy_2d, is_right=True
            ).copy()
            self._sigmas_right = _sensor_sigmas(self._anchors_right)
            self._v_right_disp = _display_transform(self._v_left, is_right=True)
            self._v_right_disp[:, 0] -= hand_sep
            self._anchors_right[:, 0] -= hand_sep
            self._pf_right = _palm_face_mask(self._v_right_disp)
            self._region_labels_right = _vertex_region_labels(
                self._v_right_disp, self._anchors_right, self._pf_right
            )
        else:
            self._v_right = self._f_right = self._anchors_right = None
            self._v_right_disp = None
            self._sigmas_right = None
            self._pf_right = None
            self._region_labels_right = None

        plt = _import_matplotlib()
        self._plt = plt
        plt.rcParams.update(
            {
                "figure.facecolor": _MANO_VIZ_BG,
                "axes.facecolor": _MANO_VIZ_BG,
                "font.family": "sans-serif",
                "font.sans-serif": ["Segoe UI", "Microsoft YaHei", "DejaVu Sans"],
            }
        )

        self.fig = plt.figure(figsize=(11.5, 5.5), dpi=100)
        self.fig.patch.set_facecolor(_MANO_VIZ_BG)
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title(title)

        self._ax = self.fig.add_axes([0.02, 0.08, 0.96, 0.82], projection="3d")
        self._ax.set_facecolor(_MANO_VIZ_BG)
        self._ax.axis("off")
        try:
            self._ax.set_proj_type("ortho")
        except AttributeError:
            pass

        self._cmap = plt.get_cmap(_MANO_CMAP)
        self._norm = Normalize(vmin=vmin, vmax=vmax)

        zero = np.zeros(len(sensor_xy_2d))
        c_left = self._vertex_colors(
            self._v_left_disp,
            self._anchors_left,
            zero,
            self._sigmas_left,
            self._pf_left,
            self._region_labels_left,
        )
        self._mesh_left = self._make_collection(self._v_left_disp, self._f_left, c_left)
        self._ax.add_collection3d(self._mesh_left)

        if show_right_hand and self._v_right_disp is not None:
            c_right = self._vertex_colors(
                self._v_right_disp,
                self._anchors_right,
                zero,
                self._sigmas_right,
                self._pf_right,
                self._region_labels_right,
            )
            self._mesh_right = self._make_collection(
                self._v_right_disp, self._f_right, c_right
            )
            self._ax.add_collection3d(self._mesh_right)
        else:
            self._mesh_right = None

        self._set_camera()
        self._fig_text(title)
        self._plt.ion()
        self._plt.show(block=False)

    def _vertex_colors(
        self,
        verts: np.ndarray,
        anchors: np.ndarray,
        values: np.ndarray,
        sigmas: np.ndarray,
        palm_face: np.ndarray,
        vertex_regions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        heat = vertex_heat(
            verts,
            anchors,
            values,
            sigmas,
            vmin=self._clim[0],
            vmax=self._clim[1],
            vertex_regions=vertex_regions,
            sensor_regions=self._sensor_regions,
        )
        heat = np.where(palm_face, heat, 0.0)
        return _vertex_rgba(_MANO_BASE_GRAY, heat, self._cmap)

    def _make_collection(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        vertex_rgba: np.ndarray,
    ):
        tris = verts[faces]
        fc = _face_colors(vertex_rgba, faces)
        coll = self._Poly3DCollection(
            tris,
            linewidths=0.08,
            edgecolors=(0.35, 0.35, 0.35, 0.25),
        )
        coll.set_facecolor(fc)
        return coll

    def _set_camera(self) -> None:
        all_v = [self._v_left_disp]
        if self._v_right_disp is not None:
            all_v.append(self._v_right_disp)
        stacked = np.vstack(all_v)
        mins = stacked.min(axis=0)
        maxs = stacked.max(axis=0)
        spans = np.maximum(maxs - mins, 1e-6)
        pad = 0.12 * float(spans.max())
        self._ax.set_xlim(mins[0] - pad, maxs[0] + pad)
        self._ax.set_ylim(mins[1] - pad, maxs[1] + pad)
        self._ax.set_zlim(mins[2] - pad, maxs[2] + pad)
        self._ax.view_init(elev=0, azim=90)
        # 按数据真实跨度设置 box aspect，避免 mpl 3.10+ 把模型压成竖线
        try:
            self._ax.set_box_aspect(tuple(float(s) for s in spans))
        except AttributeError:
            pass

    def _fig_text(self, title: str) -> None:
        self.fig.text(
            0.03,
            0.94,
            title,
            ha="left",
            va="top",
            fontsize=13,
            color="#e8e8e8",
            fontweight="bold",
        )
        self.fig.text(
            0.28,
            0.02,
            "hand_left",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#aaaaaa",
        )
        if self._show_right:
            self.fig.text(
                0.72,
                0.02,
                "hand_right",
                ha="center",
                va="bottom",
                fontsize=11,
                color="#aaaaaa",
            )
        self._subtitle = self.fig.text(
            0.5,
            0.94,
            "",
            ha="center",
            va="top",
            fontsize=9,
            color="#888888",
        )

    def _prepare_values(self, values: np.ndarray) -> Tuple[np.ndarray, str]:
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        if not self._subtract_baseline:
            return vals, ""
        if self._baseline is None:
            self._baseline_buf.append(vals)
            if len(self._baseline_buf) >= self._baseline_frames:
                self._baseline = np.median(np.stack(self._baseline_buf), axis=0)
            n = len(self._baseline_buf)
            return np.zeros_like(vals), f"Calibrating baseline {n}/{self._baseline_frames}"
        delta = np.clip(vals - self._baseline, 0.0, None)
        return delta, ""

    def update(
        self,
        hand: Any,
        *,
        sequence: Optional[int] = None,
        pressure: Optional[np.ndarray] = None,
    ) -> None:
        from tactile_serial_reader import hand_to_pressure_vector

        if pressure is not None:
            values = np.asarray(pressure, dtype=np.float64).reshape(-1)
        else:
            values = hand_to_pressure_vector(hand)

        values, note = self._prepare_values(values)
        c_left = self._vertex_colors(
            self._v_left_disp,
            self._anchors_left,
            values,
            self._sigmas_left,
            self._pf_left,
            self._region_labels_left,
        )
        self._mesh_left.set_facecolor(_face_colors(c_left, self._f_left))

        if self._mesh_right is not None and self._v_right_disp is not None:
            right_vals = values if self._show_right_heat else np.zeros_like(values)
            c_right = self._vertex_colors(
                self._v_right_disp,
                self._anchors_right,
                right_vals,
                self._sigmas_right,
                self._pf_right,
                self._region_labels_right,
            )
            self._mesh_right.set_facecolor(_face_colors(c_right, self._f_right))

        peak = float(np.max(values))
        parts = [f"Peak {peak:.0f} {self._value_unit}"]
        if sequence is not None:
            parts.append(f"Seq {sequence}")
        if note:
            parts.append(note)
        self._subtitle.set_text("  ·  ".join(parts))

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self._plt.pause(0.001)

    def close(self) -> None:
        self._plt.close(self.fig)


def plot_mano_demo(
    values: Optional[np.ndarray] = None,
    *,
    block: bool = True,
    show_right_hand: bool = True,
) -> None:
    """静态 MANO 3D 演示（无需串口）。"""
    from tactile_serial_reader import NUM_TACTILE, _SENSOR_VIZ_XY, decode_hand, TactileFrame

    if values is None:
        rng = np.random.default_rng(0)
        values = np.zeros(NUM_TACTILE)
        # 左手：掌部 F0–F47 为主；右手演示在下方单独赋值
        values[20:68] = rng.uniform(100, 200, 48)
        values[0:4] = rng.uniform(60, 120, 4)
    frame = TactileFrame(sequence=1, pressure=values, timestamp=0.0)
    hand = decode_hand(frame)
    viewer = ManoHandTactileViewer(
        _SENSOR_VIZ_XY,
        subtract_baseline=False,
        show_right_hand=show_right_hand,
        show_right_heat=True,
    )
    viewer.update(hand, sequence=1)
    if show_right_hand and viewer._mesh_right is not None:
        rng = np.random.default_rng(1)
        right_vals = np.zeros(NUM_TACTILE)
        right_vals[4:8] = rng.uniform(140, 200, 4)
        right_vals[8:12] = rng.uniform(120, 190, 4)
        right_vals[16:20] = rng.uniform(100, 170, 4)
        c_right = viewer._vertex_colors(
            viewer._v_right_disp,
            viewer._anchors_right,
            right_vals,
            viewer._sigmas_right,
            viewer._pf_right,
            viewer._region_labels_right,
        )
        viewer._mesh_right.set_facecolor(_face_colors(c_right, viewer._f_right))
        viewer._subtitle.set_text("Peak demo · Seq 1")
    viewer._plt.show(block=block)
