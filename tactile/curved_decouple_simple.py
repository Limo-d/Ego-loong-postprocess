#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整手握拳扫掠标定 + 弯曲伪触觉解耦（姿态库 + 默认 1-NN 全帧模板）。

姿态代理两种模式（pose_mode）：
  - pressure_sum：每指 4 路压力和 → 5 维（当前默认，无独立关节传感器）
  - joint：独立关节角 → 14 维（拇指 2 + 食/中/无/小各 3）

目标：标定扫掠内的空手姿势，解耦残差趋近零；抓握时残差反映接触力。

流程：
  1. record：空手缓慢握拳扫掠 → sweep.npz
  2. build：全帧姿态库（默认不下采样）+ 1-NN 模板相减
  3. viz：对比「解耦前 / 解耦后」

可选 --method knn：高斯加权（旧行为，精度通常更差）。
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from tactile_serial_reader import (
    DEFAULT_SOLVED_WAIT_TIMEOUT_S,
    FINGER_JOINT_COUNTS,
    NUM_JOINT_ANGLES,
    PALM_GRID_SHAPE,
    FingerTactile,
    HandTactile,
    SensorType,
    TactileGloveReader,
    finger_joint_slice,
    list_serial_ports,
    palm_to_grid,
    validate_joint_angles,
)

NUM_FINGERS = 5
FINGER_SHAPE = (2, 2)
CALIB_VERSION = 5
# 姿态匹配模式："nn" 最近邻 | "knn" 高斯加权
AtlasMode = str
# 姿态代理："pressure_sum" 四路压力和 | "joint" 独立关节角
PoseProxyMode = str

POSE_MODE_PRESSURE_SUM: PoseProxyMode = "pressure_sum"
POSE_MODE_JOINT: PoseProxyMode = "joint"
POSE_DIM_PRESSURE_SUM = NUM_FINGERS
POSE_DIM_JOINT = NUM_JOINT_ANGLES


def pose_dim(mode: PoseProxyMode) -> int:
    if mode == POSE_MODE_JOINT:
        return POSE_DIM_JOINT
    if mode == POSE_MODE_PRESSURE_SUM:
        return POSE_DIM_PRESSURE_SUM
    raise ValueError(f"未知 pose_mode: {mode!r}，可选 {POSE_MODE_PRESSURE_SUM!r} / {POSE_MODE_JOINT!r}")


def normalize_pose_mode(mode: Union[str, PoseProxyMode]) -> PoseProxyMode:
    m = str(mode).strip().lower()
    if m in (POSE_MODE_PRESSURE_SUM, "pressure", "bend", "sum"):
        return POSE_MODE_PRESSURE_SUM
    if m in (POSE_MODE_JOINT, "joints", "angles"):
        return POSE_MODE_JOINT
    raise ValueError(f"未知 pose_mode: {mode!r}")


def extract_pose_vector(hand: HandTactile, mode: PoseProxyMode) -> np.ndarray:
    """从 HandTactile 提取姿态向量，供姿态库匹配使用。"""
    mode = normalize_pose_mode(mode)
    if mode == POSE_MODE_PRESSURE_SUM:
        return np.array([f.bend for f in hand.fingers], dtype=np.float64)
    if hand.joint_angles is None:
        raise ValueError(
            f"pose_mode={POSE_MODE_JOINT!r} 需要 HandTactile.joint_angles ({POSE_DIM_JOINT},)。"
            "请从串口解析关节角后写入 TactileFrame.joint_angles，"
            "或调用 tactile_serial_reader.attach_joint_angles()。"
        )
    return validate_joint_angles(hand.joint_angles)


def _npz_scalar_str(arr: np.ndarray) -> str:
    return str(arr.item() if arr.ndim == 0 else arr[0])


@dataclass
class BendDecoupleCalib:
    """空手姿态库：每帧 (pose_D, 整手压力) 为模板。"""

    sensor_type: int
    finger_names: Tuple[str, ...]
    pose_mode: PoseProxyMode
    bend_center: np.ndarray  # (D,) 姿态归一化中心
    bend_scale: np.ndarray  # (D,) 姿态归一化尺度
    atlas_bends: np.ndarray  # (N, D) 姿态库；字段名保留兼容 v4
    atlas_fingers: np.ndarray  # (N, 5, 2, 2)
    atlas_palm: np.ndarray  # (N, 48)
    atlas_mode: str  # "nn" 最近邻单帧；"knn" 高斯加权
    knn_sigma: float  # 仅 knn
    knn_k: int  # 仅 knn；0=全部


def _hand_to_arrays(
    hand: HandTactile,
    pose_mode: PoseProxyMode,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose = extract_pose_vector(hand, pose_mode)
    pressures = np.stack([f.pressure for f in hand.fingers], axis=0)
    return pose, pressures, hand.palm.copy()


def _clone_hand(
    hand: HandTactile,
    *,
    pressures: Optional[np.ndarray] = None,
    palm: Optional[np.ndarray] = None,
) -> HandTactile:
    fingers: List[FingerTactile] = []
    for i, f in enumerate(hand.fingers):
        p = pressures[i] if pressures is not None else f.pressure.copy()
        fingers.append(FingerTactile(name=f.name, pressure=p, bend=f.bend))
    pal = palm if palm is not None else hand.palm.copy()
    return HandTactile(
        sensor_type=hand.sensor_type,
        fingers=fingers,
        palm=pal,
        joint_angles=None if hand.joint_angles is None else hand.joint_angles.copy(),
    )


def _normalize_bends(bends: np.ndarray, calib: BendDecoupleCalib) -> np.ndarray:
    b = np.asarray(bends, dtype=np.float64)
    c = calib.bend_center
    s = np.maximum(calib.bend_scale, 1e-6)
    return (b - c) / s


def _bend_dist2(query_bends: np.ndarray, calib: BendDecoupleCalib) -> np.ndarray:
    """query (D,) -> 与姿态库每帧的归一化欧氏距离平方 (N,)。"""
    q = _normalize_bends(query_bends, calib)
    a = _normalize_bends(calib.atlas_bends, calib)
    diff = a - q
    return np.sum(diff * diff, axis=1)


def nearest_atlas_index(bends: np.ndarray, calib: BendDecoupleCalib) -> int:
    return int(np.argmin(_bend_dist2(bends, calib)))


def _atlas_weights_knn(query_bends: np.ndarray, calib: BendDecoupleCalib) -> np.ndarray:
    dist2 = _bend_dist2(query_bends, calib)
    sigma = max(float(calib.knn_sigma), 1e-6)
    w = np.exp(-0.5 * dist2 / (sigma * sigma))

    k = int(calib.knn_k)
    if k > 0 and k < w.size:
        idx = np.argpartition(w, -k)[-k:]
        mask = np.zeros_like(w, dtype=bool)
        mask[idx] = True
        w = np.where(mask, w, 0.0)

    s = w.sum()
    if s <= 1e-12:
        j = int(np.argmin(dist2))
        out = np.zeros_like(w)
        out[j] = 1.0
        return out
    return w / s


def predict_template(
    bends: np.ndarray,
    calib: BendDecoupleCalib,
) -> Tuple[np.ndarray, np.ndarray]:
    """预测该 bend 下空手压力模板：(5,2,2), (48,)"""
    if calib.atlas_mode == "nn":
        j = nearest_atlas_index(bends, calib)
        return calib.atlas_fingers[j].copy(), calib.atlas_palm[j].copy()

    w = _atlas_weights_knn(bends, calib)
    fingers = np.tensordot(w, calib.atlas_fingers, axes=(0, 0))
    palm = np.dot(w, calib.atlas_palm)
    return fingers, palm


def hand_net_signal(hand: HandTactile, calib: BendDecoupleCalib) -> HandTactile:
    """解耦输出：原始 − 最近邻（或加权）模板。"""
    pose, _, _ = _hand_to_arrays(hand, calib.pose_mode)
    pred_f, pred_p = predict_template(pose, calib)
    pressures = np.stack([f.pressure for f in hand.fingers], axis=0) - pred_f
    palm = hand.palm - pred_p
    return _clone_hand(hand, pressures=pressures, palm=palm)


def decouple_hand(hand: HandTactile, calib: BendDecoupleCalib) -> HandTactile:
    """与 hand_net_signal 相同（1-NN 无额外偏置）。"""
    if int(hand.sensor_type) != calib.sensor_type:
        raise ValueError(
            f"手型不一致: 当前 {hand.sensor_type.name}，标定为 sensor_type={calib.sensor_type}"
        )
    return hand_net_signal(hand, calib)


def _estimate_knn_sigma(bends_norm: np.ndarray, *, k_neighbor: int = 5) -> float:
    n = bends_norm.shape[0]
    if n < 3:
        return 1.0
    k = min(k_neighbor + 1, n)
    dists: List[float] = []
    for i in range(n):
        d = np.linalg.norm(bends_norm - bends_norm[i], axis=1)
        d[i] = np.inf
        dists.append(float(np.partition(d, k - 1)[k - 1]))
    return max(float(np.median(dists)) * 1.5, 0.05)


def _subsample_indices(n: int, max_templates: int) -> np.ndarray:
    """max_templates<=0 表示保留全部帧。"""
    if max_templates <= 0 or n <= max_templates:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, max_templates, dtype=np.int64)


def build_calib_from_sweep(
    poses: np.ndarray,
    pressures: np.ndarray,
    palm: np.ndarray,
    *,
    sensor_type: int,
    finger_names: Sequence[str],
    pose_mode: PoseProxyMode = POSE_MODE_PRESSURE_SUM,
    atlas_mode: str = "nn",
    knn_sigma: Optional[float] = None,
    knn_k: int = 0,
    max_templates: int = 0,
    k_neighbor_sigma: int = 5,
) -> BendDecoupleCalib:
    pose_mode = normalize_pose_mode(pose_mode)
    dim = pose_dim(pose_mode)
    n = poses.shape[0]
    if n < 20:
        raise ValueError(f"标定帧数过少 ({n})，建议慢速扫掠至少 3–5 秒")
    if poses.shape != (n, dim):
        raise ValueError(f"poses 期望 {(n, dim)}（pose_mode={pose_mode}），得到 {poses.shape}")
    if pressures.shape != (n, NUM_FINGERS, *FINGER_SHAPE):
        raise ValueError(f"pressures 期望 {(n, NUM_FINGERS, *FINGER_SHAPE)}，得到 {pressures.shape}")
    if palm.shape != (n, 48):
        raise ValueError(f"palm 期望 {(n, 48)}，得到 {palm.shape}")

    names = tuple(str(x) for x in finger_names)
    if len(names) != NUM_FINGERS:
        raise ValueError(f"需要 {NUM_FINGERS} 个指名，得到 {len(names)}")

    idx = _subsample_indices(n, max_templates)
    if idx.size < n:
        print(f"姿态库下采样: {n} -> {idx.size} 帧（建议 nn 模式用 --max-templates 0 保留全帧）")

    atlas_bends = poses[idx].astype(np.float64)
    atlas_fingers = pressures[idx].astype(np.float64)
    atlas_palm = palm[idx].astype(np.float64)

    bend_center = np.median(atlas_bends, axis=0)
    bend_scale = np.std(atlas_bends, axis=0)
    bend_scale = np.maximum(bend_scale, 1.0)

    mode: AtlasMode = atlas_mode if atlas_mode in ("nn", "knn") else "nn"
    sigma = 0.0
    if mode == "knn":
        tmp = BendDecoupleCalib(
            sensor_type=int(sensor_type),
            finger_names=names,
            pose_mode=pose_mode,
            bend_center=bend_center,
            bend_scale=bend_scale,
            atlas_bends=atlas_bends,
            atlas_fingers=atlas_fingers,
            atlas_palm=atlas_palm,
            atlas_mode="knn",
            knn_sigma=1.0,
            knn_k=int(knn_k),
        )
        bn = _normalize_bends(atlas_bends, tmp)
        sigma = float(knn_sigma) if knn_sigma is not None and knn_sigma > 0 else _estimate_knn_sigma(bn, k_neighbor=k_neighbor_sigma)

    return BendDecoupleCalib(
        sensor_type=int(sensor_type),
        finger_names=names,
        pose_mode=pose_mode,
        bend_center=bend_center,
        bend_scale=bend_scale,
        atlas_bends=atlas_bends,
        atlas_fingers=atlas_fingers,
        atlas_palm=atlas_palm,
        atlas_mode=mode,
        knn_sigma=sigma,
        knn_k=int(knn_k),
    )


def save_calib(calib: BendDecoupleCalib, path: Union[str, Path]) -> None:
    np.savez_compressed(
        Path(path),
        calib_version=np.int32(CALIB_VERSION),
        sensor_type=np.int32(calib.sensor_type),
        finger_names=np.array(calib.finger_names, dtype=object),
        pose_mode=np.array(calib.pose_mode),
        bend_center=calib.bend_center,
        bend_scale=calib.bend_scale,
        atlas_bends=calib.atlas_bends,
        atlas_fingers=calib.atlas_fingers,
        atlas_palm=calib.atlas_palm,
        atlas_mode=np.array(calib.atlas_mode),
        knn_sigma=np.float64(calib.knn_sigma),
        knn_k=np.int32(calib.knn_k),
    )


def load_calib(path: Union[str, Path]) -> BendDecoupleCalib:
    d = np.load(Path(path), allow_pickle=True)
    ver = int(d["calib_version"]) if "calib_version" in d.files else 0
    if ver not in (4, CALIB_VERSION):
        raise ValueError(
            f"标定文件 version={ver}，当前支持 version=4/5。"
            "请重新执行: python curved_decouple_simple.py build --sweep ... --out ..."
        )
    names = tuple(str(x) for x in d["finger_names"].tolist())
    mode = str(d["atlas_mode"].item() if d["atlas_mode"].ndim == 0 else d["atlas_mode"][0])
    if mode not in ("nn", "knn"):
        mode = "nn"
    if "pose_mode" in d.files:
        pose_mode = normalize_pose_mode(_npz_scalar_str(d["pose_mode"]))
    else:
        pose_mode = POSE_MODE_PRESSURE_SUM
    atlas_bends = d["atlas_bends"].astype(np.float64)
    expected_dim = pose_dim(pose_mode)
    if atlas_bends.ndim != 2 or atlas_bends.shape[1] != expected_dim:
        raise ValueError(
            f"atlas_bends 形状 {atlas_bends.shape} 与 pose_mode={pose_mode}（D={expected_dim}）不一致"
        )
    return BendDecoupleCalib(
        sensor_type=int(d["sensor_type"]),
        finger_names=names,
        pose_mode=pose_mode,
        bend_center=d["bend_center"].astype(np.float64),
        bend_scale=d["bend_scale"].astype(np.float64),
        atlas_bends=atlas_bends,
        atlas_fingers=d["atlas_fingers"].astype(np.float64),
        atlas_palm=d["atlas_palm"].astype(np.float64),
        atlas_mode=mode,  # type: ignore[arg-type]
        knn_sigma=float(d["knn_sigma"]),
        knn_k=int(d["knn_k"]),
    )


def sweep_pose_mode(sweep: dict) -> PoseProxyMode:
    if "pose_mode" in sweep:
        return normalize_pose_mode(_npz_scalar_str(sweep["pose_mode"]))
    bends = sweep.get("bends")
    if bends is not None and bends.ndim == 2 and bends.shape[1] == POSE_DIM_JOINT:
        return POSE_MODE_JOINT
    return POSE_MODE_PRESSURE_SUM


def sweep_poses(sweep: dict) -> np.ndarray:
    if "poses" in sweep:
        return np.asarray(sweep["poses"], dtype=np.float64)
    if "bends" in sweep:
        return np.asarray(sweep["bends"], dtype=np.float64)
    raise KeyError("扫掠文件需包含 poses 或 bends 字段")


def save_sweep(
    path: Union[str, Path],
    *,
    poses: np.ndarray,
    pressures: np.ndarray,
    palm: np.ndarray,
    sensor_type: int,
    finger_names: Sequence[str],
    pose_mode: PoseProxyMode = POSE_MODE_PRESSURE_SUM,
    timestamps: Optional[np.ndarray] = None,
) -> None:
    pose_mode = normalize_pose_mode(pose_mode)
    poses = np.asarray(poses, dtype=np.float64)
    dim = pose_dim(pose_mode)
    if poses.ndim != 2 or poses.shape[1] != dim:
        raise ValueError(f"poses 期望 (*, {dim})，得到 {poses.shape}")
    kw: dict = dict(
        poses=poses,
        pose_mode=np.array(pose_mode),
        pressures=pressures.astype(np.float64),
        palm=palm.astype(np.float64),
        sensor_type=np.int32(sensor_type),
        finger_names=np.array(finger_names, dtype=object),
    )
    if pose_mode == POSE_MODE_PRESSURE_SUM:
        kw["bends"] = poses
    if timestamps is not None:
        kw["timestamps"] = timestamps.astype(np.float64)
    np.savez_compressed(Path(path), **kw)


def load_sweep(path: Union[str, Path]) -> dict:
    d = np.load(Path(path), allow_pickle=True)
    return {k: d[k] for k in d.files}


def build_calib_from_sweep_file(
    sweep_path: Union[str, Path],
    calib_path: Union[str, Path],
    *,
    atlas_mode: str = "nn",
    knn_sigma: Optional[float] = None,
    knn_k: int = 0,
    max_templates: int = 0,
) -> BendDecoupleCalib:
    sweep = load_sweep(sweep_path)
    names = tuple(str(x) for x in sweep["finger_names"].tolist())
    pose_mode = sweep_pose_mode(sweep)
    calib = build_calib_from_sweep(
        sweep_poses(sweep),
        sweep["pressures"],
        sweep["palm"],
        sensor_type=int(sweep["sensor_type"]),
        finger_names=names,
        pose_mode=pose_mode,
        atlas_mode=atlas_mode,
        knn_sigma=knn_sigma,
        knn_k=knn_k,
        max_templates=max_templates,
    )
    save_calib(calib, calib_path)
    return calib


def record_sweep(
    port: str,
    baudrate: int,
    out_path: Union[str, Path],
    *,
    duration_s: float = 12.0,
    prompt: bool = True,
    pose_mode: PoseProxyMode = POSE_MODE_PRESSURE_SUM,
) -> Path:
    out_path = Path(out_path)
    pose_mode = normalize_pose_mode(pose_mode)
    poses_list: List[np.ndarray] = []
    pressures_list: List[np.ndarray] = []
    palm_list: List[np.ndarray] = []
    ts_list: List[float] = []
    finger_names: Optional[Tuple[str, ...]] = None
    sensor_type: Optional[int] = None

    pose_hint = (
        "四路压力和（5 维）"
        if pose_mode == POSE_MODE_PRESSURE_SUM
        else f"独立关节角（{POSE_DIM_JOINT} 维：拇指{FINGER_JOINT_COUNTS[0]} + 其余各{FINGER_JOINT_COUNTS[1]}）"
    )
    if prompt:
        print(
            "【整手握拳标定】\n"
            "  1. 手放松伸直，不接触物体\n"
            "  2. 采集期间缓慢握拳至最紧，再缓慢伸直（建议往返一次）\n"
            f"  3. 时长 {duration_s:.0f} 秒\n"
            f"  4. 姿态代理: {pose_mode}（{pose_hint}）\n"
        )
        input("按 Enter 开始…")

    t0 = time.time()
    require_solved = pose_mode == POSE_MODE_JOINT
    if require_solved:
        print(
            "等待 SOLVED 就绪（TOUCH/SOLVED 分行或同行均可；"
            f"最长等待 {DEFAULT_SOLVED_WAIT_TIMEOUT_S:.0f}s；跳过含 MISS 的帧）…"
        )
    with TactileGloveReader(
        port=port,
        baudrate=baudrate,
        require_complete_solved=require_solved,
    ) as reader:
        while time.time() - t0 < duration_s:
            frame = reader.read_frame(timeout_s=2.0)
            hand = frame.decode()
            if finger_names is None:
                finger_names = tuple(f.name for f in hand.fingers)
                sensor_type = int(hand.sensor_type)
            pose, p, pal = _hand_to_arrays(hand, pose_mode)
            poses_list.append(pose)
            pressures_list.append(p)
            palm_list.append(pal)
            ts_list.append(frame.timestamp)

    save_sweep(
        out_path,
        poses=np.stack(poses_list),
        pressures=np.stack(pressures_list),
        palm=np.stack(palm_list),
        sensor_type=int(sensor_type),
        finger_names=finger_names or (),
        pose_mode=pose_mode,
        timestamps=np.array(ts_list),
    )
    print(f"已保存 {len(poses_list)} 帧 -> {out_path}  pose_mode={pose_mode}")
    return out_path


def _hand_from_sweep_row(
    sweep: dict,
    index: int,
    calib: BendDecoupleCalib,
) -> HandTactile:
    pose = sweep_poses(sweep)[index]
    fingers = [
        FingerTactile(
            name=calib.finger_names[f],
            pressure=sweep["pressures"][index, f],
            bend=float(pose[f]) if calib.pose_mode == POSE_MODE_PRESSURE_SUM else _finger_bend_proxy(
                sweep["pressures"][index, f]
            ),
        )
        for f in range(NUM_FINGERS)
    ]
    joint_angles = None
    if calib.pose_mode == POSE_MODE_JOINT:
        joint_angles = validate_joint_angles(pose)
    return HandTactile(
        sensor_type=SensorType(calib.sensor_type),
        fingers=fingers,
        palm=sweep["palm"][index],
        joint_angles=joint_angles,
    )


def _finger_bend_proxy(finger_pressure: np.ndarray) -> float:
    return float(np.sum(finger_pressure))


def evaluate_on_sweep(
    sweep: dict,
    calib: BendDecoupleCalib,
    *,
    full_atlas_expected: bool = True,
) -> dict:
    n = int(sweep_poses(sweep).shape[0])
    poses = sweep_poses(sweep)
    res_max: List[float] = []
    palm_res: List[float] = []
    self_hits = 0
    for i in range(n):
        hand = _hand_from_sweep_row(sweep, i, calib)
        dec = decouple_hand(hand, calib)
        res_max.append(float(np.max(np.abs(np.stack([f.pressure for f in dec.fingers])))))
        palm_res.append(float(np.max(np.abs(dec.palm))))
        if full_atlas_expected and calib.atlas_bends.shape[0] == n:
            j = nearest_atlas_index(poses[i], calib)
            if j == i:
                self_hits += 1

    out = {
        "frames": n,
        "pose_mode": calib.pose_mode,
        "pose_dim": pose_dim(calib.pose_mode),
        "atlas_templates": int(calib.atlas_bends.shape[0]),
        "atlas_mode": calib.atlas_mode,
        "finger_residual_max_mean": float(np.mean(res_max)),
        "finger_residual_max_p95": float(np.percentile(res_max, 95)),
        "finger_residual_max_p99": float(np.percentile(res_max, 99)),
        "palm_residual_max_p95": float(np.percentile(palm_res, 95)),
    }
    if full_atlas_expected and calib.atlas_bends.shape[0] == n:
        out["nearest_self_hit_rate"] = float(self_hits / n)
    return out


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError as e:
        raise SystemExit("可视化需要 matplotlib: pip install matplotlib") from e
    return plt, GridSpec


def _resolve_colormap(name: str) -> str:
    """兼容旧版 matplotlib（无 turbo 等），不可用时回退到高对比色图。"""
    plt, _ = _import_matplotlib()
    try:
        plt.get_cmap(name)
        return name
    except ValueError:
        for fallback in ("plasma", "hot", "gist_ncar", "inferno"):
            try:
                plt.get_cmap(fallback)
                print(f"提示: 色图 '{name}' 不可用，已改用 '{fallback}'")
                return fallback
            except ValueError:
                continue
        return "inferno"


def _pose_finger_label(hand: HandTactile, calib: BendDecoupleCalib, finger_index: int) -> str:
    if calib.pose_mode == POSE_MODE_PRESSURE_SUM:
        return f"Σ:{hand.fingers[finger_index].bend:.0f}"
    if hand.joint_angles is None:
        return "J:—"
    sl = finger_joint_slice(finger_index)
    vals = hand.joint_angles[sl]
    return "J:" + ",".join(f"{v:.0f}" for v in vals)


class DecoupleCompareViewer:
    def __init__(
        self,
        calib: BendDecoupleCalib,
        *,
        cmap: str = "plasma",
        raw_vmin: float = 0.0,
        raw_vmax: float = 500.0,
        resid_vmin: float = 0.0,
        resid_vmax: float = 50.0,
    ):
        plt, GridSpec = _import_matplotlib()
        cmap = _resolve_colormap(cmap)
        self._cmap = cmap
        self._raw_vmin = raw_vmin
        self._raw_vmax = raw_vmax
        self._resid_vmin = resid_vmin
        self._resid_vmax = resid_vmax
        self._plt = plt
        self.fig = plt.figure(figsize=(15, 9))
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("弯曲解耦对比")

        gs = GridSpec(4, 6, figure=self.fig, height_ratios=[0.12, 1.2, 1.0, 1.0], hspace=0.45, wspace=0.32)
        ax_lb = self.fig.add_subplot(gs[0, :3])
        ax_la = self.fig.add_subplot(gs[0, 3:])
        ax_lb.set_axis_off()
        ax_la.set_axis_off()
        ax_lb.set_title(f"解耦前（原始读数，{raw_vmin:.0f}–{raw_vmax:.0f}）", fontsize=11, loc="left")
        ax_la.set_title(f"解耦后（残差，{resid_vmin:.0f}–{resid_vmax:.0f}）", fontsize=11, loc="left")

        self._ax_palm_b = self.fig.add_subplot(gs[1, :3])
        self._ax_palm_a = self.fig.add_subplot(gs[1, 3:])
        z = np.zeros(PALM_GRID_SHAPE)
        self._im_palm_b = self._ax_palm_b.imshow(
            z, vmin=raw_vmin, vmax=raw_vmax, cmap=cmap, aspect="auto"
        )
        self._im_palm_a = self._ax_palm_a.imshow(
            z, vmin=resid_vmin, vmax=resid_vmax, cmap=cmap, aspect="auto"
        )
        self._ax_palm_b.set_title("手掌")
        self._ax_palm_a.set_title("手掌")

        self._im_f_b: List[Any] = []
        self._im_f_a: List[Any] = []
        self._bend_txt: List[Any] = []
        for i in range(NUM_FINGERS):
            ax_b = self.fig.add_subplot(gs[2, i])
            ax_a = self.fig.add_subplot(gs[3, i])
            im_b = ax_b.imshow(
                np.zeros(FINGER_SHAPE), vmin=raw_vmin, vmax=raw_vmax, cmap=cmap, aspect="equal"
            )
            im_a = ax_a.imshow(
                np.zeros(FINGER_SHAPE), vmin=resid_vmin, vmax=resid_vmax, cmap=cmap, aspect="equal"
            )
            self._im_f_b.append(im_b)
            self._im_f_a.append(im_a)
            name = calib.finger_names[i] if i < len(calib.finger_names) else str(i)
            ax_b.set_title(name, fontsize=9)
            ax_a.set_title(name, fontsize=9)
            self._bend_txt.append(ax_a.text(0.5, -0.15, "", transform=ax_a.transAxes, ha="center", fontsize=8))
            for ax in (ax_b, ax_a):
                ax.set_xticks([])
                ax.set_yticks([])

        self._calib = calib
        self._stats = self.fig.text(0.5, 0.02, "", ha="center", fontsize=9)
        self._title = self.fig.suptitle("", fontsize=11)
        self._plt.ion()
        self._plt.show(block=False)

    def update(
        self,
        hand_raw: HandTactile,
        hand_dec: HandTactile,
        *,
        metrics_line: str = "",
    ) -> None:
        self._im_palm_b.set_data(palm_to_grid(hand_raw.palm))
        self._im_palm_a.set_data(palm_to_grid(hand_dec.palm))
        for i in range(NUM_FINGERS):
            self._im_f_b[i].set_data(hand_raw.fingers[i].pressure)
            self._im_f_a[i].set_data(hand_dec.fingers[i].pressure)
            self._bend_txt[i].set_text(_pose_finger_label(hand_raw, self._calib, i))
        side = "左" if hand_raw.sensor_type == SensorType.LEFT_HAND else "右"
        self._title.set_text(
            f"{side}手  {self._calib.pose_mode}  {self._calib.atlas_mode.upper()} 解耦"
        )
        self._stats.set_text(metrics_line)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self._plt.pause(0.001)

    def close(self) -> None:
        self._plt.close(self.fig)


def _metrics_line(hand_raw: HandTactile, hand_dec: HandTactile, calib: BendDecoupleCalib) -> str:
    raw_m = float(np.max(np.stack([f.pressure for f in hand_raw.fingers])))
    dec_m = float(np.max(np.abs(np.stack([f.pressure for f in hand_dec.fingers]))))
    return (
        f"指尖 原始max={raw_m:.0f}  |残差|max={dec_m:.1f}  "
        f"{calib.pose_mode} {calib.atlas_mode} 库={calib.atlas_bends.shape[0]}帧"
    )


def run_viz_live(
    port: str,
    baudrate: int,
    calib_path: Union[str, Path],
) -> None:
    calib = load_calib(calib_path)
    viewer = DecoupleCompareViewer(calib)
    require_solved = calib.pose_mode == POSE_MODE_JOINT
    if require_solved:
        print(
            "等待 SOLVED 就绪（TOUCH/SOLVED 分行或同行均可；"
            f"最长等待 {DEFAULT_SOLVED_WAIT_TIMEOUT_S:.0f}s；跳过含 MISS 的帧）…"
        )
    try:
        with TactileGloveReader(
            port=port,
            baudrate=baudrate,
            require_complete_solved=require_solved,
        ) as reader:
            print(
                f"实时解耦: {port}，{calib.pose_mode} {calib.atlas_mode}，"
                f"库 {calib.atlas_bends.shape[0]} 帧"
            )
            for frame in reader.frames():
                if viewer.fig.number not in viewer._plt.get_fignums():
                    break
                hand = frame.decode()
                dec = decouple_hand(hand, calib)
                viewer.update(hand, dec, metrics_line=_metrics_line(hand, dec, calib))
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


def run_viz_replay(
    sweep_path: Union[str, Path],
    calib_path: Union[str, Path],
    *,
    speed: float = 1.0,
) -> None:
    sweep = load_sweep(sweep_path)
    calib = load_calib(calib_path)
    stats = evaluate_on_sweep(sweep, calib)
    print("扫掠自检:", stats)

    viewer = DecoupleCompareViewer(calib)
    ts = sweep.get("timestamps")
    n = int(sweep_poses(sweep).shape[0])
    try:
        for i in range(n):
            if viewer.fig.number not in viewer._plt.get_fignums():
                break
            hand = _hand_from_sweep_row(sweep, i, calib)
            dec = decouple_hand(hand, calib)
            line = _metrics_line(hand, dec, calib)
            if i == 0 or i == n - 1:
                line += f"  | p95={stats['finger_residual_max_p95']:.1f}"
            viewer.update(hand, dec, metrics_line=line)
            if ts is not None and i + 1 < n and speed > 0:
                dt = float(ts[i + 1] - ts[i]) / speed
                if dt > 0:
                    time.sleep(min(dt, 0.2))
            else:
                time.sleep(0.03)
        print("回放结束，关闭窗口退出")
        viewer._plt.show(block=True)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


def main() -> None:
    ports = list_serial_ports()
    parser = argparse.ArgumentParser(
        description="整手握拳弯曲解耦（默认全帧 1-NN）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 1. 采集：空手慢握拳扫掠（默认 pressure_sum，约 12 秒）
    python curved_decouple_simple.py record --port COM3 --out bend_sweep.npz

    # 1b. 关节传感器就绪后：用 14 维关节角作姿态代理
    python curved_decouple_simple.py record --port COM3 --pose-mode joint --out joint_sweep.npz

    # 2. 建库：全帧 1-NN（默认 --method nn --max-templates 0）
    python curved_decouple_simple.py build --sweep bend_sweep.npz --out bend_decouple_calib.npz

    # 3. 回放验证：看解耦前后对比 + 扫掠自检
    python curved_decouple_simple.py viz --calib bend_decouple_calib.npz --sweep bend_sweep.npz

    # 4. 实时对比：戴同一副手套、用同一次 build 的 calib
    python curved_decouple_simple.py viz --calib bend_decouple_calib.npz --port COM3
        """,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rec = sub.add_parser("record", help="串口采集整手握拳扫掠")
    p_rec.add_argument("--port", default=ports[0] if ports else "COM3")
    p_rec.add_argument("--baudrate", type=int, default=921600)
    p_rec.add_argument("--out", default="bend_sweep.npz")
    p_rec.add_argument("--duration", type=float, default=12.0)
    p_rec.add_argument("--no-prompt", action="store_true")
    p_rec.add_argument(
        "--pose-mode",
        choices=(POSE_MODE_PRESSURE_SUM, POSE_MODE_JOINT),
        default=POSE_MODE_JOINT,
        help="姿态代理：pressure_sum=四路压力和(5维)；joint=独立关节角(14维)",
    )

    p_build = sub.add_parser("build", help="构建姿态库（默认 nn + 全帧）")
    p_build.add_argument("--sweep", required=True)
    p_build.add_argument("--out", default="bend_decouple_calib.npz")
    p_build.add_argument(
        "--method",
        choices=("nn", "knn"),
        default="knn",
        help="nn=最近邻单帧（推荐）；knn=高斯加权",
    )
    p_build.add_argument("--sigma", type=float, default=None, help="仅 knn：高斯 σ")
    p_build.add_argument("--k", type=int, default=0, help="仅 knn：前 k 模板")
    p_build.add_argument(
        "--max-templates",
        type=int,
        default=0,
        help="姿态库帧数上限，0=扫掠全帧（推荐）",
    )

    p_viz = sub.add_parser("viz", help="解耦前后对比")
    p_viz.add_argument("--calib", required=True)
    p_viz.add_argument("--sweep", default=None)
    p_viz.add_argument("--port", default=ports[0] if ports else "COM3")
    p_viz.add_argument("--baudrate", type=int, default=921600)
    p_viz.add_argument("--speed", type=float, default=1.0)

    args = parser.parse_args()

    if args.cmd == "record":
        record_sweep(
            args.port,
            args.baudrate,
            args.out,
            duration_s=args.duration,
            prompt=not args.no_prompt,
            pose_mode=args.pose_mode,
        )
    elif args.cmd == "build":
        calib = build_calib_from_sweep_file(
            args.sweep,
            args.out,
            atlas_mode=args.method,
            knn_sigma=args.sigma,
            knn_k=args.k,
            max_templates=args.max_templates,
        )
        ev = evaluate_on_sweep(load_sweep(args.sweep), calib)
        print(
            f"已写入 -> {args.out}  pose_mode={calib.pose_mode}  "
            f"模式={calib.atlas_mode}  库={calib.atlas_bends.shape[0]}帧"
        )
        print("扫掠自检:", ev)
    elif args.cmd == "viz":
        if args.sweep:
            run_viz_replay(args.sweep, args.calib, speed=args.speed)
        else:
            run_viz_live(args.port, args.baudrate, args.calib)
    else:
        parser.error(f"未知命令: {args.cmd}")


if __name__ == "__main__":
    main()
