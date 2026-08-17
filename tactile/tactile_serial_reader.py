#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
左手触觉手套串口读取（68 路压阻 AD）

协议（ASCII 文本，一行/帧，以 \\n 或 \\r\\n 结尾）：
  TOUCH seq=33711 thumb=71,71,69,71 index=74,72,77,74 middle=... ring=... little=... palm=...
  可选同行追加 SOLVED：
  SOLVED seq=3511 status=0 flags=0x00000008 angles_mdeg=... quat_1e4=...

  angles_mdeg 共 19 个数；弯曲解耦取其中 14 个屈伸角（跳过各指 MCP 侧摆）。
  TOUCH 与 SOLVED 可在同一行，也可分两行（先 TOUCH 后 SOLVED）；分行时按最近 TOUCH 配对。
  IMU 解算预热期间 angles_mdeg / quat_1e4 可能出现 MISS；require_complete_solved=True 时
  跳过含 MISS 或无完整 SOLVED 的包，直至就绪后再解包输出。

触觉顺序：thumb×4, index×4, middle×4, ring×4, little×4, palm F0–F47。
串口 AD 值经 ExpDec1 标定为压强 (kPa)：y = A1·exp(-x/t1) + y0。
空间布局严格按手册行/列标表：pressure_to_hand_grid() / hand_to_unified_grid()。

可视化 CLI 见 tactile_serial_view.py（2D 脉冲点、MANO 3D）；本模块主要供 import。
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Generator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import serial
except ImportError as e:
    raise SystemExit("请先安装: pip install pyserial") from e


ASCII_LINE_PREFIX = "TOUCH"
SOLVED_LINE_PREFIX = "SOLVED"
SOLVED_MISS_TOKEN = "MISS"
ASCII_TOUCH_RE = re.compile(
    r"^TOUCH\s+"
    r"seq=(\d+)\s+"
    r"thumb=([\d,]+)\s+"
    r"index=([\d,]+)\s+"
    r"middle=([\d,]+)\s+"
    r"ring=([\d,]+)\s+"
    r"little=([\d,]+)\s+"
    r"palm=([\d,]+)"
)
# 兼容旧名
ASCII_LINE_RE = ASCII_TOUCH_RE
ASCII_SOLVED_RE = re.compile(
    r"SOLVED\s+"
    r"seq=(\d+)\s+"
    r"status=(\d+)\s+"
    r"flags=(0x[0-9a-fA-F]+|\d+)\s+"
    r"angles_mdeg=([-\w,]+)"
    r"(?:\s+quat_1e4=([-\w,]+))?"
)
NUM_ANGLES_MDEG = 19
# angles_mdeg 原始下标 → 弯曲解耦 14 维（拇指→小指，仅屈伸）
# 四指每指 4 槽 [MCP屈, MCP侧摆, PIP屈, DIP屈]；拇指取 MCP屈(16)、IP屈(18)
ANGLES_MDEG_FLEX_INDICES: Tuple[int, ...] = (
    16,
    18,  # thumb
    0,
    2,
    3,  # index
    4,
    6,
    7,  # middle
    8,
    10,
    11,  # ring
    12,
    14,
    15,  # little
)
NUM_TACTILE = 68
NUM_FINGERS = 5
FINGER_POINTS = 4
PALM_POINTS = 48

# 物理坐标范围（手册行标/列标）
PHYS_ROW_MIN = 5
PHYS_ROW_MAX = 14
PHYS_COL_MIN = 7
PHYS_COL_MAX = 14

# 掌区 6 行×8 列，与包内 palm 行优先顺序一致
PALM_GRID_SHAPE = (6, 8)
FINGER_GRID_SHAPE = (2, 2)
UNIFIED_GRID_SHAPE = (10, 8)  # 行 5–14 × 列 7–14（手册坐标画布）

FINGER_NAMES: Tuple[str, ...] = ("thumb", "index", "middle", "ring", "little")

# 独立关节传感器布局：拇指 2 关节，食/中/无/小各 3 关节 → 共 14 维
FINGER_JOINT_COUNTS: Tuple[int, ...] = (2, 3, 3, 3, 3)
NUM_JOINT_ANGLES = sum(FINGER_JOINT_COUNTS)

# 等待首帧完整 SOLVED 的默认超时（IMU 预热 + 分行配对可能较慢）
DEFAULT_SOLVED_WAIT_TIMEOUT_S = 60.0

# 可视化色标（固定，不随帧变化）
VIZ_AD_MIN = 0.0
VIZ_AD_MAX = 200.0
VIZ_BASELINE_FRAMES = 10

# 传感点布局（像素坐标，x 向右、y 向下，与 assets/sensor_layout.png 一致）
_PKG_ROOT = Path(__file__).resolve().parent
if (_PKG_ROOT / "assets").is_dir():
    _PROJECT_ROOT = _PKG_ROOT
elif (_PKG_ROOT.parent / "assets").is_dir():
    _PROJECT_ROOT = _PKG_ROOT.parent
else:
    _PROJECT_ROOT = _PKG_ROOT
_ASSETS_DIR = _PROJECT_ROOT / "assets"
_SENSOR_LAYOUT_IMAGE = _ASSETS_DIR / "sensor_layout.png"
_SENSOR_LAYOUT_COORDS_NPY = _ASSETS_DIR / "sensor_layout_coords.npy"
_VIZ_CANVAS_SIZE = (990, 1024)  # width, height

# 可视化主题（黑底，与手册示意图一致）
_VIZ_THEME = {
    "fig_bg": "#0a0a0a",
    "card_bg": "#000000",
    "label": "#d4d4d4",
    "label_muted": "#888888",
    "title": "#f0f0f0",
    "accent": "#ff6b6b",
    "grid": "#333333",
    "point_edge": "#1a1a1a",
}
_VIZ_CMAP = "magma"
_VIZ_PULSE_PERIOD_S = 0.85
_VIZ_CORE_SIZE_PT2 = (25.0, 180.0)  # 内核 scatter 面积 (points^2)
_VIZ_PULSE_SIZE_PT2 = (80.0, 650.0)  # 外圈脉冲面积 (points^2)

# 包内每指 4 点顺序 A0,A1,A2,A3：左列(7) 上→下，右列(8) 上→下（与手册图一致）
_FINGER_COORD_TABLE: Tuple[Tuple[Tuple[int, int, str], ...], ...] = (
    # thumb A0–A3
    ((14, 7, "A0"), (13, 7, "A1"), (14, 8, "A2"), (13, 8, "A3")),
    # index B0–B3
    ((12, 7, "B0"), (11, 7, "B1"), (12, 8, "B2"), (11, 8, "B3")),
    # middle C0–C3
    ((10, 7, "C0"), (9, 7, "C1"), (10, 8, "C2"), (9, 8, "C3")),
    # ring D0–D3
    ((8, 7, "D0"), (7, 7, "D1"), (8, 8, "D2"), (7, 8, "D3")),
    # little E0–E3
    ((6, 7, "E0"), (5, 7, "E1"), (6, 8, "E2"), (5, 8, "E3")),
)


def _phys_to_grid(row: int, col: int) -> Tuple[int, int]:
    """手册 (行标, 列标) → 10×8 画布 (row, col)，行 14 在画布顶部。"""
    if not (PHYS_ROW_MIN <= row <= PHYS_ROW_MAX and PHYS_COL_MIN <= col <= PHYS_COL_MAX):
        raise ValueError(f"坐标 ({row}, {col}) 超出手册范围")
    return PHYS_ROW_MAX - row, col - PHYS_COL_MIN


def _build_packet_sensor_layout() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, Tuple[str, ...]
]:
    """生成包内 68 点 → 画布坐标与点名。"""
    grid_rows = np.zeros(NUM_TACTILE, dtype=np.int32)
    grid_cols = np.zeros(NUM_TACTILE, dtype=np.int32)
    phys_rows = np.zeros(NUM_TACTILE, dtype=np.int32)
    phys_cols = np.zeros(NUM_TACTILE, dtype=np.int32)
    labels: List[str] = []

    idx = 0
    for finger_pts in _FINGER_COORD_TABLE:
        for phys_r, phys_c, name in finger_pts:
            gr, gc = _phys_to_grid(phys_r, phys_c)
            grid_rows[idx] = gr
            grid_cols[idx] = gc
            phys_rows[idx] = phys_r
            phys_cols[idx] = phys_c
            labels.append(name)
            idx += 1

    for f_idx in range(PALM_POINTS):
        phys_c = 9 + f_idx // 8
        phys_r = 12 - (f_idx % 8)
        gr, gc = _phys_to_grid(phys_r, phys_c)
        grid_rows[idx] = gr
        grid_cols[idx] = gc
        phys_rows[idx] = phys_r
        phys_cols[idx] = phys_c
        labels.append(f"F{f_idx}")
        idx += 1

    return grid_rows, grid_cols, phys_rows, phys_cols, tuple(labels)


(
    PACKET_GRID_ROWS,
    PACKET_GRID_COLS,
    PACKET_PHYS_ROWS,
    PACKET_PHYS_COLS,
    PACKET_SENSOR_LABELS,
) = _build_packet_sensor_layout()

_FINGER_LABEL_EN = {
    "thumb": "Thumb",
    "index": "Index",
    "middle": "Middle",
    "ring": "Ring",
    "little": "Little",
}


class SensorType(IntEnum):
    """当前设备仅左手；保留枚举以兼容弯曲解耦标定文件。"""

    LEFT_HAND = 0x01
    RIGHT_HAND = 0x02


@dataclass(frozen=True)
class PressureCalibration:
    """AD → 压强 (kPa)，厂商 ExpDec1 拟合（R²≈0.99976）。"""

    y0: float = -25.20345
    A1: float = 24.87497
    t1: float = -1099.63275


DEFAULT_PRESSURE_CALIB = PressureCalibration()


def calibrate_ad_to_kpa(
    ad: np.ndarray,
    calib: PressureCalibration = DEFAULT_PRESSURE_CALIB,
) -> np.ndarray:
    """
    将 AD 标定为压强 (kPa)。

    y = A1 * exp(-x / t1) + y0；t1<0 时等价于 A1*exp(x/|t1|)+y0。
    负值截为 0。
    """
    x = np.asarray(ad, dtype=np.float64)
    kpa = calib.A1 * np.exp(-x / calib.t1) + calib.y0
    return np.maximum(kpa, 0.0)


def _comma_field_to_ints(field: str, expected: int, name: str) -> List[int]:
    parts = [p.strip() for p in field.split(",") if p.strip()]
    try:
        vals = [int(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"{name} 含非整数") from e
    if len(vals) != expected:
        raise ValueError(f"{name} 期望 {expected} 点，得到 {len(vals)}")
    return vals


def parse_ascii_touch_line(line: str) -> Tuple[int, np.ndarray]:
    """解析 ASCII TOUCH 段 → (序号, 68 路 AD)。可与同行 SOLVED 并存。"""
    text = line.strip()
    if not text.startswith(ASCII_LINE_PREFIX):
        raise ValueError("行首应为 TOUCH")
    match = ASCII_TOUCH_RE.match(text)
    if match is None:
        raise ValueError("格式不匹配 TOUCH seq=… thumb=… index=… middle=… ring=… little=… palm=…")

    sequence = int(match.group(1))
    ad_vals: List[int] = []
    expected = (FINGER_POINTS,) * NUM_FINGERS + (PALM_POINTS,)
    names = (*FINGER_NAMES, "palm")
    for i, (exp, fname) in enumerate(zip(expected, names), start=2):
        ad_vals.extend(_comma_field_to_ints(match.group(i), exp, fname))
    if len(ad_vals) != NUM_TACTILE:
        raise ValueError(f"总路数 {len(ad_vals)} != {NUM_TACTILE}")
    return sequence, np.asarray(ad_vals, dtype=np.float64)


def _parse_flags_value(raw: str) -> int:
    text = raw.strip().lower()
    if text.startswith("0x"):
        return int(text, 16)
    return int(text)


def _parse_numeric_or_miss_field(field: str, name: str, expected: int) -> Tuple[np.ndarray, bool]:
    """解析逗号分隔整数列；遇 MISS 记为 nan 并返回 has_miss=True。"""
    parts = [p.strip() for p in field.split(",") if p.strip()]
    if len(parts) != expected:
        raise ValueError(f"{name} 期望 {expected} 个数，得到 {len(parts)}")
    has_miss = False
    out = np.full(expected, np.nan, dtype=np.float64)
    for i, token in enumerate(parts):
        if token.upper() == SOLVED_MISS_TOKEN:
            has_miss = True
            continue
        try:
            out[i] = float(int(token))
        except ValueError as e:
            raise ValueError(f"{name}[{i}] 非整数/MISS: {token!r}") from e
    return out, has_miss


@dataclass(frozen=True)
class SolvedKinematics:
    """SOLVED 段运动学解（与 TOUCH 可同包）。"""

    sequence: int
    status: int
    flags: int
    angles_mdeg: np.ndarray  # (19,) 未就绪处为 nan
    angles_has_miss: bool
    quat_1e4: Optional[np.ndarray] = None  # (8,) 未就绪处为 nan
    quat_has_miss: bool = False

    @property
    def is_complete(self) -> bool:
        """angles_mdeg 与 quat_1e4（若存在）均无 MISS。"""
        if self.angles_has_miss:
            return False
        if self.quat_1e4 is not None and self.quat_has_miss:
            return False
        return True


def parse_ascii_solved_section(line: str) -> Optional[SolvedKinematics]:
    """从整行文本中解析 SOLVED 段；无则返回 None。"""
    match = ASCII_SOLVED_RE.search(line)
    if match is None:
        return None

    angles, angles_has_miss = _parse_numeric_or_miss_field(
        match.group(4), "angles_mdeg", NUM_ANGLES_MDEG
    )

    quat_raw = match.group(5)
    quat = None
    quat_has_miss = False
    if quat_raw is not None:
        quat, quat_has_miss = _parse_numeric_or_miss_field(quat_raw, "quat_1e4", 8)

    return SolvedKinematics(
        sequence=int(match.group(1)),
        status=int(match.group(2)),
        flags=_parse_flags_value(match.group(3)),
        angles_mdeg=angles,
        angles_has_miss=angles_has_miss,
        quat_1e4=quat,
        quat_has_miss=quat_has_miss,
    )


def flex_joint_angles_from_mdeg(angles_mdeg: np.ndarray) -> np.ndarray:
    """angles_mdeg (19,) → 弯曲解耦用 joint_angles (14,)，单位：度。要求无 MISS。"""
    raw = np.asarray(angles_mdeg, dtype=np.float64).reshape(-1)
    if raw.size != NUM_ANGLES_MDEG:
        raise ValueError(f"angles_mdeg 期望 {NUM_ANGLES_MDEG} 维，得到 {raw.size}")
    if np.isnan(raw).any():
        raise ValueError("angles_mdeg 含 MISS，关节角尚未就绪")
    flex_mdeg = raw[list(ANGLES_MDEG_FLEX_INDICES)]
    return flex_mdeg / 1000.0


def packet_finger_to_2x2(four: np.ndarray) -> np.ndarray:
    """包内四值 → 2×2：前两个在上行，后两个在下行。"""
    f = np.asarray(four, dtype=np.float64).reshape(-1)
    return np.array([[f[0], f[1]], [f[2], f[3]]], dtype=np.float64)


def _finger_2x2(four: np.ndarray) -> np.ndarray:
    """兼容旧名；与 packet_finger_to_2x2 相同。"""
    return packet_finger_to_2x2(four)


def finger_2x2_from_pressure(pressure: np.ndarray, finger_idx: int) -> np.ndarray:
    """按手册坐标表，从 68 路向量提取单指 2×2（与 pressure_to_hand_grid 一致）。"""
    p = np.asarray(pressure, dtype=np.float64).reshape(-1)
    base = finger_idx * FINGER_POINTS
    rows = PACKET_GRID_ROWS[base : base + FINGER_POINTS]
    row0 = int(rows.min())
    out = np.zeros(FINGER_GRID_SHAPE, dtype=np.float64)
    for k in range(FINGER_POINTS):
        idx = base + k
        lr = int(PACKET_GRID_ROWS[idx] - row0)
        lc = int(PACKET_GRID_COLS[idx])
        out[lr, lc] = p[idx]
    return out


def _finger_bend(four: np.ndarray) -> float:
    """无独立弯曲通道：用四通道压力和作姿态代理。"""
    return float(np.sum(four))


@dataclass
class FingerTactile:
    name: str
    pressure: np.ndarray  # (2, 2) kPa
    bend: float  # 四通道 kPa 之和（姿态代理）


@dataclass
class HandTactile:
    sensor_type: SensorType
    fingers: List[FingerTactile]
    palm: np.ndarray  # (48,) F0–F47，kPa，与手册表顺序一致
    joint_angles: Optional[np.ndarray] = None  # (14,) 度；来自 SOLVED angles_mdeg 屈伸角


def finger_joint_slice(finger_index: int) -> slice:
    """第 finger_index 指在 joint_angles 向量中的切片（拇指=0 … 小指=4）。"""
    if not (0 <= finger_index < NUM_FINGERS):
        raise IndexError(f"finger_index 需在 [0, {NUM_FINGERS})，得到 {finger_index}")
    start = sum(FINGER_JOINT_COUNTS[:finger_index])
    stop = start + FINGER_JOINT_COUNTS[finger_index]
    return slice(start, stop)


def validate_joint_angles(joint_angles: np.ndarray) -> np.ndarray:
    """校验并返回 (14,) float64 关节角向量。"""
    ja = np.asarray(joint_angles, dtype=np.float64).reshape(-1)
    if ja.size != NUM_JOINT_ANGLES:
        raise ValueError(
            f"joint_angles 期望 {NUM_JOINT_ANGLES} 维 "
            f"(拇指{FINGER_JOINT_COUNTS[0]} + 其余各{FINGER_JOINT_COUNTS[1]})，得到 {ja.size}"
        )
    return ja


def pack_joint_angles(
    thumb: Sequence[float],
    index: Sequence[float],
    middle: Sequence[float],
    ring: Sequence[float],
    little: Sequence[float],
) -> np.ndarray:
    """按指别拼装 (14,) 关节角（串口解析后的便捷入口）。"""
    parts = (thumb, index, middle, ring, little)
    if len(parts) != NUM_FINGERS:
        raise ValueError(f"需要 {NUM_FINGERS} 指关节数据")
    chunks: List[np.ndarray] = []
    for i, (name, vals) in enumerate(zip(FINGER_NAMES, parts)):
        v = np.asarray(vals, dtype=np.float64).reshape(-1)
        expect = FINGER_JOINT_COUNTS[i]
        if v.size != expect:
            raise ValueError(f"{name} 期望 {expect} 个关节角，得到 {v.size}")
        chunks.append(v)
    return np.concatenate(chunks)


def attach_joint_angles(hand: HandTactile, joint_angles: np.ndarray) -> HandTactile:
    """为 HandTactile 附着关节角，供弯曲解耦 joint 姿态模式使用。"""
    ja = validate_joint_angles(joint_angles)
    return HandTactile(
        sensor_type=hand.sensor_type,
        fingers=hand.fingers,
        palm=hand.palm,
        joint_angles=ja,
    )


def pressure_to_hand_grid(pressure: np.ndarray) -> np.ndarray:
    """按手册坐标表将包内 68 点填入 10×8 左手画布（行 14 在上，列 7 在左）。"""
    p = np.asarray(pressure, dtype=np.float64).reshape(-1)
    if p.size != NUM_TACTILE:
        raise ValueError(f"期望 {NUM_TACTILE} 路，得到 {p.size}")
    grid = np.zeros(UNIFIED_GRID_SHAPE, dtype=np.float64)
    grid[PACKET_GRID_ROWS, PACKET_GRID_COLS] = p
    return grid


def palm_to_grid(palm: np.ndarray, shape: Tuple[int, int] = PALM_GRID_SHAPE) -> np.ndarray:
    """掌 F0–F47 行优先 → 6×8。"""
    h, w = shape
    if palm.size != h * w:
        raise ValueError(f"手掌点数 {palm.size} 与网格 {shape} 不匹配")
    return np.asarray(palm, dtype=np.float64).reshape(h, w)


def hand_to_pressure_vector(hand: HandTactile) -> np.ndarray:
    """HandTactile → 68 路向量（decode_hand 的逆，按手册坐标表）。"""
    p = np.zeros(NUM_TACTILE, dtype=np.float64)
    p[NUM_FINGERS * FINGER_POINTS :] = hand.palm
    for i, finger in enumerate(hand.fingers):
        base = i * FINGER_POINTS
        p[base : base + FINGER_POINTS] = finger.pressure.reshape(-1)
    return p


def hand_to_unified_grid(hand: HandTactile) -> np.ndarray:
    """HandTactile → 10×8，与 pressure_to_hand_grid 一致。"""
    return pressure_to_hand_grid(hand_to_pressure_vector(hand))


def decode_hand(frame: "TactileFrame") -> HandTactile:
    """将 68 路向量重构为五指 2×2 + 手掌 48 点。"""
    p = frame.pressure
    if p.size != NUM_TACTILE:
        raise ValueError(f"期望 {NUM_TACTILE} 路触觉，得到 {p.size}")

    fingers: List[FingerTactile] = []
    for i, name in enumerate(FINGER_NAMES):
        chunk = p[i * FINGER_POINTS : (i + 1) * FINGER_POINTS]
        fingers.append(
            FingerTactile(
                name=name,
                pressure=packet_finger_to_2x2(chunk),
                bend=_finger_bend(chunk),
            )
        )
    palm = p[NUM_FINGERS * FINGER_POINTS :].copy()
    return HandTactile(sensor_type=SensorType.LEFT_HAND, fingers=fingers, palm=palm)


@dataclass
class TactileFrame:
    sequence: int
    pressure: np.ndarray  # (68,) 标定后 kPa
    timestamp: float
    pressure_ad: Optional[np.ndarray] = None  # (68,) 原始 AD，可选
    joint_angles: Optional[np.ndarray] = None  # (14,) 度；来自 SOLVED angles_mdeg
    solved: Optional[SolvedKinematics] = None

    def decode(self) -> HandTactile:
        hand = decode_hand(self)
        if self.joint_angles is not None:
            hand = attach_joint_angles(hand, self.joint_angles)
        return hand


@dataclass
class _PendingTouch:
    """分行协议：暂存最近一条 TOUCH，待 SOLVED 行配对。"""

    sequence: int
    pressure_ad: np.ndarray
    raw: bytes


class TactileGloveReader:
    """从串口同步并解析触觉手套数据。"""

    def __init__(
        self,
        port: str,
        baudrate: int = 921600,
        timeout: float = 0.1,
        *,
        verify_crc: bool = True,
        apply_calib: bool = True,
        calib: PressureCalibration = DEFAULT_PRESSURE_CALIB,
        debug: bool = False,
        require_complete_solved: bool = False,
        solved_wait_timeout_s: float = DEFAULT_SOLVED_WAIT_TIMEOUT_S,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.verify_crc = verify_crc
        self.apply_calib = apply_calib
        self.calib = calib
        self.debug = debug
        self.require_complete_solved = require_complete_solved
        self.solved_wait_timeout_s = max(float(solved_wait_timeout_s), 1.0)
        self._ser: Optional[serial.Serial] = None
        self._buf = bytearray()
        self._bytes_read = 0
        self._last_reject: Optional[str] = None
        self._reject_counts: dict[str, int] = {}
        self._solved_ready_announced = False
        self._pending_touch: Optional[_PendingTouch] = None

    def open(self) -> None:
        self._ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
        )
        self._buf.clear()
        self._bytes_read = 0
        self._last_reject = None
        self._reject_counts.clear()
        self._solved_ready_announced = False
        self._pending_touch = None
        # 部分 USB 转串口需拉高 DTR/RTS 后设备才开始发数
        try:
            self._ser.dtr = True
            self._ser.rts = True
        except Exception:
            pass
        try:
            self._ser.reset_input_buffer()
        except Exception:
            pass
        time.sleep(0.05)

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._ser = None

    def __enter__(self) -> TactileGloveReader:
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _read_chunk(self) -> bytes:
        if not self._ser or not self._ser.is_open:
            raise RuntimeError("串口未打开，请先调用 open()")
        n = self._ser.in_waiting
        if n <= 0:
            chunk = self._ser.read(1)
        else:
            chunk = self._ser.read(n)
        self._bytes_read += len(chunk)
        return chunk

    def _note_reject(self, reason: str, snippet: bytes) -> None:
        self._last_reject = reason
        self._reject_counts[reason] = self._reject_counts.get(reason, 0) + 1
        if self.debug:
            preview = snippet[:120].decode("ascii", errors="replace")
            print(f"[debug] 丢帧: {reason}  line={preview!r}")

    def _solved_skip_reason(
        self,
        solved: Optional[SolvedKinematics],
    ) -> Optional[str]:
        if solved is None:
            return "无 SOLVED 段"
        if not solved.is_complete:
            parts: List[str] = []
            if solved.angles_has_miss:
                parts.append("angles_mdeg")
            if solved.quat_has_miss:
                parts.append("quat_1e4")
            return "SOLVED 未就绪(MISS: " + ", ".join(parts) + ")"
        return None

    def _finalize_frame(
        self,
        sequence: int,
        pressure_ad: np.ndarray,
        solved: Optional[SolvedKinematics],
        raw: bytes,
    ) -> Optional[TactileFrame]:
        joint_angles: Optional[np.ndarray] = None
        try:
            skip_reason = self._solved_skip_reason(solved)
            if self.require_complete_solved:
                if skip_reason is not None:
                    self._note_reject(skip_reason, raw)
                    return None
                joint_angles = flex_joint_angles_from_mdeg(solved.angles_mdeg)  # type: ignore[union-attr]
                if not self._solved_ready_announced:
                    self._solved_ready_announced = True
                    print(
                        f"SOLVED 已就绪（touch_seq={sequence}, solved_seq={solved.sequence}），开始输出帧"
                    )
            elif solved is not None and solved.is_complete:
                joint_angles = flex_joint_angles_from_mdeg(solved.angles_mdeg)
        except ValueError as e:
            self._note_reject(f"SOLVED: {e}", raw)
            return None

        if self.apply_calib:
            pressure = calibrate_ad_to_kpa(pressure_ad, self.calib)
        else:
            pressure = pressure_ad.copy()

        if self.debug:
            msg = (
                f"[debug] OK seq={sequence} AD[0:3]={pressure_ad[:3].astype(int)} "
                f"kPa[0]={pressure[0]:.2f}"
            )
            if solved is not None:
                msg += (
                    f"  solved_seq={solved.sequence} status={solved.status} "
                    f"joint[0:3]={joint_angles[:3] if joint_angles is not None else []}"
                )
            print(msg)

        return TactileFrame(
            sequence=sequence,
            pressure=pressure,
            timestamp=time.time(),
            pressure_ad=pressure_ad.copy(),
            joint_angles=joint_angles,
            solved=solved,
        )

    def _handle_touch_line(self, line: str, raw: bytes) -> Optional[TactileFrame]:
        try:
            sequence, pressure_ad = parse_ascii_touch_line(line)
        except ValueError as e:
            self._note_reject(str(e), raw)
            return None

        solved = parse_ascii_solved_section(line)
        if solved is not None:
            return self._finalize_frame(sequence, pressure_ad, solved, raw)

        if self.require_complete_solved:
            self._pending_touch = _PendingTouch(
                sequence=sequence,
                pressure_ad=pressure_ad.copy(),
                raw=raw,
            )
            return None

        return self._finalize_frame(sequence, pressure_ad, None, raw)

    def _handle_solved_line(self, line: str, raw: bytes) -> Optional[TactileFrame]:
        try:
            solved = parse_ascii_solved_section(line)
        except ValueError as e:
            self._note_reject(f"SOLVED: {e}", raw)
            return None
        if solved is None:
            self._note_reject("SOLVED 段格式无效", raw)
            return None

        pending = self._pending_touch
        if pending is None:
            self._note_reject("SOLVED 无配对 TOUCH", raw)
            return None

        frame = self._finalize_frame(
            pending.sequence,
            pending.pressure_ad,
            solved,
            pending.raw + b"\n" + raw,
        )
        self._pending_touch = None
        return frame

    def _process_line(self, line: str, raw: bytes) -> Optional[TactileFrame]:
        stripped = line.lstrip()
        if stripped.startswith(ASCII_LINE_PREFIX):
            return self._handle_touch_line(line, raw)
        if stripped.startswith(SOLVED_LINE_PREFIX):
            return self._handle_solved_line(line, raw)
        return None

    def _try_extract_packets(self) -> list[TactileFrame]:
        frames: list[TactileFrame] = []
        while True:
            nl_pos = self._buf.find(b"\n")
            if nl_pos < 0:
                if len(self._buf) > 8192:
                    pos = max(self._buf.rfind(b"TOUCH"), self._buf.rfind(b"SOLVED"))
                    self._buf = self._buf[pos:] if pos >= 0 else self._buf[-4096:]
                break

            line_bytes = bytes(self._buf[:nl_pos])
            del self._buf[: nl_pos + 1]
            if line_bytes.endswith(b"\r"):
                line_bytes = line_bytes[:-1]
            if not line_bytes.strip():
                continue

            try:
                line = line_bytes.decode("ascii")
            except UnicodeDecodeError:
                self._note_reject("非 ASCII 行", line_bytes)
                continue

            frame = self._process_line(line, line_bytes)
            if frame is not None:
                frames.append(frame)

        return frames

    def _timeout_diagnostics(self) -> str:
        lines = [
            f"已读 {self._bytes_read} 字节，缓冲剩余 {len(self._buf)} 字节",
        ]
        if self._buf:
            preview = bytes(self._buf[:120]).decode("ascii", errors="replace")
            lines.append(f"缓冲前 120 字符: {preview!r}")
        if self._last_reject:
            lines.append(f"最近丢帧原因: {self._last_reject}")
        if self._reject_counts:
            top = sorted(self._reject_counts.items(), key=lambda x: -x[1])[:5]
            lines.append("丢帧统计: " + "; ".join(f"{k}×{v}" for k, v in top))
        if self._bytes_read == 0:
            lines.append(
                "串口无任何数据：请确认设备上电、线接好、COM 口正确，"
                "或尝试其它波特率"
            )
        elif not self._reject_counts:
            if self.require_complete_solved:
                lines.append(
                    "有数据但未收到完整 SOLVED（可能 IMU 仍在预热、TOUCH/SOLVED 未配对、"
                    "或 angles/quat 含 MISS）"
                )
            else:
                lines.append(
                    "有数据但未找到 TOUCH/SOLVED 完整行：可能波特率不对或行未以换行结束"
                )
        if self.require_complete_solved and self._pending_touch is not None:
            lines.append(
                f"已缓存待配对 TOUCH seq={self._pending_touch.sequence}，等待 SOLVED 行"
            )
        return "\n  ".join(lines)

    def _effective_read_timeout(self, timeout_s: float) -> float:
        if self.require_complete_solved:
            return max(float(timeout_s), self.solved_wait_timeout_s)
        return float(timeout_s)

    def read_frame(self, timeout_s: float = 2.0) -> TactileFrame:
        deadline = time.time() + self._effective_read_timeout(timeout_s)
        while time.time() < deadline:
            self._buf.extend(self._read_chunk())
            got = self._try_extract_packets()
            if got:
                return got[0]
        effective = self._effective_read_timeout(timeout_s)
        raise TimeoutError(
            f"{effective:g}s 内未收到有效帧\n  {self._timeout_diagnostics()}"
        )

    def frames(self) -> Generator[TactileFrame, None, None]:
        default_timeout = (
            self.solved_wait_timeout_s if self.require_complete_solved else 5.0
        )
        while True:
            yield self.read_frame(timeout_s=default_timeout)


def list_serial_ports() -> list[str]:
    from serial.tools import list_ports

    return [p.device for p in list_ports.comports()]


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError as e:
        raise SystemExit("可视化需要 matplotlib: pip install matplotlib") from e
    return plt, GridSpec


def _load_sensor_viz_xy() -> np.ndarray:
    """68 路传感点在手册示意图上的 (x, y) 像素坐标。"""
    if _SENSOR_LAYOUT_COORDS_NPY.is_file():
        xy = np.load(_SENSOR_LAYOUT_COORDS_NPY)
        if xy.shape == (NUM_TACTILE, 2):
            return np.asarray(xy, dtype=np.float64)

    script = _PROJECT_ROOT / "scripts" / "extract_sensor_layout_coords.py"
    if _SENSOR_LAYOUT_IMAGE.is_file() and script.is_file():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "extract_sensor_layout_coords", script
        )
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            xy = mod.extract_sensor_layout_xy()
            _ASSETS_DIR.mkdir(parents=True, exist_ok=True)
            np.save(_SENSOR_LAYOUT_COORDS_NPY, xy)
            return np.asarray(xy, dtype=np.float64)

    raise FileNotFoundError(
        "未找到传感点坐标。请先运行: python scripts/extract_sensor_layout_coords.py"
    )


_SENSOR_VIZ_XY = _load_sensor_viz_xy()


def _make_viz_cmap(plt, cmap: str):
    from matplotlib.colors import ListedColormap

    cmap_base = plt.get_cmap(cmap, 256)
    if hasattr(cmap_base, "copy"):
        cmap_obj = cmap_base.copy()
    else:
        cmap_obj = ListedColormap(cmap_base(np.linspace(0.0, 1.0, 256)))
    return cmap_obj


def _value_norm(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """标量 → [0, 1]，用于着色与脉冲强度。"""
    span = max(vmax - vmin, 1e-9)
    return np.clip((np.asarray(values, dtype=np.float64) - vmin) / span, 0.0, 1.0)


def _apply_viz_rcparams(plt) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": _VIZ_THEME["fig_bg"],
            "axes.facecolor": _VIZ_THEME["card_bg"],
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Microsoft YaHei", "DejaVu Sans"],
            "font.size": 10,
        }
    )


class HandTactileViewer:
    """手册布局 68 路脉冲点（无底图；大小/颜色随数值变化，外圈呼吸动画）。"""

    def __init__(
        self,
        vmin: float = VIZ_AD_MIN,
        vmax: float = VIZ_AD_MAX,
        cmap: str = _VIZ_CMAP,
        *,
        value_unit: str = "AD",
        subtract_baseline: bool = False,
        baseline_frames: int = VIZ_BASELINE_FRAMES,
        show_values: bool = False,
        pulse_period_s: float = _VIZ_PULSE_PERIOD_S,
    ) -> None:
        self._value_unit = value_unit
        self._clim = (vmin, vmax)
        self._subtract_baseline = subtract_baseline
        self._baseline_frames = max(1, baseline_frames)
        self._baseline: Optional[np.ndarray] = None
        self._baseline_buf: List[np.ndarray] = []
        self._show_values = show_values
        self._pulse_period_s = max(0.2, pulse_period_s)
        self._sensor_xy = _SENSOR_VIZ_XY
        plt, _ = _import_matplotlib()
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        _apply_viz_rcparams(plt)
        self._plt = plt
        img_w, img_h = _VIZ_CANVAS_SIZE

        self.fig = plt.figure(figsize=(img_w / 110, img_h / 110), dpi=110)
        self.fig.patch.set_facecolor(_VIZ_THEME["fig_bg"])
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title("Tactile Glove · Left Hand")

        self._ax = self.fig.add_axes([0.08, 0.08, 0.78, 0.80])
        self._ax.set_facecolor(_VIZ_THEME["card_bg"])
        self._ax.set_xlim(0, img_w)
        self._ax.set_ylim(img_h, 0)
        self._ax.set_aspect("equal")
        self._ax.axis("off")

        self._cmap_obj = _make_viz_cmap(plt, cmap)
        self._norm = Normalize(vmin=vmin, vmax=vmax)
        n = NUM_TACTILE
        xy0 = self._sensor_xy
        zero_norm = np.zeros(n)

        self._pulse_scatter = self._ax.scatter(
            xy0[:, 0],
            xy0[:, 1],
            s=np.full(n, _VIZ_PULSE_SIZE_PT2[0]),
            c=zero_norm,
            cmap=self._cmap_obj,
            norm=self._norm,
            alpha=0.28,
            linewidths=0,
            zorder=2,
        )
        self._core_scatter = self._ax.scatter(
            xy0[:, 0],
            xy0[:, 1],
            s=np.full(n, _VIZ_CORE_SIZE_PT2[0]),
            c=zero_norm,
            cmap=self._cmap_obj,
            norm=self._norm,
            alpha=0.92,
            edgecolors=_VIZ_THEME["point_edge"],
            linewidths=0.6,
            zorder=3,
        )

        sm = ScalarMappable(cmap=self._cmap_obj, norm=self._norm)
        sm.set_array([])
        cax = self.fig.add_axes([0.895, 0.14, 0.022, 0.68])
        self._cbar = self.fig.colorbar(sm, cax=cax)
        self._cbar.set_label(value_unit, fontsize=9, color=_VIZ_THEME["label"], labelpad=8)
        self._cbar.ax.tick_params(labelsize=8, colors=_VIZ_THEME["label_muted"])
        self._cbar.outline.set_edgecolor(_VIZ_THEME["grid"])
        self._cbar.outline.set_linewidth(0.6)

        self._value_texts: List[Any] = []
        self._bend_texts: List[Any] = []
        for i, name in enumerate(FINGER_NAMES):
            x, y = self._sensor_xy[i * FINGER_POINTS]
            txt = self._ax.text(
                x,
                y - 14,
                "",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=_VIZ_THEME["label_muted"],
                zorder=5,
            )
            self._bend_texts.append(txt)

        self._title = self.fig.text(
            0.5,
            0.97,
            "Tactile Glove · Left Hand",
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            color=_VIZ_THEME["title"],
        )
        self._subtitle = self.fig.text(
            0.5,
            0.935,
            "",
            ha="center",
            va="top",
            fontsize=9.5,
            color=_VIZ_THEME["label_muted"],
        )
        self._plt.ion()
        self._plt.show(block=False)

    def _clear_value_texts(self) -> None:
        for t in self._value_texts:
            t.remove()
        self._value_texts.clear()

    def _draw_value_labels(self, values: np.ndarray) -> None:
        if not self._show_values:
            return
        vmin, vmax = self._clim
        mid = 0.5 * (vmin + vmax)
        for i, (x, y) in enumerate(self._sensor_xy):
            val = float(values[i])
            color = "#ffffff" if val > mid else "#cbd5e1"
            txt = self._ax.text(
                x,
                y,
                f"{val:.0f}",
                ha="center",
                va="center",
                fontsize=6,
                color=color,
                fontweight="medium",
                zorder=6,
            )
            self._value_texts.append(txt)

    def _pulse_scale(self) -> float:
        phase = (time.time() % self._pulse_period_s) / self._pulse_period_s
        return 0.5 + 0.5 * np.sin(2.0 * np.pi * phase)

    def _update_pulse_scatter(self, values: np.ndarray) -> None:
        vmin, vmax = self._clim
        norm = _value_norm(values, vmin, vmax)
        pulse_k = self._pulse_scale()
        c_lo, c_hi = _VIZ_CORE_SIZE_PT2
        p_lo, p_hi = _VIZ_PULSE_SIZE_PT2
        core_sizes = c_lo + norm * (c_hi - c_lo)
        pulse_sizes = (p_lo + norm * (p_hi - p_lo)) * (0.55 + 0.45 * pulse_k)
        pulse_alpha = 0.12 + 0.38 * norm * (0.45 + 0.55 * pulse_k)

        self._core_scatter.set_array(values)
        self._core_scatter.set_sizes(core_sizes)

        pulse_rgba = np.array(self._cmap_obj(self._norm(values)), dtype=np.float64)
        if pulse_rgba.shape[1] == 3:
            pulse_rgba = np.column_stack([pulse_rgba, pulse_alpha])
        else:
            pulse_rgba[:, 3] = pulse_alpha
        self._pulse_scatter.set_facecolors(pulse_rgba)
        self._pulse_scatter.set_sizes(pulse_sizes)

    def _prepare_display_hand(self, hand: HandTactile) -> Tuple[HandTactile, str]:
        if not self._subtract_baseline:
            return hand, ""
        flat = hand_to_pressure_vector(hand)
        if self._baseline is None:
            self._baseline_buf.append(flat)
            if len(self._baseline_buf) >= self._baseline_frames:
                self._baseline = np.median(np.stack(self._baseline_buf), axis=0)
            n = len(self._baseline_buf)
            empty = decode_hand(
                TactileFrame(sequence=0, pressure=np.zeros(NUM_TACTILE), timestamp=0.0)
            )
            return empty, f"Calibrating baseline {n}/{self._baseline_frames} — keep hand still"
        delta = np.clip(flat - self._baseline, 0.0, None)
        return decode_hand(TactileFrame(sequence=0, pressure=delta, timestamp=0.0)), ""

    def update(
        self,
        hand: HandTactile,
        *,
        sequence: Optional[int] = None,
        pressure: Optional[np.ndarray] = None,
    ) -> None:
        if pressure is not None:
            hand = decode_hand(
                TactileFrame(sequence=sequence or 0, pressure=pressure, timestamp=0.0)
            )

        hand, calib_note = self._prepare_display_hand(hand)
        values = hand_to_pressure_vector(hand)
        self._update_pulse_scatter(values)

        self._clear_value_texts()
        self._draw_value_labels(values)

        peak = float(np.max(values))

        for i, finger in enumerate(hand.fingers):
            self._bend_texts[i].set_text(f"Σ {finger.bend:.0f}")

        meta_parts = [f"Peak {peak:.0f} {self._value_unit}"]
        if sequence is not None:
            meta_parts.append(f"Seq {sequence}")
        if calib_note:
            meta_parts.append(calib_note)
        self._subtitle.set_text("  ·  ".join(meta_parts))

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self._plt.pause(0.001)

    def close(self) -> None:
        self._clear_value_texts()
        self._plt.close(self.fig)


def plot_hand(
    hand: HandTactile,
    *,
    sequence: Optional[int] = None,
    block: bool = True,
) -> None:
    viewer = HandTactileViewer()
    viewer.update(hand, sequence=sequence)
    viewer._plt.show(block=block)


def format_ad_line(sequence: int, ad: np.ndarray) -> str:
    """格式化 68 路 AD 为可读行。"""
    vals = np.asarray(ad, dtype=np.int32).reshape(-1)
    if vals.size != NUM_TACTILE:
        raise ValueError(f"期望 {NUM_TACTILE} 路，得到 {vals.size}")
    chunks = []
    for i, name in enumerate(FINGER_NAMES):
        s = i * FINGER_POINTS
        e = s + FINGER_POINTS
        part = " ".join(f"{v:4d}" for v in vals[s:e])
        chunks.append(f"{name}[{s}:{e}] {part}")
    palm = " ".join(f"{v:4d}" for v in vals[NUM_FINGERS * FINGER_POINTS :])
    chunks.append(f"palm[20:68] {palm}")
    return f"seq={sequence}  " + "  |  ".join(chunks)


def run_stream(
    port: str,
    baudrate: int,
    *,
    verify_crc: bool = True,
    debug: bool = False,
    refresh: bool = False,
    interval: float = 0.5,
) -> None:
    """终端实时输出 68 路 AD，不做标定、不画图。"""
    if interval <= 0:
        raise ValueError("interval 必须 > 0")
    try:
        with TactileGloveReader(
            port=port,
            baudrate=baudrate,
            verify_crc=verify_crc,
            apply_calib=False,
            debug=debug,
        ) as reader:
            print(
                f"实时 AD: {port} @ {baudrate}  "
                f"(ASCII TOUCH 行, 每 {interval:g}s 刷新)  Ctrl+C 退出"
            )
            last_print = 0.0
            for frame in reader.frames():
                ad = frame.pressure_ad if frame.pressure_ad is not None else frame.pressure
                line = format_ad_line(frame.sequence, ad)
                now = time.time()
                if now - last_print < interval:
                    continue
                last_print = now
                if refresh:
                    print("\r" + line, end="", flush=True)
                else:
                    print(line)
    except KeyboardInterrupt:
        print()
    except TimeoutError as e:
        print(e)
        print("建议: python tactile_serial_reader.py --port", port, "--debug")


def run_visualizer_3d(
    port: str,
    baudrate: int,
    *,
    verify_crc: bool = True,
    debug: bool = False,
) -> None:
    """MANO 3D 网格热力图（SynchroTactile 风格）。"""
    from mano_tactile_viz import ManoHandTactileViewer, mano_models_available

    if not mano_models_available():
        raise SystemExit(
            "3D 可视化需要 MANO 模型文件。\n"
            f"  1. 从 https://mano.is.tue.mpg.de 下载 MANO_LEFT.pkl / MANO_RIGHT.pkl\n"
            f"  2. 放入 {Path(__file__).resolve().parent / 'mano_v1_2' / 'models'}\n"
            "  3. 运行: python scripts/export_mano_meshes.py"
        )

    viewer = ManoHandTactileViewer(
        _SENSOR_VIZ_XY,
        value_unit="ΔAD",
        vmin=VIZ_AD_MIN,
        vmax=VIZ_AD_MAX,
        subtract_baseline=True,
        baseline_frames=VIZ_BASELINE_FRAMES,
    )
    try:
        with TactileGloveReader(
            port=port,
            baudrate=baudrate,
            verify_crc=verify_crc,
            apply_calib=False,
            debug=debug,
        ) as reader:
            print(
                f"MANO 3D 可视化: {port} @ {baudrate} "
                f"(ΔAD, clim 0–200)，前 {VIZ_BASELINE_FRAMES} 帧静止标基线"
            )
            for frame in reader.frames():
                if viewer.fig.number not in viewer._plt.get_fignums():
                    break
                ad = frame.pressure_ad if frame.pressure_ad is not None else frame.pressure
                hand = decode_hand(
                    TactileFrame(
                        sequence=frame.sequence,
                        pressure=ad,
                        timestamp=frame.timestamp,
                        pressure_ad=ad,
                    )
                )
                viewer.update(hand, sequence=frame.sequence)
    except KeyboardInterrupt:
        pass
    except TimeoutError as e:
        print(e)
    finally:
        viewer.close()


def run_visualizer(
    port: str,
    baudrate: int,
    *,
    verify_crc: bool = True,
    debug: bool = False,
) -> None:
    """可视化：原始 AD 减静止基线，色标固定 0–200。"""
    viewer = HandTactileViewer(
        value_unit="ΔAD",
        vmin=VIZ_AD_MIN,
        vmax=VIZ_AD_MAX,
        subtract_baseline=True,
    )
    try:
        with TactileGloveReader(
            port=port,
            baudrate=baudrate,
            verify_crc=verify_crc,
            apply_calib=False,
            debug=debug,
        ) as reader:
            print(f"可视化已启动: {port} @ {baudrate} (ΔAD, clim 0–200)，前 {VIZ_BASELINE_FRAMES} 帧静止标基线")
            for frame in reader.frames():
                if viewer.fig.number not in viewer._plt.get_fignums():
                    break
                ad = frame.pressure_ad if frame.pressure_ad is not None else frame.pressure
                hand = decode_hand(
                    TactileFrame(
                        sequence=frame.sequence,
                        pressure=ad,
                        timestamp=frame.timestamp,
                        pressure_ad=ad,
                    )
                )
                viewer.update(hand, sequence=frame.sequence)
    except KeyboardInterrupt:
        pass
    except TimeoutError as e:
        print(e)
    finally:
        viewer.close()


def main() -> None:
    ports = list_serial_ports()
    parser = argparse.ArgumentParser(description="左手触觉手套串口读取（68 点）")
    parser.add_argument("--port", default=ports[0] if ports else "COM5", help="串口名")
    parser.add_argument("--baudrate", type=int, default=921600, help="波特率")
    parser.add_argument("--viz", action="store_true", help="2D 脉冲点热力图")
    parser.add_argument(
        "--viz-3d",
        action="store_true",
        help="MANO 3D 网格热力图（需 MANO_LEFT/RIGHT.pkl）",
    )
    parser.add_argument(
        "--viz-3d-demo",
        action="store_true",
        help="MANO 3D 静态演示（无需串口，需 MANO 模型）",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="单行刷新（默认逐行打印）",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        metavar="SEC",
        help="终端刷新间隔秒数（默认 0.5）",
    )
    parser.add_argument("--no-crc", action="store_true", help="（已废弃，ASCII 协议无 CRC）")
    parser.add_argument("--no-calib", action="store_true", help="输出原始 AD，不做 kPa 标定")
    parser.add_argument("--debug", action="store_true", help="打印丢帧原因与原始字节片段")
    args = parser.parse_args()

    if args.viz_3d_demo:
        from mano_tactile_viz import mano_models_available, plot_mano_demo

        if not mano_models_available():
            raise SystemExit(
                "3D 演示需要 MANO 模型文件。\n"
                f"  请将 MANO_LEFT.pkl / MANO_RIGHT.pkl 放入 "
                f"{Path(__file__).resolve().parent / 'mano_v1_2' / 'models'}"
            )
        plot_mano_demo(block=True)
        return

    if args.viz_3d:
        run_visualizer_3d(
            args.port,
            args.baudrate,
            verify_crc=not args.no_crc,
            debug=args.debug,
        )
        return

    if args.viz:
        run_visualizer(
            args.port,
            args.baudrate,
            verify_crc=not args.no_crc,
            debug=args.debug,
        )
        return

    run_stream(
        args.port,
        args.baudrate,
        verify_crc=not args.no_crc,
        debug=args.debug,
        refresh=args.refresh,
        interval=args.interval,
    )


if __name__ == "__main__":
    main()
