"""
整掌压阻阵列上的规则式滑移检测（简易方案），对应《滑移检测.md》第 49–54 行。

假设：传感器覆盖整个手掌，为一块 H×W 网格。将文档中的「小指/无名指等边缘指位」
映射为阵列几何外圈；「主要受力点」映射为去掉一圈后的内区。

左手 68 点手套：掌 8×6 + 指 2×2 填入 10×8 统一网格（左指右掌）；
边缘区 = 手掌外圈 + 各指远端行；主要区 = 掌内部 + 各指近端行。
见 HandSlipDetectorSimple / hand_to_slip_grid。

输出：
  - slip_alert：单通道短时相对骤降 + 总力短时方差激增（可调阈值）
  - trend_slope：过去 1 s 内总力的线性拟合斜率（负值表示缓慢放松）
  - slip_risk：1 - 当前总力 / 过去 1 s 总力峰值（文档风险度）
  - severity：NONE / MILD / MODERATE / SEVERE 粗分档

注意：采样率过低（如 <50 Hz）时 50 ms 窗内不足 2 帧，骤降判定会退化；文档建议 ≥100 Hz。
纯法向力无法区分主动释放与意外滑移（见原文）。

P0/P1：active mask（排除 pad）、基线扣除、接触门控、绝对降幅 + 报警滞回。
A：骤降参照用短窗逐格 max、方差长短窗不重叠、骤降空间连通（掌 CC / 同指≥2 格）。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Deque, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from tactile_serial_reader import HandTactile

import numpy as np


class SlipSeverity(Enum):
    NONE = auto()
    MILD = auto()  # 仅边缘 1 点明显骤降，内区无同步大降
    MODERATE = auto()
    SEVERE = auto()  # 内区 ≥2 点相对骤降 ≥30% 等


@dataclass
class SlipDetectionConfig:
    sample_rate_hz: float = 100.0
    # 骤降：相对窗口起点的降幅 (F_old - F_new) / max(F_old, eps)
    drop_window_s: float = 0.05
    drop_ratio_alert: float = 0.20
    drop_ratio_severe: float = 0.30
    # 总力方差：短窗 vs 长窗
    variance_short_s: float = 0.15
    variance_long_s: float = 1.0
    variance_spike_ratio: float = 2.5
    # 趋势与风险度
    trend_window_s: float = 1.0
    peak_window_s: float = 1.0
    # 斜率判「明显为负」：按总力量级归一（除以 max(峰值, eps)）
    trend_slope_norm_threshold: float = -0.05  # 每秒内相对下降 5% 峰值以上认为趋势明显
    # 可选：输入前先低通（简单滑动平均，帧数）
    lowpass_window_frames: int = 0  # 0 表示关闭
    eps: float = 1e-6
    # P1：接触门控、基线、骤降绝对量、报警滞回
    contact_force_threshold: float = 80.0
    baseline_frames: int = 30
    drop_abs_alert: float = 12.0
    alert_hold_frames: int = 2
    alert_release_frames: int = 3
    # A：空间连通、不重叠方差窗
    min_drop_cluster_cells: int = 2


@dataclass
class SlipState:
    slip_alert: bool
    trend_slope: float
    trend_slope_normalized: float
    slip_risk: float
    severity: SlipSeverity
    total_force: float
    total_force_peak_1s: float
    short_variance_total: float
    long_variance_total: float
    num_cells_drop_alert: int
    num_major_drop_severe: int
    num_peripheral_drop_alert: int
    is_contact: bool = True
    baseline_ready: bool = False


def _peripheral_mask(height: int, width: int) -> np.ndarray:
    """外圈为 True（对应文档边缘指位/尺侧等的几何近似）。"""
    m = np.zeros((height, width), dtype=bool)
    m[0, :] = True
    m[-1, :] = True
    m[:, 0] = True
    m[:, -1] = True
    return m


def _major_mask(height: int, width: int) -> np.ndarray:
    peri = _peripheral_mask(height, width)
    major = np.ones((height, width), dtype=bool)
    major[peri] = False
    if not major.any():
        # 极小阵列：退化为全主要区
        major[:] = True
    return major


# 左手 68 点：掌 8×6 占 grid[2:10,2:8]，五指 2×2 占左两列 → 10×8
HAND_SLIP_PALM_SHAPE = (8, 6)
HAND_SLIP_GRID_SHAPE = (10, 8)
HAND_SLIP_PALM_ROW_START = 2
HAND_SLIP_PALM_COL_START = 2
HAND_SLIP_PALM_ROWS, HAND_SLIP_PALM_COLS = HAND_SLIP_PALM_SHAPE
HAND_SLIP_GRID_H, HAND_SLIP_GRID_W = HAND_SLIP_GRID_SHAPE
HAND_SLIP_FINGER_ROWS = 2
HAND_SLIP_FINGER_COLS = 2
HAND_SLIP_NUM_FINGERS = 5
# 各指在统一网格中的 2 行块 + 远端行（拇指在上、小指在下）
HAND_SLIP_FINGER_ROW_PAIRS: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
)
HAND_SLIP_FINGER_DISTAL_ROWS: Tuple[int, ...] = (0, 2, 4, 6, 9)


def hand_to_slip_grid(hand: "HandTactile") -> np.ndarray:
    """将 HandTactile 填入滑移检测用 10×8 统一网格。"""
    from tactile_serial_reader import hand_to_unified_grid

    if len(hand.fingers) != HAND_SLIP_NUM_FINGERS:
        raise ValueError(f"期望 {HAND_SLIP_NUM_FINGERS} 指，得到 {len(hand.fingers)}")
    return hand_to_unified_grid(hand)


def build_hand_slip_masks(
    grid_shape: Tuple[int, int] = HAND_SLIP_GRID_SHAPE,
    palm_shape: Tuple[int, int] = HAND_SLIP_PALM_SHAPE,
    num_fingers: int = HAND_SLIP_NUM_FINGERS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    语义分区 mask（10×8 左指右掌）：
      - 边缘：掌 8×6 外圈 + 各指远端行（2 格）
      - 主要：掌内部 + 各指近端行
    """
    gh, gw = grid_shape
    pr, pc = palm_shape
    rs, cs = HAND_SLIP_PALM_ROW_START, HAND_SLIP_PALM_COL_START
    peri = np.zeros((gh, gw), dtype=bool)
    major = np.zeros((gh, gw), dtype=bool)

    peri[rs : rs + pr, cs : cs + pc] = _peripheral_mask(pr, pc)
    major[rs : rs + pr, cs : cs + pc] = _major_mask(pr, pc)

    for (r0, r1), distal_r in zip(HAND_SLIP_FINGER_ROW_PAIRS, HAND_SLIP_FINGER_DISTAL_ROWS):
        peri[distal_r, 0:HAND_SLIP_FINGER_COLS] = True
        proximal_r = r1 if distal_r == r0 else r0
        major[proximal_r, 0:HAND_SLIP_FINGER_COLS] = True

    return peri.reshape(-1), major.reshape(-1)


def build_hand_active_mask(
    grid_shape: Tuple[int, int] = HAND_SLIP_GRID_SHAPE,
    palm_shape: Tuple[int, int] = HAND_SLIP_PALM_SHAPE,
    num_fingers: int = HAND_SLIP_NUM_FINGERS,
) -> np.ndarray:
    """真实传感器位置：左 2 列五指 + 右 6 列掌。"""
    gh, gw = grid_shape
    pr, pc = palm_shape
    rs, cs = HAND_SLIP_PALM_ROW_START, HAND_SLIP_PALM_COL_START
    active = np.zeros((gh, gw), dtype=bool)
    active[rs : rs + pr, cs : cs + pc] = True
    for r0, r1 in HAND_SLIP_FINGER_ROW_PAIRS[:num_fingers]:
        active[r0 : r1 + 1, 0:HAND_SLIP_FINGER_COLS] = True
    return active.reshape(-1)


def _connected_components_min_size(mask: np.ndarray, min_size: int) -> np.ndarray:
    """4-邻域连通域，保留像素数 >= min_size 的分量。"""
    h, w = mask.shape
    kept = np.zeros((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            component: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                component.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            if len(component) >= min_size:
                for cr, cc in component:
                    kept[cr, cc] = True
    return kept


def filter_drop_clusters(
    alert_flat: np.ndarray,
    height: int,
    width: int,
    active_flat: np.ndarray,
    *,
    palm_rows: Optional[int] = None,
    finger_cols: int = HAND_SLIP_FINGER_COLS,
    num_fingers: int = HAND_SLIP_NUM_FINGERS,
    min_cluster: int = 2,
    layout: str = "stacked",
) -> np.ndarray:
    """
    过滤孤立单点骤降。
    layout="stacked"：上掌下指（旧 10×15）；layout="hand_side"：左指右掌（10×8）。
    """
    alert = alert_flat.reshape(height, width).copy()
    active = active_flat.reshape(height, width)
    alert &= active
    kept = np.zeros((height, width), dtype=bool)

    if layout == "hand_side":
        rs, cs = HAND_SLIP_PALM_ROW_START, HAND_SLIP_PALM_COL_START
        pr, pc = HAND_SLIP_PALM_ROWS, HAND_SLIP_PALM_COLS
        palm_alert = alert[rs : rs + pr, cs : cs + pc]
        kept[rs : rs + pr, cs : cs + pc] |= _connected_components_min_size(
            palm_alert, min_cluster
        )
        for r0, r1 in HAND_SLIP_FINGER_ROW_PAIRS[:num_fingers]:
            block = alert[r0 : r1 + 1, 0:finger_cols]
            if int(np.count_nonzero(block)) >= min_cluster:
                kept[r0 : r1 + 1, 0:finger_cols] |= block
        return kept.reshape(-1)

    if palm_rows is not None and 0 < palm_rows < height:
        for i in range(num_fingers):
            c0 = i * finger_cols
            c1 = c0 + finger_cols
            block = alert[palm_rows:height, c0:c1]
            if int(np.count_nonzero(block)) >= min_cluster:
                kept[palm_rows:height, c0:c1] |= block
        palm_active = active[:palm_rows, :]
        palm_alert = alert[:palm_rows, :] & palm_active
        kept[:palm_rows, :] |= _connected_components_min_size(palm_alert, min_cluster)
    else:
        kept = _connected_components_min_size(alert, min_cluster)

    return kept.reshape(-1)


def _disjoint_force_variance(
    tf_arr: np.ndarray,
    short_w: int,
    long_w: int,
    eps: float,
) -> Tuple[float, float]:
    """短窗 [末尾 short_w] vs 更早 long_w（与短窗不重叠）。"""
    n = len(tf_arr)
    if n < short_w + 2:
        return 0.0, eps
    short_seg = tf_arr[-short_w:]
    short_var = float(np.var(short_seg))
    if n >= long_w + short_w:
        long_seg = tf_arr[-(long_w + short_w) : -short_w]
    else:
        long_seg = tf_arr[: max(1, n - short_w)]
    long_var = float(np.var(long_seg)) if len(long_seg) >= 2 else eps
    return short_var, max(long_var, eps)


class PalmSlipDetectorSimple:
    """
    实时规则式滑移检测。每帧调用 update(frame) -> SlipState。

    frame: (H, W) 或长度为 H*W 的一维数组，与标定一致的法向力（可为 V_net 或原始值）。
    """

    def __init__(
        self,
        height: int,
        width: int,
        config: Optional[SlipDetectionConfig] = None,
        *,
        peripheral_flat: Optional[np.ndarray] = None,
        major_flat: Optional[np.ndarray] = None,
        active_flat: Optional[np.ndarray] = None,
        spatial_palm_rows: Optional[int] = None,
        cluster_layout: str = "stacked",
    ) -> None:
        if height < 2 or width < 2:
            raise ValueError("手掌阵列至少为 2×2 以区分内外区")
        self.h = height
        self.w = width
        self.s = height * width
        self.cfg = config or SlipDetectionConfig()

        fs = self.cfg.sample_rate_hz
        self._drop_w = max(2, int(round(fs * self.cfg.drop_window_s)))
        self._baseline_w = max(1, int(self.cfg.baseline_frames))
        self._var_short_w = max(2, int(round(fs * self.cfg.variance_short_s)))
        self._var_long_w = max(self._var_short_w + 1, int(round(fs * self.cfg.variance_long_s)))
        self._trend_w = max(3, int(round(fs * self.cfg.trend_window_s)))
        self._peak_w = max(2, int(round(fs * self.cfg.peak_window_s)))

        max_hist = max(
            self._drop_w + 2,
            self._var_long_w + self._var_short_w + 5,
            self._trend_w + 5,
            self._peak_w + 5,
        )
        self._max_hist = max_hist
        self._spatial_palm_rows = spatial_palm_rows
        self._cluster_layout = cluster_layout

        self._frames: Deque[np.ndarray] = deque(maxlen=max_hist)
        self._total_f: Deque[float] = deque(maxlen=max_hist)
        self._lp_buf: Deque[np.ndarray] = deque(
            maxlen=max(1, self.cfg.lowpass_window_frames) if self.cfg.lowpass_window_frames > 0 else 1
        )

        if peripheral_flat is not None or major_flat is not None:
            if peripheral_flat is None or major_flat is None:
                raise ValueError("peripheral_flat 与 major_flat 须同时提供")
            peri = np.asarray(peripheral_flat, dtype=bool).reshape(-1)
            maj = np.asarray(major_flat, dtype=bool).reshape(-1)
            if peri.size != self.s or maj.size != self.s:
                raise ValueError(f"mask 长度须为 {self.s}，得到 peri={peri.size} major={maj.size}")
            self._peri_flat = peri
            self._major_flat = maj
        else:
            self._peri_flat = _peripheral_mask(height, width).reshape(-1)
            self._major_flat = _major_mask(height, width).reshape(-1)

        if active_flat is not None:
            act = np.asarray(active_flat, dtype=bool).reshape(-1)
            if act.size != self.s:
                raise ValueError(f"active mask 长度须为 {self.s}，得到 {act.size}")
            self._active_flat = act
        else:
            self._active_flat = np.ones(self.s, dtype=bool)

        self._peri_flat &= self._active_flat
        self._major_flat &= self._active_flat

        self._baseline: Optional[np.ndarray] = None
        self._baseline_ready = False
        self._baseline_buf: Deque[np.ndarray] = deque(maxlen=self._baseline_w)
        self._alert_latched = False
        self._alert_streak = 0
        self._clear_streak = 0

    def reset(self) -> None:
        self._frames.clear()
        self._total_f.clear()
        self._lp_buf.clear()
        self._baseline_buf.clear()
        self._baseline = None
        self._baseline_ready = False
        self._alert_latched = False
        self._alert_streak = 0
        self._clear_streak = 0

    def _flatten(self, frame: Union[np.ndarray, List[float]]) -> np.ndarray:
        x = np.asarray(frame, dtype=np.float64).reshape(-1)
        if x.size != self.s:
            raise ValueError(f"期望 {self.s} 个通道，得到 {x.size}")
        return x

    def _maybe_lowpass(self, flat: np.ndarray) -> np.ndarray:
        k = self.cfg.lowpass_window_frames
        if k <= 1:
            return flat
        self._lp_buf.append(flat.copy())
        if len(self._lp_buf) < k:
            return flat
        return np.mean(np.stack(self._lp_buf, axis=0), axis=0)

    def _total_force_active(self, flat: np.ndarray) -> float:
        return float(np.sum(flat[self._active_flat]))

    def _learn_baseline(self, flat: np.ndarray) -> None:
        self._baseline_buf.append(flat.copy())
        if len(self._baseline_buf) >= self._baseline_w:
            self._baseline = np.mean(np.stack(list(self._baseline_buf), axis=0), axis=0)
            self._baseline_ready = True

    def _apply_baseline(self, flat: np.ndarray) -> np.ndarray:
        if not self._baseline_ready or self._baseline is None:
            return flat
        out = flat - self._baseline
        out[~self._active_flat] = 0.0
        return np.maximum(out, 0.0)

    def _update_alert_hysteresis(self, raw_alert: bool) -> bool:
        cfg = self.cfg
        if raw_alert:
            self._alert_streak += 1
            self._clear_streak = 0
        else:
            self._clear_streak += 1
            self._alert_streak = 0
        if self._alert_streak >= cfg.alert_hold_frames:
            self._alert_latched = True
        if self._clear_streak >= cfg.alert_release_frames:
            self._alert_latched = False
        return self._alert_latched

    def _safe_state(
        self,
        tf: float,
        *,
        is_contact: bool,
        baseline_ready: bool,
        peak_1s: Optional[float] = None,
    ) -> SlipState:
        peak = peak_1s if peak_1s is not None else tf
        return SlipState(
            slip_alert=False,
            trend_slope=0.0,
            trend_slope_normalized=0.0,
            slip_risk=0.0,
            severity=SlipSeverity.NONE,
            total_force=tf,
            total_force_peak_1s=peak,
            short_variance_total=0.0,
            long_variance_total=0.0,
            num_cells_drop_alert=0,
            num_major_drop_severe=0,
            num_peripheral_drop_alert=0,
            is_contact=is_contact,
            baseline_ready=baseline_ready,
        )

    def update(self, frame: Union[np.ndarray, List[float]]) -> SlipState:
        flat_raw = self._flatten(frame)
        flat_raw = self._maybe_lowpass(flat_raw)

        if not self._baseline_ready:
            self._learn_baseline(flat_raw)
            tf_raw = self._total_force_active(flat_raw)
            return self._safe_state(tf_raw, is_contact=False, baseline_ready=False)

        flat = self._apply_baseline(flat_raw)
        tf = self._total_force_active(flat)

        cfg = self.cfg
        eps = cfg.eps
        is_contact = tf >= cfg.contact_force_threshold

        self._frames.append(flat)
        self._total_f.append(tf)

        if not is_contact:
            self._update_alert_hysteresis(False)
            return self._safe_state(
                tf,
                is_contact=False,
                baseline_ready=True,
                peak_1s=float(np.max(self._total_f)) if self._total_f else tf,
            )

        # 历史不足时返回安全默认
        if len(self._frames) < self._drop_w + 1:
            return self._safe_state(tf, is_contact=True, baseline_ready=True, peak_1s=tf)

        arr = np.stack(list(self._frames), axis=0)  # (T, S)
        # A：参照为骤降窗内逐格峰值（非单帧 F0）
        ref_slice = arr[-(self._drop_w + 1) : -1]
        if ref_slice.shape[0] == 0:
            ref_slice = arr[-2:-1]
        f_ref = np.max(ref_slice, axis=0)
        f1 = arr[-1]
        abs_drop = f_ref - f1
        rel_drop = abs_drop / np.maximum(f_ref, eps)

        alert_raw = (
            (rel_drop > cfg.drop_ratio_alert)
            & (abs_drop > cfg.drop_abs_alert)
            & self._active_flat
        )
        severe_raw = (
            (rel_drop > cfg.drop_ratio_severe)
            & (abs_drop > cfg.drop_abs_alert)
            & self._major_flat
        )
        min_cluster = max(1, int(cfg.min_drop_cluster_cells))
        alert_drop = filter_drop_clusters(
            alert_raw,
            self.h,
            self.w,
            self._active_flat,
            palm_rows=self._spatial_palm_rows,
            min_cluster=min_cluster,
            layout=self._cluster_layout,
        )
        severe_drop_major = filter_drop_clusters(
            severe_raw,
            self.h,
            self.w,
            self._active_flat,
            palm_rows=self._spatial_palm_rows,
            min_cluster=min_cluster,
            layout=self._cluster_layout,
        ) & self._major_flat

        num_cells_drop_alert = int(np.sum(alert_drop))
        num_major_drop_severe = int(np.sum(severe_drop_major))
        num_peripheral_drop_alert = int(np.sum(alert_drop & self._peri_flat))

        tf_arr = np.array(self._total_f, dtype=np.float64)
        short_var, long_var = _disjoint_force_variance(
            tf_arr, self._var_short_w, self._var_long_w, eps
        )
        var_spike = short_var > cfg.variance_spike_ratio * long_var

        raw_alert = bool(num_cells_drop_alert > 0 and var_spike)
        slip_alert = self._update_alert_hysteresis(raw_alert)

        trend_slope = 0.0
        trend_slope_norm = 0.0
        if len(tf_arr) >= self._trend_w:
            y = tf_arr[-self._trend_w :]
            x = np.arange(len(y), dtype=np.float64)
            slope, _intercept = np.polyfit(x, y, 1)
            trend_slope = float(slope)
            peak_recent = float(np.max(tf_arr[-self._peak_w :]))
            trend_slope_norm = float(slope * (self._trend_w - 1) / max(peak_recent, eps))

        peak_1s = float(np.max(tf_arr[-self._peak_w :]))
        slip_risk = 1.0 - tf / max(peak_1s, eps)
        slip_risk = float(np.clip(slip_risk, 0.0, 1.0))

        severity = SlipSeverity.NONE
        if num_major_drop_severe >= 2:
            severity = SlipSeverity.SEVERE
        elif slip_alert and num_major_drop_severe >= 1:
            severity = SlipSeverity.SEVERE
        elif num_cells_drop_alert >= 3 and num_major_drop_severe >= 1:
            severity = SlipSeverity.SEVERE
        elif num_peripheral_drop_alert == 1 and num_cells_drop_alert == 1:
            severity = SlipSeverity.MILD
        elif slip_alert or num_cells_drop_alert >= 2:
            severity = SlipSeverity.MODERATE
        elif trend_slope_norm < cfg.trend_slope_norm_threshold and slip_risk > 0.25:
            severity = SlipSeverity.MODERATE

        return SlipState(
            slip_alert=slip_alert,
            trend_slope=trend_slope,
            trend_slope_normalized=trend_slope_norm,
            slip_risk=slip_risk,
            severity=severity,
            total_force=tf,
            total_force_peak_1s=peak_1s,
            short_variance_total=short_var,
            long_variance_total=long_var,
            num_cells_drop_alert=num_cells_drop_alert,
            num_major_drop_severe=num_major_drop_severe,
            num_peripheral_drop_alert=num_peripheral_drop_alert,
            is_contact=is_contact,
            baseline_ready=True,
        )


class HandSlipDetectorSimple:
    """左手 68 点手套：HandTactile → 10×8 统一网格 + 语义 mask。"""

    def __init__(self, config: Optional[SlipDetectionConfig] = None) -> None:
        h, w = HAND_SLIP_GRID_SHAPE
        peri, major = build_hand_slip_masks()
        active = build_hand_active_mask()
        self._detector = PalmSlipDetectorSimple(
            h,
            w,
            config,
            peripheral_flat=peri,
            major_flat=major,
            active_flat=active,
            spatial_palm_rows=HAND_SLIP_PALM_ROW_START,
            cluster_layout="hand_side",
        )

    @property
    def config(self) -> SlipDetectionConfig:
        return self._detector.cfg

    def reset(self) -> None:
        self._detector.reset()

    def update(self, hand: "HandTactile") -> SlipState:
        grid = hand_to_slip_grid(hand)
        return self._detector.update(grid)


def risk_band_description(risk: float) -> str:
    """与文档 0~0.3 / 0.4~0.6 / >0.7 文字对齐的粗解释。"""
    if risk < 0.3:
        return "低风险（稳定抓握倾向）"
    if risk < 0.4:
        return "中低风险"
    if risk < 0.7:
        return "滑移趋势明显（部分脱离倾向）"
    return "高风险（剧烈滑移或已掉落倾向）"


if __name__ == "__main__":
    # 合成：8×8 阵列，稳定若干帧后模拟一侧边缘骤降 + 总力波动
    H, W = 8, 8
    det = PalmSlipDetectorSimple(H, W, SlipDetectionConfig(sample_rate_hz=100.0))
    rng = np.random.default_rng(0)
    base = np.abs(rng.normal(1.0, 0.05, size=(H, W))) + 0.5

    print("前 80 帧稳定抓握…")
    for _ in range(80):
        f = base + rng.normal(0, 0.02, size=(H, W))
        st = det.update(f)

    print("后 40 帧：外圈随机单元骤降 + 整体略抖（模拟滑移）…")
    for t in range(40):
        f = base.copy()
        if t > 5:
            # 随机边缘格骤降
            edge_idx = [(0, k) for k in range(W)] + [(H - 1, k) for k in range(W)]
            r, c = edge_idx[int(rng.integers(0, len(edge_idx)))]
            f[r, c] *= float(rng.uniform(0.2, 0.5))
            f += rng.normal(0, 0.08, size=(H, W))
        st = det.update(np.maximum(f, 0.0))
        if t % 5 == 0:
            print(
                f"  t={t:02d} alert={st.slip_alert} sev={st.severity.name} "
                f"risk={st.slip_risk:.2f} ({risk_band_description(st.slip_risk)}) "
                f"slope_norm={st.trend_slope_normalized:.3f} drops={st.num_cells_drop_alert}"
            )
