"""
进阶规则式滑移检测：融合四条物理特征。

1. 力重分布：局部骤降 + 其它点位骤升（redistribution_score）
2. 瞬态震荡：总力高通残差短/长窗能量比（oscillation_spike）
3. 动摩擦转换：总力短时整体相对下降（global_slip）
4. 疲劳/释放前兆：缓慢负趋势 + slip_risk（slip_precursor）；整体缓释抑制误报（likely_release）

复用 slip_detection_simple 的网格拼接、mask、连通域过滤等工具。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Deque, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from tactile_serial_reader import HandTactile

import numpy as np

from slip_detection_simple import (
    HAND_SLIP_FINGER_COLS,
    HAND_SLIP_GRID_SHAPE,
    HAND_SLIP_NUM_FINGERS,
    HAND_SLIP_PALM_ROW_START,
    SlipSeverity,
    _disjoint_force_variance,
    _major_mask,
    _peripheral_mask,
    build_hand_active_mask,
    build_hand_slip_masks,
    filter_drop_clusters,
    hand_to_slip_grid,
    risk_band_description,
)

__all__ = [
    "SlipDetectionAdvanceConfig",
    "SlipStateAdvance",
    "PalmSlipDetectorAdvance",
    "HandSlipDetectorAdvance",
    "SlipSeverity",
    "risk_band_description",
]


@dataclass
class SlipDetectionAdvanceConfig:
    sample_rate_hz: float = 100.0
    eps: float = 1e-6

    # 骤降 / 骤升（特征 ①）
    drop_window_s: float = 0.05
    drop_ratio_alert: float = 0.20
    drop_ratio_severe: float = 0.30
    rise_ratio_alert: float = 0.15
    redistribution_min_ratio: float = 0.35
    drop_abs_scale: float = 0.08
    rise_abs_scale: float = 0.05
    drop_abs_floor: float = 8.0
    rise_abs_floor: float = 6.0
    min_drop_cluster_cells: int = 2

    # 震荡（特征 ②）
    trend_lowpass_window_s: float = 0.10
    osc_short_s: float = 0.08
    osc_long_s: float = 0.50
    osc_spike_ratio: float = 2.0
    variance_short_s: float = 0.15
    variance_long_s: float = 1.0
    variance_spike_ratio: float = 2.5

    # 全局摩擦骤降（特征 ③）
    global_drop_ratio: float = 0.10
    global_drop_abs_scale: float = 0.06
    global_drop_abs_floor: float = 12.0

    # 前兆 / 趋势（特征 ④）
    trend_window_s: float = 1.0
    peak_window_s: float = 1.0
    trend_slope_norm_threshold: float = -0.05
    precursor_risk_threshold: float = 0.18
    precursor_trend_norm: float = -0.03
    release_redist_max: float = 0.20

    # 事件融合
    weight_redistribution: float = 0.45
    weight_oscillation: float = 0.30
    weight_global_drop: float = 0.35
    alert_score_threshold: float = 1.15
    require_redistribution_for_alert: bool = True

    # 接触 / 基线 / 滞回
    contact_force_threshold: float = 80.0
    baseline_frames: int = 30
    alert_hold_frames: int = 2
    alert_release_frames: int = 3
    input_smooth_frames: int = 2


@dataclass
class SlipStateAdvance:
    """与 SlipState 兼容的核心字段 + 进阶诊断量。"""

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

    redistribution_score: float = 0.0
    drop_mass: float = 0.0
    rise_mass: float = 0.0
    num_cells_rise_alert: int = 0
    global_rel_drop: float = 0.0
    global_slip: bool = False
    oscillation_spike: bool = False
    oscillation_energy_short: float = 0.0
    oscillation_energy_long: float = 0.0
    var_spike: bool = False
    slip_precursor: bool = False
    likely_release: bool = False
    slip_event_score: float = 0.0


def _rms(seg: np.ndarray) -> float:
    if len(seg) < 1:
        return 0.0
    return float(np.sqrt(np.mean(np.square(seg))))


def _disjoint_segment_rms(
    arr: np.ndarray,
    short_w: int,
    long_w: int,
    eps: float,
) -> Tuple[float, float]:
    n = len(arr)
    if n < short_w + 1:
        return 0.0, eps
    e_short = _rms(arr[-short_w:])
    if n >= long_w + short_w:
        e_long = _rms(arr[-(long_w + short_w) : -short_w])
    else:
        seg = arr[: max(1, n - short_w)]
        e_long = _rms(seg) if len(seg) >= 1 else eps
    return e_short, max(e_long, eps)


class PalmSlipDetectorAdvance:
    """四特征融合滑移检测。"""

    def __init__(
        self,
        height: int,
        width: int,
        config: Optional[SlipDetectionAdvanceConfig] = None,
        *,
        peripheral_flat: Optional[np.ndarray] = None,
        major_flat: Optional[np.ndarray] = None,
        active_flat: Optional[np.ndarray] = None,
        spatial_palm_rows: Optional[int] = None,
        cluster_layout: str = "stacked",
    ) -> None:
        if height < 2 or width < 2:
            raise ValueError("手掌阵列至少为 2×2")
        self.h = height
        self.w = width
        self.s = height * width
        self.cfg = config or SlipDetectionAdvanceConfig()
        fs = self.cfg.sample_rate_hz

        self._drop_w = max(2, int(round(fs * self.cfg.drop_window_s)))
        self._baseline_w = max(1, int(self.cfg.baseline_frames))
        self._var_short_w = max(2, int(round(fs * self.cfg.variance_short_s)))
        self._var_long_w = max(self._var_short_w + 1, int(round(fs * self.cfg.variance_long_s)))
        self._trend_w = max(3, int(round(fs * self.cfg.trend_window_s)))
        self._peak_w = max(2, int(round(fs * self.cfg.peak_window_s)))
        self._osc_short_w = max(2, int(round(fs * self.cfg.osc_short_s)))
        self._osc_long_w = max(self._osc_short_w + 1, int(round(fs * self.cfg.osc_long_s)))
        self._trend_lp_w = max(2, int(round(fs * self.cfg.trend_lowpass_window_s)))

        max_hist = max(
            self._drop_w + 2,
            self._var_long_w + self._var_short_w + 5,
            self._trend_w + 5,
            self._peak_w + 5,
            self._osc_long_w + self._osc_short_w + 5,
        )
        self._max_hist = max_hist
        self._spatial_palm_rows = spatial_palm_rows
        self._cluster_layout = cluster_layout

        self._frames: Deque[np.ndarray] = deque(maxlen=max_hist)
        self._total_f: Deque[float] = deque(maxlen=max_hist)
        self._total_f_lp: Deque[float] = deque(maxlen=max_hist)
        self._hf_tf: Deque[float] = deque(maxlen=max_hist)
        self._smooth_buf: Deque[np.ndarray] = deque(
            maxlen=max(1, self.cfg.input_smooth_frames) if self.cfg.input_smooth_frames > 0 else 1
        )

        if peripheral_flat is not None or major_flat is not None:
            if peripheral_flat is None or major_flat is None:
                raise ValueError("peripheral_flat 与 major_flat 须同时提供")
            peri = np.asarray(peripheral_flat, dtype=bool).reshape(-1)
            maj = np.asarray(major_flat, dtype=bool).reshape(-1)
            self._peri_flat = peri
            self._major_flat = maj
        else:
            self._peri_flat = _peripheral_mask(height, width).reshape(-1)
            self._major_flat = _major_mask(height, width).reshape(-1)

        if active_flat is not None:
            act = np.asarray(active_flat, dtype=bool).reshape(-1)
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
        self._ema_tf: Optional[float] = None

    def reset(self) -> None:
        self._frames.clear()
        self._total_f.clear()
        self._total_f_lp.clear()
        self._hf_tf.clear()
        self._smooth_buf.clear()
        self._baseline_buf.clear()
        self._baseline = None
        self._baseline_ready = False
        self._alert_latched = False
        self._alert_streak = 0
        self._clear_streak = 0
        self._ema_tf = None

    def _flatten(self, frame: Union[np.ndarray, List[float]]) -> np.ndarray:
        x = np.asarray(frame, dtype=np.float64).reshape(-1)
        if x.size != self.s:
            raise ValueError(f"期望 {self.s} 个通道，得到 {x.size}")
        return x

    def _maybe_smooth(self, flat: np.ndarray) -> np.ndarray:
        k = self.cfg.input_smooth_frames
        if k <= 1:
            return flat
        self._smooth_buf.append(flat.copy())
        if len(self._smooth_buf) < k:
            return flat
        return np.mean(np.stack(self._smooth_buf, axis=0), axis=0)

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

    def _update_trend_channel(self, tf: float) -> float:
        """EMA 低通总力，返回高通残差。"""
        alpha = 2.0 / (self._trend_lp_w + 1.0)
        if self._ema_tf is None:
            self._ema_tf = tf
        else:
            self._ema_tf = (1.0 - alpha) * self._ema_tf + alpha * tf
        self._total_f_lp.append(self._ema_tf)
        hf = tf - self._ema_tf
        self._hf_tf.append(hf)
        return hf

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

    def _scaled_threshold(self, peak: float, scale: float, floor: float) -> float:
        return max(floor, scale * max(peak, self.cfg.eps))

    def _safe_state(
        self,
        tf: float,
        *,
        is_contact: bool,
        baseline_ready: bool,
        peak_1s: Optional[float] = None,
    ) -> SlipStateAdvance:
        peak = peak_1s if peak_1s is not None else tf
        return SlipStateAdvance(
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

    def update(self, frame: Union[np.ndarray, List[float]]) -> SlipStateAdvance:
        flat_raw = self._flatten(frame)
        flat_raw = self._maybe_smooth(flat_raw)

        if not self._baseline_ready:
            self._learn_baseline(flat_raw)
            return self._safe_state(
                self._total_force_active(flat_raw),
                is_contact=False,
                baseline_ready=False,
            )

        flat = self._apply_baseline(flat_raw)
        tf = self._total_force_active(flat)
        cfg = self.cfg
        eps = cfg.eps
        is_contact = tf >= cfg.contact_force_threshold

        self._update_trend_channel(tf)
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

        if len(self._frames) < self._drop_w + 1:
            return self._safe_state(tf, is_contact=True, baseline_ready=True, peak_1s=tf)

        tf_arr = np.array(self._total_f, dtype=np.float64)
        peak_1s = float(np.max(tf_arr[-self._peak_w :]))
        drop_abs_th = self._scaled_threshold(peak_1s, cfg.drop_abs_scale, cfg.drop_abs_floor)
        rise_abs_th = self._scaled_threshold(peak_1s, cfg.rise_abs_scale, cfg.rise_abs_floor)
        global_abs_th = self._scaled_threshold(
            peak_1s, cfg.global_drop_abs_scale, cfg.global_drop_abs_floor
        )

        arr = np.stack(list(self._frames), axis=0)
        ref_slice = arr[-(self._drop_w + 1) : -1]
        if ref_slice.shape[0] == 0:
            ref_slice = arr[-2:-1]
        f_ref = np.max(ref_slice, axis=0)
        f1 = arr[-1]
        abs_drop = f_ref - f1
        rel_drop = abs_drop / np.maximum(f_ref, eps)
        abs_rise = f1 - f_ref
        rel_rise = abs_rise / np.maximum(f_ref, eps)

        alert_drop_raw = (
            (rel_drop > cfg.drop_ratio_alert)
            & (abs_drop > drop_abs_th)
            & self._active_flat
        )
        rise_alert_raw = (
            (rel_rise > cfg.rise_ratio_alert)
            & (abs_rise > rise_abs_th)
            & self._active_flat
        )
        severe_raw = (
            (rel_drop > cfg.drop_ratio_severe)
            & (abs_drop > drop_abs_th)
            & self._major_flat
        )
        min_cluster = max(1, int(cfg.min_drop_cluster_cells))

        alert_drop = filter_drop_clusters(
            alert_drop_raw,
            self.h,
            self.w,
            self._active_flat,
            palm_rows=self._spatial_palm_rows,
            min_cluster=min_cluster,
            layout=self._cluster_layout,
        )
        rise_alert = filter_drop_clusters(
            rise_alert_raw,
            self.h,
            self.w,
            self._active_flat,
            palm_rows=self._spatial_palm_rows,
            min_cluster=min_cluster,
            layout=self._cluster_layout,
        )
        severe_drop_major = (
            filter_drop_clusters(
                severe_raw,
                self.h,
                self.w,
                self._active_flat,
                palm_rows=self._spatial_palm_rows,
                min_cluster=min_cluster,
                layout=self._cluster_layout,
            )
            & self._major_flat
        )

        drop_mass = float(np.sum(abs_drop[alert_drop]))
        rise_mass = float(np.sum(abs_rise[rise_alert]))
        redistribution_score = float(
            np.clip(rise_mass / max(drop_mass, eps), 0.0, 1.0)
        )
        redistribution_ok = redistribution_score >= cfg.redistribution_min_ratio

        num_cells_drop_alert = int(np.sum(alert_drop))
        num_cells_rise_alert = int(np.sum(rise_alert))
        num_major_drop_severe = int(np.sum(severe_drop_major))
        num_peripheral_drop_alert = int(np.sum(alert_drop & self._peri_flat))

        tf_ref = float(np.max(tf_arr[-(self._drop_w + 1) : -1])) if len(tf_arr) > 1 else tf
        global_rel_drop = float((tf_ref - tf) / max(tf_ref, eps))
        global_slip = (global_rel_drop > cfg.global_drop_ratio) and (
            (tf_ref - tf) > global_abs_th
        )

        hf_arr = np.array(self._hf_tf, dtype=np.float64)
        osc_short, osc_long = _disjoint_segment_rms(
            hf_arr, self._osc_short_w, self._osc_long_w, eps
        )
        oscillation_spike = osc_short > cfg.osc_spike_ratio * osc_long

        short_var, long_var = _disjoint_force_variance(
            tf_arr, self._var_short_w, self._var_long_w, eps
        )
        var_spike = short_var > cfg.variance_spike_ratio * long_var
        texture_spike = oscillation_spike or (
            var_spike and (redistribution_ok or num_cells_drop_alert > 0)
        )

        trend_slope = 0.0
        trend_slope_norm = 0.0
        if len(tf_arr) >= self._trend_w:
            y = tf_arr[-self._trend_w :]
            x = np.arange(len(y), dtype=np.float64)
            slope, _ = np.polyfit(x, y, 1)
            trend_slope = float(slope)
            trend_slope_norm = float(
                slope * (self._trend_w - 1) / max(peak_1s, eps)
            )

        risk_peak = 1.0 - tf / max(peak_1s, eps)
        slip_risk = float(np.clip(max(risk_peak, global_rel_drop), 0.0, 1.0))

        slip_precursor = (
            trend_slope_norm < cfg.precursor_trend_norm
            and slip_risk > cfg.precursor_risk_threshold
            and not texture_spike
            and num_cells_drop_alert == 0
        )

        likely_release = (
            global_slip
            and redistribution_score < cfg.release_redist_max
            and (slip_precursor or trend_slope_norm < cfg.trend_slope_norm_threshold)
        )

        w1, w2, w3 = (
            cfg.weight_redistribution,
            cfg.weight_oscillation,
            cfg.weight_global_drop,
        )
        slip_event_score = (
            w1 * redistribution_score
            + w2 * float(texture_spike)
            + w3 * float(np.clip(global_rel_drop / max(cfg.global_drop_ratio, eps), 0.0, 1.5))
        )

        has_spatial_event = num_cells_drop_alert > 0 and (
            redistribution_ok or not cfg.require_redistribution_for_alert
        )
        raw_alert = (
            slip_event_score >= cfg.alert_score_threshold
            and (has_spatial_event or global_slip)
            and not likely_release
        )
        slip_alert = self._update_alert_hysteresis(raw_alert)

        severity = SlipSeverity.NONE
        if num_major_drop_severe >= 2 and redistribution_ok:
            severity = SlipSeverity.SEVERE
        elif slip_alert and (num_major_drop_severe >= 1 or global_slip):
            severity = SlipSeverity.SEVERE
        elif num_cells_drop_alert >= 3 and redistribution_ok:
            severity = SlipSeverity.SEVERE
        elif num_peripheral_drop_alert == 1 and num_cells_drop_alert == 1 and not redistribution_ok:
            severity = SlipSeverity.MILD
        elif slip_alert or (num_cells_drop_alert >= 2 and redistribution_ok):
            severity = SlipSeverity.MODERATE
        elif slip_precursor or (
            trend_slope_norm < cfg.trend_slope_norm_threshold and slip_risk > 0.25
        ):
            severity = SlipSeverity.MODERATE
        elif likely_release:
            severity = SlipSeverity.MILD

        return SlipStateAdvance(
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
            redistribution_score=redistribution_score,
            drop_mass=drop_mass,
            rise_mass=rise_mass,
            num_cells_rise_alert=num_cells_rise_alert,
            global_rel_drop=global_rel_drop,
            global_slip=global_slip,
            oscillation_spike=oscillation_spike,
            oscillation_energy_short=osc_short,
            oscillation_energy_long=osc_long,
            var_spike=var_spike,
            slip_precursor=slip_precursor,
            likely_release=likely_release,
            slip_event_score=slip_event_score,
        )


class HandSlipDetectorAdvance:
    """左手 68 点手套：HandTactile → 10×8 + 四特征检测。"""

    def __init__(self, config: Optional[SlipDetectionAdvanceConfig] = None) -> None:
        h, w = HAND_SLIP_GRID_SHAPE
        peri, major = build_hand_slip_masks()
        active = build_hand_active_mask()
        self._detector = PalmSlipDetectorAdvance(
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
    def config(self) -> SlipDetectionAdvanceConfig:
        return self._detector.cfg

    def reset(self) -> None:
        self._detector.reset()

    def update(self, hand: "HandTactile") -> SlipStateAdvance:
        return self._detector.update(hand_to_slip_grid(hand))


if __name__ == "__main__":
    from slip_detection_simple import PalmSlipDetectorSimple, SlipDetectionConfig

    H, W = 8, 8
    rng = np.random.default_rng(1)
    base = np.abs(rng.normal(1.0, 0.05, size=(H, W))) + 0.8

    det_adv = PalmSlipDetectorAdvance(H, W, SlipDetectionAdvanceConfig(sample_rate_hz=100.0))
    det_sim = PalmSlipDetectorSimple(H, W, SlipDetectionConfig(sample_rate_hz=100.0))

    for _ in range(60):
        det_adv.update(base + rng.normal(0, 0.02, size=(H, W)))
        det_sim.update(base + rng.normal(0, 0.02, size=(H, W)))

    print("模拟滑移：边缘骤降 + 对侧略升 + 抖动…")
    for t in range(30):
        f = base.copy()
        if t > 3:
            f[0, 2] *= 0.3
            f[3, 5] *= 1.4
            f += rng.normal(0, 0.12, size=(H, W))
        st = det_adv.update(np.maximum(f, 0.0))
        if t % 5 == 0:
            print(
                f"  t={t:02d} alert={st.slip_alert} score={st.slip_event_score:.2f} "
                f"redist={st.redistribution_score:.2f} osc={st.oscillation_spike} "
                f"global={st.global_slip} precursor={st.slip_precursor} release={st.likely_release}"
            )
