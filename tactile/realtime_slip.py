#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
串口实时采集 + 方案 B 滑移检测（掌 8×6 + 指 2×2 → 10×8）。

用法:
  python realtime_slip.py --port COM3 --baudrate 921600
  python realtime_slip.py --port COM3 --viz
  python realtime_slip.py --port COM3 --mode advance
  python realtime_slip.py --port COM3 --mode simple
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
from typing import Deque, Optional, Union

from slip_detection_simple import (
    HandSlipDetectorSimple,
    SlipDetectionConfig,
    SlipSeverity,
    SlipState,
    risk_band_description,
)
from slip_detection_advance import (
    HandSlipDetectorAdvance,
    SlipDetectionAdvanceConfig,
    SlipStateAdvance,
)
from tactile_serial_reader import HandTactile, HandTactileViewer, TactileGloveReader, list_serial_ports

DetectorMode = str  # "simple" | "advance"


def _estimate_sample_rate_hz(timestamps: Deque[float], default: float = 100.0) -> float:
    if len(timestamps) < 2:
        return default
    dt = timestamps[-1] - timestamps[0]
    if dt <= 0:
        return default
    return max(10.0, min(500.0, (len(timestamps) - 1) / dt))


def make_detector(
    mode: DetectorMode,
    sample_rate_hz: float,
    *,
    simple_cfg: Optional[SlipDetectionConfig] = None,
    advance_cfg: Optional[SlipDetectionAdvanceConfig] = None,
) -> Union[HandSlipDetectorSimple, HandSlipDetectorAdvance]:
    if mode == "advance":
        cfg = advance_cfg or SlipDetectionAdvanceConfig(sample_rate_hz=sample_rate_hz)
        if cfg.sample_rate_hz != sample_rate_hz:
            cfg = replace(cfg, sample_rate_hz=sample_rate_hz)
        return HandSlipDetectorAdvance(cfg)
    cfg = simple_cfg or SlipDetectionConfig(sample_rate_hz=sample_rate_hz)
    if cfg.sample_rate_hz != sample_rate_hz:
        cfg = replace(cfg, sample_rate_hz=sample_rate_hz)
    return HandSlipDetectorSimple(cfg)


def recalibrate_detector(
    detector: Union[HandSlipDetectorSimple, HandSlipDetectorAdvance],
    mode: DetectorMode,
    sample_rate_hz: float,
) -> Union[HandSlipDetectorSimple, HandSlipDetectorAdvance]:
    """采样率变化时按当前配置重建检测器（保留阈值等字段）。"""
    if mode == "advance":
        assert isinstance(detector, HandSlipDetectorAdvance)
        return HandSlipDetectorAdvance(
            replace(detector.config, sample_rate_hz=sample_rate_hz)
        )
    assert isinstance(detector, HandSlipDetectorSimple)
    return HandSlipDetectorSimple(
        replace(detector.config, sample_rate_hz=sample_rate_hz)
    )


def _baseline_frames(
    detector: Union[HandSlipDetectorSimple, HandSlipDetectorAdvance],
) -> int:
    return int(detector.config.baseline_frames)


def _format_status_line(
    mode: DetectorMode,
    st: Union[SlipState, SlipStateAdvance],
    *,
    frame_index: int,
    hand_name: str,
) -> str:
    base = (
        f"[{frame_index}] {hand_name} "
        f"alert={st.slip_alert} sev={st.severity.name} "
        f"risk={st.slip_risk:.2f} ({risk_band_description(st.slip_risk)}) "
        f"drops={st.num_cells_drop_alert} peri={st.num_peripheral_drop_alert}"
    )
    if mode != "advance" or not isinstance(st, SlipStateAdvance):
        return base
    return (
        f"{base} score={st.slip_event_score:.2f} "
        f"redist={st.redistribution_score:.2f} rises={st.num_cells_rise_alert} "
        f"osc={st.oscillation_spike} global={st.global_slip} "
        f"precursor={st.slip_precursor} release={st.likely_release}"
    )


def _format_viz_title(
    mode: DetectorMode,
    hand: HandTactile,
    st: Union[SlipState, SlipStateAdvance],
) -> str:
    side = "左" if hand.sensor_type.value == 1 else "右"
    contact = "接触" if st.is_contact else "未接触"
    tag = "ADV" if mode == "advance" else "SIM"
    line = f"{side}手 [{tag}] {contact} {st.severity.name} risk={st.slip_risk:.2f} alert={st.slip_alert}"
    if mode == "advance" and isinstance(st, SlipStateAdvance):
        line += f" R={st.redistribution_score:.2f}"
        if st.slip_precursor:
            line += " 前兆"
        if st.likely_release:
            line += " 释?"
    return line


def run(
    port: str,
    baudrate: int,
    sample_rate_hz: float,
    viz: bool,
    calibrate_frames: int,
    mode: DetectorMode,
) -> None:
    detector = make_detector(mode, sample_rate_hz)
    viewer: Optional[HandTactileViewer] = None
    if viz:
        viewer = HandTactileViewer()

    timestamps: Deque[float] = deque(maxlen=30)
    mode_label = "进阶 (advance)" if mode == "advance" else "简易 (simple)"

    try:
        with TactileGloveReader(port=port, baudrate=baudrate) as reader:
            print(f"已连接 {port} @ {baudrate}  检测模式: {mode_label}  Hz≈{sample_rate_hz:.0f}")
            if calibrate_frames > 0:
                print(f"前 {calibrate_frames} 帧用于估计采样率…")

            baseline_frames = _baseline_frames(detector)

            for i, frame in enumerate(reader.frames()):
                hand = frame.decode()
                timestamps.append(frame.timestamp)

                if i == calibrate_frames and calibrate_frames > 0:
                    est = _estimate_sample_rate_hz(timestamps, sample_rate_hz)
                    if abs(est - detector.config.sample_rate_hz) > 5:
                        detector = recalibrate_detector(detector, mode, est)
                        baseline_frames = _baseline_frames(detector)
                        print(f"采样率校准为 {est:.1f} Hz")

                st = detector.update(hand)

                if i == baseline_frames and st.baseline_ready:
                    print("基线标定完成，开始滑移检测")

                if viewer is not None:
                    if viewer.fig.number not in viewer._plt.get_fignums():
                        break
                    viewer.update(hand, sequence=frame.sequence)
                    viewer._title.set_text(_format_viz_title(mode, hand, st))
                    viewer.fig.canvas.draw_idle()
                    viewer.fig.canvas.flush_events()
                    viewer._plt.pause(0.001)
                else:
                    show_adv_extra = mode == "advance" and isinstance(st, SlipStateAdvance)
                    interesting = (
                        st.slip_alert
                        or st.severity != SlipSeverity.NONE
                        or (show_adv_extra and (st.slip_precursor or st.likely_release))
                        or i % 50 == 0
                    )
                    if interesting:
                        print(_format_status_line(mode, st, frame_index=i, hand_name=hand.sensor_type.name))

    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        if viewer is not None:
            viewer.close()


def main() -> None:
    ports = list_serial_ports()
    parser = argparse.ArgumentParser(description="触觉手套实时滑移检测")
    parser.add_argument("--port", default=ports[0] if ports else "COM3")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--sample-rate", type=float, default=100.0, help="初始采样率 (Hz)")
    parser.add_argument("--calibrate-frames", type=int, default=50, help="用于估计真实采样率的帧数")
    parser.add_argument("--viz", action="store_true", help="同时显示触觉热力图")
    parser.add_argument(
        "--mode",
        choices=("simple", "advance"),
        default="advance",
        help="滑移算法：simple=slip_detection_simple，advance=slip_detection_advance（四特征）",
    )
    args = parser.parse_args()

    print("可用串口:", ports)
    run(
        port=args.port,
        baudrate=args.baudrate,
        sample_rate_hz=args.sample_rate,
        viz=args.viz,
        calibrate_frames=args.calibrate_frames,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
