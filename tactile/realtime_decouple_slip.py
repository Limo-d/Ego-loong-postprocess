#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
串口实时采集 + 弯曲解耦（1-NN 姿态库）+ 方案 B 滑移检测。

流程：decode → decouple_hand → HandSlipDetector*.update

用法:
  python realtime_decouple_slip.py --port COM12 --calib bend_decouple_calib.npz
  python realtime_decouple_slip.py --port COM12 --calib bend_decouple_calib.npz --viz
  python realtime_decouple_slip.py --port COM12 --calib bend_decouple_calib.npz --mode simple
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Union

from curved_decouple_simple import BendDecoupleCalib, POSE_MODE_JOINT, decouple_hand, load_calib
from realtime_slip import (
    DetectorMode,
    _baseline_frames,
    _estimate_sample_rate_hz,
    _format_status_line,
    _format_viz_title,
    make_detector,
    recalibrate_detector,
)
from slip_detection_advance import SlipStateAdvance
from slip_detection_simple import SlipSeverity, SlipState
from tactile_serial_reader import HandTactileViewer, TactileGloveReader, list_serial_ports


def run(
    port: str,
    baudrate: int,
    calib_path: Union[str, Path],
    sample_rate_hz: float,
    viz: bool,
    calibrate_frames: int,
    mode: DetectorMode,
) -> None:
    calib: BendDecoupleCalib = load_calib(calib_path)
    detector = make_detector(mode, sample_rate_hz)
    viewer: Optional[HandTactileViewer] = None
    if viz:
        viewer = HandTactileViewer()

    timestamps: Deque[float] = deque(maxlen=30)
    mode_label = "进阶 (advance)" if mode == "advance" else "简易 (simple)"
    require_solved = calib.pose_mode == POSE_MODE_JOINT
    if require_solved:
        print("  等待 SOLVED 就绪（跳过 IMU 预热期含 MISS 的帧）…")

    try:
        with TactileGloveReader(
            port=port,
            baudrate=baudrate,
            require_complete_solved=require_solved,
        ) as reader:
            print(
                f"已连接 {port} @ {baudrate}\n"
                f"  弯曲解耦: {calib_path}  ({calib.pose_mode}, {calib.atlas_mode}, "
                f"{calib.atlas_bends.shape[0]} 帧)\n"
                f"  滑移检测: {mode_label}  Hz≈{sample_rate_hz:.0f}"
            )
            if calibrate_frames > 0:
                print(f"前 {calibrate_frames} 帧用于估计采样率…")

            baseline_frames = _baseline_frames(detector)

            for i, frame in enumerate(reader.frames()):
                hand_raw = frame.decode()
                if int(hand_raw.sensor_type) != calib.sensor_type:
                    raise ValueError(
                        f"手套手型 {hand_raw.sensor_type.name} 与标定 "
                        f"(sensor_type={calib.sensor_type}) 不一致，请重新 build 标定文件"
                    )
                hand = decouple_hand(hand_raw, calib)
                timestamps.append(frame.timestamp)

                if i == calibrate_frames and calibrate_frames > 0:
                    est = _estimate_sample_rate_hz(timestamps, sample_rate_hz)
                    if abs(est - detector.config.sample_rate_hz) > 5:
                        detector = recalibrate_detector(detector, mode, est)
                        baseline_frames = _baseline_frames(detector)
                        print(f"采样率校准为 {est:.1f} Hz")

                st = detector.update(hand)

                if i == baseline_frames and st.baseline_ready:
                    print("滑移基线标定完成（输入为解耦后残差）")

                if viewer is not None:
                    if viewer.fig.number not in viewer._plt.get_fignums():
                        break
                    viewer.update(hand, sequence=frame.sequence)
                    viewer._title.set_text(
                        _format_viz_title(mode, hand, st) + "  [解耦后]"
                    )
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
                        print(
                            _format_status_line(
                                mode,
                                st,
                                frame_index=i,
                                hand_name=hand_raw.sensor_type.name,
                            )
                        )

    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        if viewer is not None:
            viewer.close()


def _default_calib_path() -> str:
    """IDE 直接运行时默认查找项目目录下的标定文件。"""
    script_dir = Path(__file__).resolve().parent
    for candidate in (
        Path.cwd() / "bend_decouple_calib.npz",
        script_dir / "bend_decouple_calib.npz",
    ):
        if candidate.is_file():
            return str(candidate)
    return str(script_dir / "bend_decouple_calib.npz")


def main() -> None:
    ports = list_serial_ports()
    parser = argparse.ArgumentParser(
        description="弯曲解耦 + 实时滑移检测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python realtime_decouple_slip.py --port COM12 --calib bend_decouple_calib.npz
  python realtime_decouple_slip.py --port COM12 --calib bend_decouple_calib.npz --viz --mode advance

标定文件由 curved_decouple_simple.py 的 record + build 生成。
        """,
    )
    parser.add_argument("--port", default=ports[0] if ports else "COM3")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument(
        "--calib",
        default=_default_calib_path(),
        help="弯曲解耦标定 npz（默认: 当前目录或脚本目录下的 bend_decouple_calib.npz）",
    )
    parser.add_argument("--sample-rate", type=float, default=100.0, help="初始采样率 (Hz)")
    parser.add_argument("--calibrate-frames", type=int, default=50, help="用于估计真实采样率的帧数")
    parser.add_argument("--viz", action="store_true", help="显示解耦后触觉热力图")
    parser.add_argument(
        "--mode",
        choices=("simple", "advance"),
        default="advance",
        help="滑移算法：simple / advance",
    )
    args = parser.parse_args()

    calib_path = Path(args.calib)
    if not calib_path.is_file():
        raise SystemExit(
            f"找不到标定文件: {calib_path.resolve()}\n"
            "请先执行:\n"
            "  python curved_decouple_simple.py record --port COM12 --out bend_sweep.npz\n"
            "  python curved_decouple_simple.py build --sweep bend_sweep.npz --out bend_decouple_calib.npz"
        )

    print("可用串口:", ports)
    print(f"使用标定: {calib_path.resolve()}")
    run(
        port=args.port,
        baudrate=args.baudrate,
        calib_path=calib_path,
        sample_rate_hz=args.sample_rate,
        viz=args.viz,
        calibrate_frames=args.calibrate_frames,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
