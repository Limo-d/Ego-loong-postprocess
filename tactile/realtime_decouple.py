#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
串口实时采集 + 弯曲解耦（curved_decouple_simple 姿态库）。

流程：decode → decouple_hand → 输出残差（控制台或解耦前后对比可视化）

用法:
  python realtime_decouple.py --port COM12 --calib bend_decouple_calib.npz
  python realtime_decouple.py --port COM12 --calib bend_decouple_calib.npz --viz
  python realtime_decouple.py --port COM12 --viz

标定文件由 curved_decouple_simple.py 的 record + build 生成。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

from curved_decouple_simple import (
    BendDecoupleCalib,
    DecoupleCompareViewer,
    POSE_MODE_JOINT,
    _metrics_line,
    decouple_hand,
    load_calib,
)
from tactile_serial_reader import TactileGloveReader, list_serial_ports


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


def run(
    port: str,
    baudrate: int,
    calib_path: Union[str, Path],
    *,
    viz: bool = False,
    print_every: int = 50,
    raw_vmax: float = 500.0,
    resid_vmax: float = 50.0,
) -> None:
    calib: BendDecoupleCalib = load_calib(calib_path)
    viewer: Optional[DecoupleCompareViewer] = None
    if viz:
        viewer = DecoupleCompareViewer(calib, raw_vmax=raw_vmax, resid_vmax=resid_vmax)

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
                f"  弯曲解耦: {calib_path}\n"
                f"  姿态代理: {calib.pose_mode}  |  匹配: {calib.atlas_mode.upper()}  "
                f"|  库 {calib.atlas_bends.shape[0]} 帧"
            )
            if viz:
                print("  可视化: 解耦前 / 解耦后对比（关闭窗口退出）")
            else:
                print(f"  控制台: 每 {print_every} 帧输出一次指标，Ctrl+C 停止")

            for i, frame in enumerate(reader.frames()):
                hand_raw = frame.decode()
                if int(hand_raw.sensor_type) != calib.sensor_type:
                    raise ValueError(
                        f"手套手型 {hand_raw.sensor_type.name} 与标定 "
                        f"(sensor_type={calib.sensor_type}) 不一致，请重新 build 标定文件"
                    )
                hand_dec = decouple_hand(hand_raw, calib)

                if viewer is not None:
                    if viewer.fig.number not in viewer._plt.get_fignums():
                        break
                    line = _metrics_line(hand_raw, hand_dec, calib)
                    line += f"  seq={frame.sequence}"
                    viewer.update(hand_raw, hand_dec, metrics_line=line)
                elif print_every > 0 and i % print_every == 0:
                    line = _metrics_line(hand_raw, hand_dec, calib)
                    print(f"[{i:5d}] {line}  seq={frame.sequence}")

    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        if viewer is not None:
            viewer.close()


def main() -> None:
    ports = list_serial_ports()
    parser = argparse.ArgumentParser(
        description="串口实时弯曲解耦",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python realtime_decouple.py --port COM12 --calib bend_decouple_calib.npz
  python realtime_decouple.py --port COM12 --calib bend_decouple_calib.npz --viz
  python realtime_decouple.py --port COM12 --viz --raw-vmax 400 --resid-vmax 30

标定:
  python curved_decouple_simple.py record --port COM12 --out bend_sweep.npz
  python curved_decouple_simple.py build --sweep bend_sweep.npz --out bend_decouple_calib.npz --method nn
        """,
    )
    parser.add_argument("--port", default=ports[0] if ports else "COM3")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument(
        "--calib",
        default=_default_calib_path(),
        help="弯曲解耦标定 npz（默认: 当前目录或脚本目录下的 bend_decouple_calib.npz）",
    )
    parser.add_argument("--viz", action="store_true", help="显示解耦前/后对比热力图")
    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="无 --viz 时每隔多少帧打印一行（0=不打印）",
    )
    parser.add_argument("--raw-vmax", type=float, default=500.0, help="可视化：原始读数色标上限 (kPa)")
    parser.add_argument("--resid-vmax", type=float, default=50.0, help="可视化：残差色标上限 (kPa)")
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
        viz=args.viz,
        print_every=args.print_every,
        raw_vmax=args.raw_vmax,
        resid_vmax=args.resid_vmax,
    )


if __name__ == "__main__":
    main()
