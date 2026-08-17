#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live 68-channel tactile viewer for the glove Modbus RTU protocol.

The glove is a Modbus RTU slave (default address 0x01).  This program acts as
the master and repeatedly reads 68 Holding Registers starting at 0x2000, then
serves the values through the existing PALMSCOPE-style web viewer.

Serial defaults follow ``数采手套寄存器地址说明--初版.pdf``: 3,000,000 baud,
8 data bits, no parity, one stop bit.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import termios
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

import live_tactile_68_web as viewer


TOUCH_START_REGISTER = 0x2000
TOUCH_REGISTER_COUNT = 68
TOUCH_STATUS_START_REGISTER = 0x2080
TOUCH_STATUS_REGISTER_COUNT = 8
MODBUS_READ_HOLDING_REGISTERS = 0x03


def crc16_modbus(data: bytes) -> int:
    """Return the Modbus RTU CRC16 value (wire order is low byte first)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc & 0xFFFF


def append_crc(payload: bytes) -> bytes:
    crc = crc16_modbus(payload)
    return payload + bytes((crc & 0xFF, crc >> 8))


def build_read_holding_request(slave: int, start: int, count: int) -> bytes:
    if not 1 <= count <= 125:
        raise ValueError("Modbus register count must be in 1..125")
    if not 0 <= slave <= 247:
        raise ValueError("Modbus slave address must be in 0..247")
    payload = bytes(
        (
            slave,
            MODBUS_READ_HOLDING_REGISTERS,
            (start >> 8) & 0xFF,
            start & 0xFF,
            (count >> 8) & 0xFF,
            count & 0xFF,
        )
    )
    return append_crc(payload)


class ModbusError(RuntimeError):
    pass


class ModbusTactileReader(viewer.SerialTouchReader):
    def __init__(
        self,
        port: str,
        baudrate: int,
        slave: int,
        poll_hz: float,
        response_timeout: float,
        baseline_frames: int,
        noise_gate: float,
        ema_rise: float,
        ema_fall: float,
    ) -> None:
        super().__init__(
            port=port,
            baudrate=baudrate,
            baseline_frames=baseline_frames,
            noise_gate=noise_gate,
            ema_rise=ema_rise,
            ema_fall=ema_fall,
        )
        self.slave = slave
        self.poll_hz = max(0.1, poll_hz)
        self.response_timeout = max(0.01, response_timeout)
        self._rx = bytearray()
        self._request_count = 0
        self._crc_errors = 0
        self._timeouts = 0

    def _read_response(self, fd: int, register_count: int) -> list[int]:
        byte_count = register_count * 2
        normal_length = 3 + byte_count + 2
        deadline = time.monotonic() + self.response_timeout

        while not self.stop_event.is_set():
            # Discard bytes until a response for this slave begins.
            while self._rx and self._rx[0] != self.slave:
                del self._rx[0]

            if len(self._rx) >= 2 and self._rx[1] == (MODBUS_READ_HOLDING_REGISTERS | 0x80):
                if len(self._rx) < 5:
                    pass
                else:
                    frame = bytes(self._rx[:5])
                    del self._rx[:5]
                    if crc16_modbus(frame[:-2]) != int.from_bytes(frame[-2:], "little"):
                        self._crc_errors += 1
                        raise ModbusError("CRC error in Modbus exception response")
                    raise ModbusError(f"Modbus exception 0x{frame[2]:02X}")

            if len(self._rx) >= 3:
                if self._rx[1] != MODBUS_READ_HOLDING_REGISTERS:
                    del self._rx[0]
                    continue
                if self._rx[2] != byte_count:
                    del self._rx[0]
                    continue
                if len(self._rx) >= normal_length:
                    frame = bytes(self._rx[:normal_length])
                    del self._rx[:normal_length]
                    received_crc = int.from_bytes(frame[-2:], "little")
                    if crc16_modbus(frame[:-2]) != received_crc:
                        self._crc_errors += 1
                        raise ModbusError("CRC error in read response")
                    payload = frame[3:-2]
                    return [
                        int.from_bytes(payload[index:index + 2], "big")
                        for index in range(0, len(payload), 2)
                    ]

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._timeouts += 1
                self._rx.clear()
                raise TimeoutError("Modbus response timeout")
            ready, _, _ = select.select([fd], [], [], min(remaining, 0.02))
            if ready:
                chunk = os.read(fd, 4096)
                if chunk:
                    self._rx.extend(chunk)

        raise InterruptedError("reader stopped")

    def _read_holding(self, fd: int, start: int, count: int) -> list[int]:
        # A silent interval and input flush avoid matching a late response to a
        # new request after a timeout.
        request = build_read_holding_request(self.slave, start, count)
        os.write(fd, request)
        termios.tcdrain(fd)
        self._request_count += 1
        return self._read_response(fd, count)

    def _publish_protocol_status(self, values: list[int]) -> None:
        if len(values) != TOUCH_STATUS_REGISTER_COUNT:
            return
        timestamp_us = (
            values[0]
            | (values[1] << 16)
            | (values[2] << 32)
            | (values[3] << 48)
        )
        flags = values[4]
        point_count = values[5]
        capacity = values[6]
        with self.lock:
            self.latest["touch_timestamp_us"] = timestamp_us
            self.latest["touch_status_flags"] = flags
            self.latest["snapshot_valid"] = bool(flags & 0x0001)
            self.latest["touch_valid"] = bool(flags & 0x0002)
            self.latest["point_count"] = point_count
            self.latest["capacity"] = capacity
            self.latest["requests"] = self._request_count
            self.latest["crc_errors"] = self._crc_errors
            self.latest["timeouts"] = self._timeouts

    def run(self) -> None:
        fd: Optional[int] = None
        try:
            fd = self._open_serial()
            self._set_status("Modbus RTU open · polling 0x2000", connected=True)
            period = 1.0 / self.poll_hz
            next_poll = time.monotonic()
            next_status_poll = 0.0
            seq = 0

            while not self.stop_event.is_set():
                now = time.monotonic()
                if now < next_poll:
                    self.stop_event.wait(next_poll - now)
                    continue
                next_poll = max(next_poll + period, now)
                try:
                    raw = self._read_holding(
                        fd, TOUCH_START_REGISTER, TOUCH_REGISTER_COUNT
                    )
                    seq += 1
                    self._apply_frame(seq, raw)
                    with self.lock:
                        self.latest["status"] = "streaming · Modbus RTU 0x03"
                        self.latest["requests"] = self._request_count
                        self.latest["crc_errors"] = self._crc_errors
                        self.latest["timeouts"] = self._timeouts

                    if now >= next_status_poll:
                        status = self._read_holding(
                            fd,
                            TOUCH_STATUS_START_REGISTER,
                            TOUCH_STATUS_REGISTER_COUNT,
                        )
                        self._publish_protocol_status(status)
                        next_status_poll = now + 1.0
                except (TimeoutError, ModbusError) as exc:
                    self._set_status(str(exc), connected=True)
                    self.stop_event.wait(0.05)
        except Exception as exc:
            self._set_status(f"serial error: {exc}", connected=False)
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve a live 68-channel Modbus RTU tactile heatmap."
    )
    parser.add_argument("--port", default="/dev/ttyACM0", help="USB/RS485 serial device")
    parser.add_argument("--baudrate", type=int, default=3_000_000)
    parser.add_argument("--slave", type=lambda value: int(value, 0), default=0x01)
    parser.add_argument("--poll_hz", type=float, default=100.0)
    parser.add_argument("--response_timeout", type=float, default=0.10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--web_port", type=int, default=8790)
    parser.add_argument(
        "--hand_asset",
        default="/home/lenovo/Downloads/ourhost/assets/hand_live.png",
    )
    parser.add_argument("--baseline_frames", type=int, default=24)
    parser.add_argument("--noise_gate", type=float, default=1.5)
    parser.add_argument("--ema_rise", type=float, default=0.45)
    parser.add_argument("--ema_fall", type=float, default=0.22)
    args = parser.parse_args()

    hand_asset = Path(args.hand_asset).expanduser().resolve()
    if not hand_asset.exists():
        raise SystemExit(f"hand asset not found: {hand_asset}")

    reader = ModbusTactileReader(
        port=args.port,
        baudrate=args.baudrate,
        slave=args.slave,
        poll_hz=args.poll_hz,
        response_timeout=args.response_timeout,
        baseline_frames=max(1, args.baseline_frames),
        noise_gate=args.noise_gate,
        ema_rise=args.ema_rise,
        ema_fall=args.ema_fall,
    )
    reader.start()

    viewer.TactileRequestHandler.reader = reader
    viewer.TactileRequestHandler.hand_asset = hand_asset
    viewer.TactileRequestHandler.html = viewer.make_html(max(1, args.baseline_frames))
    server = ThreadingHTTPServer((args.host, args.web_port), viewer.TactileRequestHandler)

    def stop(_signum: int, _frame: Any) -> None:
        reader.stop_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(
        json.dumps(
            {
                "url": f"http://{args.host}:{args.web_port}",
                "serial": args.port,
                "baudrate": args.baudrate,
                "protocol": "Modbus RTU",
                "slave": args.slave,
                "read": "0x2000 + 68 holding registers",
                "poll_hz": args.poll_hz,
            },
            ensure_ascii=False,
        )
    )
    try:
        server.serve_forever()
    finally:
        reader.stop_event.set()
        reader.join(timeout=1.0)
        server.server_close()


if __name__ == "__main__":
    main()
