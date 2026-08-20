#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract sampler ROS2 bags into the existing preprocess/all_data layout.

This script is intentionally ROS-facing. Run it from a shell that has sourced:

  source /opt/ros/jazzy/setup.bash
  source /path/to/hand_msg_ws/install/setup.bash

It writes the minimal frame layout consumed by the existing HaMeR/filtering
scripts:

  <session>/preprocess/all_data/000000/rgb.png
  <session>/preprocess/all_data/000000/depth.png
  <session>/preprocess/all_data/000000/depth_aligned.png
  <session>/preprocess/all_data/000000/aria_cam_rgb.json
"""

from __future__ import annotations

import argparse
import bisect
import ctypes
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
import warnings
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import rosbag2_py
from builtin_interfaces.msg import Time
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


RGB_TOPIC = "/factor_perception/rgb/image_rect/compressed"
DEPTH_TOPIC = "/factor_perception/depth/image_rect"
DEPTH_REGISTERED_COMPRESSED_TOPIC = "/factor_perception/depth_registered/image_rect/compressedDepth"
DEPTH_TOPICS = (DEPTH_TOPIC, DEPTH_REGISTERED_COMPRESSED_TOPIC)
RGB_INFO_TOPIC = "/factor_perception/rgb/camera_info"
DEPTH_INFO_TOPIC = "/factor_perception/depth/camera_info"
DEPTH_REGISTERED_INFO_TOPIC = "/factor_perception/depth_registered/camera_info"
DEPTH_INFO_TOPICS = (DEPTH_INFO_TOPIC, DEPTH_REGISTERED_INFO_TOPIC)
ODOM_TOPIC = "/factor_perception/odom"
TF_TOPIC = "/tf"
TF_STATIC_TOPIC = "/tf_static"
HAND_FRAME_TOPIC = "/hand_frame"
GLOVE_TOPIC = "/glove"
RESOLVE_DRIVER_DEFAULT = Path("/home/lenovo/Retarget/data/ros_ws/resolve_check/resolve_driver")
_NATIVE_RVL = None
_NATIVE_RVL_ATTEMPTED = False

DEFAULT_T_RGB_DEPTH = np.array(
    [
        [0.9999770522, -0.0000959671, -0.0067714620, -0.0380730700],
        [0.0001288063, 0.9999882579, 0.0048491457, 0.0003812458],
        [0.0067709172, -0.0048499065, 0.9999653697, -0.0000050605],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass
class StampedMsg:
    topic: str
    bag_time_ns: int
    msg_time_ns: int
    msg: Any


def time_to_ns(stamp: Time) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def header_stamp_ns(msg: Any, fallback_ns: int) -> int:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return int(fallback_ns)
    ns = time_to_ns(stamp)
    return ns if ns > 0 else int(fallback_ns)


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [safe_list(v) for v in value]
    return value


def camera_info_to_dict(msg: Any) -> Dict[str, Any]:
    return {
        "height": int(msg.height),
        "width": int(msg.width),
        "distortion_model": str(msg.distortion_model),
        "d": [float(x) for x in msg.d],
        "k": np.asarray(msg.k, dtype=np.float64).reshape(3, 3).tolist(),
        "r": np.asarray(msg.r, dtype=np.float64).reshape(3, 3).tolist(),
        "p": np.asarray(msg.p, dtype=np.float64).reshape(3, 4).tolist(),
        "frame_id": str(msg.header.frame_id),
        "stamp_ns": time_to_ns(msg.header.stamp),
    }


def compressed_to_bgr(msg: Any) -> np.ndarray:
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode compressed RGB image")
    return img


def _decode_rvl_python(payload: bytes, width: int, height: int) -> np.ndarray:
    padded_payload = payload + b"\0" * ((-len(payload)) % 4)
    words = np.frombuffer(padded_payload, dtype="<u4")
    word_idx = 0
    word = 0
    nibbles_left = 0

    def decode_vle() -> int:
        nonlocal word_idx, word, nibbles_left
        value = 0
        shift = 0
        while True:
            if nibbles_left == 0:
                if word_idx >= len(words):
                    raise ValueError("Truncated RVL depth word stream")
                word = int(words[word_idx])
                word_idx += 1
                nibbles_left = 8
            nibble = (word >> 28) & 0xF
            word = (word << 4) & 0xFFFFFFFF
            nibbles_left -= 1
            value |= (nibble & 0x7) << shift
            if not (nibble & 0x8):
                return value
            shift += 3
            if shift > 63:
                raise ValueError("Invalid RVL variable-length integer")

    pixel_count = int(width) * int(height)
    depth = np.empty(pixel_count, dtype=np.uint16)
    pixel_idx = 0
    previous = 0
    uint16_max = np.iinfo(np.uint16).max
    while pixel_idx < pixel_count:
        zeros = decode_vle()
        if zeros > pixel_count - pixel_idx:
            raise ValueError("RVL zero run exceeds depth image size")
        depth[pixel_idx : pixel_idx + zeros] = 0
        pixel_idx += zeros

        nonzeros = decode_vle()
        if nonzeros > pixel_count - pixel_idx:
            raise ValueError("RVL nonzero run exceeds depth image size")
        for _ in range(nonzeros):
            positive = decode_vle()
            delta = (positive >> 1) ^ -(positive & 1)
            previous += delta
            if previous < 0 or previous > uint16_max:
                raise ValueError(f"RVL decoded value outside uint16 range: {previous}")
            depth[pixel_idx] = previous
            pixel_idx += 1

    return depth.reshape(int(height), int(width))


def _load_native_rvl():
    global _NATIVE_RVL, _NATIVE_RVL_ATTEMPTED
    if _NATIVE_RVL_ATTEMPTED:
        return _NATIVE_RVL
    _NATIVE_RVL_ATTEMPTED = True
    if os.environ.get("EGOLOONG_DISABLE_NATIVE_RVL", "0") == "1":
        return None
    source = Path(__file__).resolve().parent / "native" / "rvl_decode.cpp"
    if not source.is_file():
        return None
    try:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        build_dir = Path(tempfile.gettempdir()) / "ego_loong_native"
        build_dir.mkdir(parents=True, exist_ok=True)
        library_path = build_dir / f"librvl_decode_{digest}.so"
        if not library_path.is_file():
            temporary_path = build_dir / f".{library_path.name}.{os.getpid()}.tmp"
            subprocess.run(
                ["g++", "-O3", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(temporary_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            os.replace(temporary_path, library_path)
        library = ctypes.CDLL(str(library_path))
        function = library.ego_loong_decode_rvl_u16
        function.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_size_t,
        ]
        function.restype = ctypes.c_int
        _NATIVE_RVL = (library, function)
    except Exception as exc:
        warnings.warn(f"Native RVL decoder unavailable; using slow Python fallback: {exc}")
        _NATIVE_RVL = None
    return _NATIVE_RVL


def decode_rvl_depth(data: bytes) -> np.ndarray:
    """Decode compressed_depth_image_transport RVL into a uint16 image."""
    rvl_header_size = 20
    if len(data) < rvl_header_size:
        raise ValueError(
            f"Compressed RVL depth payload too small: got {len(data)} bytes, "
            f"expected at least {rvl_header_size}"
        )
    width, height = struct.unpack_from("<II", data, 12)
    if width == 0 or height == 0 or width > 16384 or height > 16384:
        raise ValueError(f"Invalid RVL depth dimensions: {width}x{height}")
    payload = data[rvl_header_size:]
    native = _load_native_rvl()
    if native is None:
        return _decode_rvl_python(payload, width, height)
    payload_array = np.frombuffer(payload, dtype=np.uint8)
    depth = np.empty(int(width) * int(height), dtype=np.uint16)
    result = native[1](
        payload_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        payload_array.size,
        depth.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        depth.size,
    )
    errors = {
        -1: "invalid native RVL arguments",
        -2: "truncated or invalid RVL variable-length integer",
        -3: "RVL zero run exceeds depth image size",
        -4: "RVL nonzero run exceeds depth image size",
        -5: "RVL decoded value outside uint16 range",
    }
    if result != 0:
        raise ValueError(errors.get(result, f"native RVL decoder failed with code {result}"))
    return depth.reshape(int(height), int(width))


def depth_image_to_array(msg: Any) -> np.ndarray:
    if hasattr(msg, "encoding"):
        encoding = str(msg.encoding).lower()
        if encoding in ("16uc1", "mono16"):
            dtype = np.uint16
        elif encoding in ("32fc1",):
            dtype = np.float32
        else:
            raise ValueError(f"Unsupported depth encoding: {msg.encoding}")
        arr = np.frombuffer(msg.data, dtype=dtype)
        expected = int(msg.height) * int(msg.width)
        if arr.size < expected:
            raise ValueError(f"Depth payload too small: got {arr.size}, expected {expected}")
        return arr[:expected].reshape(int(msg.height), int(msg.width)).copy()

    # sensor_msgs/CompressedImage with compressedDepth transport may use PNG or
    # RVL. OAK registered 16UC1 depth is expressed in millimeters.
    data = bytes(msg.data)
    compressed_format = str(getattr(msg, "format", "")).lower()
    if "compresseddepth rvl" in compressed_format:
        if not compressed_format.startswith(("16uc1", "mono16")):
            raise ValueError(f"Unsupported RVL depth format: {msg.format}")
        return decode_rvl_depth(data)

    png_sig = b"\x89PNG\r\n\x1a\n"
    start = data.find(png_sig)
    if start < 0:
        raise ValueError(f"Unsupported compressed depth format: {getattr(msg, 'format', '')!r}")
    payload = data[start:]
    arr = np.frombuffer(payload, dtype=np.uint8)
    depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError("Failed to decode compressed depth image")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth.copy()


def depth_units_per_meter(depth: np.ndarray) -> float:
    if depth.dtype == np.uint16:
        return 1000.0
    return 1.0


def align_depth_to_rgb(
    depth: np.ndarray,
    depth_k: np.ndarray,
    rgb_k: np.ndarray,
    rgb_hw: Tuple[int, int],
    t_rgb_depth: np.ndarray,
) -> np.ndarray:
    """Project depth pixels into RGB optical frame using T_rgb_depth.

    The aligned image stores Z in the RGB optical frame, in the same depth unit
    as the input image. For the sampler bags this is uint16 millimeters.
    """
    rgb_h, rgb_w = rgb_hw
    units = depth_units_per_meter(depth)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((rgb_h, rgb_w), dtype=depth.dtype)

    vd, ud = np.nonzero(valid)
    z_d = depth[vd, ud].astype(np.float64) / units
    x_d = (ud.astype(np.float64) - depth_k[0, 2]) * z_d / depth_k[0, 0]
    y_d = (vd.astype(np.float64) - depth_k[1, 2]) * z_d / depth_k[1, 1]
    pts_d = np.stack([x_d, y_d, z_d], axis=0)

    pts_rgb = t_rgb_depth[:3, :3] @ pts_d + t_rgb_depth[:3, 3:4]
    z_rgb = pts_rgb[2]
    in_front = z_rgb > 1e-6
    if not np.any(in_front):
        return np.zeros((rgb_h, rgb_w), dtype=depth.dtype)

    x_rgb = pts_rgb[0, in_front]
    y_rgb = pts_rgb[1, in_front]
    z_rgb = z_rgb[in_front]
    u_rgb = np.rint(rgb_k[0, 0] * x_rgb / z_rgb + rgb_k[0, 2]).astype(np.int32)
    v_rgb = np.rint(rgb_k[1, 1] * y_rgb / z_rgb + rgb_k[1, 2]).astype(np.int32)

    inside = (u_rgb >= 0) & (u_rgb < rgb_w) & (v_rgb >= 0) & (v_rgb < rgb_h)
    if not np.any(inside):
        return np.zeros((rgb_h, rgb_w), dtype=depth.dtype)

    u_rgb = u_rgb[inside]
    v_rgb = v_rgb[inside]
    z_units = z_rgb[inside] * units
    if np.issubdtype(depth.dtype, np.integer):
        z_values = np.rint(z_units).clip(1, np.iinfo(depth.dtype).max).astype(depth.dtype)
        fill_value = np.iinfo(depth.dtype).max
    else:
        z_values = z_units.astype(depth.dtype)
        fill_value = np.inf

    flat_idx = v_rgb * rgb_w + u_rgb
    aligned_flat = np.full(rgb_h * rgb_w, fill_value, dtype=depth.dtype)
    np.minimum.at(aligned_flat, flat_idx, z_values)
    aligned = aligned_flat.reshape(rgb_h, rgb_w)
    aligned[aligned == fill_value] = 0
    return aligned


def resize_depth_to_rgb(depth: np.ndarray, rgb_hw: Tuple[int, int]) -> np.ndarray:
    rgb_h, rgb_w = rgb_hw
    if depth.shape[:2] == (rgb_h, rgb_w):
        return depth.copy()
    return cv2.resize(depth, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)


def parse_extrinsic(value: str) -> np.ndarray:
    raw = json.loads(value)
    mat = np.asarray(raw, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"--t_rgb_depth must be a 4x4 JSON matrix, got {mat.shape}")
    return mat


def load_camera_extrinsic(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        raw = payload["depth_to_rgb"]["matrix_4x4_m"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Camera extrinsics must contain depth_to_rgb.matrix_4x4_m: {path}"
        ) from exc
    mat = np.asarray(raw, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(
            f"depth_to_rgb.matrix_4x4_m must be a 4x4 matrix in {path}, got {mat.shape}"
        )
    return mat


def quat_to_matrix_wxyz(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def odom_to_c2w(msg: Any) -> np.ndarray:
    pose = msg.pose.pose
    p = pose.position
    q = pose.orientation
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = quat_to_matrix_wxyz([q.w, q.x, q.y, q.z])
    mat[:3, 3] = [float(p.x), float(p.y), float(p.z)]
    return mat


def transform_dict_to_matrix(record: Dict[str, Any]) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = quat_to_matrix_wxyz(record["rotation_wxyz"])
    mat[:3, 3] = np.asarray(record["translation"], dtype=np.float64)
    return mat


def nearest_tf(records: List[Dict[str, Any]], stamp_ns: int, max_dt_ns: Optional[int], time_key: str = "stamp_ns") -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
    if not records:
        return None, None
    stamps = [int(r[time_key]) for r in records]
    pos = bisect.bisect_left(stamps, stamp_ns)
    candidates = []
    if pos < len(records):
        candidates.append(records[pos])
    if pos > 0:
        candidates.append(records[pos - 1])
    if not candidates:
        return None, None
    best = min(candidates, key=lambda r: abs(int(r[time_key]) - stamp_ns))
    dt = int(best[time_key]) - int(stamp_ns)
    if max_dt_ns is not None and abs(dt) > max_dt_ns:
        return None, dt
    return best, dt


def dynamic_tf_records(tf_messages: List[Dict[str, Any]], parent: str, child: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for msg in tf_messages:
        for tr in msg.get("transforms", []):
            if tr.get("parent") == parent and tr.get("child") == child:
                row = dict(tr)
                row["bag_time_ns"] = int(msg.get("bag_time_ns", 0))
                rows.append(row)
    rows.sort(key=lambda r: int(r.get("bag_time_ns", r["stamp_ns"])))
    return rows


def transform_to_dict(tr: Any) -> Dict[str, Any]:
    t = tr.transform.translation
    q = tr.transform.rotation
    return {
        "stamp_ns": time_to_ns(tr.header.stamp),
        "parent": str(tr.header.frame_id),
        "child": str(tr.child_frame_id),
        "translation": [float(t.x), float(t.y), float(t.z)],
        "rotation_wxyz": [float(q.w), float(q.x), float(q.y), float(q.z)],
    }


def hand_imu_sample_to_dict(sample: Any) -> Dict[str, Any]:
    return {
        "q_wxyz": [float(sample.q_w), float(sample.q_x), float(sample.q_y), float(sample.q_z)],
        "acc": [float(sample.acc_x), float(sample.acc_y), float(sample.acc_z)],
        "gyr": [float(sample.gyr_x), float(sample.gyr_y), float(sample.gyr_z)],
    }


def hand_frame_to_dict(msg: Any, bag_time_ns: int) -> Dict[str, Any]:
    return {
        "bag_time_ns": int(bag_time_ns),
        "imu_stamp_left_ns": time_to_ns(msg.imu_stamp_left),
        "pressure_stamp_left_ns": time_to_ns(msg.pressure_stamp_left),
        "imu_stamp_right_ns": time_to_ns(msg.imu_stamp_right),
        "pressure_stamp_right_ns": time_to_ns(msg.pressure_stamp_right),
        "imu_left": [hand_imu_sample_to_dict(v) for v in msg.imu_left],
        "imu_right": [hand_imu_sample_to_dict(v) for v in msg.imu_right],
        "pressure_left": [float(v) for v in msg.pressure_left],
        "pressure_right": [float(v) for v in msg.pressure_right],
        "solve_state_left": [float(v) for v in msg.solve_state_left],
        "solve_state_right": [float(v) for v in msg.solve_state_right],
    }




def _time_ns_from_sec_nsec(sec: int, nanosec: int) -> int:
    return int(sec) * 1_000_000_000 + int(nanosec)


def parse_glove_packet_cdr(data: bytes, bag_time_ns: int) -> Dict[str, Any]:
    endian = "<" if len(data) > 1 and data[1] == 1 else ">"
    off = 4

    def read_time() -> int:
        nonlocal off
        sec, nanosec = struct.unpack_from(endian + "iI", data, off)
        off += 8
        return _time_ns_from_sec_nsec(sec, nanosec)

    def read_imu_samples() -> List[Dict[str, Any]]:
        nonlocal off
        samples = []
        for _ in range(16):
            vals = struct.unpack_from(endian + "10f", data, off)
            off += 40
            samples.append({
                "q_wxyz": [float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])],
                "acc": [float(vals[4]), float(vals[5]), float(vals[6])],
                "gyr": [float(vals[7]), float(vals[8]), float(vals[9])],
            })
        return samples

    def read_pressure() -> List[float]:
        nonlocal off
        vals = struct.unpack_from(endian + "68f", data, off)
        off += 68 * 4
        return [float(v) for v in vals]

    imu_stamp_left_ns = read_time()
    imu_left = read_imu_samples()
    pressure_stamp_left_ns = read_time()
    pressure_left = read_pressure()
    imu_stamp_right_ns = read_time()
    imu_right = read_imu_samples()
    pressure_stamp_right_ns = read_time()
    pressure_right = read_pressure()
    return {
        "bag_time_ns": int(bag_time_ns),
        "imu_stamp_left_ns": imu_stamp_left_ns,
        "pressure_stamp_left_ns": pressure_stamp_left_ns,
        "imu_stamp_right_ns": imu_stamp_right_ns,
        "pressure_stamp_right_ns": pressure_stamp_right_ns,
        "imu_left": imu_left,
        "imu_right": imu_right,
        "pressure_left": pressure_left,
        "pressure_right": pressure_right,
        "solve_state_left": [],
        "solve_state_right": [],
        "source_topic": GLOVE_TOPIC,
    }


def find_handcal_for_session(session_path: Path) -> Optional[Path]:
    candidates = [
        session_path / "calibrations" / "hand_calibration.txt",
        session_path / "calibration" / "hand_calibration.txt",
        session_path / "calibration" / "handcal.txt",
        session_path / "data" / "calibration" / "handcal.txt",
        session_path.parent / "hand_calibration.txt",
        session_path.parent / "handcal.txt",
        session_path.parent / "calibrations" / "hand_calibration.txt",
        session_path.parent / "calibration" / "handcal.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def parse_handcal(path: Path, hand: str) -> Tuple[List[Tuple[float, float, float, float]], List[Tuple[float, float, float, float]]]:
    base = 0 if hand == "left" else 16
    c_map: Dict[int, Tuple[float, float, float, float]] = {}
    m_map: Dict[int, Tuple[float, float, float, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != 6 or parts[0] not in ("C", "M"):
            continue
        idx = int(parts[1])
        quat = tuple(float(v) for v in parts[2:6])
        if parts[0] == "C":
            c_map[idx] = quat
        else:
            m_map[idx] = quat
    return [c_map[base + i] for i in range(16)], [m_map[base + i] for i in range(16)]


def solve_states_from_imu(
    hand_frames: List[Dict[str, Any]],
    handcal_path: Optional[Path],
    resolve_driver: Path,
    hand: str = "left",
) -> int:
    if not hand_frames:
        return 0
    if handcal_path is None or not handcal_path.is_file():
        raise RuntimeError("Legacy /glove packets require calibration/handcal.txt to solve state27")
    if not resolve_driver.is_file() or not os.access(resolve_driver, os.X_OK):
        raise RuntimeError(f"Legacy /glove packets require an executable resolve_driver: {resolve_driver}")
    imu_key = f"imu_{hand}"
    state_key = f"solve_state_{hand}"
    c_quats, m_quats = parse_handcal(handcal_path, hand)
    frames = []
    valid_indices = []
    for idx, row in enumerate(hand_frames):
        samples = row.get(imu_key) or []
        if len(samples) < 16:
            continue
        quats = [tuple(float(v) for v in sample.get("q_wxyz", [0, 0, 0, 0])) for sample in samples[:16]]
        frames.append(quats)
        valid_indices.append(idx)
    if not frames:
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        inp = tmp_path / "resolve_in.txt"
        outp = tmp_path / "resolve_out.txt"
        with inp.open("w", encoding="utf-8") as f:
            for quat in c_quats:
                f.write("%.9g %.9g %.9g %.9g\n" % quat)
            for quat in m_quats:
                f.write("%.9g %.9g %.9g %.9g\n" % quat)
            f.write(f"{len(frames)}\n")
            for frame in frames:
                for quat in frame:
                    f.write("%.9g %.9g %.9g %.9g\n" % quat)
        subprocess.run([str(resolve_driver), str(inp), str(outp)], check=True)
        states = []
        for line in outp.read_text(encoding="utf-8", errors="ignore").splitlines():
            vals = line.split()
            if len(vals) == 27:
                states.append([float(v) for v in vals])
    count = 0
    for idx, state in zip(valid_indices, states):
        hand_frames[idx][state_key] = state
        count += 1
    return count


def nearest_by_bag_time(items: List[StampedMsg], bag_time_ns: int, max_dt_ns: Optional[int]) -> Optional[StampedMsg]:
    if not items:
        return None
    stamps = [it.bag_time_ns for it in items]
    pos = bisect.bisect_left(stamps, bag_time_ns)
    candidates = []
    if pos < len(items):
        candidates.append(items[pos])
    if pos > 0:
        candidates.append(items[pos - 1])
    if not candidates:
        return None
    best = min(candidates, key=lambda it: abs(it.bag_time_ns - bag_time_ns))
    if max_dt_ns is not None and abs(best.bag_time_ns - bag_time_ns) > max_dt_ns:
        return None
    return best

def nearest_by_stamp(items: List[StampedMsg], stamp_ns: int, max_dt_ns: Optional[int]) -> Optional[StampedMsg]:
    if not items:
        return None
    stamps = [it.msg_time_ns for it in items]
    pos = bisect.bisect_left(stamps, stamp_ns)
    candidates = []
    if pos < len(items):
        candidates.append(items[pos])
    if pos > 0:
        candidates.append(items[pos - 1])
    if not candidates:
        return None
    best = min(candidates, key=lambda it: abs(it.msg_time_ns - stamp_ns))
    if max_dt_ns is not None and abs(best.msg_time_ns - stamp_ns) > max_dt_ns:
        return None
    return best


def read_bag(
    bag_dir: Path,
    max_frames: Optional[int],
    handcal_path: Optional[Path] = None,
    resolve_driver: Path = RESOLVE_DRIVER_DEFAULT,
) -> Dict[str, Any]:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    wanted = {
        RGB_TOPIC,
        *DEPTH_TOPICS,
        RGB_INFO_TOPIC,
        *DEPTH_INFO_TOPICS,
        ODOM_TOPIC,
        TF_TOPIC,
        TF_STATIC_TOPIC,
        HAND_FRAME_TOPIC,
        GLOVE_TOPIC,
    }
    msg_types = {}
    for topic in wanted:
        type_name = topic_types.get(topic)
        if not type_name:
            continue
        try:
            msg_types[topic] = get_message(type_name)
        except ModuleNotFoundError:
            print(f"[ExtractRosbagSampler] skip topic with unavailable message type: {topic} ({type_name})")

    rgb: List[StampedMsg] = []
    depth_streams: Dict[str, List[StampedMsg]] = {topic: [] for topic in DEPTH_TOPICS}
    odom: List[StampedMsg] = []
    hand_frames: List[Dict[str, Any]] = []
    tf_messages: List[Dict[str, Any]] = []
    tf_static: List[Dict[str, Any]] = []
    camera_info: Dict[str, Dict[str, Any]] = {}
    counts: Dict[str, int] = {}

    while reader.has_next():
        topic, data, bag_time_ns = reader.read_next()
        if topic not in wanted:
            continue
        if topic == GLOVE_TOPIC and topic not in msg_types:
            counts[topic] = counts.get(topic, 0) + 1
            hand_frames.append(parse_glove_packet_cdr(data, bag_time_ns))
            continue
        if topic not in msg_types:
            continue
        counts[topic] = counts.get(topic, 0) + 1
        msg = deserialize_message(data, msg_types[topic])
        msg_time_ns = header_stamp_ns(msg, bag_time_ns)

        if topic == RGB_TOPIC:
            if max_frames is None or len(rgb) < max_frames:
                rgb.append(StampedMsg(topic, bag_time_ns, msg_time_ns, msg))
        elif topic in DEPTH_TOPICS:
            depth_streams[topic].append(StampedMsg(topic, bag_time_ns, msg_time_ns, msg))
        elif topic == ODOM_TOPIC:
            odom.append(StampedMsg(topic, bag_time_ns, msg_time_ns, msg))
        elif topic == RGB_INFO_TOPIC:
            camera_info["rgb"] = camera_info_to_dict(msg)
        elif topic == DEPTH_INFO_TOPIC:
            camera_info["depth_raw"] = camera_info_to_dict(msg)
        elif topic == DEPTH_REGISTERED_INFO_TOPIC:
            camera_info["depth_registered"] = camera_info_to_dict(msg)
        elif topic == HAND_FRAME_TOPIC:
            hand_frames.append(hand_frame_to_dict(msg, bag_time_ns))
        elif topic in (TF_TOPIC, TF_STATIC_TOPIC):
            records = [transform_to_dict(tr) for tr in msg.transforms]
            if topic == TF_STATIC_TOPIC:
                tf_static.extend(records)
            else:
                tf_messages.append({"bag_time_ns": int(bag_time_ns), "transforms": records})

    rgb.sort(key=lambda it: it.msg_time_ns)
    registered_declared = DEPTH_REGISTERED_COMPRESSED_TOPIC in topic_types
    if registered_declared:
        selected_depth_topic = DEPTH_REGISTERED_COMPRESSED_TOPIC
        selected_info_topic = DEPTH_REGISTERED_INFO_TOPIC
        selected_info_key = "depth_registered"
        selection_mode = "registered_preferred"
    elif DEPTH_TOPIC in topic_types:
        selected_depth_topic = DEPTH_TOPIC
        selected_info_topic = DEPTH_INFO_TOPIC
        selected_info_key = "depth_raw"
        selection_mode = "raw_fallback_no_registered_topic"
    else:
        raise RuntimeError(f"Missing depth topic; expected {DEPTH_REGISTERED_COMPRESSED_TOPIC} or {DEPTH_TOPIC}")

    depth = depth_streams[selected_depth_topic]
    depth.sort(key=lambda it: it.msg_time_ns)
    camera_info["depth"] = camera_info.get(selected_info_key)
    depth_selection = {
        "policy": "Use registered depth exclusively when its topic is declared; use raw only when registered is absent.",
        "mode": selection_mode,
        "selected_depth_topic": selected_depth_topic,
        "selected_camera_info_topic": selected_info_topic,
        "selected_is_registered": selected_depth_topic == DEPTH_REGISTERED_COMPRESSED_TOPIC,
        "available_depth_topics": [topic for topic in DEPTH_TOPICS if topic in topic_types],
        "selected_messages": len(depth),
        "ignored_depth_messages": sum(len(items) for topic, items in depth_streams.items() if topic != selected_depth_topic),
    }
    print(
        "[ExtractRosbagSampler] depth source: "
        f"{selected_depth_topic} ({selection_mode}), messages={len(depth)}, "
        f"ignored_other_depth_messages={depth_selection['ignored_depth_messages']}"
    )
    odom.sort(key=lambda it: it.msg_time_ns)
    hand_frames.sort(key=lambda it: it["bag_time_ns"])
    if hand_frames and any(row.get("source_topic") == GLOVE_TOPIC for row in hand_frames):
        solved_left = solve_states_from_imu(
            hand_frames,
            handcal_path,
            resolve_driver=resolve_driver,
            hand="left",
        )
        solved_right = solve_states_from_imu(
            hand_frames,
            handcal_path,
            resolve_driver=resolve_driver,
            hand="right",
        )
        print(
            f"[ExtractRosbagSampler] /glove packets: {len(hand_frames)}, "
            f"solved left states: {solved_left}, solved right states: {solved_right}"
        )
    return {
        "topic_types": topic_types,
        "counts": counts,
        "rgb": rgb,
        "depth": depth,
        "depth_selection": depth_selection,
        "odom": odom,
        "camera_info": camera_info,
        "hand_frames": hand_frames,
        "tf": tf_messages,
        "tf_static": tf_static,
    }


def estimate_fps(stamps_ns: List[int]) -> float:
    if len(stamps_ns) < 2:
        return 30.0
    diffs = np.diff(np.asarray(stamps_ns, dtype=np.int64)).astype(np.float64) / 1e9
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 30.0
    return float(1.0 / np.median(diffs))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image or fail the extraction stage immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        written = cv2.imwrite(str(path), image)
    except cv2.error as exc:
        raise RuntimeError(f"Failed to encode/write image: {path}: {exc}") from exc
    if not written:
        raise RuntimeError(f"Failed to encode/write image: {path}")


def write_image_with_hardlink(path: Path, alias_path: Path, image: np.ndarray) -> None:
    """Encode once and expose the same bytes at a second required path."""
    write_image(path, image)
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        alias_path.unlink(missing_ok=True)
        os.link(path, alias_path)
    except OSError:
        write_image(alias_path, image)


class ParallelImageWriter:
    """Bounded image encoder/writer pool; bounds queued arrays to control RAM."""

    def __init__(self, workers: int):
        self.workers = max(1, int(workers))
        self.executor = None if self.workers == 1 else ThreadPoolExecutor(
            max_workers=self.workers,
            thread_name_prefix="extract-image",
        )
        self.pending = set()
        self.max_pending = self.workers * 4

    def submit(self, path: Path, image: np.ndarray) -> None:
        if self.executor is None:
            write_image(path, image)
            return
        self.pending.add(self.executor.submit(write_image, path, image))
        if len(self.pending) >= self.max_pending:
            done, self.pending = wait(self.pending, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()

    def submit_with_hardlink(self, path: Path, alias_path: Path, image: np.ndarray) -> None:
        if self.executor is None:
            write_image_with_hardlink(path, alias_path, image)
            return
        self.pending.add(self.executor.submit(write_image_with_hardlink, path, alias_path, image))
        if len(self.pending) >= self.max_pending:
            done, self.pending = wait(self.pending, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()

    def close(self) -> None:
        if self.executor is None:
            return
        done, self.pending = wait(self.pending)
        try:
            for future in done:
                future.result()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)


def write_outputs(
    session_path: Path,
    output_dir: Path,
    extracted: Dict[str, Any],
    overwrite: bool,
    max_sync_dt_ns: Optional[int],
    t_rgb_depth: Optional[np.ndarray],
    image_write_workers: int,
) -> Dict[str, Any]:
    preprocess_dir = output_dir
    all_data_dir = preprocess_dir / "all_data"
    ensure_clean_dir(all_data_dir, overwrite=overwrite)
    (preprocess_dir / "vis").mkdir(parents=True, exist_ok=True)

    rgb_info = extracted["camera_info"].get("rgb")
    depth_info = extracted["camera_info"].get("depth")
    if rgb_info is None:
        raise RuntimeError(f"Missing {RGB_INFO_TOPIC}")
    if depth_info is None:
        selected_info_topic = extracted["depth_selection"]["selected_camera_info_topic"]
        raise RuntimeError(f"Missing CameraInfo for selected depth source: {selected_info_topic}")

    rgb_k = np.asarray(rgb_info["k"], dtype=np.float64)
    depth_k = np.asarray(depth_info["k"], dtype=np.float64)
    fps = estimate_fps([it.msg_time_ns for it in extracted["rgb"]])
    map_odom_tf = dynamic_tf_records(extracted["tf"], "map", "odom")

    frame_rows: List[Dict[str, Any]] = []
    odom_rows: List[Dict[str, Any]] = []
    depth_sync_fail = 0
    odom_sync_fail = 0
    deduplicated_registered_depth = 0
    image_writer = ParallelImageWriter(image_write_workers)

    for idx, rgb_item in enumerate(extracted["rgb"]):
        frame_key = f"{idx:05d}"
        frame_dir = all_data_dir / frame_key
        frame_dir.mkdir(parents=True, exist_ok=True)

        rgb_img = compressed_to_bgr(rgb_item.msg)
        rgb_path = frame_dir / "rgb.png"
        image_writer.submit(rgb_path, rgb_img)

        depth_item = nearest_by_stamp(extracted["depth"], rgb_item.msg_time_ns, max_sync_dt_ns)
        depth_path = frame_dir / "depth.png"
        depth_aligned_path = frame_dir / "depth_aligned.png"
        depth_dt_ns = None
        depth_alignment = {
            "method": "none",
            "note": "no depth frame matched",
            "has_depth_to_rgb_extrinsic": False,
        }
        if depth_item is None:
            depth_sync_fail += 1
            depth_arr = np.zeros((int(depth_info["height"]), int(depth_info["width"])), dtype=np.uint16)
            image_writer.submit(depth_path, depth_arr)
            image_writer.submit(
                depth_aligned_path,
                cv2.resize(depth_arr, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_NEAREST),
            )
        else:
            depth_dt_ns = int(depth_item.msg_time_ns - rgb_item.msg_time_ns)
            depth_arr = depth_image_to_array(depth_item.msg)
            depth_is_registered = depth_item.topic == DEPTH_REGISTERED_COMPRESSED_TOPIC
            if depth_is_registered:
                depth_aligned = resize_depth_to_rgb(depth_arr, (rgb_img.shape[0], rgb_img.shape[1]))
                depth_alignment = {
                    "method": "registered_depth_rgb_frame",
                    "note": "depth is already registered to the RGB optical frame; copied/resized without depth-to-RGB extrinsic",
                    "has_depth_to_rgb_extrinsic": False,
                    "source_frame": depth_info["frame_id"],
                    "target_frame": rgb_info["frame_id"],
                    "source_topic": depth_item.topic,
                }
            elif t_rgb_depth is None:
                depth_aligned = resize_depth_to_rgb(depth_arr, (rgb_img.shape[0], rgb_img.shape[1]))
                depth_alignment = {
                    "method": "resize_nearest_no_extrinsic",
                    "note": "depth resized to RGB resolution; depth optical -> rgb optical extrinsic disabled",
                    "has_depth_to_rgb_extrinsic": False,
                }
            else:
                depth_aligned = align_depth_to_rgb(
                    depth=depth_arr,
                    depth_k=depth_k,
                    rgb_k=rgb_k,
                    rgb_hw=(rgb_img.shape[0], rgb_img.shape[1]),
                    t_rgb_depth=t_rgb_depth,
                )
                depth_alignment = {
                    "method": "project_depth_to_rgb",
                    "note": "depth pixels projected from depth optical frame to RGB optical frame using T_rgb_depth",
                    "has_depth_to_rgb_extrinsic": True,
                    "source_frame": depth_info["frame_id"],
                    "target_frame": rgb_info["frame_id"],
                    "t_rgb_depth": t_rgb_depth.tolist(),
                }
            if depth_is_registered and depth_arr.shape == depth_aligned.shape:
                image_writer.submit_with_hardlink(depth_path, depth_aligned_path, depth_arr)
                deduplicated_registered_depth += 1
            else:
                image_writer.submit(depth_path, depth_arr)
                image_writer.submit(depth_aligned_path, depth_aligned)

        odom_item = nearest_by_bag_time(extracted["odom"], rgb_item.bag_time_ns, max_sync_dt_ns)
        pose_source = {
            "method": "identity_missing_odom",
            "world_frame": "unknown",
            "base_frame": "base_link",
            "uses_map_to_odom_tf": False,
            "map_odom_stamp_ns": None,
            "map_odom_dt_ns": None,
        }
        if odom_item is None:
            odom_sync_fail += 1
            c2w = np.eye(4, dtype=np.float64)
            odom_dt_ns = None
            map_odom_record = None
            map_odom_dt_ns = None
        else:
            odom_base = odom_to_c2w(odom_item.msg)
            odom_dt_ns = int(odom_item.bag_time_ns - rgb_item.bag_time_ns)
            map_odom_record, map_odom_dt_ns = nearest_tf(map_odom_tf, rgb_item.bag_time_ns, max_sync_dt_ns, time_key="bag_time_ns")
            if map_odom_record is not None:
                map_odom = transform_dict_to_matrix(map_odom_record)
                c2w = map_odom @ odom_base
                pose_source = {
                    "method": "tf_chain_map_odom_base_link",
                    "world_frame": "map",
                    "base_frame": str(odom_item.msg.child_frame_id),
                    "uses_map_to_odom_tf": True,
                    "map_odom_stamp_ns": int(map_odom_record["stamp_ns"]),
                    "map_odom_dt_ns": int(map_odom_dt_ns),
                    "odom_stamp_ns": int(odom_item.msg_time_ns),
                    "odom_bag_time_ns": int(odom_item.bag_time_ns),
                    "odom_dt_ns": int(odom_dt_ns),
                    "odom_header_dt_ns": int(odom_item.msg_time_ns - rgb_item.msg_time_ns),
                }
            else:
                c2w = odom_base
                pose_source = {
                    "method": "odom_base_link_fallback_no_map_odom",
                    "world_frame": str(odom_item.msg.header.frame_id),
                    "base_frame": str(odom_item.msg.child_frame_id),
                    "uses_map_to_odom_tf": False,
                    "map_odom_stamp_ns": None,
                    "map_odom_dt_ns": None if map_odom_dt_ns is None else int(map_odom_dt_ns),
                    "odom_stamp_ns": int(odom_item.msg_time_ns),
                    "odom_bag_time_ns": int(odom_item.bag_time_ns),
                    "odom_dt_ns": int(odom_dt_ns),
                    "odom_header_dt_ns": int(odom_item.msg_time_ns - rgb_item.msg_time_ns),
                }

            p = odom_item.msg.pose.pose.position
            q = odom_item.msg.pose.pose.orientation
            odom_rows.append(
                {
                    "frame": frame_key,
                    "rgb_stamp_ns": int(rgb_item.msg_time_ns),
                    "odom_stamp_ns": int(odom_item.msg_time_ns),
                    "odom_bag_time_ns": int(odom_item.bag_time_ns),
                    "dt_ns": odom_dt_ns,
                    "header_dt_ns": int(odom_item.msg_time_ns - rgb_item.msg_time_ns),
                    "frame_id": str(odom_item.msg.header.frame_id),
                    "child_frame_id": str(odom_item.msg.child_frame_id),
                    "position": [float(p.x), float(p.y), float(p.z)],
                    "orientation_wxyz": [float(q.w), float(q.x), float(q.y), float(q.z)],
                    "odom_base_c2w": odom_base.tolist(),
                    "map_odom": None if map_odom_record is None else map_odom_record,
                    "pose_source": pose_source,
                    "c2w": c2w.tolist(),
                }
            )

        cam_json = {
            "idx": idx,
            "ts": int(rgb_item.msg_time_ns),
            "h": int(rgb_img.shape[0]),
            "w": int(rgb_img.shape[1]),
            "k": rgb_k.tolist(),
            "d": [float(x) for x in rgb_info["d"]],
            "c2w": c2w.tolist(),
            "pose_source": pose_source,
            "c2d": np.eye(4, dtype=np.float64).tolist(),
            "d2w": np.eye(4, dtype=np.float64).tolist(),
            "fps": fps,
            "rgb_frame_id": rgb_info["frame_id"],
            "depth_frame_id": depth_info["frame_id"],
            "source_rgb_path": str(rgb_path),
            "source_depth_path": str(depth_path),
            "source_depth_aligned_path": str(depth_aligned_path),
            "source": "rosbag2_sampler",
            "sync": {
                "rgb_bag_time_ns": int(rgb_item.bag_time_ns),
                "depth_stamp_ns": None if depth_item is None else int(depth_item.msg_time_ns),
                "depth_dt_ns": depth_dt_ns,
                "odom_stamp_ns": None if odom_item is None else int(odom_item.msg_time_ns),
                "odom_bag_time_ns": None if odom_item is None else int(odom_item.bag_time_ns),
                "odom_dt_ns": odom_dt_ns,
                "odom_header_dt_ns": None if odom_item is None else int(odom_item.msg_time_ns - rgb_item.msg_time_ns),
                "map_odom_stamp_ns": pose_source.get("map_odom_stamp_ns"),
                "map_odom_dt_ns": pose_source.get("map_odom_dt_ns"),
            },
            "depth_alignment": depth_alignment,
        }
        with (frame_dir / "aria_cam_rgb.json").open("w", encoding="utf-8") as f:
            json.dump(cam_json, f, indent=4)
            f.write("\n")

        frame_rows.append(
            {
                "frame": frame_key,
                "idx": idx,
                "rgb_stamp_ns": int(rgb_item.msg_time_ns),
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path),
                "depth_aligned_path": str(depth_aligned_path),
                "depth_dt_ns": depth_dt_ns,
                "odom_dt_ns": odom_dt_ns,
                "odom_header_dt_ns": None if odom_item is None else int(odom_item.msg_time_ns - rgb_item.msg_time_ns),
                "pose_source": pose_source,
            }
        )

    image_writer.close()

    camera_summary = {
        "session_path": str(session_path),
        "rgb": rgb_info,
        "depth": depth_info,
        "depth_selection": extracted["depth_selection"],
        "rgb_k": rgb_k.tolist(),
        "depth_k": depth_k.tolist(),
        "fps": fps,
        "frames": len(frame_rows),
        "depth_alignment_default": (
            "registered_depth_rgb_frame"
            if extracted["depth_selection"]["selected_is_registered"]
            else ("project_depth_to_rgb" if t_rgb_depth is not None else "resize_nearest_no_extrinsic")
        ),
        "depth_to_rgb_extrinsic_used": bool(
            not extracted["depth_selection"]["selected_is_registered"] and t_rgb_depth is not None
        ),
        "t_rgb_depth": None if t_rgb_depth is None else t_rgb_depth.tolist(),
    }
    with (preprocess_dir / "camera_info.json").open("w", encoding="utf-8") as f:
        json.dump(camera_summary, f, indent=4)
        f.write("\n")

    with (preprocess_dir / "extract_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "session_path": str(session_path),
                "topic_counts": extracted["counts"],
                "topic_types": extracted["topic_types"],
                "depth_selection": extracted["depth_selection"],
                "frames": len(frame_rows),
                "fps": fps,
                "depth_sync_fail": depth_sync_fail,
                "odom_sync_fail": odom_sync_fail,
                "map_odom_tf_count": len(map_odom_tf),
                "pose_method": "tf_chain_map_odom_base_link_when_available",
                "max_sync_dt_ns": max_sync_dt_ns,
                "image_write_workers": int(image_write_workers),
                "deduplicated_registered_depth": deduplicated_registered_depth,
            },
            f,
            indent=4,
        )
        f.write("\n")

    write_jsonl(preprocess_dir / "timestamps.jsonl", frame_rows)
    write_jsonl(preprocess_dir / "odom.jsonl", odom_rows)
    write_jsonl(preprocess_dir / "hand_frame.jsonl", extracted["hand_frames"])
    write_jsonl(preprocess_dir / "tf.jsonl", extracted["tf"])
    write_jsonl(preprocess_dir / "tf_static.jsonl", extracted["tf_static"])

    return {
        "frames": len(frame_rows),
        "fps": fps,
        "depth_sync_fail": depth_sync_fail,
        "odom_sync_fail": odom_sync_fail,
        "hand_frames": len(extracted["hand_frames"]),
        "out_dir": str(preprocess_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract sampler ROS2 bag RGBD/pose/hand_frame into preprocess layout.")
    parser.add_argument("--session_path", required=True, help="Session directory containing bag/metadata.yaml and bag/*.db3.")
    parser.add_argument("--bag_dir", default=None, help="Override bag directory. Defaults to <session_path>/bag.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to <session_path>/preprocess for backward compatibility.",
    )
    parser.add_argument("--max_frames", type=int, default=None, help="Limit RGB frames for quick tests.")
    parser.add_argument("--max_sync_dt_ms", type=float, default=80.0, help="Max nearest-neighbor sync delta in ms; <0 disables.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing preprocess/all_data before writing.")
    parser.add_argument("--image_write_workers", type=int, default=8, help="Parallel PNG encode/write workers; use 1 for serial writes.")
    parser.add_argument(
        "--resolve_driver",
        default=str(RESOLVE_DRIVER_DEFAULT),
        help="Legacy /glove IMU-to-state executable. Not used by bags that already contain /hand_frame solve states.",
    )
    parser.add_argument(
        "--t_rgb_depth",
        default=None,
        help="Optional 4x4 JSON matrix mapping depth/LEFT optical points into RGB optical frame. Defaults to the calibrated sampler OAK matrix.",
    )
    parser.add_argument(
        "--camera_extrinsics",
        default=None,
        help="Optional camera_extrinsics.json containing depth_to_rgb.matrix_4x4_m.",
    )
    parser.add_argument(
        "--handcal_path",
        default=None,
        help="Optional explicit hand calibration file for legacy /glove packets.",
    )
    parser.add_argument(
        "--disable_depth_to_rgb_extrinsic",
        action="store_true",
        help="Fall back to nearest-neighbor resizing instead of calibrated depth-to-RGB projection.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    session_path = Path(args.session_path).expanduser().resolve()
    bag_dir = Path(args.bag_dir).expanduser().resolve() if args.bag_dir else session_path / "bag"
    if not bag_dir.is_dir():
        raise FileNotFoundError(f"Missing bag directory: {bag_dir}")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else session_path / "preprocess"
    max_sync_dt_ns = None if args.max_sync_dt_ms < 0 else int(args.max_sync_dt_ms * 1e6)
    if args.disable_depth_to_rgb_extrinsic:
        t_rgb_depth = None
    elif args.t_rgb_depth:
        t_rgb_depth = parse_extrinsic(args.t_rgb_depth)
    elif args.camera_extrinsics:
        t_rgb_depth = load_camera_extrinsic(Path(args.camera_extrinsics).expanduser().resolve())
    else:
        t_rgb_depth = DEFAULT_T_RGB_DEPTH

    handcal_path = (
        Path(args.handcal_path).expanduser().resolve()
        if args.handcal_path
        else find_handcal_for_session(session_path)
    )
    resolve_driver = Path(args.resolve_driver).expanduser().resolve()
    extracted = read_bag(
        bag_dir,
        max_frames=args.max_frames,
        handcal_path=handcal_path,
        resolve_driver=resolve_driver,
    )
    stats = write_outputs(
        session_path=session_path,
        output_dir=output_dir,
        extracted=extracted,
        overwrite=args.overwrite,
        max_sync_dt_ns=max_sync_dt_ns,
        t_rgb_depth=t_rgb_depth,
        image_write_workers=args.image_write_workers,
    )
    print(f"[ExtractRosbagSampler] stats: {json.dumps(stats, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
