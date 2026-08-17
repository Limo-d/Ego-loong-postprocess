#!/usr/bin/env python3
"""从 MANO pkl 导出左右手网格缓存，并标定 68 路传感点 3D 锚点。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mano_tactile_viz import (  # noqa: E402
    calibrate_sensor_anchors,
    export_mano_meshes,
    load_mano_mesh,
    _ANCHORS_LEFT_NPY,
    _ANCHORS_RIGHT_NPY,
)
from tactile_serial_reader import _SENSOR_VIZ_XY  # noqa: E402


def main() -> None:
    left_path, right_path = export_mano_meshes()
    print(f"exported {left_path}")
    print(f"exported {right_path}")

    v_left, _ = load_mano_mesh(is_right=False)
    v_right, _ = load_mano_mesh(is_right=True)
    anchors_l = calibrate_sensor_anchors(v_left, _SENSOR_VIZ_XY, is_right=False)
    anchors_r = calibrate_sensor_anchors(v_right, _SENSOR_VIZ_XY, is_right=True)
    _ANCHORS_LEFT_NPY.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.save(_ANCHORS_LEFT_NPY, anchors_l)
    np.save(_ANCHORS_RIGHT_NPY, anchors_r)
    print(f"anchors {_ANCHORS_LEFT_NPY}")
    print(f"anchors {_ANCHORS_RIGHT_NPY}")


if __name__ == "__main__":
    main()
