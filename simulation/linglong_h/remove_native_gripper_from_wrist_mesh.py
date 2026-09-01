#!/usr/bin/env python3
"""Remove the LingLong native gripper components from terminal wrist STL meshes."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


TRIANGLE_DTYPE = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ]
)


def connected_components(vertices: np.ndarray) -> list[np.ndarray]:
    parent = np.arange(len(vertices), dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owner: dict[tuple[int, int, int], int] = {}
    quantized = np.rint(vertices.astype(np.float64) / 1e-6).astype(np.int64)
    for face_index, face in enumerate(quantized):
        for vertex in face:
            key = tuple(int(value) for value in vertex)
            union(face_index, owner.setdefault(key, face_index))

    groups: dict[int, list[int]] = {}
    for face_index in range(len(vertices)):
        groups.setdefault(find(face_index), []).append(face_index)
    return [np.asarray(indices, dtype=np.int64) for indices in groups.values()]


def strip_native_gripper(source: Path, destination: Path) -> None:
    raw = source.read_bytes()
    triangle_count = struct.unpack_from("<I", raw, 80)[0]
    records = np.frombuffer(
        raw, dtype=TRIANGLE_DTYPE, offset=84, count=triangle_count
    ).copy()
    remove = np.zeros(triangle_count, dtype=bool)
    removed_components: list[tuple[int, list[float], list[float]]] = []
    for indices in connected_components(records["vertices"]):
        points = records["vertices"][indices].reshape(-1, 3).astype(np.float64)
        lower, upper = points.min(axis=0), points.max(axis=0)
        is_native_gripper = (
            (lower[0] >= 0.0625 and upper[0] >= 0.1715 and len(indices) >= 1000)
            or lower[0] >= 0.1715
        )
        if is_native_gripper:
            remove[indices] = True
            removed_components.append(
                (len(indices), lower.round(6).tolist(), upper.round(6).tolist())
            )
    if len(removed_components) != 3:
        raise RuntimeError(
            f"{source}: expected 3 native-gripper components, found "
            f"{len(removed_components)}: {removed_components}"
        )
    kept = records[~remove]
    header = (
        f"{source.name} without native gripper".encode("ascii")[:80].ljust(80, b"\0")
    )
    destination.write_bytes(header + struct.pack("<I", len(kept)) + kept.tobytes())
    print(
        f"{source.name}: kept {len(kept)}/{triangle_count} triangles; "
        f"removed {removed_components}; wrote {destination}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh_dir", type=Path, default=Path(__file__).parent / "assets/meshes")
    args = parser.parse_args()
    for side in ("left", "right"):
        strip_native_gripper(
            args.mesh_dir / f"{side}_wrist_yaw_link.STL",
            args.mesh_dir / f"{side}_wrist_yaw_link_without_gripper.STL",
        )


if __name__ == "__main__":
    main()
