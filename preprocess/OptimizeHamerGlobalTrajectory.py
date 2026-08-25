#!/usr/bin/env python3
"""Smooth FK wrist translation and palm orientation in a shared world frame."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation, Slerp

try:
    from preprocess.Timebase import row_stamp_ns
except ModuleNotFoundError:
    from Timebase import row_stamp_ns

MCP = np.asarray([5, 9, 13, 17], dtype=np.int64)
PALM = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def project_so3(matrix: np.ndarray) -> np.ndarray:
    u, _s, vt = np.linalg.svd(matrix)
    out = u @ vt
    if np.linalg.det(out) < 0:
        u[:, -1] *= -1
        out = u @ vt
    return out


def palm_basis(points: np.ndarray, previous: Optional[np.ndarray]) -> Optional[np.ndarray]:
    x = np.mean(points[MCP], axis=0) - points[0]
    lateral = points[5] - points[17]
    if np.linalg.norm(x) < 1e-8 or np.linalg.norm(lateral) < 1e-8:
        return None
    x /= np.linalg.norm(x)
    lateral /= np.linalg.norm(lateral)
    z = np.cross(x, lateral)
    if np.linalg.norm(z) < 1e-8:
        return None
    z /= np.linalg.norm(z)
    if previous is None:
        if z[2] > 0:  # Dorsal side is visible: point toward the head camera.
            z = -z
    elif z @ previous[:, 2] < 0:
        z = -z
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return project_so3(np.column_stack([x, y, z]))


def rotation_6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    a1, a2 = value[..., :3], value[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack((b1, b2, torch.cross(b1, b2, dim=-1)), dim=-1)


def matrix_to_6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[:, 0], matrix[:, 1]])


def interpolate_series(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = values.copy()
    indices = np.arange(len(values))
    good = indices[valid]
    if good.size == 0:
        return result
    for dim in range(values.shape[1]):
        result[:, dim] = np.interp(indices, good, values[good, dim])
    return result


def interpolate_rotations(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    indices = np.arange(len(values))
    good = indices[valid]
    if good.size == 0:
        return values
    if good.size == 1:
        return np.repeat(values[good[0]][None], len(values), axis=0)
    slerp = Slerp(good.astype(float), Rotation.from_matrix(values[good]))
    query = np.clip(indices, good[0], good[-1]).astype(float)
    return slerp(query).as_matrix()


def angular_steps(rotations: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = []
    previous = None
    for rotation, ok in zip(rotations, valid):
        if not ok:
            continue
        if previous is not None:
            relative = previous.T @ rotation
            out.append(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) * .5, -1, 1))))
        previous = rotation
    return np.asarray(out, dtype=np.float64)


def local_pose_steps(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = []
    previous = None
    for pose, ok in zip(points, valid):
        if not ok:
            continue
        if previous is not None:
            displacement = np.linalg.norm(pose - previous, axis=1)
            out.append(float(np.sqrt(np.mean(displacement ** 2))))
        previous = pose
    return np.asarray(out, dtype=np.float64)


def translation_steps(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = []
    for i in range(1, len(points)):
        if valid[i - 1] and valid[i]:
            out.append(float(np.linalg.norm(points[i] - points[i - 1])))
    return np.asarray(out, dtype=np.float64)


def translation_second_differences(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = []
    for i in range(1, len(points) - 1):
        if valid[i - 1] and valid[i] and valid[i + 1]:
            out.append(float(np.linalg.norm(points[i + 1] - 2.0 * points[i] + points[i - 1])))
    return np.asarray(out, dtype=np.float64)


def branch_rejections(rotations: np.ndarray, valid: np.ndarray, threshold_deg: float, max_length: int) -> np.ndarray:
    """Mark short orientation branches that jump away and later return."""
    parity = np.zeros(len(valid), dtype=np.uint8)
    previous = None
    for i in range(len(valid)):
        parity[i] = parity[i - 1] if i else 0
        if not valid[i]:
            continue
        if previous is not None:
            relative = rotations[previous].T @ rotations[i]
            angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1) * .5, -1, 1)))
            if angle > threshold_deg:
                parity[i] = 1 - parity[i]
        previous = i
    rejected = np.zeros(len(valid), dtype=bool)
    start = 0
    for i in range(1, len(parity) + 1):
        if i < len(parity) and parity[i] == parity[start]:
            continue
        end = i - 1
        if parity[start] == 1 and start > 0 and end < len(parity) - 1 and end - start + 1 <= max_length:
            rejected[start:end + 1] = True
        start = i
    return rejected


def stats(values: np.ndarray) -> Dict[str, Optional[float]]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "median": None, "p95": None, "p99": None, "max": None}
    return {"count": int(values.size), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values))}


def smooth_map(path: Optional[str]) -> Dict[tuple, np.ndarray]:
    if not path:
        return {}
    result = {}
    for row in read_jsonl(Path(path).expanduser().resolve()):
        payload = row.get("visual_2d_smooth") or {}
        points = np.asarray(payload.get("kpts_2d"), dtype=np.float64)
        if payload.get("valid") and points.shape == (21, 2) and np.isfinite(points).all():
            stamp = row_stamp_ns(row)
            result[("stamp", stamp) if stamp else ("frame", str(row.get("frame")))] = points
    return result


def optimize_side(rows: List[Dict[str, Any]], side: str, targets: Dict[tuple, np.ndarray], args: argparse.Namespace) -> Dict[str, Any]:
    count = len(rows)
    c2w = np.zeros((count, 4, 4), np.float64)
    intrinsics = np.zeros((count, 3, 3), np.float64)
    local = np.zeros((count, 21, 3), np.float64)
    root_world = np.zeros((count, 3), np.float64)
    rotation_world = np.repeat(np.eye(3)[None], count, axis=0)
    target_px = np.zeros((count, 21, 2), np.float64)
    fk_valid = np.zeros(count, bool)
    valid = np.zeros(count, bool)
    previous_camera_rotation = None
    for i, row in enumerate(rows):
        hand = ((row.get("hands") or {}).get(side) or {})
        glove = hand.get("glove") or {}
        points = np.asarray(
            glove.get("kpts_3d_camera_m_before_palm_level") or glove.get("kpts_3d_camera_m"),
            dtype=np.float64,
        )
        camera = row.get("camera") or {}
        C = np.asarray(camera.get("c2w"), dtype=np.float64)
        K = np.asarray(camera.get("k"), dtype=np.float64)
        stamp = row_stamp_ns(row)
        target = targets.get(("stamp", stamp) if stamp else ("frame", str(row.get("frame"))))
        if points.shape != (21, 3) or C.shape != (4, 4) or K.shape != (3, 3):
            continue
        if not np.isfinite(points).all() or not np.isfinite(C).all() or np.min(points[:, 2]) <= 0:
            continue
        R_cam = palm_basis(points, previous_camera_rotation)
        if R_cam is None:
            continue
        previous_camera_rotation = R_cam
        R_wc = project_so3(C[:3, :3])
        c2w[i] = C
        c2w[i, :3, :3] = R_wc
        intrinsics[i] = K
        local[i] = (R_cam.T @ (points - points[0]).T).T
        root_world[i] = R_wc @ points[0] + C[:3, 3]
        rotation_world[i] = project_so3(R_wc @ R_cam)
        fk_valid[i] = True
        if target is not None:
            target_px[i] = target
            valid[i] = True
    if not fk_valid.any():
        return {"side": side, "valid": 0, "status": "no_valid_frames"}

    rejected = np.zeros(count, dtype=bool)
    root_init = interpolate_series(root_world, fk_valid)
    rotation_init = interpolate_rotations(rotation_world, fk_valid)
    local_filled = interpolate_series(local.reshape(count, -1), fk_valid).reshape(count, 21, 3)
    # A depth/root jump is characterized by disagreement with the local
    # constant-velocity prediction. Downweight it instead of deleting the
    # observation so sustained real motion remains represented.
    root_prediction_residual = np.zeros(count, dtype=np.float64)
    root_observation_weight = np.ones(count, dtype=np.float64)
    for i in range(1, count - 1):
        if fk_valid[i - 1] and fk_valid[i] and fk_valid[i + 1]:
            prediction = 0.5 * (root_world[i - 1] + root_world[i + 1])
            residual = float(np.linalg.norm(root_world[i] - prediction))
            root_prediction_residual[i] = residual
            ratio = residual / max(float(args.translation_outlier_threshold_m), 1e-9)
            root_observation_weight[i] = max(
                float(args.min_root_observation_weight), 1.0 / (1.0 + ratio * ratio)
            )
    root_observation_weight[~fk_valid] = 0.0
    translation_outliers = (
        fk_valid & (root_prediction_residual > float(args.translation_outlier_threshold_m))
    )

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64
    t = torch.nn.Parameter(torch.as_tensor(root_init, dtype=dtype, device=device))
    r6 = torch.nn.Parameter(torch.as_tensor(np.stack([matrix_to_6d(r) for r in rotation_init]), dtype=dtype, device=device))
    log_scale = torch.nn.Parameter(torch.zeros((), dtype=dtype, device=device))
    local_t = torch.as_tensor(local_filled, dtype=dtype, device=device)
    root_obs = torch.as_tensor(root_init, dtype=dtype, device=device)
    root_weight = torch.as_tensor(root_observation_weight, dtype=dtype, device=device)
    rot_obs = torch.as_tensor(rotation_init, dtype=dtype, device=device)
    valid_t = torch.as_tensor(valid, dtype=torch.bool, device=device)
    fk_valid_t = torch.as_tensor(fk_valid, dtype=torch.bool, device=device)
    palm_t = torch.as_tensor(PALM, dtype=torch.long, device=device)
    K = torch.as_tensor(intrinsics, dtype=dtype, device=device)
    C = torch.as_tensor(c2w, dtype=dtype, device=device)
    target = torch.as_tensor(target_px, dtype=dtype, device=device)

    def losses(temporal: bool) -> Dict[str, torch.Tensor]:
        R = rotation_6d_to_matrix(r6)
        scale = torch.ones((), dtype=dtype, device=device)
        world = torch.einsum("nij,nkj->nki", R, local_t * scale) + t[:, None, :]
        Rc = C[:, :3, :3].transpose(1, 2)
        camera = torch.einsum("nij,nkj->nki", Rc, world - C[:, None, :3, 3])
        uvw = torch.einsum("nij,nkj->nki", K, camera)
        projected = uvw[..., :2] / uvw[..., 2:].clamp_min(1e-4)
        reproj = torch.zeros((), dtype=dtype, device=device)
        root_element = F.smooth_l1_loss(t, root_obs, beta=.01, reduction="none").mean(dim=1)
        root = (root_element * root_weight).sum() / root_weight.sum().clamp_min(1e-9)
        rotation_obs = ((R[fk_valid_t] - rot_obs[fk_valid_t]) ** 2).mean()
        scale_prior = torch.zeros((), dtype=dtype, device=device)
        trans_acc = ((t[2:] - 2 * t[1:-1] + t[:-2]) ** 2).mean()
        trans_jerk = ((t[3:] - 3 * t[2:-1] + 3 * t[1:-2] - t[:-3]) ** 2).mean()
        translation_step = torch.linalg.vector_norm(t[1:] - t[:-1], dim=1)
        translation_speed = torch.relu(
            translation_step - float(args.max_translation_step_m)
        ).square().mean()
        rot_acc = ((R[2:] - 2 * R[1:-1] + R[:-2]) ** 2).mean()
        rot_vel = ((R[1:] - R[:-1]) ** 2).mean()
        step_fro_sq = ((R[1:] - R[:-1]) ** 2).sum(dim=(1, 2))
        jump_limit = 8.0 * np.sin(np.radians(args.max_rotation_step_deg) * .5) ** 2
        rotation_jump = torch.relu(step_fro_sq - jump_limit).square().mean()
        total = args.w_root * root + args.w_rotation_obs * rotation_obs
        if temporal:
            total = (total + args.w_translation_acc * trans_acc
                     + args.w_translation_jerk * trans_jerk
                     + args.w_translation_speed * translation_speed
                     + args.w_rotation_acc * rot_acc
                     + args.w_rotation_vel * rot_vel + args.w_rotation_jump * rotation_jump)
        return {"total": total, "reprojection": reproj, "root": root, "rotation_observation": rotation_obs,
                "scale_prior": scale_prior, "translation_acceleration": trans_acc,
                "translation_jerk": trans_jerk, "translation_speed": translation_speed,
                "rotation_acceleration": rot_acc, "rotation_velocity": rot_vel,
                "rotation_jump": rotation_jump}

    adam = torch.optim.Adam([t, r6, log_scale], lr=args.root_lr)
    for _ in range(args.root_iterations):
        adam.zero_grad(); loss = losses(False)["total"]; loss.backward(); adam.step()
    lbfgs = torch.optim.LBFGS([t, r6, log_scale], lr=1.0, max_iter=args.smooth_iterations,
                              tolerance_grad=1e-8, tolerance_change=1e-10, line_search_fn="strong_wolfe")
    def closure() -> torch.Tensor:
        lbfgs.zero_grad(); loss = losses(True)["total"]; loss.backward(); return loss
    lbfgs.step(closure)
    final_losses = {key: float(value.detach().cpu()) for key, value in losses(True).items()}
    R_opt = rotation_6d_to_matrix(r6).detach().cpu().numpy()
    root_opt = t.detach().cpu().numpy()
    scale = 1.0
    world_opt = np.einsum("nij,nkj->nki", R_opt, local_filled * scale) + root_opt[:, None, :]

    reprojection_errors = []
    for i, row in enumerate(rows):
        hand = ((row.get("hands") or {}).get(side) or {})
        payload: Dict[str, Any] = {
            "method": "fk_local_world_root_orientation_robust_translation_v4",
            "observed_valid": bool(fk_valid[i]),
            "visual_constraint_valid": bool(valid[i]),
            "interpolated_state": bool(not fk_valid[i]),
            "branch_repaired": bool(rejected[i]),
            "translation_observation_weight": float(root_observation_weight[i]),
            "translation_prediction_residual_m": float(root_prediction_residual[i]),
            "translation_outlier": bool(translation_outliers[i]),
            "global_scale": scale,
        }
        if fk_valid[i]:
            C = c2w[i]
            camera_opt = (C[:3, :3].T @ (world_opt[i] - C[:3, 3]).T).T
            R_camera = project_so3(C[:3, :3].T @ R_opt[i])
            q = (intrinsics[i] @ camera_opt.T).T
            projected = q[:, :2] / q[:, 2:3]
            error = np.linalg.norm(projected - target_px[i], axis=1) if valid[i] else np.full(21, np.nan)
            if valid[i]:
                reprojection_errors.extend(error[PALM].tolist())
            payload.update({
                "kpts_3d_world_m_optimized": world_opt[i].tolist(),
                "kpts_3d_camera_m_optimized": camera_opt.tolist(),
                "wrist_translation_world_m": root_opt[i].tolist(),
                "wrist_translation_camera_m": camera_opt[0].tolist(),
                "palm_rotation_world": R_opt[i].tolist(),
                "palm_rotation_camera": R_camera.tolist(),
                "reprojection_error_px": {
                    "palm_median": float(np.nanmedian(error[PALM])) if valid[i] else None,
                    "palm_p95": float(np.nanpercentile(error[PALM], 95)) if valid[i] else None,
                },
            })
        payload["local_pose_source"] = "glove_fk_wrist_relative"
        payload["visual_constraint_source"] = None
        hand["optimized_trajectory"] = payload
        hand.pop("hamer_trajectory", None)
        hand.pop("palm_frame", None)
        row.setdefault("hands", {})[side] = hand
    return {
        "side": side, "status": "complete", "frames": count, "valid": int(fk_valid.sum()),
        "visual_constraint_frames": int(valid.sum()),
        "branch_repaired_frames": int(rejected.sum()),
        "global_scale": scale, "device": str(device), "losses": final_losses,
        "raw_rotation_step_deg": stats(angular_steps(rotation_world, fk_valid)),
        "optimized_rotation_step_deg": stats(angular_steps(R_opt, fk_valid)),
        "raw_wrist_translation_step_m": stats(translation_steps(root_init, fk_valid)),
        "optimized_wrist_translation_step_m": stats(translation_steps(root_opt, fk_valid)),
        "raw_wrist_translation_second_difference_m": stats(
            translation_second_differences(root_init, fk_valid)
        ),
        "optimized_wrist_translation_second_difference_m": stats(
            translation_second_differences(root_opt, fk_valid)
        ),
        "translation_outlier_frames": int(translation_outliers.sum()),
        "translation_downweighted_frames": int(((root_observation_weight < 0.8) & fk_valid).sum()),
        "translation_min_observation_weight": float(root_observation_weight[fk_valid].min()),
        "fk_local_pose_rms_step_m": stats(local_pose_steps(local_filled, fk_valid)),
        "optimized_palm_reprojection_error_px": None,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    input_path = Path(args.input_jsonl).expanduser().resolve()
    output_path = Path(args.output_jsonl).expanduser().resolve()
    rows = read_jsonl(input_path)
    maps = {"left": smooth_map(args.left_smooth_jsonl), "right": smooth_map(args.right_smooth_jsonl)}
    sides = [side for side in args.sides.split(",") if side in {"left", "right"}]
    summaries = [optimize_side(rows, side, maps[side], args) for side in sides]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, output_path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise
    summary = {"input_jsonl": str(input_path), "output_jsonl": str(output_path), "params": vars(args), "sides": summaries}
    if args.summary_json:
        path = Path(args.summary_json).expanduser().resolve(); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", required=True); parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--left_smooth_jsonl"); parser.add_argument("--right_smooth_jsonl")
    parser.add_argument("--summary_json"); parser.add_argument("--sides", default="left,right")
    parser.add_argument("--device", default="auto"); parser.add_argument("--root_iterations", type=int, default=120)
    parser.add_argument("--smooth_iterations", type=int, default=70); parser.add_argument("--root_lr", type=float, default=.02)
    parser.add_argument("--pixel_norm", type=float, default=424.0)
    parser.add_argument("--min_scale", type=float, default=.75); parser.add_argument("--max_scale", type=float, default=1.25)
    parser.add_argument("--w_reproj", type=float, default=10.0); parser.add_argument("--w_root", type=float, default=8.0)
    parser.add_argument("--w_rotation_obs", type=float, default=2.0); parser.add_argument("--w_scale", type=float, default=.5)
    parser.add_argument("--w_translation_acc", type=float, default=120.0)
    parser.add_argument("--w_translation_jerk", type=float, default=120.0)
    parser.add_argument("--max_translation_step_m", type=float, default=.02)
    parser.add_argument("--w_translation_speed", type=float, default=4000.0)
    parser.add_argument("--translation_outlier_threshold_m", type=float, default=.025)
    parser.add_argument("--min_root_observation_weight", type=float, default=.1)
    parser.add_argument("--w_rotation_acc", type=float, default=6.0); parser.add_argument("--w_rotation_vel", type=float, default=.2)
    parser.add_argument("--max_rotation_step_deg", type=float, default=41.0)
    parser.add_argument("--w_rotation_jump", type=float, default=20.0)
    parser.add_argument("--branch_jump_deg", type=float, default=75.0)
    parser.add_argument("--branch_max_frames", type=int, default=60)
    args = parser.parse_args(); print(json.dumps(run(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
