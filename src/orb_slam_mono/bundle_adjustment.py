"""Gauge-fixed sparse local bundle adjustment for one monocular map segment."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .camera import CalibratedCamera
from .map_state import SparseMap


@dataclass(frozen=True)
class BundleConfig:
    enabled: bool = False
    every_n_keyframes: int = 2
    max_points: int = 60
    max_observations: int = 300
    min_points: int = 12
    max_nfev: int = 12
    loss: str = "soft_l1"
    loss_scale_px: float = 2.0
    max_median_reprojection_px: float = 4.0
    max_heldout_increase_px: float = 0.5
    max_heldout_p90_increase_px: float = 1.0
    max_heldout_max_increase_px: float = 2.0
    max_validation_reprojection_px: float = 4.0
    min_validation_observations: int = 10

    def __post_init__(self) -> None:
        if (self.every_n_keyframes < 1 or self.max_points < 1 or self.max_observations < 4
                or self.min_points < 1 or self.max_nfev < 1 or self.loss not in ("huber", "soft_l1")
                or self.loss_scale_px <= 0 or self.max_median_reprojection_px <= 0
                or self.max_heldout_increase_px < 0
                or self.max_heldout_p90_increase_px < 0
                or self.max_heldout_max_increase_px < 0
                or self.max_validation_reprojection_px <= 0
                or self.min_validation_observations < 1):
            raise ValueError("Invalid bundle adjustment configuration")


def jacobian_sparsity(pose_columns: list[int | None], point_columns: list[int],
                      variable_count: int):
    """Two pixel rows depend on one landmark and at most one variable camera."""
    pattern = lil_matrix((2 * len(point_columns), variable_count), dtype=np.int8)
    for row, (pose_start, point_start) in enumerate(zip(pose_columns, point_columns)):
        pattern[2 * row:2 * row + 2, point_start:point_start + 3] = 1
        if pose_start is not None:
            pattern[2 * row:2 * row + 2, pose_start:pose_start + 6] = 1
    return pattern.tocsr()


def _projection(R: np.ndarray, t: np.ndarray, xyz: np.ndarray,
                camera: CalibratedCamera) -> tuple[np.ndarray, np.ndarray]:
    points = (R @ xyz.T).T + t
    z = points[:, 2]
    safe_z = np.where(np.abs(z) < 1e-8, np.where(z < 0, -1e-8, 1e-8), z)
    uv = np.column_stack((camera.fx * points[:, 0] / safe_z + camera.cx,
                          camera.fy * points[:, 1] / safe_z + camera.cy))
    return uv, z


def optimize_local_map(sparse_map: SparseMap, camera: CalibratedCamera,
                       config: BundleConfig) -> dict:
    started = perf_counter()
    report = {"status": "rejected", "reason": None, "applied": False,
              "keyframes": 0, "landmarks": 0, "observations": 0,
              "initial_cost": None, "final_cost": None,
              "initial_median_px": None, "final_median_px": None,
              "heldout_initial_median_px": None, "heldout_final_median_px": None,
              "heldout_initial_p90_px": None, "heldout_final_p90_px": None,
              "heldout_initial_max_px": None, "heldout_final_max_px": None,
              "heldout_observations": 0, "heldout_outside_window_observations": 0,
              "heldout_new_quality_violations": None,
              "solver_termination": None,
              "nfev": None, "solver_success": None, "solver_status": None,
              "solver_message": None, "runtime_ms": None}

    def reject(reason: str) -> dict:
        report["reason"] = reason
        report["runtime_ms"] = (perf_counter() - started) * 1000
        return report

    ids = list(sparse_map.active_keyframe_ids)
    fixed = {sparse_map.anchor_frame_id, sparse_map.second_frame_id}
    if len(ids) < 3 or not fixed.issubset(ids) or len(ids) > 10:
        return reject("underconstrained_keyframe_window")
    variable_poses = [frame_id for frame_id in ids if frame_id not in fixed]
    if not variable_poses:
        return reject("no_variable_pose")
    eligible = []
    for point in sparse_map.points.values():
        views = [i for i in ids if i in point.observations]
        if point.valid and len(views) >= 2:
            eligible.append((point, views))
    eligible.sort(key=lambda item: (-len(item[1]), item[0].point_id))
    eligible = eligible[:config.max_points]
    if len(eligible) < config.min_points:
        return reject("insufficient_multiview_points")

    # Bipartite observation graph must connect every optimized pose/point to
    # the two immutable initialization cameras. They fix SE(3) and baseline scale.
    reachable_frames = {sparse_map.anchor_frame_id}
    reachable_points: set[int] = set()
    while True:
        new_points = {p.point_id for p, views in eligible if set(views) & reachable_frames}
        new_frames = {i for p, views in eligible if p.point_id in new_points for i in views}
        if new_points == reachable_points and new_frames <= reachable_frames:
            break
        reachable_points |= new_points
        reachable_frames |= new_frames
    if (sparse_map.second_frame_id not in reachable_frames
            or any(i not in reachable_frames for i in variable_poses) or any(
            p.point_id not in reachable_points for p, _ in eligible)):
        return reject("disconnected_observation_graph")

    observations = []
    for point, views in eligible:
        for frame_id in views:
            frame = sparse_map.frames[frame_id]
            feature = point.observations[frame_id]
            observations.append((frame_id, point.point_id,
                                 np.array(frame.features.keypoints[feature].pt, dtype=float)))
    observations.sort(key=lambda row: (row[0], row[1]))
    if len(observations) > config.max_observations:
        # Whole-point selection avoids single-view truncation of a landmark.
        allowed = set()
        used = 0
        for point, views in eligible:
            if used + len(views) <= config.max_observations:
                allowed.add(point.point_id)
                used += len(views)
        eligible = [(p, views) for p, views in eligible if p.point_id in allowed]
        observations = [row for row in observations if row[1] in allowed]
    if len(eligible) < config.min_points:
        return reject("insufficient_bounded_points")
    bounded_reachable = {sparse_map.anchor_frame_id}
    while True:
        expanded = bounded_reachable | {i for _, views in eligible
                                         if set(views) & bounded_reachable for i in views}
        if expanded == bounded_reachable:
            break
        bounded_reachable = expanded
    if sparse_map.second_frame_id not in bounded_reachable or any(i not in bounded_reachable for i in variable_poses):
        return reject("bounded_graph_disconnected")
    pose_offset = {i: 6 * row for row, i in enumerate(variable_poses)}
    point_offset = {p.point_id: 6 * len(variable_poses) + 3 * row
                    for row, (p, _) in enumerate(eligible)}
    variable_count = 6 * len(variable_poses) + 3 * len(eligible)
    if 2 * len(observations) <= variable_count:
        return reject("underdetermined_observation_graph")
    x0 = np.zeros(variable_count, dtype=float)
    for frame_id in variable_poses:
        frame = sparse_map.frames[frame_id]
        x0[pose_offset[frame_id]:pose_offset[frame_id] + 3] = cv2.Rodrigues(frame.R_cw)[0].reshape(3)
        x0[pose_offset[frame_id] + 3:pose_offset[frame_id] + 6] = frame.t_cw
    for point, _ in eligible:
        x0[point_offset[point.point_id]:point_offset[point.point_id] + 3] = point.xyz_world
    pattern = jacobian_sparsity([pose_offset.get(i) for i, _, _ in observations],
                                [point_offset[j] for _, j, _ in observations], variable_count)
    pixels = np.array([uv for _, _, uv in observations])

    def unpack(values):
        poses = {i: (sparse_map.frames[i].R_cw, sparse_map.frames[i].t_cw) for i in fixed}
        for i in variable_poses:
            k = pose_offset[i]
            poses[i] = (cv2.Rodrigues(values[k:k + 3])[0], values[k + 3:k + 6])
        xyz = {p.point_id: values[point_offset[p.point_id]:point_offset[p.point_id] + 3]
               for p, _ in eligible}
        return poses, xyz

    def errors_and_depth(values):
        poses, points = unpack(values)
        projected = np.empty_like(pixels)
        depths = np.empty(len(observations))
        for row, (frame_id, point_id, _) in enumerate(observations):
            R, t = poses[frame_id]
            uv, depth = _projection(R, t, points[point_id].reshape(1, 3), camera)
            projected[row], depths[row] = uv[0], depth[0]
        return projected - pixels, depths

    def residual(values):
        err, depth = errors_and_depth(values)
        if not np.isfinite(err).all():
            return np.full(2 * len(observations), 1e6)
        # Negative depth is never an acceptable final map; large residual keeps
        # the optimizer away from such candidate configurations.
        return (err + np.maximum(0, 1e-4 - depth)[:, None] * 100).ravel()

    initial_err, initial_depth = errors_and_depth(x0)
    if not np.isfinite(initial_err).all() or np.any(initial_depth <= 0):
        return reject("invalid_initial_geometry")
    report.update(keyframes=len(ids), landmarks=len(eligible), observations=len(observations),
                  initial_median_px=float(np.median(np.linalg.norm(initial_err, axis=1))))
    try:
        solution = least_squares(residual, x0, jac_sparsity=pattern, method="trf",
                                 loss=config.loss, f_scale=config.loss_scale_px,
                                 max_nfev=config.max_nfev)
    except (ValueError, FloatingPointError):
        return reject("optimizer_failed")
    report.update(solver_success=bool(solution.success), solver_status=int(solution.status),
                  solver_message=str(solution.message),
                  solver_termination=("converged" if solution.success else
                                      "budget_exhausted" if solution.status == 0 else "failed"))
    report.update(initial_cost=float(np.sum(residual(x0) ** 2)),
                  final_cost=float(np.sum(residual(solution.x) ** 2)), nfev=int(solution.nfev))
    if not np.isfinite(solution.x).all():
        return reject("nonfinite_solution")
    if solution.status < 0 or (not solution.success and solution.status != 0):
        return reject("solver_failed")
    final_err, final_depth = errors_and_depth(solution.x)
    median = float(np.median(np.linalg.norm(final_err, axis=1)))
    report["final_median_px"] = median
    if (not np.isfinite(final_err).all() or np.any(final_depth <= 0)
            or report["final_cost"] >= report["initial_cost"]
            or median > max(config.max_median_reprojection_px,
                            report["initial_median_px"] * 1.05)):
        return reject("optimization_quality_gate")
    poses, points = unpack(solution.x)
    if any(not np.isfinite(R).all() or abs(np.linalg.det(R) - 1) > 1e-4
           for R, _ in poses.values()):
        return reject("invalid_rotation")

    # Every stored measurement affected by a moving camera or landmark, but not
    # used for training, validates the candidate in this segment's world frame.
    trained = {(i, j) for i, j, _ in observations}
    heldout = []
    for point in sparse_map.points.values():
        for frame_id, feature in point.observations.items():
            if ((point.point_id in points or frame_id in variable_poses)
                    and (frame_id, point.point_id) not in trained):
                frame = sparse_map.frames[frame_id]
                if frame.R_cw is None or frame.t_cw is None:
                    return reject("invalid_validation_pose")
                heldout.append((frame_id, point.point_id,
                                np.array(frame.features.keypoints[feature].pt)))
    report["heldout_observations"] = len(heldout)
    report["heldout_outside_window_observations"] = sum(i not in ids for i, _, _ in heldout)

    def heldout_errors(candidate: bool) -> np.ndarray:
        errors = []
        for frame_id, point_id, uv in heldout:
            R, t = (poses[frame_id] if candidate and frame_id in variable_poses else
                    (sparse_map.frames[frame_id].R_cw, sparse_map.frames[frame_id].t_cw))
            xyz = points.get(point_id, sparse_map.points[point_id].xyz_world) if candidate else sparse_map.points[point_id].xyz_world
            projected, depth = _projection(R, t, xyz.reshape(1, 3), camera)
            errors.append(float(np.linalg.norm(projected[0] - uv))
                          if depth[0] > 0 and np.isfinite(projected).all() else np.inf)
        return np.asarray(errors)

    if len(heldout) < config.min_validation_observations:
        return reject("insufficient_validation_observations")
    before, after = heldout_errors(False), heldout_errors(True)
    report.update(heldout_initial_median_px=float(np.median(before)),
                  heldout_final_median_px=float(np.median(after)),
                  heldout_initial_p90_px=float(np.percentile(before, 90)),
                  heldout_final_p90_px=float(np.percentile(after, 90)),
                  heldout_initial_max_px=float(np.max(before)),
                  heldout_final_max_px=float(np.max(after)))
    newly_bad = int(np.sum((before <= config.max_validation_reprojection_px)
                           & (after > config.max_validation_reprojection_px)))
    report["heldout_new_quality_violations"] = newly_bad
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        return reject("invalid_validation_geometry")
    if newly_bad or (report["heldout_final_median_px"] > report["heldout_initial_median_px"] + config.max_heldout_increase_px
                     or report["heldout_final_p90_px"] > report["heldout_initial_p90_px"] + config.max_heldout_p90_increase_px
                     or report["heldout_final_max_px"] > report["heldout_initial_max_px"] + config.max_heldout_max_increase_px):
        return reject("heldout_reprojection_regression")
    # Transaction: only validated candidate values reach the live map.
    snapshots = {i: (sparse_map.frames[i].R_cw.copy(), sparse_map.frames[i].t_cw.copy())
                 for i in variable_poses}
    old_points = {j: sparse_map.points[j].xyz_world.copy() for j in points}
    try:
        for i in variable_poses:
            sparse_map.frames[i].R_cw, sparse_map.frames[i].t_cw = poses[i][0].copy(), poses[i][1].copy()
        for j, xyz in points.items():
            sparse_map.points[j].xyz_world = xyz.copy()
        sparse_map.validate()
    except (ValueError, AssertionError):
        for i, (R, t) in snapshots.items():
            sparse_map.frames[i].R_cw, sparse_map.frames[i].t_cw = R, t
        for j, xyz in old_points.items():
            sparse_map.points[j].xyz_world = xyz
        return reject("transaction_rollback")
    report.update(status="accepted", reason=None, applied=True,
                  runtime_ms=(perf_counter() - started) * 1000)
    return report
