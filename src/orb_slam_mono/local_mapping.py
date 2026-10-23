"""Segment-local keyframes, calibrated triangulation, and conservative point quality."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .camera import CalibratedCamera
from .map_state import Frame, Keyframe, MapPoint, SparseMap
from .matching import DescriptorMatcher


@dataclass(frozen=True)
class MappingConfig:
    enabled: bool = False  # Exact M3 baseline unless explicitly enabled.
    update_descriptors: bool = False
    min_keyframe_gap: int = 5
    max_keyframe_gap: int = 10
    min_tracked_fraction: float = 0.6
    min_center_baseline: float = 0.5
    window_size: int = 8
    min_triangulation_baseline: float = 0.08
    min_parallax_deg: float = 0.7
    max_reprojection_px: float = 3.0
    max_depth_baselines: float = 200.0
    duplicate_distance_baselines: float = 0.025
    max_new_points_per_keyframe: int = 80
    max_active_points: int = 1000
    stale_after_frames: int = 25
    descriptor_evidence_limit: int = 8
    cull_after_observations: int = 3
    max_point_median_reprojection_px: float = 6.0

    def __post_init__(self) -> None:
        if (self.min_keyframe_gap < 1 or self.max_keyframe_gap < self.min_keyframe_gap
                or not 5 <= self.window_size <= 10 or self.max_new_points_per_keyframe < 1
                or self.descriptor_evidence_limit < 1 or self.cull_after_observations < 3
                or self.max_active_points < 50 or self.stale_after_frames < 1
                or not 0 < self.min_tracked_fraction <= 1
                or min(self.min_center_baseline, self.min_triangulation_baseline,
                       self.min_parallax_deg, self.max_reprojection_px,
                       self.max_depth_baselines, self.duplicate_distance_baselines,
                       self.max_point_median_reprojection_px) <= 0):
            raise ValueError("Invalid local mapping configuration")


def insert_keyframe(sparse_map: SparseMap, frame: Frame, window_size: int) -> Keyframe:
    if frame.frame_id not in sparse_map.frames or sparse_map.frames[frame.frame_id] is not frame:
        raise ValueError("Keyframe must belong to the active map")
    if frame.frame_id in sparse_map.keyframes or not 5 <= window_size <= 10:
        raise ValueError("Duplicate keyframe or invalid window")
    keyframe = Keyframe(frame.frame_id, sparse_map.segment_id, frame)
    sparse_map.keyframes[frame.frame_id] = keyframe
    sparse_map.active_keyframe_ids.append(frame.frame_id)
    if len(sparse_map.active_keyframe_ids) > window_size:
        # Keep the two initialization views as immutable gauge anchors.
        for frame_id in list(sparse_map.active_keyframe_ids):
            if frame_id not in (sparse_map.anchor_frame_id, sparse_map.second_frame_id):
                sparse_map.active_keyframe_ids.remove(frame_id)
                break
    sparse_map.validate()
    return keyframe


def should_insert_keyframe(sparse_map: SparseMap, frame: Frame,
                           config: MappingConfig) -> bool:
    if not config.enabled or frame.R_cw is None or not sparse_map.keyframes:
        return False
    previous = sparse_map.keyframes[max(sparse_map.keyframes)].frame
    gap = frame.frame_id - previous.frame_id
    if gap < config.min_keyframe_gap:
        return False
    tracked_fraction = len(frame.pnp_inlier_features) / max(1, len(previous.feature_to_point))
    baseline = float(np.linalg.norm(frame.camera_center_world - previous.camera_center_world))
    return (gap >= config.max_keyframe_gap or tracked_fraction < config.min_tracked_fraction
            or baseline >= config.min_center_baseline)


def _project(frame: Frame, xyz: np.ndarray, camera: CalibratedCamera) -> tuple[np.ndarray, float]:
    point = frame.R_cw @ xyz + frame.t_cw
    if not np.all(np.isfinite(point)) or point[2] <= 1e-8:
        return np.array([np.inf, np.inf]), float(point[2])
    return np.array([camera.fx * point[0] / point[2] + camera.cx,
                     camera.fy * point[1] / point[2] + camera.cy]), float(point[2])


def triangulate_new_points(sparse_map: SparseMap, first: Frame, second: Frame,
                           camera: CalibratedCamera, matcher: DescriptorMatcher,
                           config: MappingConfig, next_point_id: int) -> dict:
    """Use estimated T_cw directly; new points retain this segment's established scale."""
    started = perf_counter()
    report = {"candidate_matches": 0, "unassigned_matches": 0, "geometry_passed": 0,
              "duplicates_rejected": 0, "capacity_rejected": 0,
              "capacity_culled": 0, "new_points": 0, "runtime_ms": 0.0}
    if (first.frame_id not in sparse_map.keyframes or second.frame_id not in sparse_map.keyframes
            or first.frame_id == second.frame_id or first.features.descriptors is None
            or second.features.descriptors is None):
        return report
    center_a, center_b = first.camera_center_world, second.camera_center_world
    baseline = float(np.linalg.norm(center_b - center_a))
    if baseline < config.min_triangulation_baseline:
        return report
    report["capacity_culled"] = enforce_active_capacity(sparse_map, config, second.frame_id)
    matches = matcher.match(first.features.descriptors, second.features.descriptors).matches
    report["candidate_matches"] = len(matches)
    matches = [m for m in matches if m.queryIdx not in first.feature_to_point
               and m.trainIdx not in second.feature_to_point]
    report["unassigned_matches"] = len(matches)
    if not matches:
        return report
    uv_a = np.array([first.features.keypoints[m.queryIdx].pt for m in matches], dtype=np.float64).T
    uv_b = np.array([second.features.keypoints[m.trainIdx].pt for m in matches], dtype=np.float64).T
    p_a = camera.matrix @ np.column_stack((first.R_cw, first.t_cw))
    p_b = camera.matrix @ np.column_stack((second.R_cw, second.t_cw))
    homogeneous = cv2.triangulatePoints(p_a, p_b, uv_a, uv_b)
    existing = np.array([p.xyz_world for p in sparse_map.points.values() if p.valid], dtype=float).reshape(-1, 3)
    tree = cKDTree(existing) if len(existing) else None
    proposals: list[tuple[int, int, np.ndarray]] = []
    for index, match in enumerate(matches):
        h = homogeneous[:, index]
        if not np.isfinite(h).all() or abs(h[3]) < 1e-10:
            continue
        xyz = h[:3] / h[3]
        if not np.isfinite(xyz).all():
            continue
        ray_a, ray_b = xyz - center_a, xyz - center_b
        distances = np.array([np.linalg.norm(ray_a), np.linalg.norm(ray_b)])
        if min(distances) <= 0 or max(distances) > config.max_depth_baselines * baseline:
            continue
        angle = np.degrees(np.arccos(np.clip(np.dot(ray_a, ray_b) / np.prod(distances), -1, 1)))
        if angle < config.min_parallax_deg:
            continue
        projected_a, depth_a = _project(first, xyz, camera)
        projected_b, depth_b = _project(second, xyz, camera)
        if (min(depth_a, depth_b) <= 0 or
                np.linalg.norm(projected_a - uv_a[:, index]) > config.max_reprojection_px or
                np.linalg.norm(projected_b - uv_b[:, index]) > config.max_reprojection_px):
            continue
        report["geometry_passed"] += 1
        if (tree is not None and tree.query(xyz)[0] < config.duplicate_distance_baselines * baseline) or any(
                np.linalg.norm(xyz - other[2]) < config.duplicate_distance_baselines * baseline
                for other in proposals):
            report["duplicates_rejected"] += 1
            continue
        proposals.append((match.queryIdx, match.trainIdx, xyz.copy()))
        if len(proposals) >= config.max_new_points_per_keyframe:
            break
    # Existing high-quality points have priority. Proposal order follows the
    # deterministic matcher order and preserves the original ORB indices.
    remaining = max(0, config.max_active_points - sum(p.valid for p in sparse_map.points.values()))
    report["capacity_rejected"] = max(0, len(proposals) - remaining)
    proposals = proposals[:remaining]
    # Geometric screening finishes before any map mutation.
    for row, (query, train, xyz) in enumerate(proposals):
        point_id = next_point_id + row
        sparse_map.points[point_id] = MapPoint(point_id, xyz, first.features.descriptors[query].copy())
        sparse_map.add_observation(point_id, first, query)
        sparse_map.add_observation(point_id, second, train)
        if config.update_descriptors:
            sparse_map.points[point_id].update_descriptor(
                second.features.descriptors[train], config.descriptor_evidence_limit)
    sparse_map.validate()
    report["new_points"] = len(proposals)
    report["runtime_ms"] = (perf_counter() - started) * 1000
    return report


def cull_inconsistent_points(sparse_map: SparseMap, camera: CalibratedCamera,
                             config: MappingConfig, current_frame_id: int | None = None,
                             candidate_ids: set[int] | None = None) -> int:
    """Deactivate only well-observed points with persistent geometric disagreement."""
    culled = 0
    inspected = (sparse_map.points[i] for i in candidate_ids if i in sparse_map.points) if candidate_ids is not None else sparse_map.points.values()
    for point in inspected:
        if not point.valid or point.observation_count < config.cull_after_observations:
            continue
        errors = []
        for frame_id, feature_index in point.observations.items():
            frame = sparse_map.frames[frame_id]
            if frame.R_cw is None:
                continue
            projected, depth = _project(frame, point.xyz_world, camera)
            errors.append(float(np.linalg.norm(projected - frame.features.keypoints[feature_index].pt))
                         if depth > 0 else np.inf)
        if len(errors) >= config.cull_after_observations and np.median(errors) > config.max_point_median_reprojection_px:
            point.valid = False
            culled += 1
    if current_frame_id is not None:
        culled += enforce_active_capacity(sparse_map, config, current_frame_id)
    sparse_map.validate()
    return culled


def enforce_active_capacity(sparse_map: SparseMap, config: MappingConfig,
                            current_frame_id: int) -> int:
    """Hard active-point bound; keep stronger/recent points, retain all identities.

    Already-overfull maps are corrected deterministically: stale points first,
    then lower observation support, older last observation, and point ID. No
    observation or descriptor is deleted; invalid points are excluded by PnP.
    """
    active = [point for point in sparse_map.points.values() if point.valid]
    excess = len(active) - config.max_active_points
    if excess <= 0:
        return 0
    def priority(point: MapPoint) -> tuple[int, int, int, int]:
        last = max(point.observations, default=-1)
        stale = current_frame_id - last >= config.stale_after_frames
        return (0 if stale else 1, point.observation_count, last, point.point_id)
    for point in sorted(active, key=priority)[:excess]:
        point.valid = False
    sparse_map.validate()
    return excess
