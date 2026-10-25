"""Sequential initialization and fixed-landmark PnP tracking, with no truth input."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .camera import CalibratedCamera
from .datasets import ImageSample
from .features import OrbExtractor
from .geometry import GeometryConfig, initialize_two_view, _valid_rotation
from .map_state import Frame, SparseMap, TrackingState, map_from_initialization
from .matching import DescriptorMatcher
from .local_mapping import (MappingConfig, insert_keyframe, should_insert_keyframe,
                            triangulate_new_points, cull_inconsistent_points,
                            enforce_active_capacity)
from .bundle_adjustment import BundleConfig, optimize_local_map


@dataclass(frozen=True)
class TrackingConfig:
    initialization_min_gap: int = 3
    initialization_attempt_step: int = 2
    initialization_max_gap: int = 11
    min_pnp_correspondences: int = 12
    min_pnp_inliers: int = 12
    min_pnp_inlier_fraction: float = 0.5
    pnp_ransac_reprojection_px: float = 3.0
    pnp_ransac_confidence: float = 0.999
    pnp_ransac_iterations: int = 100
    min_positive_depth_fraction: float = 0.9
    max_median_reprojection_px: float = 2.5
    max_p90_reprojection_px: float = 4.0
    max_recovery_frames: int = 60  # Processed frames after failed PnP; source-ID gaps do not consume budget.

    def __post_init__(self) -> None:
        if isinstance(self.max_recovery_frames, bool) or not isinstance(self.max_recovery_frames, int):
            raise ValueError("max_recovery_frames must count whole processed frames")
        if (self.initialization_min_gap < 1 or self.initialization_attempt_step < 1
                or self.initialization_max_gap < self.initialization_min_gap
                or self.min_pnp_correspondences < 4 or self.min_pnp_inliers < 4
                or self.pnp_ransac_iterations < 1 or self.max_recovery_frames < 1):
            raise ValueError("Invalid positive tracking count or initialization schedule")
        values = (self.min_pnp_inlier_fraction, self.pnp_ransac_reprojection_px,
                  self.pnp_ransac_confidence, self.min_positive_depth_fraction,
                  self.max_median_reprojection_px, self.max_p90_reprojection_px)
        if (not np.all(np.isfinite(values)) or not 0 < self.min_pnp_inlier_fraction <= 1
                or not 0 < self.pnp_ransac_confidence < 1
                or not 0 < self.min_positive_depth_fraction <= 1
                or min(self.pnp_ransac_reprojection_px, self.max_median_reprojection_px,
                       self.max_p90_reprojection_px) <= 0):
            raise ValueError("Invalid PnP threshold")


@dataclass
class TrackingResult:
    frame_id: int
    state: TrackingState
    segment_id: int | None = None
    R_cw: np.ndarray | None = None
    t_cw: np.ndarray | None = None
    failure_reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    timing_ms: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        center = None if self.R_cw is None else -self.R_cw.T @ self.t_cw
        return {"frame_id": self.frame_id, "state": self.state.value,
                "segment_id": self.segment_id, "failure_reason": self.failure_reason,
                "pose_valid": self.R_cw is not None,
                "T_cw_R": None if self.R_cw is None else self.R_cw.tolist(),
                "T_cw_t": None if self.t_cw is None else self.t_cw.tolist(),
                "camera_center_world": None if center is None else center.tolist(),
                "diagnostics": self.diagnostics, "timing_ms": self.timing_ms}


@dataclass
class PnPResult:
    accepted: bool = False
    reason: str | None = None
    R_cw: np.ndarray | None = None
    t_cw: np.ndarray | None = None
    inlier_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    diagnostics: dict[str, Any] = field(default_factory=dict)


def solve_map_pnp(xyz_world: np.ndarray, pixels: np.ndarray,
                  camera: CalibratedCamera, config: TrackingConfig,
                  diagnostic_capture: dict[str, Any] | None = None) -> PnPResult:
    """Return a pose only after numerical, inlier, depth and reprojection gates."""
    result = PnPResult(diagnostics={"pnp_ransac_attempted": False})
    try:
        xyz = np.asarray(xyz_world, dtype=np.float64)
        uv = np.asarray(pixels, dtype=np.float64)
    except (TypeError, ValueError):
        result.reason = "invalid_correspondences"
        return result
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or uv.shape != (len(xyz), 2) or not np.isfinite(xyz).all() or not np.isfinite(uv).all():
        result.reason = "invalid_correspondences"
        return result
    n = len(xyz)
    result.diagnostics["pnp_correspondences"] = n
    if n < config.min_pnp_correspondences:
        result.reason = "insufficient_pnp_correspondences"
        return result
    result.diagnostics["pnp_ransac_attempted"] = True
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            xyz, uv, camera.matrix, None, iterationsCount=config.pnp_ransac_iterations,
            reprojectionError=config.pnp_ransac_reprojection_px,
            confidence=config.pnp_ransac_confidence, flags=cv2.SOLVEPNP_EPNP)
    except cv2.error:
        result.reason = "pnp_solver_failed"
        return result
    if diagnostic_capture is not None:
        diagnostic_capture.update(
            ransac_success=bool(success),
            raw_rvec=None if not success or rvec is None else np.asarray(rvec, dtype=np.float64).reshape(-1).copy(),
            raw_tvec=None if not success or tvec is None else np.asarray(tvec, dtype=np.float64).reshape(-1).copy(),
            raw_inlier_indices=None if not success or inliers is None else np.asarray(inliers, dtype=int).reshape(-1).copy(),
            pose_stage="unrefined RANSAC candidate; never an accepted pose by itself")
    if not success or inliers is None or rvec is None or tvec is None:
        result.reason = "pnp_solver_failed"
        return result
    idx = np.asarray(inliers, dtype=int).reshape(-1)
    if len(np.unique(idx)) != len(idx) or np.any(idx < 0) or np.any(idx >= n):
        result.reason = "invalid_pnp_inliers"
        return result
    result.diagnostics["pnp_inliers"] = len(idx)
    result.diagnostics["pnp_inlier_fraction"] = len(idx) / n
    if len(idx) < config.min_pnp_inliers or len(idx) / n < config.min_pnp_inlier_fraction:
        result.reason = "insufficient_pnp_inliers"
        return result
    try:
        # Inlier-only iterative refinement is constrained by RANSAC support.
        refined, rvec, tvec = cv2.solvePnP(xyz[idx], uv[idx], camera.matrix, None,
                                           rvec, tvec, useExtrinsicGuess=True,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
        rotation, _ = cv2.Rodrigues(rvec)
    except cv2.error:
        result.reason = "pnp_refinement_failed"
        return result
    translation = np.asarray(tvec, dtype=np.float64).reshape(-1)
    if not refined or not _valid_rotation(rotation) or translation.shape != (3,) or not np.isfinite(translation).all():
        result.reason = "invalid_pnp_pose"
        return result
    points_camera = (rotation @ xyz[idx].T).T + translation
    depth_fraction = float(np.mean(points_camera[:, 2] > 1e-8))
    result.diagnostics["positive_depth_fraction"] = depth_fraction
    if depth_fraction < config.min_positive_depth_fraction:
        result.reason = "insufficient_positive_depth"
        return result
    projected, _ = cv2.projectPoints(xyz[idx], rvec, tvec, camera.matrix, None)
    errors = np.linalg.norm(projected.reshape(-1, 2) - uv[idx], axis=1)
    if not np.isfinite(errors).all():
        result.reason = "invalid_reprojection"
        return result
    median, p90 = float(np.median(errors)), float(np.percentile(errors, 90))
    result.diagnostics.update(median_reprojection_px=median, p90_reprojection_px=p90)
    if median > config.max_median_reprojection_px or p90 > config.max_p90_reprojection_px:
        result.reason = "excessive_pnp_reprojection_error"
        return result
    result.accepted = True
    result.R_cw, result.t_cw, result.inlier_indices = rotation, translation, idx
    return result


class MonocularTracker:
    """One-frame-at-a-time controller; initial map points remain fixed in M3."""

    LEGAL_TRANSITIONS = {
        TrackingState.INITIALIZING: {TrackingState.INITIALIZING, TrackingState.TRACKING, TrackingState.LOST},
        TrackingState.TRACKING: {TrackingState.TRACKING, TrackingState.LOST},
        TrackingState.LOST: {TrackingState.LOST, TrackingState.INITIALIZING},
    }

    def __init__(self, camera: CalibratedCamera, extractor: OrbExtractor,
                 matcher: DescriptorMatcher, geometry: GeometryConfig,
                 config: TrackingConfig, mapping: MappingConfig | None = None,
                 bundle: BundleConfig | None = None,
                 *, capture_pnp_diagnostics: bool = False):
        self.camera, self.extractor, self.matcher = camera, extractor, matcher
        self.geometry, self.config = geometry, config
        self.mapping = mapping or MappingConfig()
        self.bundle = bundle or BundleConfig()
        self.keyframe_events: list[dict[str, Any]] = []
        self.mapping_events: list[dict[str, Any]] = []
        self.ba_events: list[dict[str, Any]] = []
        self.state = TrackingState.INITIALIZING
        self.anchor: Frame | None = None
        self.anchor_image: np.ndarray | None = None
        self.active_map: SparseMap | None = None
        self.segments: list[SparseMap] = []
        self.attempts: list[dict[str, Any]] = []
        self.loss_events: list[dict[str, Any]] = []
        self.recovery_events: list[dict[str, Any]] = []
        self.last_successful_pose: dict[str, Any] | None = None
        self.recovery_start: int | None = None
        self.recovery_processed_frames = 0
        self.terminal_lost = False
        self.last_lost_segment_id: int | None = None
        self._next_segment = 0
        self._next_point_id = 0
        self._last_frame_id: int | None = None
        self.visual_event: dict[str, Any] | None = None
        self.capture_pnp_diagnostics = capture_pnp_diagnostics
        self.diagnostic_event: dict[str, Any] | None = None

    def _transition(self, next_state: TrackingState) -> None:
        if next_state not in self.LEGAL_TRANSITIONS[self.state]:
            raise ValueError(f"Illegal tracker transition {self.state.value} -> {next_state.value}")
        self.state = next_state

    def process(self, sample: ImageSample) -> TrackingResult:
        if self._last_frame_id is not None and sample.index <= self._last_frame_id:
            raise ValueError("Frame IDs must increase strictly within a tracker run")
        self._last_frame_id = sample.index
        start = perf_counter()
        self.visual_event = None
        self.diagnostic_event = None
        if self.terminal_lost:
            return self._terminal_result(sample.index, start, first_timeout=False)
        if self.recovery_start is not None and self.state != TrackingState.TRACKING:
            if self.recovery_processed_frames >= self.config.max_recovery_frames:
                self._transition(TrackingState.LOST)
                self.terminal_lost = True
                self.anchor = None
                self.anchor_image = None
                self.recovery_events.append({"frame_id": sample.index, "reason": "recovery_window_exhausted",
                                             "recovery_start_frame": self.recovery_start,
                                             "recovery_frames_processed": self.recovery_processed_frames,
                                             "lost_segment_id": self.last_lost_segment_id})
                return self._terminal_result(sample.index, start, first_timeout=True)
            self.recovery_processed_frames += 1
            if self.state == TrackingState.LOST:
                self._transition(TrackingState.INITIALIZING)
        extracted = perf_counter()
        frame = Frame.from_sample(sample, self.extractor.extract(sample.image))
        timing = {"orb": (perf_counter() - extracted) * 1000.0}
        result = TrackingResult(frame.frame_id, self.state,
                                self.active_map.segment_id if self.active_map else None,
                                diagnostics={"features_detected": frame.features.count,
                                             "available_map_points": len(self.active_map.points) if self.active_map else 0},
                                timing_ms=timing)
        if self.recovery_start is not None and self.state == TrackingState.INITIALIZING:
            result.diagnostics.update(recovery_search_active=True,
                                      recovery_start_frame=self.recovery_start,
                                      recovery_frames_processed=self.recovery_processed_frames,
                                      recovery_frames_remaining=self.config.max_recovery_frames - self.recovery_processed_frames,
                                      lost_segment_id=self.last_lost_segment_id)
        if self.state == TrackingState.TRACKING and self.active_map is not None:
            self._track(frame, result, sample.image)
        else:
            self._initialize(frame, result, sample.image)
        timing["total"] = (perf_counter() - start) * 1000.0
        result.state = self.state
        return result

    def _terminal_result(self, frame_id: int, started: float, first_timeout: bool) -> TrackingResult:
        """A terminally lost tracker accepts later IDs without retrying or publishing poses."""
        return TrackingResult(
            frame_id, TrackingState.LOST, failure_reason=("recovery_window_exhausted" if first_timeout
                                                          else "terminal_recovery_exhausted"),
            diagnostics={"terminal_recovery_exhausted": True,
                         "recovery_search_active": False, "initialization_attempted": False,
                         "recovery_start_frame": self.recovery_start,
                         "recovery_frames_processed": self.recovery_processed_frames,
                         "recovery_budget_frames": self.config.max_recovery_frames,
                         "lost_segment_id": self.last_lost_segment_id,
                         "last_successful_frame": None if self.last_successful_pose is None
                         else self.last_successful_pose["frame_id"],
                         "features_detected": None, "available_map_points": 0,
                         "feature_extraction_skipped": True},
            timing_ms={"total": (perf_counter() - started) * 1000.0})

    def _initialize(self, frame: Frame, result: TrackingResult, image: np.ndarray) -> None:
        result.state = TrackingState.INITIALIZING
        if self.anchor is None:
            self.anchor = frame
            self.anchor_image = image
            result.failure_reason = "waiting_for_initialization_gap"
            return
        gap = frame.frame_id - self.anchor.frame_id
        if gap < self.config.initialization_min_gap:
            result.failure_reason = "waiting_for_initialization_gap"
            return
        if gap > self.config.initialization_max_gap:
            self.anchor = frame
            self.anchor_image = image
            result.failure_reason = "initialization_window_exhausted_anchor_replaced"
            return
        if (gap - self.config.initialization_min_gap) % self.config.initialization_attempt_step:
            result.failure_reason = "waiting_for_initialization_schedule"
            return
        started = perf_counter()
        matches = self.matcher.match(self.anchor.features.descriptors, frame.features.descriptors)
        result.diagnostics["initialization_attempted"] = True
        result.timing_ms["association"] = (perf_counter() - started) * 1000.0
        result.diagnostics.update(anchor_frame=self.anchor.frame_id, initialization_gap=gap,
                                  descriptor_knn_pairs=matches.knn_pairs,
                                  descriptor_ratio_passed=matches.ratio_passed,
                                  descriptor_matches=matches.retained)
        first = np.array([self.anchor.features.keypoints[m.queryIdx].pt for m in matches.matches], dtype=np.float64).reshape(-1, 2)
        second = np.array([frame.features.keypoints[m.trainIdx].pt for m in matches.matches], dtype=np.float64).reshape(-1, 2)
        started = perf_counter()
        geometry = initialize_two_view(first, second, self.camera, self.geometry)
        self.visual_event = {"kind": "initialization", "first_image": self.anchor_image,
                             "second_image": image, "first_keypoints": self.anchor.features.keypoints,
                             "second_keypoints": frame.features.keypoints, "matches": matches.matches,
                             "essential_mask": geometry.essential_mask.copy(),
                             "accepted_mask": geometry.accepted_mask.copy(),
                             "status": geometry.status}
        result.timing_ms["initialization"] = (perf_counter() - started) * 1000.0
        result.diagnostics["geometry"] = geometry.diagnostics
        attempt = {"anchor_frame": self.anchor.frame_id, "candidate_frame": frame.frame_id,
                   "gap": gap, "matches": matches.retained, **geometry.as_dict()}
        self.attempts.append(attempt)
        if geometry.status != "accepted":
            result.failure_reason = geometry.failure_reason
            if gap + self.config.initialization_attempt_step > self.config.initialization_max_gap:
                self.anchor = frame
                self.anchor_image = image
                result.diagnostics["anchor_replaced"] = True
            return
        started = perf_counter()
        sparse_map = map_from_initialization(self._next_segment, self.anchor, frame,
                                              matches.matches, geometry, self._next_point_id)
        if self.mapping.enabled:
            enforce_active_capacity(sparse_map, self.mapping, frame.frame_id)
            insert_keyframe(sparse_map, sparse_map.frames[sparse_map.anchor_frame_id], self.mapping.window_size)
            insert_keyframe(sparse_map, frame, self.mapping.window_size)
            self.keyframe_events.extend({"segment_id": sparse_map.segment_id,
                                         "frame_id": i, "reason": "initialization"}
                                        for i in (sparse_map.anchor_frame_id, frame.frame_id))
        self._next_segment += 1
        self._next_point_id += len(sparse_map.points)
        self.active_map = sparse_map
        self.segments.append(sparse_map)
        self._transition(TrackingState.TRACKING)
        self.recovery_start = None
        self.recovery_processed_frames = 0
        self.last_lost_segment_id = None
        self.anchor = None
        self.anchor_image = None
        result.state = TrackingState.TRACKING
        result.segment_id = sparse_map.segment_id
        result.R_cw, result.t_cw = frame.R_cw.copy(), frame.t_cw.copy()
        result.diagnostics["initialization_pair"] = [sparse_map.anchor_frame_id, frame.frame_id]
        result.diagnostics["map_points"] = len(sparse_map.points)
        result.timing_ms["map_update"] = (perf_counter() - started) * 1000.0
        self.last_successful_pose = result.as_dict()

    def _track(self, frame: Frame, result: TrackingResult, image: np.ndarray) -> None:
        sparse_map = self.active_map
        assert sparse_map is not None
        started = perf_counter()
        point_ids = sorted(point_id for point_id, point in sparse_map.points.items() if point.valid)
        descriptors = np.stack([sparse_map.points[point_id].descriptor for point_id in point_ids]) if point_ids else None
        match_capture: dict[str, Any] | None = {} if self.capture_pnp_diagnostics else None
        matches = (self.matcher.match(descriptors, frame.features.descriptors)
                   if match_capture is None else
                   self.matcher.match(descriptors, frame.features.descriptors, diagnostic=match_capture))
        result.timing_ms["association"] = (perf_counter() - started) * 1000.0
        result.diagnostics.update(candidate_map_associations=matches.knn_pairs,
                                  ratio_passed_map_associations=matches.ratio_passed,
                                  unique_map_associations=matches.retained,
                                  map_points=len(sparse_map.points))
        xyz = np.array([sparse_map.points[point_ids[m.queryIdx]].xyz_world for m in matches.matches], dtype=np.float64).reshape(-1, 3)
        uv = np.array([frame.features.keypoints[m.trainIdx].pt for m in matches.matches], dtype=np.float64).reshape(-1, 2)
        pnp_capture: dict[str, Any] | None = {} if self.capture_pnp_diagnostics else None
        if self.capture_pnp_diagnostics:
            self.diagnostic_event = {
                "frame_id": frame.frame_id, "segment_id": sparse_map.segment_id,
                "keypoints_xy": np.asarray([keypoint.pt for keypoint in frame.features.keypoints], dtype=np.float64),
                "candidate_point_ids": list(point_ids), "matcher": match_capture,
                "active_landmarks": [
                    {"point_id": point_id, "xyz_world": sparse_map.points[point_id].xyz_world.copy(),
                     "observation_count": sparse_map.points[point_id].observation_count,
                     "first_observed_frame": min(sparse_map.points[point_id].observations),
                     "last_observed_frame": max(sparse_map.points[point_id].observations),
                     "age_since_observation_frames": frame.frame_id - max(sparse_map.points[point_id].observations),
                     "descriptor_evidence_count": len(sparse_map.points[point_id].descriptor_evidence)}
                    for point_id in point_ids],
                "correspondences": [
                    {"point_id": point_ids[match.queryIdx], "image_feature_index": int(match.trainIdx),
                     "pixel_xy": uv[row].copy(), "xyz_world": xyz[row].copy(),
                     "hamming": float(match.distance)}
                    for row, match in enumerate(matches.matches)],
                "previous_accepted_pose": None if self.last_successful_pose is None else {
                    key: self.last_successful_pose[key] for key in
                    ("frame_id", "segment_id", "T_cw_R", "T_cw_t")},
                "pnp_candidate": pnp_capture,
            }
        started = perf_counter()
        pnp = (solve_map_pnp(xyz, uv, self.camera, self.config)
               if pnp_capture is None else
               solve_map_pnp(xyz, uv, self.camera, self.config, diagnostic_capture=pnp_capture))
        if self.diagnostic_event is not None:
            self.diagnostic_event["pnp_accepted"] = pnp.accepted
            self.diagnostic_event["pnp_rejection_reason"] = pnp.reason
        self.visual_event = {"kind": "pnp", "image": image, "pixels": uv,
                             "inlier_indices": pnp.inlier_indices.copy(),
                             "status": "accepted" if pnp.accepted else "rejected"}
        result.timing_ms["pnp"] = (perf_counter() - started) * 1000.0
        result.diagnostics.update(pnp.diagnostics)
        result.diagnostics["pnp_status"] = "accepted" if pnp.accepted else "rejected"
        result.diagnostics["pnp_failure_reason"] = pnp.reason
        if not pnp.accepted:
            frame.state, frame.failure_reason = TrackingState.LOST, pnp.reason
            result.state, result.failure_reason = TrackingState.LOST, pnp.reason
            result.diagnostics["last_successful_frame"] = None if self.last_successful_pose is None else self.last_successful_pose["frame_id"]
            self.loss_events.append({"frame_id": frame.frame_id, "segment_id": sparse_map.segment_id,
                                     "reason": pnp.reason, "last_successful_frame": result.diagnostics["last_successful_frame"]})
            self._transition(TrackingState.LOST)
            self.active_map = None
            self.anchor = None
            self.anchor_image = None
            self.recovery_start = frame.frame_id
            self.recovery_processed_frames = 0
            self.last_lost_segment_id = sparse_map.segment_id
            return
        started = perf_counter()
        frame.R_cw, frame.t_cw = pnp.R_cw, pnp.t_cw
        frame.state = TrackingState.TRACKING
        sparse_map.frames[frame.frame_id] = frame
        for idx in pnp.inlier_indices:
            match = matches.matches[int(idx)]
            point_id = point_ids[match.queryIdx]
            sparse_map.add_observation(point_id, frame, match.trainIdx)
            frame.pnp_inlier_features.add(match.trainIdx)
            if self.mapping.enabled and self.mapping.update_descriptors:
                sparse_map.points[point_id].update_descriptor(
                    frame.features.descriptors[match.trainIdx], self.mapping.descriptor_evidence_limit)
        if not self.mapping.enabled:
            sparse_map.validate()
        result.timing_ms["map_update"] = (perf_counter() - started) * 1000.0
        result.R_cw, result.t_cw = pnp.R_cw.copy(), pnp.t_cw.copy()
        result.diagnostics["map_points"] = len(sparse_map.points)
        if should_insert_keyframe(sparse_map, frame, self.mapping):
            previous = sparse_map.keyframes[max(sparse_map.keyframes)].frame
            started = perf_counter()
            insert_keyframe(sparse_map, frame, self.mapping.window_size)
            self.keyframe_events.append({"segment_id": sparse_map.segment_id,
                                         "frame_id": frame.frame_id, "reason": "coverage_or_baseline",
                                         "active_window": list(sparse_map.active_keyframe_ids)})
            result.diagnostics["keyframe_inserted"] = True
            prior_ids = set(sparse_map.points)
            prior_previous = previous.feature_to_point.copy()
            prior_current = frame.feature_to_point.copy()
            try:
                mapping_report = triangulate_new_points(sparse_map, previous, frame, self.camera,
                                                        self.matcher, self.mapping, self._next_point_id)
            except (cv2.error, ValueError, FloatingPointError) as error:
                for point_id in set(sparse_map.points) - prior_ids:
                    del sparse_map.points[point_id]
                previous.feature_to_point = prior_previous
                frame.feature_to_point = prior_current
                sparse_map.validate()
                mapping_report = {"status": "rejected", "reason": type(error).__name__,
                                  "new_points": 0, "runtime_ms": (perf_counter() - started) * 1000}
            self._next_point_id += mapping_report["new_points"]
            self.mapping_events.append({"segment_id": sparse_map.segment_id,
                                        "frame_id": frame.frame_id, **mapping_report})
            result.diagnostics["mapping"] = mapping_report
            result.timing_ms["local_mapping"] = (perf_counter() - started) * 1000
            if self.bundle.enabled and (len(sparse_map.keyframes) - 2) % self.bundle.every_n_keyframes == 0:
                ba = optimize_local_map(sparse_map, self.camera, self.bundle)
                self.ba_events.append({"segment_id": sparse_map.segment_id,
                                       "frame_id": frame.frame_id, **ba})
                result.diagnostics["bundle_adjustment"] = ba
                result.timing_ms["bundle_adjustment"] = ba["runtime_ms"]
                if ba["applied"]:
                    result.R_cw, result.t_cw = frame.R_cw.copy(), frame.t_cw.copy()
            culled = cull_inconsistent_points(sparse_map, self.camera, self.mapping,
                                              frame.frame_id, set(frame.feature_to_point.values()))
            result.diagnostics["culled_points"] = culled
            result.diagnostics["map_points"] = len(sparse_map.points)
            result.diagnostics["active_map_points"] = sum(p.valid for p in sparse_map.points.values())
        if self.mapping.enabled:
            result.diagnostics["keyframes"] = len(sparse_map.keyframes)
        self.last_successful_pose = result.as_dict()
