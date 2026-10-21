"""Calibrated two-view monocular initialization in an arbitrary unit-baseline frame.

Convention: T_cw maps world to camera, X_c = R_cw X_w + t_cw. Camera 1 is
the world origin; camera 2 has (R_21, t_21), with ||t_21|| = 1. Input and
reprojection coordinates are pixels, so P1/P2 include K exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .camera import CalibratedCamera


@dataclass(frozen=True)
class GeometryConfig:
    ransac_confidence: float = 0.999
    ransac_threshold_px: float = 1.5
    min_correspondences: int = 8
    min_essential_inliers: int = 40
    min_essential_inlier_ratio: float = 0.4
    min_pose_inliers: int = 30
    min_pose_inlier_fraction: float = 0.9
    min_triangulated_points: int = 25
    min_rotation_only_residual_px: float = 1.5
    min_point_parallax_deg: float = 0.5
    min_median_parallax_deg: float = 1.0
    min_homography_coverage: float = 0.9
    min_homography_advantage: float = 1.05
    max_dominant_homography_parallax_deg: float = 2.0
    max_reprojection_error_px: float = 3.0
    max_depth_in_baselines: float = 200.0
    pose_stability_trials: int = 8
    min_pose_stability_candidates: int = 4
    min_pose_stability_accepted_trials: int = 4
    max_median_rotation_disagreement_deg: float = 10.0
    pose_stability_seed: int = 202

    def __post_init__(self) -> None:
        thresholds = [self.ransac_confidence, self.ransac_threshold_px,
                      self.min_essential_inlier_ratio, self.min_pose_inlier_fraction,
                      self.min_rotation_only_residual_px, self.min_point_parallax_deg,
                      self.min_median_parallax_deg, self.min_homography_coverage,
                      self.min_homography_advantage, self.max_dominant_homography_parallax_deg,
                      self.max_reprojection_error_px, self.max_depth_in_baselines,
                      self.max_median_rotation_disagreement_deg]
        if not np.all(np.isfinite(thresholds)):
            raise ValueError("Geometry thresholds must be finite")
        if (not 0 < self.ransac_confidence < 1
                or not 0 < self.min_essential_inlier_ratio <= 1
                or not 0 < self.min_pose_inlier_fraction <= 1):
            raise ValueError("Confidence and inlier ratio must be in (0, 1]")
        if min(self.min_correspondences, self.min_essential_inliers, self.min_pose_inliers,
               self.min_triangulated_points) < 1:
            raise ValueError("Minimum correspondence and point counts must be positive")
        if min(self.ransac_threshold_px, self.min_rotation_only_residual_px,
               self.min_point_parallax_deg,
               self.min_median_parallax_deg, self.max_reprojection_error_px,
               self.max_depth_in_baselines, self.max_dominant_homography_parallax_deg,
               self.min_homography_advantage,
               self.max_median_rotation_disagreement_deg) <= 0 or not 0 < self.min_homography_coverage <= 1:
            raise ValueError("Geometry thresholds must be positive")
        if (not isinstance(self.pose_stability_trials, int) or self.pose_stability_trials < 0
                or not isinstance(self.min_pose_stability_candidates, int)
                or self.min_pose_stability_candidates < 1
                or (self.pose_stability_trials > 0
                    and self.min_pose_stability_candidates > self.pose_stability_trials)
                or not isinstance(self.min_pose_stability_accepted_trials, int)
                or isinstance(self.min_pose_stability_accepted_trials, bool)
                or self.min_pose_stability_accepted_trials < 1
                or (self.pose_stability_trials > 0
                    and self.min_pose_stability_accepted_trials > self.pose_stability_trials)
                or not isinstance(self.pose_stability_seed, int)
                or self.pose_stability_seed < 0):
            raise ValueError("Invalid pose-stability trial configuration")


@dataclass
class TriangulationFilterResult:
    points_xyz: np.ndarray
    accepted_source_indices: np.ndarray
    accepted_local_mask: np.ndarray
    parallax_deg: np.ndarray
    reprojection_error_first_px: np.ndarray
    reprojection_error_second_px: np.ndarray
    counts: Dict[str, int]
    malformed: bool = False


@dataclass
class InitializationResult:
    status: str = "rejected"
    failure_reason: Optional[str] = None
    R_21: Optional[np.ndarray] = None
    t_21: Optional[np.ndarray] = None
    debug_provisional_R_21: Optional[np.ndarray] = None
    debug_provisional_t_21: Optional[np.ndarray] = None
    essential_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    pose_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    accepted_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    points_xyz: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    timing_ms: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "initialization_status": self.status,
            "failure_reason": self.failure_reason,
            "scale": "arbitrary; unit translation baseline",
            "pose_convention": "T_cw: X_c = R_cw X_w + t_cw; camera 1 is world origin",
            "R_21": None if self.R_21 is None else self.R_21.tolist(),
            "t_21_unit": None if self.t_21 is None else self.t_21.tolist(),
            "C2_in_camera1": None if self.R_21 is None or self.t_21 is None
            else (-self.R_21.T @ self.t_21).tolist(),
            "valid_for_mapping": self.status == "accepted",
            "debug_provisional_pose_invalid_for_mapping": None if self.status == "accepted" else {
                "R_21": None if self.debug_provisional_R_21 is None else self.debug_provisional_R_21.tolist(),
                "t_21_unit": None if self.debug_provisional_t_21 is None else self.debug_provisional_t_21.tolist(),
            },
            "diagnostics": self.diagnostics,
            "timing_ms": self.timing_ms,
            "essential_inlier_indices": np.flatnonzero(self.essential_mask).tolist(),
            "pose_inlier_indices": np.flatnonzero(self.pose_mask).tolist(),
            "accepted_point_indices": np.flatnonzero(self.accepted_mask).tolist(),
        }


def _essential_candidates(matrix: np.ndarray) -> List[np.ndarray]:
    """Handle OpenCV's single, vertically stacked, or horizontally stacked E."""
    e = np.asarray(matrix, dtype=np.float64)
    if e.ndim != 2:
        return []
    if e.shape == (3, 3):
        candidates = [e]
    elif e.shape[1] == 3 and e.shape[0] % 3 == 0:
        candidates = [e[i:i + 3] for i in range(0, e.shape[0], 3)]
    elif e.shape[0] == 3 and e.shape[1] % 3 == 0:
        candidates = [e[:, i:i + 3] for i in range(0, e.shape[1], 3)]
    else:
        return []
    return [candidate for candidate in candidates
            if np.all(np.isfinite(candidate)) and np.linalg.norm(candidate) > 1e-12]


def _valid_rotation(rotation: np.ndarray) -> bool:
    return (rotation.shape == (3, 3) and np.all(np.isfinite(rotation))
            and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
            and abs(float(np.linalg.det(rotation)) - 1.0) < 1e-3)


def _bearings(points_px: np.ndarray, camera: CalibratedCamera) -> np.ndarray:
    rays = np.column_stack(((points_px[:, 0] - camera.cx) / camera.fx,
                            (points_px[:, 1] - camera.cy) / camera.fy,
                            np.ones(len(points_px))))
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def parallax_degrees(points1_px: np.ndarray, points2_px: np.ndarray,
                     camera: CalibratedCamera, R_21: np.ndarray) -> np.ndarray:
    """Angle between corresponding 3D viewing rays in camera-1 coordinates."""
    rays1 = _bearings(points1_px, camera)
    rays2_in_first = (R_21.T @ _bearings(points2_px, camera).T).T
    cosine = np.clip(np.sum(rays1 * rays2_in_first, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def rotation_only_median_residual_px(points1_px: np.ndarray, points2_px: np.ndarray,
                                     camera: CalibratedCamera) -> float:
    """Median pixel residual after fitting one proper rotation to bearing pairs.

    A near-zero residual means translation is not discernible at this image
    precision. This fit is a quality diagnostic, never an estimated SLAM pose.
    """
    first = _bearings(points1_px, camera)
    second = _bearings(points2_px, camera)
    u, _, vt = np.linalg.svd(second.T @ first)
    rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    rotated = (rotation @ first.T).T
    projected = (camera.matrix @ rotated.T).T
    valid = np.isfinite(projected).all(axis=1) & (projected[:, 2] > 1e-12)
    if not valid.any():
        return float("nan")
    pixels = projected[valid, :2] / projected[valid, 2:3]
    return float(np.median(np.linalg.norm(pixels - points2_px[valid], axis=1)))


def filter_triangulated_points(
    homogeneous: np.ndarray,
    points1_px: np.ndarray,
    points2_px: np.ndarray,
    camera: CalibratedCamera,
    R_21: np.ndarray,
    t_21: np.ndarray,
    source_indices: np.ndarray,
    config: GeometryConfig,
) -> TriangulationFilterResult:
    """Apply sequential numerical, depth, parallax, and reprojection gates.

    source_indices are positions in the original input correspondence array.
    This function is separately testable, including malformed homogeneous data.
    """
    count = len(points1_px)
    stage = {"triangulation_candidates": count, "homogeneous_valid": 0,
             "positive_depth": 0, "depth_range": 0, "parallax_pass": 0,
             "reprojection_pass": 0, "accepted_points": 0}
    empty = TriangulationFilterResult(np.empty((0, 3)), np.empty(0, dtype=int),
                                      np.zeros(count, dtype=bool), np.empty(0),
                                      np.empty(0), np.empty(0), stage, malformed=True)
    h = np.asarray(homogeneous)
    if (h.shape != (4, count) or points2_px.shape != (count, 2)
            or np.asarray(source_indices).shape != (count,)):
        return empty
    if count == 0:
        empty.malformed = False
        return empty
    try:
        h = h.astype(np.float64, copy=False)
    except (TypeError, ValueError):
        return empty
    valid_h = np.all(np.isfinite(h), axis=0) & (np.abs(h[3]) > 1e-12)
    xyz = np.full((count, 3), np.nan)
    xyz[valid_h] = (h[:3, valid_h] / h[3, valid_h]).T
    valid_h &= np.all(np.isfinite(xyz), axis=1)
    stage["homogeneous_valid"] = int(valid_h.sum())

    second_xyz = (R_21 @ xyz.T).T + t_21.reshape(1, 3)
    positive = valid_h & (xyz[:, 2] > 1e-8) & (second_xyz[:, 2] > 1e-8)
    stage["positive_depth"] = int(positive.sum())
    depth_range = positive & (xyz[:, 2] <= config.max_depth_in_baselines) & (
        second_xyz[:, 2] <= config.max_depth_in_baselines)
    stage["depth_range"] = int(depth_range.sum())

    angles = parallax_degrees(points1_px, points2_px, camera, R_21)
    parallax = depth_range & np.isfinite(angles) & (angles >= config.min_point_parallax_deg)
    stage["parallax_pass"] = int(parallax.sum())

    err1 = np.full(count, np.nan)
    err2 = np.full(count, np.nan)
    if np.any(parallax):
        selected = np.flatnonzero(parallax)
        projected1 = (camera.matrix @ xyz[selected].T).T
        projected2 = (camera.matrix @ second_xyz[selected].T).T
        projected1 = projected1[:, :2] / projected1[:, 2:3]
        projected2 = projected2[:, :2] / projected2[:, 2:3]
        err1[selected] = np.linalg.norm(projected1 - points1_px[selected], axis=1)
        err2[selected] = np.linalg.norm(projected2 - points2_px[selected], axis=1)
    accepted = parallax & np.isfinite(err1) & np.isfinite(err2) & (
        err1 <= config.max_reprojection_error_px) & (err2 <= config.max_reprojection_error_px)
    stage["reprojection_pass"] = int(accepted.sum())
    stage["accepted_points"] = int(accepted.sum())
    return TriangulationFilterResult(
        xyz[accepted], np.asarray(source_indices)[accepted], accepted,
        angles[accepted], err1[accepted], err2[accepted], stage)


def _initialize_two_view_once(points1_px: np.ndarray, points2_px: np.ndarray,
                              camera: CalibratedCamera,
                              config: Optional[GeometryConfig] = None) -> InitializationResult:
    """Estimate a unit-baseline relative pose and filtered sparse 3D points.

    Input arrays are matching pixel coordinates in the same order. Every mask
    is indexed against these original arrays; no truth pose/depth is accepted.
    Ordinary bad data and degenerate motion yield a structured rejection.
    """
    started = perf_counter()
    cfg = config or GeometryConfig()
    result = InitializationResult()
    counts: Dict[str, Any] = {
        "input_correspondences": 0, "essential_inliers": None,
        "essential_inlier_ratio": None, "pose_inliers": None,
        "pose_inlier_fraction": None, "rotation_only_median_residual_px": None,
        "homography_inliers": None, "homography_dominant": None,
        "triangulation_candidates": None, "homogeneous_valid": None,
        "positive_depth": None, "depth_range": None, "parallax_pass": None,
        "reprojection_pass": None, "accepted_points": None,
        "median_candidate_parallax_deg": None, "candidate_parallax_p10_deg": None,
        "candidate_parallax_p90_deg": None, "median_accepted_parallax_deg": None,
        "median_reprojection_error_first_px": None,
        "median_reprojection_error_second_px": None,
        "pose_stability_valid_candidates": None,
        "pose_stability_provisional_candidates": None,
        "pose_stability_accepted_trials": None,
        "pose_stability_median_rotation_disagreement_deg": None,
    }
    result.diagnostics = counts

    def finish(reason: Optional[str]) -> InitializationResult:
        result.failure_reason = reason
        result.status = "accepted" if reason is None else "rejected"
        if reason is not None:
            # Diagnostic masks/counts remain, but public pose and map fields
            # cannot be consumed as an accepted initialization.
            result.debug_provisional_R_21 = result.R_21
            result.debug_provisional_t_21 = result.t_21
            result.R_21 = None
            result.t_21 = None
            result.points_xyz = np.empty((0, 3), dtype=np.float64)
            result.accepted_mask = np.zeros_like(result.accepted_mask)
        counts["usable_map_points"] = len(result.points_xyz)
        result.timing_ms["total"] = round((perf_counter() - started) * 1000, 3)
        return result

    if not isinstance(camera, CalibratedCamera):
        return finish("invalid_calibration")
    try:
        first = np.asarray(points1_px, dtype=np.float64)
        second = np.asarray(points2_px, dtype=np.float64)
    except (TypeError, ValueError):
        return finish("invalid_input")
    if first.ndim != 2 or first.shape[1:] != (2,) or second.shape != first.shape:
        return finish("invalid_input")
    n = len(first)
    counts["input_correspondences"] = n
    result.essential_mask = np.zeros(n, dtype=bool)
    result.pose_mask = np.zeros(n, dtype=bool)
    result.accepted_mask = np.zeros(n, dtype=bool)
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        return finish("invalid_input")
    if n < cfg.min_correspondences:
        return finish("insufficient_correspondences")

    essential_started = perf_counter()
    try:
        essential, raw_mask = cv2.findEssentialMat(
            first, second, camera.matrix, method=cv2.RANSAC,
            prob=cfg.ransac_confidence, threshold=cfg.ransac_threshold_px)
    except cv2.error:
        return finish("essential_estimation_failed")
    result.timing_ms["essential"] = round((perf_counter() - essential_started) * 1000, 3)
    if essential is None or raw_mask is None:
        return finish("essential_estimation_failed")
    candidates = _essential_candidates(essential)
    if not candidates or np.asarray(raw_mask).size != n:
        return finish("essential_estimation_failed")
    result.essential_mask = np.asarray(raw_mask).reshape(-1) != 0
    essential_count = int(result.essential_mask.sum())
    counts["essential_inliers"] = essential_count
    counts["essential_inlier_ratio"] = round(essential_count / n, 4)
    if (essential_count < cfg.min_essential_inliers
            or essential_count / n < cfg.min_essential_inlier_ratio):
        return finish("insufficient_essential_inliers")
    rotation_only_error = rotation_only_median_residual_px(
        first[result.essential_mask], second[result.essential_mask], camera)
    counts["rotation_only_median_residual_px"] = rotation_only_error
    if not np.isfinite(rotation_only_error):
        return finish("invalid_geometry")

    # A homography is only a degeneracy diagnostic. Planar translation can also
    # produce a strong homography, so it rejects only alongside low parallax.
    try:
        _, homography_mask = cv2.findHomography(
            first, second, cv2.RANSAC, cfg.ransac_threshold_px)
    except cv2.error:
        homography_mask = None
    if homography_mask is not None and np.asarray(homography_mask).size == n:
        homography_count = int(np.count_nonzero(homography_mask))
        counts["homography_inliers"] = homography_count
        counts["homography_dominant"] = bool(
            homography_count / n >= cfg.min_homography_coverage
            and homography_count >= cfg.min_homography_advantage * essential_count)

    pose_started = perf_counter()
    best = None
    for candidate in candidates:
        try:
            _, rotation, translation, pose_mask_raw = cv2.recoverPose(
                candidate, first, second, camera.matrix,
                mask=result.essential_mask.astype(np.uint8).reshape(-1, 1) * 255)
        except cv2.error:
            continue
        if pose_mask_raw is None or np.asarray(pose_mask_raw).size != n:
            continue
        translation = np.asarray(translation, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(translation))
        if not _valid_rotation(rotation) or translation.shape != (3,) or not np.isfinite(norm) or norm < 1e-12:
            continue
        pose_mask = (np.asarray(pose_mask_raw).reshape(-1) != 0) & result.essential_mask
        score = int(pose_mask.sum())
        if best is None or score > best[0]:
            best = (score, rotation, translation / norm, pose_mask)
    result.timing_ms["pose"] = round((perf_counter() - pose_started) * 1000, 3)
    if best is None:
        return finish("pose_recovery_failed")
    pose_count, result.R_21, result.t_21, result.pose_mask = best
    counts["pose_inliers"] = pose_count
    counts["pose_inlier_fraction"] = pose_count / essential_count
    if pose_count < cfg.min_pose_inliers:
        return finish("insufficient_pose_inliers")
    if counts["pose_inlier_fraction"] < cfg.min_pose_inlier_fraction:
        return finish("low_cheirality_fraction")
    if rotation_only_error < cfg.min_rotation_only_residual_px:
        return finish("rotation_only_explains_matches")

    source_indices = np.flatnonzero(result.pose_mask)
    counts["triangulation_candidates"] = len(source_indices)
    candidate_angles = parallax_degrees(first[source_indices], second[source_indices],
                                        camera, result.R_21)
    if not np.all(np.isfinite(candidate_angles)):
        return finish("invalid_geometry")
    counts["median_candidate_parallax_deg"] = float(np.median(candidate_angles))
    counts["candidate_parallax_p10_deg"] = float(np.percentile(candidate_angles, 10))
    counts["candidate_parallax_p90_deg"] = float(np.percentile(candidate_angles, 90))
    if (counts["homography_dominant"] and counts["median_candidate_parallax_deg"]
            < cfg.max_dominant_homography_parallax_deg):
        return finish("rotation_or_planar_degeneracy")
    if counts["median_candidate_parallax_deg"] < cfg.min_median_parallax_deg:
        return finish("low_parallax")

    triangulation_started = perf_counter()
    p1 = camera.matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
    p2 = camera.matrix @ np.column_stack((result.R_21, result.t_21))
    try:
        homogeneous = cv2.triangulatePoints(
            p1, p2, first[source_indices].T, second[source_indices].T)
    except cv2.error:
        return finish("invalid_triangulation")
    filtered = filter_triangulated_points(
        homogeneous, first[source_indices], second[source_indices], camera,
        result.R_21, result.t_21, source_indices, cfg)
    result.timing_ms["triangulation"] = round((perf_counter() - triangulation_started) * 1000, 3)
    counts.update(filtered.counts)
    if filtered.malformed or filtered.counts["homogeneous_valid"] == 0:
        return finish("invalid_triangulation")
    result.points_xyz = filtered.points_xyz
    result.accepted_mask[filtered.accepted_source_indices] = True
    if len(filtered.points_xyz):
        counts["median_accepted_parallax_deg"] = float(np.median(filtered.parallax_deg))
        counts["median_reprojection_error_first_px"] = float(np.median(filtered.reprojection_error_first_px))
        counts["median_reprojection_error_second_px"] = float(np.median(filtered.reprojection_error_second_px))
    if filtered.counts["triangulation_candidates"] < cfg.min_triangulated_points:
        return finish("insufficient_triangulated_points")
    if filtered.counts["positive_depth"] < cfg.min_triangulated_points:
        return finish("insufficient_positive_depth")
    if filtered.counts["depth_range"] < cfg.min_triangulated_points:
        return finish("unstable_geometry")
    if filtered.counts["parallax_pass"] < cfg.min_triangulated_points:
        return finish("low_parallax")
    if filtered.counts["accepted_points"] < cfg.min_triangulated_points:
        return finish("excessive_reprojection_error")
    if counts["median_accepted_parallax_deg"] < cfg.min_median_parallax_deg:
        return finish("low_parallax")
    return finish(None)


def initialize_two_view(points1_px: np.ndarray, points2_px: np.ndarray,
                        camera: CalibratedCamera,
                        config: Optional[GeometryConfig] = None) -> InitializationResult:
    """Initialize once, then reject poses unstable to match-order perturbations.

    The additional trials see the same 2D matches and calibration in permuted
    order. They do not receive truth, alter the chosen pose, or set metric scale.
    Rejected trials can still provide a provisional rotation for the angular
    diagnostic, but only fully accepted trials satisfy the acceptance minimum.
    """
    cfg = config or GeometryConfig()
    result = _initialize_two_view_once(points1_px, points2_px, camera, cfg)
    if result.status != "accepted" or cfg.pose_stability_trials == 0:
        return result
    started = perf_counter()
    first = np.asarray(points1_px, dtype=np.float64)
    second = np.asarray(points2_px, dtype=np.float64)
    rng = np.random.default_rng(cfg.pose_stability_seed)
    disagreements = []
    accepted_trials = 0
    provisional_candidates = 0
    for _ in range(cfg.pose_stability_trials):
        order = rng.permutation(len(first))
        trial = _initialize_two_view_once(first[order], second[order], camera, cfg)
        if trial.status == "accepted":
            accepted_trials += 1
        rotation = (trial.R_21 if trial.R_21 is not None
                    else trial.debug_provisional_R_21)
        if rotation is not None:
            if trial.status != "accepted":
                provisional_candidates += 1
            delta = rotation @ result.R_21.T
            disagreements.append(float(np.degrees(np.arccos(
                np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)))))
    median_disagreement = None if not disagreements else float(np.median(disagreements))
    counts = result.diagnostics
    counts["pose_stability_valid_candidates"] = len(disagreements)
    counts["pose_stability_provisional_candidates"] = provisional_candidates
    counts["pose_stability_accepted_trials"] = accepted_trials
    counts["pose_stability_median_rotation_disagreement_deg"] = median_disagreement
    result.timing_ms["pose_stability"] = round((perf_counter() - started) * 1000, 3)
    result.timing_ms["total"] = round(result.timing_ms["total"] + result.timing_ms["pose_stability"], 3)
    if (len(disagreements) < cfg.min_pose_stability_candidates
            or median_disagreement > cfg.max_median_rotation_disagreement_deg
            or accepted_trials < cfg.min_pose_stability_accepted_trials):
        result.status = "rejected"
        result.failure_reason = ("insufficient_accepted_pose_stability_trials"
                                 if (len(disagreements) >= cfg.min_pose_stability_candidates
                                     and median_disagreement <= cfg.max_median_rotation_disagreement_deg)
                                 else "unstable_pose_under_resampling")
        result.debug_provisional_R_21 = result.R_21
        result.debug_provisional_t_21 = result.t_21
        result.R_21 = None
        result.t_21 = None
        result.points_xyz = np.empty((0, 3), dtype=np.float64)
        result.accepted_mask = np.zeros_like(result.accepted_mask)
        counts["usable_map_points"] = 0
    return result
