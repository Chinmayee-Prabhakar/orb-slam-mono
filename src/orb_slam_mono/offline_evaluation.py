"""Synthetic truth-based diagnostics; never imported by the online initializer."""

from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

from .geometry import InitializationResult, parallax_degrees
from .synthetic3d import SyntheticScene


def _quantiles(values: np.ndarray) -> Optional[Dict[str, float]]:
    if not len(values):
        return None
    values = np.asarray(values, dtype=float)
    return {"p10": float(np.percentile(values, 10)),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90))}


def _angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    cosine = np.clip(float(np.dot(first, second) /
                           (np.linalg.norm(first) * np.linalg.norm(second))), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _true_pose_reference(scene: SyntheticScene) -> Dict[str, Any]:
    """Offline noisy-observation triangulation with the known generating pose."""
    indices = np.flatnonzero(scene.true_match_mask)
    if not len(indices) or np.linalg.norm(scene.t_21_metric_truth) <= 0:
        return {"median_3d_error": None, "p90_3d_error": None}
    unit_t = scene.t_21_metric_truth / np.linalg.norm(scene.t_21_metric_truth)
    p1 = scene.camera.matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
    p2 = scene.camera.matrix @ np.column_stack((scene.R_21_truth, unit_t))
    homogeneous = cv2.triangulatePoints(p1, p2, scene.points1_px[indices].T,
                                        scene.points2_px[indices].T)
    valid = np.all(np.isfinite(homogeneous), axis=0) & (np.abs(homogeneous[3]) > 1e-12)
    reference = (homogeneous[:3, valid] / homogeneous[3, valid]).T
    reference *= np.linalg.norm(scene.t_21_metric_truth)
    errors = np.linalg.norm(reference - scene.points_world[indices[valid]], axis=1)
    return {"median_3d_error": float(np.median(errors)) if len(errors) else None,
            "p90_3d_error": float(np.percentile(errors, 90)) if len(errors) else None}


def evaluate_synthetic(result: InitializationResult, scene: SyntheticScene,
                       point_identity_known: bool = True) -> Dict[str, Any]:
    """Evaluate an estimate after initialization; truth never affects acceptance."""
    report: Dict[str, Any] = {
        "rotation_error_deg": None, "translation_direction_error_deg": None,
        "pose_evaluation_source": None,
        "false_correspondences_among_map_points": None,
        "median_3d_error_after_offline_scale_alignment": None,
        "scale_only_alignment": None,
        "true_pose_triangulation_reference": (_true_pose_reference(scene)
                                              if point_identity_known else None),
        "point_identity_known": point_identity_known,
    }
    rotation = result.R_21 if result.R_21 is not None else result.debug_provisional_R_21
    translation = result.t_21 if result.t_21 is not None else result.debug_provisional_t_21
    if rotation is not None:
        report["pose_evaluation_source"] = ("accepted" if result.status == "accepted" else
                                             "debug_provisional_invalid_for_mapping")
        delta = rotation @ scene.R_21_truth.T
        report["rotation_error_deg"] = float(np.degrees(np.arccos(
            np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))))
    if translation is not None and np.linalg.norm(scene.t_21_metric_truth) > 0:
        report["translation_direction_error_deg"] = _angle_degrees(
            translation, scene.t_21_metric_truth)
    if result.status != "accepted" or not len(result.points_xyz) or not point_identity_known:
        return report
    indices = np.flatnonzero(result.accepted_mask)
    if len(indices) != len(result.points_xyz) or len(result.accepted_mask) != len(scene.points_world):
        raise ValueError("Accepted geometry and synthetic source identities disagree")
    report["false_correspondences_among_map_points"] = int((~scene.true_match_mask[indices]).sum())
    valid = scene.true_match_mask[indices]
    estimated = result.points_xyz[valid]
    truth = scene.points_world[indices[valid]]
    if not len(estimated):
        return report
    denominator = float(np.sum(estimated * estimated))
    if denominator <= 0 or not np.isfinite(denominator):
        return report
    # Origin-fixed scalar fit, not Umeyama Sim(3). Used only for this report.
    scale = float(np.sum(estimated * truth) / denominator)
    aligned = scale * estimated
    errors = np.linalg.norm(aligned - truth, axis=1)
    depth_midpoint = float(np.median(truth[:, 2]))
    near, far = truth[:, 2] <= depth_midpoint, truth[:, 2] > depth_midpoint
    ray_angles = parallax_degrees(scene.points1_px[indices[valid]],
                                  scene.points2_px[indices[valid]],
                                  scene.camera, scene.R_21_truth)
    angle_midpoint = float(np.median(ray_angles))
    low_angle, high_angle = ray_angles <= angle_midpoint, ray_angles > angle_midpoint
    report["scale_only_alignment"] = {
        "definition": "origin-fixed scalar fit; not Umeyama Sim(3); evaluation only",
        "fitted_scale_scene_units_per_estimated_unit": scale,
        "true_translation_baseline_scene_units_for_reference": float(np.linalg.norm(scene.t_21_metric_truth)),
        "median_3d_error_scene_units": float(np.median(errors)),
        "p90_3d_error_scene_units": float(np.percentile(errors, 90)),
        "estimated_depth_unit_baselines": _quantiles(estimated[:, 2]),
        "aligned_estimated_depth_scene_units": _quantiles(aligned[:, 2]),
        "true_depth_scene_units": _quantiles(truth[:, 2]),
        "error_near_depth_median": float(np.median(errors[near])) if near.any() else None,
        "error_far_depth_median": float(np.median(errors[far])) if far.any() else None,
        "error_low_parallax_median": float(np.median(errors[low_angle])) if low_angle.any() else None,
        "error_high_parallax_median": float(np.median(errors[high_angle])) if high_angle.any() else None,
    }
    report["median_3d_error_after_offline_scale_alignment"] = float(np.median(errors))
    return report
