"""Offline-only evaluation of saved online monocular trajectories against KITTI.

No online estimator imports this module. Ground truth is used only after a run
has been frozen to disk. All input frame identifiers are original KITTI IDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import scipy

from .kitti_pose_evaluation import load_kitti_poses


VARIANTS = ("baseline", "growth", "descriptors", "full")
GAPS = (1, 5, 10)
MIN_ALIGNMENT_POSES = 5
MIN_SECOND_SINGULAR_RATIO = 0.05
MIN_GT_DISPLACEMENT_M = 0.05
MIN_EST_DISPLACEMENT = 1e-8
ROTATION_TOL = 1e-3


@dataclass(frozen=True)
class OnlinePose:
    frame_id: int
    segment_id: int
    R_cw: np.ndarray
    t_cw: np.ndarray
    center: np.ndarray

    @property
    def R_wc(self) -> np.ndarray:
        return self.R_cw.T


@dataclass(frozen=True)
class Similarity:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    estimated_singular_values: Tuple[float, float, float]
    truth_singular_values: Tuple[float, float, float]

    def transform(self, centers: np.ndarray) -> np.ndarray:
        return self.scale * (np.asarray(centers) @ self.rotation.T) + self.translation


def _rotation(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite 3x3 rotation")
    if (np.max(np.abs(result @ result.T - np.eye(3))) > ROTATION_TOL
            or abs(np.linalg.det(result) - 1.0) > ROTATION_TOL):
        raise ValueError(f"{label} must be proper and orthogonal")
    return result


def camera_center(R_cw: np.ndarray, t_cw: np.ndarray) -> np.ndarray:
    rotation = _rotation(R_cw, "R_cw")
    translation = np.asarray(t_cw, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("t_cw must be a finite 3-vector")
    return -rotation.T @ translation


def inverse_t_cw(R_cw: np.ndarray, t_cw: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = _rotation(R_cw, "R_cw").T
    result[:3, 3] = camera_center(R_cw, t_cw)
    return result


def rotation_error_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    _rotation(R_a, "R_a")
    _rotation(R_b, "R_b")
    relative = R_a @ R_b.T
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))


def direction_error_deg(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= MIN_EST_DISPLACEMENT or nb < MIN_GT_DISPLACEMENT_M:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1))))


def validate_source_index(sequence: Path, gt_count: int, start: int, end: int) -> Dict[str, Any]:
    """Validate the positional loader's exact source-ID/timestamp contract.

    This fixed KITTI prefix must start at zero. A future restarted subsequence
    can use a full original-ID sequence directory and select start/end here.
    """
    if start < 0 or end < start:
        raise ValueError("Invalid inclusive source frame range")
    images = sorted((sequence / "image_0").glob("*.png"))
    if not images:
        raise ValueError("No source PNG images")
    ids: List[int] = []
    for path in images:
        if len(path.stem) != 6 or not path.stem.isdecimal():
            raise ValueError(f"Ambiguous KITTI image identifier: {path.name}")
        ids.append(int(path.stem))
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source image identifier")
    if ids != list(range(len(ids))):
        raise ValueError("Image names must be the contiguous original zero-based KITTI IDs")
    if end >= len(ids) or end >= gt_count:
        raise ValueError("Requested source frame exceeds images or GT rows")
    times_path = sequence / "times.txt"
    lines = times_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(ids):
        raise ValueError("Timestamp count does not match source image count")
    try:
        times = np.asarray([float(line) for line in lines], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("Malformed timestamp") from exc
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing")
    if abs(float(times[0])) > 1e-9:
        raise ValueError("KITTI source timestamp zero must correspond to image 000000")
    return {"image_count": len(ids), "first_source_id": ids[0],
            "last_source_id": ids[-1], "requested_range": [start, end],
            "first_timestamp_s": float(times[start]), "last_timestamp_s": float(times[end]),
            "timestamp_rows": len(times), "gt_rows": gt_count,
            "mapping": "image stem == positional loader index == times row == GT row"}


def load_online_rows(path: Path, start: int, end: int) -> Tuple[List[Dict[str, Any]], Dict[int, OnlinePose]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = list(range(start, end + 1))
    if [row.get("frame_id") for row in rows] != expected:
        raise ValueError(f"{path}: online frame IDs must match exact source range")
    poses: Dict[int, OnlinePose] = {}
    for row in rows:
        frame = row["frame_id"]
        values = (row.get("T_cw_R"), row.get("T_cw_t"), row.get("camera_center_world"))
        if not row.get("pose_valid"):
            if any(value is not None for value in values):
                raise ValueError(f"Frame {frame}: unestimated pose has pose fields")
            continue
        if row.get("state") != "tracking" or row.get("segment_id") is None or any(v is None for v in values):
            raise ValueError(f"Frame {frame}: inconsistent online valid pose")
        R = _rotation(values[0], f"frame {frame} R_cw")
        t = np.asarray(values[1], dtype=np.float64)
        c = np.asarray(values[2], dtype=np.float64)
        if t.shape != (3,) or c.shape != (3,) or not np.isfinite(t).all() or not np.isfinite(c).all():
            raise ValueError(f"Frame {frame}: malformed pose vectors")
        if not np.allclose(c, camera_center(R, t), atol=1e-6):
            raise ValueError(f"Frame {frame}: exported camera center disagrees with T_cw")
        seg = row["segment_id"]
        if not isinstance(seg, int) or seg < 0:
            raise ValueError(f"Frame {frame}: invalid segment ID")
        poses[frame] = OnlinePose(frame, seg, R, t, c)
    return rows, poses


def _conditioning(centers: np.ndarray) -> Tuple[Optional[str], Tuple[Optional[float], Optional[float], Optional[float]]]:
    if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
        return "nonfinite_or_malformed_centers", (None, None, None)
    if not len(centers):
        return "fewer_than_five_online_poses", (None, None, None)
    singular = np.linalg.svd(centers - centers.mean(axis=0), compute_uv=False)
    values = tuple(float(x) for x in singular) + (0.0,) * (3 - len(singular))
    if len(centers) < MIN_ALIGNMENT_POSES:
        return "fewer_than_five_online_poses", values
    extent = max(1.0, float(np.max(np.linalg.norm(centers, axis=1))))
    if singular[0] <= 1e-9 * extent:
        return "repeated_or_negligible_centers", values
    if singular[1] / singular[0] < MIN_SECOND_SINGULAR_RATIO:
        return "near_collinear_motion", values
    return None, values


def fit_similarity(estimated: np.ndarray, truth: np.ndarray) -> Tuple[Optional[Similarity], Optional[str]]:
    """Least-squares proper Umeyama fit, estimated world centers to GT meters."""
    X, Y = np.asarray(estimated, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if X.shape != Y.shape:
        return None, "mismatched_center_shapes"
    reason_x, sx = _conditioning(X)
    reason_y, sy = _conditioning(Y)
    if reason_x or reason_y:
        return None, f"estimated:{reason_x}" if reason_x else f"truth:{reason_y}"
    mx, my = X.mean(axis=0), Y.mean(axis=0)
    xc, yc = X - mx, Y - my
    U, s, Vt = np.linalg.svd((yc.T @ xc) / len(X))
    correction = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    Q = U @ correction @ Vt
    variance = float(np.mean(np.sum(xc * xc, axis=1)))
    scale = float(np.sum(s * np.diag(correction)) / variance)
    if not np.isfinite(scale) or scale <= 0 or np.linalg.det(Q) < 1 - ROTATION_TOL:
        return None, "nonpositive_scale_or_improper_rotation"
    b = my - scale * (Q @ mx)
    return Similarity(scale, Q, b, sx, sy), None


def _stats(values: Iterable[float], unit: str) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0, "rmse": None, "median": None, "p90": None, "unit": unit}
    return {"count": int(len(array)), "rmse": float(np.sqrt(np.mean(array ** 2))),
            "median": float(np.median(array)), "p90": float(np.percentile(array, 90)), "unit": unit}


def _pose_error(pose: OnlinePose, gt_wc: np.ndarray, fit: Similarity) -> Dict[str, Any]:
    gt_center, gt_rotation = gt_wc[:3, 3], gt_wc[:3, :3]
    aligned_center = fit.transform(pose.center)
    aligned_rotation = fit.rotation @ pose.R_wc
    return {"frame_id": pose.frame_id, "segment_id": pose.segment_id,
            "ate_m": float(np.linalg.norm(aligned_center - gt_center)),
            "orientation_deg": rotation_error_deg(aligned_rotation, gt_rotation),
            "aligned_center_m": aligned_center.tolist(), "gt_center_m": gt_center.tolist()}


def _relative(a: OnlinePose, b: OnlinePose, gt: np.ndarray,
              scale: Optional[float]) -> Dict[str, Any]:
    """Both translations are expressed in the first camera's axes."""
    ga, gb = gt[a.frame_id], gt[b.frame_id]
    estimated_rotation = b.R_cw @ a.R_cw.T
    truth_rotation = gb[:3, :3].T @ ga[:3, :3]
    est_disp = a.R_cw @ (b.center - a.center)
    gt_disp = ga[:3, :3].T @ (gb[:3, 3] - ga[:3, 3])
    gt_len, est_len = float(np.linalg.norm(gt_disp)), float(np.linalg.norm(est_disp))
    reliable = gt_len >= MIN_GT_DISPLACEMENT_M and est_len > MIN_EST_DISPLACEMENT
    return {"first_frame": a.frame_id, "second_frame": b.frame_id,
            "source_gap": b.frame_id - a.frame_id, "segment_id": a.segment_id,
            "rotation_error_deg": rotation_error_deg(estimated_rotation, truth_rotation),
            "translation_direction_error_deg": direction_error_deg(est_disp, gt_disp),
            "estimated_displacement_units": est_len, "gt_displacement_m": gt_len,
            "scaled_displacement_m": None if scale is None else scale * est_len,
            "displacement_error_m": None if scale is None else abs(scale * est_len - gt_len),
            "relative_translation_vector_error_m": None if scale is None else
            float(np.linalg.norm(scale * est_disp - gt_disp)),
            "scale_ratio": None if scale is None or not reliable else scale * est_len / gt_len,
            "translation_reliable": reliable}


def _segment_groups(poses: Dict[int, OnlinePose]) -> Dict[int, List[OnlinePose]]:
    groups: Dict[int, List[OnlinePose]] = defaultdict(list)
    for pose in poses.values():
        groups[pose.segment_id].append(pose)
    for group in groups.values():
        group.sort(key=lambda pose: pose.frame_id)
    return dict(sorted(groups.items()))


def evaluate_segments(poses: Dict[int, OnlinePose], gt: np.ndarray,
                      anchors: Dict[int, int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    segments, frame_errors, relative_rows = [], [], []
    for segment_id, group in _segment_groups(poses).items():
        ids = [p.frame_id for p in group]
        X = np.asarray([p.center for p in group])
        Y = gt[ids, :3, 3]
        x_condition, x_singular = _conditioning(X)
        y_condition, y_singular = _conditioning(Y)
        fit, reason = fit_similarity(X, Y)
        errors = [] if fit is None else [_pose_error(p, gt[p.frame_id], fit) for p in group]
        frame_errors.extend(errors)
        pair_rows = []
        by_frame = {p.frame_id: p for p in group}
        for first in group:
            for gap in GAPS:
                second = by_frame.get(first.frame_id + gap)
                if second is not None:
                    pair_rows.append(_relative(first, second, gt, None if fit is None else fit.scale))
        relative_rows.extend(pair_rows)
        anchor = anchors.get(segment_id)
        anchor_errors = []
        if anchor is not None:
            # The initial map's anchor camera is the segment world origin.
            anchor_rotation = gt[anchor, :3, :3]
            anchor_errors = [rotation_error_deg(anchor_rotation @ p.R_wc,
                                                gt[p.frame_id, :3, :3]) for p in group]
        prefix = {"status": "unavailable", "reason": "segment_shorter_than_ten_poses_or_ineligible"}
        if fit is not None and len(group) >= 10:
            cutoff = int(math.ceil(len(group) / 2))
            prefix_fit, prefix_reason = fit_similarity(X[:cutoff], Y[:cutoff])
            if prefix_fit is not None:
                suffix_errors = [_pose_error(p, gt[p.frame_id], prefix_fit) for p in group[cutoff:]]
                prefix = {"status": "eligible", "fit_pose_count": cutoff,
                          "suffix_pose_count": len(suffix_errors), "prefix_scale_m_per_unit": prefix_fit.scale,
                          "suffix_ate_m": _stats((e["ate_m"] for e in suffix_errors), "m"),
                          "suffix_orientation_deg": _stats((e["orientation_deg"] for e in suffix_errors), "deg")}
            else:
                prefix = {"status": "unavailable", "reason": prefix_reason}
        by_gap: Dict[str, Any] = {}
        for gap in GAPS:
            rows = [r for r in pair_rows if r["source_gap"] == gap]
            by_gap[str(gap)] = {"rotation_deg": _stats((r["rotation_error_deg"] for r in rows), "deg"),
                                "direction_deg": _stats((r["translation_direction_error_deg"] for r in rows
                                                          if r["translation_direction_error_deg"] is not None), "deg"),
                                "displacement_error_m": _stats((r["displacement_error_m"] for r in rows
                                                                 if r["displacement_error_m"] is not None), "m"),
                                "relative_translation_vector_error_m": _stats(
                                    (r["relative_translation_vector_error_m"] for r in rows
                                     if r["relative_translation_vector_error_m"] is not None), "m"),
                                "scale_ratio": _stats((r["scale_ratio"] for r in rows
                                                       if r["scale_ratio"] is not None), "ratio")}
        drift_rows = [r for r in pair_rows if r["source_gap"] == 5 and r["scale_ratio"] is not None]
        drift_rows.sort(key=lambda row: row["first_frame"])
        if len(drift_rows) >= 4:
            midpoint = len(drift_rows) // 2
            early = float(np.median([r["scale_ratio"] for r in drift_rows[:midpoint]]))
            late = float(np.median([r["scale_ratio"] for r in drift_rows[midpoint:]]))
            drift = {"status": "observable", "gap": 5, "pair_count": len(drift_rows),
                     "early_median_scale_ratio": early, "late_median_scale_ratio": late,
                     "late_minus_early_ratio": late - early,
                     "definition": "One full-segment scale fit; Δ5 scaled estimated displacement / GT displacement, first versus second half of source-ordered reliable pairs."}
        else:
            drift = {"status": "unavailable", "reason": "fewer_than_four_reliable_exact_gap_5_pairs"}
        segments.append({"segment_id": segment_id, "first_valid_frame": ids[0], "last_valid_frame": ids[-1],
                         "online_valid_poses": len(group), "evaluation_eligible_poses": len(errors),
                         "status": "eligible" if fit else "ineligible", "ineligibility_reason": reason,
                         "eligibility_diagnostics": {
                             "estimated_condition": x_condition, "truth_condition": y_condition,
                             "estimated_singular_values": x_singular,
                             "gt_singular_values_m": y_singular,
                             "estimated_second_over_first": None if not x_singular[0] else x_singular[1] / x_singular[0],
                             "gt_second_over_first": None if not y_singular[0] else y_singular[1] / y_singular[0]},
                         "alignment": None if fit is None else {
                             "scale_m_per_estimated_unit": fit.scale,
                             "Q_est_world_to_gt_world": fit.rotation.tolist(),
                             "b_gt_m": fit.translation.tolist(),
                             "estimated_singular_values": fit.estimated_singular_values,
                             "gt_singular_values_m": fit.truth_singular_values,
                             "det_Q": float(np.linalg.det(fit.rotation))},
                         "in_sample_ate_m": _stats((e["ate_m"] for e in errors), "m"),
                         "in_sample_orientation_deg": _stats((e["orientation_deg"] for e in errors), "deg"),
                         "anchor_frame": anchor,
                         "anchor_referenced_orientation_deg": _stats(anchor_errors, "deg"),
                         "relative_by_gap": by_gap, "scale_drift_gap5": drift,
                         "prefix_fit_suffix_evaluation": prefix})
    return segments, frame_errors, relative_rows


def _intervals(rows: List[Dict[str, Any]], valid: bool) -> List[Dict[str, int]]:
    runs: List[Dict[str, int]] = []
    first: Optional[int] = None
    last = -1
    for row in rows:
        frame = row["frame_id"]
        if bool(row["pose_valid"]) == valid:
            if first is None:
                first = frame
            last = frame
        elif first is not None:
            runs.append({"first": first, "last": last, "length": last - first + 1})
            first = None
    if first is not None:
        runs.append({"first": first, "last": last, "length": last - first + 1})
    return runs


def coverage_report(rows: List[Dict[str, Any]], poses: Dict[int, OnlinePose],
                    attempts: List[Dict[str, Any]], losses: List[Dict[str, Any]],
                    summary: Dict[str, Any], keyframe_events: List[Dict[str, Any]],
                    growth_events: List[Dict[str, Any]],
                    recovery_events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    valid_runs, missing_runs = _intervals(rows, True), _intervals(rows, False)
    segments = _segment_groups(poses)
    accepted = [a for a in attempts if a["initialization_status"] == "accepted"]
    rejection_counts = Counter(a["failure_reason"] for a in attempts if a["initialization_status"] != "accepted")
    pnp_failures = []
    for loss in losses:
        row = rows[loss["frame_id"] - rows[0]["frame_id"]]
        diag = row["diagnostics"]
        pnp_failures.append({"frame_id": row["frame_id"], "segment_id": loss["segment_id"],
                             "reason": loss["reason"], "last_successful_frame": loss["last_successful_frame"],
                             "candidate_map_associations": diag.get("candidate_map_associations"),
                             "ratio_passed_map_associations": diag.get("ratio_passed_map_associations"),
                             "unique_map_associations": diag.get("unique_map_associations"),
                             "pnp_correspondences": diag.get("pnp_correspondences"),
                             "pnp_inliers": diag.get("pnp_inliers"),
                             "pnp_inlier_fraction": diag.get("pnp_inlier_fraction")})
    accepted_by_second = {a["candidate_frame"]: a["anchor_frame"] for a in accepted}
    anchors = {}
    for segment_id, group in segments.items():
        first = group[0].frame_id
        if first not in accepted_by_second:
            raise ValueError(f"Segment {segment_id} has no accepted initialization at first online pose")
        anchor = accepted_by_second[first]
        if anchor in poses:
            raise ValueError("Initialization anchor must not count as an online pose")
        anchors[segment_id] = anchor
    if len(accepted) != len(segments):
        raise ValueError("Accepted initialization count disagrees with segment count")
    recoveries = []
    for loss in losses:
        next_valid = next((frame for frame in sorted(poses) if frame > loss["frame_id"]), None)
        recoveries.append({"loss_frame": loss["frame_id"], "next_online_valid_frame": next_valid,
                           "no_pose_frames_until_next_valid": None if next_valid is None else
                           next_valid - loss["frame_id"],
                           "new_segment_id": None if next_valid is None else poses[next_valid].segment_id})
    count = len(rows)
    result = {"frame_range_inclusive": [rows[0]["frame_id"], rows[-1]["frame_id"]],
              "total_frames": count, "online_valid_poses": len(poses),
              "online_valid_fraction": len(poses) / count,
              "no_online_pose_frames": count - len(poses),
              "longest_uninterrupted_valid_span": max((r["length"] for r in valid_runs), default=0),
              "valid_intervals": valid_runs, "missing_pose_intervals": missing_runs,
              "segment_count": len(segments), "segment_lengths": {str(k): len(v) for k, v in segments.items()},
              "segment_anchor_frames": {str(k): v for k, v in anchors.items()},
              "initialization_attempts": len(attempts), "accepted_initializations": len(accepted),
              "initialization_rejections": dict(sorted(rejection_counts.items())),
              "pnp_tracking_losses": len(losses), "pnp_failures": pnp_failures,
              "reinitializations": summary.get("reinitializations"),
              "post_loss_reinitialization": recoveries,
              "recovery_events": len(recovery_events or []),
              "keyframe_events": len(keyframe_events), "keyframes_summary": summary.get("keyframes"),
              "map_growth_events": len(growth_events),
              "new_map_points_summary": summary.get("new_map_points"),
              "map_points_per_segment": summary.get("map_points_per_segment"),
              "ba_attempts": summary.get("ba_attempts"), "ba_accepted": summary.get("ba_accepted"),
              "state_counts": dict(sorted(Counter(row["state"] for row in rows).items()))}
    for key, computed in (("valid_poses", len(poses)),
                          ("longest_valid_span", result["longest_uninterrupted_valid_span"]),
                          ("no_pose_frames", count - len(poses)),
                          ("initialization_attempts", len(attempts))):
        if summary.get(key) != computed:
            raise ValueError(f"Saved summary {key} disagrees with online records")
    return result


def common_frame_comparison(variants: Dict[str, Dict[str, Any]], gt: np.ndarray) -> Dict[str, Any]:
    """Use shared online IDs and fit every segment using only shared IDs.

    Iteratively remove frames whose enclosing segment is not eligible in every
    variant; refit after each removal. This is a matched-frame, matched-fit-ID
    comparison, with no per-frame alignment.
    """
    common = set.intersection(*(set(variants[name]["poses"]) for name in VARIANTS))
    initial = len(common)
    initial_ids = set(common)
    eligibility_iterations = []
    reasons: Dict[str, Any] = {}
    fits: Dict[str, Dict[int, Similarity]] = {}
    for _ in range(initial + 1):
        eligible_sets: List[set] = []
        fits = {}
        reasons = {}
        for name in VARIANTS:
            poses = variants[name]["poses"]
            by_segment: Dict[int, List[int]] = defaultdict(list)
            for frame in sorted(common):
                by_segment[poses[frame].segment_id].append(frame)
            variant_fits: Dict[int, Similarity] = {}
            eligible = set()
            variant_reasons = {}
            for segment_id, ids in by_segment.items():
                fit, reason = fit_similarity(np.asarray([poses[i].center for i in ids]), gt[ids, :3, 3])
                if fit is None:
                    variant_reasons[str(segment_id)] = reason
                else:
                    variant_fits[segment_id] = fit
                    eligible.update(ids)
            fits[name], reasons[name] = variant_fits, variant_reasons
            eligible_sets.append(eligible)
        revised = set.intersection(*eligible_sets) if eligible_sets else set()
        eligibility_iterations.append({"input_count": len(common), "retained_count": len(revised),
                                       "ineligible_segments": reasons})
        if revised == common:
            break
        common = revised
    else:
        raise RuntimeError("Common eligibility did not converge")
    scored: Dict[str, Any] = {}
    for name in VARIANTS:
        poses = variants[name]["poses"]
        errors = [_pose_error(poses[i], gt[i], fits[name][poses[i].segment_id]) for i in sorted(common)]
        scored[name] = {"ate_m": _stats((e["ate_m"] for e in errors), "m"),
                        "orientation_deg": _stats((e["orientation_deg"] for e in errors), "deg"),
                        "alignment_scale_by_segment_m_per_unit": {
                            str(k): v.scale for k, v in sorted(fits[name].items())}}
    return {"initial_common_online_frame_count": initial,
            "eligible_common_frame_count": len(common), "eligible_common_frame_ids": sorted(common),
            "excluded_initial_common_frame_ids": sorted(initial_ids - common),
            "eligibility_iterations": eligibility_iterations,
            "convention": "Intersect original online-valid IDs; fit each variant segment using only shared IDs; remove IDs in ineligible segments in any variant and repeat until stable. Report no metric if empty.",
            "ineligible_segment_reasons_final_iteration": reasons, "variants": scored}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_anchor_keyframes(path: Path, anchors: Dict[int, int]) -> Dict[str, Any]:
    """Use the final export only to verify anchor map coordinates, never coverage."""
    data = json.loads(path.read_text(encoding="utf-8"))
    poses = data.get("poses", [])
    if not poses:
        return {"source_pose_stage": data.get("pose_stage"), "verification_status": "not_exported",
                "use": "baseline has no final keyframe export; anchor identity follows the saved initialization T_cw convention, not an online pose",
                "anchors": [{"segment_id": segment, "anchor_frame": anchor,
                             "anchor_T_cw_identity_verified": False} for segment, anchor in sorted(anchors.items())]}
    verified = []
    for segment, anchor in sorted(anchors.items()):
        matches = [row for row in poses if row.get("segment_id") == segment and row.get("frame_id") == anchor]
        if len(matches) != 1:
            raise ValueError(f"Segment {segment} requires exactly one saved anchor keyframe")
        row = matches[0]
        R = _rotation(row["T_cw_R"], "anchor R_cw")
        t = np.asarray(row["T_cw_t"], dtype=np.float64)
        if t.shape != (3,) or not np.isfinite(t).all() or not np.allclose(R, np.eye(3), atol=1e-8) or not np.allclose(t, 0, atol=1e-8):
            raise ValueError(f"Segment {segment} saved anchor is not the segment world origin")
        verified.append({"segment_id": segment, "anchor_frame": anchor,
                         "anchor_T_cw_identity_verified": True})
    return {"source_pose_stage": data.get("pose_stage"), "verification_status": "verified",
            "use": "anchor coordinate convention validation only; no online coverage", "anchors": verified}


def _timeline(rows: List[Dict[str, Any]], path: Path) -> None:
    width, height = 1500, 260
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    start, end = rows[0]["frame_id"], rows[-1]["frame_id"]
    def x(frame: int) -> int:
        return 80 + int((frame - start) * 1340 / max(1, end - start))
    colors = {"tracking": (50, 155, 50), "initializing": (180, 180, 180), "lost": (40, 40, 230)}
    previous_segment = None
    palette = [(30, 90, 230), (230, 110, 30), (60, 160, 60), (160, 50, 160), (50, 160, 200)]
    for row in rows:
        xx = x(row["frame_id"])
        cv2.line(canvas, (xx, 75), (xx, 130), colors.get(row["state"], (0, 0, 0)), 4)
        if row["pose_valid"]:
            segment = row["segment_id"]
            cv2.line(canvas, (xx, 155), (xx, 180), palette[segment % len(palette)], 4)
            if segment != previous_segment:
                cv2.line(canvas, (xx, 140), (xx, 188), (0, 0, 0), 1)
                cv2.putText(canvas, str(segment), (xx, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            previous_segment = segment
    cv2.putText(canvas, "state: gray initializing, red lost, green tracking", (50, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(canvas, "online pose: colors distinguish segments; gaps unestimated", (50, 235),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(canvas, str(start), (65, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(canvas, str(end), (1410, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Could not write {path}")


def _trajectory_plot(segments: List[Dict[str, Any]], errors: List[Dict[str, Any]],
                     gt: np.ndarray, path: Path, start: int, end: int) -> None:
    canvas = np.full((900, 1300, 3), 255, dtype=np.uint8)
    by_segment: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in errors:
        by_segment[row["segment_id"]].append(row)
    all_points = [gt[start:end + 1, :3, 3][:, [0, 2]]]
    all_points += [np.asarray([e["aligned_center_m"] for e in group])[:, [0, 2]]
                   for group in by_segment.values()]
    points = np.vstack(all_points)
    low, high = points.min(axis=0), points.max(axis=0)
    span = np.maximum(high - low, 1e-9)
    ratio = min(1100 / span[0], 700 / span[1])
    def xy(p: np.ndarray) -> Tuple[int, int]:
        return int(100 + (p[0] - low[0]) * ratio), int(800 - (p[1] - low[1]) * ratio)
    ground = gt[start:end + 1, :3, 3]
    for a, b in zip(ground[:-1], ground[1:]):
        cv2.line(canvas, xy(a[[0, 2]]), xy(b[[0, 2]]), (0, 0, 0), 1)
    palette = [(30, 90, 230), (230, 110, 30), (60, 160, 60), (160, 50, 160), (50, 160, 200)]
    for segment_id, group in sorted(by_segment.items()):
        group.sort(key=lambda e: e["frame_id"])
        color = palette[segment_id % len(palette)]
        for e in group:
            cv2.circle(canvas, xy(np.asarray(e["aligned_center_m"])[[0, 2]]), 3, color, -1)
        for a, b in zip(group[:-1], group[1:]):
            if b["frame_id"] == a["frame_id"] + 1:
                cv2.line(canvas, xy(np.asarray(a["aligned_center_m"])[[0, 2]]),
                         xy(np.asarray(b["aligned_center_m"])[[0, 2]]), color, 2)
    cv2.putText(canvas, "KITTI GT black; eligible aligned online segments colored; gaps unconnected", (25, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)
    cv2.putText(canvas, "Top view X/Z (meters); per-segment offline Sim(3)", (25, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    for line, segment in enumerate(s for s in segments if s["status"] == "eligible"):
        color = palette[segment["segment_id"] % len(palette)]
        yy = 105 + line * 30
        cv2.line(canvas, (930, yy - 5), (960, yy - 5), color, 3)
        label = f"S{segment['segment_id']}: {segment['first_valid_frame']}-{segment['last_valid_frame']}"
        cv2.putText(canvas, label, (970, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Could not write {path}")


def evaluate_frozen(sequence: Path, ground_truth: Path, results_root: Path,
                    output_dir: Path, start: int, end: int, config: Path) -> Dict[str, Any]:
    """Evaluate four existing result folders; create only the new output directory."""
    if output_dir.exists():
        raise FileExistsError(f"Evaluation output already exists: {output_dir}")
    gt = load_kitti_poses(ground_truth)
    index = validate_source_index(sequence, len(gt), start, end)
    variants: Dict[str, Dict[str, Any]] = {}
    for name in VARIANTS:
        folder = results_root / name
        rows, poses = load_online_rows(folder / "per_frame.jsonl", start, end)
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        attempts = json.loads((folder / "initialization_attempts.json").read_text(encoding="utf-8"))
        losses = json.loads((folder / "loss_events.json").read_text(encoding="utf-8"))
        keyframes = json.loads((folder / "keyframe_events.json").read_text(encoding="utf-8"))
        growth = json.loads((folder / "map_growth_events.json").read_text(encoding="utf-8"))
        recovery = json.loads((folder / "recovery_events.json").read_text(encoding="utf-8"))
        coverage = coverage_report(rows, poses, attempts, losses, summary, keyframes, growth, recovery)
        anchors = {int(k): v for k, v in coverage["segment_anchor_frames"].items()}
        anchor_validation = validate_anchor_keyframes(folder / "final_keyframes.json", anchors)
        segments, errors, relative_rows = evaluate_segments(poses, gt, anchors)
        variants[name] = {"rows": rows, "poses": poses, "coverage": coverage,
                          "segments": segments, "errors": errors, "relative": relative_rows,
                          "anchor_validation": anchor_validation}
    common = common_frame_comparison(variants, gt)
    comparison = []
    for name in VARIANTS:
        value = variants[name]
        errors = value["errors"]
        comparison.append({"variant": name, "total_frames": end - start + 1,
                           "online_valid_poses": value["coverage"]["online_valid_poses"],
                           "eligible_poses": len(errors),
                           "eligible_segments": sum(s["status"] == "eligible" for s in value["segments"]),
                           "own_eligible_ate_rmse_m": _stats((e["ate_m"] for e in errors), "m")["rmse"],
                           "own_eligible_ate_median_m": _stats((e["ate_m"] for e in errors), "m")["median"],
                           "own_eligible_orientation_median_deg": _stats((e["orientation_deg"] for e in errors), "deg")["median"],
                           "common_eligible_pose_count": common["eligible_common_frame_count"],
                           "common_ate_rmse_m": common["variants"][name]["ate_m"]["rmse"],
                           "common_orientation_median_deg": common["variants"][name]["orientation_deg"]["median"]})
    published_fields = ("frame_id", "state", "segment_id", "pose_valid", "T_cw_R", "T_cw_t",
                        "camera_center_world")
    full_descriptors_equal = all(
        all(a.get(key) == b.get(key) for key in published_fields)
        for a, b in zip(variants["full"]["rows"], variants["descriptors"]["rows"]))
    fingerprint_files = {"ground_truth": ground_truth, "times": sequence / "times.txt",
                         "calibration": sequence / "calib.txt", "config": config,
                         "evaluation_source": Path(__file__)}
    for name in VARIANTS:
        for basename in ("per_frame.jsonl", "summary.json", "initialization_attempts.json",
                         "loss_events.json", "recovery_events.json", "final_keyframes.json"):
            fingerprint_files[f"{name}/{basename}"] = results_root / name / basename
    image_digests = [(p.name, _sha(p)) for p in sorted((sequence / "image_0").glob("*.png"))]
    image_manifest_hash = hashlib.sha256(json.dumps(image_digests, separators=(",", ":")).encode()).hexdigest()
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[2],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    provenance = {"source_index": index, "sha256": {name: _sha(path) for name, path in fingerprint_files.items()},
                  "all_image_content_manifest_sha256": image_manifest_hash,
                  "image_count_hashed": len(image_digests), "git_head": head,
                  "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                  "opencv": cv2.__version__, "scipy": scipy.__version__,
                                  "platform": platform.platform()},
                  "input_paths": {"sequence": str(sequence.resolve()), "ground_truth": str(ground_truth.resolve()),
                                  "results_root": str(results_root.resolve()), "config": str(config.resolve())},
                  "pose_source": "per_frame.jsonl only; online T_cw; GT T_wc",
                  "full_and_descriptors_published_online_poses_identical": full_descriptors_equal,
                  "eligibility": {"min_online_poses": MIN_ALIGNMENT_POSES,
                                  "min_second_over_first_center_singular_value": MIN_SECOND_SINGULAR_RATIO,
                                  "center_extent_floor": "first singular > 1e-9 * max(1, max center norm), each cloud",
                                  "finite_centers_required": True, "third_singular_value_required": False},
                  "relative_translation": "C_j-C_i rotated by R_cw_i and compared with GT displacement rotated by R_wc_i.T; fitted segment scale converts estimated units to meters; displacement error is absolute length difference; scale ratio is fitted estimated length / GT length",
                  "relative_direction_min_gt_m": MIN_GT_DISPLACEMENT_M,
                  "prefix_suffix_rule": "For eligible segments >=10 poses, fit first ceil(n/2) if eligible; evaluate remaining poses without refitting.",
                  "source_timestamp_limit": "KITTI pose rows contain no independent timestamps; validated positional filename/times/pose-row contract cannot prove external synchronization."}
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "provenance.json", provenance)
    _write_json(output_dir / "common_frame_comparison.json", common)
    _write_json(output_dir / "four_way_comparison.json", comparison)
    _write_csv(output_dir / "four_way_comparison.csv", comparison, tuple(comparison[0]))
    for name in VARIANTS:
        value = variants[name]
        folder = output_dir / name
        folder.mkdir()
        _write_json(folder / "segments.json", value["segments"])
        _write_json(folder / "coverage_failures.json", value["coverage"])
        _write_json(folder / "anchor_validation.json", value["anchor_validation"])
        _write_json(folder / "per_frame_errors.json", value["errors"])
        _write_csv(folder / "per_frame_errors.csv", value["errors"],
                   ("frame_id", "segment_id", "ate_m", "orientation_deg", "aligned_center_m", "gt_center_m"))
        _write_json(folder / "relative_pairs.json", value["relative"])
        _write_csv(folder / "relative_pairs.csv", value["relative"],
                   ("first_frame", "second_frame", "source_gap", "segment_id", "rotation_error_deg",
                    "translation_direction_error_deg", "estimated_displacement_units", "gt_displacement_m",
                    "scaled_displacement_m", "displacement_error_m", "relative_translation_vector_error_m",
                    "scale_ratio", "translation_reliable"))
        timeline = [{"frame_id": r["frame_id"], "state": r["state"],
                     "segment_id": r["segment_id"], "pose_valid": r["pose_valid"],
                     "failure_reason": r["failure_reason"]} for r in value["rows"]]
        _write_json(folder / "frame_timeline.json", timeline)
        _write_csv(folder / "frame_timeline.csv", timeline,
                   ("frame_id", "state", "segment_id", "pose_valid", "failure_reason"))
        _timeline(value["rows"], folder / "frame_timeline.png")
        _trajectory_plot(value["segments"], value["errors"], gt,
                         folder / "aligned_trajectory_xz.png", start, end)
    reproduction = ("Run in PowerShell from the repository root:\n"
                    "$env:PYTHONPATH = (Resolve-Path .\\src).Path\n"
                    "python -m orb_slam_mono.trajectory_evaluation "
                    f"--sequence {sequence} --ground-truth {ground_truth} "
                    f"--results-root {results_root} --output-dir {output_dir} "
                    f"--config {config} --start {start} --end {end}\n\n"
                    "The output directory must not already exist.\n"
                    "Full test suite: python -m unittest discover -s tests -v")
    (output_dir / "REPRODUCE.txt").write_text(reproduction + "\n", encoding="utf-8")
    (output_dir / "METRICS.md").write_text(
        "# Frozen M5-A metric definitions\n\n"
        "All online successes come only from chronological per_frame.jsonl. "
        "GT is KITTI T_wc, indexed by original six-digit image ID. Where exported, "
        "anchor-only final keyframes validate the map origin but never enter online coverage. "
        "The baseline exports no final keyframes; its anchor orientation uses the saved "
        "initialization coordinate convention.\n\n"
        "For each eligible segment, fit C_gt = s Q C_est + b by proper Umeyama least squares "
        "with s > 0 and det(Q) = +1. Eligibility: at least five online poses, finite centers, "
        "first centered singular value > 1e-9 times max(1, maximum center norm), and "
        "second/first singular value >= 0.05 in both center clouds. A planar third singular "
        "value is allowed. No fit crosses segments.\n\n"
        "ATE is Euclidean aligned center error in meters. In-sample RMSE is sqrt(mean(error^2)); "
        "median and p90 are over eligible online poses. Aligned orientation is geodesic "
        "angle of Q R_wc_est relative to R_wc_gt, in degrees. Anchor-referenced orientation "
        "uses GT R_wc at the independent map's saved identity anchor in place of Q; it is "
        "reported separately.\n\n"
        "RPE pairs share one segment and have exact original source-frame gaps 1, 5, or 10. "
        "Relative rotation is R_cw_j R_cw_i.T versus R_wc_gt_j.T R_wc_gt_i. Translation "
        "vectors are C_j-C_i in first-camera axes; only direction is scale free. A direction "
        "requires estimated length > 1e-8 and GT length >= 0.05 m. Displacement error "
        "uses one fixed full-segment fitted scale: absolute difference of estimated and "
        "GT lengths in meters. Scale ratio is scaled estimated length / GT length. "
        "Relative translation vector error is the norm of the difference between the "
        "two first-camera displacement vectors after fixed-scale conversion, in meters. "
        "The drift diagnostic compares early and late medians of reliable exact-gap-5 "
        "scale ratios; at least four pairs are required.\n\n"
        "Prefix/suffix: eligible segments with >=10 poses fit the first ceil(n/2) poses if "
        "that prefix is eligible, then score the remaining poses without refitting. "
        "Common-frame comparison intersects online-valid original IDs and fits each "
        "variant segment only on those shared IDs, removing ineligible segments in any "
        "variant until the set stabilizes. An empty set yields unavailable metrics. "
        "Coverage always uses all requested source frames.\n\n"
        "KITTI pose rows contain no independent timestamps; the evaluator validates "
        "image-ID, timestamp-row, and pose-row indexing and the first zero timestamp, "
        "but cannot independently prove external synchronization.\n",
        encoding="utf-8")
    return {"comparison": comparison, "common": common,
            "segments": {name: variants[name]["segments"] for name in VARIANTS},
            "coverage": {name: variants[name]["coverage"] for name in VARIANTS}}


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args(argv)
    result = evaluate_frozen(args.sequence, args.ground_truth, args.results_root,
                             args.output_dir, args.start, args.end, args.config)
    print(json.dumps({"comparison": result["comparison"],
                      "common_eligible_frames": result["common"]["eligible_common_frame_count"]}, indent=2))


if __name__ == "__main__":
    main()
