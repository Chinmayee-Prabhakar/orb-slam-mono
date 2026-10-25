"""Opt-in, offline analysis of saved M4 tracking failures and a diagnostic replay.

This module never supplies truth, correspondences, thresholds, or poses to the
tracker. It only reads immutable copies emitted by opt-in capture hooks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import scipy

from .bundle_adjustment import BundleConfig
from .datasets import KittiGraySequence
from .features import OrbExtractor
from .geometry import GeometryConfig
from .local_mapping import MappingConfig
from .matching import DescriptorMatcher
from .tracking import MonocularTracker, TrackingConfig


FAILURE_IDS = (8, 20, 43, 82, 122, 202, 207, 282)
PRIORITY_IDS = (43, 82, 207, 282)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default, allow_nan=False))


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "p90": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(array), "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90)), "min": float(array.min()),
            "max": float(array.max())}


def _spatial(pixels: np.ndarray, width: int, height: int) -> Dict[str, Any]:
    pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if not len(pixels):
        return {"count": 0, "hull_image_fraction": None, "occupied_4x4_cells": 0,
                "x_range_px": None, "y_range_px": None}
    cells = {(min(3, max(0, int(x * 4 / width))), min(3, max(0, int(y * 4 / height))))
             for x, y in pixels}
    hull = None if len(pixels) < 3 else float(cv2.contourArea(cv2.convexHull(pixels.astype(np.float32))) / (width * height))
    return {"count": len(pixels), "hull_image_fraction": hull,
            "occupied_4x4_cells": len(cells),
            "x_range_px": [float(pixels[:, 0].min()), float(pixels[:, 0].max())],
            "y_range_px": [float(pixels[:, 1].min()), float(pixels[:, 1].max())]}


def _candidate_projection(event: Dict[str, Any], camera_matrix: np.ndarray) -> Dict[str, Any]:
    raw = event["pnp_candidate"]
    correspondences = event["correspondences"]
    if (raw.get("ransac_success") is not True or raw.get("raw_rvec") is None
            or raw.get("raw_tvec") is None or raw.get("raw_inlier_indices") is None):
        return {"available": False, "reason": "PnP stopped before a usable RANSAC candidate",
                "pose_stage": "no candidate; no accepted pose", "raw_inlier_indices": [],
                "reprojection_error_px": None, "depth": None}
    rvec = np.asarray(raw["raw_rvec"], dtype=np.float64).reshape(3)
    tvec = np.asarray(raw["raw_tvec"], dtype=np.float64).reshape(3)
    xyz = np.asarray([row["xyz_world"] for row in correspondences], dtype=np.float64).reshape(-1, 3)
    uv = np.asarray([row["pixel_xy"] for row in correspondences], dtype=np.float64).reshape(-1, 2)
    projected = cv2.projectPoints(xyz, rvec, tvec, camera_matrix, None)[0].reshape(-1, 2)
    R, _ = cv2.Rodrigues(rvec)
    depth = ((R @ xyz.T).T + tvec)[:, 2]
    residuals = np.linalg.norm(projected - uv, axis=1)
    indices = np.asarray(raw["raw_inlier_indices"], dtype=int).reshape(-1)
    return {"available": True, "pose_stage": raw["pose_stage"],
            "raw_inlier_indices": indices.tolist(), "candidate_R_cw": R.tolist(),
            "candidate_t_cw": tvec.tolist(),
            "reprojection_error_px": residuals.tolist(), "depth": depth.tolist(),
            "projected_pixels": projected.tolist(),
            "all_residual_px": _stats(residuals.tolist()),
            "ransac_inlier_residual_px": _stats(residuals[indices].tolist()),
            "positive_depth_fraction_all": float(np.mean(depth > 0))}


def diagnose_pnp_event(event: Dict[str, Any], result: Dict[str, Any],
                       camera_matrix: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return compact per-frame record and detailed non-mutating evidence."""
    height, width = image_shape
    matches = event["matcher"]
    keypoints = event["keypoints_xy"]
    active = event["active_landmarks"]
    candidates = []
    for row in matches["candidates"]:
        index = row["map_descriptor_index"]
        enriched = dict(row)
        enriched["point_id"] = event["candidate_point_ids"][index]
        enriched["pixel_xy"] = keypoints[row["image_feature_index"]].tolist()
        candidates.append(enriched)
    correspondences = [dict(row) for row in event["correspondences"]]
    candidate = _candidate_projection(event, camera_matrix)
    inliers = set(candidate["raw_inlier_indices"])
    for index, row in enumerate(correspondences):
        row["ransac_inlier"] = None if not candidate["available"] else index in inliers
        row["candidate_reprojection_error_px"] = None if not candidate["available"] else candidate["reprojection_error_px"][index]
        row["candidate_depth"] = None if not candidate["available"] else candidate["depth"][index]
    uv = np.asarray([row["pixel_xy"] for row in correspondences], dtype=np.float64).reshape(-1, 2)
    inlier_uv = uv[sorted(inliers)] if candidate["available"] else np.empty((0, 2))
    outlier_uv = uv[[i for i in range(len(uv)) if i not in inliers]] if candidate["available"] else np.empty((0, 2))
    active_xyz = np.asarray([row["xyz_world"] for row in active], dtype=np.float64).reshape(-1, 3)
    singular = np.linalg.svd(active_xyz - active_xyz.mean(axis=0), compute_uv=False).tolist() if len(active) else []
    prior = event["previous_accepted_pose"]
    prior_visible = None
    if prior is not None and len(active):
        R = np.asarray(prior["T_cw_R"], dtype=np.float64)
        t = np.asarray(prior["T_cw_t"], dtype=np.float64)
        camera_xyz = (R @ active_xyz.T).T + t
        valid = camera_xyz[:, 2] > 1e-8
        projected = np.full((len(active), 2), np.nan)
        projected[valid, 0] = camera_matrix[0, 0] * camera_xyz[valid, 0] / camera_xyz[valid, 2] + camera_matrix[0, 2]
        projected[valid, 1] = camera_matrix[1, 1] * camera_xyz[valid, 1] / camera_xyz[valid, 2] + camera_matrix[1, 2]
        visible = valid & (projected[:, 0] >= 0) & (projected[:, 0] < width) & (projected[:, 1] >= 0) & (projected[:, 1] < height)
        prior_visible = int(np.count_nonzero(visible))
    by_point = {row["point_id"]: row for row in active}
    retained_meta = [by_point[row["point_id"]] for row in correspondences]
    distances = [row["best_hamming"] for row in candidates]
    ratio_distances = [row["best_hamming"] for row in candidates if row["ratio_pass"]]
    retained_distances = [row["hamming"] for row in correspondences]
    inlier_distances = [correspondences[i]["hamming"] for i in sorted(inliers)]
    outlier_distances = [row["hamming"] for i, row in enumerate(correspondences) if i not in inliers] if candidate["available"] else []
    compact = {
        "frame_id": result["frame_id"], "state": result["state"], "segment_id": result["segment_id"],
        "pose_valid": result["pose_valid"], "failure_reason": result["failure_reason"],
        "features_detected": result["diagnostics"].get("features_detected"),
        "stored_map_landmarks": len(active), "forward_complete_pairs": matches["forward_complete_pairs"],
        "ratio_passed": matches["ratio_passed"], "mutual_passed_unique_correspondences": matches["mutual_passed"],
        "pnp_correspondences": result["diagnostics"].get("pnp_correspondences"),
        "pnp_ransac_attempted": result["diagnostics"].get("pnp_ransac_attempted"),
        "pnp_raw_inliers": None if not candidate["available"] else len(inliers),
        "pnp_raw_inlier_fraction": None if not candidate["available"] or not len(uv) else len(inliers) / len(uv),
        "match_spatial": _spatial(uv, width, height),
        "inlier_spatial": _spatial(inlier_uv, width, height) if candidate["available"] else None,
        "outlier_spatial": _spatial(outlier_uv, width, height) if candidate["available"] else None,
        "descriptor_hamming_all_best": _stats(distances),
        "descriptor_hamming_ratio_passed": _stats(ratio_distances),
        "descriptor_hamming_retained": _stats(retained_distances),
        "descriptor_hamming_inlier": _stats(inlier_distances),
        "descriptor_hamming_outlier": _stats(outlier_distances),
        "matched_landmark_observations": _stats([row["observation_count"] for row in retained_meta]),
        "matched_landmark_age_frames": _stats([row["age_since_observation_frames"] for row in retained_meta]),
        "active_landmark_observations": _stats([row["observation_count"] for row in active]),
        "active_landmark_age_frames": _stats([row["age_since_observation_frames"] for row in active]),
        "active_landmark_xyz_singular_values": singular,
        "visible_from_previous_accepted_pose": prior_visible,
        "previous_accepted_frame": None if prior is None else prior["frame_id"],
        "raw_candidate_residual_px": candidate.get("all_residual_px"),
        "raw_candidate_positive_depth_fraction": candidate.get("positive_depth_fraction_all"),
    }
    detail = {"summary": compact, "matcher_candidates": candidates,
              "correspondences": correspondences, "active_landmarks": active,
              "candidate": candidate, "previous_accepted_pose": prior,
              "keypoints_xy": keypoints.tolist(),
              "interpretation": "RANSAC candidate remains rejected unless the saved online pose is valid"}
    return _plain(compact), _plain(detail)


def _normalized(value: Any) -> Any:
    """Drop only measured runtime fields when comparing deterministic outputs."""
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()
                if key != "timing_ms" and key != "runtime_ms" and not key.endswith("_ms")}
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    return value


def _ply_bytes(points: List[np.ndarray]) -> bytes:
    lines = ["ply", "format ascii 1.0", "comment arbitrary monocular scale; segment-local first-camera world frame",
             f"element vertex {len(points)}", "property float x", "property float y", "property float z", "end_header"]
    lines.extend(f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g}" for point in points)
    # sequence_demo uses text mode on Windows, which writes CRLF to frozen PLY.
    return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def compare_replay(rows: List[Dict[str, Any]], tracker: MonocularTracker, frozen_dir: Path) -> Dict[str, Any]:
    frozen = [json.loads(line) for line in (frozen_dir / "per_frame.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != len(frozen):
        raise ValueError("Replay and frozen frame counts differ")
    core = ("frame_id", "state", "pose_valid", "segment_id", "failure_reason",
            "T_cw_R", "T_cw_t", "camera_center_world")
    mismatches = []
    comparisons = 0
    maximum_pose_delta = 0.0
    for replay, reference in zip(rows, frozen):
        for key in core:
            comparisons += 1
            if replay[key] != reference[key]:
                mismatches.append({"frame_id": replay["frame_id"], "field": key})
            if key in ("T_cw_R", "T_cw_t") and replay[key] is not None and reference[key] is not None:
                maximum_pose_delta = max(maximum_pose_delta, float(np.max(np.abs(
                    np.asarray(replay[key]) - np.asarray(reference[key])))))
        comparisons += 1
        if _normalized(replay["diagnostics"]) != _normalized(reference["diagnostics"]):
            mismatches.append({"frame_id": replay["frame_id"], "field": "non_timing_diagnostics"})
    events = {"initialization_attempts": tracker.attempts, "loss_events": tracker.loss_events,
              "recovery_events": tracker.recovery_events, "keyframe_events": tracker.keyframe_events,
              "map_growth_events": tracker.mapping_events, "ba_events": tracker.ba_events}
    event_results = {}
    for name, values in events.items():
        reference = json.loads((frozen_dir / f"{name}.json").read_text(encoding="utf-8"))
        equal = _normalized(_plain(values)) == _normalized(reference)
        event_results[name] = {"replay_count": len(values), "frozen_count": len(reference),
                               "non_timing_equal": equal}
        if not equal:
            mismatches.append({"field": name})
    final_reference = json.loads((frozen_dir / "final_keyframes.json").read_text(encoding="utf-8"))["poses"]
    final_replay = [{"segment_id": segment.segment_id, "frame_id": keyframe.frame_id,
                     "T_cw_R": keyframe.frame.R_cw.tolist(), "T_cw_t": keyframe.frame.t_cw.tolist(),
                     "camera_center_world": keyframe.frame.camera_center_world.tolist()}
                    for segment in tracker.segments for keyframe in segment.keyframes.values()]
    final_keyframes_equal = final_replay == final_reference
    if not final_keyframes_equal:
        mismatches.append({"field": "final_keyframes"})
    ply_results = {}
    for segment in tracker.segments:
        filename = f"segment-{segment.segment_id:03d}.ply"
        points = [point.xyz_world for point in segment.points.values() if point.valid]
        actual = _ply_bytes(points)
        reference = (frozen_dir / "maps" / filename).read_bytes()
        equal = actual == reference
        ply_results[filename] = {"replay_valid_points": len(points), "byte_equal": equal,
                                 "sha256": hashlib.sha256(actual).hexdigest()}
        if not equal:
            mismatches.append({"field": f"maps/{filename}"})
    replay_valid = sum(row["pose_valid"] for row in rows)
    frozen_summary = json.loads((frozen_dir / "summary.json").read_text(encoding="utf-8"))
    summary_equal = (replay_valid == frozen_summary["valid_poses"] and
                     len(tracker.segments) == len(frozen_summary["accepted_initialization_pairs"]) and
                     len(tracker.loss_events) == frozen_summary["tracking_failures"] and
                     sum(event["applied"] for event in tracker.ba_events) == frozen_summary["ba_accepted"])
    if not summary_equal:
        mismatches.append({"field": "summary_counts"})
    return {"passed": not mismatches, "frames_compared": len(rows),
            "per_frame_fields_compared": comparisons, "pose_tolerance": 0.0,
            "maximum_absolute_pose_difference": maximum_pose_delta,
            "timing_policy": "Only timing_ms and nested *_ms runtime measurements excluded",
            "mismatches": mismatches, "event_lists": event_results,
            "final_keyframes_equal": final_keyframes_equal, "final_map_ply": ply_results,
            "summary_counts_equal": summary_equal,
            "frozen_map_identity_limit": "Frozen PLY omits point IDs and observation links; byte-identical coordinates and recorded event/count evolution are the strongest available comparison."}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default,
                               allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _flat_record(row: Dict[str, Any]) -> Dict[str, Any]:
    fields = ("frame_id", "state", "segment_id", "pose_valid", "failure_reason",
              "features_detected", "stored_map_landmarks", "forward_complete_pairs",
              "ratio_passed", "mutual_passed_unique_correspondences", "pnp_correspondences",
              "pnp_ransac_attempted", "pnp_raw_inliers", "pnp_raw_inlier_fraction",
              "visible_from_previous_accepted_pose", "previous_accepted_frame")
    output = {key: row.get(key) for key in fields}
    for group in ("match_spatial", "inlier_spatial", "outlier_spatial"):
        value = row.get(group)
        output[group + "_cells"] = None if value is None else value["occupied_4x4_cells"]
        output[group + "_hull_fraction"] = None if value is None else value["hull_image_fraction"]
    for group in ("descriptor_hamming_retained", "descriptor_hamming_inlier",
                  "descriptor_hamming_outlier", "matched_landmark_observations",
                  "matched_landmark_age_frames", "raw_candidate_residual_px"):
        value = row.get(group)
        output[group + "_median"] = None if value is None else value["median"]
        output[group + "_p90"] = None if value is None else value["p90"]
    return output


def _write_image(path: Path, pixels: np.ndarray) -> None:
    if not cv2.imwrite(str(path), pixels):
        raise OSError(f"Unable to write {path}")


def _canvas(title: str, width: int = 1440, height: int = 480) -> np.ndarray:
    pixels = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(pixels, title, (26, 38), cv2.FONT_HERSHEY_SIMPLEX, .85, (20, 20, 20), 2, cv2.LINE_AA)
    return pixels


def _label(image: np.ndarray, label: str, x: int, y: int,
           color: Tuple[int, int, int] = (30, 30, 30), scale: float = .55) -> None:
    cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _timeline(rows: List[Dict[str, Any]], path: Path) -> None:
    image = _canvas("KITTI 00 frames 0-299: online tracking state (frozen M4 equivalent)", height=360)
    left, right = 75, 1370
    cv2.line(image, (left, 260), (right, 260), (90, 90, 90), 1)
    for row in rows:
        frame = row["frame_id"]
        x = round(left + frame * (right-left) / 299)
        color = ((35, 155, 55) if row["pose_valid"] else
                 ((35, 45, 220) if row["state"] == "lost" else (185, 185, 185)))
        cv2.line(image, (x, 110), (x, 248), color, 3)
        if row["state"] == "lost":
            cv2.circle(image, (x, 90), 6, (35, 45, 220), -1)
            _label(image, str(frame), x-9, 72, scale=.42)
        if row["pose_valid"] and (frame == 0 or rows[frame-1]["segment_id"] != row["segment_id"]):
            _label(image, f"S{row['segment_id']}", x-6, 280, scale=.43)
    for frame in range(0, 300, 25):
        x = round(left + frame * (right-left) / 299)
        _label(image, str(frame), x-9, 310, scale=.48)
    _label(image, "Green = accepted online pose; gray = reinitializing/no pose; red = PnP loss", 75, 336)
    _write_image(path, image)


def _series(image: np.ndarray, rows: List[Dict[str, Any]], key: str, x0: int, x1: int,
            y0: int, y1: int, limit: float, color: Tuple[int, int, int]) -> None:
    previous = None
    for row in rows:
        value = row.get(key)
        if value is None:
            previous = None
            continue
        x = round(x0 + row["frame_id"] * (x1-x0) / 299)
        y = round(y1 - min(float(value)/limit, 1.0)*(y1-y0))
        if previous is not None:
            cv2.line(image, previous, (x, y), color, 2)
        cv2.circle(image, (x, y), 2, color, -1)
        previous = (x, y)


def _support(rows: List[Dict[str, Any]], path: Path) -> None:
    image = _canvas("PnP support: all map-tracking attempts; gaps mean unavailable", height=815)
    x0, x1 = 72, 1370
    panels = ((85, 280, 700, (("stored_map_landmarks", (90, 90, 90)),)),
              (338, 553, 130, (("ratio_passed", (180, 50, 150)),
                               ("mutual_passed_unique_correspondences", (215, 110, 40)),
                               ("pnp_raw_inliers", (40, 150, 45)))),
              (615, 750, 1.0, (("pnp_raw_inlier_fraction", (40, 150, 45)),)))
    for y0, y1, limit, entries in panels:
        cv2.rectangle(image, (x0, y0), (x1, y1), (185, 185, 185), 1)
        for key, color in entries:
            _series(image, rows, key, x0, x1, y0, y1, limit, color)
    count_gate_y = round(553-12/130*(553-338))
    cv2.line(image, (x0, count_gate_y), (x1, count_gate_y), (30, 30, 220), 1)
    _label(image, "12", x1-30, count_gate_y-4, (30, 30, 180), .43)
    threshold_y = round(750-.5*(750-615))
    cv2.line(image, (x0, threshold_y), (x1, threshold_y), (30, 30, 220), 1)
    _label(image, "RANSAC inlier fraction >= 0.50", 82, threshold_y-6, (30, 30, 180))
    _label(image, "Top: stored landmarks = complete forward KNN pairs here (0-700); middle: filtered support (0-130)", 80, 310)
    _label(image, "Middle gate: >=12 unique correspondences and >=12 RANSAC inliers", 80, 580)
    for frame in FAILURE_IDS:
        x = round(x0 + frame*(x1-x0)/299)
        cv2.line(image, (x, 85), (x, 280), (40, 40, 220), 1)
        cv2.line(image, (x, 338), (x, 553), (40, 40, 220), 1)
        _label(image, str(frame), x-9, 78 if frame != 207 else 64, (40, 40, 180), .4)
    _label(image, "Gray active map/KNN | magenta ratio | blue unique 2D-3D | green raw inliers/fraction", 80, 783)
    _label(image, "Missing inlier counts remain blank, never zero; red verticals mark rejected frames", 80, 806, scale=.48)
    _write_image(path, image)


def _overlay(sequence: KittiGraySequence, frame_id: int, detail: Dict[str, Any], path: Path) -> None:
    image = cv2.cvtColor(sequence[frame_id].image, cv2.COLOR_GRAY2BGR)
    height, width = image.shape[:2]
    keypoints = detail["keypoints_xy"]
    for x, y in keypoints:
        cv2.circle(image, (round(x), round(y)), 1, (120, 120, 120), -1)
    candidates = detail["matcher_candidates"]
    for row in candidates:
        if row["ratio_pass"]:
            x, y = row["pixel_xy"]
            cv2.circle(image, (round(x), round(y)), 3, (190, 80, 180), 1)
    for row in detail["correspondences"]:
        x, y = row["pixel_xy"]
        color = ((30, 160, 35) if row["ransac_inlier"] else
                 ((35, 45, 225) if row["ransac_inlier"] is False else (20, 150, 230)))
        cv2.circle(image, (round(x), round(y)), 6, color, 2)
    summary = detail["summary"]
    candidate = detail["candidate"]
    header = np.full((112, width, 3), (245, 245, 245), np.uint8)
    _label(header, f"Frame {frame_id} | {summary['failure_reason']} | REJECTED RANSAC CANDIDATE" if candidate["available"]
           else f"Frame {frame_id} | {summary['failure_reason']} | no PnP candidate", 12, 24, scale=.62)
    _label(header, f"ORB {summary['features_detected']} | map {summary['stored_map_landmarks']} | KNN {summary['forward_complete_pairs']} | ratio {summary['ratio_passed']} | unique {summary['pnp_correspondences']}", 12, 51)
    count = summary["pnp_raw_inliers"]
    fraction = summary["pnp_raw_inlier_fraction"]
    _label(header, f"Raw inliers {count if count is not None else 'N/A'} | fraction {fraction:.3f}" if fraction is not None
           else "Raw inliers N/A | fraction N/A", 12, 78)
    _label(header, "Gray ORB | magenta ratio-pass | orange retained, no candidate | green inlier | red outlier", 12, 102, scale=.46)
    _write_image(path, np.vstack((header, image)))


def _pair_comparison(sequence: KittiGraySequence, before: int, after: int,
                     details: Dict[int, Dict[str, Any]], rows: List[Dict[str, Any]], path: Path) -> None:
    panels = []
    for frame_id in (before, after):
        image = cv2.cvtColor(sequence[frame_id].image, cv2.COLOR_GRAY2BGR)
        detail = details.get(frame_id)
        if detail is not None:
            for row in detail["correspondences"]:
                x, y = row["pixel_xy"]
                color = ((30, 160, 35) if row["ransac_inlier"] else
                         ((35, 45, 225) if row["ransac_inlier"] is False else (20, 150, 230)))
                cv2.circle(image, (round(x), round(y)), 5, color, 2)
        summary = None if detail is None else detail["summary"]
        header = np.full((72, image.shape[1], 3), 245, np.uint8)
        if summary is None:
            _label(header, f"Frame {frame_id}: ACCEPTED initialization; no PnP attempted", 12, 27, scale=.68)
            _label(header, f"New map segment {rows[frame_id]['segment_id']} | ORB {rows[frame_id]['diagnostics'].get('features_detected')} | PnP quantities unavailable", 12, 55)
        else:
            _label(header, f"Frame {frame_id}: {'ACCEPTED' if summary['pose_valid'] else 'REJECTED candidate/no candidate'}", 12, 27, scale=.68)
            _label(header, f"Map {summary['stored_map_landmarks']} | ratio {summary['ratio_passed']} | unique {summary['pnp_correspondences']} | raw inliers {summary['pnp_raw_inliers'] if summary['pnp_raw_inliers'] is not None else 'N/A'}", 12, 55)
        panels.append(np.vstack((header, image)))
    combined = np.hstack(panels)
    footer = np.full((40, combined.shape[1], 3), 245, np.uint8)
    _label(footer, "Green raw RANSAC inlier | red raw outlier | orange no candidate. Raw support does not imply accepted pose.", 12, 27)
    _write_image(path, np.vstack((combined, footer)))


def _residuals(details: Dict[int, Dict[str, Any]], path: Path) -> None:
    image = _canvas("Rejected RANSAC candidates: all correspondence reprojection residuals", height=570)
    x0, x1 = 150, 1350
    y0, y1 = 80, 490
    cv2.rectangle(image, (x0, y0), (x1, y1), (190, 190, 190), 1)
    for row, frame_id in enumerate(FAILURE_IDS):
        y = 107 + row*48
        detail = details[frame_id]
        candidate = detail["candidate"]
        _label(image, f"{frame_id}", 45, y+4)
        if not candidate["available"]:
            _label(image, "no candidate", 157, y+4, (100, 100, 100))
            continue
        for index, error in enumerate(candidate["reprojection_error_px"]):
            x = round(x0 + min(error, 30)/30*(x1-x0))
            color = (30, 150, 35) if index in candidate["raw_inlier_indices"] else (35, 45, 225)
            cv2.circle(image, (x, y), 3, color, -1)
    for px in (0, 3, 10, 20, 30):
        x = round(x0 + px/30*(x1-x0))
        _label(image, str(px), x-8, 518)
    _label(image, "Reprojection error (pixels), clipped at 30 for display | green raw inlier, red outlier", 170, 549)
    _write_image(path, image)


def _trajectory_zoom(records: List[Dict[str, Any]], segment_id: int, first: int,
                     last: int, path: Path) -> None:
    selected = [row for row in records if row["segment_id"] == segment_id and first <= row["frame_id"] <= last]
    if not selected:
        raise ValueError(f"No M5-A records for segment {segment_id}")
    image = _canvas(f"Segment {segment_id}, frames {first}-{last}: offline in-sample Sim(3) alignment", 1300, 730)
    gt = np.asarray([row["gt_center_m"] for row in selected])[:, [0, 2]]
    estimated = np.asarray([row["aligned_center_m"] for row in selected])[:, [0, 2]]
    cloud = np.vstack((gt, estimated))
    low, high = cloud.min(axis=0), cloud.max(axis=0)
    extent = max(float(np.max(high-low)), 1.0)
    center = (low+high)/2
    x0, y0, side = 90, 95, 570
    def pixel(xz: np.ndarray) -> Tuple[int, int]:
        return (round(x0 + side/2 + (xz[0]-center[0])/extent*side*.87),
                round(y0 + side/2 - (xz[1]-center[1])/extent*side*.87))
    cv2.rectangle(image, (x0, y0), (x0+side, y0+side), (150, 150, 150), 1)
    for points, color in ((gt, (35, 145, 35)), (estimated, (30, 90, 215))):
        for first_point, second_point in zip(points, points[1:]):
            cv2.line(image, pixel(first_point), pixel(second_point), color, 2)
        for point in points:
            cv2.circle(image, pixel(point), 3, color, -1)
    _label(image, "X (m) ->", 330, 690)
    _label(image, "Z (m) up", 15, 88)
    _label(image, f"Equal metric axis scale; span {extent:.1f} m on X and Z", 720, 160)
    _label(image, "Green KITTI GT", 720, 200, (35, 145, 35))
    _label(image, "Orange offline aligned estimate", 720, 230, (30, 90, 215))
    _label(image, "Alignment fitted within this segment", 720, 270)
    _label(image, "Online estimator did not see GT", 720, 300)
    _write_image(path, image)


def _build_tracker(sequence: KittiGraySequence, config: Dict[str, Any]) -> MonocularTracker:
    orb = config["orb"]
    mapping = dict(config.get("mapping", {}))
    mapping.update(enabled=True, update_descriptors=True)
    bundle = dict(config.get("bundle_adjustment", {}))
    bundle["enabled"] = True
    return MonocularTracker(
        sequence.camera,
        OrbExtractor(orb["nfeatures"], orb["scaleFactor"], orb["nlevels"], orb["fastThreshold"]),
        DescriptorMatcher(**config["matching"]), GeometryConfig(**config["geometry"]),
        TrackingConfig(**config.get("tracking", {})), MappingConfig(**mapping), BundleConfig(**bundle),
        capture_pnp_diagnostics=True)


def _fingerprint_file(path: Path, root: Path) -> Dict[str, Any]:
    return {"path": path.relative_to(root).as_posix(), "sha256": _sha(path), "size_bytes": path.stat().st_size}


def _verify_preflight(root: Path, output: Path) -> Dict[str, Any]:
    preflight = json.loads((output / "frozen_input_hashes.json").read_text(encoding="utf-8"))
    mismatches = []
    for item in preflight["files"]:
        path = root / item["path"]
        if not path.is_file() or _sha(path) != item["sha256"]:
            mismatches.append(item["path"])
    return {"file_count": len(preflight["files"]), "all_sha256_unchanged": not mismatches,
            "mismatches": mismatches, "preflight_manifest_sha256": _sha(output / "frozen_input_hashes.json")}


def _failure_report(frame_id: int, details: Dict[int, Dict[str, Any]],
                    rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    detail = details[frame_id]
    before = details.get(frame_id-1)
    after = next((row for row in rows[frame_id+1:] if row["pose_valid"]), None)
    def summary(value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return None if value is None else value["summary"]
    preceding = summary(before)
    if preceding is None:
        prior = rows[frame_id-1]
        preceding = {"frame_id": prior["frame_id"], "state": prior["state"],
                     "segment_id": prior["segment_id"], "pose_valid": prior["pose_valid"],
                     "stage": "accepted initialization; PnP quantities unavailable",
                     "features_detected": prior["diagnostics"].get("features_detected"),
                     "stored_map_landmarks_after_initialization": prior["diagnostics"].get("map_points")}
    return {"failure_frame": frame_id, "failure": detail, "preceding_tracked_frame": preceding,
            "next_accepted_pose": None if after is None else
            {"frame_id": after["frame_id"], "segment_id": after["segment_id"]},
            "immediate_next_state": rows[frame_id+1]["state"],
            "immediate_next_failure_reason": rows[frame_id+1]["failure_reason"],
            "notes": ["Online frame has no accepted pose", "Raw RANSAC candidate, when present, is rejected evidence only",
                      "Previous-pose visibility is a frozen-pose projection proxy, not a current-frame visibility oracle"]}


def run_diagnostics(root: Path, output: Path) -> Dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    base = root / "artifacts" / "m45a-pnp-diagnostics"
    if output != base and base not in output.parents:
        raise ValueError("M4.5-A outputs must stay under artifacts/m45a-pnp-diagnostics")
    if (output / "equivalence.json").exists():
        raise FileExistsError("Diagnostic evidence already exists; use a fresh output after preserving it")
    preflight = _verify_preflight(root, output)
    if not preflight["all_sha256_unchanged"]:
        raise ValueError("Frozen inputs changed since preflight")
    config_path = root / "configs" / "default.json"
    sequence_path = root / "data" / "sequences" / "00"
    frozen_dir = root / "artifacts" / "m4-kitti-00-ba-validation-final" / "full"
    m5a_path = root / "artifacts" / "m5a-frozen-evaluation" / "full" / "per_frame_errors.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sequence = KittiGraySequence(sequence_path)
    tracker = _build_tracker(sequence, config)
    rows: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []
    details: Dict[int, Dict[str, Any]] = {}
    replay_start = perf_counter()
    for index in range(300):
        sample = sequence[index]
        result = tracker.process(sample)
        record = _plain(result.as_dict())
        rows.append(record)
        if tracker.diagnostic_event is None:
            diagnostics.append({"frame_id": index, "state": record["state"],
                                "segment_id": record["segment_id"], "pose_valid": record["pose_valid"],
                                "failure_reason": record["failure_reason"],
                                "features_detected": record["diagnostics"].get("features_detected"),
                                "stored_map_landmarks": None, "forward_complete_pairs": None,
                                "ratio_passed": None, "mutual_passed_unique_correspondences": None,
                                "pnp_correspondences": None, "pnp_ransac_attempted": False,
                                "pnp_raw_inliers": None, "pnp_raw_inlier_fraction": None})
        else:
            compact, detail = diagnose_pnp_event(tracker.diagnostic_event, record,
                                                 sequence.camera.matrix, sample.image.shape)
            diagnostics.append(compact)
            details[index] = detail
        if (index+1) % 25 == 0:
            print(f"diagnostic replay {index+1}/300, frame {index}", flush=True)
    replay_wall_ms = (perf_counter() - replay_start) * 1000
    equivalence = compare_replay(rows, tracker, frozen_dir)
    equivalence["frozen_input_integrity_before_replay"] = preflight
    _write_json(output / "equivalence.json", equivalence)
    if not equivalence["passed"]:
        raise AssertionError(f"Diagnostic replay differs from frozen M4: {equivalence['mismatches'][:5]}")
    replay_times = np.asarray([row["timing_ms"]["total"] for row in rows], dtype=np.float64)
    frozen_summary = json.loads((frozen_dir / "summary.json").read_text(encoding="utf-8"))
    runtime = {"instrumented_replay_total_ms_median": float(np.median(replay_times)),
               "instrumented_replay_total_ms_p90": float(np.percentile(replay_times, 90)),
               "instrumented_replay_total_ms_sum": float(replay_times.sum()),
               "instrumented_replay_wall_ms_including_diagnostic_extraction": replay_wall_ms,
               "frozen_m4_total_ms_median": frozen_summary["total_runtime_ms_median"],
               "frozen_m4_total_ms_p90": frozen_summary["total_runtime_ms_p90"],
               "frozen_m4_total_ms_sum": frozen_summary["total_runtime_ms_sum"],
               "comparison_limit": "Separate executions; diagnostic extraction and offline plots are not online baseline timing, and machine load differs."}
    _write_json(output / "runtime_overhead.json", runtime)
    with (output / "per_frame_pnp_diagnostics.jsonl").open("w", encoding="utf-8") as stream:
        for row in diagnostics:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    flat = [_flat_record(row) for row in diagnostics]
    _write_csv(output / "per_frame_pnp_diagnostics.csv", flat, list(flat[0]))
    failure_reports = [_failure_report(frame, details, rows) for frame in FAILURE_IDS]
    _write_json(output / "failure_reports.json", failure_reports)
    failure_dir = output / "failures"
    failure_dir.mkdir()
    for report in failure_reports:
        _write_json(failure_dir / f"failure_{report['failure_frame']:03d}.json", report)
    fig = output / "figures"
    fig.mkdir()
    _timeline(rows, fig / "tracking_timeline.png")
    _support(diagnostics, fig / "pnp_support_by_frame.png")
    for frame in FAILURE_IDS:
        _overlay(sequence, frame, details[frame], fig / f"failure_{frame:03d}_correspondences.png")
    for frame in PRIORITY_IDS:
        _pair_comparison(sequence, frame-1, frame, details, rows,
                         fig / f"before_failure_{frame-1:03d}_{frame:03d}.png")
    _residuals(details, fig / "rejected_candidate_residuals.png")
    m5a_records = json.loads(m5a_path.read_text(encoding="utf-8"))
    _trajectory_zoom(m5a_records, 4, 88, 121, fig / "m5a_segment_4_88_121_zoom.png")
    _trajectory_zoom(m5a_records, 5, 178, 201, fig / "m5a_segment_5_178_201_zoom.png")
    integrity = _verify_preflight(root, output)
    if not integrity["all_sha256_unchanged"]:
        raise AssertionError("Frozen inputs changed during diagnostic replay")
    _write_json(output / "frozen_input_integrity_after_replay.json", integrity)
    provenance = {"purpose": "M4.5-A decision-preserving diagnostic replay",
                  "git_head_before_changes": json.loads((output / "frozen_input_hashes.json").read_text(encoding="utf-8")).get("git_head"),
                  "config": _fingerprint_file(config_path, root),
                  "frozen_full_summary": _fingerprint_file(frozen_dir / "summary.json", root),
                  "m5a_per_frame_errors": _fingerprint_file(m5a_path, root),
                  "instrumented_sources": [_fingerprint_file(root / name, root) for name in
                                           ("src/orb_slam_mono/matching.py", "src/orb_slam_mono/tracking.py",
                                            "src/orb_slam_mono/tracking_diagnostics.py")],
                  "sequence_path": sequence_path.relative_to(root).as_posix(),
                  "frame_range_inclusive": [0, 299], "configuration": "full M4: map growth, descriptor maintenance, BA enabled",
                  "python": platform.python_version(), "opencv": cv2.__version__,
                  "numpy": np.__version__, "scipy": scipy.__version__,
                  "equivalence_passed": True, "input_integrity": integrity}
    _write_json(output / "provenance.json", provenance)
    return {"equivalence": equivalence, "failures": [report["failure"]["summary"] for report in failure_reports],
            "figures": [path.name for path in sorted(fig.glob("*.png"))], "integrity": integrity,
            "runtime": runtime}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-subdir", type=str,
                        help="Fresh child directory under artifacts/m45a-pnp-diagnostics for a non-overwriting rerun")
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "artifacts" / "m45a-pnp-diagnostics"
    output = base
    if args.output_subdir is not None:
        name = args.output_subdir
        if not name or not all(char.isalnum() or char in "-_" for char in name):
            parser.error("--output-subdir must be a simple directory name")
        output = base / name
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing diagnostic directory: {output}")
        output.mkdir(parents=True)
        (output / "frozen_input_hashes.json").write_bytes((base / "frozen_input_hashes.json").read_bytes())
    result = run_diagnostics(root, output)
    print(json.dumps({"equivalence_passed": result["equivalence"]["passed"],
                      "figures": result["figures"], "input_integrity": result["integrity"],
                      "runtime": result["runtime"]}, indent=2))


if __name__ == "__main__":
    main()
