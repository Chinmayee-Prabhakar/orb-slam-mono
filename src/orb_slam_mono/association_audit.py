"""Offline association identity audit; never feeds pairs or poses to SLAM.

The production matcher, tracker and M5-A evaluator are imported only as
read-only sources. Counterfactual pairs and PnP candidates stay in this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .datasets import KittiGraySequence
from .features import OrbExtractor
from .tracking import TrackingConfig, solve_map_pnp
from .tracking_diagnostics import _build_tracker, _plain, compare_replay


PRIORITY = (43, 207, 82, 282)
PREDECESSORS = (42, 206, 81, 281)
SECONDARY = (8, 20, 122, 202)
TARGETS = tuple(sorted(set(PRIORITY + PREDECESSORS + SECONDARY)))
RADII_PX = (8.0, 16.0, 32.0)
RATIO = 0.75
MAX_SOURCE_GAP = 12
MAX_SOURCE_OBSERVATIONS = 3
FLOW_FB_MAX_PX = 1.0
PATCH_NCC_MIN = 0.80
TARGET_FEATURE_RADIUS_PX = 3.0
WRONG_FEATURE_SEPARATION_PX = 5.0


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _hamming(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.uint8).reshape(32)
    second = np.asarray(second, dtype=np.uint8).reshape(-1, 32)
    return np.unpackbits(np.bitwise_xor(second, first), axis=1).sum(axis=1)


def _project_points(points: Sequence[Dict[str, Any]], prior_pose: Dict[str, Any],
                    camera_matrix: np.ndarray, image_shape: Tuple[int, int],
                    pixel_bias: Tuple[float, float] = (0.0, 0.0)) -> Dict[int, Dict[str, Any]]:
    """Use only the previous accepted estimated T_cw; never KITTI truth."""
    R = np.asarray(prior_pose["T_cw_R"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(prior_pose["T_cw_t"], dtype=np.float64).reshape(3)
    height, width = image_shape
    predictions = {}
    for point in points:
        if point.get("segment_id") != prior_pose.get("segment_id"):
            raise ValueError("Counterfactual association cannot cross map segments")
        xyz = np.asarray(point["xyz_world"], dtype=np.float64).reshape(3)
        camera_xyz = R @ xyz + t
        depth = float(camera_xyz[2])
        if not np.isfinite(camera_xyz).all() or depth <= 1e-8:
            predictions[point["point_id"]] = {"status": "behind_or_invalid", "pixel_xy": None,
                                               "depth": depth}
            continue
        u = camera_matrix[0, 0]*camera_xyz[0]/depth + camera_matrix[0, 2] + pixel_bias[0]
        v = camera_matrix[1, 1]*camera_xyz[1]/depth + camera_matrix[1, 2] + pixel_bias[1]
        pixel = [float(u), float(v)]
        status = "in_image" if 0 <= u < width and 0 <= v < height else "outside_image"
        predictions[point["point_id"]] = {"status": status, "pixel_xy": pixel, "depth": depth}
    return predictions


def projected_associations(points: Sequence[Dict[str, Any]], feature_xy: np.ndarray,
                           feature_descriptors: np.ndarray, prior_pose: Dict[str, Any],
                           camera_matrix: np.ndarray, image_shape: Tuple[int, int],
                           radius_px: float, pixel_bias: Tuple[float, float] = (0.0, 0.0),
                           ratio_threshold: float = RATIO) -> Dict[str, Any]:
    """Counterfactual local Hamming KNN + ratio + mutual best, without state mutation."""
    if radius_px <= 0 or not 0 < ratio_threshold < 1:
        raise ValueError("Invalid counterfactual matching threshold")
    xy = np.asarray(feature_xy, dtype=np.float64).reshape(-1, 2)
    descriptors = np.asarray(feature_descriptors, dtype=np.uint8).reshape(-1, 32)
    if len(xy) != len(descriptors):
        raise ValueError("Feature coordinates and descriptors differ")
    predictions = _project_points(points, prior_pose, camera_matrix, image_shape, pixel_bias)
    tree = cKDTree(xy) if len(xy) else None
    forward = []
    ratios = []
    feature_candidates = defaultdict(list)
    for point in points:
        point_id = point["point_id"]
        prediction = predictions[point_id]
        if prediction["status"] != "in_image" or tree is None:
            continue
        nearby = sorted(tree.query_ball_point(prediction["pixel_xy"], radius_px))
        for feature_id in nearby:
            distance = int(_hamming(np.asarray(point["descriptor"], dtype=np.uint8),
                                    descriptors[feature_id:feature_id+1])[0])
            feature_candidates[feature_id].append((distance, point_id))
        if len(nearby) < 2:
            continue
        distances = _hamming(np.asarray(point["descriptor"], dtype=np.uint8), descriptors[nearby])
        ranked = sorted(zip(distances.tolist(), nearby), key=lambda item: (item[0], item[1]))
        best_distance, feature_id = ranked[0]
        second_distance = ranked[1][0]
        ratio = None if second_distance == 0 else float(best_distance/second_distance)
        pixel = xy[feature_id]
        entry = {"point_id": point_id, "feature_id": int(feature_id),
                 "hamming": int(best_distance), "second_hamming": int(second_distance),
                 "ratio": ratio, "radius_px": float(radius_px),
                 "projected_pixel_xy": prediction["pixel_xy"], "pixel_xy": pixel.tolist(),
                 "projection_error_px": float(np.linalg.norm(pixel-prediction["pixel_xy"])),
                 "local_candidate_count": len(nearby), "ratio_pass": bool(best_distance < ratio_threshold*second_distance)}
        forward.append(entry)
        if entry["ratio_pass"]:
            ratios.append(entry)
    reverse_best = {feature_id: min(values)[1] for feature_id, values in feature_candidates.items()}
    retained = [entry for entry in ratios if reverse_best.get(entry["feature_id"]) == entry["point_id"]]
    retained.sort(key=lambda item: (item["hamming"], item["point_id"], item["feature_id"]))
    assert len({entry["point_id"] for entry in retained}) == len(retained)
    assert len({entry["feature_id"] for entry in retained}) == len(retained)
    return {"radius_px": float(radius_px), "pixel_bias": list(pixel_bias),
            "projected_in_image": sum(value["status"] == "in_image" for value in predictions.values()),
            "projected_behind_or_invalid": sum(value["status"] == "behind_or_invalid" for value in predictions.values()),
            "projected_outside_image": sum(value["status"] == "outside_image" for value in predictions.values()),
            "forward_complete_pairs": len(forward), "ratio_passed": len(ratios),
            "mutual_passed": len(retained), "forward": forward,
            "retained": retained, "predictions": predictions}


def _patch_ncc(source: np.ndarray, source_xy: Sequence[float],
               target: np.ndarray, target_xy: Sequence[float]) -> Optional[float]:
    h0, w0 = source.shape
    h1, w1 = target.shape
    x0, y0 = source_xy
    x1, y1 = target_xy
    if not (6 <= x0 < w0-6 and 6 <= y0 < h0-6 and 6 <= x1 < w1-6 and 6 <= y1 < h1-6):
        return None
    a = cv2.getRectSubPix(source, (11, 11), (float(x0), float(y0))).astype(np.float64).reshape(-1)
    b = cv2.getRectSubPix(target, (11, 11), (float(x1), float(y1))).astype(np.float64).reshape(-1)
    a -= a.mean()
    b -= b.mean()
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return None if norm < 1e-8 else float(np.dot(a, b)/norm)


def estimate_track_identity(points: Sequence[Dict[str, Any]], source_images: Dict[int, np.ndarray],
                            target_image: np.ndarray, target_feature_xy: np.ndarray,
                            target_frame_id: int) -> Dict[int, Dict[str, Any]]:
    """Two-source image-track evidence independent of tested descriptor/PnP pairs."""
    target_xy = np.asarray(target_feature_xy, dtype=np.float64).reshape(-1, 2)
    tree = cKDTree(target_xy) if len(target_xy) else None
    groups = defaultdict(list)
    for point in points:
        collected = 0
        for observation in sorted(point["observations"], key=lambda row: row["frame_id"], reverse=True):
            source_id = observation["frame_id"]
            if source_id >= target_frame_id or target_frame_id-source_id > MAX_SOURCE_GAP:
                continue
            if source_id not in source_images:
                continue
            groups[source_id].append((point["point_id"], observation))
            if sum(point_id == point["point_id"] for point_id, _ in groups[source_id]) > 1:
                raise ValueError("Duplicate map point observation in a source frame")
            collected += 1
            if collected >= MAX_SOURCE_OBSERVATIONS:
                break
    evidence = defaultdict(list)
    settings = {"winSize": (21, 21), "maxLevel": 3,
                "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)}
    for source_id, rows in sorted(groups.items()):
        source_image = source_images[source_id]
        original = np.asarray([row[1]["pixel_xy"] for row in rows], dtype=np.float32).reshape(-1, 1, 2)
        predicted, ok_forward, _ = cv2.calcOpticalFlowPyrLK(source_image, target_image, original, None, **settings)
        if predicted is None:
            predicted = np.full_like(original, np.nan)
            ok_forward = np.zeros((len(rows), 1), dtype=np.uint8)
        safe_predicted = np.where(np.isfinite(predicted), predicted, original).astype(np.float32)
        backward, ok_back, _ = cv2.calcOpticalFlowPyrLK(target_image, source_image, safe_predicted, None, **settings)
        if backward is None:
            backward = np.full_like(original, np.nan)
            ok_back = np.zeros((len(rows), 1), dtype=np.uint8)
        for index, (point_id, observation) in enumerate(rows):
            pixel = predicted[index, 0].astype(float)
            fb = float(np.linalg.norm(backward[index, 0]-original[index, 0]))
            ncc = _patch_ncc(source_image, observation["pixel_xy"], target_image, pixel)
            usable = (bool(ok_forward[index, 0]) and bool(ok_back[index, 0]) and
                      np.isfinite(pixel).all() and fb <= FLOW_FB_MAX_PX and
                      ncc is not None and ncc >= PATCH_NCC_MIN and tree is not None)
            feature_id = None
            nearest_distance = None
            if usable:
                distances, indices = tree.query(pixel, k=min(2, len(target_xy)))
                distances = np.atleast_1d(distances)
                indices = np.atleast_1d(indices)
                nearest_distance = float(distances[0])
                if nearest_distance <= TARGET_FEATURE_RADIUS_PX and (len(distances) == 1 or distances[1] > TARGET_FEATURE_RADIUS_PX):
                    feature_id = int(indices[0])
            evidence[point_id].append({"source_frame_id": source_id,
                                       "source_feature_id": observation["feature_id"],
                                       "source_pixel_xy": observation["pixel_xy"],
                                       "tracked_pixel_xy": pixel.tolist() if np.isfinite(pixel).all() else None,
                                       "forward_backward_error_px": fb if np.isfinite(fb) else None,
                                       "patch_ncc": ncc, "nearest_target_feature_id": feature_id,
                                       "nearest_target_distance_px": nearest_distance,
                                       "usable": bool(usable and feature_id is not None)})
    claims = {}
    for point in points:
        point_id = point["point_id"]
        tracks = evidence[point_id]
        usable = [row for row in tracks if row["usable"]]
        status = "not_evaluable" if not usable else "uncertain"
        feature_id = None
        if (len(usable) >= 2 and len({row["source_frame_id"] for row in usable}) >= 2 and
                len({row["nearest_target_feature_id"] for row in usable}) == 1):
            locations = np.asarray([row["tracked_pixel_xy"] for row in usable], dtype=float)
            distances = np.linalg.norm(locations[:, None, :]-locations[None, :, :], axis=2)
            if float(distances.max()) <= TARGET_FEATURE_RADIUS_PX:
                status = "supported"
                feature_id = usable[0]["nearest_target_feature_id"]
        claims[point_id] = {"status": status, "supported_feature_id": feature_id,
                            "tracks": tracks, "usable_track_count": len(usable)}
    owners = defaultdict(list)
    for point_id, value in claims.items():
        if value["status"] == "supported":
            owners[value["supported_feature_id"]].append(point_id)
    for feature_id, point_ids in owners.items():
        if len(point_ids) > 1:
            for point_id in point_ids:
                claims[point_id]["status"] = "uncertain"
                claims[point_id]["ambiguity"] = f"feature_claimed_by_{len(point_ids)}_points"
                claims[point_id]["supported_feature_id"] = None
    return claims


def label_association(point_id: int, feature_id: int, claims: Dict[int, Dict[str, Any]],
                      feature_xy: np.ndarray) -> str:
    claim = claims[point_id]
    if claim["status"] == "not_evaluable":
        return "not_evaluable"
    if claim["status"] != "supported":
        return "uncertain"
    supported = claim["supported_feature_id"]
    if feature_id == supported:
        return "supported_correct"
    separation = float(np.linalg.norm(np.asarray(feature_xy[feature_id])-np.asarray(feature_xy[supported])))
    return "supported_incorrect" if separation >= WRONG_FEATURE_SEPARATION_PX else "uncertain"


def association_metrics(pairs: Sequence[Dict[str, Any]], claims: Dict[int, Dict[str, Any]],
                        feature_xy: np.ndarray, width: int, height: int) -> Dict[str, Any]:
    labeled = []
    for pair in pairs:
        label = label_association(pair["point_id"], pair["feature_id"], claims, feature_xy)
        labeled.append({**pair, "identity_label": label})
    counts = Counter(row["identity_label"] for row in labeled)
    correct, incorrect = counts["supported_correct"], counts["supported_incorrect"]
    supported_points = sum(value["status"] == "supported" for value in claims.values())
    cells = {(min(3, max(0, int(feature_xy[row["feature_id"], 0]*4/width))),
              min(3, max(0, int(feature_xy[row["feature_id"], 1]*4/height)))) for row in labeled}
    return {"pair_count": len(labeled), "supported_correct": correct,
            "supported_incorrect": incorrect, "uncertain": counts["uncertain"],
            "not_evaluable": counts["not_evaluable"],
            "labeled_precision": None if correct+incorrect == 0 else correct/(correct+incorrect),
            "partial_recall": None if supported_points == 0 else correct/supported_points,
            "supported_point_denominator": supported_points,
            "occupied_4x4_cells": len(cells), "pairs": labeled}


def _snapshot_map(tracker: Any, frame_id: int) -> Optional[Dict[str, Any]]:
    sparse_map = tracker.active_map
    if sparse_map is None:
        return None
    points = []
    for point_id, point in sorted(sparse_map.points.items()):
        if not point.valid:
            continue
        observations = []
        for source_id, feature_id in sorted(point.observations.items()):
            if source_id >= frame_id:
                raise AssertionError("Future map observation would leak into identity audit")
            source = sparse_map.frames[source_id]
            observations.append({"frame_id": int(source_id), "feature_id": int(feature_id),
                                 "pixel_xy": list(map(float, source.features.keypoints[feature_id].pt))})
        points.append({"point_id": int(point_id), "segment_id": int(sparse_map.segment_id),
                       "xyz_world": point.xyz_world.tolist(), "descriptor": point.descriptor.tolist(),
                       "descriptor_evidence_hex": [value.tobytes().hex() for value in point.descriptor_evidence],
                       "observations": observations})
    prior = tracker.last_successful_pose
    prior_pose = None if prior is None else {key: prior[key] for key in
                                             ("frame_id", "segment_id", "T_cw_R", "T_cw_t")}
    return {"frame_id": frame_id, "segment_id": sparse_map.segment_id,
            "prior_pose": _plain(prior_pose), "points": points}


def _baseline_pairs(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"point_id": int(row["point_id"]),
             "feature_id": int(row["image_feature_index"]),
             "hamming": float(row["hamming"]), "pixel_xy": row["pixel_xy"]}
            for row in event["correspondences"]]


def _offline_pnp(pairs: Sequence[Dict[str, Any]], point_lookup: Dict[int, Dict[str, Any]],
                 feature_xy: np.ndarray, camera: Any, config: TrackingConfig,
                 seed: int) -> Dict[str, Any]:
    xyz = np.asarray([point_lookup[row["point_id"]]["xyz_world"] for row in pairs], dtype=float).reshape(-1, 3)
    uv = np.asarray([feature_xy[row["feature_id"]] for row in pairs], dtype=float).reshape(-1, 2)
    cv2.setRNGSeed(seed)
    capture: Dict[str, Any] = {}
    result = solve_map_pnp(xyz, uv, camera, config, diagnostic_capture=capture)
    return {"role": "offline counterfactual geometric support only; never published",
            "input_pairs": len(pairs), "passes_unchanged_pnp_gates_offline": bool(result.accepted),
            "reason": result.reason, "raw_inliers": None if capture.get("raw_inlier_indices") is None
            else int(len(capture["raw_inlier_indices"])),
            "raw_inlier_fraction": None if capture.get("raw_inlier_indices") is None or not pairs
            else float(len(capture["raw_inlier_indices"])/len(pairs)),
            "diagnostics": _plain(result.diagnostics)}


def _check_preflight(root: Path, output: Path) -> Dict[str, Any]:
    manifest = json.loads((root / "artifacts" / "m45b1-association-audit" / "preflight_sha256.json").read_text(encoding="utf-8"))
    changed = []
    for item in manifest["files"]:
        if item["path"] in ("STATUS.md", "NOTES.md"):
            continue
        path = root / item["path"]
        if not path.is_file() or _sha(path) != item["sha256"]:
            changed.append(item["path"])
    m45a = root / "artifacts" / "m45a-pnp-diagnostics"
    image_manifest = json.loads((m45a / "image_sha256.json").read_text(encoding="utf-8"))
    image_changed = []
    for item in image_manifest["files"]:
        path = root / item["path"]
        if not path.is_file() or _sha(path) != item["sha256"]:
            image_changed.append(item["path"])
    return {"protected_file_count": manifest["file_count"]-2,
            "protected_hash_mismatches": changed, "image_count": len(image_manifest["files"]),
            "image_hash_mismatches": image_changed,
            "m45a_equivalence_passed": json.loads((m45a / "equivalence.json").read_text(encoding="utf-8"))["passed"]}


def run_audit(root: Path, output: Path) -> Dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    base = root / "artifacts" / "m45b1-association-audit"
    if output != base and output.parent != base:
        raise ValueError("Write M4.5-B1 evidence only under artifacts/m45b1-association-audit")
    if output.exists() and any(output.iterdir()) and output != base:
        raise FileExistsError("Audit rerun directory exists; preserve it and choose a new name")
    if (output / "audit_summary.json").exists():
        raise FileExistsError("Audit evidence exists; preserve it before a new run")
    preflight = _check_preflight(root, output)
    if preflight["protected_hash_mismatches"] or preflight["image_hash_mismatches"] or not preflight["m45a_equivalence_passed"]:
        raise ValueError("Accepted input evidence changed")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / "configs" / "default.json").read_text(encoding="utf-8"))
    sequence = KittiGraySequence(root / "data" / "sequences" / "00")
    tracker = _build_tracker(sequence, config)
    frozen_dir = root / "artifacts" / "m4-kitti-00-ba-validation-final" / "full"
    snapshots: Dict[int, Optional[Dict[str, Any]]] = {}
    events: Dict[int, Dict[str, Any]] = {}
    rows = []
    for frame_id in range(300):
        if frame_id in TARGETS:
            snapshots[frame_id] = _snapshot_map(tracker, frame_id)
        result = tracker.process(sequence[frame_id])
        record = _plain(result.as_dict())
        rows.append(record)
        if frame_id in TARGETS and tracker.diagnostic_event is not None:
            events[frame_id] = _plain(tracker.diagnostic_event)
        if (frame_id+1) % 50 == 0:
            print(f"unchanged audit replay {frame_id+1}/300", flush=True)
    equivalence = compare_replay(rows, tracker, frozen_dir)
    _write_json(output / "replay_equivalence.json", equivalence)
    if not equivalence["passed"]:
        raise AssertionError("Audit replay differs from corrected M4")
    if events.get(206) is not None or rows[206]["diagnostics"].get("initialization_pair") is None:
        raise AssertionError("Frame 206 must remain an initialization view without PnP")
    feature_extractor = OrbExtractor(config["orb"]["nfeatures"], config["orb"]["scaleFactor"],
                                     config["orb"]["nlevels"], config["orb"]["fastThreshold"])
    source_images: Dict[int, np.ndarray] = {}
    summaries = []
    details = []
    identity_rows = []
    point_histories = []
    for frame_id in TARGETS:
        if frame_id == 206:
            summaries.append({"frame_id": 206, "stage": "accepted initialization", "pnp_available": False,
                              "pose_valid": True, "segment_id": rows[206]["segment_id"],
                              "reason": "No map PnP associations exist at the second initialization view"})
            continue
        event = events.get(frame_id)
        snapshot = snapshots.get(frame_id)
        if event is None or snapshot is None:
            raise AssertionError(f"Expected PnP tracking event at frame {frame_id}")
        if snapshot["segment_id"] != event["segment_id"] or snapshot["prior_pose"]["segment_id"] != event["segment_id"]:
            raise AssertionError("Cross-segment association leak")
        image = sequence[frame_id].image
        features = feature_extractor.extract(image)
        xy = np.asarray([keypoint.pt for keypoint in features.keypoints], dtype=np.float64).reshape(-1, 2)
        if features.descriptors is None or not np.array_equal(xy, np.asarray(event["keypoints_xy"])):
            raise AssertionError(f"Offline ORB re-extraction differs from saved online keypoints at frame {frame_id}")
        points = snapshot["points"]
        if [row["point_id"] for row in points] != event["candidate_point_ids"]:
            raise AssertionError("Snapshot point IDs differ from online association order")
        point_lookup = {point["point_id"]: point for point in points}
        for point in points:
            for obs in point["observations"]:
                source_id = obs["frame_id"]
                if frame_id-source_id <= MAX_SOURCE_GAP and source_id not in source_images:
                    source_images[source_id] = sequence[source_id].image
        claims = estimate_track_identity(points, source_images, image, xy, frame_id)
        supported = {point_id: value["supported_feature_id"] for point_id, value in claims.items()
                     if value["status"] == "supported"}
        baseline = _baseline_pairs(event)
        if len(baseline) != rows[frame_id]["diagnostics"].get("unique_map_associations"):
            raise AssertionError("Baseline pair count differs from online diagnostics")
        baseline_metrics = association_metrics(baseline, claims, xy, sequence.camera.width, sequence.camera.height)
        baseline_metrics["method"] = "frozen_global_descriptor"
        baseline_metrics["forward_complete_pairs"] = event["matcher"]["forward_complete_pairs"]
        baseline_metrics["ratio_passed"] = event["matcher"]["ratio_passed"]
        baseline_metrics["offline_pnp"] = _offline_pnp(baseline, point_lookup, xy, sequence.camera,
                                                       TrackingConfig(**config["tracking"]), 9000+frame_id)
        projected_results = {}
        for radius in RADII_PX:
            local = projected_associations(points, xy, features.descriptors, snapshot["prior_pose"],
                                           sequence.camera.matrix, image.shape, radius)
            metrics = association_metrics(local["retained"], claims, xy, sequence.camera.width, sequence.camera.height)
            metrics.update({key: local[key] for key in ("projected_in_image", "projected_behind_or_invalid",
                                                       "projected_outside_image", "forward_complete_pairs",
                                                       "ratio_passed", "mutual_passed")})
            metrics["method"] = f"projected_{int(radius)}px"
            metrics["offline_pnp"] = _offline_pnp(local["retained"], point_lookup, xy, sequence.camera,
                                                   TrackingConfig(**config["tracking"]), 9000+frame_id)
            projected_results[str(int(radius))] = {"summary": {key: value for key, value in metrics.items() if key != "pairs"},
                                                   "pairs": metrics["pairs"], "forward": local["forward"],
                                                   "predictions": local["predictions"]}
        stress = projected_associations(points, xy, features.descriptors, snapshot["prior_pose"],
                                        sequence.camera.matrix, image.shape, 16.0, pixel_bias=(50.0, 0.0))
        stress_metrics = association_metrics(stress["retained"], claims, xy, sequence.camera.width, sequence.camera.height)
        global_candidates = []
        for candidate in event["matcher"]["candidates"]:
            point_id = event["candidate_point_ids"][candidate["map_descriptor_index"]]
            feature_id = candidate["image_feature_index"]
            global_candidates.append({"point_id": point_id, "feature_id": feature_id,
                                      "best_hamming": candidate["best_hamming"],
                                      "second_hamming": candidate["second_hamming"],
                                      "ratio": None if candidate["second_hamming"] == 0 else
                                      candidate["best_hamming"]/candidate["second_hamming"],
                                      "ratio_pass": candidate["ratio_pass"],
                                      "mutual_pass": candidate["mutual_pass"],
                                      "identity_label": label_association(point_id, feature_id, claims, xy)})
        statuses = Counter(value["status"] for value in claims.values())
        prior_predictions = _project_points(points, snapshot["prior_pose"], sequence.camera.matrix, image.shape)
        errors = [float(np.linalg.norm(xy[feature_id]-prior_predictions[point_id]["pixel_xy"]))
                  for point_id, feature_id in supported.items()
                  if prior_predictions[point_id]["status"] == "in_image"]
        summary = {"frame_id": frame_id, "segment_id": event["segment_id"], "state": rows[frame_id]["state"],
                   "pose_valid": rows[frame_id]["pose_valid"], "failure_reason": rows[frame_id]["failure_reason"],
                   "prior_accepted_frame": snapshot["prior_pose"]["frame_id"],
                   "active_landmarks": len(points), "orb_features": len(xy),
                   "identity_claim_counts": dict(statuses),
                   "prior_projection_error_on_supported_points_px": {
                       "count": len(errors), "median": None if not errors else float(np.median(errors)),
                       "p90": None if not errors else float(np.percentile(errors, 90)),
                       "over_8px": sum(error > 8 for error in errors),
                       "over_16px": sum(error > 16 for error in errors),
                       "over_32px": sum(error > 32 for error in errors)},
                   "baseline": {key: value for key, value in baseline_metrics.items() if key != "pairs"},
                   "projected": {radius: data["summary"] for radius, data in projected_results.items()},
                   "biased_16px": {key: value for key, value in stress_metrics.items() if key != "pairs"}}
        summaries.append(summary)
        details.append({"frame_id": frame_id, "baseline_pairs": baseline_metrics["pairs"],
                        "global_candidates": global_candidates, "projected": projected_results,
                        "biased_16px_pairs": stress_metrics["pairs"], "claims": claims})
        identity_rows.extend({"frame_id": frame_id, "segment_id": event["segment_id"],
                              "point_id": point_id, **value} for point_id, value in sorted(claims.items()))
        if frame_id in PRIORITY:
            point_histories.append({"frame_id": frame_id, "segment_id": event["segment_id"],
                                    "points": points})
        print(f"identity audit frame {frame_id}: {len(points)} points, {len(supported)} two-track claims", flush=True)
    _write_json(output / "audit_summary.json", summaries)
    with (output / "association_details.jsonl").open("w", encoding="utf-8") as stream:
        for value in details:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    with (output / "identity_evidence.jsonl").open("w", encoding="utf-8") as stream:
        for value in identity_rows:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    with (output / "point_history_priority.jsonl").open("w", encoding="utf-8") as stream:
        for value in point_histories:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    flat = []
    for row in summaries:
        if row["frame_id"] == 206:
            continue
        for method, result in [("baseline", row["baseline"])] + [(f"projected_{radius}px", value)
                                                              for radius, value in row["projected"].items()]:
            flat.append({"frame_id": row["frame_id"], "segment_id": row["segment_id"],
                         "method": method, "active_landmarks": row["active_landmarks"],
                         "supported_point_denominator": result["supported_point_denominator"],
                         **{key: result.get(key) for key in
                            ("forward_complete_pairs", "ratio_passed", "pair_count", "supported_correct",
                             "supported_incorrect", "uncertain", "not_evaluable", "labeled_precision",
                             "partial_recall", "occupied_4x4_cells")},
                         "offline_pnp_raw_inliers": result["offline_pnp"]["raw_inliers"],
                         "offline_pnp_passes_gates": result["offline_pnp"]["passes_unchanged_pnp_gates_offline"]})
    with (output / "method_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    postflight = _check_preflight(root, output)
    _write_json(output / "accepted_input_integrity_after.json", postflight)
    if postflight["protected_hash_mismatches"] or postflight["image_hash_mismatches"]:
        raise AssertionError("Accepted evidence changed during audit")
    return {"summaries": summaries, "details": details, "equivalence": equivalence,
            "input_integrity": postflight}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-subdir", type=str,
                        help="Fresh single-component rerun name inside artifacts/m45b1-association-audit")
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "artifacts" / "m45b1-association-audit"
    if args.output_subdir and not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.output_subdir):
        parser.error("--output-subdir must be a simple lowercase name")
    result = run_audit(root, base / args.output_subdir if args.output_subdir else base)
    print(json.dumps({"replay_equivalent": result["equivalence"]["passed"],
                      "frames_audited": len(result["summaries"]),
                      "priority_labels": {str(row["frame_id"]): row["identity_claim_counts"]
                                          for row in result["summaries"] if row["frame_id"] in PRIORITY},
                      "input_integrity": result["input_integrity"]}, indent=2))


if __name__ == "__main__":
    main()
