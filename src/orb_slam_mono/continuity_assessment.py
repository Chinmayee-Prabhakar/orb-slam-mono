"""Offline, decision-preserving M4.5-D continuity and recovery assessment.

No ground truth is imported. Counterfactual PnP and two-view results are never
written to the online frame stream or applied to a tracker/map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .datasets import KittiGraySequence
from .geometry import initialize_two_view
from .tracking import solve_map_pnp
from .tracking_diagnostics import _build_tracker, compare_replay


LONG_RECOVERY_ANCHORS = (123, 208)
PROBE_OFFSETS = (1, 2, 3)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _point_snapshot(sparse_map: Any) -> list[dict[str, Any]]:
    return [{"id": int(pid), "xyz": p.xyz_world.copy(), "descriptor": p.descriptor.copy(),
             "last_observed": max(p.observations), "observation_count": len(p.observations),
             "evidence_count": len(p.descriptor_evidence),
             "observations": [{"frame_id": fid, "feature_id": feature_id,
                               "pixel_xy": sparse_map.frames[fid].features.keypoints[feature_id].pt}
                              for fid, feature_id in sorted(p.observations.items())]}
            for pid, p in sorted(sparse_map.points.items()) if p.valid]


def _grid_occupancy(xy: np.ndarray, width: int, height: int) -> int:
    if len(xy) == 0:
        return 0
    cells = {(min(3, int(x // (width / 4))), min(2, int(y // (height / 3))))
             for x, y in xy}
    return len(cells)


def _initial_map_record(sparse_map: Any, attempt: dict[str, Any], width: int,
                        height: int) -> dict[str, Any]:
    second = sparse_map.frames[sparse_map.second_frame_id]
    point_list = list(sparse_map.points.values())
    xy = np.array([second.features.keypoints[p.observations[second.frame_id]].pt
                   for p in point_list], dtype=float).reshape(-1, 2)
    xyz = np.array([p.xyz_world for p in point_list], dtype=float).reshape(-1, 3)
    depth = (second.R_cw @ xyz.T).T[:, 2] + second.t_cw[2]
    d = attempt["diagnostics"]
    return {"segment_id": sparse_map.segment_id, "anchor_frame": sparse_map.anchor_frame_id,
            "second_frame": second.frame_id, "initial_points": len(point_list),
            "initial_point_ids": [p.point_id for p in point_list],
            "descriptor_available": sum(p.descriptor.shape == (32,) for p in point_list),
            "initial_observations_per_point": sorted(set(len(p.observations) for p in point_list)),
            "image_grid_4x3_occupied": _grid_occupancy(xy, width, height),
            "image_xy_p10_p90": None if not len(xy) else
            {"x": np.percentile(xy[:, 0], [10, 90]).tolist(),
             "y": np.percentile(xy[:, 1], [10, 90]).tolist()},
            "initial_landmark_pixels": xy.tolist(),
            "second_camera_depth_p10_median_p90_arbitrary": np.percentile(depth, [10, 50, 90]).tolist(),
            "essential_inliers": d.get("essential_inliers"),
            "pose_inliers": d.get("pose_inliers"),
            "pose_inlier_fraction": d.get("pose_inlier_fraction"),
            "median_accepted_parallax_deg": d.get("median_accepted_parallax_deg"),
            "accepted_pose_stability_trials": d.get("pose_stability_accepted_trials")}


def recovery_intervals(rows: list[dict[str, Any]], attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return maximal no-pose runs, retaining original source IDs and nulls."""
    if len({r["frame_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source frame ID")
    intervals = []
    start = None
    for index in range(len(rows) + 1):
        row = rows[index] if index < len(rows) else None
        if row is not None and not row["pose_valid"]:
            if start is None:
                start = index
        elif start is not None:
            subset = rows[start:index]
            first, last = subset[0]["frame_id"], subset[-1]["frame_id"]
            candidate_attempts = [a for a in attempts if first <= a["candidate_frame"] <= last]
            intervals.append({"first_frame": first, "last_frame": last,
                              "processed_frames": len(subset),
                              "next_online_pose_frame": None if row is None else row["frame_id"],
                              "attempt_count": len(candidate_attempts),
                              "anchors": sorted({a["anchor_frame"] for a in candidate_attempts}),
                              "candidate_frames": [a["candidate_frame"] for a in candidate_attempts],
                              "gaps": [a["gap"] for a in candidate_attempts],
                              "rejections": dict(Counter(a["reason"] for a in candidate_attempts
                                                       if a["reason"])),
                              "unscheduled_or_waiting_frames": sum(
                                  r["failure_reason"] in ("waiting_for_initialization_gap",
                                                           "waiting_for_initialization_schedule")
                                  for r in subset)})
            start = None
    return intervals


def offline_map_probe(snapshot: dict[str, Any], features: Any, frame_id: int,
                      matcher: Any, camera: Any, config: Any) -> dict[str, Any]:
    """Global descriptor/PnP probe against a copy of the map as it existed at loss."""
    if frame_id <= snapshot["lost_frame"] or frame_id - snapshot["lost_frame"] not in PROBE_OFFSETS:
        raise ValueError("Probe frame outside predeclared post-loss horizon")
    points = snapshot["points"]
    descriptors = np.stack([p["descriptor"] for p in points]) if points else None
    start = perf_counter()
    matches = matcher.match(descriptors, features.descriptors)
    xyz = np.array([points[m.queryIdx]["xyz"] for m in matches.matches], dtype=float).reshape(-1, 3)
    uv = np.array([features.keypoints[m.trainIdx].pt for m in matches.matches], dtype=float).reshape(-1, 2)
    capture: dict[str, Any] = {}
    cv2.setRNGSeed(19)
    pnp = solve_map_pnp(xyz, uv, camera, config, diagnostic_capture=capture)
    ages = [frame_id - points[m.queryIdx]["last_observed"] for m in matches.matches]
    return {"role": "offline counterfactual, never an accepted online pose",
            "lost_frame": snapshot["lost_frame"], "frame_id": frame_id,
            "segment_id": snapshot["segment_id"], "staleness_frames": frame_id - snapshot["last_pose_frame"],
            "valid_landmarks": len(points), "descriptor_knn_pairs": matches.knn_pairs,
            "ratio_passed": matches.ratio_passed, "unique_correspondences": matches.retained,
            "match_age_p10_median_p90_frames": None if not ages else np.percentile(ages, [10, 50, 90]).tolist(),
            "matches_from_initial_pair": sum(points[m.queryIdx]["observation_count"] == 2
                                             for m in matches.matches),
            "raw_inliers": None if capture.get("raw_inlier_indices") is None else
            len(capture["raw_inlier_indices"]),
            "inlier_fraction": pnp.diagnostics.get("pnp_inlier_fraction"),
            "passes_unchanged_pnp_gates_offline": bool(pnp.accepted), "reason": pnp.reason,
            "positive_depth_fraction": pnp.diagnostics.get("positive_depth_fraction"),
            "median_reprojection_px": pnp.diagnostics.get("median_reprojection_px"),
            "p90_reprojection_px": pnp.diagnostics.get("p90_reprojection_px"),
            "runtime_ms": (perf_counter() - start) * 1000,
            "pairs": [{"point_id": points[m.queryIdx]["id"], "feature_id": int(m.trainIdx),
                       "pixel_xy": uv[i].tolist(), "hamming": float(m.distance)}
                      for i, m in enumerate(matches.matches)]}


def offline_gap_four(anchor: Any, candidate: Any, matcher: Any, camera: Any,
                     geometry_config: Any) -> dict[str, Any]:
    """Test a fixed gap-4 pair using unchanged matching and geometry thresholds."""
    if candidate.frame_id - anchor.frame_id != 4:
        raise ValueError("Only the predeclared gap-4 schedule is permitted")
    start = perf_counter()
    matches = matcher.match(anchor.features.descriptors, candidate.features.descriptors)
    first = np.array([anchor.features.keypoints[m.queryIdx].pt for m in matches.matches],
                     dtype=float).reshape(-1, 2)
    second = np.array([candidate.features.keypoints[m.trainIdx].pt for m in matches.matches],
                      dtype=float).reshape(-1, 2)
    cv2.setRNGSeed(19)
    result = initialize_two_view(first, second, camera, geometry_config)
    return {"role": "offline counterfactual geometry, never an online segment",
            "anchor_frame": anchor.frame_id, "candidate_frame": candidate.frame_id,
            "gap": 4, "matches": matches.retained, "accepted_by_unchanged_geometry_offline":
            result.status == "accepted", "reason": result.failure_reason,
            "diagnostics": result.diagnostics, "runtime_ms": (perf_counter() - start) * 1000}


def run_assessment(root: Path, output: Path) -> dict[str, Any]:
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty output directory; earlier evidence is immutable")
    output.mkdir(parents=True, exist_ok=True)
    frozen = root / "artifacts/m45c-ba-depth-correction/corrected_full"
    sequence = KittiGraySequence(root / "data/sequences/00")
    config_path = root / "configs/default.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protected_paths = [config_path, root / "data/sequences/00/calib.txt",
                       root / "data/sequences/00/times.txt"]
    protected_paths += sorted(frozen.rglob("*"))
    protected_paths += sorted((root / "artifacts/m5a-frozen-evaluation").rglob("*"))
    protected_paths += sorted((root / "data/sequences/00/image_0").glob("*.png"))
    hashes = {p.relative_to(root).as_posix(): _sha(p) for p in protected_paths if p.is_file()}
    _dump(output / "protected_sha256_before.json", hashes)
    tracker = _build_tracker(sequence, config)
    rows: list[dict[str, Any]] = []
    segment_records: list[dict[str, Any]] = []
    loss_snapshots: dict[int, dict[str, Any]] = {}
    diagnostic_events: dict[int, dict[str, Any]] = {}
    gap_trials: list[dict[str, Any]] = []
    scheduled_gap_pairs: list[tuple[Any, int]] = []
    for frame_id in range(300):
        sample = sequence[frame_id]
        # The only schedule counterfactual: two long-recovery anchors at gap 4.
        if (tracker.anchor is not None and tracker.anchor.frame_id in LONG_RECOVERY_ANCHORS
                and frame_id - tracker.anchor.frame_id == 4):
            scheduled_gap_pairs.append((tracker.anchor, frame_id))
        result = tracker.process(sample)
        row = result.as_dict()
        rows.append(row)
        event = tracker.diagnostic_event
        if event is not None and (frame_id in (42, 43, 81, 82, 121, 122, 201, 202, 206, 207, 281, 282)
                                  or (tracker.segments and
                                      event["segment_id"] == tracker.segments[-1].segment_id and
                                      frame_id <= tracker.segments[-1].second_frame_id + 5)):
            diagnostic_events[frame_id] = {"frame_id": frame_id,
                                           "segment_id": event["segment_id"],
                                           "correspondences": [
                                               {"point_id": c["point_id"],
                                                "pixel_xy": c["pixel_xy"].tolist(),
                                                "feature_id": c["image_feature_index"]}
                                               for c in event["correspondences"]]}
        if row["pose_valid"] and "initialization_pair" in row["diagnostics"]:
            attempt = tracker.attempts[-1]
            segment_records.append(_initial_map_record(tracker.active_map, attempt,
                                                       sequence.camera.width, sequence.camera.height))
        if row["state"] == "lost" and tracker.loss_events and tracker.loss_events[-1]["frame_id"] == frame_id:
            sparse_map = tracker.segments[-1]
            loss_snapshots[frame_id] = {
                "lost_frame": frame_id, "segment_id": sparse_map.segment_id,
                "last_pose_frame": tracker.last_successful_pose["frame_id"],
                "points": _point_snapshot(sparse_map)}
    equivalence = compare_replay(rows, tracker, frozen)
    if not equivalence["passed"]:
        _dump(output / "replay_equivalence_failed.json", equivalence)
        raise AssertionError("Diagnostic replay differs from corrected M4.5-C baseline")
    for anchor, frame_id in scheduled_gap_pairs:
        sample = sequence[frame_id]
        candidate = type(anchor).from_sample(sample, tracker.extractor.extract(sample.image))
        gap_trials.append(offline_gap_four(anchor, candidate, tracker.matcher,
                                           tracker.camera, tracker.geometry))
    probes = []
    for lost_frame, snapshot in loss_snapshots.items():
        next_online = next((r["frame_id"] for r in rows[lost_frame + 1:] if r["pose_valid"]), 300)
        for offset in PROBE_OFFSETS:
            frame_id = lost_frame + offset
            if frame_id >= min(300, next_online):
                break
            features = tracker.extractor.extract(sequence[frame_id].image)
            probes.append(offline_map_probe(snapshot, features, frame_id, tracker.matcher,
                                            tracker.camera, tracker.config))
    # Independent two-source optical-flow labels are sparse and are evaluated
    # only on the two gate-passing recovery proposals; unknown remains unknown.
    from .association_audit import estimate_track_identity, label_association
    for probe in probes:
        if not probe["passes_unchanged_pnp_gates_offline"]:
            continue
        snapshot = loss_snapshots[probe["lost_frame"]]
        frame_id = probe["frame_id"]
        target = sequence[frame_id].image
        features = tracker.extractor.extract(target)
        xy = np.asarray([kp.pt for kp in features.keypoints], dtype=float).reshape(-1, 2)
        obs = [{"point_id": point["id"], "observations": point["observations"]}
               for point in snapshot["points"]]
        sources = {o["frame_id"] for point in obs for o in point["observations"]
                   if 0 < frame_id - o["frame_id"] <= 12}
        images = {source: sequence[source].image for source in sorted(sources)}
        claims = estimate_track_identity(obs, images, target, xy, frame_id)
        labels = Counter(label_association(pair["point_id"], pair["feature_id"], claims, xy)
                         for pair in probe["pairs"])
        probe["independent_two_track_labels"] = dict(labels)
        probe["supported_point_count"] = sum(c["status"] == "supported" for c in claims.values())
    by_segment = {r["segment_id"]: r for r in segment_records}
    for segment in tracker.segments:
        info = by_segment[segment.segment_id]
        initial_ids = set(info["initial_point_ids"])
        valid = [r for r in rows if r["segment_id"] == segment.segment_id and r["pose_valid"]]
        info["valid_pose_count"] = len(valid)
        info["last_valid_frame"] = valid[-1]["frame_id"]
        loss = next((e for e in tracker.loss_events if e["segment_id"] == segment.segment_id), None)
        info["loss_frame"] = None if loss is None else loss["frame_id"]
        info["first_pnp_frame"] = segment.second_frame_id + 1
        info["first_pnp_support"] = (rows[info["first_pnp_frame"]]["diagnostics"]
                                     if info["first_pnp_frame"] < len(rows) else None)
        info["keyframe_ids"] = sorted(segment.keyframes)
        info["new_map_points"] = len(segment.points) - info["initial_points"]
        info["final_point_observations_p10_median_p90"] = np.percentile(
            [p.observation_count for p in segment.points.values()], [10, 50, 90]).tolist()
        info["final_descriptor_evidence_p10_median_p90"] = np.percentile(
            [len(p.descriptor_evidence) for p in segment.points.values()], [10, 50, 90]).tolist()
        info["first_five_support"] = [
            {"frame_id": r["frame_id"], "map_points": r["diagnostics"].get("map_points"),
             "unique_pairs": r["diagnostics"].get("unique_map_associations"),
             "initial_landmark_pairs": sum(c["point_id"] in initial_ids for c in
                                           diagnostic_events.get(r["frame_id"], {}).get("correspondences", [])),
             "initial_landmark_accepted_observations": sum(
                 pid in initial_ids for pid in segment.frames[r["frame_id"]].feature_to_point.values())
             if r["frame_id"] in segment.frames else 0,
             "pnp_inliers": r["diagnostics"].get("pnp_inliers"),
             "pnp_status": r["diagnostics"].get("pnp_status"),
             "new_points": r["diagnostics"].get("mapping", {}).get("new_points")}
            for r in rows[segment.second_frame_id+1:min(len(rows), segment.second_frame_id+6)]
            if r["segment_id"] == segment.segment_id]
        info["initial_landmarks_reobserved_after_initialization"] = sum(
            any(fid > segment.second_frame_id for fid in segment.points[pid].observations)
            for pid in initial_ids)
    attempts = [{"anchor_frame": a["anchor_frame"], "candidate_frame": a["candidate_frame"],
                 "gap": a["gap"], "matches": a["matches"], "status": a["initialization_status"],
                 "reason": a["failure_reason"], "diagnostics": a["diagnostics"]}
                for a in tracker.attempts]
    intervals = recovery_intervals(rows, attempts)
    for name, value in (("segment_analysis.json", segment_records), ("recovery_intervals.json", intervals),
                        ("initialization_attempts_summary.json", attempts),
                        ("existing_map_probes.json", probes), ("gap_four_probes.json", gap_trials),
                        ("replay_equivalence.json", equivalence)):
        _dump(output / name, value)
    _dump(output / "selected_frame_correspondences.json", diagnostic_events)
    current = {p.relative_to(root).as_posix(): _sha(p) for p in protected_paths if p.is_file()}
    preservation = {"protected_file_count": len(hashes), "all_sha256_unchanged": hashes == current,
                    "changed": [path for path in hashes if hashes[path] != current.get(path)]}
    _dump(output / "protected_integrity_after.json", preservation)
    if not preservation["all_sha256_unchanged"]:
        raise AssertionError("Protected baseline input changed")
    return {"segments": segment_records, "intervals": intervals, "attempts": attempts,
            "existing_map_probes": probes, "gap_four_probes": gap_trials,
            "rows": rows, "events": diagnostic_events,
            "preservation": preservation, "equivalence": equivalence}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_assessment(args.root, args.output)
    print(json.dumps({"segments": len(result["segments"]), "attempts": len(result["attempts"]),
                      "map_probes": len(result["existing_map_probes"]),
                      "gap_four_probes": len(result["gap_four_probes"]),
                      "replay_equal": result["equivalence"]["passed"]}))


if __name__ == "__main__":
    main()
