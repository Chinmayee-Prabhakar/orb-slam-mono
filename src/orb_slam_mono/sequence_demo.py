"""M3 sequential ORB/map/PnP experiment and arbitrary-scale exports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .datasets import KittiGraySequence
from .features import OrbExtractor
from .geometry import GeometryConfig
from .matching import DescriptorMatcher
from .synthetic_sequence import generate_sequence
from .tracking import MonocularTracker, TrackingConfig
from .local_mapping import MappingConfig
from .bundle_adjustment import BundleConfig
from .map_state import SparseMap


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _save_visual(event: dict[str, Any], path: Path, limit: int = 120) -> None:
    if event["kind"] == "initialization":
        left, right = event["first_image"], event["second_image"]
        canvas = np.hstack((cv2.cvtColor(left, cv2.COLOR_GRAY2BGR),
                            cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)))
        mask = event["accepted_mask"] if event["status"] == "accepted" else event["essential_mask"]
        for index, match in enumerate(event["matches"][:limit]):
            x1, y1 = map(int, np.rint(event["first_keypoints"][match.queryIdx].pt))
            x2, y2 = map(int, np.rint(event["second_keypoints"][match.trainIdx].pt))
            color = (45, 190, 45) if index < len(mask) and mask[index] else (45, 45, 220)
            cv2.line(canvas, (x1, y1), (x2 + left.shape[1], y2), color, 1)
        caption = f"{event['status']} | green: accepted/essential inliers, red: other matches"
    else:
        canvas = cv2.cvtColor(event["image"], cv2.COLOR_GRAY2BGR)
        inliers = set(map(int, event["inlier_indices"]))
        for index, point in enumerate(event["pixels"][:limit]):
            color = (45, 190, 45) if index in inliers else (45, 45, 220)
            cv2.circle(canvas, tuple(np.rint(point).astype(int)), 4, color, 1)
        caption = f"PnP {event['status']} | green: RANSAC inliers, red: other associations"
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 850), 27), (10, 10, 10), -1)
    cv2.putText(canvas, caption, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Could not write {path}")


def _plot(values: list[float | None], path: Path, label: str,
          color: tuple[int, int, int], signed: bool = False) -> None:
    width, height = 1100, 320
    image = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(image, label, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2)
    cv2.line(image, (50, 160 if signed else 270),
             (width - 15, 160 if signed else 270), (100, 100, 100), 1)
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    maximum = max(finite, default=1.0) or 1.0
    previous = None
    for index, value in enumerate(values):
        if value is None or not np.isfinite(value):
            previous = None
            continue
        x = int(50 + index * (width - 70) / max(1, len(values) - 1))
        y = int(160 - float(value) * 95) if signed else int(270 - min(float(value) / maximum, 1.0) * 205)
        cv2.circle(image, (x, y), 2, color, -1)
        if previous is not None:
            cv2.line(image, previous, (x, y), color, 2)
        previous = (x, y)
    cv2.imwrite(str(path), image)


def _write_ply(path: Path, points: list[np.ndarray]) -> None:
    with path.open("w", encoding="utf-8") as output:
        output.write("ply\nformat ascii 1.0\ncomment arbitrary monocular scale; segment-local first-camera world frame\n")
        output.write(f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        for point in points:
            output.write(f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n")


def _draw_segment_map(path: Path, points: list[np.ndarray], poses: list[dict[str, Any]],
                      segment_id: int, final_keyframes: bool = False) -> None:
    """Top-down X/Z view with an explicit map/pose stage."""
    canvas = np.full((700, 900, 3), 255, np.uint8)
    centers = np.array([pose["camera_center_world"] for pose in poses], dtype=float).reshape(-1, 3)
    xyz = np.array(points, dtype=float).reshape(-1, 3)
    cloud = np.vstack((xyz[:, [0, 2]], centers[:, [0, 2]]))
    if len(cloud):
        low = np.percentile(cloud, 2, axis=0)
        high = np.percentile(cloud, 98, axis=0)
        span = np.maximum(high - low, 1e-6)
        def pixel(point: np.ndarray) -> tuple[int, int]:
            normalized = (point[[0, 2]] - low) / span
            return (int(np.clip(40 + normalized[0] * 820, 0, 899)),
                    int(np.clip(650 - normalized[1] * 590, 0, 699)))
        for point in xyz:
            cv2.circle(canvas, pixel(point), 1, (140, 140, 140), -1)
        projected_centers = [pixel(center) for center in centers]
        for first, second in zip(projected_centers, projected_centers[1:]):
            cv2.line(canvas, first, second, (180, 70, 40), 2)
        for center in projected_centers:
            cv2.circle(canvas, center, 4, (40, 40, 210), -1)
    stage = "final map/keyframes" if final_keyframes else "online map/trajectory"
    cv2.putText(canvas, f"Segment {segment_id}: {stage}, arbitrary scale",
                (20, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    cv2.imwrite(str(path), canvas)


def _final_map_pose_records(segment: SparseMap,
                            online_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Final-map plots use every adjusted keyframe, including the online-null anchor."""
    if segment.keyframes:
        return [{"frame_id": keyframe.frame_id, "segment_id": segment.segment_id,
                 "pose_valid": True,
                 "camera_center_world": keyframe.frame.camera_center_world.tolist()}
                for keyframe in segment.keyframes.values()]
    return [record for record in online_results
            if record["segment_id"] == segment.segment_id and record["pose_valid"]]


def run_sequence(mode: str, output_dir: Path, config: dict[str, Any],
                 kitti_sequence: Path | None = None, start: int = 0,
                 end: int | None = None, seed: int = 19,
                 ablation: str | None = None,
                 vary_visibility: bool = False) -> dict[str, Any]:
    if mode == "kitti":
        if kitti_sequence is None:
            raise ValueError("--kitti-sequence is required")
        sequence = KittiGraySequence(kitti_sequence)
    elif mode == "synthetic":
        sequence = generate_sequence(seed=seed, vary_visibility=vary_visibility,
                                     count=18 if vary_visibility else 12)
    else:
        raise ValueError("mode must be kitti or synthetic")
    if end is None:
        end = len(sequence) - 1
    if not 0 <= start <= end < len(sequence):
        raise ValueError("Invalid inclusive frame range")
    orb = config["orb"]
    if ablation not in (None, "baseline", "growth", "descriptors", "full"):
        raise ValueError("Unknown M4 ablation")
    mapping_settings = dict(config.get("mapping", {}))
    bundle_settings = dict(config.get("bundle_adjustment", {}))
    if ablation is not None:
        mapping_settings.update(enabled=ablation != "baseline",
                                update_descriptors=ablation in ("descriptors", "full"))
        bundle_settings["enabled"] = ablation == "full"
    tracker = MonocularTracker(
        sequence.camera,
        OrbExtractor(orb["nfeatures"], orb["scaleFactor"], orb["nlevels"], orb["fastThreshold"]),
        DescriptorMatcher(**config["matching"]), GeometryConfig(**config["geometry"]),
        TrackingConfig(**config.get("tracking", {})),
        MappingConfig(**mapping_settings), BundleConfig(**bundle_settings))
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = output_dir / "visualizations"
    visual_dir.mkdir(exist_ok=True)
    results = []
    visual_counts = Counter()
    for index in range(start, end + 1):
        result = tracker.process(sequence[index])
        results.append(result.as_dict())
        event = tracker.visual_event
        if event is not None:
            if event["kind"] == "initialization":
                name = f"init-{index:06d}-{event['status']}"
                _save_visual(event, visual_dir / f"{name}.png")
                visual_counts[name] += 1
            elif event["status"] == "rejected" or visual_counts[event["status"]] < 1 or index % 50 == 0:
                name = f"pnp-{index:06d}-{event['status']}"
                _save_visual(event, visual_dir / f"{name}.png")
                visual_counts[event["status"]] += 1
        if (index - start + 1) % 25 == 0:
            print(f"processed {index - start + 1}/{end - start + 1}: frame {index}, {result.state.value}", flush=True)
    rows = []
    for record in results:
        d = record["diagnostics"]
        geometry = d.get("geometry", {})
        row = {"frame": record["frame_id"], "state": record["state"],
                     "segment_id": record["segment_id"], "pose_valid": record["pose_valid"],
                     "failure_reason": record["failure_reason"], "features": d.get("features_detected"),
                     "map_points": d.get("map_points", d.get("available_map_points")),
                     "init_anchor": d.get("anchor_frame"), "init_gap": d.get("initialization_gap"),
                     "descriptor_matches": d.get("descriptor_matches"),
                     "essential_inliers": geometry.get("essential_inliers"),
                     "pose_inliers": geometry.get("pose_inliers"),
                     "init_accepted_points": geometry.get("accepted_points"),
                     "association_candidates": d.get("candidate_map_associations"),
                     "association_ratio_passed": d.get("ratio_passed_map_associations"),
                     "unique_associations": d.get("unique_map_associations"),
                     "map_associations": d.get("unique_map_associations"),
                     "pnp_correspondences": d.get("pnp_correspondences"),
                     "pnp_ransac_attempted": d.get("pnp_ransac_attempted"),
                     "pnp_inliers": d.get("pnp_inliers"),
                     "pnp_status": d.get("pnp_status"),
                     "pnp_failure_reason": d.get("pnp_failure_reason"),
                     "pnp_inlier_fraction": d.get("pnp_inlier_fraction"),
                     "median_reprojection_px": d.get("median_reprojection_px"),
                     "positive_depth_fraction": d.get("positive_depth_fraction"),
                     "recovery_frames_processed": d.get("recovery_frames_processed"),
                     "recovery_frames_remaining": d.get("recovery_frames_remaining"),
                     "terminal_recovery_exhausted": d.get("terminal_recovery_exhausted", False),
                     "lost_segment_id": d.get("lost_segment_id"),
                     "orb_ms": record["timing_ms"].get("orb"),
                     "association_ms": record["timing_ms"].get("association"),
                     "initialization_ms": record["timing_ms"].get("initialization"),
                     "pnp_ms": record["timing_ms"].get("pnp"),
                     "map_update_ms": record["timing_ms"].get("map_update"),
                     "total_ms": record["timing_ms"].get("total")}
        if ablation is not None:
            mapping = d.get("mapping", {})
            ba = d.get("bundle_adjustment", {})
            row.update(keyframe_inserted=d.get("keyframe_inserted", False),
                       keyframes=d.get("keyframes"),
                       new_points=mapping.get("new_points", 0),
                       active_map_points=d.get("active_map_points"),
                       culled_points=d.get("culled_points", 0),
                       ba_status=ba.get("status"), ba_reason=ba.get("reason"),
                       ba_initial_median_px=ba.get("initial_median_px"),
                       ba_final_median_px=ba.get("final_median_px"),
                       local_mapping_ms=record["timing_ms"].get("local_mapping"),
                       bundle_adjustment_ms=record["timing_ms"].get("bundle_adjustment"))
        rows.append(row)
    with (output_dir / "per_frame.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "per_frame.jsonl").open("w", encoding="utf-8") as output:
        for record in results:
            output.write(json.dumps(record, default=_json_default) + "\n")
    (output_dir / "initialization_attempts.json").write_text(
        json.dumps(tracker.attempts, indent=2, default=_json_default) + "\n", encoding="utf-8")
    (output_dir / "loss_events.json").write_text(
        json.dumps(tracker.loss_events, indent=2, default=_json_default) + "\n", encoding="utf-8")
    (output_dir / "recovery_events.json").write_text(
        json.dumps(tracker.recovery_events, indent=2, default=_json_default) + "\n", encoding="utf-8")
    if ablation is not None:
        for name, events in (("keyframe_events", tracker.keyframe_events),
                             ("map_growth_events", tracker.mapping_events),
                             ("ba_events", tracker.ba_events)):
            (output_dir / f"{name}.json").write_text(
                json.dumps(events, indent=2, default=_json_default) + "\n", encoding="utf-8")
    pose_export = {"pose_convention": "T_cw: X_c = R_cw X_w + t_cw",
                   "camera_center": "C_w = -R_cw.T @ t_cw",
                   "scale": "arbitrary per independent segment; no cross-segment alignment",
                   "invalid_poses": "null; frames before accepted second initialization view stay unestimated",
                   "frames": [{key: record[key] for key in ("frame_id", "segment_id", "pose_valid", "T_cw_R", "T_cw_t", "camera_center_world")}
                              for record in results]}
    if ablation is not None:
        pose_export["pose_stage"] = ("Final segment-local keyframe poses after accepted BA; "
                                     "other valid poses remain their online PnP estimates. "
                                     "Per-frame JSONL retains online published states.")
        by_identity = {(record["segment_id"], record["frame_id"]): record
                       for record in pose_export["frames"] if record["pose_valid"]}
        for segment in tracker.segments:
            for keyframe in segment.keyframes.values():
                record = by_identity.get((segment.segment_id, keyframe.frame_id))
                if record is not None:
                    frame = keyframe.frame
                    record["T_cw_R"] = frame.R_cw.tolist()
                    record["T_cw_t"] = frame.t_cw.tolist()
                    record["camera_center_world"] = frame.camera_center_world.tolist()
    (output_dir / "estimated_poses.json").write_text(json.dumps(pose_export, indent=2) + "\n", encoding="utf-8")
    if ablation is not None:
        final_keyframes = {"pose_convention": "T_cw: X_c = R_cw X_w + t_cw",
                           "pose_stage": "Final segment-local keyframe state after accepted BA",
                           "scale": "arbitrary per independent segment",
                           "poses": [{"segment_id": segment.segment_id,
                                      "frame_id": keyframe.frame_id,
                                      "T_cw_R": keyframe.frame.R_cw.tolist(),
                                      "T_cw_t": keyframe.frame.t_cw.tolist(),
                                      "camera_center_world": keyframe.frame.camera_center_world.tolist()}
                                     for segment in tracker.segments
                                     for keyframe in segment.keyframes.values()]}
        (output_dir / "final_keyframes.json").write_text(
            json.dumps(final_keyframes, indent=2) + "\n", encoding="utf-8")
    map_dir = output_dir / "maps"
    map_dir.mkdir(exist_ok=True)
    for segment in tracker.segments:
        points = [point.xyz_world for point in segment.points.values() if point.valid]
        _write_ply(map_dir / f"segment-{segment.segment_id:03d}.ply", points)
        _draw_segment_map(map_dir / f"segment-{segment.segment_id:03d}.png", points,
                          _final_map_pose_records(segment, results),
                          segment.segment_id, bool(segment.keyframes))
    _plot([1 if row["state"] == "tracking" else 0 if row["state"] == "initializing" else -1 for row in rows],
          output_dir / "tracking_states.png", "Tracking state: tracking=1, initializing=0, lost=-1",
          (80, 100, 190), signed=True)
    _plot([row["pnp_inliers"] for row in rows], output_dir / "pnp_inliers.png", "PnP RANSAC inliers", (40, 150, 50))
    _plot([row["median_reprojection_px"] for row in rows], output_dir / "reprojection_px.png", "PnP median inlier reprojection (pixels)", (190, 80, 50))
    if ablation is not None:
        _plot([row["active_map_points"] for row in rows], output_dir / "active_map_points.png",
              "Active map points", (40, 100, 180))
        _plot([row["keyframes"] for row in rows], output_dir / "keyframes.png",
              "Persistent keyframes", (130, 100, 40))
        _plot([row["ba_initial_median_px"] for row in rows], output_dir / "ba_before_px.png",
              "BA before median reprojection (pixels)", (180, 80, 80))
        _plot([row["ba_final_median_px"] for row in rows], output_dir / "ba_after_px.png",
              "BA after median reprojection (pixels)", (80, 150, 90))
    states = Counter(row["state"] for row in rows)
    accepted_pairs = [[a["anchor_frame"], a["candidate_frame"]] for a in tracker.attempts if a["initialization_status"] == "accepted"]
    pnp_inliers = [row["pnp_inliers"] for row in rows if row["pnp_inliers"] is not None]
    pnp_correspondences = [row["pnp_correspondences"] for row in rows if row["pnp_correspondences"] is not None]
    accepted_pnp_rows = [row for row in rows if row["pnp_status"] == "accepted"]
    rejected_pnp_rows = [row for row in rows if row["pnp_status"] == "rejected"]
    timings = [row["total_ms"] for row in rows]
    def distribution(values: list[float]) -> dict[str, float] | None:
        return None if not values else {"p10": float(np.percentile(values, 10)),
                                        "median": float(np.median(values)),
                                        "p90": float(np.percentile(values, 90)),
                                        "sum": float(np.sum(values))}
    runtime = {name: distribution([float(row[name]) for row in rows if row[name] is not None])
               for name in ("orb_ms", "association_ms", "initialization_ms", "pnp_ms",
                            "map_update_ms", "total_ms")}
    summary = {"mode": mode, "frame_range_inclusive": [start, end], "total_frames": len(rows),
               "camera": {"width": sequence.camera.width, "height": sequence.camera.height,
                          "fx": sequence.camera.fx, "fy": sequence.camera.fy,
                          "cx": sequence.camera.cx, "cy": sequence.camera.cy},
               "scale": "arbitrary per independent segment", "state_counts": dict(states),
               "lost_frame_fraction": states["lost"] / len(rows),
               "no_pose_frames": sum(not row["pose_valid"] for row in rows),
               "no_pose_fraction": sum(not row["pose_valid"] for row in rows) / len(rows),
               "initialization_attempts": len(tracker.attempts), "accepted_initialization_pairs": accepted_pairs,
               "initialization_rejections": dict(Counter(a["failure_reason"] for a in tracker.attempts if a["failure_reason"])),
               "pnp_tracked_frames": sum(row["state"] == "tracking" and row["pnp_inliers"] is not None for row in rows),
               "tracking_failures": len(tracker.loss_events), "loss_events": tracker.loss_events,
               "pnp_rejection_reasons": dict(Counter(row["pnp_failure_reason"] for row in rejected_pnp_rows)),
               "recovery_timeouts": len(tracker.recovery_events),
               "terminal_lost_frames": sum(bool(row["terminal_recovery_exhausted"]) for row in rows),
               "recovery_events": tracker.recovery_events,
               "reinitializations": max(0, len(tracker.segments) - 1),
               "map_points_per_segment": {str(s.segment_id): len(s.points) for s in tracker.segments},
               "pnp_correspondence_distribution": distribution(pnp_correspondences),
               "pnp_inlier_distribution": distribution(pnp_inliers),
               "pnp_distribution_definitions": {
                   "pnp_correspondence_distribution": "All map-based tracking attempts, accepted or rejected",
                   "pnp_inlier_distribution": "RANSAC attempts that returned an inlier count, accepted or rejected",
                   "accepted": "Only frames whose PnP pose passed all tracking gates",
                   "rejected_solver_returns": "Rejected frames with a returned RANSAC inlier count"},
               "pnp_correspondence_distribution_accepted": distribution(
                   [row["pnp_correspondences"] for row in accepted_pnp_rows]),
               "pnp_inlier_distribution_accepted": distribution(
                   [row["pnp_inliers"] for row in accepted_pnp_rows]),
               "pnp_correspondence_distribution_rejected": distribution(
                   [row["pnp_correspondences"] for row in rejected_pnp_rows
                    if row["pnp_correspondences"] is not None]),
               "pnp_inlier_distribution_rejected_solver_returns": distribution(
                   [row["pnp_inliers"] for row in rejected_pnp_rows if row["pnp_inliers"] is not None]),
               "stage_runtime_ms": runtime,
               "pnp_inliers_median": None if not pnp_inliers else float(np.median(pnp_inliers)),
               "pnp_inliers_p10": None if not pnp_inliers else float(np.percentile(pnp_inliers, 10)),
               "total_runtime_ms_median": float(np.median(timings)),
               "total_runtime_ms_p90": float(np.percentile(timings, 90)),
               "total_runtime_ms_sum": float(np.sum(timings))}
    if ablation is not None:
        span, longest, previous_segment = 0, 0, None
        for row in rows:
            if row["pose_valid"]:
                span = span + 1 if previous_segment == row["segment_id"] else 1
                previous_segment = row["segment_id"]
                longest = max(longest, span)
            else:
                span, previous_segment = 0, None
        summary.update(ablation=ablation, valid_poses=len(rows)-summary["no_pose_frames"],
                       longest_valid_span=longest, keyframes=len(tracker.keyframe_events),
                       new_map_points=sum(event["new_points"] for event in tracker.mapping_events),
                       active_map_points_at_end=0 if tracker.active_map is None else sum(
                           point.valid for point in tracker.active_map.points.values()),
                       valid_map_points_all_segments=sum(point.valid for seg in tracker.segments
                                                         for point in seg.points.values()),
                       culled_map_points=sum(not point.valid for seg in tracker.segments for point in seg.points.values()),
                       observation_count=sum(point.observation_count for seg in tracker.segments for point in seg.points.values()),
                       ba_attempts=len(tracker.ba_events),
                       ba_accepted=sum(event["applied"] for event in tracker.ba_events),
                       ba_rejected=sum(not event["applied"] for event in tracker.ba_events),
                       local_mapping_runtime_ms=distribution([row["local_mapping_ms"] for row in rows if row["local_mapping_ms"] is not None]),
                       ba_runtime_ms=distribution([row["bundle_adjustment_ms"] for row in rows if row["bundle_adjustment_ms"] is not None]))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "kitti"), default="synthetic")
    parser.add_argument("--kitti-sequence", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/m3-synthetic"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--ablation", choices=("baseline", "growth", "descriptors", "full"))
    parser.add_argument("--vary-visibility", action="store_true",
                        help="Synthetic only: landmarks enter and leave the view")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run_sequence(args.mode, args.output_dir, config, args.kitti_sequence,
                                  args.start, args.end, args.seed, args.ablation,
                                  args.vary_visibility), indent=2))


if __name__ == "__main__":
    main()
