"""Offline, truth-free correspondence audit of one saved KITTI M2 attempt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from orb_slam_mono.datasets import KittiGraySequence
from orb_slam_mono.features import OrbExtractor
from orb_slam_mono.geometry import GeometryConfig, _essential_candidates, initialize_two_view
from orb_slam_mono.kitti_pose_evaluation import rotation_angle_degrees
from orb_slam_mono.matching import DescriptorMatcher


def _bearing_rotation(first: np.ndarray, second: np.ndarray, indices: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(second[indices].T @ first[indices])
    return u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt


def _rotation_fit(first_px: np.ndarray, second_px: np.ndarray,
                  camera, robust: bool) -> dict:
    first = np.column_stack(((first_px[:, 0] - camera.cx) / camera.fx,
                             (first_px[:, 1] - camera.cy) / camera.fy,
                             np.ones(len(first_px))))
    second = np.column_stack(((second_px[:, 0] - camera.cx) / camera.fx,
                              (second_px[:, 1] - camera.cy) / camera.fy,
                              np.ones(len(second_px))))
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second /= np.linalg.norm(second, axis=1, keepdims=True)

    def residuals(rotation):
        image = (camera.matrix @ (rotation @ first.T)).T
        return np.linalg.norm(image[:, :2] / image[:, 2:3] - second_px, axis=1)

    if robust:
        rng = np.random.default_rng(17)
        choices = (rng.choice(len(first), 3, replace=False) for _ in range(500))
        rotation = min((_bearing_rotation(first, second, indices) for indices in choices),
                       key=lambda candidate: float(np.median(residuals(candidate))))
        inliers = residuals(rotation) < 3.0
        if inliers.sum() >= 3:
            rotation = _bearing_rotation(first, second, np.flatnonzero(inliers))
    else:
        rotation = _bearing_rotation(first, second, np.arange(len(first)))
    errors = residuals(rotation)
    return {"rotation_deg": rotation_angle_degrees(rotation),
            "median_residual_px": float(np.median(errors)),
            "p90_residual_px": float(np.percentile(errors, 90)),
            "within_3px": int((errors < 3.0).sum())}


def audit(sequence_path: Path, report_path: Path, config_path: Path,
          output_dir: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sequence = KittiGraySequence(sequence_path)
    first_index, second_index = report["first_index"], report["second_index"]
    first, second = sequence[first_index].image, sequence[second_index].image
    orb = config["orb"]
    extractor = OrbExtractor(orb["nfeatures"], orb["scaleFactor"], orb["nlevels"],
                             orb["fastThreshold"])
    features1, features2 = extractor.extract(first), extractor.extract(second)
    matches = DescriptorMatcher(**config["matching"]).match(features1.descriptors,
                                                             features2.descriptors).matches
    points1 = np.array([features1.keypoints[item.queryIdx].pt for item in matches])
    points2 = np.array([features2.keypoints[item.trainIdx].pt for item in matches])
    if len(matches) != report["retained_matches"]:
        raise ValueError("ORB match count changed from saved baseline")
    essential, mask = cv2.findEssentialMat(points1, points2, sequence.camera.matrix,
                                           method=cv2.RANSAC,
                                           prob=config["geometry"]["ransac_confidence"],
                                           threshold=config["geometry"]["ransac_threshold_px"])
    same_mask = np.array_equal(np.flatnonzero(mask), report["essential_inlier_indices"])
    indices = np.asarray(report["essential_inlier_indices"], dtype=int)
    selected1, selected2 = points1[indices], points2[indices]
    flow = selected2 - selected1
    magnitude = np.linalg.norm(flow, axis=1)
    cell_x = np.clip((selected1[:, 0] / (sequence.camera.width / 4)).astype(int), 0, 3)
    cell_y = np.clip((selected1[:, 1] / (sequence.camera.height / 3)).astype(int), 0, 2)
    grid = np.bincount(cell_y * 4 + cell_x, minlength=12).reshape(3, 4)
    original_rotation = np.asarray(report["R_21"], dtype=float)
    settings = dict(config["geometry"], pose_stability_trials=0)
    cfg = GeometryConfig(**settings)
    rng = np.random.default_rng(config["geometry"]["pose_stability_seed"])
    trials = []
    for trial_index in range(config["geometry"]["pose_stability_trials"]):
        order = rng.permutation(len(points1))
        attempt = initialize_two_view(points1[order], points2[order], sequence.camera, cfg)
        rotation = attempt.R_21 if attempt.R_21 is not None else attempt.debug_provisional_R_21
        translation = attempt.t_21 if attempt.t_21 is not None else attempt.debug_provisional_t_21
        trials.append({"trial": trial_index,
                       "status": attempt.status, "reason": attempt.failure_reason,
                       "essential_inliers": attempt.diagnostics["essential_inliers"],
                       "pose_inliers": attempt.diagnostics["pose_inliers"],
                       "rotation_deg": None if rotation is None else rotation_angle_degrees(rotation),
                       "translation_direction": None if translation is None else translation.tolist(),
                       "median_parallax_deg": attempt.diagnostics["median_candidate_parallax_deg"],
                       "rotation_disagreement_deg": None if rotation is None else
                       rotation_angle_degrees(rotation @ original_rotation.T)})
    seed_trials = []
    for seed in range(5):
        cv2.setRNGSeed(seed)
        attempt = initialize_two_view(points1, points2, sequence.camera, cfg)
        rotation = attempt.R_21 if attempt.R_21 is not None else attempt.debug_provisional_R_21
        seed_trials.append({"opencv_rng_seed": seed,
                            "essential_inliers": attempt.diagnostics["essential_inliers"],
                            "status": attempt.status,
                            "rotation_deg": None if rotation is None else rotation_angle_degrees(rotation)})
    disagreements = [row["rotation_disagreement_deg"] for row in trials
                     if row["rotation_disagreement_deg"] is not None]
    result = {
        "pair": [first_index, second_index],
        "feature_counts": [features1.count, features2.count],
        "retained_matches": len(matches),
        "essential_output_shape": list(np.shape(essential)),
        "essential_candidate_count": len(_essential_candidates(essential)),
        "essential_mask_reproduces_original": same_mask,
        "essential_inliers": len(indices),
        "inlier_grid_4x3": grid.tolist(),
        "largest_grid_cell_fraction": float(grid.max() / len(indices)),
        "occupied_grid_cells": int(np.count_nonzero(grid)),
        "median_flow_xy_px": np.median(flow, axis=0).tolist(),
        "flow_magnitude_p10_p50_p90_px": np.percentile(magnitude, [10, 50, 90]).tolist(),
        "least_squares_rotation_only_fit": _rotation_fit(selected1, selected2,
                                                          sequence.camera, robust=False),
        "robust_rotation_only_fit": _rotation_fit(selected1, selected2,
                                                   sequence.camera, robust=True),
        "original_estimated_rotation_deg": rotation_angle_degrees(original_rotation),
        "original_estimated_parallax_deg": report["diagnostics"]["median_candidate_parallax_deg"],
        "permutation_trials": trials,
        "original_order_rng_seed_trials": seed_trials,
        "median_rotation_disagreement_deg": float(np.median(disagreements)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "correspondence_audit.json").write_text(json.dumps(result, indent=2) + "\n",
                                                          encoding="utf-8")
    canvas = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
    for column in range(1, 4):
        x = round(column * sequence.camera.width / 4)
        cv2.line(canvas, (x, 0), (x, sequence.camera.height - 1), (255, 190, 0), 1)
    for row in range(1, 3):
        y = round(row * sequence.camera.height / 3)
        cv2.line(canvas, (0, y), (sequence.camera.width - 1, y), (255, 190, 0), 1)
    for point, delta, distance in zip(selected1, flow, magnitude):
        start = tuple(np.rint(point).astype(int))
        end = tuple(np.rint(point + 2 * delta).astype(int))
        color = (0, 210, 0) if distance < 10 else (0, 80, 240)
        cv2.circle(canvas, start, 2, color, -1)
        cv2.arrowedLine(canvas, start, end, color, 1, tipLength=0.15)
    cv2.putText(canvas, f"{first_index}->{second_index}: E inliers {len(indices)}; largest cell "
                f"{grid.max()}/{len(indices)}; arrows 2x flow", (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(output_dir / "inlier_grid_flow.png"), canvas):
        raise OSError("Could not write inlier_grid_flow.png")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, default=Path("data/sequences/00"))
    parser.add_argument("--report", type=Path,
                        default=Path("artifacts/kitti-00-validation/pair-000020-000025/stats.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/kitti-00-pose-audit/anomaly-20-25"))
    args = parser.parse_args()
    result = audit(args.sequence, args.report, args.config, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
