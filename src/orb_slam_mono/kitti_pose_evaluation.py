"""Offline KITTI pose audit. This module is never imported by online geometry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


ORIGINAL_PAIRS = ((0, 1), (0, 5), (10, 15), (20, 25), (40, 45), (60, 65), (80, 85))
HELDOUT_PAIRS = ((100, 105), (120, 125), (140, 145), (180, 185), (220, 225))
# Direction becomes ill-conditioned as the true baseline approaches zero.
MIN_RELIABLE_TRUE_BASELINE_M = 0.05


def load_kitti_poses(path: Path) -> np.ndarray:
    """Read KITTI camera-to-world 3x4 rows as N homogeneous T_wc matrices."""
    rows = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if rows.ndim != 2 or rows.shape[1] != 12 or not len(rows) or not np.isfinite(rows).all():
        raise ValueError("KITTI pose file must have finite 12-value rows")
    poses = np.repeat(np.eye(4)[None, :, :], len(rows), axis=0)
    poses[:, :3, :] = rows.reshape(-1, 3, 4)
    rotations = poses[:, :3, :3]
    if (np.max(np.abs(rotations @ np.transpose(rotations, (0, 2, 1)) - np.eye(3))) > 1e-3
            or np.max(np.abs(np.linalg.det(rotations) - 1.0)) > 1e-3):
        raise ValueError("KITTI pose rotations must be proper and orthogonal")
    return poses


def relative_world_to_camera(poses_wc: np.ndarray, first: int,
                             second: int) -> Tuple[np.ndarray, np.ndarray]:
    """T_cj_ci = inverse(T_wc_j) @ T_wc_i, in the first camera's frame."""
    if not 0 <= first < second < len(poses_wc):
        raise IndexError("Pose frame indices are outside the sequence")
    first_pose, second_pose = poses_wc[first], poses_wc[second]
    rotation = second_pose[:3, :3].T @ first_pose[:3, :3]
    translation = second_pose[:3, :3].T @ (first_pose[:3, 3] - second_pose[:3, 3])
    return rotation, translation


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))


def direction_error_degrees(estimated: np.ndarray, truth: np.ndarray) -> float:
    cosine = float(np.dot(estimated, truth) / (np.linalg.norm(estimated) * np.linalg.norm(truth)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def evaluate_pair(report: Dict[str, Any], poses_wc: np.ndarray) -> Dict[str, Any]:
    first, second = report["first_index"], report["second_index"]
    truth_rotation, truth_translation = relative_world_to_camera(poses_wc, first, second)
    baseline = float(np.linalg.norm(truth_translation))
    reliable_direction = baseline >= MIN_RELIABLE_TRUE_BASELINE_M
    truth_direction = truth_translation / baseline if reliable_direction else None
    accepted = report["initialization_status"] == "accepted"
    if accepted and (report["R_21"] is None or report["t_21_unit"] is None):
        raise ValueError("Accepted report has no public pose")
    if not accepted and (report["R_21"] is not None or report["t_21_unit"] is not None
                         or report["valid_for_mapping"]):
        raise ValueError("Rejected report exposes a usable pose")
    estimated_rotation = np.asarray(report["R_21"], dtype=float) if accepted else None
    estimated_direction = np.asarray(report["t_21_unit"], dtype=float) if accepted else None
    if accepted and (estimated_rotation.shape != (3, 3) or estimated_direction.shape != (3,)
                     or not np.isfinite(estimated_rotation).all()
                     or not np.isfinite(estimated_direction).all()):
        raise ValueError("Accepted pose is malformed")
    diagnostics = report["diagnostics"]
    return {
        "first_index": first, "second_index": second,
        "status": report["initialization_status"], "failure_reason": report["failure_reason"],
        "estimated_rotation_deg": None if not accepted else rotation_angle_degrees(estimated_rotation),
        "truth_rotation_deg": rotation_angle_degrees(truth_rotation),
        "rotation_error_deg": None if not accepted else rotation_angle_degrees(
            estimated_rotation @ truth_rotation.T),
        "estimated_translation_direction": None if not accepted else estimated_direction.tolist(),
        "truth_translation_direction": None if truth_direction is None else truth_direction.tolist(),
        "translation_direction_error_deg": None if not accepted or not reliable_direction else
        direction_error_degrees(estimated_direction, truth_translation),
        "truth_baseline_m_evaluation_only": baseline,
        "truth_direction_reliable": reliable_direction,
        "essential_inliers": diagnostics["essential_inliers"],
        "pose_inliers": diagnostics["pose_inliers"],
        "median_parallax_deg": diagnostics["median_candidate_parallax_deg"],
        "accepted_points": diagnostics["accepted_points"],
        "median_reprojection_first_px": diagnostics["median_reprojection_error_first_px"],
        "median_reprojection_second_px": diagnostics["median_reprojection_error_second_px"],
    }


def evaluate_directory(poses_path: Path, results_dir: Path, output_dir: Path,
                       pairs: Sequence[Tuple[int, int]], label: str) -> Dict[str, Any]:
    poses = load_kitti_poses(poses_path)
    rows = []
    for first, second in pairs:
        path = results_dir / f"pair-{first:06d}-{second:06d}" / "stats.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report["dataset"] != "kitti_odometry_gray"
                or (report["first_index"], report["second_index"]) != (first, second)):
            raise ValueError(f"Wrong KITTI result in {path}")
        rows.append(evaluate_pair(report, poses))
    result = {"label": label, "ground_truth_convention": "T_wc camera to first-camera world",
              "relative_composition": "T_cj_ci = inverse(T_wc_j) @ T_wc_i",
              "truth_use": "offline evaluation only; no estimator input or scale selection",
              "ground_truth_sha256": hashlib.sha256(poses_path.read_bytes()).hexdigest(),
              "pose_rows": len(poses), "rows": rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pose_accuracy.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "pose_accuracy.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-set", choices=("original", "heldout"), required=True)
    args = parser.parse_args(argv)
    pairs = ORIGINAL_PAIRS if args.pair_set == "original" else HELDOUT_PAIRS
    report = evaluate_directory(args.ground_truth, args.results_dir, args.output_dir,
                                pairs, args.pair_set)
    for row in report["rows"]:
        print(f"{row['first_index']}->{row['second_index']}: {row['status']}; "
              f"rot error={row['rotation_error_deg']}; "
              f"direction error={row['translation_direction_error_deg']}")
    print(f"Saved {len(report['rows'])} offline evaluations to {args.output_dir}")


if __name__ == "__main__":
    main()
