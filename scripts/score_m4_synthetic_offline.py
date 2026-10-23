"""Offline known-baseline scoring of a varied-visibility M4 synthetic run.

Truth is generated only here, after the estimator's saved pose exports exist.
This script is not imported by online tracking, mapping, or bundle adjustment.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from orb_slam_mono.synthetic_sequence import generate_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_dir
    truth = generate_sequence(seed=19, count=18, vary_visibility=True)
    results = {}
    for name in ("baseline", "growth", "descriptors", "full"):
        summary = json.loads((root / name / "summary.json").read_text())
        poses = json.loads((root / name / "estimated_poses.json").read_text())["frames"]
        anchor, second = summary["accepted_initialization_pairs"][0]
        base_R = truth.truth_R_cw[anchor]
        base_C = -base_R.T @ truth.truth_t_cw[anchor]
        second_C = -truth.truth_R_cw[second].T @ truth.truth_t_cw[second]
        estimated_second = next(p for p in poses if p["frame_id"] == second and p["pose_valid"])
        scale = np.linalg.norm(second_C - base_C) / np.linalg.norm(estimated_second["camera_center_world"])
        rotation_errors, center_errors = [], []
        for pose in poses:
            if not pose["pose_valid"] or pose["segment_id"] != 0:
                continue
            i = pose["frame_id"]
            expected_R = truth.truth_R_cw[i] @ base_R.T
            estimated_R = np.asarray(pose["T_cw_R"])
            angle = np.degrees(np.arccos(np.clip((np.trace(estimated_R @ expected_R.T) - 1) / 2, -1, 1)))
            expected_C = base_R @ (-truth.truth_R_cw[i].T @ truth.truth_t_cw[i] - base_C)
            center = scale * np.asarray(pose["camera_center_world"])
            rotation_errors.append(float(angle))
            center_errors.append(float(np.linalg.norm(center - expected_C)))
        results[name] = {
            "scored_first_segment_poses": len(rotation_errors),
            "offline_known_baseline_scale": float(scale),
            "rotation_median_deg": float(np.median(rotation_errors)),
            "rotation_max_deg": float(np.max(rotation_errors)),
            "center_median_scene_units": float(np.median(center_errors)),
            "center_max_scene_units": float(np.max(center_errors)),
        }
    (root / "synthetic_pose_accuracy.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
