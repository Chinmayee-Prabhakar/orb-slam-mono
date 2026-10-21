"""Summarize the seven fixed sequence-00 M2 attempts without rerunning estimation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PAIRS = ((0, 1), (0, 5), (10, 15), (20, 25), (40, 45), (60, 65), (80, 85))
FIELDS = (
    "first_index", "second_index", "first_features", "second_features",
    "knn_pairs", "ratio_passed", "retained_matches", "essential_inliers",
    "essential_inlier_percentage", "pose_inliers", "pose_essential_inlier_fraction",
    "rotation_only_median_residual_px", "median_parallax_deg",
    "triangulation_candidates", "positive_depth", "parallax_pass", "reprojection_pass",
    "accepted_triangulated_points", "median_reprojection_first_px",
    "median_reprojection_second_px", "initialization_status", "failure_reason",
    "end_to_end_runtime_ms",
)


def summarize(results_dir: Path) -> dict:
    rows = []
    for first, second in PAIRS:
        path = results_dir / f"pair-{first:06d}-{second:06d}" / "stats.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report["dataset"] != "kitti_odometry_gray" or report["mode"] != "kitti"
                or (report["first_index"], report["second_index"]) != (first, second)):
            raise ValueError(f"Wrong dataset or pair in {path}")
        if report["initialization_status"] not in ("accepted", "rejected"):
            raise ValueError(f"Missing explicit outcome in {path}")
        if report["initialization_status"] == "rejected" and not report["failure_reason"]:
            raise ValueError(f"Missing rejection reason in {path}")
        diagnostics = report["diagnostics"]
        inlier_ratio = diagnostics["essential_inlier_ratio"]
        rows.append({
            "first_index": first,
            "second_index": second,
            "first_features": report["first_features"],
            "second_features": report["second_features"],
            "knn_pairs": report["knn_pairs"],
            "ratio_passed": report["ratio_passed"],
            "retained_matches": report["retained_matches"],
            "essential_inliers": diagnostics["essential_inliers"],
            "essential_inlier_percentage": None if inlier_ratio is None else round(100 * inlier_ratio, 2),
            "pose_inliers": diagnostics["pose_inliers"],
            "pose_essential_inlier_fraction": diagnostics["pose_inlier_fraction"],
            "rotation_only_median_residual_px": diagnostics["rotation_only_median_residual_px"],
            "median_parallax_deg": diagnostics["median_candidate_parallax_deg"],
            "triangulation_candidates": diagnostics["triangulation_candidates"],
            "positive_depth": diagnostics["positive_depth"],
            "parallax_pass": diagnostics["parallax_pass"],
            "reprojection_pass": diagnostics["reprojection_pass"],
            "accepted_triangulated_points": diagnostics["accepted_points"],
            "median_reprojection_first_px": diagnostics["median_reprojection_error_first_px"],
            "median_reprojection_second_px": diagnostics["median_reprojection_error_second_px"],
            "initialization_status": report["initialization_status"],
            "failure_reason": report["failure_reason"],
            "end_to_end_runtime_ms": report["timing_ms"]["end_to_end_total"],
        })
    accepted = sum(row["initialization_status"] == "accepted" for row in rows)
    return {
        "dataset": "official_kitti_odometry_gray_sequence_00",
        "configuration": "configs/default.json (unchanged)",
        "scale": "arbitrary; unit translation baseline",
        "attempts": len(rows),
        "accepted": accepted,
        "rejected": len(rows) - accepted,
        "acceptance_fraction": accepted / len(rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("artifacts/kitti-00-validation"))
    args = parser.parse_args()
    result = summarize(args.results_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.results_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(result["rows"])
    (args.results_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["rows"]:
        print(f"{row['first_index']:02d}->{row['second_index']:02d}: "
              f"{row['initialization_status']}"
              f"{'' if row['failure_reason'] is None else ' (' + row['failure_reason'] + ')'}")
    print(f"Accepted {result['accepted']}/{result['attempts']}; saved summary.csv and summary.json")


if __name__ == "__main__":
    main()
