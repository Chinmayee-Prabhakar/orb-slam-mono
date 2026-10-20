"""Fixed synthetic M2 audit grid; ground truth is used only after estimation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np

from .geometry import GeometryConfig, initialize_two_view, parallax_degrees
from .geometry_demo import DEFAULT_CONFIG
from .offline_evaluation import evaluate_synthetic
from .synthetic3d import generate_scene

SEEDS = (7, 11, 19, 23, 31)
CONDITIONS = (
    ("clean_reference", 0.0, 0.0, 0.8),
    ("nominal", 0.35, 0.0, 0.8),
    ("increased_noise", 1.0, 0.0, 0.8),
    ("heavy_noise", 2.0, 0.0, 0.8),
    ("incorrect_matches", 0.35, 0.30, 0.8),
    ("weak_baseline", 0.35, 0.0, 0.15),
    ("very_weak_baseline", 0.35, 0.0, 0.04),
    ("pure_rotation", 0.0, 0.0, 0.0),
)


def run_study(output_dir: Path, config: Optional[GeometryConfig] = None,
              label: str = "audit") -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or GeometryConfig()
    rows = []
    for condition, noise, outliers, baseline in CONDITIONS:
        for seed in SEEDS:
            scene = generate_scene(seed=seed, noise_px=noise,
                                   outlier_fraction=outliers, baseline=baseline)
            cv2.setRNGSeed(0)
            result = initialize_two_view(scene.points1_px, scene.points2_px, scene.camera, cfg)
            evaluation = evaluate_synthetic(result, scene)
            counts = result.diagnostics
            angles = None
            diagnostic_rotation = (result.R_21 if result.R_21 is not None
                                   else result.debug_provisional_R_21)
            if diagnostic_rotation is not None and result.pose_mask.any():
                angles = parallax_degrees(scene.points1_px[result.pose_mask],
                                          scene.points2_px[result.pose_mask],
                                          scene.camera, diagnostic_rotation)
            pose_fraction = None
            if counts["essential_inliers"] and counts["pose_inliers"] is not None:
                pose_fraction = counts["pose_inliers"] / counts["essential_inliers"]
            row = {
                "condition": condition, "seed": seed, "noise_px": noise,
                "outlier_fraction": outliers, "baseline_scene_units": baseline,
                "status": result.status, "failure_reason": result.failure_reason,
                "input_correspondences": counts["input_correspondences"],
                "essential_inliers": counts["essential_inliers"],
                "pose_inliers": counts["pose_inliers"],
                "pose_over_essential_fraction": pose_fraction,
                "rotation_only_median_residual_px": counts["rotation_only_median_residual_px"],
                "homography_inliers": counts["homography_inliers"],
                "homography_dominant": counts["homography_dominant"],
                "triangulation_filter_survivors": counts["accepted_points"],
                "usable_map_points": len(result.points_xyz) if result.status == "accepted" else 0,
                "candidate_parallax_p10_deg": None if angles is None else float(np.percentile(angles, 10)),
                "candidate_parallax_median_deg": None if angles is None else float(np.median(angles)),
                "candidate_parallax_p90_deg": None if angles is None else float(np.percentile(angles, 90)),
                "median_reprojection_first_px": counts["median_reprojection_error_first_px"],
                "median_reprojection_second_px": counts["median_reprojection_error_second_px"],
                "rotation_error_deg": evaluation["rotation_error_deg"],
                "translation_direction_error_deg": evaluation["translation_direction_error_deg"],
                "false_correspondences_among_map_points": evaluation["false_correspondences_among_map_points"],
                "runtime_ms": result.timing_ms["total"],
            }
            rows.append(row)
    summary: Dict[str, Any] = {}
    for condition, _, _, _ in CONDITIONS:
        group = [row for row in rows if row["condition"] == condition]
        accepted = [row for row in group if row["status"] == "accepted"]
        summary[condition] = {
            "accepted": len(accepted), "rejected": len(group) - len(accepted),
            "accepted_direction_error_over_45_deg": sum(
                row["translation_direction_error_deg"] is not None
                and row["translation_direction_error_deg"] > 45 for row in accepted),
            "median_accepted_rotation_error_deg": None if not accepted else float(np.median(
                [row["rotation_error_deg"] for row in accepted])),
            "median_accepted_direction_error_deg": None if not accepted else float(np.median(
                [row["translation_direction_error_deg"] for row in accepted
                 if row["translation_direction_error_deg"] is not None]))
                if any(row["translation_direction_error_deg"] is not None for row in accepted) else None,
        }
    report = {"label": label, "seeds": list(SEEDS),
              "config": vars(cfg), "rows": rows, "summary": summary,
              "truth_policy": "Synthetic truth is read only after initialize_two_view returns"}
    (output_dir / "robustness.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "robustness.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"{label}: fixed 5-seed x 8-condition synthetic study"]
    for condition, values in summary.items():
        lines.append(f"{condition}: {values['accepted']}/5 accepted; "
                     f"severe accepted direction errors (>45 deg): "
                     f"{values['accepted_direction_error_over_45_deg']}")
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--label", default="audit")
    args = parser.parse_args(argv)
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    report = run_study(args.output_dir, GeometryConfig(**settings["geometry"]), args.label)
    print((args.output_dir / "summary.txt").read_text(encoding="utf-8"), end="")
    print(f"Saved {len(report['rows'])} runs to {args.output_dir.absolute()}")


if __name__ == "__main__":
    main()
