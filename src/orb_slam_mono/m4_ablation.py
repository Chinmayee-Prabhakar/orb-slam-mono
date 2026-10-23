"""Run fixed M4 configurations on identical chronological frames; preserve M3 artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .sequence_demo import run_sequence


ABLATIONS = ("baseline", "growth", "descriptors", "full")


def run_ablations(mode: str, output_dir: Path, config: dict,
                  kitti_sequence: Path | None = None, start: int = 0,
                  end: int | None = None, seed: int = 19,
                  vary_visibility: bool = False) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name in ABLATIONS:
        summary = run_sequence(mode, output_dir / name, config, kitti_sequence,
                               start, end, seed, name, vary_visibility)
        results.append({key: summary[key] for key in (
            "ablation", "total_frames", "valid_poses", "pnp_tracked_frames",
            "no_pose_frames", "initialization_attempts", "tracking_failures",
            "reinitializations", "recovery_timeouts", "longest_valid_span",
            "keyframes", "new_map_points", "active_map_points_at_end",
            "valid_map_points_all_segments", "culled_map_points",
            "observation_count", "ba_attempts", "ba_accepted", "ba_rejected",
            "total_runtime_ms_median", "total_runtime_ms_p90", "total_runtime_ms_sum")}
                       | {"accepted_initialization_pairs": len(summary["accepted_initialization_pairs"]),
                          "pnp_correspondences_median": None if summary["pnp_correspondence_distribution"] is None
                          else summary["pnp_correspondence_distribution"]["median"],
                          "pnp_inliers_median": summary["pnp_inliers_median"]})
    (output_dir / "comparison.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "kitti"), required=True)
    parser.add_argument("--kitti-sequence", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--vary-visibility", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/m4-ablation"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run_ablations(args.mode, args.output_dir, config, args.kitti_sequence,
                                   args.start, args.end, args.seed, args.vary_visibility), indent=2))


if __name__ == "__main__":
    main()
