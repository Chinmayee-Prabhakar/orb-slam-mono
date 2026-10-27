"""Evaluate the three saved KITTI runs without private milestone artifacts.

This is offline orchestration only. Numerical scoring and repeat comparisons
are delegated unchanged to the project's existing evaluation functions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

# Allow `python scripts/evaluate_release.py` from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orb_slam_mono.kitti_pose_evaluation import load_kitti_poses
from orb_slam_mono.m5b_evaluation import (
    RUNS, compare_restart, evaluate_saved_run, repeatability_report,
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def evaluate_release(root: Path, output: Path) -> dict[str, Any]:
    """Score saved runs; do not run SLAM, overwrite outputs, or certify old audits."""
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Preserving existing output: {output}")
    sequence = root / "data/sequences/00"
    gt_path = root / "data/ground_truth/kitti_odometry/00.txt"
    base = root / "artifacts/m5b-final-evaluation"
    gt = load_kitti_poses(gt_path)
    values = {
        name: evaluate_saved_run(base / name, sequence, gt, *bounds)
        for name, bounds in RUNS.items()
    }
    comparison = {
        "first_half": compare_restart(values["run-a1"], values["run-b1"], gt),
        "second_half": compare_restart(values["run-a1"], values["run-c1"], gt),
    }
    # The six separately launched runs remain mandatory for repeatability.
    repeats = repeatability_report(root)
    scope = {
        "mode": "public saved-run evaluation",
        "scoring": "Existing M5-A/M5-B mathematical functions, unchanged",
        "ground_truth": "Offline evaluation only; never passed to an online runner",
        "historical_baseline_comparison": {
            "status": "not_performed",
            "reason": "Private historical M4/M4.5/M5-A artifacts are not required for a fresh checkout."
        },
        "repeatability": "Non-timing outputs of A1/A2, B1/B2 and C1/C2 compared",
    }
    # Finish scoring before publishing; never overwrite a previous evaluation.
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".evaluation-", dir=output.parent))
    try:
        for name, value in values.items():
            for filename, key in (("metrics", "metrics"), ("segments", "segments"),
                                  ("per_frame_errors", "errors"), ("relative_pairs", "relative"),
                                  ("anchor_validation", "anchors")):
                _write(staging / name / f"{filename}.json", value[key])
        _write(staging / "restart_comparison.json", comparison)
        _write(staging / "repeatability.json", repeats)
        _write(staging / "evaluation_scope.json", scope)
        if output.exists():
            raise FileExistsError(f"Output appeared during evaluation: {output}")
        # A same-directory rename can be denied by managed Windows folders.
        # copytree still refuses an output that appeared after the precheck.
        shutil.copytree(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"output": str(output), "runs": {
        name: {"coverage_percent": v["metrics"]["online_pose_fraction_percent"],
               "eligible_poses": v["metrics"]["eligible_online_poses"],
               "ate_rmse_m": v["metrics"]["in_sample_ate_m"]["rmse"]}
        for name, v in values.items()}, "scope": scope}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/m5b-final-evaluation/evaluation"),
                        help="Absolute path or path relative to --root; must not exist")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else args.root / args.output
    try:
        result = evaluate_release(args.root, output)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"Evaluation stopped: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
