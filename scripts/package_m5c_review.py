"""Verify M5-C inputs and create a manifest-checked local review archive."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

from orb_slam_mono.m5c_data import sha256, validate_sources
from orb_slam_mono.m5c_offline_gt import comparison_data


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def package(root: Path) -> dict:
    root = root.resolve()
    base = root / "artifacts/m5c-visual-demonstration"
    zip_path = base / "M5C-independent-review.zip"
    if zip_path.exists():
        raise FileExistsError(f"Preserve the existing review package: {zip_path}")
    checked = validate_sources(root)
    original = json.loads((base / "capture/input_manifest.json").read_text(encoding="utf-8"))
    if checked != original:
        raise ValueError("Frozen M5-B, KITTI input, config, or GT differs from capture input manifest")
    equivalence = json.loads((base / "capture/replay_equivalence.json").read_text(encoding="utf-8"))
    if not equivalence["passed"] or equivalence["maximum_absolute_pose_difference"] != 0:
        raise ValueError("Coordinate replay has not passed frozen online equivalence")
    if not all(entry["byte_equal"] for entry in equivalence["final_map_ply"].values()):
        raise ValueError("A frozen segment map differs from diagnostic replay")
    media = json.loads((base / "MEDIA_MANIFEST.json").read_text(encoding="utf-8"))
    if len(media["videos"]) != 2 or len(media["figures"]) != 4:
        raise ValueError("Expected two videos and four non-GT figures")
    for item in media["videos"] + media["figures"]:
        if sha256(root / item["path"]) != item["sha256"]:
            raise ValueError(f"Rendered medium changed: {item['path']}")
    for video, expected in zip(media["videos"], ((4, 35, 88, 122), (7, 31, 252, 282))):
        segment, frames, first, last = expected
        if (video["map_segment"], video["decoded_frame_count"], video["source_ids"]) != (
                segment, frames, list(range(first, last+1))):
            raise ValueError("Video source IDs, frame count, or map segment changed")
    gt = comparison_data(root)
    gt_path = base / "figures/segment_004_gt_comparison.png"
    if cv2.imread(str(gt_path)) is None:
        raise ValueError("Offline GT figure does not decode")
    test_log = (base / "test-suite.log").read_text(encoding="utf-8", errors="replace")
    if "Ran 156 tests" not in test_log or not test_log.rstrip().endswith("OK"):
        raise ValueError("Complete passing 156-test log is missing")
    branch = git(root, "branch", "--show-current")
    head = git(root, "rev-parse", "HEAD")
    source_changes = git(root, "diff", "--name-only", "--", "src/orb_slam_mono/tracking.py",
                         "src/orb_slam_mono/map_state.py", "src/orb_slam_mono/local_mapping.py",
                         "src/orb_slam_mono/bundle_adjustment.py", "configs/default.json")
    if source_changes:
        raise ValueError(f"Production estimator or config changed:\n{source_changes}")
    evidence = (f"branch: {branch}\nHEAD: {head}\n\ngit status --short:\n"
                f"{git(root, 'status', '--short')}\n\ngit diff --stat:\n"
                f"{git(root, 'diff', '--stat')}\n\ngit log -1 --format=fuller:\n"
                f"{git(root, 'log', '-1', '--format=fuller')}\n\ngit remote -v:\n"
                f"{git(root, 'remote', '-v')}\n")
    (base / "git-evidence.txt").write_text(evidence, encoding="utf-8")
    provenance = {
        "frozen_run": "artifacts/m5b-final-evaluation/run-a1",
        "git_branch": branch, "git_head": head,
        "production_estimator_or_config_diff_empty": True,
        "input_manifest": "capture/input_manifest.json",
        "source_frame_count": 300,
        "capture_frame_count": checked["capture_frame_count"],
        "coordinate_replay": {
            "passed": equivalence["passed"],
            "frames_compared": equivalence["frames_compared"],
            "per_frame_fields_compared": equivalence["per_frame_fields_compared"],
            "maximum_absolute_pose_difference": equivalence["maximum_absolute_pose_difference"],
            "all_final_map_ply_byte_equal": True,
            "timing_policy": equivalence["timing_policy"],
        },
        "full_window_online": checked["baseline"],
        "missing_poses": 155,
        "segment_clips": [
            {"segment_id": 4, "source_frames_inclusive": [88, 122],
             "accepted_online_pose_frames_inclusive": [88, 121], "accepted_count": 34,
             "lost_frame": 122, "retained_pairs": 33, "raw_ransac_inliers": 11,
             "final_map_point_count": 670},
            {"segment_id": 7, "source_frames_inclusive": [252, 282],
             "accepted_online_pose_frames_inclusive": [252, 281], "accepted_count": 30,
             "lost_frame": 282, "retained_pairs": 70, "raw_ransac_inliers": 31,
             "final_map_point_count": 558, "metric_ate": None},
        ],
        "segment_004_offline_gt": {
            "source_frames_inclusive": [88, 121], "pose_count": 34,
            "in_sample_ate_rmse_m": gt["in_sample_rmse"],
            "prefix_fit_pose_count": 17, "suffix_score_pose_count": 17,
            "prefix_fit_suffix_ate_rmse_m": gt["prefix_fit_suffix_rmse"],
            "alignment_scale_m_per_estimated_unit": gt["scale_m_per_unit"],
            "alignment_rotation_determinant": gt["det_Q"],
            "ground_truth_used_online": False,
        },
        "full_run_two_segment_in_sample_ate_rmse_m": .184586,
        "full_run_two_segment_eligible_pose_count": 58,
        "media": {"videos": media["videos"], "figures": media["figures"],
                  "offline_gt_figure": {
                      "path": gt_path.relative_to(root).as_posix(),
                      "sha256": sha256(gt_path), "source_frames_inclusive": [88, 121]}},
        "python_version": sys.version.split()[0],
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        "test_suite": {"passed": 156, "log": "test-suite.log"},
        "visual_human_signoff": "pending",
    }
    (base / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    files = []
    files.extend((root / "src/orb_slam_mono").glob("*.py"))
    files.extend((root / "tests").glob("*.py"))
    files.extend(root / name for name in ("README.md", "PROJECT_PLAN.md", "STATUS.md",
                                           "NOTES.md", "REFERENCES.md", "requirements.txt",
                                           "environment.yml", "configs/default.json",
                                           "scripts/package_m5c_review.py",
                                           "data/sequences/00/calib.txt",
                                           "data/sequences/00/times.txt",
                                           "data/ground_truth/kitti_odometry/00.txt"))
    m5b = root / "artifacts/m5b-final-evaluation"
    files.extend(m5b / "run-a1" / name for name in (
        "per_frame.jsonl", "estimated_poses.json", "final_keyframes.json",
        "loss_events.json", "keyframe_events.json", "map_growth_events.json",
        "summary.json", "maps/segment-004.ply", "maps/segment-007.ply"))
    files.append(m5b / "evaluation/run-a1/segments.json")
    for name in ("REPORT.md", "FIGURE_GUIDE.md", "VIDEO_GUIDE.md", "REPRODUCE.txt",
                 "MEDIA_MANIFEST.json", "PROVENANCE.json", "test-suite.log", "git-evidence.txt"):
        files.append(base / name)
    files.extend((base / "capture").glob("*.json"))
    files.extend((base / "figures").glob("*.png"))
    files.extend((base / "videos").glob("*.mp4"))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Review package missing files: " + ", ".join(missing))
    files = sorted(set(files), key=lambda path: path.relative_to(root).as_posix())
    entries = {path.relative_to(root).as_posix(): {"sha256": sha256(path), "bytes": path.stat().st_size}
               for path in files}
    manifest = {"description": "M5-C independent review; raw KITTI images omitted",
                "entry_count": len(entries), "entries": entries,
                "external_raw_images_required_for_rerender": "data/sequences/00/image_0/000000.png–000299.png",
                "video_files_included": True}
    manifest_path = base / "REVIEW_MANIFEST.json"
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr(manifest_path.relative_to(root).as_posix(), manifest_bytes)
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Review archive failed ZIP CRC check")
        for name, item in entries.items():
            import hashlib
            if hashlib.sha256(archive.read(name)).hexdigest() != item["sha256"]:
                raise ValueError(f"ZIP entry failed SHA-256: {name}")
    return {"path": zip_path.relative_to(root).as_posix(), "sha256": sha256(zip_path),
            "bytes": zip_path.stat().st_size, "entries": len(entries)+1,
            "videos_included": True, "frozen_inputs_match_capture": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(package(args.root), indent=2))
