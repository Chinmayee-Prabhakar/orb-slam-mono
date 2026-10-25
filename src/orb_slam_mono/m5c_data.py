"""Read-only M5-C source validation, geometry, and opt-in feature replay.

Online decisions come exclusively from saved Run A. A separate diagnostic
replay copies coordinates for display and must match all frozen decisions,
keyframes, and final PLY bytes before its overlays can be used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .datasets import KittiGraySequence
from .tracking_diagnostics import _build_tracker, _plain, compare_replay


CLIPS = {4: (88, 122), 7: (252, 282)}
STORYBOARD = (121, 122, 203, 206, 207, 252)
CAPTURE_IDS = sorted(set(STORYBOARD) | set(range(88, 123)) | set(range(252, 283)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 300 or [row["frame_id"] for row in rows] != list(range(300)):
        raise ValueError("Run A must contain original KITTI source IDs 0–299 exactly once")
    return rows


def read_ply(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines[:3] != ["ply", "format ascii 1.0",
                     "comment arbitrary monocular scale; segment-local first-camera world frame"]:
        raise ValueError(f"Unexpected segment-local PLY header: {path}")
    if not lines[3].startswith("element vertex ") or lines[7] != "end_header":
        raise ValueError(f"Malformed PLY vertex header: {path}")
    count = int(lines[3].split()[-1])
    points = np.asarray([[float(value) for value in line.split()] for line in lines[8:]], dtype=float).reshape(-1, 3)
    if len(points) != count or not np.isfinite(points).all():
        raise ValueError(f"PLY coordinates/count invalid: {path}")
    return points


def pose_geometry(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if not row["pose_valid"]:
        raise ValueError("An invalid online frame has no camera geometry")
    R = np.asarray(row["T_cw_R"], dtype=float).reshape(3, 3)
    t = np.asarray(row["T_cw_t"], dtype=float).reshape(3)
    if not np.isfinite(R).all() or not np.isfinite(t).all() or not np.allclose(R.T @ R, np.eye(3), atol=1e-5):
        raise ValueError("Non-rigid online pose")
    center = -R.T @ t
    if not np.allclose(center, row["camera_center_world"], atol=1e-9):
        raise ValueError("Exported camera center disagrees with T_cw")
    return center, R.T


def frustum_world(row: dict[str, Any], K: np.ndarray, width: int, height: int,
                  depth: float = 0.7) -> np.ndarray:
    """Camera center and four sensor-corner rays in segment-local coordinates."""
    center, R_wc = pose_geometry(row)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    corners = np.array([[(x-cx)/fx*depth, (y-cy)/fy*depth, depth]
                        for x, y in ((0, 0), (width, 0), (width, height), (0, height))])
    return np.vstack((center, center + (R_wc @ corners.T).T))


def validate_sources(root: Path) -> dict[str, Any]:
    root = root.resolve()
    run = root / "artifacts/m5b-final-evaluation/run-a1"
    sequence = KittiGraySequence(root / "data/sequences/00")
    if len(sequence) != 300 or [p.name for p in sequence.paths] != [f"{i:06d}.png" for i in range(300)]:
        raise ValueError("Original KITTI image IDs are missing or reordered")
    rows = load_rows(run / "per_frame.jsonl")
    for segment, (first, lost) in CLIPS.items():
        for frame in range(first, lost):
            row = rows[frame]
            if not row["pose_valid"] or row["segment_id"] != segment:
                raise ValueError(f"Wrong accepted pose/segment at source frame {frame}")
            pose_geometry(row)
        last = rows[lost]
        if last["pose_valid"] or last["state"] != "lost" or last["segment_id"] != segment:
            raise ValueError(f"Loss frame {lost} must have no accepted pose")
        if any(last[key] is not None for key in ("T_cw_R", "T_cw_t", "camera_center_world")):
            raise ValueError("Lost frame exports non-null camera geometry")
    if rows[203]["pose_valid"] or rows[203]["state"] != "initializing":
        raise ValueError("Frame 203 is only an initialization view")
    if not rows[206]["pose_valid"] or rows[206]["segment_id"] != 6 or rows[207]["pose_valid"]:
        raise ValueError("Fresh segment 006 and immediate loss changed")
    maps = {segment: read_ply(run / "maps" / f"segment-{segment:03d}.ply") for segment in CLIPS}
    if {key: len(value) for key, value in maps.items()} != {4: 670, 7: 558}:
        raise ValueError("Final map landmark populations changed")
    exported = json.loads((run / "estimated_poses.json").read_text(encoding="utf-8"))["frames"]
    for segment, (first, lost) in CLIPS.items():
        for frame in range(first, lost + 1):
            for field in ("frame_id", "segment_id", "pose_valid", "T_cw_R", "T_cw_t", "camera_center_world"):
                if rows[frame][field] != exported[frame][field]:
                    raise ValueError("Final pose export differs from accepted online pose")
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    segment_count = len(summary["accepted_initialization_pairs"])
    if (summary["valid_poses"], segment_count, summary["longest_valid_span"],
            summary["ba_accepted"], summary["ba_attempts"]) != (145, 9, 34, 0, 10):
        raise ValueError("Accepted full-window baseline changed")
    hashes = {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in
              [run / name for name in ("per_frame.jsonl", "estimated_poses.json", "final_keyframes.json",
                                           "loss_events.json", "keyframe_events.json", "map_growth_events.json",
                                           "summary.json")]
              + [run / "maps" / f"segment-{seg:03d}.ply" for seg in CLIPS]
              + [root / "data/sequences/00" / "image_0" / f"{i:06d}.png" for i in CAPTURE_IDS]
              + [root / "data/sequences/00/calib.txt", root / "data/sequences/00/times.txt",
                 root / "data/ground_truth/kitti_odometry/00.txt", root / "configs/default.json"]}
    return {"source_frame_count": len(sequence), "capture_frame_count": len(CAPTURE_IDS),
            "capture_ids": CAPTURE_IDS, "clip_ranges_inclusive": {str(s): [a, b] for s, (a, b) in CLIPS.items()},
            "map_point_counts": {str(s): len(v) for s, v in maps.items()},
            "baseline": {"valid_poses": summary["valid_poses"], "segment_count": segment_count,
                         "longest_valid_span": summary["longest_valid_span"],
                         "ba_accepted": summary["ba_accepted"], "ba_attempts": summary["ba_attempts"]},
            "files_sha256": hashes}


def capture_replay(root: Path, output: Path) -> dict[str, Any]:
    """Capture actual 2D features/associations; refuse release if replay differs."""
    if output.exists():
        raise FileExistsError(output)
    inputs = validate_sources(root)
    sequence = KittiGraySequence(root / "data/sequences/00")
    config = json.loads((root / "configs/default.json").read_text(encoding="utf-8"))
    tracker = _build_tracker(sequence, config)
    rows = []
    captures: dict[str, Any] = {}
    for frame_id in range(300):
        result = tracker.process(sequence[frame_id])
        row = _plain(result.as_dict())
        rows.append(row)
        if frame_id not in CAPTURE_IDS:
            continue
        event = tracker.diagnostic_event
        visual = tracker.visual_event
        keypoints = []
        associations = []
        inlier_indices: list[int] = []
        provenance = "unavailable"
        if event is not None:
            keypoints = event["keypoints_xy"].tolist()
            associations = [{"feature_index": item["image_feature_index"],
                             "pixel_xy": item["pixel_xy"].tolist(),
                             "point_id": item["point_id"]} for item in event["correspondences"]]
            if visual is None or visual["kind"] != "pnp":
                raise ValueError("PnP diagnostic lacked its image event")
            raw_indices = event["pnp_candidate"].get("raw_inlier_indices")
            inlier_indices = [] if raw_indices is None else [int(value) for value in raw_indices]
            provenance = "existing opt-in tracker PnP diagnostic capture"
        elif visual is not None and visual["kind"] == "initialization":
            keypoints = [list(k.pt) for k in visual["second_keypoints"]]
            provenance = "existing initialization image event; no map PnP on this view"
        if keypoints and len(keypoints) != row["diagnostics"].get("features_detected"):
            raise ValueError(f"ORB coordinate count differs at {frame_id}")
        if event is not None:
            if len(associations) != row["diagnostics"].get("pnp_correspondences"):
                raise ValueError(f"Association coordinate count differs at {frame_id}")
            if row["diagnostics"].get("pnp_inliers") is not None and len(inlier_indices) != row["diagnostics"]["pnp_inliers"]:
                raise ValueError(f"RANSAC inlier coordinates differ at {frame_id}")
        captures[str(frame_id)] = {"frame_id": frame_id, "state": row["state"],
                                   "segment_id": row["segment_id"], "pose_valid": row["pose_valid"],
                                   "keypoints_xy": keypoints, "associations": associations,
                                   "ransac_inlier_indices": inlier_indices, "provenance": provenance,
                                   "pnp_status": row["diagnostics"].get("pnp_status")}
        if (frame_id + 1) % 50 == 0:
            print(f"read-only coordinate capture {frame_id+1}/300", flush=True)
    equivalence = compare_replay(rows, tracker, root / "artifacts/m5b-final-evaluation/run-a1")
    if not equivalence["passed"]:
        raise ValueError(f"Read-only replay changed accepted online outputs: {equivalence['mismatches'][:5]}")
    output.mkdir(parents=True)
    (output / "coordinate_capture.json").write_text(json.dumps(captures, separators=(",", ":")) + "\n", encoding="utf-8")
    (output / "replay_equivalence.json").write_text(json.dumps(equivalence, indent=2) + "\n", encoding="utf-8")
    (output / "input_manifest.json").write_text(json.dumps(inputs, indent=2) + "\n", encoding="utf-8")
    return {"captured_frame_count": len(captures), "equivalence": equivalence["passed"],
            "map_ply_byte_equal": all(v["byte_equal"] for v in equivalence["final_map_ply"].values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(capture_replay(args.root.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
