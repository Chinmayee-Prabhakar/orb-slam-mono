"""Run M2 two-view initialization on physical synthetic data or KITTI images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np

from .datasets import KittiGraySequence
from .features import OrbExtractor
from .geometry import GeometryConfig, InitializationResult, initialize_two_view
from .matching import DescriptorMatcher
from .offline_evaluation import evaluate_synthetic
from .synthetic3d import generate_scene, render_projected_landmarks

DEFAULT_CONFIG = Path(__file__).absolute().parents[2] / "configs" / "default.json"


def _camera_dict(camera: Any) -> Dict[str, Any]:
    return {key: getattr(camera, key) for key in ("width", "height", "fx", "fy", "cx", "cy")}


def _draw_correspondences(first: np.ndarray, second: np.ndarray, points1: np.ndarray,
                          points2: np.ndarray, mask: np.ndarray, destination: Path,
                          max_drawn: int, result: InitializationResult) -> None:
    left = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(second, cv2.COLOR_GRAY2BGR)
    canvas = np.concatenate((left, right), axis=1)
    count = len(points1)
    if count:
        selected = np.linspace(0, count - 1, min(count, max_drawn), dtype=int)
        for index in selected:
            color = (60, 220, 60) if mask[index] else (70, 70, 230)
            a = tuple(np.rint(points1[index]).astype(int))
            b = tuple(np.rint(points2[index] + [first.shape[1], 0]).astype(int))
            cv2.line(canvas, a, b, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, a, 3, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, b, 3, color, 1, cv2.LINE_AA)
    if result.status == "rejected":
        cv2.putText(canvas, f"REJECTED - diagnostic only: {result.failure_reason}",
                    (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(destination), canvas):
        raise OSError(f"Could not save {destination}")


def _write_ply(path: Path, result: InitializationResult) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("ply\nformat ascii 1.0\ncomment camera-1 world frame; arbitrary unit baseline\n")
        output.write(f"comment initialization_status {result.status}\n")
        if result.failure_reason:
            output.write(f"comment failure_reason {result.failure_reason}\n")
        output.write(f"element vertex {len(result.points_xyz)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        for xyz in result.points_xyz:
            output.write(f"{xyz[0]:.8g} {xyz[1]:.8g} {xyz[2]:.8g}\n")


def _write_poses(path: Path, result: InitializationResult) -> None:
    identity = np.eye(4)
    second = None
    if result.R_21 is not None and result.t_21 is not None:
        second = identity.copy()
        second[:3, :3] = result.R_21
        second[:3, 3] = result.t_21
    data = {"initialization_status": result.status, "failure_reason": result.failure_reason,
            "valid_for_mapping": result.status == "accepted",
            "coordinate_frame": "camera 1 optical frame: +x right, +y down, +z forward",
            "transform_convention": "T_cw maps world to camera; X_c = R_cw X_w + t_cw",
            "scale": "arbitrary; ||t_21|| = 1; never metric",
            "poses_T_cw": [{"camera": 1, "matrix": identity.tolist()},
                           {"camera": 2, "matrix": None if second is None else second.tolist()}],
            "camera_centers_world": [[0.0, 0.0, 0.0], None if second is None
                                     else (-result.R_21.T @ result.t_21).tolist()]}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _draw_map(path: Path, result: InitializationResult, camera: Any) -> None:
    canvas = np.full((800, 1100, 3), 250, np.uint8)
    title = ("Sparse two-view map - arbitrary monocular scale" if result.status == "accepted"
             else "Initialization rejected - no usable map")
    cv2.putText(canvas, title, (30, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (35, 35, 35), 2, cv2.LINE_AA)
    subtitle = ("Oblique view: x right, z diagonal, y vertical" if result.status == "accepted"
                else "Diagnostic attempt; no 3D reconstruction exported")
    cv2.putText(canvas, subtitle, (30, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (70, 70, 70), 1, cv2.LINE_AA)
    if result.R_21 is None or result.t_21 is None:
        cv2.putText(canvas, f"REJECTED: {result.failure_reason}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 160), 2, cv2.LINE_AA)
        cv2.putText(canvas, "No usable camera-2 pose or landmarks", (30, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 160), 1, cv2.LINE_AA)
    else:
        center2 = -result.R_21.T @ result.t_21
        corners = np.array([[0, 0, 1], [-camera.cx / camera.fx, -camera.cy / camera.fy, 1],
                            [(camera.width - camera.cx) / camera.fx, -camera.cy / camera.fy, 1],
                            [(camera.width - camera.cx) / camera.fx,
                             (camera.height - camera.cy) / camera.fy, 1],
                            [-camera.cx / camera.fx, (camera.height - camera.cy) / camera.fy, 1]], dtype=float)
        rays1 = corners[1:] * 0.6
        rays2 = center2 + (result.R_21.T @ rays1.T).T
        world = np.vstack((result.points_xyz, [0, 0, 0], center2, rays1, rays2))
        view = np.column_stack((world[:, 0] - 0.35 * world[:, 2],
                                0.58 * world[:, 2] - 0.85 * world[:, 1]))
        lo, hi = np.min(view, axis=0), np.max(view, axis=0)
        span = np.maximum(hi - lo, 1.0)
        factor = min(1000 / span[0], 620 / span[1])
        pixel = np.rint((view - lo) * factor + [50, 115]).astype(int)
        point_count = len(result.points_xyz)
        for i, point in enumerate(pixel[:point_count]):
            y_color = int(np.clip(130 + 25 * result.points_xyz[i, 1], 40, 220))
            cv2.circle(canvas, tuple(point), 2, (180, y_color, 35), -1, cv2.LINE_AA)
        cam1, cam2 = pixel[point_count:point_count + 2]
        for origin, start in ((cam1, point_count + 2), (cam2, point_count + 6)):
            corners_px = pixel[start:start + 4]
            for corner in corners_px:
                cv2.line(canvas, tuple(origin), tuple(corner), (200, 80, 40), 2, cv2.LINE_AA)
            cv2.polylines(canvas, [corners_px], True, (200, 80, 40), 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(cam1), 6, (15, 15, 190), -1)
        cv2.circle(canvas, tuple(cam2), 6, (190, 90, 15), -1)
        for label, point, offset in (("camera 1", cam1, [-115, -18]),
                                     ("camera 2", cam2, [12, -18])):
            cv2.putText(canvas, label, tuple(point + offset), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (30, 30, 30), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Could not save {path}")


def run_geometry_demo(mode: str, output_dir: Path, config: Dict[str, Any],
                      kitti_sequence: Optional[Path] = None, first_index: int = 0,
                      second_index: int = 5, seed: int = 7) -> Dict[str, Any]:
    started = perf_counter()
    scene = None
    if mode == "kitti":
        if kitti_sequence is None:
            raise ValueError("--kitti-sequence is required in kitti mode")
        sequence = KittiGraySequence(kitti_sequence)
        if not 0 <= first_index < second_index < len(sequence):
            raise IndexError("Require 0 <= first-index < second-index < KITTI sequence length")
        sample1, sample2 = sequence[first_index], sequence[second_index]
        first, second, camera = sample1.image, sample2.image, sequence.camera
        source1, source2 = sample1.source, sample2.source
    elif mode in ("synthetic", "synthetic-images"):
        scene = generate_scene(seed=seed)
        first, second = render_projected_landmarks(scene)
        camera = scene.camera
        source1, source2 = "synthetic_3d_view_1", "synthetic_3d_view_2"
        first_index, second_index = 0, 1
    else:
        raise ValueError(f"Unknown mode: {mode}")

    feature_counts: Dict[str, Any] = {"first_features": None, "second_features": None,
                                      "knn_pairs": None, "ratio_passed": None,
                                      "retained_matches": None}
    stage_ms: Dict[str, Any] = {"orb_and_matching": None}
    if mode == "synthetic":
        assert scene is not None
        points1, points2 = scene.points1_px, scene.points2_px
    else:
        feature_started = perf_counter()
        extractor = OrbExtractor(nfeatures=config["orb"]["nfeatures"],
                                 scale_factor=config["orb"]["scaleFactor"],
                                 levels=config["orb"]["nlevels"],
                                 fast_threshold=config["orb"]["fastThreshold"])
        f1, f2 = extractor.extract(first), extractor.extract(second)
        matches = DescriptorMatcher(**config["matching"]).match(f1.descriptors, f2.descriptors)
        points1 = np.array([f1.keypoints[m.queryIdx].pt for m in matches.matches], dtype=float).reshape(-1, 2)
        points2 = np.array([f2.keypoints[m.trainIdx].pt for m in matches.matches], dtype=float).reshape(-1, 2)
        feature_counts = {"first_features": f1.count, "second_features": f2.count,
                          "knn_pairs": matches.knn_pairs, "ratio_passed": matches.ratio_passed,
                          "retained_matches": matches.retained}
        stage_ms["orb_and_matching"] = round((perf_counter() - feature_started) * 1000, 3)

    geometry = initialize_two_view(points1, points2, camera, GeometryConfig(**config["geometry"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    max_drawn = int(config["visualization"]["max_drawn_matches"])
    _draw_correspondences(first, second, points1, points2, geometry.essential_mask,
                          output_dir / "essential_correspondences.png", max_drawn, geometry)
    _draw_correspondences(first, second, points1, points2, geometry.pose_mask,
                          output_dir / "pose_correspondences.png", max_drawn, geometry)
    _draw_correspondences(first, second, points1, points2, geometry.accepted_mask,
                          output_dir / "triangulated_correspondences.png", max_drawn, geometry)
    _draw_map(output_dir / "sparse_map.png", geometry, camera)
    _write_ply(output_dir / "landmarks.ply", geometry)
    _write_poses(output_dir / "poses.json", geometry)
    stage_ms.update(geometry.timing_ms)
    stage_ms["end_to_end_total"] = round((perf_counter() - started) * 1000, 3)
    report = {"dataset": "kitti_odometry_gray" if mode == "kitti" else "synthetic_3d",
              "mode": mode, "first_source": source1, "second_source": source2,
              "first_index": first_index, "second_index": second_index,
              "camera": _camera_dict(camera), **feature_counts,
              **geometry.as_dict(), "timing_ms": stage_ms,
              "synthetic_scene": None if scene is None else
              {"seed": scene.seed, "noise_px": scene.noise_px, "visible_landmarks": len(scene.points_world),
               "translation_baseline_metric_truth_for_evaluation_only": scene.baseline},
              "offline_synthetic_evaluation": None if scene is None else
              evaluate_synthetic(geometry, scene, point_identity_known=(mode == "synthetic")),
              "outputs": ["essential_correspondences.png", "pose_correspondences.png",
                          "triangulated_correspondences.png",
                          "sparse_map.png", "landmarks.ply", "poses.json", "stats.json"]}
    (output_dir / "stats.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "synthetic-images", "kitti"), default="synthetic")
    parser.add_argument("--kitti-sequence", type=Path)
    parser.add_argument("--first-index", type=int, default=0)
    parser.add_argument("--second-index", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/geometry-synthetic"))
    args = parser.parse_args(argv)
    try:
        configuration = json.loads(args.config.read_text(encoding="utf-8"))
        report = run_geometry_demo(args.mode, args.output_dir, configuration,
                                   args.kitti_sequence, args.first_index, args.second_index, args.seed)
    except (FileNotFoundError, ValueError, IndexError) as error:
        parser.error(str(error))
    summary = {key: report[key] for key in ("dataset", "mode", "first_index", "second_index",
                                            "first_features", "second_features", "retained_matches",
                                            "initialization_status", "failure_reason")}
    summary.update(report["diagnostics"])
    summary["offline_synthetic_evaluation"] = report["offline_synthetic_evaluation"]
    print(json.dumps(summary, indent=2))
    print(f"Saved M2 results to {args.output_dir.absolute()}")


if __name__ == "__main__":
    main()
