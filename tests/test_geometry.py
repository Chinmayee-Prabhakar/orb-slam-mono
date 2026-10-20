"""M2 geometry correctness and conservative initialization failures."""

import json
import contextlib
import io
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from orb_slam_mono.geometry import (GeometryConfig, _essential_candidates,
                                    filter_triangulated_points, initialize_two_view)
from orb_slam_mono.geometry_demo import DEFAULT_CONFIG, main, run_geometry_demo
from orb_slam_mono.offline_evaluation import evaluate_synthetic
from orb_slam_mono.synthetic3d import generate_scene, render_projected_landmarks


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = generate_scene()
        cls.result = initialize_two_view(cls.scene.points1_px, cls.scene.points2_px, cls.scene.camera)

    def test_known_scene_pose_and_counts(self):
        scene, result = self.scene, self.result
        self.assertEqual(result.status, "accepted")
        self.assertGreaterEqual(result.diagnostics["essential_inliers"], 350)
        self.assertGreaterEqual(result.diagnostics["accepted_points"], 300)
        delta = result.R_21 @ scene.R_21_truth.T
        rotation_error = np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1)))
        direction = np.dot(result.t_21, scene.t_21_metric_truth) / np.linalg.norm(scene.t_21_metric_truth)
        translation_error = np.degrees(np.arccos(np.clip(direction, -1, 1)))
        self.assertLess(rotation_error, 3.0)
        self.assertLess(translation_error, 8.0)
        self.assertAlmostEqual(np.linalg.norm(result.t_21), 1.0, places=8)
        self.assertLess(np.linalg.norm(result.R_21.T @ result.R_21 - np.eye(3)), 1e-3)
        self.assertAlmostEqual(np.linalg.det(result.R_21), 1.0, places=3)

    def test_world_camera_projection_and_centers(self):
        result, camera = self.result, self.scene.camera
        xyz = result.points_xyz
        idx = np.flatnonzero(result.accepted_mask)
        camera2_xyz = (result.R_21 @ xyz.T).T + result.t_21
        self.assertTrue(np.all(xyz[:, 2] > 0))
        self.assertTrue(np.all(camera2_xyz[:, 2] > 0))
        p1 = (camera.matrix @ xyz.T).T
        p2 = (camera.matrix @ camera2_xyz.T).T
        err1 = np.linalg.norm(p1[:, :2] / p1[:, 2:3] - self.scene.points1_px[idx], axis=1)
        err2 = np.linalg.norm(p2[:, :2] / p2[:, 2:3] - self.scene.points2_px[idx], axis=1)
        self.assertLess(float(np.max(err1)), 3.0)
        self.assertLess(float(np.max(err2)), 3.0)
        center2 = -result.R_21.T @ result.t_21
        np.testing.assert_allclose(result.R_21 @ center2 + result.t_21, np.zeros(3), atol=1e-12)
        self.assertEqual(len(result.points_xyz), int(result.accepted_mask.sum()))

    def test_triangulation_with_known_pose_in_unit_baseline_frame(self):
        scene = self.scene
        unit_t = scene.t_21_metric_truth / np.linalg.norm(scene.t_21_metric_truth)
        p1 = scene.camera.matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        p2 = scene.camera.matrix @ np.column_stack((scene.R_21_truth, unit_t))
        homogeneous = cv2.triangulatePoints(p1, p2, scene.points1_px.T, scene.points2_px.T)
        source = np.arange(len(scene.points1_px))
        filtered = filter_triangulated_points(homogeneous, scene.points1_px, scene.points2_px,
                                              scene.camera, scene.R_21_truth, unit_t, source,
                                              GeometryConfig())
        truth_in_unit_baselines = scene.points_world[filtered.accepted_source_indices] / np.linalg.norm(scene.t_21_metric_truth)
        error = np.linalg.norm(filtered.points_xyz - truth_in_unit_baselines, axis=1)
        self.assertGreater(len(error), 400)
        self.assertLess(float(np.median(error)), 0.10)

    def test_filter_preserves_indices_and_rejects_bad_reprojection(self):
        scene = self.scene
        unit_t = scene.t_21_metric_truth / np.linalg.norm(scene.t_21_metric_truth)
        points1 = scene.points1_px[:8]
        points2 = scene.points2_px[:8]
        p1 = scene.camera.matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        p2 = scene.camera.matrix @ np.column_stack((scene.R_21_truth, unit_t))
        homogeneous = cv2.triangulatePoints(p1, p2, points1.T, points2.T)
        corrupted = points2.copy()
        corrupted[3] += [35, 20]
        source = np.arange(100, 108)
        filtered = filter_triangulated_points(homogeneous, points1, corrupted, scene.camera,
                                              scene.R_21_truth, unit_t, source, GeometryConfig())
        self.assertFalse(filtered.accepted_local_mask[3])
        self.assertNotIn(103, filtered.accepted_source_indices)
        self.assertEqual(len(filtered.accepted_local_mask), 8)
        self.assertTrue(np.all(np.isin(filtered.accepted_source_indices, source)))

    def test_each_triangulation_filter_stage_and_source_identity(self):
        scene = self.scene
        unit_t = scene.t_21_metric_truth / np.linalg.norm(scene.t_21_metric_truth)
        points1, points2 = scene.points1_px[:8].copy(), scene.points2_px[:8].copy()
        p1 = scene.camera.matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        p2 = scene.camera.matrix @ np.column_stack((scene.R_21_truth, unit_t))
        h = cv2.triangulatePoints(p1, p2, points1.T, points2.T)
        h[:, 0] = [np.nan, 0, 1, 1]  # invalid homogeneous point
        h[:, 1] = [0, 0, -5, 1]      # negative depth in camera 1
        h[:, 2] = [100, 0, 1, 1]     # positive first depth, negative second depth
        h[:, 3] = [0, 0, 500, 1]     # beyond maximum depth in baseline units
        ray1 = np.array([(points1[4, 0] - scene.camera.cx) / scene.camera.fx,
                         (points1[4, 1] - scene.camera.cy) / scene.camera.fy, 1.0])
        rotated = scene.R_21_truth @ ray1
        points2[4] = (scene.camera.matrix @ rotated)[:2] / rotated[2]  # zero parallax
        points1[5] += [35, 20]       # first-view reprojection error
        points2[6] += [35, 20]       # second-view reprojection error
        source = np.arange(100, 108)
        filtered = filter_triangulated_points(h, points1, points2, scene.camera,
                                              scene.R_21_truth, unit_t, source, GeometryConfig())
        counts = filtered.counts
        stages = [counts[key] for key in ("triangulation_candidates", "homogeneous_valid",
                                           "positive_depth", "depth_range", "parallax_pass",
                                           "reprojection_pass", "accepted_points")]
        self.assertEqual(stages, [8, 7, 5, 4, 3, 1, 1])
        self.assertTrue(all(left >= right for left, right in zip(stages, stages[1:])))
        self.assertEqual(filtered.accepted_source_indices.tolist(), [107])
        self.assertEqual(np.flatnonzero(filtered.accepted_local_mask).tolist(), [7])
        self.assertLess((scene.R_21_truth @ np.array([100., 0., 1.]) + unit_t)[2], 0)

    def test_rejected_provisional_points_are_not_map(self):
        scene = self.scene
        result = initialize_two_view(scene.points1_px, scene.points2_px, scene.camera,
                                     GeometryConfig(min_triangulated_points=430))
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.failure_reason, "insufficient_triangulated_points")
        self.assertEqual(result.diagnostics["accepted_points"], 427)
        self.assertEqual(result.diagnostics["usable_map_points"], 0)
        self.assertEqual(len(result.points_xyz), 0)
        self.assertEqual(int(result.accepted_mask.sum()), 0)
        self.assertIsNone(result.R_21)
        self.assertIsNone(result.t_21)
        self.assertIsNotNone(result.debug_provisional_R_21)
        self.assertFalse(result.as_dict()["valid_for_mapping"])

    def test_rejected_rerun_overwrites_only_own_artifacts(self):
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            good = run_geometry_demo("synthetic", folder, config)
            self.assertEqual(good["initialization_status"], "accepted")
            self.assertIn("element vertex 427", (folder / "landmarks.ply").read_text())
            prior_map_png = (folder / "sparse_map.png").read_bytes()
            unrelated = folder / "keep_me.txt"
            unrelated.write_text("user data", encoding="utf-8")
            config["geometry"]["min_triangulated_points"] = 430
            bad = run_geometry_demo("synthetic", folder, config)
            self.assertEqual(bad["initialization_status"], "rejected")
            self.assertEqual(bad["diagnostics"]["accepted_points"], 427)
            self.assertEqual(bad["diagnostics"]["usable_map_points"], 0)
            self.assertFalse(bad["valid_for_mapping"])
            self.assertIsNone(bad["R_21"])
            self.assertEqual(unrelated.read_text(encoding="utf-8"), "user data")
            ply = (folder / "landmarks.ply").read_text(encoding="utf-8")
            self.assertIn("element vertex 0", ply)
            self.assertIn("initialization_status rejected", ply)
            poses = json.loads((folder / "poses.json").read_text(encoding="utf-8"))
            self.assertFalse(poses["valid_for_mapping"])
            self.assertIsNone(poses["poses_T_cw"][1]["matrix"])
            self.assertNotEqual((folder / "sparse_map.png").read_bytes(), prior_map_png)

    def test_malformed_homogeneous_output_is_structured(self):
        scene = self.scene
        for bad in (np.ones((3, 5)), np.ones((4, 5), dtype=object) * "bad",
                    np.full((4, 5), np.nan), np.vstack((np.ones((3, 5)), np.zeros((1, 5))))):
            result = filter_triangulated_points(bad, scene.points1_px[:5], scene.points2_px[:5],
                                                scene.camera, np.eye(3), np.array([1., 0., 0.]),
                                                np.arange(5), GeometryConfig())
            self.assertEqual(result.counts["accepted_points"], 0)

    def test_essential_candidate_shapes(self):
        e = np.eye(3)
        self.assertEqual(len(_essential_candidates(np.vstack((e, 2 * e)))), 2)
        self.assertEqual(len(_essential_candidates(np.hstack((e, 2 * e)))), 2)
        self.assertEqual(_essential_candidates(np.ones((4, 4))), [])
        self.assertEqual(_essential_candidates(np.full((3, 3), np.nan)), [])

    def test_empty_malformed_and_nonfinite_input(self):
        scene = self.scene
        empty = initialize_two_view(np.empty((0, 2)), np.empty((0, 2)), scene.camera)
        self.assertEqual(empty.failure_reason, "insufficient_correspondences")
        self.assertIsNone(empty.diagnostics["essential_inliers"])
        self.assertEqual(initialize_two_view(scene.points1_px[:5], scene.points2_px[:5],
                                             scene.camera).failure_reason, "insufficient_correspondences")
        for first, second in ((np.ones((8, 3)), np.ones((8, 3))),
                              (np.ones((8, 2)), np.ones((7, 2))),
                              (np.full((8, 2), np.nan), np.ones((8, 2)))):
            self.assertEqual(initialize_two_view(first, second, scene.camera).failure_reason, "invalid_input")
        self.assertEqual(initialize_two_view(scene.points1_px, scene.points2_px, np.eye(3)).failure_reason,
                         "invalid_calibration")

    def test_pure_rotation_and_low_baseline_reject(self):
        for kwargs in ({"baseline": 0.0, "noise_px": 0.0},
                       {"baseline": 0.005, "noise_px": 0.35}):
            scene = generate_scene(**kwargs)
            result = initialize_two_view(scene.points1_px, scene.points2_px, scene.camera)
            self.assertEqual(result.status, "rejected")
            self.assertIn(result.failure_reason, ("rotation_or_planar_degeneracy", "low_parallax",
                                                  "insufficient_pose_inliers", "insufficient_essential_inliers",
                                                  "low_cheirality_fraction", "rotation_only_explains_matches"))

    def test_excess_outliers_reject(self):
        scene = generate_scene(outlier_fraction=0.85)
        result = initialize_two_view(scene.points1_px, scene.points2_px, scene.camera)
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.failure_reason, "insufficient_essential_inliers")

    def test_reviewed_weak_baseline_is_not_false_accepted(self):
        scene = generate_scene(seed=7, baseline=0.15, noise_px=0.35)
        result = initialize_two_view(scene.points1_px, scene.points2_px, scene.camera)
        self.assertEqual(result.failure_reason, "low_cheirality_fraction")
        self.assertLess(result.diagnostics["pose_inlier_fraction"], 0.9)
        self.assertEqual(len(result.points_xyz), 0)
        self.assertIsNone(result.t_21)

    def test_rotation_only_diagnostic_rejects_spurious_pure_rotation(self):
        scene = generate_scene(seed=23, baseline=0.0, noise_px=0.0)
        result = initialize_two_view(scene.points1_px, scene.points2_px, scene.camera)
        self.assertEqual(result.failure_reason, "rotation_only_explains_matches")
        self.assertLess(result.diagnostics["rotation_only_median_residual_px"], 1.5)
        self.assertEqual(result.diagnostics["usable_map_points"], 0)

    def test_scale_only_offline_evaluation_is_not_sim3(self):
        scene, result = self.scene, self.result
        estimate_before = result.points_xyz.copy()
        evaluation = evaluate_synthetic(result, scene)
        alignment = evaluation["scale_only_alignment"]
        self.assertIn("not Umeyama Sim(3)", alignment["definition"])
        truth = scene.points_world[result.accepted_mask]
        estimate = result.points_xyz
        expected_scale = float(np.sum(estimate * truth) / np.sum(estimate * estimate))
        self.assertAlmostEqual(alignment["fitted_scale_scene_units_per_estimated_unit"], expected_scale)
        self.assertAlmostEqual(alignment["median_3d_error_scene_units"],
                               evaluation["median_3d_error_after_offline_scale_alignment"])
        self.assertLess(evaluation["true_pose_triangulation_reference"]["median_3d_error"], 0.1)
        self.assertGreater(alignment["median_3d_error_scene_units"],
                           evaluation["true_pose_triangulation_reference"]["median_3d_error"])
        np.testing.assert_array_equal(result.points_xyz, estimate_before)
        self.assertEqual(list(inspect.signature(initialize_two_view).parameters),
                         ["points1_px", "points2_px", "camera", "config"])

    def test_fixed_nominal_seeds_stay_accepted(self):
        for seed in (7, 11, 19, 23, 31):
            scene = generate_scene(seed=seed, baseline=0.8, noise_px=0.35)
            result = initialize_two_view(scene.points1_px, scene.points2_px, scene.camera)
            self.assertEqual(result.status, "accepted", f"seed {seed}: {result.failure_reason}")
            self.assertGreater(result.diagnostics["pose_inlier_fraction"], 0.9)

    def test_excessive_reprojection_error_reject(self):
        scene = self.scene
        result = initialize_two_view(scene.points1_px, scene.points2_px, scene.camera,
                                     GeometryConfig(max_reprojection_error_px=1e-5))
        self.assertEqual(result.failure_reason, "excessive_reprojection_error")
        self.assertLess(result.diagnostics["accepted_points"], 25)

    def test_malformed_essential_and_failed_pose_reject(self):
        scene = self.scene
        with patch("orb_slam_mono.geometry.cv2.findEssentialMat", return_value=(np.ones((4, 4)),
                                                                          np.ones((len(scene.points1_px), 1)))):
            result = initialize_two_view(scene.points1_px, scene.points2_px, scene.camera)
            self.assertEqual(result.failure_reason, "essential_estimation_failed")
        with patch("orb_slam_mono.geometry.cv2.recoverPose", side_effect=cv2.error("pose failed")):
            result = initialize_two_view(scene.points1_px, scene.points2_px, scene.camera)
            self.assertEqual(result.failure_reason, "pose_recovery_failed")

    def test_other_fixed_seeds(self):
        for seed in (2, 19):
            scene = generate_scene(seed=seed)
            result = initialize_two_view(scene.points1_px, scene.points2_px, scene.camera)
            self.assertEqual(result.status, "accepted")
            self.assertGreater(result.diagnostics["accepted_points"], 300)

    def test_synthetic_image_orb_path_and_exports(self):
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            report = run_geometry_demo("synthetic-images", folder, config)
            self.assertEqual(report["initialization_status"], "accepted")
            self.assertGreater(report["retained_matches"], 100)
            self.assertLessEqual(report["diagnostics"]["pose_inliers"], report["diagnostics"]["essential_inliers"])
            self.assertTrue((folder / "landmarks.ply").read_text().startswith("ply\n"))
            self.assertIn(f"element vertex {report['diagnostics']['accepted_points']}",
                          (folder / "landmarks.ply").read_text(encoding="utf-8"))
            poses = json.loads((folder / "poses.json").read_text(encoding="utf-8"))
            self.assertIn("T_cw", poses["transform_convention"])
            transform = np.array(poses["poses_T_cw"][1]["matrix"])
            center = np.array(poses["camera_centers_world"][1])
            np.testing.assert_allclose(transform[:3, :3] @ center + transform[:3, 3],
                                       np.zeros(3), atol=1e-12)
            for name in report["outputs"]:
                self.assertTrue((folder / name).is_file())

    def test_kitti_missing_data(self):
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                run_geometry_demo("kitti", Path(directory) / "output", config,
                                  Path(directory) / "absent")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exit_status:
                    main(["--mode", "kitti", "--kitti-sequence", str(Path(directory) / "absent"),
                          "--output-dir", str(Path(directory) / "output")])
            self.assertEqual(exit_status.exception.code, 2)

    def test_kitti_layout_cli_path_with_rendered_fixture(self):
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        scene = self.scene
        images = render_projected_landmarks(scene)
        with tempfile.TemporaryDirectory() as directory:
            sequence = Path(directory) / "sequences" / "00"
            image_dir = sequence / "image_0"
            image_dir.mkdir(parents=True)
            for index, image in enumerate(images):
                self.assertTrue(cv2.imwrite(str(image_dir / f"{index:06d}.png"), image))
            k = scene.camera.matrix
            projection = np.column_stack((k, np.zeros(3)))
            (sequence / "calib.txt").write_text(
                "P0: " + " ".join(str(value) for value in projection.reshape(-1)) + "\n",
                encoding="utf-8")
            report = run_geometry_demo("kitti", Path(directory) / "result", config,
                                       sequence, 0, 1)
            self.assertEqual(report["dataset"], "kitti_odometry_gray")
            self.assertEqual(report["initialization_status"], "accepted")
            self.assertEqual(report["first_index"], 0)
            self.assertEqual(report["second_index"], 1)


if __name__ == "__main__":
    unittest.main()
