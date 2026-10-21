"""Order-stability gate checks without supplying truth to the initializer."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from orb_slam_mono.camera import CalibratedCamera
from orb_slam_mono.datasets import KittiGraySequence
from orb_slam_mono.features import OrbExtractor
from orb_slam_mono.geometry import GeometryConfig, InitializationResult, initialize_two_view
from orb_slam_mono.matching import DescriptorMatcher
from orb_slam_mono.synthetic3d import generate_scene


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = PROJECT_ROOT / "data" / "sequences" / "00"


class PoseStabilityTests(unittest.TestCase):
    @staticmethod
    def _mock_initialization(accepted=True, angle_deg=0):
        rotation, _ = cv2.Rodrigues(np.array([0.0, 0.0, np.deg2rad(angle_deg)]))
        mask = np.array([True, False, True, False, True, False, False, False])
        return InitializationResult(
            status="accepted" if accepted else "rejected",
            failure_reason=None if accepted else "low_parallax",
            R_21=rotation if accepted else None,
            t_21=np.array([1.0, 0.0, 0.0]) if accepted else None,
            debug_provisional_R_21=None if accepted else rotation,
            debug_provisional_t_21=None if accepted else np.array([1.0, 0.0, 0.0]),
            essential_mask=np.ones(8, dtype=bool),
            pose_mask=np.ones(8, dtype=bool),
            accepted_mask=mask.copy() if accepted else np.zeros(8, dtype=bool),
            points_xyz=np.ones((3, 3)) if accepted else np.empty((0, 3)),
            diagnostics={"usable_map_points": 3 if accepted else 0},
            timing_ms={"total": 1.0},
        )

    def test_accepted_trial_minimum_with_provisional_rotations(self):
        camera = CalibratedCamera(1280, 720, 700, 700, 640, 360)
        points = np.zeros((8, 2))
        for accepted_count in (0, 3, 4):
            with self.subTest(accepted_trials=accepted_count):
                original = self._mock_initialization()
                trials = [self._mock_initialization(i < accepted_count)
                          for i in range(8)]
                with patch("orb_slam_mono.geometry._initialize_two_view_once",
                           side_effect=[original, *trials]) as mocked:
                    result = initialize_two_view(points, points, camera)
                self.assertEqual(mocked.call_count, 9)
                self.assertEqual(result.diagnostics["pose_stability_valid_candidates"], 8)
                self.assertEqual(result.diagnostics["pose_stability_provisional_candidates"],
                                 8 - accepted_count)
                self.assertEqual(result.diagnostics["pose_stability_accepted_trials"],
                                 accepted_count)
                self.assertAlmostEqual(result.diagnostics[
                    "pose_stability_median_rotation_disagreement_deg"], 0)
                if accepted_count < 4:
                    self.assertEqual(result.status, "rejected")
                    self.assertEqual(result.failure_reason,
                                     "insufficient_accepted_pose_stability_trials")
                    self.assertIsNone(result.R_21)
                    self.assertIsNone(result.t_21)
                    self.assertEqual(len(result.points_xyz), 0)
                    self.assertFalse(result.accepted_mask.any())
                    self.assertEqual(result.diagnostics["usable_map_points"], 0)
                    self.assertIsNotNone(result.debug_provisional_R_21)
                else:
                    self.assertEqual(result.status, "accepted")
                    self.assertIsNone(result.failure_reason)
                    np.testing.assert_array_equal(result.R_21, original.R_21)
                    np.testing.assert_array_equal(result.t_21, original.t_21)
                    np.testing.assert_array_equal(result.points_xyz, original.points_xyz)
                    np.testing.assert_array_equal(result.essential_mask, original.essential_mask)
                    np.testing.assert_array_equal(result.pose_mask, original.pose_mask)
                    np.testing.assert_array_equal(result.accepted_mask, original.accepted_mask)

    def test_accepted_trials_do_not_override_rotation_disagreement(self):
        camera = CalibratedCamera(1280, 720, 700, 700, 640, 360)
        points = np.zeros((8, 2))
        original = self._mock_initialization()
        trials = [self._mock_initialization(angle_deg=30) for _ in range(8)]
        with patch("orb_slam_mono.geometry._initialize_two_view_once",
                   side_effect=[original, *trials]):
            result = initialize_two_view(points, points, camera)
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.failure_reason, "unstable_pose_under_resampling")
        self.assertEqual(result.diagnostics["pose_stability_accepted_trials"], 8)
        self.assertGreater(result.diagnostics[
            "pose_stability_median_rotation_disagreement_deg"], 10)
        self.assertIsNone(result.R_21)
        self.assertFalse(result.accepted_mask.any())

    def test_disabled_stability_keeps_one_shot_result(self):
        camera = CalibratedCamera(1280, 720, 700, 700, 640, 360)
        points = np.zeros((8, 2))
        original = self._mock_initialization()
        with patch("orb_slam_mono.geometry._initialize_two_view_once",
                   return_value=original) as mocked:
            result = initialize_two_view(points, points, camera,
                                         GeometryConfig(pose_stability_trials=0))
        self.assertIs(result, original)
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(result.status, "accepted")
        self.assertNotIn("pose_stability_accepted_trials", result.diagnostics)

    @staticmethod
    def _projected_case(center, angle_deg, layout):
        rng = np.random.default_rng(11)
        camera = CalibratedCamera(1280, 720, 700, 700, 640, 360)
        x = rng.uniform(-3, 3, 650)
        y = rng.uniform(-1.5, 1.5, 650)
        z = rng.uniform(4, 12, 650)
        if layout == "nonuniform":
            x = rng.uniform(0.4, 3, 650)
            y = rng.uniform(-0.2, 0.8, 650)
        elif layout == "planar":
            z = np.full(650, 8.0)
        elif layout == "near_planar":
            z = 8 + rng.normal(0, 0.1, 650)
        points = np.column_stack((x, y, z))
        rotation, _ = cv2.Rodrigues(np.deg2rad([0.2, angle_deg, -0.3]))
        translation = -rotation @ np.asarray(center, dtype=float)
        second_points = (rotation @ points.T).T + translation

        def project(array):
            return np.column_stack((700 * array[:, 0] / array[:, 2] + 640,
                                    700 * array[:, 1] / array[:, 2] + 360))

        first, second = project(points), project(second_points)
        visible = ((second_points[:, 2] > 0)
                   & (first[:, 0] > 24) & (first[:, 0] < 1256)
                   & (first[:, 1] > 24) & (first[:, 1] < 696)
                   & (second[:, 0] > 24) & (second[:, 0] < 1256)
                   & (second[:, 1] > 24) & (second[:, 1] < 696))
        count = int(visible.sum())
        first = first[visible] + rng.normal(0, 0.35, (count, 2))
        second = second[visible] + rng.normal(0, 0.35, (count, 2))
        return first, second, camera, rotation

    def test_large_camera_rotation_synthetic_scene_remains_accepted(self):
        scene = generate_scene(rotation_degrees=20.0)
        result = initialize_two_view(scene.points1_px, scene.points2_px, scene.camera)
        self.assertEqual(result.status, "accepted")
        self.assertGreaterEqual(result.diagnostics["pose_stability_valid_candidates"], 4)
        self.assertLess(result.diagnostics["pose_stability_median_rotation_disagreement_deg"], 10)
        error = np.degrees(np.arccos(np.clip(
            (np.trace(result.R_21 @ scene.R_21_truth.T) - 1) / 2, -1, 1)))
        self.assertLess(error, 3)

    def test_invalid_stability_configuration(self):
        with self.assertRaises(ValueError):
            GeometryConfig(pose_stability_trials=3, min_pose_stability_candidates=4)
        with self.assertRaises(ValueError):
            GeometryConfig(max_median_rotation_disagreement_deg=0)
        for invalid in (0, -1, 9, 1.5, True):
            with self.subTest(min_accepted_trials=invalid), self.assertRaises(ValueError):
                GeometryConfig(min_pose_stability_accepted_trials=invalid)

    def test_controlled_motion_and_spatial_counterexamples(self):
        cases = (
            ("small_translation_moderate_rotation", [0.15, 0.006, 0.003], 10, "general", "conservative_rejection"),
            ("strong_forward_motion", [0, 0, 0.8], 3, "general", "accepted"),
            ("lateral_motion", [0.8, 0, 0], 3, "general", "accepted"),
            ("nonuniform_features", [0.8, 0, 0], 3, "nonuniform", "rejected"),
            ("planar", [0.8, 0, 0], 3, "planar", "conservative_rejection"),
            ("near_planar", [0.8, 0, 0], 3, "near_planar", "accepted"),
        )
        for name, center, angle, layout, expected in cases:
            with self.subTest(name=name):
                first, second, camera, truth_rotation = self._projected_case(center, angle, layout)
                baseline = initialize_two_view(first, second, camera,
                                               GeometryConfig(pose_stability_trials=0))
                result = initialize_two_view(first, second, camera)
                self.assertEqual(baseline.status, "accepted")
                self.assertEqual(result.status,
                                 "rejected" if expected == "conservative_rejection" else expected)
                if expected == "accepted":
                    self.assertLess(result.diagnostics[
                        "pose_stability_median_rotation_disagreement_deg"], 10)
                    self.assertGreaterEqual(result.diagnostics[
                        "pose_stability_accepted_trials"], 4)
                elif expected == "conservative_rejection":
                    true_translation = -truth_rotation @ np.asarray(center, dtype=float)
                    rotation_error = np.degrees(np.arccos(np.clip(
                        (np.trace(baseline.R_21 @ truth_rotation.T) - 1) / 2, -1, 1)))
                    direction_error = np.degrees(np.arccos(np.clip(
                        np.dot(baseline.t_21, true_translation)
                        / (np.linalg.norm(baseline.t_21) * np.linalg.norm(true_translation)),
                        -1, 1)))
                    if name == "small_translation_moderate_rotation":
                        self.assertLess(rotation_error, 2)
                        self.assertGreater(direction_error, 8)
                        self.assertLess(direction_error, 13)
                    else:
                        self.assertGreater(rotation_error, 2)
                        self.assertLess(rotation_error, 5)
                        self.assertGreater(direction_error, 20)
                        self.assertLess(direction_error, 30)
                    counts = result.diagnostics
                    self.assertEqual(result.failure_reason,
                                     "insufficient_accepted_pose_stability_trials")
                    self.assertGreaterEqual(counts["pose_stability_valid_candidates"], 4)
                    self.assertLess(counts["pose_stability_median_rotation_disagreement_deg"], 10)
                    self.assertGreater(counts["pose_stability_accepted_trials"], 0)
                    self.assertLess(counts["pose_stability_accepted_trials"], 4)
                    self.assertGreater(counts["pose_stability_provisional_candidates"], 0)
                    self.assertIsNone(result.R_21)
                    self.assertIsNone(result.t_21)
                    self.assertEqual(len(result.points_xyz), 0)
                    self.assertFalse(result.accepted_mask.any())
                    self.assertEqual(counts["usable_map_points"], 0)

                    # Truth scores these fixed observations after estimation;
                    # it is never passed to any initialization trial.
                    rejected_direction_errors = []
                    rng = np.random.default_rng(GeometryConfig().pose_stability_seed)
                    for _ in range(GeometryConfig().pose_stability_trials):
                        order = rng.permutation(len(first))
                        trial = initialize_two_view(first[order], second[order], camera,
                                                    GeometryConfig(pose_stability_trials=0))
                        if trial.status != "rejected" or trial.debug_provisional_t_21 is None:
                            continue
                        direction = trial.debug_provisional_t_21
                        error = np.degrees(np.arccos(np.clip(
                            np.dot(direction, true_translation)
                            / (np.linalg.norm(direction) * np.linalg.norm(true_translation)),
                            -1, 1)))
                        rejected_direction_errors.append(error)
                    if name == "small_translation_moderate_rotation":
                        self.assertTrue(any(error > 150 for error in rejected_direction_errors))
                    else:
                        self.assertGreaterEqual(sum(error > 80 for error in rejected_direction_errors), 3)
                else:
                    self.assertEqual(result.failure_reason, "unstable_pose_under_resampling")
                    error = np.degrees(np.arccos(np.clip(
                        (np.trace(baseline.R_21 @ truth_rotation.T) - 1) / 2, -1, 1)))
                    self.assertGreater(error, 10)

    @unittest.skipUnless((SEQUENCE / "image_0" / "000025.png").is_file(),
                         "Local KITTI sequence-00 subset is not installed")
    def test_kitti_20_25_no_longer_exports_false_acceptance(self):
        config = json.loads((PROJECT_ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
        sequence = KittiGraySequence(SEQUENCE)
        settings = config["orb"]
        extractor = OrbExtractor(settings["nfeatures"], settings["scaleFactor"],
                                 settings["nlevels"], settings["fastThreshold"])
        first = extractor.extract(sequence[20].image)
        second = extractor.extract(sequence[25].image)
        matches = DescriptorMatcher(**config["matching"]).match(first.descriptors,
                                                                  second.descriptors).matches
        points1 = np.array([first.keypoints[match.queryIdx].pt for match in matches])
        points2 = np.array([second.keypoints[match.trainIdx].pt for match in matches])
        unchecked = dict(config["geometry"], pose_stability_trials=0)
        baseline = initialize_two_view(points1, points2, sequence.camera,
                                       GeometryConfig(**unchecked))
        guarded = initialize_two_view(points1, points2, sequence.camera,
                                      GeometryConfig(**config["geometry"]))
        self.assertEqual(baseline.status, "accepted")
        self.assertEqual(guarded.failure_reason, "unstable_pose_under_resampling")
        self.assertGreater(guarded.diagnostics["pose_stability_median_rotation_disagreement_deg"], 10)
        self.assertIsNone(guarded.R_21)
        self.assertIsNone(guarded.t_21)
        self.assertEqual(guarded.diagnostics["usable_map_points"], 0)
        self.assertEqual(len(guarded.points_xyz), 0)
        self.assertFalse(guarded.accepted_mask.any())


if __name__ == "__main__":
    unittest.main()
