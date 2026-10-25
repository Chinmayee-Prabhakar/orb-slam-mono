"""Deterministic safety controls for M4.5-D offline experiments."""

from __future__ import annotations

import copy
import inspect
import unittest

import cv2
import numpy as np

from orb_slam_mono.camera import CalibratedCamera
from orb_slam_mono.continuity_assessment import (offline_gap_four, offline_map_probe,
                                                recovery_intervals)
from orb_slam_mono.features import FeatureSet
from orb_slam_mono.geometry import GeometryConfig
from orb_slam_mono.map_state import Frame
from orb_slam_mono.matching import DescriptorMatcher
from orb_slam_mono.tracking import TrackingConfig


def _features(xy: np.ndarray, descriptors: np.ndarray) -> FeatureSet:
    return FeatureSet([cv2.KeyPoint(float(x), float(y), 8) for x, y in xy], descriptors)


class ContinuityAssessmentTests(unittest.TestCase):
    def test_recovery_intervals_use_source_ids_and_keep_unknown_attempts(self) -> None:
        rows = [{"frame_id": 40, "pose_valid": True, "failure_reason": None},
                {"frame_id": 41, "pose_valid": False, "failure_reason": "waiting_for_initialization_gap"},
                {"frame_id": 43, "pose_valid": False, "failure_reason": "low_parallax"},
                {"frame_id": 47, "pose_valid": True, "failure_reason": None}]
        attempts = [{"anchor_frame": 41, "candidate_frame": 43, "gap": 2,
                     "reason": "low_parallax"}]
        intervals = recovery_intervals(rows, attempts)
        self.assertEqual(len(intervals), 1)
        self.assertEqual((intervals[0]["first_frame"], intervals[0]["last_frame"],
                          intervals[0]["processed_frames"], intervals[0]["next_online_pose_frame"]),
                         (41, 43, 2, 47))
        self.assertEqual(intervals[0]["rejections"], {"low_parallax": 1})
        self.assertEqual(intervals[0]["unscheduled_or_waiting_frames"], 1)

    def test_duplicate_frame_ids_rejected(self) -> None:
        with self.assertRaises(ValueError):
            recovery_intervals([{"frame_id": 2}, {"frame_id": 2}], [])

    def _positive_scene(self):
        rng = np.random.default_rng(47)
        camera = CalibratedCamera(640, 480, 500, 500, 320, 240)
        xyz = np.column_stack((rng.uniform(-2, 2, 32), rng.uniform(-1.4, 1.4, 32),
                               rng.uniform(4, 9, 32)))
        uv, _ = cv2.projectPoints(xyz, np.zeros(3), np.array([0.12, 0.01, 0.02]),
                                  camera.matrix, None)
        uv = uv.reshape(-1, 2)
        descriptors = rng.integers(0, 256, (len(xyz), 32), dtype=np.uint8)
        snapshot = {"lost_frame": 10, "last_pose_frame": 9, "segment_id": 7,
                    "points": [{"id": 700 + i, "xyz": xyz[i].copy(),
                                "descriptor": descriptors[i].copy(), "last_observed": 9,
                                "observation_count": 3} for i in range(len(xyz))]}
        return camera, snapshot, _features(uv, descriptors)

    def test_existing_map_positive_control_and_no_state_mutation(self) -> None:
        camera, snapshot, features = self._positive_scene()
        before = copy.deepcopy(snapshot)
        first = offline_map_probe(snapshot, features, 11, DescriptorMatcher(), camera,
                                  TrackingConfig())
        second = offline_map_probe(snapshot, features, 11, DescriptorMatcher(), camera,
                                   TrackingConfig())
        self.assertTrue(first["passes_unchanged_pnp_gates_offline"])
        self.assertEqual(first["raw_inliers"], len(snapshot["points"]))
        self.assertEqual(first["segment_id"], 7)
        self.assertNotIn("T_cw", first)
        self.assertEqual([{k: v for k, v in row.items() if k != "runtime_ms"}
                          for row in (first, second)][0],
                         {k: v for k, v in second.items() if k != "runtime_ms"})
        for old, new in zip(before["points"], snapshot["points"]):
            np.testing.assert_array_equal(old["xyz"], new["xyz"])
            np.testing.assert_array_equal(old["descriptor"], new["descriptor"])

    def test_wrong_associations_reject_and_short_fresh_map_is_insufficient(self) -> None:
        camera, snapshot, features = self._positive_scene()
        wrong_xy = np.array([keypoint.pt for keypoint in features.keypoints])[::-1].copy()
        wrong = offline_map_probe(snapshot, _features(wrong_xy, features.descriptors), 11,
                                  DescriptorMatcher(), camera, TrackingConfig())
        self.assertFalse(wrong["passes_unchanged_pnp_gates_offline"])
        short = dict(snapshot, points=snapshot["points"][:10], segment_id=8)
        result = offline_map_probe(short, features, 11, DescriptorMatcher(), camera,
                                   TrackingConfig())
        self.assertEqual(result["reason"], "insufficient_pnp_correspondences")
        self.assertEqual(result["valid_landmarks"], 10)
        self.assertEqual(result["segment_id"], 8)

    def test_probe_horizon_is_bounded(self) -> None:
        camera, snapshot, features = self._positive_scene()
        with self.assertRaises(ValueError):
            offline_map_probe(snapshot, features, 10, DescriptorMatcher(), camera,
                              TrackingConfig())
        with self.assertRaises(ValueError):
            offline_map_probe(snapshot, features, 14, DescriptorMatcher(), camera,
                              TrackingConfig())

    def test_gap_four_low_parallax_control_and_wrong_gap(self) -> None:
        camera = CalibratedCamera(640, 480, 500, 500, 320, 240)
        rng = np.random.default_rng(12)
        xy = rng.uniform([40, 30], [600, 450], (70, 2))
        descriptors = rng.integers(0, 256, (70, 32), dtype=np.uint8)
        anchor = Frame(123, None, _features(xy, descriptors))
        candidate = Frame(127, None, _features(xy.copy(), descriptors.copy()))
        result = offline_gap_four(anchor, candidate, DescriptorMatcher(), camera,
                                  GeometryConfig())
        self.assertFalse(result["accepted_by_unchanged_geometry_offline"])
        self.assertIsNotNone(result["reason"])
        self.assertIsNone(anchor.R_cw)
        with self.assertRaises(ValueError):
            offline_gap_four(anchor, Frame(128, None, candidate.features),
                             DescriptorMatcher(), camera, GeometryConfig())

    def test_offline_module_has_no_truth_or_pose_export_dependency(self) -> None:
        import orb_slam_mono.continuity_assessment as module
        source = inspect.getsource(module)
        self.assertNotIn("ground_truth", source)
        self.assertNotIn("trajectory_evaluation", source)
        self.assertNotIn("estimated_poses.json", source)


if __name__ == "__main__":
    unittest.main()
