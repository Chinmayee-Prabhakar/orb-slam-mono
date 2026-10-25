"""Decision-preserving checks for opt-in tracking diagnostics."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from orb_slam_mono.bundle_adjustment import BundleConfig
from orb_slam_mono.features import OrbExtractor
from orb_slam_mono.geometry import GeometryConfig
from orb_slam_mono.local_mapping import MappingConfig
from orb_slam_mono.matching import DescriptorMatcher
from orb_slam_mono.synthetic_sequence import generate_sequence
from orb_slam_mono.tracking import MonocularTracker, TrackingConfig, solve_map_pnp
from orb_slam_mono.tracking_diagnostics import _normalized, diagnose_pnp_event


def _result_snapshot(tracker: MonocularTracker, results: list) -> dict:
    segments = []
    for segment in tracker.segments:
        segments.append({"segment_id": segment.segment_id,
                         "points": [(point_id, point.xyz_world.tolist(), point.valid,
                                     sorted(point.observations), point.descriptor.tolist(),
                                     [evidence.tolist() for evidence in point.descriptor_evidence])
                                    for point_id, point in sorted(segment.points.items())],
                         "frames": [(frame_id, sorted(frame.feature_to_point.items()))
                                    for frame_id, frame in sorted(segment.frames.items())],
                         "keyframes": list(segment.keyframes),
                         "active_keyframes": list(segment.active_keyframe_ids)})
    return {"results": [{key: value for key, value in result.as_dict().items() if key != "timing_ms"}
                        for result in results], "segments": segments,
            "attempts": tracker.attempts, "losses": tracker.loss_events,
            "mapping": tracker.mapping_events, "ba": tracker.ba_events}


class TrackingDiagnosticsTests(unittest.TestCase):
    def test_match_capture_preserves_matches_and_rng(self) -> None:
        rng = np.random.default_rng(2)
        first = rng.integers(0, 256, (24, 32), dtype=np.uint8)
        second = np.vstack((first[:12], rng.integers(0, 256, (12, 32), dtype=np.uint8)))
        matcher = DescriptorMatcher()
        cv2.setRNGSeed(42)
        normal = matcher.match(first, second)
        after_normal = np.empty((8, 8), np.float32)
        cv2.randu(after_normal, 0, 1)
        cv2.setRNGSeed(42)
        capture = {}
        observed = matcher.match(first, second, diagnostic=capture)
        after_observed = np.empty((8, 8), np.float32)
        cv2.randu(after_observed, 0, 1)
        self.assertEqual((normal.knn_pairs, normal.ratio_passed, normal.retained),
                         (observed.knn_pairs, observed.ratio_passed, observed.retained))
        self.assertEqual([(m.queryIdx, m.trainIdx, m.distance) for m in normal.matches],
                         [(m.queryIdx, m.trainIdx, m.distance) for m in observed.matches])
        self.assertEqual(capture["forward_complete_pairs"], normal.knn_pairs)
        self.assertEqual(capture["ratio_passed"], normal.ratio_passed)
        self.assertEqual(capture["mutual_passed"], normal.retained)
        self.assertEqual(len(capture["candidates"]), normal.knn_pairs)
        np.testing.assert_array_equal(after_normal, after_observed)

    def test_pnp_capture_preserves_pose_and_rng(self) -> None:
        camera = generate_sequence(count=5).camera
        rng = np.random.default_rng(12)
        xyz = np.column_stack((rng.uniform(-3, 3, 50), rng.uniform(-2, 2, 50),
                               rng.uniform(6, 15, 50)))
        uv = cv2.projectPoints(xyz, np.zeros(3), np.array([.1, 0, 0]),
                               camera.matrix, None)[0].reshape(-1, 2)
        config = TrackingConfig()
        cv2.setRNGSeed(99)
        plain = solve_map_pnp(xyz, uv, camera, config)
        after_plain = np.empty((8, 8), np.float32)
        cv2.randu(after_plain, 0, 1)
        cv2.setRNGSeed(99)
        capture = {}
        observed = solve_map_pnp(xyz, uv, camera, config, diagnostic_capture=capture)
        after_observed = np.empty((8, 8), np.float32)
        cv2.randu(after_observed, 0, 1)
        self.assertEqual((plain.accepted, plain.reason, plain.diagnostics),
                         (observed.accepted, observed.reason, observed.diagnostics))
        np.testing.assert_array_equal(plain.R_cw, observed.R_cw)
        np.testing.assert_array_equal(plain.t_cw, observed.t_cw)
        np.testing.assert_array_equal(plain.inlier_indices, observed.inlier_indices)
        np.testing.assert_array_equal(after_plain, after_observed)
        self.assertTrue(capture["ransac_success"])
        self.assertEqual(len(capture["raw_inlier_indices"]), 50)

    def test_pnp_no_candidate_capture_is_safe(self) -> None:
        camera = generate_sequence(count=5).camera
        xyz = np.column_stack((np.arange(12), np.zeros(12), np.full(12, 8)))
        uv = np.zeros((12, 2))
        with patch("orb_slam_mono.tracking.cv2.solvePnPRansac", return_value=(False, None, None, None)):
            capture = {}
            observed = solve_map_pnp(xyz, uv, camera, TrackingConfig(), diagnostic_capture=capture)
        self.assertFalse(observed.accepted)
        self.assertEqual(observed.reason, "pnp_solver_failed")
        self.assertIsNone(capture["raw_rvec"])
        self.assertIsNone(capture["raw_inlier_indices"])
        with patch("orb_slam_mono.tracking.cv2.solvePnPRansac", return_value=(False, "invalid", "invalid", "invalid")):
            capture = {}
            observed = solve_map_pnp(xyz, uv, camera, TrackingConfig(), diagnostic_capture=capture)
        self.assertEqual(observed.reason, "pnp_solver_failed")
        self.assertIsNone(capture["raw_rvec"])

    def test_short_support_failure_does_not_fabricate_candidate(self) -> None:
        sequence = generate_sequence(count=8)
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/default.json").read_text())
        orb = config["orb"]
        tracker = MonocularTracker(sequence.camera,
                                   OrbExtractor(orb["nfeatures"], orb["scaleFactor"], orb["nlevels"], orb["fastThreshold"]),
                                   DescriptorMatcher(**config["matching"]), GeometryConfig(**config["geometry"]),
                                   TrackingConfig(min_pnp_correspondences=100000),
                                   MappingConfig(enabled=False), BundleConfig(enabled=False),
                                   capture_pnp_diagnostics=True)
        observed = None
        for sample in sequence.samples:
            result = tracker.process(sample)
            if tracker.diagnostic_event is not None:
                observed = (result, tracker.diagnostic_event)
                break
        self.assertIsNotNone(observed)
        result, event = observed
        compact, detail = diagnose_pnp_event(event, result.as_dict(),
                                              sequence.camera.matrix, sequence.samples[result.frame_id].image.shape)
        self.assertFalse(compact["pose_valid"])
        self.assertIsNone(compact["pnp_raw_inliers"])
        self.assertFalse(detail["candidate"]["available"])

    def test_tracker_capture_preserves_estimator_state_and_rng(self) -> None:
        sequence = generate_sequence(count=10, blank_frames=(9,))
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/default.json").read_text())
        orb = config["orb"]
        snapshots = []
        for enabled in (False, True):
            cv2.setRNGSeed(85)
            tracker = MonocularTracker(
                sequence.camera,
                OrbExtractor(orb["nfeatures"], orb["scaleFactor"], orb["nlevels"], orb["fastThreshold"]),
                DescriptorMatcher(**config["matching"]), GeometryConfig(**config["geometry"]),
                TrackingConfig(**config["tracking"]), MappingConfig(enabled=False),
                BundleConfig(enabled=False), capture_pnp_diagnostics=enabled)
            results = [tracker.process(sample) for sample in sequence.samples]
            tail = np.empty((8, 8), np.float32)
            cv2.randu(tail, 0, 1)
            snapshots.append((_result_snapshot(tracker, results), tail))
        self.assertEqual(_normalized(snapshots[0][0]), _normalized(snapshots[1][0]))
        np.testing.assert_array_equal(snapshots[0][1], snapshots[1][1])


if __name__ == "__main__":
    unittest.main()
