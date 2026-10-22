"""M3 identity, state-machine, PnP, and physical multi-frame regressions."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from orb_slam_mono.camera import CalibratedCamera
from orb_slam_mono.datasets import ImageSample, KittiGraySequence
from orb_slam_mono.features import FeatureSet, OrbExtractor
from orb_slam_mono.geometry import GeometryConfig, InitializationResult
from orb_slam_mono.map_state import Frame, MapPoint, TrackingState, map_from_initialization
from orb_slam_mono.matching import DescriptorMatcher
from orb_slam_mono.synthetic_sequence import generate_sequence
from orb_slam_mono.sequence_demo import run_sequence
from orb_slam_mono.tracking import MonocularTracker, TrackingConfig, solve_map_pnp


def _frame(index, n=12):
    keys = [cv2.KeyPoint(float(i * 4), float(i * 5), 8) for i in range(n)]
    desc = np.arange(n * 32, dtype=np.uint8).reshape(n, 32).copy()
    return Frame(index, None, FeatureSet(keys, desc))


def _tracker(camera=None, config=None):
    return MonocularTracker(camera or CalibratedCamera(640, 480, 500, 500, 320, 240),
                            OrbExtractor(), DescriptorMatcher(), GeometryConfig(),
                            config or TrackingConfig())


class MapTests(unittest.TestCase):
    def test_shuffled_nonconsecutive_match_indices_preserve_xyz(self):
        first, second = _frame(4), _frame(9)
        matches = [cv2.DMatch(_queryIdx=q, _trainIdx=t, _distance=5)
                   for q, t in [(8, 5), (2, 10), (11, 0), (4, 7)]]
        geometry = InitializationResult(status="accepted", R_21=np.eye(3),
                                        t_21=np.array([1., 0, 0]),
                                        accepted_mask=np.array([False, True, False, True]),
                                        points_xyz=np.array([[1., 2, 7], [3, 1, 8]]))
        mapped = map_from_initialization(3, first, second, matches, geometry)
        self.assertEqual(mapped.segment_id, 3)
        self.assertEqual(first.feature_to_point, {2: 0, 4: 1})
        self.assertEqual(second.feature_to_point, {10: 0, 7: 1})
        np.testing.assert_array_equal(mapped.points[0].xyz_world, [1, 2, 7])
        np.testing.assert_array_equal(mapped.points[1].descriptor, first.features.descriptors[4])
        self.assertEqual(mapped.points[0].observations, {4: 2, 9: 10})
        mapped.validate()
        with self.assertRaises(ValueError):
            mapped.add_observation(0, second, 11)
        with self.assertRaises(ValueError):
            mapped.add_observation(1, first, 2)

    def test_rejected_geometry_never_creates_map(self):
        with self.assertRaises(ValueError):
            map_from_initialization(0, _frame(0), _frame(1), [], InitializationResult())

    def test_invalid_point_coordinates_and_descriptor(self):
        for xyz in ([np.nan, 0, 1], [np.inf, 1, 1]):
            with self.assertRaises(ValueError):
                MapPoint(1, np.array(xyz), np.zeros(32, np.uint8))
        with self.assertRaises(ValueError):
            MapPoint(1, np.ones(3), np.zeros(31, np.uint8))


class PnPTests(unittest.TestCase):
    def setUp(self):
        self.camera = CalibratedCamera(960, 600, 650, 650, 480, 300)
        rng = np.random.default_rng(3)
        self.xyz = np.column_stack((rng.uniform(-3, 3, 120), rng.uniform(-2, 2, 120),
                                    rng.uniform(5, 12, 120)))
        self.rotation, _ = cv2.Rodrigues(np.array([0.02, -0.03, 0.01]))
        self.translation = np.array([-0.7, 0.03, 0.01])
        rvec, _ = cv2.Rodrigues(self.rotation)
        self.uv = cv2.projectPoints(self.xyz, rvec, self.translation, self.camera.matrix, None)[0].reshape(-1, 2)
        self.uv += rng.normal(0, 0.3, self.uv.shape)

    def test_pnp_pose_and_world_to_camera_convention(self):
        estimate = solve_map_pnp(self.xyz, self.uv, self.camera, TrackingConfig())
        self.assertTrue(estimate.accepted, estimate.reason)
        self.assertGreaterEqual(len(estimate.inlier_indices), 110)
        self.assertAlmostEqual(np.linalg.det(estimate.R_cw), 1.0, places=5)
        np.testing.assert_allclose(estimate.R_cw.T @ estimate.R_cw, np.eye(3), atol=1e-5)
        self.assertLess(np.linalg.norm(estimate.t_cw - self.translation), 0.03)
        self.assertLess(np.linalg.norm(estimate.R_cw - self.rotation), 0.01)
        np.testing.assert_allclose(-estimate.R_cw.T @ estimate.t_cw,
                                   -self.rotation.T @ self.translation, atol=0.03)

    def test_invalid_and_insufficient_correspondences(self):
        for xyz, uv in [(np.empty((0, 3)), np.empty((0, 2))),
                        (self.xyz[:5], self.uv[:5]),
                        (self.xyz, self.uv[:, :1]),
                        (np.full_like(self.xyz, np.nan), self.uv)]:
            result = solve_map_pnp(xyz, uv, self.camera, TrackingConfig())
            self.assertFalse(result.accepted)
            self.assertIsNone(result.R_cw)

    def test_incorrect_associations_and_reprojection_gate(self):
        wrong = self.uv[np.random.default_rng(6).permutation(len(self.uv))]
        self.assertFalse(solve_map_pnp(self.xyz, wrong, self.camera, TrackingConfig()).accepted)
        strict = TrackingConfig(max_median_reprojection_px=0.01, max_p90_reprojection_px=0.02)
        result = solve_map_pnp(self.xyz, self.uv, self.camera, strict)
        self.assertEqual(result.reason, "excessive_pnp_reprojection_error")
        self.assertIsNone(result.R_cw)


class StateTests(unittest.TestCase):
    def test_only_documented_state_transitions_are_allowed(self):
        tracker = _tracker()
        tracker._transition(TrackingState.TRACKING)
        with self.assertRaises(ValueError):
            tracker._transition(TrackingState.INITIALIZING)
        tracker._transition(TrackingState.LOST)
        with self.assertRaises(ValueError):
            tracker._transition(TrackingState.TRACKING)
        tracker._transition(TrackingState.INITIALIZING)

    def test_rejection_then_success_preserves_unestimated_earlier_frames(self):
        sequence = generate_sequence()
        tracker = _tracker(sequence.camera, TrackingConfig(initialization_min_gap=1,
                                                             initialization_attempt_step=2))
        outcomes = [tracker.process(sample) for sample in sequence.samples[:5]]
        self.assertEqual([(a["anchor_frame"], a["candidate_frame"], a["initialization_status"])
                          for a in tracker.attempts], [(0, 1, "rejected"), (0, 3, "accepted")])
        self.assertTrue(all(outcome.R_cw is None for outcome in outcomes[:3]))
        self.assertEqual(outcomes[3].state, TrackingState.TRACKING)
        self.assertEqual(len(tracker.segments), 1)

    def test_attempt_schedule_and_anchor_replacement(self):
        tracker = _tracker(config=TrackingConfig(initialization_min_gap=2,
                                                  initialization_attempt_step=2,
                                                  initialization_max_gap=4))
        blank = np.zeros((480, 640), np.uint8)
        outcomes = [tracker.process(ImageSample(i, blank, "blank")) for i in range(9)]
        self.assertEqual([(a["anchor_frame"], a["candidate_frame"]) for a in tracker.attempts],
                         [(0, 2), (0, 4), (4, 6), (4, 8)])
        self.assertTrue(all(outcome.R_cw is None for outcome in outcomes))
        self.assertEqual(len(tracker.segments), 0)
        self.assertTrue(all(a["failure_reason"] == "insufficient_correspondences" for a in tracker.attempts))

    def test_physical_multiframe_tracking_and_accuracy(self):
        sequence = generate_sequence()
        tracker = _tracker(sequence.camera)
        results = [tracker.process(sample) for sample in sequence.samples]
        self.assertEqual([r.state for r in results[:3]], [TrackingState.INITIALIZING] * 3)
        self.assertTrue(all(r.R_cw is None for r in results[:3]))
        self.assertEqual(tracker.segments[0].anchor_frame_id, 0)
        self.assertEqual(tracker.segments[0].second_frame_id, 3)
        self.assertEqual(len(tracker.attempts), 1)
        self.assertEqual(tracker.attempts[0]["initialization_status"], "accepted")
        self.assertGreaterEqual(len(tracker.segments[0].points), 100)
        self.assertTrue(all(r.state == TrackingState.TRACKING for r in results[3:]))
        self.assertTrue(all(r.diagnostics.get("pnp_inliers", 100) >= 50 for r in results[4:]))
        # Offline known-baseline normalization only; tracker never sees metric_step.
        metric_baseline = np.linalg.norm(-sequence.truth_R_cw[3].T @ sequence.truth_t_cw[3])
        for result in results[3:]:
            index = result.frame_id
            true_center = -sequence.truth_R_cw[index].T @ sequence.truth_t_cw[index]
            estimated_center = -result.R_cw.T @ result.t_cw
            self.assertLess(np.linalg.norm(metric_baseline * estimated_center - true_center), 0.12)
            angle = np.degrees(np.arccos(np.clip((np.trace(result.R_cw @ sequence.truth_R_cw[index].T) - 1) / 2, -1, 1)))
            self.assertLess(angle, 1.5)
        tracker.segments[0].validate()

    def test_loss_then_fresh_segment_without_false_pose(self):
        sequence = generate_sequence(count=18, blank_frames=(7,))
        tracker = _tracker(sequence.camera, TrackingConfig(max_recovery_frames=4))
        results = [tracker.process(sample) for sample in sequence.samples[:7]]
        first_map = tracker.segments[0]
        prior_frame_ids = set(first_map.frames)
        prior_points = {point_id: (point.xyz_world.copy(), point.descriptor.copy(),
                                    point.observations.copy())
                        for point_id, point in first_map.points.items()}
        prior_pose = (first_map.frames[6].R_cw.copy(), first_map.frames[6].t_cw.copy())
        failed = tracker.process(sequence[7])
        results.append(failed)
        self.assertIsNone(tracker.active_map)
        self.assertEqual(set(first_map.frames), prior_frame_ids)
        self.assertNotIn(7, first_map.frames)
        self.assertEqual(tracker.last_successful_pose["frame_id"], 6)
        self.assertEqual(failed.segment_id, first_map.segment_id)
        self.assertEqual(failed.diagnostics["pnp_status"], "rejected")
        self.assertEqual(failed.diagnostics["pnp_failure_reason"], failed.failure_reason)
        self.assertFalse(failed.diagnostics["pnp_ransac_attempted"])
        self.assertIsNone(failed.as_dict()["T_cw_R"])
        for point_id, (xyz, descriptor, observations) in prior_points.items():
            point = first_map.points[point_id]
            np.testing.assert_array_equal(point.xyz_world, xyz)
            np.testing.assert_array_equal(point.descriptor, descriptor)
            self.assertEqual(point.observations, observations)
        np.testing.assert_array_equal(first_map.frames[6].R_cw, prior_pose[0])
        np.testing.assert_array_equal(first_map.frames[6].t_cw, prior_pose[1])
        results.extend(tracker.process(sample) for sample in sequence.samples[8:])
        self.assertEqual(results[7].state, TrackingState.LOST)
        self.assertIsNone(results[7].R_cw)
        self.assertEqual(results[7].failure_reason, "insufficient_pnp_correspondences")
        self.assertEqual([r.state for r in results[8:11]], [TrackingState.INITIALIZING] * 3)
        self.assertTrue(all(r.R_cw is None for r in results[8:11]))
        self.assertEqual(results[11].state, TrackingState.TRACKING)
        self.assertEqual([s.segment_id for s in tracker.segments], [0, 1])
        self.assertEqual([(s.anchor_frame_id, s.second_frame_id) for s in tracker.segments],
                         [(0, 3), (8, 11)])
        self.assertNotEqual(results[6].segment_id, results[11].segment_id)
        self.assertTrue(set(tracker.segments[0].points).isdisjoint(tracker.segments[1].points))
        self.assertEqual(tracker.recovery_events, [])

    def test_recovery_budget_exhaustion_is_terminal_and_clears_anchor(self):
        sequence = generate_sequence(count=12, blank_frames=(7,))
        tracker = _tracker(sequence.camera, TrackingConfig(max_recovery_frames=3,
                                                             initialization_min_gap=1))
        for sample in sequence.samples[:8]:
            tracker.process(sample)
        self.assertEqual(tracker.state, TrackingState.LOST)
        prior_map = tracker.segments[0]
        prior_frames = set(prior_map.frames)
        prior_observations = {point_id: point.observations.copy()
                              for point_id, point in prior_map.points.items()}
        blank = np.zeros_like(sequence.samples[0].image)
        recovery = [tracker.process(ImageSample(i, blank, "blank")) for i in (8, 9, 10)]
        self.assertEqual([r.state for r in recovery], [TrackingState.INITIALIZING] * 3)
        self.assertEqual([r.diagnostics["recovery_frames_processed"] for r in recovery], [1, 2, 3])
        self.assertEqual(recovery[-1].diagnostics["recovery_frames_remaining"], 0)
        self.assertIsNotNone(tracker.anchor)
        attempts_at_deadline = len(tracker.attempts)
        timeout = tracker.process(ImageSample(11, blank, "blank"))
        self.assertEqual(timeout.state, TrackingState.LOST)
        self.assertEqual(timeout.failure_reason, "recovery_window_exhausted")
        self.assertTrue(timeout.diagnostics["terminal_recovery_exhausted"])
        self.assertFalse(timeout.diagnostics["initialization_attempted"])
        self.assertTrue(tracker.terminal_lost)
        self.assertIsNone(tracker.anchor)
        self.assertIsNone(tracker.anchor_image)
        self.assertIsNone(tracker.active_map)
        self.assertEqual(len(tracker.recovery_events), 1)
        with patch.object(tracker, "_transition", wraps=tracker._transition) as transition:
            later = [tracker.process(ImageSample(i, blank, "blank")) for i in (12, 13, 90)]
            transition.assert_not_called()
        self.assertTrue(all(r.state == TrackingState.LOST and r.R_cw is None and r.t_cw is None
                            and r.failure_reason == "terminal_recovery_exhausted" for r in later))
        self.assertTrue(all(r.diagnostics["feature_extraction_skipped"] for r in later))
        self.assertTrue(all(r.segment_id is None and r.diagnostics["lost_segment_id"] == 0 for r in later))
        self.assertEqual(len(tracker.attempts), attempts_at_deadline)
        self.assertEqual(set(prior_map.frames), prior_frames)
        self.assertEqual({point_id: point.observations for point_id, point in prior_map.points.items()},
                         prior_observations)
        # Restart is explicit: a fresh tracker owns its own independent segment.
        fresh = _tracker(sequence.camera)
        restarted = [fresh.process(sample) for sample in generate_sequence().samples[:4]]
        self.assertEqual(restarted[-1].state, TrackingState.TRACKING)
        self.assertEqual(fresh.segments[0].segment_id, 0)
        self.assertTrue(tracker.terminal_lost)

    def test_recovery_budget_counts_processed_frames_not_source_id_distance(self):
        sequence = generate_sequence(count=12, blank_frames=(7,))
        tracker = _tracker(sequence.camera, TrackingConfig(max_recovery_frames=3,
                                                             initialization_min_gap=1))
        for sample in sequence.samples[:8]:
            tracker.process(sample)
        blank = np.zeros_like(sequence.samples[0].image)
        results = [tracker.process(ImageSample(i, blank, "blank")) for i in (20, 40, 60, 80)]
        self.assertEqual([r.state for r in results], [TrackingState.INITIALIZING] * 3 + [TrackingState.LOST])
        self.assertEqual([r.diagnostics["recovery_frames_processed"] for r in results], [1, 2, 3, 3])
        self.assertEqual(results[-1].failure_reason, "recovery_window_exhausted")
        self.assertEqual(tracker.recovery_events[0]["frame_id"], 80)

    def test_export_diagnostics_separate_accepted_and_rejected_pnp(self):
        import csv
        config = {"orb": {"nfeatures": 1500, "scaleFactor": 1.2, "nlevels": 8, "fastThreshold": 20},
                  "matching": {"ratio_threshold": 0.75, "mutual_check": True},
                  "geometry": {}, "tracking": {}}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            summary = run_sequence("synthetic", output, config)
            with (output / "per_frame.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 12)
            self.assertEqual(summary["pnp_tracked_frames"], 8)
            self.assertIsNotNone(summary["pnp_inlier_distribution_accepted"])
            self.assertEqual(summary["recovery_timeouts"], 0)
            self.assertEqual(json.loads((output / "recovery_events.json").read_text()), [])
            self.assertTrue(all(row["pnp_status"] == "accepted" for row in rows[4:]))
            self.assertTrue(all(row["pnp_ransac_attempted"] == "True" for row in rows[4:]))

    def test_synthetic_timeout_export_has_explicit_terminal_and_failure_populations(self):
        import csv
        sequence = generate_sequence(count=12, blank_frames=(7, 8, 9, 10, 11))
        config = {"orb": {"nfeatures": 1500, "scaleFactor": 1.2, "nlevels": 8, "fastThreshold": 20},
                  "matching": {"ratio_threshold": 0.75, "mutual_check": True},
                  "geometry": {}, "tracking": {"max_recovery_frames": 3,
                                                 "initialization_min_gap": 1}}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch("orb_slam_mono.sequence_demo.generate_sequence", return_value=sequence):
                summary = run_sequence("synthetic", output, config)
            with (output / "per_frame.csv").open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 12)
            self.assertEqual(summary["recovery_timeouts"], 1)
            self.assertEqual(summary["terminal_lost_frames"], 1)
            self.assertEqual(summary["pnp_rejection_reasons"], {"insufficient_pnp_correspondences": 1})
            self.assertIsNotNone(summary["pnp_inlier_distribution_accepted"])
            self.assertIsNone(summary["pnp_inlier_distribution_rejected_solver_returns"])
            self.assertEqual(rows[7]["pnp_status"], "rejected")
            self.assertEqual(rows[7]["pnp_ransac_attempted"], "False")
            self.assertEqual(rows[11]["terminal_recovery_exhausted"], "True")
            self.assertEqual(rows[11]["pose_valid"], "False")
            self.assertEqual(len(json.loads((output / "recovery_events.json").read_text())), 1)

    def test_rejects_duplicate_or_nonchronological_frame_id(self):
        tracker = _tracker()
        image = np.zeros((480, 640), np.uint8)
        tracker.process(ImageSample(4, image, "blank"))
        with self.assertRaises(ValueError):
            tracker.process(ImageSample(4, image, "blank"))
        with self.assertRaises(ValueError):
            tracker.process(ImageSample(3, image, "blank"))

    def test_pure_rotation_and_weak_baseline_do_not_map(self):
        for step in (0.0, 2e-4):
            sequence = generate_sequence(count=6, metric_step=step)
            tracker = _tracker(sequence.camera)
            results = [tracker.process(sample) for sample in sequence.samples]
            self.assertEqual(len(tracker.segments), 0)
            self.assertTrue(all(result.R_cw is None for result in results))


class KittiSmokeTests(unittest.TestCase):
    def test_sequence_00_loader_and_initialization_outcome(self):
        root = Path(__file__).resolve().parents[1] / "data" / "sequences" / "00"
        if not root.is_dir():
            self.skipTest("Local KITTI sequence 00 is absent")
        sequence = KittiGraySequence(root)
        self.assertGreaterEqual(len(sequence), 6)
        tracker = _tracker(sequence.camera)
        outcomes = [tracker.process(sequence[i]) for i in range(6)]
        self.assertEqual(len(outcomes), 6)
        self.assertGreaterEqual(len(tracker.attempts), 1)
        self.assertTrue(all(outcome.state in TrackingState for outcome in outcomes))


if __name__ == "__main__":
    unittest.main()
