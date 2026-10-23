"""Focused M4 identity, triangulation, optimization, and ablation checks."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from orb_slam_mono.bundle_adjustment import BundleConfig, _projection, jacobian_sparsity, optimize_local_map
from orb_slam_mono.camera import CalibratedCamera
from orb_slam_mono.datasets import KittiGraySequence
from orb_slam_mono.features import FeatureSet, OrbExtractor
from orb_slam_mono.geometry import GeometryConfig
from orb_slam_mono.local_mapping import (MappingConfig, cull_inconsistent_points,
                                          enforce_active_capacity, insert_keyframe, should_insert_keyframe,
                                          triangulate_new_points)
from orb_slam_mono.map_state import Frame, MapPoint, SparseMap, TrackingState
from orb_slam_mono.matching import DescriptorMatcher
from orb_slam_mono.sequence_demo import _final_map_pose_records, run_sequence
from orb_slam_mono.synthetic_sequence import generate_sequence
from orb_slam_mono.tracking import MonocularTracker, TrackingConfig


CAMERA = CalibratedCamera(960, 600, 650, 650, 480, 300)


def _fixture(count=30, noise=0.0):
    rng = np.random.default_rng(42)
    xyz = np.column_stack((rng.uniform(-2, 2, count), rng.uniform(-1, 1, count),
                           rng.uniform(6, 12, count)))
    descriptors = rng.integers(0, 256, (count, 32), dtype=np.uint8)
    sparse_map = SparseMap(4, 0, 1)
    for i in range(3):
        R = np.eye(3)
        t = np.array([-float(i), 0., 0.])
        uv = np.column_stack((CAMERA.fx * (xyz[:, 0] - i) / xyz[:, 2] + CAMERA.cx,
                              CAMERA.fy * xyz[:, 1] / xyz[:, 2] + CAMERA.cy))
        uv += rng.normal(0, noise, uv.shape)
        features = FeatureSet([cv2.KeyPoint(float(u), float(v), 10) for u, v in uv],
                              descriptors.copy())
        frame = Frame(i, None, features, TrackingState.TRACKING, R.copy(), t.copy())
        sparse_map.frames[i] = frame
        insert_keyframe(sparse_map, frame, 8)
    return sparse_map, xyz


class MappingTests(unittest.TestCase):
    def test_capacity_one_slot_stale_priority_and_segment_scope(self):
        mapped, _ = _fixture(55)
        config = MappingConfig(enabled=True, max_active_points=50, stale_after_frames=5)
        for i in range(49):
            mapped.points[i] = MapPoint(i, np.array([i * .1, 0., 8.]),
                                        mapped.frames[0].features.descriptors[i].copy())
            for frame in mapped.frames.values():
                mapped.add_observation(i, frame, i)
        report = triangulate_new_points(mapped, mapped.frames[0], mapped.frames[1], CAMERA,
                                        DescriptorMatcher(), config, 100)
        self.assertEqual(report["new_points"], 1)
        self.assertEqual(sum(p.valid for p in mapped.points.values()), 50)
        mapped.validate()
        separate, _ = _fixture(8)
        self.assertEqual(enforce_active_capacity(separate, config, 20), 0)
        self.assertEqual(len(mapped.points), 50)
        mapped.points[0].valid = False
        self.assertEqual(enforce_active_capacity(mapped, config, 20), 0)
        self.assertEqual(sum(p.valid for p in mapped.points.values()), 49)

    def test_overfull_map_prefers_stale_landmarks_for_deactivation(self):
        mapped, _ = _fixture(55)
        config = MappingConfig(enabled=True, max_active_points=50, stale_after_frames=5)
        for i in range(55):
            mapped.points[i] = MapPoint(i, np.array([i * .1, 0., 8.]),
                                        mapped.frames[0].features.descriptors[i].copy())
            for frame in mapped.frames.values():
                mapped.add_observation(i, frame, i)
        recent = Frame(20, None, mapped.frames[2].features, TrackingState.TRACKING,
                       np.eye(3), np.array([-20., 0., 0.]))
        mapped.frames[20] = recent
        for i in range(50):
            mapped.add_observation(i, recent, i)
        self.assertEqual(enforce_active_capacity(mapped, config, 20), 5)
        self.assertEqual([i for i in range(55) if not mapped.points[i].valid], list(range(50, 55)))
        mapped.validate()

    def test_hard_capacity_admits_only_free_slots_and_preserves_ids(self):
        mapped, _ = _fixture(60)
        config = MappingConfig(enabled=True, max_active_points=50)
        first = triangulate_new_points(mapped, mapped.frames[0], mapped.frames[1], CAMERA,
                                       DescriptorMatcher(), config, 100)
        self.assertEqual(first["new_points"], 50)
        self.assertGreater(first["capacity_rejected"], 0)
        self.assertEqual(sum(p.valid for p in mapped.points.values()), 50)
        self.assertEqual(triangulate_new_points(mapped, mapped.frames[0], mapped.frames[1], CAMERA,
                                                DescriptorMatcher(), config, 200)["new_points"], 0)
        mapped.validate()
        for point in mapped.points.values():
            self.assertEqual(point.observations[0], point.observations[1])
        other, _ = _fixture(55)
        for i in range(55):
            other.points[i] = MapPoint(i, np.array([i * .1, 0., 8.]),
                                       other.frames[0].features.descriptors[i].copy())
            for frame in other.frames.values():
                other.add_observation(i, frame, i)
        removed = enforce_active_capacity(other, config, 2)
        self.assertEqual(removed, 5)
        self.assertEqual(sum(p.valid for p in other.points.values()), 50)
        self.assertEqual([i for i in range(55) if not other.points[i].valid], list(range(5)))
        other.validate()

    def test_triangulates_from_estimated_poses_and_preserves_feature_indices(self):
        mapped, truth = _fixture()
        # Truth is used solely by this assertion; mapping receives pixel features and estimated poses.
        report = triangulate_new_points(mapped, mapped.frames[0], mapped.frames[1], CAMERA,
                                        DescriptorMatcher(), MappingConfig(enabled=True), 50)
        self.assertGreaterEqual(report["new_points"], 25)
        self.assertEqual(sorted(mapped.points)[0], 50)
        for point in mapped.points.values():
            q = point.observations[0]
            self.assertEqual(point.observations[1], q)
            self.assertLess(np.linalg.norm(point.xyz_world - truth[q]), 1e-4)
        mapped.validate()
        self.assertEqual(triangulate_new_points(mapped, mapped.frames[0], mapped.frames[1], CAMERA,
                                                DescriptorMatcher(), MappingConfig(enabled=True), 100)["new_points"], 0)

    def test_keyframe_policy_window_and_segment_invariants(self):
        mapped, _ = _fixture(8)
        config = MappingConfig(enabled=True, min_keyframe_gap=3)
        self.assertFalse(should_insert_keyframe(mapped, mapped.frames[2], config))
        candidate = Frame(5, None, mapped.frames[2].features, TrackingState.TRACKING,
                          np.eye(3), np.array([-3., 0., 0.]))
        mapped.frames[5] = candidate
        self.assertTrue(should_insert_keyframe(mapped, candidate, config))
        del mapped.frames[5]
        for i in range(3, 15):
            frame = mapped.frames[2]
            copy = Frame(i, None, frame.features, TrackingState.TRACKING,
                         np.eye(3), np.array([-float(i), 0., 0.]))
            mapped.frames[i] = copy
            insert_keyframe(mapped, copy, 8)
        self.assertEqual(len(mapped.active_keyframe_ids), 8)
        self.assertEqual(mapped.active_keyframe_ids[:2], [0, 1])
        self.assertEqual(len(mapped.keyframes), 15)
        mapped.validate()
        with self.assertRaises(ValueError):
            insert_keyframe(mapped, mapped.frames[14], 8)
        mapped.keyframes[14].segment_id = 99
        with self.assertRaises(ValueError):
            mapped.validate()

    def test_descriptor_medoid_bounded_and_invalid_points_excluded(self):
        descriptor = np.zeros(32, np.uint8)
        point = MapPoint(0, np.array([0., 0., 8.]), descriptor)
        for value in (255, 0, 0, 255, 0):
            point.update_descriptor(np.full(32, value, np.uint8), 3)
        self.assertEqual(len(point.descriptor_evidence), 3)
        np.testing.assert_array_equal(point.descriptor, np.zeros(32, np.uint8))
        with self.assertRaises(ValueError):
            point.update_descriptor(np.zeros(31, np.uint8))
        mapped, _ = _fixture(8)
        mapped.points[0] = point
        for i in range(3):
            mapped.add_observation(0, mapped.frames[i], 0)
        mapped.frames[2].features.keypoints[0].pt = (0., 0.)
        self.assertEqual(cull_inconsistent_points(mapped, CAMERA,
                         MappingConfig(enabled=True, max_point_median_reprojection_px=0.1)), 1)
        self.assertFalse(point.valid)
        mapped.validate()

    def test_configuration_rejects_invalid_bounds(self):
        for kwargs in ({"window_size": 11}, {"min_keyframe_gap": 0},
                       {"descriptor_evidence_limit": 0}):
            with self.assertRaises(ValueError):
                MappingConfig(**kwargs)


class BundleTests(unittest.TestCase):
    def _noisy_map(self):
        mapped, truth = _fixture(30, noise=0.35)
        rng = np.random.default_rng(5)
        mapped.frames[2].t_cw += np.array([0.05, -0.03, 0.02])
        mapped.frames[2].R_cw = cv2.Rodrigues(np.array([0.004, -0.003, 0.002]))[0]
        for i in range(30):
            point = MapPoint(i, truth[i] + rng.normal(0, 0.04, 3),
                             mapped.frames[0].features.descriptors[i].copy())
            mapped.points[i] = point
            for frame in mapped.frames.values():
                mapped.add_observation(i, frame, i)
        self._add_outside_frame(mapped, truth, 3, noise=0.35)
        return mapped

    @staticmethod
    def _add_outside_frame(mapped, xyz, frame_id=3, noise=0.0):
        rng = np.random.default_rng(101)
        uv = np.column_stack((CAMERA.fx * (xyz[:, 0] - frame_id) / xyz[:, 2] + CAMERA.cx,
                              CAMERA.fy * xyz[:, 1] / xyz[:, 2] + CAMERA.cy))
        uv += rng.normal(0, noise, uv.shape)
        frame = Frame(frame_id, None,
                      FeatureSet([cv2.KeyPoint(float(u), float(v), 10) for u, v in uv],
                                 mapped.frames[0].features.descriptors.copy()),
                      TrackingState.TRACKING, np.eye(3), np.array([-float(frame_id), 0., 0.]))
        mapped.frames[frame_id] = frame
        for i in range(len(xyz)):
            mapped.add_observation(i, frame, i)
        return frame

    def _controlled_candidate(self, outside_shift: bool):
        mapped, truth = _fixture(30)
        shifted = truth + np.array([0.09, 0., 0.])
        for i in range(30):
            mapped.points[i] = MapPoint(i, truth[i].copy(),
                                        mapped.frames[0].features.descriptors[i].copy())
            for frame in mapped.frames.values():
                uv, _ = _projection(frame.R_cw, frame.t_cw, shifted[i:i + 1], CAMERA)
                frame.features.keypoints[i].pt = tuple(uv[0])
                mapped.add_observation(i, frame, i)
        self._add_outside_frame(mapped, shifted if outside_shift else truth)
        def candidate(_residual, x0, **_kwargs):
            values = x0.copy()
            values[6:] = shifted.reshape(-1)
            return SimpleNamespace(x=values, success=False, status=0,
                                   message="maximum number of function evaluations exceeded", nfev=12)
        return mapped, candidate

    def test_outside_window_damage_rejected_and_transaction_untouched(self):
        mapped, candidate = self._controlled_candidate(False)
        old_poses = {i: (f.R_cw.copy(), f.t_cw.copy()) for i, f in mapped.frames.items()}
        old_xyz = {i: p.xyz_world.copy() for i, p in mapped.points.items()}
        old_desc = {i: p.descriptor.copy() for i, p in mapped.points.items()}
        with patch("orb_slam_mono.bundle_adjustment.least_squares", side_effect=candidate):
            report = optimize_local_map(mapped, CAMERA, BundleConfig())
        self.assertFalse(report["applied"])
        self.assertEqual(report["reason"], "heldout_reprojection_regression")
        self.assertEqual(report["heldout_observations"], 30)
        self.assertEqual(report["heldout_outside_window_observations"], 30)
        self.assertLess(report["heldout_initial_median_px"], 1e-3)
        self.assertGreater(report["heldout_final_median_px"], 3)
        self.assertGreater(report["heldout_new_quality_violations"], 0)
        for i, frame in mapped.frames.items():
            np.testing.assert_array_equal(frame.R_cw, old_poses[i][0])
            np.testing.assert_array_equal(frame.t_cw, old_poses[i][1])
        for i, point in mapped.points.items():
            np.testing.assert_array_equal(point.xyz_world, old_xyz[i])
            np.testing.assert_array_equal(point.descriptor, old_desc[i])
        mapped.validate()

    def test_safe_budget_candidate_uses_external_validation(self):
        mapped, candidate = self._controlled_candidate(True)
        with patch("orb_slam_mono.bundle_adjustment.least_squares", side_effect=candidate):
            report = optimize_local_map(mapped, CAMERA, BundleConfig())
        self.assertTrue(report["applied"], report)
        self.assertEqual(report["solver_termination"], "budget_exhausted")
        self.assertEqual(report["heldout_observations"], 30)
        self.assertLess(report["heldout_final_median_px"], report["heldout_initial_median_px"])
        mapped.validate()

    def test_budget_candidate_without_external_evidence_is_rejected(self):
        mapped, candidate = self._controlled_candidate(True)
        outside = mapped.frames.pop(3)
        for point in mapped.points.values():
            feature = point.observations.pop(3)
            del outside.feature_to_point[feature]
        mapped.validate()
        with patch("orb_slam_mono.bundle_adjustment.least_squares", side_effect=candidate):
            report = optimize_local_map(mapped, CAMERA, BundleConfig())
        self.assertFalse(report["applied"])
        self.assertEqual(report["solver_termination"], "budget_exhausted")
        self.assertEqual(report["heldout_observations"], 0)
        self.assertEqual(report["reason"], "insufficient_validation_observations")

    def test_unoptimized_landmark_seen_by_variable_camera_is_validated(self):
        mapped, candidate = self._controlled_candidate(True)
        frame = mapped.frames[2]
        xyz = np.array([0.4, 0.1, 8.])
        uv, _ = _projection(frame.R_cw, frame.t_cw, xyz.reshape(1, 3), CAMERA)
        feature_index = frame.features.count
        frame.features = FeatureSet(
            frame.features.keypoints + [cv2.KeyPoint(float(uv[0, 0]), float(uv[0, 1]), 10)],
            np.vstack((frame.features.descriptors, np.zeros((1, 32), np.uint8))))
        mapped.points[1000] = MapPoint(1000, xyz, np.zeros(32, np.uint8))
        mapped.add_observation(1000, frame, feature_index)
        with patch("orb_slam_mono.bundle_adjustment.least_squares", side_effect=candidate):
            report = optimize_local_map(mapped, CAMERA, BundleConfig())
        self.assertTrue(report["applied"], report)
        self.assertEqual(report["heldout_observations"], 31)
        self.assertEqual(report["heldout_outside_window_observations"], 30)

    def test_exact_projection_zero_residual_and_pose_convention(self):
        mapped, truth = _fixture(20)
        frame = mapped.frames[2]
        projected = CAMERA.matrix @ np.column_stack((frame.R_cw, frame.t_cw))
        for i, xyz in enumerate(truth):
            uv = projected @ np.r_[xyz, 1.]
            uv = uv[:2] / uv[2]
            np.testing.assert_allclose(uv, frame.features.keypoints[i].pt, atol=1e-5)
        np.testing.assert_allclose(frame.camera_center_world, [2., 0., 0.])
        self.assertAlmostEqual(np.linalg.det(frame.R_cw), 1.)

    def test_noisy_ba_improves_cost_and_preserves_fixed_gauge(self):
        mapped = self._noisy_map()
        before = [(mapped.frames[i].R_cw.copy(), mapped.frames[i].t_cw.copy()) for i in (0, 1)]
        report = optimize_local_map(mapped, CAMERA, BundleConfig(max_nfev=18))
        self.assertTrue(report["applied"], report)
        self.assertLess(report["final_cost"], report["initial_cost"])
        self.assertLess(report["final_median_px"], report["initial_median_px"])
        for i, (R, t) in zip((0, 1), before):
            np.testing.assert_array_equal(mapped.frames[i].R_cw, R)
            np.testing.assert_array_equal(mapped.frames[i].t_cw, t)
        mapped.validate()

    def test_sparse_jacobian_pattern_has_local_dependence(self):
        pattern = jacobian_sparsity([None, 0], [6, 9], 12).toarray()
        self.assertEqual(pattern.shape, (4, 12))
        np.testing.assert_array_equal(pattern[0, :6], 0)
        np.testing.assert_array_equal(pattern[2, :6], 1)
        np.testing.assert_array_equal(pattern[0, 6:9], 1)
        np.testing.assert_array_equal(pattern[2, 9:12], 1)
        self.assertEqual(int(pattern.sum()), 24)

    def test_sparse_pattern_contains_finite_difference_sensitivities(self):
        # One fixed-pose/point observation and one variable-pose/other-point observation.
        pattern = jacobian_sparsity([None, 0], [6, 9], 12).toarray().astype(bool)
        values = np.r_[np.array([0.01, 0.02, -0.01, -1., 0.03, 0.02]),
                       np.array([0.4, 0.1, 8., -0.3, 0.2, 9.])]
        def residual(x):
            R = cv2.Rodrigues(x[:3])[0]
            first = _projection(np.eye(3), np.zeros(3), x[6:9].reshape(1, 3), CAMERA)[0][0]
            second = _projection(R, x[3:6], x[9:12].reshape(1, 3), CAMERA)[0][0]
            return np.r_[first, second]
        eps = 1e-6
        fd = np.column_stack([(residual(values + eps * np.eye(12)[i]) -
                               residual(values - eps * np.eye(12)[i])) / (2 * eps)
                              for i in range(12)])
        self.assertTrue(np.isfinite(fd).all())
        self.assertLess(np.max(np.abs(fd[~pattern])), 1e-7)
        self.assertGreater(np.max(np.abs(fd[pattern])), 1.0)

    def test_underconstrained_invalid_depth_and_failed_optimizer_rollback(self):
        mapped, _ = _fixture(4)
        self.assertEqual(optimize_local_map(mapped, CAMERA, BundleConfig())["reason"],
                         "insufficient_multiview_points")
        mapped = self._noisy_map()
        mapped.points[0].xyz_world[2] = -8
        self.assertEqual(optimize_local_map(mapped, CAMERA, BundleConfig())["reason"],
                         "invalid_initial_geometry")
        mapped = self._noisy_map()
        old_pose = mapped.frames[2].t_cw.copy()
        old_points = {i: point.xyz_world.copy() for i, point in mapped.points.items()}
        with patch("orb_slam_mono.bundle_adjustment.least_squares", side_effect=ValueError("failure")):
            self.assertFalse(optimize_local_map(mapped, CAMERA, BundleConfig())["applied"])
        np.testing.assert_array_equal(mapped.frames[2].t_cw, old_pose)
        for i, xyz in old_points.items():
            np.testing.assert_array_equal(mapped.points[i].xyz_world, xyz)

    def test_nonfinite_optimizer_result_is_rejected_without_mutation(self):
        mapped = self._noisy_map()
        old_pose = mapped.frames[2].t_cw.copy()
        old_points = {i: point.xyz_world.copy() for i, point in mapped.points.items()}
        class BadResult:
            success = False
            status = 0
            message = "nonfinite"
            nfev = 1
            x = np.full(6 + 3 * 30, np.nan)
        with patch("orb_slam_mono.bundle_adjustment.least_squares", return_value=BadResult()):
            report = optimize_local_map(mapped, CAMERA, BundleConfig())
        self.assertFalse(report["applied"])
        self.assertEqual(report["reason"], "nonfinite_solution")
        np.testing.assert_array_equal(mapped.frames[2].t_cw, old_pose)
        for i, xyz in old_points.items():
            np.testing.assert_array_equal(mapped.points[i].xyz_world, xyz)

    def test_outlier_observation_does_not_publish_invalid_geometry(self):
        mapped = self._noisy_map()
        mapped.frames[2].features.keypoints[3].pt = (950., 550.)
        before = [mapped.frames[i].t_cw.copy() for i in (0, 1)]
        report = optimize_local_map(mapped, CAMERA, BundleConfig(max_nfev=8))
        self.assertIn(report["status"], ("accepted", "rejected"))
        mapped.validate()
        for i, t in zip((0, 1), before):
            np.testing.assert_array_equal(mapped.frames[i].t_cw, t)
        self.assertTrue(all(np.isfinite(point.xyz_world).all() for point in mapped.points.values()))

    def test_segment_scope_and_window_boundary(self):
        mapped = self._noisy_map()
        mapped.active_keyframe_ids = [0, 1]
        self.assertEqual(optimize_local_map(mapped, CAMERA, BundleConfig())["reason"],
                         "underconstrained_keyframe_window")
        mapped.active_keyframe_ids = [0, 1, 2]
        mapped.keyframes[2].segment_id = 5
        with self.assertRaises(ValueError):
            mapped.validate()

    def test_disconnected_anchor_graph_is_rejected(self):
        mapped = self._noisy_map()
        for point in mapped.points.values():
            feature = point.observations.pop(0)
            del mapped.frames[0].feature_to_point[feature]
        mapped.validate()
        report = optimize_local_map(mapped, CAMERA, BundleConfig())
        self.assertEqual(report["reason"], "disconnected_observation_graph")
        self.assertFalse(report["applied"])


class IntegratedTests(unittest.TestCase):
    def test_final_map_plot_uses_adjusted_keyframe_pose_stage(self):
        mapped, _ = _fixture(8)
        online = [{"frame_id": 2, "segment_id": 4, "pose_valid": True,
                   "camera_center_world": [2., 0., 0.]}]
        mapped.frames[2].t_cw = np.array([-2.2, 0., 0.])
        selected = _final_map_pose_records(mapped, online)
        self.assertEqual({record["frame_id"] for record in selected}, {0, 1, 2})
        adjusted = next(record for record in selected if record["frame_id"] == 2)
        self.assertAlmostEqual(adjusted["camera_center_world"][0], 2.2)
        self.assertNotEqual(adjusted["camera_center_world"], online[0]["camera_center_world"])

    def test_m4_loss_reinitializes_in_new_segment(self):
        sequence = generate_sequence(count=18, blank_frames=(7,))
        tracker = MonocularTracker(sequence.camera, OrbExtractor(), DescriptorMatcher(),
                                   GeometryConfig(), TrackingConfig(), MappingConfig(enabled=True))
        results = [tracker.process(sample) for sample in sequence.samples]
        self.assertEqual(results[7].state, TrackingState.LOST)
        self.assertIsNone(results[7].R_cw)
        self.assertGreaterEqual(len(tracker.segments), 2)
        self.assertNotEqual(tracker.segments[0].segment_id, tracker.segments[1].segment_id)
        self.assertTrue(set(tracker.segments[0].points).isdisjoint(tracker.segments[1].points))
        for segment in tracker.segments:
            segment.validate()

    def test_mapping_exception_keeps_accepted_tracking_pose(self):
        sequence = generate_sequence()
        tracker = MonocularTracker(sequence.camera, OrbExtractor(), DescriptorMatcher(),
                                   GeometryConfig(), TrackingConfig(), MappingConfig(enabled=True))
        with patch("orb_slam_mono.tracking.triangulate_new_points", side_effect=ValueError("bad candidate")):
            results = [tracker.process(sample) for sample in sequence.samples[:9]]
        self.assertTrue(any(r.diagnostics.get("mapping", {}).get("status") == "rejected"
                            for r in results))
        self.assertTrue(all(r.R_cw is not None for r in results if r.state == TrackingState.TRACKING))
        for segment in tracker.segments:
            segment.validate()

    def test_rejected_pnp_does_not_change_descriptor_evidence(self):
        sequence = generate_sequence(count=12, blank_frames=(7,))
        tracker = MonocularTracker(sequence.camera, OrbExtractor(), DescriptorMatcher(),
                                   GeometryConfig(), TrackingConfig(),
                                   MappingConfig(enabled=True, update_descriptors=True))
        for sample in sequence.samples[:7]:
            tracker.process(sample)
        mapped = tracker.active_map
        evidence = {i: [descriptor.copy() for descriptor in point.descriptor_evidence]
                    for i, point in mapped.points.items()}
        failed = tracker.process(sequence.samples[7])
        self.assertEqual(failed.state, TrackingState.LOST)
        self.assertIsNone(failed.R_cw)
        for i, point in mapped.points.items():
            self.assertEqual(len(point.descriptor_evidence), len(evidence[i]))
            for before, after in zip(evidence[i], point.descriptor_evidence):
                np.testing.assert_array_equal(before, after)

    def test_varied_visibility_creates_image_derived_points_used_later(self):
        sequence = generate_sequence(count=18, vary_visibility=True)
        extractor = OrbExtractor()
        tracker = MonocularTracker(sequence.camera, extractor, DescriptorMatcher(),
                                   GeometryConfig(), TrackingConfig(), MappingConfig(enabled=True))
        results = [tracker.process(sample) for sample in sequence.samples]
        self.assertGreater(sum(e["new_points"] for e in tracker.mapping_events), 30)
        self.assertTrue(any(len(seg.keyframes) >= 3 for seg in tracker.segments))
        self.assertTrue(any(r.R_cw is None for r in results))
        self.assertTrue(all(r.R_cw is None for r in results if r.state != TrackingState.TRACKING))
        for segment in tracker.segments:
            segment.validate()
            first_new_id = min(segment.points) + len([p for p in segment.points.values()
                                                      if segment.anchor_frame_id in p.observations])
            self.assertTrue(any(point.point_id >= first_new_id and point.observation_count >= 3
                                for point in segment.points.values()))

    def test_ablation_switches_and_exports(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/default.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            a = run_sequence("synthetic", Path(directory) / "a", config, ablation="baseline")
            b = run_sequence("synthetic", Path(directory) / "b", config, ablation="growth")
            self.assertEqual(a["new_map_points"], 0)
            self.assertGreater(b["new_map_points"], 0)
            self.assertEqual(a["ba_attempts"], b["ba_attempts"])
            self.assertTrue((Path(directory) / "b" / "map_growth_events.json").exists())
            final = json.loads((Path(directory) / "b" / "final_keyframes.json").read_text())
            self.assertEqual(final["pose_stage"], "Final segment-local keyframe state after accepted BA")
            self.assertTrue(any(p["frame_id"] == 0 for p in final["poses"]))

    def test_optional_kitti_m4_smoke(self):
        root = Path(__file__).resolve().parents[1] / "data/sequences/00"
        if not root.is_dir():
            self.skipTest("Local KITTI sequence 00 is absent")
        sequence = KittiGraySequence(root)
        tracker = MonocularTracker(sequence.camera, OrbExtractor(), DescriptorMatcher(),
                                   GeometryConfig(), TrackingConfig(), MappingConfig(enabled=True))
        outcomes = [tracker.process(sequence[i]) for i in range(16)]
        self.assertEqual(len(outcomes), 16)
        self.assertTrue(all(r.state in TrackingState for r in outcomes))
        for segment in tracker.segments:
            segment.validate()


if __name__ == "__main__":
    unittest.main()
