"""Independent synthetic and I/O contracts for frozen offline trajectory evaluation."""

import ast
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from orb_slam_mono.trajectory_evaluation import (
    OnlinePose, _relative, _write_csv, _write_json, camera_center, common_frame_comparison,
    coverage_report, direction_error_deg, evaluate_segments, fit_similarity,
    inverse_t_cw, load_online_rows, rotation_error_deg, validate_source_index,
    validate_anchor_keyframes,
)


def rot(vector):
    return cv2.Rodrigues(np.asarray(vector, dtype=np.float64))[0]


def curved_points(n=12):
    t = np.arange(n, dtype=np.float64)
    return np.stack((t, 0.8 * np.sin(t * 0.6), 0.6 * np.cos(t * 0.8)), axis=1)


class TrajectoryEvaluationTests(unittest.TestCase):
    def test_nontrivial_t_cw_inverse_and_center(self):
        R_wc = rot([0.2, -0.3, 0.1])
        center = np.array([2.5, -3.0, 7.2])
        R_cw, t_cw = R_wc.T, -R_wc.T @ center
        np.testing.assert_allclose(camera_center(R_cw, t_cw), center, atol=1e-12)
        pose = inverse_t_cw(R_cw, t_cw)
        np.testing.assert_allclose(pose[:3, :3], R_wc, atol=1e-12)
        np.testing.assert_allclose(pose[:3, 3], center, atol=1e-12)
        world = np.array([-1.0, 4.2, 9.0])
        camera = R_cw @ world + t_cw
        np.testing.assert_allclose(pose[:3, :3] @ camera + center, world)

    def test_sim3_recovers_scale_proper_rotation_and_translation(self):
        X = curved_points()
        Q, scale, b = rot([0.3, -0.2, 0.4]), 3.7, np.array([4.0, -2.0, 9.0])
        Y = scale * X @ Q.T + b
        fit, reason = fit_similarity(X, Y)
        self.assertIsNone(reason)
        self.assertAlmostEqual(fit.scale, scale, places=10)
        np.testing.assert_allclose(fit.rotation, Q, atol=1e-10)
        np.testing.assert_allclose(fit.translation, b, atol=1e-10)
        np.testing.assert_allclose(fit.transform(X), Y, atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(fit.rotation), 1.0, places=10)

    def test_reflection_cannot_be_perfectly_fitted(self):
        X = curved_points()
        reflected = X @ np.diag([-1., 1., 1.])
        fit, reason = fit_similarity(X, reflected)
        self.assertIsNone(reason)
        self.assertGreater(fit.scale, 0)
        self.assertGreater(np.linalg.det(fit.rotation), 0.999)
        self.assertGreater(np.sqrt(np.mean(np.sum((fit.transform(X) - reflected) ** 2, axis=1))), 0.01)

    def test_short_repeated_collinear_planar_and_nonfinite(self):
        planar = curved_points()[:, :2]
        planar = np.column_stack((planar, np.zeros(len(planar))))
        fit, reason = fit_similarity(planar, 2.0 * planar + [1., 2., 3.])
        self.assertIsNone(reason)
        self.assertGreater(fit.scale, 0)
        self.assertIsNotNone(fit_similarity(planar[:4], planar[:4])[1])
        self.assertIn("repeated", fit_similarity(np.zeros((8, 3)), np.zeros((8, 3)))[1])
        line = np.column_stack((np.arange(8.), np.zeros(8), np.zeros(8)))
        self.assertIn("near_collinear", fit_similarity(line, line)[1])
        bad = planar.copy()
        bad[0, 0] = np.nan
        self.assertIn("nonfinite", fit_similarity(bad, planar)[1])

    def test_kitti_t_wc_and_relative_units(self):
        Ra, Rb = rot([0.12, -0.09, 0.07]), rot([-0.1, 0.15, 0.02])
        Ca, Cb = np.array([4., -2., 5.]), np.array([5., -1., 6.])
        gt = np.repeat(np.eye(4)[None], 6, axis=0)
        gt[1, :3, :3], gt[1, :3, 3] = Ra, Ca
        gt[5, :3, :3], gt[5, :3, 3] = Rb, Cb
        a = OnlinePose(1, 0, Ra.T, -Ra.T @ Ca, Ca)
        b = OnlinePose(5, 0, Rb.T, -Rb.T @ Cb, Cb)
        row = _relative(a, b, gt, 1.0)
        self.assertEqual(row["source_gap"], 4)
        self.assertLess(row["rotation_error_deg"], 1e-5)
        self.assertLess(row["translation_direction_error_deg"], 1e-5)
        self.assertAlmostEqual(row["displacement_error_m"], 0.0, places=10)
        self.assertAlmostEqual(row["relative_translation_vector_error_m"], 0.0, places=10)
        self.assertAlmostEqual(row["scale_ratio"], 1.0, places=10)
        self.assertAlmostEqual(row["gt_displacement_m"], np.sqrt(3), places=10)
        self.assertAlmostEqual(rotation_error_deg(Ra, Ra @ rot([0, 0, np.pi / 2])), 90.0, places=8)
        self.assertAlmostEqual(direction_error_deg([1, 0, 0], [0, 1, 0]), 90.0, places=8)

    def test_exact_gap_pairing_and_segment_isolation(self):
        gt = np.repeat(np.eye(4)[None], 20, axis=0)
        points = {1: [0, 0, 0], 2: [1, 1, 0], 6: [3, 2, 0], 7: [4, 1, 0],
                  11: [2, 0, 0], 12: [3, 1, 0]}
        poses = {}
        for frame, center in points.items():
            center = np.asarray(center, dtype=float)
            gt[frame, :3, 3] = center
            poses[frame] = OnlinePose(frame, 0 if frame < 10 else 1, np.eye(3), -center, center)
        _, _, relative = evaluate_segments(poses, gt, {0: 0, 1: 10})
        pairs = {(r["first_frame"], r["second_frame"], r["source_gap"]) for r in relative}
        self.assertIn((1, 2, 1), pairs)
        self.assertIn((1, 6, 5), pairs)
        self.assertNotIn((7, 11, 4), pairs)
        self.assertTrue(all(r["segment_id"] == poses[r["second_frame"]].segment_id for r in relative))
        self.assertTrue(all(r["scaled_displacement_m"] is None for r in relative))

    def test_independent_segment_scales_and_deterministic_metrics(self):
        X = curved_points(8)
        gt = np.repeat(np.eye(4)[None], 19, axis=0)
        poses = {}
        for segment, first, scale, Q, b in (
                (0, 1, 2.0, rot([0.2, -0.1, 0.05]), np.array([1., 2., 3.])),
                (1, 11, 5.0, rot([-0.15, 0.1, 0.3]), np.array([-2., 4., 8.]))):
            gt[first - 1, :3, :3] = Q
            for offset, center in enumerate(X):
                frame = first + offset
                poses[frame] = OnlinePose(frame, segment, np.eye(3), -center, center)
                gt[frame, :3, :3] = Q
                gt[frame, :3, 3] = scale * Q @ center + b
        first_result = evaluate_segments(poses, gt, {0: 0, 1: 10})
        second_result = evaluate_segments(poses, gt, {0: 0, 1: 10})
        segments = first_result[0]
        self.assertEqual([s["status"] for s in segments], ["eligible", "eligible"])
        self.assertAlmostEqual(segments[0]["alignment"]["scale_m_per_estimated_unit"], 2.0)
        self.assertAlmostEqual(segments[1]["alignment"]["scale_m_per_estimated_unit"], 5.0)
        self.assertEqual(json.dumps(first_result, sort_keys=True), json.dumps(second_result, sort_keys=True))

    def test_segment_fit_orientation_and_anchor_exclusion(self):
        X = curved_points(12)
        Q, b, scale = rot([0.3, 0.2, -0.1]), np.array([2., 1., -3.]), 2.4
        initial_rotation = rot([0.13, -0.21, 0.09])
        gt = np.repeat(np.eye(4)[None], 13, axis=0)
        gt[0, :3, :3] = Q
        poses = {}
        for i, center in enumerate(X, start=1):
            R_wc = initial_rotation @ rot([0.01 * i, 0.02 * i, -0.015 * i])
            gt[i, :3, :3] = Q @ R_wc
            gt[i, :3, 3] = scale * Q @ center + b
            poses[i] = OnlinePose(i, 0, R_wc.T, -R_wc.T @ center, center)
        segments, errors, _ = evaluate_segments(poses, gt, {0: 0})
        self.assertEqual(segments[0]["status"], "eligible")
        self.assertEqual(len(errors), 12)
        self.assertAlmostEqual(segments[0]["in_sample_ate_m"]["rmse"], 0.0, places=9)
        self.assertLess(segments[0]["in_sample_orientation_deg"]["rmse"], 1e-5)
        self.assertLess(segments[0]["anchor_referenced_orientation_deg"]["median"], 1e-5)
        self.assertEqual(segments[0]["prefix_fit_suffix_evaluation"]["status"], "eligible")
        self.assertEqual(segments[0]["prefix_fit_suffix_evaluation"]["suffix_pose_count"], 6)

    def test_frame_mapping_and_timestamps(self):
        with tempfile.TemporaryDirectory() as temporary:
            sequence = Path(temporary)
            images = sequence / "image_0"
            images.mkdir()
            for i in range(6):
                (images / f"{i:06d}.png").write_bytes(b"image")
            (sequence / "times.txt").write_text("\n".join(str(i * .1) for i in range(6)))
            result = validate_source_index(sequence, 7, 2, 5)
            self.assertEqual(result["requested_range"], [2, 5])
            self.assertEqual(result["first_timestamp_s"], .2)
            with self.assertRaisesRegex(ValueError, "GT rows"):
                validate_source_index(sequence, 5, 0, 5)
            (sequence / "times.txt").write_text("0\n1\n2")
            with self.assertRaisesRegex(ValueError, "Timestamp count"):
                validate_source_index(sequence, 7, 0, 5)
            (sequence / "times.txt").write_text("0\n1\n2\n3\n4\n5")
            (sequence / "times.txt").write_text("1\n2\n3\n4\n5\n6")
            with self.assertRaisesRegex(ValueError, "timestamp zero"):
                validate_source_index(sequence, 7, 0, 5)
            (sequence / "times.txt").write_text("0\n1\n2\n3\n4\n5")
            (images / "000003.png").unlink()
            with self.assertRaisesRegex(ValueError, "contiguous"):
                validate_source_index(sequence, 7, 0, 5)
            (images / "000003.png").write_bytes(b"image")
            (images / "000006.png").write_bytes(b"image")
            with self.assertRaisesRegex(ValueError, "Timestamp count"):
                validate_source_index(sequence, 7, 0, 5)
            (images / "000006.png").unlink()
            (images / "000005.png").unlink()
            (images / "000010.png").write_bytes(b"image")
            with self.assertRaisesRegex(ValueError, "contiguous"):
                validate_source_index(sequence, 11, 0, 4)

    def test_online_null_poses_and_wrong_frame_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "per_frame.jsonl"
            rows = [{"frame_id": 0, "state": "initializing", "segment_id": None,
                     "pose_valid": False, "T_cw_R": None, "T_cw_t": None, "camera_center_world": None},
                    {"frame_id": 1, "state": "tracking", "segment_id": 0,
                     "pose_valid": True, "T_cw_R": np.eye(3).tolist(),
                     "T_cw_t": [-1, 0, 0], "camera_center_world": [1, 0, 0]}]
            path.write_text("\n".join(json.dumps(row) for row in rows))
            _, poses = load_online_rows(path, 0, 1)
            self.assertEqual(list(poses), [1])
            rows[0]["frame_id"] = 10
            path.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError, "source range"):
                load_online_rows(path, 0, 1)
            rows[0]["frame_id"] = 0
            rows[0]["camera_center_world"] = [0, 0, 0]
            path.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError, "unestimated"):
                load_online_rows(path, 0, 1)

    def test_saved_anchor_is_identity_but_not_online_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "final_keyframes.json"
            row = {"frame_id": 0, "segment_id": 0, "T_cw_R": np.eye(3).tolist(),
                   "T_cw_t": [0., 0., 0.]}
            path.write_text(json.dumps({"pose_stage": "final", "poses": [row]}))
            report = validate_anchor_keyframes(path, {0: 0})
            self.assertEqual(len(report["anchors"]), 1)
            row["T_cw_t"] = [1., 0., 0.]
            path.write_text(json.dumps({"pose_stage": "final", "poses": [row]}))
            with self.assertRaisesRegex(ValueError, "not the segment world origin"):
                validate_anchor_keyframes(path, {0: 0})

    def test_coverage_excludes_initialization_anchor(self):
        rows = [{"frame_id": 0, "state": "initializing", "segment_id": None,
                 "pose_valid": False, "failure_reason": None}]
        rows += [{"frame_id": i, "state": "tracking", "segment_id": 0,
                  "pose_valid": True, "failure_reason": None} for i in range(1, 6)]
        poses = {i: OnlinePose(i, 0, np.eye(3), -np.array([float(i), 0, 0]),
                               np.array([float(i), 0, 0])) for i in range(1, 6)}
        report = coverage_report(rows, poses,
                                 [{"initialization_status": "accepted", "candidate_frame": 1,
                                   "anchor_frame": 0, "failure_reason": None}], [],
                                 {"valid_poses": 5, "longest_valid_span": 5,
                                  "no_pose_frames": 1, "initialization_attempts": 1}, [], [])
        self.assertEqual(report["online_valid_poses"], 5)
        self.assertEqual(report["segment_anchor_frames"], {"0": 0})
        self.assertEqual(report["missing_pose_intervals"], [{"first": 0, "last": 0, "length": 1}])

    def test_common_frame_alignment_uses_shared_ids(self):
        X = curved_points(8)
        gt = np.repeat(np.eye(4)[None], 10, axis=0)
        variants = {}
        for name, scale in (("baseline", 1.), ("growth", 2.), ("descriptors", 3.), ("full", 4.)):
            poses = {}
            for i, center in enumerate(X, start=1):
                gt[i, :3, 3] = center
                local = center / scale
                poses[i] = OnlinePose(i, 0, np.eye(3), -local, local)
            variants[name] = {"poses": poses}
        report = common_frame_comparison(variants, gt)
        self.assertEqual(report["eligible_common_frame_count"], 8)
        for name in variants:
            self.assertAlmostEqual(report["variants"][name]["ate_m"]["rmse"], 0.0, places=9)

    def test_online_sources_have_no_gt_evaluator_import(self):
        source_dir = Path(__file__).parents[1] / "src" / "orb_slam_mono"
        online = ("tracking.py", "sequence_demo.py", "datasets.py", "matching.py", "geometry.py",
                  "features.py", "map_state.py", "local_mapping.py", "bundle_adjustment.py")
        for name in online:
            tree = ast.parse((source_dir / name).read_text(encoding="utf-8"))
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
            self.assertFalse(any("trajectory_evaluation" in value or "kitti_pose_evaluation" in value
                                 or "offline_evaluation" in value for value in imports), name)

    def test_json_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            a, b = Path(temporary) / "a.json", Path(temporary) / "b.json"
            _write_json(a, {"z": 2, "a": [1, 2]})
            _write_json(b, {"a": [1, 2], "z": 2})
            self.assertEqual(a.read_bytes(), b.read_bytes())
            _write_csv(a, [{"frame_id": 1, "ate_m": 0.25}], ("frame_id", "ate_m"))
            _write_csv(b, [{"ate_m": 0.25, "frame_id": 1}], ("frame_id", "ate_m"))
            self.assertEqual(a.read_bytes(), b.read_bytes())


if __name__ == "__main__":
    unittest.main()
