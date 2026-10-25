"""Synthetic controls for M5-B isolation, measurement, and offline scoring."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from time import perf_counter

import numpy as np

from orb_slam_mono.camera import CalibratedCamera
from orb_slam_mono.datasets import ImageSample
from orb_slam_mono.features import OrbExtractor
from orb_slam_mono.geometry import GeometryConfig
from orb_slam_mono.local_mapping import MappingConfig
from orb_slam_mono.matching import DescriptorMatcher
from orb_slam_mono.m5b_evaluation import (common_pose_accuracy, compare_restart,
                                          compare_saved_outputs, describe,
                                          relative_summary)
from orb_slam_mono.m5b_runner import MemorySampler, timing_summary
from orb_slam_mono.tracking import MonocularTracker, TrackingConfig
from orb_slam_mono.trajectory_evaluation import (OnlinePose, evaluate_segments,
                                                 load_online_rows, validate_source_index)


def pose(frame: int, segment: int, center: np.ndarray) -> OnlinePose:
    center = np.asarray(center, dtype=float)
    return OnlinePose(frame, segment, np.eye(3), -center, center)


class M5BTests(unittest.TestCase):
    def test_fresh_tracker_has_no_inherited_map_pose_or_descriptors(self):
        camera = CalibratedCamera(100, 80, 90, 90, 50, 40)
        def new_tracker():
            return MonocularTracker(camera, OrbExtractor(), DescriptorMatcher(),
                                    GeometryConfig(), TrackingConfig(), MappingConfig(enabled=True))
        earlier = new_tracker()
        earlier.process(ImageSample(0, np.zeros((80, 100), dtype=np.uint8), "synthetic"))
        restarted = new_tracker()
        self.assertIsNot(earlier, restarted)
        self.assertIsNotNone(earlier.anchor)
        self.assertIsNone(restarted.anchor)
        self.assertIsNone(restarted.active_map)
        self.assertIsNone(restarted.last_successful_pose)
        self.assertEqual(restarted.segments, [])
        self.assertEqual(restarted.keyframe_events, [])
        self.assertEqual(restarted._next_point_id, 0)

    def test_original_source_ids_and_index_contract_at_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "image_0"
            images.mkdir()
            for frame in range(155):
                (images / f"{frame:06d}.png").write_bytes(b"fixture")
            (root / "times.txt").write_text("\n".join(str(i * 0.1) for i in range(155)))
            index = validate_source_index(root, 160, 150, 154)
            self.assertEqual(index["requested_range"], [150, 154])
            self.assertAlmostEqual(index["first_timestamp_s"], 15.0)
            online = root / "per_frame.jsonl"
            rows = [{"frame_id": i, "state": "initializing", "pose_valid": False,
                     "T_cw_R": None, "T_cw_t": None, "camera_center_world": None}
                    for i in range(150, 155)]
            online.write_text("\n".join(json.dumps(row) for row in rows))
            actual, poses = load_online_rows(online, 150, 154)
            self.assertEqual([r["frame_id"] for r in actual], list(range(150, 155)))
            self.assertEqual(poses, {})
            with self.assertRaises(ValueError):
                load_online_rows(online, 0, 4)

    def test_restart_categories_and_state_agreement_use_original_ids(self):
        def row(frame, valid, state):
            return {"frame_id": frame, "pose_valid": valid, "state": state}
        full = {"rows": [row(150, True, "tracking"), row(151, False, "initializing"),
                         row(152, True, "tracking"), row(153, False, "lost")],
                "poses": {}, "segments": [],
                "events": {"loss_events": [], "initialization_attempts": []}}
        restarted = {"rows": [row(150, False, "initializing"), row(151, False, "initializing"),
                              row(152, True, "tracking"), row(153, True, "tracking")],
                     "poses": {}, "segments": [],
                     "events": {"loss_events": [], "initialization_attempts": []}}
        result = compare_restart(full, restarted, np.repeat(np.eye(4)[None], 154, axis=0))
        self.assertEqual(result["category_counts"],
                         {"both_valid": 1, "full_only": 1, "restart_only": 1, "both_invalid": 1})
        self.assertEqual(result["valid_intersection_ids"], [152])
        self.assertEqual(result["valid_union_ids"], [150, 152, 153])
        self.assertEqual(result["state_agreement_count"], 2)
        self.assertEqual(result["common_pose_accuracy"]["eligible_common_frame_count"], 0)

    def test_common_fit_is_independent_and_proper_per_run_segment(self):
        ids = list(range(150, 156))
        centers = np.array([[0, 0, 0], [1, 0, 0], [2, 0.4, 0],
                            [3, 0.3, 0], [4, -0.3, 0], [5, 0.2, 0]], dtype=float)
        gt = np.repeat(np.eye(4)[None], 156, axis=0)
        gt[ids, :3, 3] = centers * 3 + [10, 4, 0]
        full = {i: pose(i, 0, centers[j]) for j, i in enumerate(ids)}
        restart = {i: pose(i, 7, centers[j] * 2) for j, i in enumerate(ids)}
        result = common_pose_accuracy(full, restart, gt)
        self.assertEqual(result["eligible_common_frame_count"], 6)
        self.assertAlmostEqual(result["scores"]["full"]["segment_scales_m_per_unit"]["0"], 3)
        self.assertAlmostEqual(result["scores"]["restart"]["segment_scales_m_per_unit"]["7"], 1.5)
        self.assertLess(result["scores"]["full"]["ate_m"]["maximum"], 1e-10)
        self.assertLess(result["scores"]["restart"]["ate_m"]["maximum"], 1e-10)
        unavailable = common_pose_accuracy(dict(list(full.items())[:3]),
                                           dict(list(restart.items())[:3]), gt)
        self.assertEqual(unavailable["eligible_common_frame_count"], 0)
        self.assertIsNone(unavailable["scores"]["full"]["ate_m"]["rmse"])

    def test_no_cross_segment_exact_gap_pair(self):
        gt = np.repeat(np.eye(4)[None], 12, axis=0)
        gt[:, 0, 3] = np.arange(12)
        poses = {5: pose(5, 0, [0, 0, 0]), 6: pose(6, 0, [1, 0, 0]),
                 7: pose(7, 1, [0, 0, 0]), 8: pose(8, 1, [1, 0, 0])}
        segments, errors, relative = evaluate_segments(poses, gt, {0: 4, 1: 6})
        self.assertEqual(len(segments), 2)
        self.assertEqual(errors, [])
        self.assertEqual([(r["first_frame"], r["second_frame"]) for r in relative],
                         [(5, 6), (7, 8)])
        self.assertTrue(all(r["scale_ratio"] is None for r in relative))
        self.assertEqual(relative_summary(relative, set())["1"]["fixed_scale_ratio"]["count"], 0)

    def test_runtime_missing_stages_and_wall_scope(self):
        rows = [{"timing_ms": {"total": 10.0, "orb": 5.0}},
                {"timing_ms": {"total": 20.0, "pnp": 2.0}}]
        report = timing_summary(rows, 0.5)
        self.assertEqual(report["stage_runtime_ms"]["total"]["count"], 2)
        self.assertEqual(report["stage_runtime_ms"]["pnp"]["count"], 1)
        self.assertEqual(report["stage_runtime_ms"]["initialization"]["count"], 0)
        self.assertIsNone(report["stage_runtime_ms"]["initialization"]["median_ms"])
        self.assertAlmostEqual(report["recorded_frame_processing_seconds"], .03)
        self.assertAlmostEqual(report["frames_per_second_including_loading_exports"], 4)

    def test_memory_os_peak_covers_unsampled_transient(self):
        observations = iter([{"working_set_bytes": 100, "os_peak_working_set_bytes": 100,
                              "private_bytes": 50},
                             {"working_set_bytes": 120, "os_peak_working_set_bytes": 300,
                              "private_bytes": 70},
                             {"working_set_bytes": 110, "os_peak_working_set_bytes": 300,
                              "private_bytes": 60}])
        sampler = MemorySampler(probe=lambda: next(observations))
        sampler._started = perf_counter()
        sampler.sample()
        sampler.sample()
        sampler.sample()
        report = sampler.report()
        self.assertEqual(report["sampled_peak_working_set_bytes"], 120)
        self.assertEqual(report["os_peak_working_set_bytes"], 300)
        self.assertEqual(report["peak_increase_from_start_bytes"], 200)
        self.assertEqual(report["sampled_peak_private_bytes"], 70)

    def test_repeat_comparison_ignores_timing_not_online_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            folders = [Path(directory) / name for name in ("a", "b")]
            for folder, elapsed in zip(folders, (1, 99)):
                (folder / "maps").mkdir(parents=True)
                row = {"frame_id": 150, "state": "initializing", "segment_id": None,
                       "pose_valid": False, "failure_reason": "waiting",
                       "T_cw_R": None, "T_cw_t": None, "camera_center_world": None,
                       "diagnostics": {"features_detected": 4}, "timing_ms": {"total": elapsed}}
                (folder / "per_frame.jsonl").write_text(json.dumps(row) + "\n")
                for name in ("estimated_poses.json", "final_keyframes.json"):
                    (folder / name).write_text("{}")
                for name in ("initialization_attempts", "loss_events", "recovery_events",
                             "keyframe_events", "map_growth_events", "ba_events"):
                    (folder / (name + ".json")).write_text("[]")
            self.assertTrue(compare_saved_outputs(*folders)["passed"])
            path = folders[1] / "per_frame.jsonl"
            row = json.loads(path.read_text())
            row["pose_valid"] = True
            path.write_text(json.dumps(row) + "\n")
            self.assertFalse(compare_saved_outputs(*folders)["passed"])

    def test_ground_truth_absent_from_online_runner(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("m5b_runner.py", "sequence_demo.py", "tracking.py", "local_mapping.py"):
            source = (root / "src/orb_slam_mono" / name).read_text(encoding="utf-8")
            self.assertNotIn("ground_truth", source)
            self.assertNotIn("trajectory_evaluation", source)

    def test_unavailable_statistics_remain_null(self):
        report = describe([], "m")
        self.assertEqual(report["count"], 0)
        self.assertIsNone(report["maximum"])
        self.assertIsNone(report["rmse"])


if __name__ == "__main__":
    unittest.main()
