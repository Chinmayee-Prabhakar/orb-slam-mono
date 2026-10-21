"""Offline truth convention tests; online geometry has no truth interface."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from orb_slam_mono.kitti_pose_evaluation import (direction_error_degrees,
                                                 load_kitti_poses,
                                                 relative_world_to_camera,
                                                 rotation_angle_degrees)


class KittiPoseEvaluationTests(unittest.TestCase):
    def test_relative_composition_and_translation_frame(self):
        base_rotation, _ = cv2.Rodrigues(np.array([0.10, -0.15, 0.08]))
        relative_rotation, _ = cv2.Rodrigues(np.array([0.04, 0.02, -0.06]))
        base = np.eye(4)
        base[:3, :3] = base_rotation
        base[:3, 3] = [3.0, -1.0, 2.0]
        relative = np.eye(4)
        relative[:3, :3] = relative_rotation
        relative[:3, 3] = [-0.7, 0.2, -1.1]
        second = base @ np.linalg.inv(relative)
        rotation, translation = relative_world_to_camera(np.stack((base, second)), 0, 1)
        np.testing.assert_allclose(rotation, relative_rotation, atol=1e-12)
        np.testing.assert_allclose(translation, relative[:3, 3], atol=1e-12)
        point_first = np.array([0.5, 0.2, 6.0])
        point_world = base[:3, :3] @ point_first + base[:3, 3]
        point_second = second[:3, :3].T @ (point_world - second[:3, 3])
        np.testing.assert_allclose(point_second, rotation @ point_first + translation)
        self.assertLess(rotation_angle_degrees(rotation @ relative_rotation.T), 1e-5)
        self.assertLess(direction_error_degrees(translation, relative[:3, 3]), 1e-5)
        self.assertGreater(direction_error_degrees(-translation, relative[:3, 3]), 179.99)

    def test_pose_file_shape_finite_and_rotation_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "00.txt"
            path.write_text("1 0 0 0 0 1 0 0 0 0 1 0\n"
                            "1 0 0 1 0 1 0 0 0 0 1 0\n", encoding="utf-8")
            poses = load_kitti_poses(path)
            self.assertEqual(poses.shape, (2, 4, 4))
            _, t = relative_world_to_camera(poses, 0, 1)
            np.testing.assert_allclose(t, [-1, 0, 0])
            path.write_text("1 0 0 0 0 2 0 0 0 0 1 0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_kitti_poses(path)
            path.write_text("1 0 0 0 0 1 0 0 0 nan 1 0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_kitti_poses(path)


if __name__ == "__main__":
    unittest.main()
