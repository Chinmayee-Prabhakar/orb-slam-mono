import tempfile
import unittest
from pathlib import Path

import numpy as np

from orb_slam_mono.camera import CalibratedCamera, load_kitti_p0


class CameraTests(unittest.TestCase):
    def test_matrix_round_trip(self):
        matrix = np.array([[700.0, 0.0, 600.0], [0.0, 701.0, 180.0], [0.0, 0.0, 1.0]])
        camera = CalibratedCamera.from_matrix(matrix, width=1241, height=376)
        np.testing.assert_array_equal(camera.matrix, matrix)
        self.assertEqual((camera.width, camera.height), (1241, 376))

    def test_rejects_invalid_intrinsics(self):
        with self.assertRaises(ValueError):
            CalibratedCamera(640, 480, -1, 500, 320, 240)
        with self.assertRaises(ValueError):
            CalibratedCamera.from_matrix(np.eye(4), 640, 480)
        with self.assertRaises(ValueError):
            CalibratedCamera.from_matrix(np.array([[500, 2, 320], [0, 500, 240], [0, 0, 1]]), 640, 480)

    def test_kitti_p0(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calib.txt"
            path.write_text("P0: 700 0 600 0 0 701 180 0 0 0 1 0\nP1: 1 2 3\n", encoding="utf-8")
            camera = load_kitti_p0(path, 1241, 376)
            self.assertEqual((camera.fx, camera.fy, camera.cx, camera.cy), (700, 701, 600, 180))

    def test_kitti_p0_rejects_missing_or_malformed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calib.txt"
            path.write_text("P1: 1 2 3\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_kitti_p0(path, 640, 480)
            path.write_text("P0: 1 2 3\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_kitti_p0(path, 640, 480)
            path.write_text("P0: 700 0 320 0 0 bad 240 0 0 0 1 0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_kitti_p0(path, 640, 480)


if __name__ == "__main__":
    unittest.main()
