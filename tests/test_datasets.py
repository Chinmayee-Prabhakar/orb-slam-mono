import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from orb_slam_mono.datasets import KittiGraySequence
from orb_slam_mono.synthetic import SyntheticSequence


class DatasetTests(unittest.TestCase):
    def test_kitti_loading_grayscale_and_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "image_0"
            image_dir.mkdir()
            first = np.full((12, 16), 17, np.uint8)
            second = np.full((12, 16), 29, np.uint8)
            cv2.imwrite(str(image_dir / "000000.png"), first)
            cv2.imwrite(str(image_dir / "000001.png"), second)
            (root / "calib.txt").write_text("P0: 10 0 8 0 0 11 6 0 0 0 1 0\n", encoding="utf-8")
            (root / "times.txt").write_text("0.0\n0.1\n", encoding="utf-8")
            sequence = KittiGraySequence(root)
            self.assertEqual(len(sequence), 2)
            self.assertEqual(sequence.camera.matrix[0, 0], 10)
            self.assertEqual(sequence[1].timestamp_seconds, 0.1)
            self.assertEqual(sequence[1].image.dtype, np.uint8)
            np.testing.assert_array_equal(sequence[0].image, first)

    def test_kitti_rejects_changed_image_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "image_0"
            image_dir.mkdir()
            cv2.imwrite(str(image_dir / "000000.png"), np.zeros((12, 16), np.uint8))
            cv2.imwrite(str(image_dir / "000001.png"), np.zeros((10, 16), np.uint8))
            (root / "calib.txt").write_text("P0: 10 0 8 0 0 11 6 0 0 0 1 0\n", encoding="utf-8")
            sequence = KittiGraySequence(root)
            with self.assertRaises(ValueError):
                sequence[1]

    def test_synthetic_fallback_has_two_grayscale_images(self):
        sequence = SyntheticSequence()
        self.assertEqual(len(sequence), 2)
        self.assertEqual(sequence[0].image.shape, (480, 640))
        self.assertFalse(np.array_equal(sequence[0].image, sequence[1].image))


if __name__ == "__main__":
    unittest.main()
