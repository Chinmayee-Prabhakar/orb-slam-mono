import unittest

import numpy as np

from orb_slam_mono.matching import DescriptorMatcher


class MatchingTests(unittest.TestCase):
    def test_retains_clear_correspondences(self):
        first = np.array([[0] * 32, [255] * 32], dtype=np.uint8)
        second = np.array([[0] * 32, [255] * 32, [170] * 32], dtype=np.uint8)
        result = DescriptorMatcher().match(first, second)
        self.assertEqual(result.knn_pairs, 2)
        self.assertEqual(result.ratio_passed, 2)
        self.assertEqual(result.retained, 2)
        self.assertEqual([(m.queryIdx, m.trainIdx) for m in result.matches], [(0, 0), (1, 1)])

    def test_mutual_check_removes_duplicate_assignment(self):
        first = np.zeros((2, 32), dtype=np.uint8)
        second = np.array([[0] * 32, [255] * 32], dtype=np.uint8)
        result = DescriptorMatcher(mutual_check=True).match(first, second)
        self.assertEqual(result.ratio_passed, 2)
        self.assertEqual(result.retained, 1)

    def test_empty_and_invalid_descriptors(self):
        matcher = DescriptorMatcher()
        self.assertEqual(matcher.match(None, None).retained, 0)
        with self.assertRaises(ValueError):
            matcher.match(np.zeros((1, 16), np.uint8), np.zeros((2, 32), np.uint8))


if __name__ == "__main__":
    unittest.main()
