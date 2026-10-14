"""Small regression checks for M1 feature and matcher edge cases."""

import unittest

import numpy as np

from orb_slam_mono.features import OrbExtractor
from orb_slam_mono.matching import DescriptorMatcher
from orb_slam_mono.synthetic import SyntheticSequence


class M1RegressionTests(unittest.TestCase):
    def test_blank_image_and_empty_descriptors(self):
        extractor = OrbExtractor()
        blank = np.zeros((240, 320), dtype=np.uint8)
        features = extractor.extract(blank)
        self.assertEqual(features.count, 0)
        self.assertIsNone(features.descriptors)
        match = DescriptorMatcher().match(features.descriptors, features.descriptors)
        self.assertEqual(match.retained, 0)
        self.assertEqual(match.knn_pairs, 0)

    def test_descriptor_format_and_reproducible_synthetic_counts(self):
        sequence = SyntheticSequence()
        extractor = OrbExtractor()
        first = extractor.extract(sequence[0].image)
        second = extractor.extract(sequence[1].image)
        self.assertGreaterEqual(first.count, 100)
        self.assertGreaterEqual(second.count, 100)
        self.assertEqual(first.descriptors.dtype, np.uint8)
        self.assertEqual(first.descriptors.shape, (first.count, 32))
        match = DescriptorMatcher().match(first.descriptors, second.descriptors)
        self.assertGreaterEqual(match.retained, 30)
        repeat = DescriptorMatcher().match(first.descriptors, second.descriptors)
        self.assertEqual([m.queryIdx for m in match.matches], [m.queryIdx for m in repeat.matches])


if __name__ == "__main__":
    unittest.main()
