"""Optional smoke check against the user's local official KITTI subset."""

import json
import tempfile
import unittest
from pathlib import Path

from orb_slam_mono.datasets import KittiGraySequence
from orb_slam_mono.geometry_demo import run_geometry_demo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = PROJECT_ROOT / "data" / "sequences" / "00"


@unittest.skipUnless((SEQUENCE / "image_0" / "000005.png").is_file()
                     and (SEQUENCE / "calib.txt").is_file(),
                     "Local KITTI sequence-00 subset is not installed")
class RealKittiSmokeTest(unittest.TestCase):
    def test_loader_and_explicit_initialization_outcome(self):
        sequence = KittiGraySequence(SEQUENCE)
        self.assertGreaterEqual(len(sequence), 6)
        self.assertEqual(sequence[0].image.shape, (376, 1241))
        self.assertAlmostEqual(sequence.camera.fx, 718.856)
        config = json.loads((PROJECT_ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            result = run_geometry_demo("kitti", Path(temporary), config,
                                       kitti_sequence=SEQUENCE, first_index=0, second_index=5)
        self.assertIn(result["initialization_status"], ("accepted", "rejected"))
        if result["initialization_status"] == "rejected":
            self.assertIsNotNone(result["failure_reason"])
        self.assertEqual(result["dataset"], "kitti_odometry_gray")
        self.assertIsNone(result["synthetic_scene"])
        self.assertIsNone(result["offline_synthetic_evaluation"])
        self.assertEqual(result["diagnostics"]["input_correspondences"],
                         result["retained_matches"])


if __name__ == "__main__":
    unittest.main()
