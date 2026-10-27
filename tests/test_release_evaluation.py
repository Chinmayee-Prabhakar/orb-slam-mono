"""Synthetic contract checks for the public evaluation entry point."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_eval", ROOT / "scripts/evaluate_release.py")
release_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_eval)


def fake_result(*args):
    return {"metrics": {"online_pose_fraction_percent": 50.0,
                        "eligible_online_poses": 0, "in_sample_ate_m": {"rmse": None}},
            "segments": [], "errors": [], "relative": [], "anchors": {}}


class ReleaseEvaluationTests(unittest.TestCase):
    def test_refuses_existing_output_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(release_eval, "load_kitti_poses") as load:
            with self.assertRaises(FileExistsError):
                release_eval.evaluate_release(Path(temp), Path(temp))
            load.assert_not_called()

    def test_missing_ground_truth_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "output"
            with self.assertRaises(OSError):
                release_eval.evaluate_release(Path(temp), out)
            self.assertFalse(out.exists())

    def test_expected_ranges_and_existing_scorer_are_used(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(release_eval, "load_kitti_poses", return_value="truth"), \
             patch.object(release_eval, "evaluate_saved_run", side_effect=fake_result) as score, \
             patch.object(release_eval, "compare_restart", return_value={}), \
             patch.object(release_eval, "repeatability_report", return_value={"verified": True}) as repeat:
            out = Path(temp) / "output"
            result = release_eval.evaluate_release(Path(temp), out)
            self.assertEqual([tuple(c.args[-2:]) for c in score.call_args_list],
                             [(0, 299), (0, 149), (150, 299)])
            repeat.assert_called_once_with(Path(temp).resolve())
            self.assertIsNone(result["runs"]["run-c1"]["ate_rmse_m"])
            self.assertFalse((out / "frozen_comparison.json").exists())
            scope = json.loads((out / "evaluation_scope.json").read_text())
            self.assertEqual(scope["historical_baseline_comparison"]["status"], "not_performed")
            self.assertTrue((out / "run-a1/segments.json").exists())
            self.assertTrue((out / "run-c1/relative_pairs.json").exists())

    def test_failed_repeat_check_publishes_no_results(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(release_eval, "load_kitti_poses", return_value="truth"), \
             patch.object(release_eval, "evaluate_saved_run", side_effect=fake_result), \
             patch.object(release_eval, "compare_restart", return_value={}), \
             patch.object(release_eval, "repeatability_report", side_effect=ValueError("repeat differs")):
            out = Path(temp) / "output"
            with self.assertRaisesRegex(ValueError, "repeat differs"):
                release_eval.evaluate_release(Path(temp), out)
            self.assertFalse(out.exists())

    def test_nonfinite_data_is_not_published_as_json(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(release_eval, "load_kitti_poses", return_value="truth"), \
             patch.object(release_eval, "evaluate_saved_run", side_effect=fake_result), \
             patch.object(release_eval, "compare_restart", return_value={"bad": float("nan")}), \
             patch.object(release_eval, "repeatability_report", return_value={}):
            out = Path(temp) / "output"
            with self.assertRaises(ValueError):
                release_eval.evaluate_release(Path(temp), out)
            self.assertFalse(out.exists())
            self.assertFalse(list(Path(temp).glob(".evaluation-*")))


if __name__ == "__main__":
    unittest.main()
