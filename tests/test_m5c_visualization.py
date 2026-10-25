"""Geometry, source-index, replay, and media checks for the M5-C display."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from orb_slam_mono import m5c_data, m5c_offline_gt, m5c_render


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/m5c-visual-demonstration"
RUN = ROOT / "artifacts/m5b-final-evaluation/run-a1"


class M5CGeometryTests(unittest.TestCase):
    def row(self, R=None, center=None, valid=True):
        R = np.eye(3) if R is None else np.asarray(R, dtype=float)
        center = np.zeros(3) if center is None else np.asarray(center, dtype=float)
        return {"pose_valid": valid, "T_cw_R": R.tolist() if valid else None,
                "T_cw_t": (-R @ center).tolist() if valid else None,
                "camera_center_world": center.tolist() if valid else None}

    def test_camera_center_and_frustum_invert_world_to_camera_pose(self):
        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
        row = self.row(R, [2, -3, 5])
        center, R_wc = m5c_data.pose_geometry(row)
        np.testing.assert_allclose(center, [2, -3, 5])
        np.testing.assert_allclose(R_wc, R.T)
        K = np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]], dtype=float)
        frustum = m5c_data.frustum_world(row, K, 100, 80, depth=2)
        np.testing.assert_allclose(frustum[0], center)
        np.testing.assert_allclose(frustum[1], center + R.T @ [-1, -.8, 2])
        np.testing.assert_allclose(frustum.mean(axis=0)[1], center[1])

    def test_missing_pose_has_no_geometry_or_fabricated_marker(self):
        with self.assertRaises(ValueError):
            m5c_data.pose_geometry(self.row(valid=False))
        with self.assertRaises(ValueError):
            m5c_data.frustum_world(self.row(valid=False), np.eye(3), 100, 80)

    def test_nonrigid_pose_and_wrong_exported_center_reject(self):
        row = self.row()
        row["T_cw_R"][0][0] = 1.1
        with self.assertRaises(ValueError):
            m5c_data.pose_geometry(row)
        row = self.row()
        row["camera_center_world"][0] = 1
        with self.assertRaises(ValueError):
            m5c_data.pose_geometry(row)

    def test_projection_is_fixed_and_equal_unit_within_segment(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        projection = m5c_render.Projection(points, (0, 0, 500, 300))
        before = projection.xy(points)
        self.assertTrue(np.array_equal(before, projection.xy(points)))
        self.assertGreater(projection.scale, 0)
        self.assertTrue(np.allclose(np.linalg.norm(projection.basis, axis=1), 1))
        self.assertAlmostEqual(np.dot(*projection.basis), 0)

    def test_overlay_uses_only_recorded_pixels(self):
        gray = np.full((40, 60), 90, dtype=np.uint8)
        capture = {"keypoints_xy": [[8, 10]],
                   "associations": [{"pixel_xy": [20, 25]}, {"pixel_xy": [45, 30]}],
                   "ransac_inlier_indices": [1]}
        actual = m5c_render.draw_overlay(gray, capture)
        self.assertEqual(actual.shape, (40, 60, 3))
        for x, y, color, radius in ((8, 10, m5c_render.CYAN, 3),
                                     (20, 25, m5c_render.AMBER, 6),
                                     (45, 30, m5c_render.GREEN, 6)):
            patch = actual[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1]
            self.assertLess(np.min(np.linalg.norm(patch.astype(float)-color, axis=2)), 40)
        self.assertTrue(np.array_equal(actual[0, 0], [90, 90, 90]))

    def test_ply_parser_preserves_coordinates_and_checks_population(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.ply"
            path.write_text("ply\nformat ascii 1.0\n"
                            "comment arbitrary monocular scale; segment-local first-camera world frame\n"
                            "element vertex 2\nproperty float x\nproperty float y\nproperty float z\n"
                            "end_header\n1.25 -2 3\n-4 5.5 6\n", encoding="utf-8")
            np.testing.assert_array_equal(m5c_data.read_ply(path), [[1.25, -2, 3], [-4, 5.5, 6]])
            path.write_text(path.read_text().replace("vertex 2", "vertex 3"), encoding="utf-8")
            with self.assertRaises(ValueError):
                m5c_data.read_ply(path)

    def test_declared_ranges_are_inclusive_original_ids_and_isolated(self):
        self.assertEqual(m5c_data.CLIPS, {4: (88, 122), 7: (252, 282)})
        self.assertEqual(len(range(88, 123)), 35)
        self.assertEqual(len(range(252, 283)), 31)
        self.assertFalse(set(range(88, 123)) & set(range(252, 283)))
        self.assertEqual(m5c_data.STORYBOARD, (121, 122, 203, 206, 207, 252))


@unittest.skipUnless((RUN / "per_frame.jsonl").exists() and OUTPUT.exists(), "KITTI M5-C artifacts unavailable")
class M5CRealEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = m5c_data.load_rows(RUN / "per_frame.jsonl")
        cls.capture = json.loads((OUTPUT / "capture/coordinate_capture.json").read_text())
        cls.media = json.loads((OUTPUT / "MEDIA_MANIFEST.json").read_text())

    def test_original_png_ids_accepted_gaps_and_independent_maps(self):
        inspected = m5c_data.validate_sources(ROOT)
        self.assertEqual(inspected["baseline"], {"valid_poses": 145, "segment_count": 9,
                                                  "longest_valid_span": 34, "ba_accepted": 0,
                                                  "ba_attempts": 10})
        for seg, (first, lost) in m5c_data.CLIPS.items():
            self.assertTrue(all(self.rows[i]["pose_valid"] and self.rows[i]["segment_id"] == seg
                                for i in range(first, lost)))
            self.assertFalse(self.rows[lost]["pose_valid"])
            self.assertIsNone(self.rows[lost]["camera_center_world"])
            self.assertTrue((ROOT / "data/sequences/00/image_0" / f"{first:06d}.png").is_file())
        self.assertFalse(self.rows[203]["pose_valid"])
        self.assertTrue(self.rows[206]["pose_valid"])
        self.assertFalse(self.rows[207]["pose_valid"])

    def test_capture_coordinates_and_candidate_support_match_frozen_counts(self):
        equivalence = json.loads((OUTPUT / "capture/replay_equivalence.json").read_text())
        self.assertTrue(equivalence["passed"])
        self.assertEqual(equivalence["maximum_absolute_pose_difference"], 0)
        self.assertTrue(all(entry["byte_equal"] for entry in equivalence["final_map_ply"].values()))
        for frame in m5c_data.CAPTURE_IDS:
            row, capture = self.rows[frame], self.capture[str(frame)]
            self.assertEqual(capture["frame_id"], frame)
            self.assertEqual(capture["pose_valid"], row["pose_valid"])
            if capture["provenance"].startswith("existing opt-in"):
                self.assertEqual(len(capture["associations"]), row["diagnostics"]["pnp_correspondences"])
                if row["diagnostics"].get("pnp_inliers") is not None:
                    self.assertEqual(len(capture["ransac_inlier_indices"]), row["diagnostics"]["pnp_inliers"])
                self.assertTrue(all(0 <= i < len(capture["associations"])
                                    for i in capture["ransac_inlier_indices"]))

    def test_final_maps_and_camera_coordinates_are_original(self):
        self.assertEqual(len(m5c_data.read_ply(RUN / "maps/segment-004.ply")), 670)
        self.assertEqual(len(m5c_data.read_ply(RUN / "maps/segment-007.ply")), 558)
        for frame in (88, 105, 121, 252, 267, 281):
            center, _ = m5c_data.pose_geometry(self.rows[frame])
            np.testing.assert_allclose(center, self.rows[frame]["camera_center_world"], atol=1e-9)

    def test_video_sequence_codec_and_final_frame_decoding(self):
        for video in self.media["videos"]:
            ids = video["source_ids"]
            self.assertEqual(ids, list(range(ids[0], ids[-1]+1)))
            self.assertEqual(len(ids), 35 if video["map_segment"] == 4 else 31)
            self.assertFalse(video["last_frame_pose_valid"])
            path = ROOT / video["path"]
            self.assertEqual(m5c_data.sha256(path), video["sha256"])
            reader = cv2.VideoCapture(str(path))
            self.assertTrue(reader.isOpened())
            fourcc = int(reader.get(cv2.CAP_PROP_FOURCC))
            actual_codec = "".join(chr((fourcc >> (8*i)) & 255) for i in range(4))
            self.assertEqual(actual_codec, video["codec"])
            self.assertEqual(video["requested_encoder"], "mp4v")
            decoded = 0
            last = None
            while True:
                ok, image = reader.read()
                if not ok:
                    break
                self.assertEqual(image.shape, (1080, 1920, 3))
                decoded += 1
                last = image
            reader.release()
            self.assertEqual(decoded, len(ids))
            self.assertIsNotNone(last)
            self.assertEqual(video["decoded_final_frame"], ids[-1])

    def test_every_encoded_image_matches_its_original_source_id(self):
        image_dir = ROOT / "data/sequences/00/image_0"
        for video in self.media["videos"]:
            reader = cv2.VideoCapture(str(ROOT / video["path"]))
            ids = video["source_ids"]
            for index, frame_id in enumerate(ids):
                ok, image = reader.read()
                self.assertTrue(ok, frame_id)
                encoded = cv2.cvtColor(image[80:644, 30:1890], cv2.COLOR_BGR2GRAY).astype(float)
                original = cv2.imread(str(image_dir / f"{frame_id:06d}.png"), cv2.IMREAD_GRAYSCALE)
                self.assertIsNotNone(original, frame_id)
                expected = cv2.resize(original, (1860, 564)).astype(float)
                mean_difference = float(np.abs(encoded-expected).mean())
                self.assertLess(mean_difference, 12, frame_id)
                for neighbor in (index-1, index+1):
                    if 0 <= neighbor < len(ids):
                        other = cv2.imread(str(image_dir / f"{ids[neighbor]:06d}.png"), cv2.IMREAD_GRAYSCALE)
                        other = cv2.resize(other, (1860, 564)).astype(float)
                        self.assertLess(mean_difference, float(np.abs(encoded-other).mean()), frame_id)
            reader.release()

    def test_figures_read_and_offline_truth_is_separate(self):
        for figure in self.media["figures"]:
            path = ROOT / figure["path"]
            self.assertEqual(m5c_data.sha256(path), figure["sha256"])
            self.assertIsNotNone(cv2.imread(str(path)))
        gt_path = OUTPUT / "figures/segment_004_gt_comparison.png"
        self.assertIsNotNone(cv2.imread(str(gt_path)))
        scored = m5c_offline_gt.comparison_data(ROOT)
        self.assertAlmostEqual(scored["in_sample_rmse"], .22503195165117487, places=10)
        self.assertAlmostEqual(scored["prefix_fit_suffix_rmse"], .61488583, places=6)
        self.assertEqual(scored["ids"].tolist(), list(range(88, 122)))
        self.assertEqual(len(scored["errors"]), 34)
        self.assertFalse(self.media["ground_truth_in_online_media"])

    def test_frozen_input_hashes_and_map_point_population(self):
        manifest = json.loads((OUTPUT / "capture/input_manifest.json").read_text())
        for name, digest in manifest["files_sha256"].items():
            self.assertEqual(m5c_data.sha256(ROOT / name), digest, name)
        self.assertEqual(manifest["map_point_counts"], {"4": 670, "7": 558})


if __name__ == "__main__":
    unittest.main()
