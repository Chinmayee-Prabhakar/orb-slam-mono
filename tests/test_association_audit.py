"""Known-identity synthetic controls for the offline M4.5-B1 audit."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import cv2
import numpy as np

from orb_slam_mono.association_audit import (association_metrics, estimate_track_identity,
                                             label_association, projected_associations, run_audit)


K = np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])
POSE = {"segment_id": 2, "T_cw_R": np.eye(3).tolist(), "T_cw_t": [0., 0., 0.]}


def point(point_id: int, descriptor: np.ndarray, xyz=(0., 0., 10.),
          segment_id: int = 2, observations=()) -> dict:
    return {"point_id": point_id, "segment_id": segment_id, "xyz_world": list(xyz),
            "descriptor": descriptor.tolist(), "observations": list(observations)}


def textured_frames() -> tuple[dict[int, np.ndarray], np.ndarray, list[dict]]:
    rng = np.random.default_rng(9)
    patch = rng.integers(40, 240, (25, 25), dtype=np.uint8)
    images = {}
    observations = []
    for frame_id, center_x in ((0, 49), (1, 52), (2, 55)):
        image = np.full((120, 120), 20, dtype=np.uint8)
        image[48:73, center_x-12:center_x+13] = patch
        images[frame_id] = image
        if frame_id < 2:
            observations.append({"frame_id": frame_id, "feature_id": 10+frame_id,
                                 "pixel_xy": [float(center_x), 60.]})
    return {0: images[0], 1: images[1]}, images[2], observations


class AssociationAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.zero = np.zeros(32, dtype=np.uint8)
        self.one = np.full(32, 255, dtype=np.uint8)

    def test_repeated_texture_tie_rejects_ratio(self) -> None:
        points = [point(1, self.zero)]
        xy = np.array([[50., 50.], [52., 50.]])
        descriptors = np.stack((self.zero, self.zero))
        result = projected_associations(points, xy, descriptors, POSE, K, (100, 100), 8)
        self.assertEqual(result["forward_complete_pairs"], 1)
        self.assertEqual(result["ratio_passed"], 0)
        self.assertEqual(result["mutual_passed"], 0)

    def test_known_identity_unique_and_competing_points(self) -> None:
        nearby_descriptor = self.zero.copy()
        nearby_descriptor[0] = 1
        points = [point(1, self.zero), point(2, nearby_descriptor)]
        xy = np.array([[50., 50.], [52., 50.]])
        descriptors = np.stack((self.zero, self.one))
        result = projected_associations(points, xy, descriptors, POSE, K, (100, 100), 8)
        self.assertEqual(result["ratio_passed"], 2)
        self.assertEqual(result["mutual_passed"], 1)
        self.assertEqual((result["retained"][0]["point_id"], result["retained"][0]["feature_id"]), (1, 0))

    def test_wrong_prior_and_large_jump_require_fallback(self) -> None:
        points = [point(1, self.zero)]
        xy = np.array([[80., 50.], [82., 50.]])
        descriptors = np.stack((self.zero, self.one))
        small = projected_associations(points, xy, descriptors, POSE, K, (100, 100), 16)
        large = projected_associations(points, xy, descriptors, POSE, K, (100, 100), 32)
        self.assertEqual(small["mutual_passed"], 0)
        self.assertEqual(large["mutual_passed"], 1)
        wrong_pose = {**POSE, "T_cw_t": [5., 0., 0.]}
        wrong = projected_associations(points, xy, descriptors, wrong_pose, K, (100, 100), 32)
        self.assertEqual(wrong["mutual_passed"], 0)

    def test_occlusion_and_behind_camera_do_not_create_pairs(self) -> None:
        points = [point(1, self.zero), point(2, self.zero, xyz=(0., 0., -10.))]
        xy = np.array([[80., 80.], [82., 80.]])
        result = projected_associations(points, xy, np.stack((self.zero, self.one)),
                                        POSE, K, (100, 100), 8)
        self.assertEqual(result["projected_behind_or_invalid"], 1)
        self.assertEqual(result["mutual_passed"], 0)

    def test_independent_segments_cannot_be_mixed(self) -> None:
        points = [point(1, self.zero), point(2, self.one, segment_id=3)]
        xy = np.array([[50., 50.], [52., 50.]])
        with self.assertRaises(ValueError):
            projected_associations(points, xy, np.stack((self.zero, self.one)),
                                   POSE, K, (100, 100), 8)

    def test_two_image_tracks_support_known_identity_and_reject_wrong_feature(self) -> None:
        sources, target, observations = textured_frames()
        points = [point(1, self.zero, observations=observations)]
        xy = np.array([[55., 60.], [90., 90.]])
        claims = estimate_track_identity(points, sources, target, xy, 2)
        self.assertEqual(claims[1]["status"], "supported")
        self.assertEqual(claims[1]["supported_feature_id"], 0)
        self.assertEqual(label_association(1, 0, claims, xy), "supported_correct")
        self.assertEqual(label_association(1, 1, claims, xy), "supported_incorrect")
        metrics = association_metrics([{"point_id": 1, "feature_id": 0}], claims, xy, 120, 120)
        self.assertEqual(metrics["labeled_precision"], 1.0)
        self.assertEqual(metrics["partial_recall"], 1.0)

    def test_ambiguous_feature_claims_remain_uncertain(self) -> None:
        sources, target, observations = textured_frames()
        points = [point(1, self.zero, observations=observations),
                  point(2, self.one, observations=observations)]
        xy = np.array([[55., 60.], [90., 90.]])
        claims = estimate_track_identity(points, sources, target, xy, 2)
        self.assertEqual(claims[1]["status"], "uncertain")
        self.assertEqual(claims[2]["status"], "uncertain")
        self.assertEqual(label_association(1, 0, claims, xy), "uncertain")
        points = [point(1, self.zero, observations=[])]
        claims = estimate_track_identity(points, sources, target, xy, 2)
        self.assertEqual(label_association(1, 0, claims, xy), "not_evaluable")

    def test_offline_analysis_does_not_mutate_inputs_or_import_truth_into_online_code(self) -> None:
        sources, target, observations = textured_frames()
        points = [point(1, self.zero, observations=observations)]
        xy = np.array([[55., 60.], [57., 60.]])
        descriptors = np.stack((self.zero, self.one))
        point_copy, xy_copy, descriptors_copy = copy.deepcopy(points), xy.copy(), descriptors.copy()
        images_copy = {key: image.copy() for key, image in sources.items()}
        target_copy = target.copy()
        projected_associations(points, xy, descriptors, POSE, K, target.shape, 8)
        estimate_track_identity(points, sources, target, xy, 2)
        self.assertEqual(points, point_copy)
        np.testing.assert_array_equal(xy, xy_copy)
        np.testing.assert_array_equal(descriptors, descriptors_copy)
        np.testing.assert_array_equal(target, target_copy)
        for key in sources:
            np.testing.assert_array_equal(sources[key], images_copy[key])
        root = Path(__file__).resolve().parents[1]
        for name in ("matching.py", "tracking.py", "map_state.py", "local_mapping.py"):
            content = (root / "src" / "orb_slam_mono" / name).read_text(encoding="utf-8")
            self.assertNotIn("association_audit", content)
            self.assertNotIn("ground_truth", content)

    def test_audit_output_is_scoped_and_existing_evidence_cannot_be_overwritten(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with self.assertRaises(ValueError):
            run_audit(root, root / "artifacts" / "other-study")
        with self.assertRaises(FileExistsError):
            run_audit(root, root / "artifacts" / "m45b1-association-audit")


if __name__ == "__main__":
    unittest.main()
