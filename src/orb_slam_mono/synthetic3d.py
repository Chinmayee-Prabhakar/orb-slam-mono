"""Deterministic physical 3D landmark projections for geometry verification."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import CalibratedCamera


@dataclass(frozen=True)
class SyntheticScene:
    camera: CalibratedCamera
    points_world: np.ndarray
    points1_px: np.ndarray
    points2_px: np.ndarray
    R_21_truth: np.ndarray
    t_21_metric_truth: np.ndarray
    true_match_mask: np.ndarray
    seed: int
    noise_px: float
    baseline: float


def _project(points: np.ndarray, camera: CalibratedCamera) -> np.ndarray:
    return np.column_stack((camera.fx * points[:, 0] / points[:, 2] + camera.cx,
                            camera.fy * points[:, 1] / points[:, 2] + camera.cy))


def generate_scene(seed: int = 7, n_landmarks: int = 450, noise_px: float = 0.35,
                   outlier_fraction: float = 0.0, baseline: float = 0.8,
                   rotation_degrees: float = 3.0) -> SyntheticScene:
    """Generate noncoplanar points and noisy, jointly visible image observations.

    Truth is returned for test/offline evaluation. It is never passed to the
    initializer, which receives only points1_px, points2_px, and camera.
    """
    if n_landmarks < 200 or noise_px < 0 or not 0 <= outlier_fraction < 1 or baseline < 0:
        raise ValueError("Invalid synthetic scene parameters")
    rng = np.random.default_rng(seed)
    camera = CalibratedCamera(1280, 720, 700.0, 700.0, 640.0, 360.0)
    xyz = np.column_stack((rng.uniform(-3.0, 3.0, n_landmarks),
                           rng.uniform(-1.6, 1.6, n_landmarks),
                           rng.uniform(4.0, 12.0, n_landmarks)))
    rotation, _ = cv2.Rodrigues(np.deg2rad(np.array([0.2, rotation_degrees, -0.3])))
    center2 = np.array([baseline, 0.04 * baseline, 0.02 * baseline])
    translation = -rotation @ center2
    xyz2 = (rotation @ xyz.T).T + translation
    image1 = _project(xyz, camera)
    image2 = _project(xyz2, camera)
    visible = ((xyz2[:, 2] > 0) &
               (image1[:, 0] >= 24) & (image1[:, 0] < camera.width - 24) &
               (image1[:, 1] >= 24) & (image1[:, 1] < camera.height - 24) &
               (image2[:, 0] >= 24) & (image2[:, 0] < camera.width - 24) &
               (image2[:, 1] >= 24) & (image2[:, 1] < camera.height - 24))
    xyz, image1, image2 = xyz[visible], image1[visible], image2[visible]
    image1 = image1 + rng.normal(0.0, noise_px, image1.shape)
    image2 = image2 + rng.normal(0.0, noise_px, image2.shape)
    true_mask = np.ones(len(image1), dtype=bool)
    wrong_count = int(round(outlier_fraction * len(image1)))
    if wrong_count:
        wrong_indices = rng.choice(len(image1), wrong_count, replace=False)
        image2[wrong_indices] = np.column_stack((
            rng.uniform(24, camera.width - 24, wrong_count),
            rng.uniform(24, camera.height - 24, wrong_count)))
        true_mask[wrong_indices] = False
    return SyntheticScene(camera, xyz, image1, image2, rotation, translation,
                          true_mask, seed, noise_px, baseline)


def render_projected_landmarks(scene: SyntheticScene) -> tuple[np.ndarray, np.ndarray]:
    """Render textured sprites at the two 3D landmark projections for ORB tests.

    The landmark centers obey 3D projection. Sprite appearance is simplified;
    quantitative geometry tests use the separate exact point observations.
    """
    rng = np.random.default_rng(scene.seed + 991)
    images = [np.full((scene.camera.height, scene.camera.width), 18, np.uint8)
              for _ in range(2)]
    # Use truth projection, not outlier-corrupted observations, to make images.
    clean_first = _project(scene.points_world, scene.camera)
    clean_second = _project((scene.R_21_truth @ scene.points_world.T).T
                            + scene.t_21_metric_truth, scene.camera)
    used_first: list[np.ndarray] = []
    used_second: list[np.ndarray] = []
    for first_point, second_point in zip(clean_first, clean_second):
        if (any(np.linalg.norm(first_point - point) < 31 for point in used_first)
                or any(np.linalg.norm(second_point - point) < 31 for point in used_second)):
            continue
        used_first.append(first_point)
        used_second.append(second_point)
        patch = rng.integers(35, 235, (25, 25), dtype=np.uint8)
        for image, point in zip(images, (first_point, second_point)):
            x, y = np.rint(point).astype(int)
            image[y - 12:y + 13, x - 12:x + 13] = patch
    return images[0], images[1]
