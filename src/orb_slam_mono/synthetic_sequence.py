"""Physical multi-view rendered landmarks; truth is for offline assertions only."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import CalibratedCamera
from .datasets import ImageSample


@dataclass(frozen=True)
class SyntheticSequence:
    camera: CalibratedCamera
    samples: list[ImageSample]
    truth_R_cw: list[np.ndarray]
    truth_t_cw: list[np.ndarray]
    world_points: np.ndarray
    metric_step: float

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ImageSample:
        return self.samples[index]


def generate_sequence(seed: int = 19, count: int = 12, metric_step: float = 0.22,
                      blank_frames: tuple[int, ...] = (),
                      vary_visibility: bool = False) -> SyntheticSequence:
    if count < 5 or metric_step < 0:
        raise ValueError("Require at least five frames and nonnegative baseline")
    rng = np.random.default_rng(seed)
    camera = CalibratedCamera(960, 600, 650.0, 650.0, 480.0, 300.0)
    world = np.column_stack((rng.uniform(-7.0, 11.0, 2600) if vary_visibility else rng.uniform(-4.0, 5.0, 1800),
                             rng.uniform(-2.5, 2.5, 2600 if vary_visibility else 1800),
                             rng.uniform(5.0, 13.0, 2600 if vary_visibility else 1800)))
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    projections: list[np.ndarray] = []
    for index in range(count):
        rotation, _ = cv2.Rodrigues(np.array([0.001 * index, 0.006 * index, 0.0]))
        center = metric_step * np.array([float(index), 0.04 * index, 0.015 * index])
        translation = -rotation @ center
        xyz = (rotation @ world.T).T + translation
        uv = np.column_stack((camera.fx * xyz[:, 0] / xyz[:, 2] + camera.cx,
                              camera.fy * xyz[:, 1] / xyz[:, 2] + camera.cy))
        rotations.append(rotation)
        translations.append(translation)
        projections.append(uv)
    per_view_visible = [((uv[:, 0] >= 18) & (uv[:, 0] < camera.width - 18)
                         & (uv[:, 1] >= 18) & (uv[:, 1] < camera.height - 18))
                        for uv in projections]
    visible = (np.sum(per_view_visible, axis=0) >= 3 if vary_visibility
               else np.logical_and.reduce(per_view_visible))
    candidates = np.flatnonzero(visible)
    rng.shuffle(candidates)
    selected: list[int] = []
    for candidate in candidates:
        if all(all(not (per_view_visible[view][candidate] and per_view_visible[view][other])
                   or np.linalg.norm(projections[view][candidate] - projections[view][other]) >= 24
                   for other in selected) for view in range(count)):
            selected.append(int(candidate))
        if len(selected) == (400 if vary_visibility else 270):
            break
    images = [np.full((camera.height, camera.width), 20, dtype=np.uint8) for _ in range(count)]
    for landmark in selected:
        patch = rng.integers(35, 240, (19, 19), dtype=np.uint8)
        for view, image in enumerate(images):
            if not per_view_visible[view][landmark]:
                continue
            x, y = np.rint(projections[view][landmark]).astype(int)
            image[y - 9:y + 10, x - 9:x + 10] = patch
    samples = [ImageSample(index, np.full_like(image, 20) if index in blank_frames else image,
                           f"synthetic_frame_{index:06d}", index * 0.1)
               for index, image in enumerate(images)]
    return SyntheticSequence(camera, samples, rotations, translations,
                             world[np.array(selected, dtype=int)], metric_step)
