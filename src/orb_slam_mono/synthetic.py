"""Deterministic textured pair used when a KITTI sequence is unavailable."""

from __future__ import annotations

import cv2
import numpy as np

from .camera import CalibratedCamera
from .datasets import ImageSample


class SyntheticSequence:
    def __init__(self, width: int = 640, height: int = 480, dx: int = 9, dy: int = 5):
        self.camera = CalibratedCamera(width, height, 520.0, 520.0, width / 2, height / 2)
        self.dx = dx
        self.dy = dy
        rng = np.random.default_rng(23)
        image = np.full((height, width), 24, dtype=np.uint8)
        for _ in range(300):
            x = int(rng.integers(20, width - 20))
            y = int(rng.integers(20, height - 20))
            shade = int(rng.integers(90, 255))
            size = int(rng.integers(2, 8))
            cv2.rectangle(image, (x - size, y - size), (x + size, y + size), shade, -1)
            cv2.line(image, (x - size, y), (x + size, y), 255 - shade, 1)
        cv2.putText(image, "ORB MONO TEST", (115, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 220, 2, cv2.LINE_AA)
        self.first = image
        transform = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
        self.second = cv2.warpAffine(image, transform, (width, height), borderValue=24)

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> ImageSample:
        if index not in (0, 1):
            raise IndexError(index)
        return ImageSample(index, self.first.copy() if index == 0 else self.second.copy(), f"synthetic:{index}")
