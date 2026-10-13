"""OpenCV ORB extraction on uint8 grayscale images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class FeatureSet:
    keypoints: List[cv2.KeyPoint]
    descriptors: Optional[np.ndarray]

    @property
    def count(self) -> int:
        return len(self.keypoints)


class OrbExtractor:
    def __init__(self, nfeatures: int = 1500, scale_factor: float = 1.2, levels: int = 8, fast_threshold: int = 20):
        if nfeatures <= 0 or scale_factor <= 1 or levels <= 0 or fast_threshold < 0:
            raise ValueError("Invalid ORB extractor configuration")
        self._orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=scale_factor,
            nlevels=levels,
            fastThreshold=fast_threshold,
        )

    def extract(self, image: np.ndarray) -> FeatureSet:
        if image.ndim != 2 or image.dtype != np.uint8:
            raise ValueError("ORB input must be a uint8 grayscale image")
        keypoints, descriptors = self._orb.detectAndCompute(image, None)
        return FeatureSet(keypoints or [], descriptors)
