"""Descriptor-only matching and filtering; no geometric estimation yet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class MatchResult:
    matches: List[cv2.DMatch]
    knn_pairs: int
    ratio_passed: int

    @property
    def retained(self) -> int:
        return len(self.matches)


class DescriptorMatcher:
    def __init__(self, ratio_threshold: float = 0.75, mutual_check: bool = True):
        if not 0 < ratio_threshold < 1:
            raise ValueError("ratio_threshold must be between 0 and 1")
        self.ratio_threshold = ratio_threshold
        self.mutual_check = mutual_check
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    @staticmethod
    def _validate(descriptors: Optional[np.ndarray]) -> bool:
        if descriptors is None or len(descriptors) == 0:
            return False
        if descriptors.ndim != 2 or descriptors.dtype != np.uint8 or descriptors.shape[1] != 32:
            raise ValueError("Expected ORB descriptors as a uint8 Nx32 array")
        return True

    def match(self, first: Optional[np.ndarray], second: Optional[np.ndarray],
              *, diagnostic: Optional[dict[str, Any]] = None) -> MatchResult:
        if not self._validate(first) or not self._validate(second):
            if diagnostic is not None:
                diagnostic.update(forward_complete_pairs=0, ratio_passed=0,
                                  mutual_passed=0, candidates=[])
            return MatchResult([], 0, 0)
        assert first is not None and second is not None
        forward = self._matcher.knnMatch(first, second, k=2)
        complete_pairs = [pair for pair in forward if len(pair) == 2]
        ratio_matches = [pair[0] for pair in complete_pairs
                         if pair[0].distance < self.ratio_threshold * pair[1].distance]
        ratio_passed = len(ratio_matches)
        reverse_best = None
        if self.mutual_check:
            reverse = self._matcher.knnMatch(second, first, k=1)
            reverse_best = {pair[0].queryIdx: pair[0].trainIdx for pair in reverse if pair}
            ratio_matches = [match for match in ratio_matches if reverse_best.get(match.trainIdx) == match.queryIdx]
        ratio_matches.sort(key=lambda match: (match.distance, match.queryIdx, match.trainIdx))
        if diagnostic is not None:
            diagnostic.update(
                forward_complete_pairs=len(complete_pairs), ratio_passed=ratio_passed,
                mutual_passed=len(ratio_matches),
                candidates=[{"map_descriptor_index": int(pair[0].queryIdx),
                             "image_feature_index": int(pair[0].trainIdx),
                             "best_hamming": float(pair[0].distance),
                             "second_hamming": float(pair[1].distance),
                             "ratio_pass": bool(pair[0].distance < self.ratio_threshold * pair[1].distance),
                             "mutual_pass": bool(
                                 pair[0].distance < self.ratio_threshold * pair[1].distance and
                                 (reverse_best is None or
                                  reverse_best.get(pair[0].trainIdx) == pair[0].queryIdx))}
                            for pair in complete_pairs])
        return MatchResult(ratio_matches, len(complete_pairs), ratio_passed)
