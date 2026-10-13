"""Calibrated pinhole camera data; no pose or ground-truth input."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CalibratedCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        values = np.array([self.fx, self.fy, self.cx, self.cy], dtype=float)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera image dimensions must be positive")
        if not np.all(np.isfinite(values)) or self.fx <= 0 or self.fy <= 0:
            raise ValueError("Camera intrinsics must be finite with positive focal lengths")

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, width: int, height: int) -> "CalibratedCamera":
        k = np.asarray(matrix, dtype=float)
        if k.shape != (3, 3) or not np.all(np.isfinite(k)):
            raise ValueError("Camera matrix must be a finite 3x3 array")
        if not np.allclose(k[2], [0.0, 0.0, 1.0]) or not np.isclose(k[0, 1], 0.0):
            raise ValueError("Expected a normalized, zero-skew pinhole camera matrix")
        if not np.isclose(k[1, 0], 0.0):
            raise ValueError("Expected a zero-skew pinhole camera matrix")
        return cls(width, height, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]))


def load_kitti_p0(calibration_file: Path, width: int, height: int) -> CalibratedCamera:
    """Read rectified left grayscale camera intrinsics from KITTI odometry P0."""
    path = Path(calibration_file)
    entries = [line.split(":", 1) for line in path.read_text(encoding="utf-8").splitlines() if ":" in line]
    p0_values = [value for key, value in entries if key.strip() == "P0"]
    if len(p0_values) != 1:
        raise ValueError(f"Expected exactly one P0 line in {path}")
    tokens = p0_values[0].split()
    if len(tokens) != 12:
        raise ValueError(f"KITTI P0 must contain 12 finite numbers in {path}")
    try:
        numbers = np.array([float(token) for token in tokens])
    except ValueError as error:
        raise ValueError(f"KITTI P0 must contain 12 finite numbers in {path}") from error
    if not np.all(np.isfinite(numbers)):
        raise ValueError(f"KITTI P0 must contain 12 finite numbers in {path}")
    projection = numbers.reshape(3, 4)
    return CalibratedCamera.from_matrix(projection[:, :3], width, height)
