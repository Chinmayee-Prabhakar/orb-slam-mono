"""Small image sequence interface and KITTI odometry grayscale loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import cv2
import numpy as np

from .camera import CalibratedCamera, load_kitti_p0


@dataclass(frozen=True)
class ImageSample:
    index: int
    image: np.ndarray
    source: str
    timestamp_seconds: Optional[float] = None


class ImageSequence(Protocol):
    camera: CalibratedCamera

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> ImageSample: ...


class KittiGraySequence:
    """One KITTI odometry sequence directory containing image_0/ and calib.txt.

    Only camera images, optional times, and P0 calibration are read. Pose files
    are deliberately outside this loader's interface.
    """

    def __init__(self, sequence_dir: Path):
        self.sequence_dir = Path(sequence_dir)
        image_dir = self.sequence_dir / "image_0"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"KITTI grayscale image directory missing: {image_dir}")
        self.paths = sorted(image_dir.glob("*.png"))
        if not self.paths:
            raise ValueError(f"No PNG images found in {image_dir}")
        first = self._read(self.paths[0])
        height, width = first.shape
        self.camera = load_kitti_p0(self.sequence_dir / "calib.txt", width, height)
        times_path = self.sequence_dir / "times.txt"
        self.times = None
        if times_path.is_file():
            self.times = [float(line.strip()) for line in times_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(self.times) != len(self.paths):
                raise ValueError("KITTI times.txt line count must match image count")

    @staticmethod
    def _read(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read grayscale image: {path}")
        return image

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> ImageSample:
        if index < 0 or index >= len(self.paths):
            raise IndexError(index)
        path = self.paths[index]
        image = self._read(path)
        if image.shape != (self.camera.height, self.camera.width):
            raise ValueError(f"Image dimensions differ from calibration source: {path}")
        timestamp = None if self.times is None else self.times[index]
        return ImageSample(index, image, str(path), timestamp)
