"""Small, identity-preserving sparse map for one arbitrary-scale T_cw segment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from .datasets import ImageSample
from .features import FeatureSet
from .geometry import InitializationResult


class TrackingState(str, Enum):
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    LOST = "lost"


@dataclass
class Frame:
    frame_id: int
    timestamp_seconds: float | None
    features: FeatureSet
    state: TrackingState = TrackingState.INITIALIZING
    R_cw: np.ndarray | None = None
    t_cw: np.ndarray | None = None
    failure_reason: str | None = None
    feature_to_point: dict[int, int] = field(default_factory=dict)
    pnp_inlier_features: set[int] = field(default_factory=set)

    @classmethod
    def from_sample(cls, sample: ImageSample, features: FeatureSet) -> "Frame":
        return cls(sample.index, sample.timestamp_seconds, features)

    @property
    def camera_center_world(self) -> np.ndarray | None:
        if self.R_cw is None or self.t_cw is None:
            return None
        return -self.R_cw.T @ self.t_cw


@dataclass
class MapPoint:
    point_id: int
    xyz_world: np.ndarray
    descriptor: np.ndarray
    observations: dict[int, int] = field(default_factory=dict)
    valid: bool = True

    def __post_init__(self) -> None:
        self.xyz_world = np.asarray(self.xyz_world, dtype=np.float64).reshape(3)
        self.descriptor = np.asarray(self.descriptor)
        if not np.all(np.isfinite(self.xyz_world)) or self.descriptor.shape != (32,) or self.descriptor.dtype != np.uint8:
            raise ValueError("Map point requires finite xyz and one uint8 ORB descriptor")

    @property
    def observation_count(self) -> int:
        return len(self.observations)


@dataclass
class SparseMap:
    segment_id: int
    anchor_frame_id: int
    second_frame_id: int
    frames: dict[int, Frame] = field(default_factory=dict)
    points: dict[int, MapPoint] = field(default_factory=dict)
    coordinate_convention: str = "T_cw; first camera is world origin; arbitrary unit baseline"

    def add_observation(self, point_id: int, frame: Frame, feature_index: int) -> None:
        if point_id not in self.points or frame.frame_id not in self.frames or self.frames[frame.frame_id] is not frame:
            raise ValueError("Observation must refer to this map's point and frame")
        point = self.points[point_id]
        if frame.frame_id in point.observations or feature_index in frame.feature_to_point:
            raise ValueError("Duplicate point/frame or feature/point observation")
        if not 0 <= feature_index < frame.features.count:
            raise ValueError("Feature index outside frame")
        point.observations[frame.frame_id] = feature_index
        frame.feature_to_point[feature_index] = point_id

    def validate(self) -> None:
        for point_id, point in self.points.items():
            if not point.valid or not np.all(np.isfinite(point.xyz_world)):
                raise ValueError("Invalid active map point")
            for frame_id, feature_index in point.observations.items():
                if frame_id not in self.frames or self.frames[frame_id].feature_to_point.get(feature_index) != point_id:
                    raise ValueError("Broken point-to-frame reference")
        for frame_id, frame in self.frames.items():
            for feature_index, point_id in frame.feature_to_point.items():
                if point_id not in self.points or self.points[point_id].observations.get(frame_id) != feature_index:
                    raise ValueError("Broken frame-to-point reference")


def map_from_initialization(segment_id: int, anchor: Frame, second: Frame,
                            matches: list[cv2.DMatch], result: InitializationResult,
                            first_point_id: int = 0) -> SparseMap:
    """Translate accepted-mask positions through original DMatch identities."""
    if result.status != "accepted" or result.R_21 is None or result.t_21 is None:
        raise ValueError("Only fully accepted two-view geometry can create a map")
    indices = np.flatnonzero(result.accepted_mask)
    if len(result.accepted_mask) != len(matches) or len(indices) != len(result.points_xyz):
        raise ValueError("Initialization mask, matches, and xyz are misaligned")
    if anchor.frame_id == second.frame_id or anchor.features.descriptors is None:
        raise ValueError("Distinct frames with anchor descriptors are required")
    sparse_map = SparseMap(segment_id, anchor.frame_id, second.frame_id,
                           {anchor.frame_id: anchor, second.frame_id: second})
    used_first: set[int] = set()
    used_second: set[int] = set()
    for row, match_index in enumerate(indices):
        match = matches[int(match_index)]
        query, train = int(match.queryIdx), int(match.trainIdx)
        if (query in used_first or train in used_second or query < 0 or train < 0
                or query >= anchor.features.count or train >= second.features.count):
            raise ValueError("Duplicate or invalid initialization feature identity")
        used_first.add(query)
        used_second.add(train)
        point_id = first_point_id + row
        sparse_map.points[point_id] = MapPoint(point_id, result.points_xyz[row],
                                                anchor.features.descriptors[query].copy())
        sparse_map.add_observation(point_id, anchor, query)
        sparse_map.add_observation(point_id, second, train)
    anchor.R_cw, anchor.t_cw = np.eye(3), np.zeros(3)
    second.R_cw, second.t_cw = result.R_21.copy(), result.t_21.copy()
    anchor.state = second.state = TrackingState.TRACKING
    sparse_map.validate()
    return sparse_map
