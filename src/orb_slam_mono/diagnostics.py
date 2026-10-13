"""Machine-readable counts and explicit placeholders for future geometry stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PairDiagnostics:
    dataset: str
    camera: Dict[str, float]
    first_source: str
    second_source: str
    first_index: int
    second_index: int
    first_features: int
    second_features: int
    knn_pairs: int
    ratio_passed: int
    retained_matches: int
    ratio_threshold: float
    mutual_check: bool
    geometric_inliers: Optional[int] = None
    pnp_inliers: Optional[int] = None
    mean_reprojection_error_px: Optional[float] = None
    tracking_state: str = "not_estimated"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
