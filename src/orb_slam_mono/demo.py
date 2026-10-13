"""Save ORB feature and consecutive-frame descriptor match visualizations."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import cv2

from .datasets import ImageSequence, KittiGraySequence
from .diagnostics import PairDiagnostics
from .features import OrbExtractor
from .matching import DescriptorMatcher
from .synthetic import SyntheticSequence

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path(__file__).absolute().parents[2] / "configs" / "default.json"


def _write_image(path: Path, image: Any) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save visualization: {path}")


def run_demo(sequence: ImageSequence, first_index: int, output_dir: Path, config: Dict[str, Any], dataset_name: str) -> PairDiagnostics:
    if first_index < 0 or first_index + 1 >= len(sequence):
        raise IndexError("first-index must select two consecutive images")
    first = sequence[first_index]
    second = sequence[first_index + 1]
    extractor = OrbExtractor(
        nfeatures=config["orb"]["nfeatures"],
        scale_factor=config["orb"]["scaleFactor"],
        levels=config["orb"]["nlevels"],
        fast_threshold=config["orb"]["fastThreshold"],
    )
    matcher = DescriptorMatcher(**config["matching"])
    first_features = extractor.extract(first.image)
    second_features = extractor.extract(second.image)
    result = matcher.match(first_features.descriptors, second_features.descriptors)
    diagnostics = PairDiagnostics(
        dataset=dataset_name,
        camera={
            "width": sequence.camera.width,
            "height": sequence.camera.height,
            "fx": sequence.camera.fx,
            "fy": sequence.camera.fy,
            "cx": sequence.camera.cx,
            "cy": sequence.camera.cy,
        },
        first_source=first.source,
        second_source=second.source,
        first_index=first.index,
        second_index=second.index,
        first_features=first_features.count,
        second_features=second_features.count,
        knn_pairs=result.knn_pairs,
        ratio_passed=result.ratio_passed,
        retained_matches=result.retained,
        ratio_threshold=matcher.ratio_threshold,
        mutual_check=matcher.mutual_check,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample, features in ((first, first_features), (second, second_features)):
        drawn = cv2.drawKeypoints(sample.image, features.keypoints, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        _write_image(output_dir / f"features_{sample.index:06d}.png", drawn)
    max_drawn = int(config["visualization"]["max_drawn_matches"])
    if max_drawn < 0:
        raise ValueError("max_drawn_matches must be nonnegative")
    match_image = cv2.drawMatches(
        first.image, first_features.keypoints,
        second.image, second_features.keypoints,
        result.matches[:max_drawn], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    _write_image(output_dir / "matches.png", match_image)
    (output_dir / "stats.json").write_text(json.dumps(diagnostics.as_dict(), indent=2) + "\n", encoding="utf-8")
    LOGGER.info("features=%d/%d ratio_passed=%d retained=%d", diagnostics.first_features,
                diagnostics.second_features, diagnostics.ratio_passed, diagnostics.retained_matches)
    return diagnostics


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitti-sequence", type=Path, help="KITTI odometry sequence directory, e.g. data/sequences/00")
    parser.add_argument("--first-index", type=int, default=0, help="first of two consecutive frames (default: 0)")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/demo"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sequence: ImageSequence = (KittiGraySequence(args.kitti_sequence) if args.kitti_sequence
                               else SyntheticSequence())
    dataset_name = "kitti_odometry_gray" if args.kitti_sequence else "synthetic"
    diagnostics = run_demo(sequence, args.first_index, args.output_dir, config, dataset_name)
    print(json.dumps(diagnostics.as_dict(), indent=2))
    print(f"Saved visualizations and stats to {args.output_dir.absolute()}")


if __name__ == "__main__":
    main()
