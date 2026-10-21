"""Extract only sequence-00 KITTI training poses into ignored offline data."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
import zlib
from pathlib import Path

from orb_slam_mono.kitti_pose_evaluation import load_kitti_poses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("data/ground_truth/kitti_odometry/00.txt"))
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        info = archive.getinfo("dataset/poses/00.txt")
        content = archive.read(info)
    if len(content) != info.file_size or zlib.crc32(content) != info.CRC:
        raise ValueError("Sequence-00 ZIP member failed CRC/size verification")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.output.read_bytes() != content:
            raise ValueError("Existing offline pose file differs from supplied ZIP")
    else:
        args.output.write_bytes(content)
    poses = load_kitti_poses(args.output)
    print(f"Verified {len(poses)} finite sequence-00 T_wc rows; "
          f"ZIP CRC-32 {info.CRC:08x}; SHA-256 {hashlib.sha256(content).hexdigest()}; "
          f"saved {args.output}")


if __name__ == "__main__":
    main()
