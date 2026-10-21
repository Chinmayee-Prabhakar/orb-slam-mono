"""Extract a byte-preserved KITTI odometry sequence-00 prefix from local ZIPs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
import zlib
from pathlib import Path
from typing import Optional, Tuple


PREFIX = "dataset/sequences/00/"


def _read_checked(archive: zipfile.ZipFile, name: str) -> Tuple[bytes, zipfile.ZipInfo]:
    info = archive.getinfo(name)
    content = archive.read(info)  # ZIP reader also checks CRC.
    if len(content) != info.file_size or zlib.crc32(content) != info.CRC:
        raise ValueError(f"ZIP CRC or size mismatch: {name}")
    return content, info


def _keep_exact(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing file differs from official ZIP: {path}")
    else:
        path.write_bytes(content)


def extract(gray_path: Path, calib_path: Optional[Path], target: Path,
            manifest_path: Path, frame_count: int) -> dict:
    if not 1 <= frame_count <= 4541:
        raise ValueError("frame-count must be between 1 and 4541 for sequence 00")
    records = []
    with zipfile.ZipFile(gray_path) as gray:
        names = set(gray.namelist())
        frame_pattern = re.compile(rf"{re.escape(PREFIX)}image_0/(\d{{6}})\.png")
        available = sorted(int(match.group(1)) for name in names
                           if (match := frame_pattern.fullmatch(name)))
        if available[:frame_count] != list(range(frame_count)):
            raise ValueError("Sequence-00 image_0 prefix is missing or discontinuous")
        calibration, calibration_info = _read_checked(gray, PREFIX + "calib.txt")
        times, times_info = _read_checked(gray, PREFIX + "times.txt")
        if calib_path is not None:
            with zipfile.ZipFile(calib_path) as separate:
                other_calibration, _ = _read_checked(separate, PREFIX + "calib.txt")
                other_times, _ = _read_checked(separate, PREFIX + "times.txt")
            def p0(data: bytes) -> str:
                return next(line for line in data.decode("ascii").splitlines()
                            if line.startswith("P0:"))
            if p0(calibration) != p0(other_calibration) or times != other_times:
                raise ValueError("Separate sequence-00 calibration/times disagree")
        timestamp_lines = times.splitlines(keepends=True)
        if len(timestamp_lines) < frame_count:
            raise ValueError("Too few sequence-00 timestamps")
        _keep_exact(target / "calib.txt", calibration)
        _keep_exact(target / "times.txt", b"".join(timestamp_lines[:frame_count]))
        for number in range(frame_count):
            name = PREFIX + f"image_0/{number:06d}.png"
            content, info = _read_checked(gray, name)
            _keep_exact(target / "image_0" / f"{number:06d}.png", content)
            records.append({"frame": number, "zip_crc32": f"{info.CRC:08x}",
                            "size_bytes": len(content),
                            "sha256": hashlib.sha256(content).hexdigest()})
    manifest = {"source": "user-provided official KITTI odometry grayscale ZIP",
                "gray_archive": str(gray_path),
                "calibration_archive": None if calib_path is None else str(calib_path),
                "sequence": "00", "camera": "image_0", "frame_count": frame_count,
                "calibration_crc32": f"{calibration_info.CRC:08x}",
                "times_full_crc32": f"{times_info.CRC:08x}",
                "images": records}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gray-archive", type=Path, required=True)
    parser.add_argument("--calib-archive", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/sequences/00"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("artifacts/kitti-00-validation/subset_manifest.json"))
    parser.add_argument("--frame-count", type=int, default=300)
    args = parser.parse_args()
    manifest = extract(args.gray_archive, args.calib_archive, args.output_dir,
                       args.manifest, args.frame_count)
    print(f"Verified {manifest['frame_count']} original image bytes and ZIP CRCs; "
          f"subset at {args.output_dir}")


if __name__ == "__main__":
    main()
