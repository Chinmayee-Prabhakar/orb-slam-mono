"""Build and verify the self-contained M5-B offline review ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


RUN_NAMES = ("run-a1", "run-a2", "run-b1", "run-b2", "run-c1", "run-c2")
RUN_FILES = ("per_frame.jsonl", "per_frame.csv", "summary.json",
             "initialization_attempts.json", "loss_events.json", "recovery_events.json",
             "keyframe_events.json", "map_growth_events.json", "ba_events.json",
             "estimated_poses.json", "final_keyframes.json", "resource_profile.json",
             "memory_samples.json")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_manifest(root: Path) -> dict[str, Any]:
    inputs = [root / name for name in ("configs/default.json", "data/sequences/00/calib.txt",
              "data/sequences/00/times.txt", "data/ground_truth/kitti_odometry/00.txt",
              "artifacts/m4-kitti-00-ba-validation-final/full/per_frame.jsonl",
              "artifacts/m45c-ba-depth-correction/corrected_full/per_frame.jsonl",
              "artifacts/m5a-frozen-evaluation/full/segments.json",
              "artifacts/m5a-frozen-evaluation/full/per_frame_errors.json",
              "artifacts/m5a-frozen-evaluation/full/relative_pairs.json")]
    images = sorted((root / "data/sequences/00/image_0").glob("*.png"))
    if [p.stem for p in images] != [f"{i:06d}" for i in range(300)]:
        raise ValueError("Expected the approved 300 KITTI source images")
    rows = {p.relative_to(root).as_posix(): sha(p) for p in inputs + images}
    image_entries = [(p.name, rows[p.relative_to(root).as_posix()]) for p in images]
    return {"method": "SHA-256 of approved input bytes; source-image names retain original IDs",
            "image_count": 300,
            "image_manifest_sha256": hashlib.sha256(json.dumps(image_entries,
                                                            separators=(",", ":")).encode()).hexdigest(),
            "files": rows}


def git_state(root: Path) -> str:
    sections = []
    for name, args in (("status --short", ("status", "--short")),
                       ("diff --stat", ("diff", "--stat")),
                       ("log -1 --format=fuller", ("log", "-1", "--format=fuller")),
                       ("remote -v", ("remote", "-v"))):
        process = subprocess.run(("git",) + args, cwd=root, check=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        sections += [f"## git {name}", process.stdout or "(none)"]
    return "\n".join(sections)


def selected_files(root: Path, base: Path) -> list[Path]:
    included: set[Path] = set()
    for relative in ("PROJECT_PLAN.md", "STATUS.md", "NOTES.md", "README.md",
                     "requirements.txt", "environment.yml", "configs/default.json",
                     "data/sequences/00/calib.txt", "data/sequences/00/times.txt",
                     "data/ground_truth/kitti_odometry/00.txt"):
        included.add(root / relative)
    included.update((root / "src/orb_slam_mono").glob("*.py"))
    included.update((root / "tests").glob("test_*.py"))
    for name in RUN_NAMES:
        folder = base / name
        included.update(folder / filename for filename in RUN_FILES)
        included.update((folder / "maps").glob("*.ply"))
    included.update((base / "evaluation").rglob("*.json"))
    included.update((base / "figures").glob("*.png"))
    for filename in ("REPORT.md", "REPRODUCE.txt", "FIGURE_GUIDE.md",
                     "FIGURE_GUIDE.json", "full_test_log.txt", "input_manifest.json",
                     "protected_evidence_check.json", "git_state.txt", "RUN_COMMANDS.txt",
                     "M5A_METRICS_REFERENCE.md", "offline_evaluation_timing.json",
                     "provenance.json"):
        included.add(base / filename)
    paths = sorted(included)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing review inputs: " + ", ".join(missing))
    if any("image_0" in p.parts for p in paths):
        raise ValueError("Raw KITTI images must not enter review ZIP")
    return paths


def verify_package(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as package:
        manifest = json.loads(package.read("PACKAGE_MANIFEST.json"))
        expected = {item["path"] for item in manifest["entries"]} | {"PACKAGE_MANIFEST.json"}
        actual = set(package.namelist())
        if expected != actual:
            raise ValueError("ZIP entry list differs from its manifest")
        for item in manifest["entries"]:
            data = package.read(item["path"])
            if len(data) != item["size_bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"ZIP content hash mismatch: {item['path']}")
    return {"verified": True, "entries": len(expected), "bytes": path.stat().st_size,
            "zip_sha256": sha(path)}


def build(root: Path, base: Path) -> dict[str, Any]:
    archive = base / "M5B-independent-review.zip"
    if archive.exists():
        raise FileExistsError(archive)
    manifest_path = base / "input_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    manifest_path.write_text(json.dumps(input_manifest(root), indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    verifier = root / "artifacts/m45d-continuity-recovery/verify_protected.py"
    process = subprocess.run((sys.executable, str(verifier), "verify"), cwd=root,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    (base / "protected_evidence_check.json").write_text(process.stdout, encoding="utf-8")
    (base / "git_state.txt").write_text(git_state(root), encoding="utf-8")
    (base / "M5A_METRICS_REFERENCE.md").write_bytes(
        (root / "artifacts/m5a-frozen-evaluation/METRICS.md").read_bytes())
    paths = selected_files(root, base)
    entries = [{"path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size, "sha256": sha(path)} for path in paths]
    package_manifest = {"format": "M5-B independent offline review",
                        "paths": "repository-relative", "entries": entries,
                        "exclusions": "raw KITTI images, Python environment, caches, old review packages, run visualization PNGs"}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as package:
        for path, entry in zip(paths, entries):
            package.write(path, entry["path"])
        package.writestr("PACKAGE_MANIFEST.json", json.dumps(package_manifest, indent=2) + "\n")
    verification = verify_package(archive)
    (base / "PACKAGE_VERIFICATION.json").write_text(json.dumps(verification, indent=2) + "\n",
                                                     encoding="utf-8")
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--base", type=Path, default=Path("artifacts/m5b-final-evaluation"))
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_package(args.verify), indent=2))
    else:
        print(json.dumps(build(args.root.resolve(), args.base.resolve()), indent=2))


if __name__ == "__main__":
    main()
