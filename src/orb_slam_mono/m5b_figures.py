"""Data-driven M5-B inspection figures; offline GT never enters tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .kitti_pose_evaluation import load_kitti_poses
from .m5b_evaluation import RUNS


COLORS = {"run-a1": "#1261a0", "run-b1": "#dc7a00", "run-c1": "#6b3fa0"}
LABELS = {"run-a1": "Full 0–299", "run-b1": "Fresh 0–149", "run-c1": "Fresh 150–299"}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _ply(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    end = lines.index("end_header")
    points = np.asarray([[float(v) for v in line.split()[:3]] for line in lines[end + 1:]], dtype=float)
    return points.reshape(-1, 3)


def _setup() -> None:
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160,
                         "font.size": 9, "axes.grid": True,
                         "grid.alpha": .22})


def make_figures(root: Path, output: Path) -> list[dict[str, Any]]:
    if output.exists():
        raise FileExistsError(output)
    _setup()
    base = root / "artifacts/m5b-final-evaluation"
    evaluation = base / "evaluation"
    data = {}
    for name in RUNS:
        data[name] = {"rows": _rows(base / name / "per_frame.jsonl"),
                      "metrics": _read(evaluation / name / "metrics.json"),
                      "segments": _read(evaluation / name / "segments.json"),
                      "errors": _read(evaluation / name / "per_frame_errors.json"),
                      "relative": _read(evaluation / name / "relative_pairs.json"),
                      "memory": _read(base / name / "memory_samples.json")}
    gt = load_kitti_poses(root / "data/ground_truth/kitti_odometry/00.txt")
    output.mkdir(parents=True)
    descriptions = []

    def save(fig, filename: str, explanation: str, frames: str, check: str) -> None:
        fig.tight_layout()
        path = output / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        descriptions.append({"relative_path": path.relative_to(root).as_posix(),
                             "absolute_path": str(path.resolve()), "description": explanation,
                             "source_frame_range": frames, "inspection": check,
                             "status": "Programmatically generated; human visual approval pending"})

    fig, axes = plt.subplots(3, 1, figsize=(15, 5), sharex=True)
    for ax, name in zip(axes, RUNS):
        values = np.full(300, np.nan)
        for row in data[name]["rows"]:
            values[row["frame_id"]] = 1 if row["pose_valid"] else 0
        ax.scatter(np.where(values == 1)[0], np.ones(int(np.sum(values == 1))),
                   color="#1b9e77", s=9, label="accepted online pose")
        ax.scatter(np.where(values == 0)[0], np.zeros(int(np.sum(values == 0))),
                   color="#d95f02", s=9, label="no online pose")
        ax.set_ylim(-.25, 1.25)
        ax.set_yticks([0, 1], ["missing", "valid"])
        ax.set_ylabel(LABELS[name])
        ax.axvspan(0, RUNS[name][0], color="0.92") if RUNS[name][0] else None
        ax.axvspan(RUNS[name][1], 299, color="0.92") if RUNS[name][1] < 299 else None
    axes[-1].set_xlim(-2, 301)
    axes[-1].set_xlabel("Original KITTI 00 source-frame ID; grey = outside requested window")
    axes[0].legend(loc="upper right", ncol=2)
    fig.suptitle("Online coverage; no interpolated or anchor-only poses")
    save(fig, "coverage_comparison.png", "Online accepted/missing frame raster for all three runs.",
         "0–299 (B 0–149; C 150–299)", "Check grey outside windows and no connecting lines through missing frames.")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, name in zip(axes, RUNS):
        start, end = RUNS[name]
        ax.plot(gt[start:end+1, 0, 3], gt[start:end+1, 2, 3], color="0.55",
                lw=1.4, label="KITTI GT T_wc (offline)")
        by_seg: dict[int, list[dict[str, Any]]] = {}
        for error in data[name]["errors"]:
            by_seg.setdefault(error["segment_id"], []).append(error)
        for segment, group in by_seg.items():
            group.sort(key=lambda row: row["frame_id"])
            centers = np.asarray([row["aligned_center_m"] for row in group])
            ax.plot(centers[:, 0], centers[:, 2], marker=".", ms=3,
                    lw=1.2, label=f"segment {segment}: own Sim(3)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(LABELS[name] + "; offline in-sample alignment")
        ax.set_xlabel("GT X (m)")
        ax.set_ylabel("GT Z (m)")
        ax.legend(fontsize=6, loc="best")
    save(fig, "segment_trajectory_comparison.png",
         "KITTI GT and eligible segments individually Sim(3)-aligned offline; no segments joined.",
         "A 0–299, B 0–149, C 150–299", "Check equal physical axis scales, segment labels and missing ineligible estimates.")

    fig, axes = plt.subplots(3, 1, figsize=(15, 7), sharex=True)
    for ax, name in zip(axes, RUNS):
        by_seg: dict[int, list[dict[str, Any]]] = {}
        for error in data[name]["errors"]:
            by_seg.setdefault(error["segment_id"], []).append(error)
        for segment, group in by_seg.items():
            ax.scatter([e["frame_id"] for e in group], [e["ate_m"] for e in group],
                       s=15, label=f"segment {segment}")
        ax.set_ylabel(LABELS[name] + "\nATE (m)")
        ax.legend(fontsize=6, loc="upper left") if by_seg else None
    axes[-1].set_xlim(0, 299)
    axes[-1].set_xlabel("Original source frame; absent points are missing or alignment-ineligible")
    fig.suptitle("Offline in-sample Sim(3) ATE; never an online accuracy signal")
    save(fig, "ate_by_frame.png", "Scored eligible-frame ATE, with absent frames left blank.",
         "A 0–299, B 0–149, C 150–299", "Check short eligible populations and no zeros at unscored frames.")

    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    x = np.arange(3)
    for offset, name in enumerate(RUNS):
        metrics = data[name]["metrics"]["relative_by_gap"]
        values = [metrics[str(g)]["rotation_deg_all"]["median"] for g in (1, 5, 10)]
        counts = [metrics[str(g)]["rotation_deg_all"]["count"] for g in (1, 5, 10)]
        xpos = x + (offset - 1) * .24
        available = [(xx, yy) for xx, yy in zip(xpos, values) if yy is not None]
        if available:
            axes[0].bar([pair[0] for pair in available], [pair[1] for pair in available], .22,
                        color=COLORS[name], label=LABELS[name])
        for xx, yy, count in zip(xpos, values, counts):
            axes[0].text(xx, (yy + .1) if yy is not None else .1,
                         f"n={count}" if yy is not None else "unavailable",
                         ha="center", fontsize=7)
        eligible_segments = {s["segment_id"] for s in data[name]["segments"] if s["status"] == "eligible"}
        rows = [r for r in data[name]["relative"] if r["source_gap"] == 5 and
                r["segment_id"] in eligible_segments and r["scale_ratio"] is not None]
        axes[1].scatter([r["first_frame"] for r in rows], [r["scale_ratio"] for r in rows],
                        s=14, color=COLORS[name], label=f"{LABELS[name]} n={len(rows)}")
    axes[0].set_xticks(x, ["Δ1", "Δ5", "Δ10"])
    axes[0].set_ylabel("Median relative rotation error (deg)")
    axes[0].legend()
    axes[1].axhline(1, color="0.4", ls="--", lw=.8)
    axes[1].set_ylabel("Fixed-scale Δ5 displacement / GT")
    axes[1].set_xlabel("First original source frame; missing = no eligible reliable exact-gap pair")
    axes[1].legend(fontsize=7)
    fig.suptitle("Offline exact-gap motion and scale; segments retain separate fitted scales")
    save(fig, "rpe_and_scale_drift.png", "Exact-gap rotation medians and reliable eligible gap-5 scale ratios.",
         "A 0–299, B 0–149, C 150–299", "Check pair counts; absent bars and points mean unavailable, not zero error.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name in RUNS:
        totals = [r["timing_ms"].get("total") for r in data[name]["rows"]]
        axes[0].hist(totals, bins=28, alpha=.45, color=COLORS[name], label=LABELS[name])
    axes[0].set_xlabel("Recorded tracker.process per frame (ms), excluding I/O")
    axes[0].set_ylabel("Frames")
    axes[0].legend()
    stages = ("orb", "association", "initialization", "pnp", "local_mapping", "bundle_adjustment")
    for offset, name in enumerate(RUNS):
        stage = data[name]["metrics"]["resource"]["timing"]["stage_runtime_ms"]
        available = [(index + (offset-1)*.24, stage[s]["median_ms"])
                     for index, s in enumerate(stages) if stage[s]["median_ms"] is not None]
        axes[1].bar([pair[0] for pair in available], [pair[1] for pair in available], .22,
                    color=COLORS[name], label=LABELS[name])
    axes[1].set_xticks(np.arange(len(stages)),
                      [f"{s}\n(n={data['run-a1']['metrics']['resource']['timing']['stage_runtime_ms'][s]['count']} A)"
                       for s in stages], rotation=35, ha="right", fontsize=7)
    axes[1].set_ylabel("Applicable-frame median (ms); zero bar may mean unavailable")
    axes[1].legend()
    fig.suptitle("CPU timing distributions; stage populations differ")
    save(fig, "runtime_distribution.png", "Per-frame processing histogram and applicable-stage medians.",
         "A 0–299, B 0–149, C 150–299", "Check long BA tail, stage counts and exclusion of image I/O from frame timings.")

    fig, axes = plt.subplots(3, 1, figsize=(14, 7))
    for ax, name in zip(axes, RUNS):
        samples = data[name]["memory"]
        report = data[name]["metrics"]["resource"]["memory"]
        ax.plot([s["elapsed_seconds"] for s in samples],
                [s["working_set_bytes"] / 1048576 for s in samples],
                color=COLORS[name], lw=1, label="sampled process working set")
        peak = report["os_peak_working_set_bytes"] / 1048576
        ax.axhline(peak, color="red", ls="--", lw=.8, label=f"OS peak {peak:.1f} MiB")
        ax.set_ylabel(LABELS[name] + "\nMiB")
        ax.legend(fontsize=7)
    axes[-1].set_xlabel("Elapsed run_sequence wall seconds; entire process incl. export")
    fig.suptitle("Windows process working set; OS peak includes unsampled native transients")
    save(fig, "memory_profile.png", "Sampled process resident working set and OS peak for each isolated run.",
         "A 0–299, B 0–149, C 150–299", "Check sample curves below OS high-water lines; this is not SLAM-only or GPU memory.")

    comparison = _read(evaluation / "restart_comparison.json")
    fig, axes = plt.subplots(2, 1, figsize=(15, 5), sharex=True)
    palette = {"both_valid": "#1b9e77", "full_only": "#d95f02",
               "restart_only": "#7570b3", "both_invalid": "#cccccc"}
    for ax, key in zip(axes, ("first_half", "second_half")):
        record = comparison[key]
        for category, ids in record["categories"].items():
            ax.scatter(ids, np.zeros(len(ids)), marker="s", s=28,
                       color=palette[category], label=f"{category}: {len(ids)}")
        for s in record["restart_segment_boundaries"]:
            ax.axvline(s["first_valid_frame"], color="black", lw=.5, alpha=.4)
        ax.set_yticks([])
        ax.set_ylabel(key.replace("_", " "))
        ax.legend(ncol=4, fontsize=7, loc="upper center")
    axes[-1].set_xlim(0, 299)
    axes[-1].set_xlabel("Original source frame; vertical lines = restart segment first online pose")
    fig.suptitle("Full window versus independent restarts; frame-ID intersections")
    save(fig, "restart_comparison.png", "Four-way valid-frame overlap with restarted segment boundaries.",
         "first 0–149, second 150–299", "Check source IDs in second panel begin at 150; colors count both/full-only/restart-only/neither.")

    counts = [len(data[name]["segments"]) for name in RUNS]
    fig, axes = plt.subplots(3, max(counts), figsize=(max(18, max(counts)*2.2), 8), squeeze=False)
    for row_index, name in enumerate(RUNS):
        records = data[name]["rows"]
        for col in range(max(counts)):
            ax = axes[row_index, col]
            if col >= counts[row_index]:
                ax.axis("off")
                continue
            seg = data[name]["segments"][col]
            segment_id = seg["segment_id"]
            points = _ply(base / name / "maps" / f"segment-{segment_id:03d}.ply")
            if len(points):
                ax.scatter(points[:, 0], points[:, 2], s=.25, color="0.55", rasterized=True)
            centers = [r["camera_center_world"] for r in records
                       if r["pose_valid"] and r["segment_id"] == segment_id]
            if centers:
                c = np.asarray(centers)
                ax.plot(c[:, 0], c[:, 2], marker=".", ms=2,
                        color=COLORS[name], lw=.7)
            ax.scatter([0], [0], marker="x", s=12, color="black")
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_title(f"{LABELS[name]} seg {segment_id}\n{seg['first_valid_frame']}–{seg['last_valid_frame']}", fontsize=7)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel("local Z (arbitrary units)", fontsize=7)
            if row_index == 2:
                ax.set_xlabel("local X (arbitrary units)", fontsize=7)
    fig.suptitle("Segment-local final sparse maps and online camera centers; scales/frames never merged")
    save(fig, "map_and_trajectory_overview.png", "All segment-local final PLY landmarks and online camera centers in separate panels.",
         "A 0–299, B 0–149, C 150–299", "Check no paths cross panels; axes are arbitrary segment-local units, not GT meters.")

    for record in descriptions:
        path = root / record["relative_path"]
        if not path.is_file() or path.stat().st_size < 1000:
            raise ValueError(f"Missing or empty plot: {path}")
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            record["pixel_dimensions"] = list(image.size)
    (output.parent / "FIGURE_GUIDE.json").write_text(json.dumps(descriptions, indent=2) + "\n", encoding="utf-8")
    markdown = ["# M5-B figure inspection guide", "",
                "All KITTI ground truth and Sim(3) alignment appear only in offline evaluation figures. Visual approval is pending.", ""]
    for item in descriptions:
        markdown += [f"## {Path(item['relative_path']).name}", "",
                     f"- Relative: `{item['relative_path']}`",
                     f"- Absolute: `{item['absolute_path']}`",
                     f"- Source frames: {item['source_frame_range']}",
                     f"- Contents: {item['description']}",
                     f"- Inspect: {item['inspection']}",
                     f"- Verified PNG dimensions: {item['pixel_dimensions']}", ""]
    (output.parent / "FIGURE_GUIDE.md").write_text("\n".join(markdown), encoding="utf-8")
    return descriptions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(make_figures(args.root.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
