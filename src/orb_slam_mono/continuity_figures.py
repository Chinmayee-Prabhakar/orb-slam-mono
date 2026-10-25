"""Render M4.5-D frozen-baseline and labeled offline-counterfactual figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_figures(root: Path, analysis: Path, figure_dir: Path) -> list[Path]:
    root, analysis, figure_dir = Path(root), Path(analysis), Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    segments = _read(analysis / "segment_analysis.json")
    attempts = _read(analysis / "initialization_attempts_summary.json")
    intervals = _read(analysis / "recovery_intervals.json")
    probes = _read(analysis / "existing_map_probes.json")
    gap_four = _read(analysis / "gap_four_probes.json")
    events = _read(analysis / "selected_frame_correspondences.json")
    created = []

    fig, ax = plt.subplots(figsize=(15, 5.5))
    for item in segments:
        y = item["segment_id"]
        first, last = item["second_frame"], item["last_valid_frame"]
        ax.broken_barh([(first, last-first+1)], (y-.32, .64), facecolors="#2677b8")
        ax.plot(first, y, "o", color="#168c62", ms=8)
        if item["loss_frame"] is not None:
            ax.plot(item["loss_frame"], y, "x", color="#c23838", ms=9, mew=2)
        ax.text(last+2, y, f'{item["valid_pose_count"]} poses', va="center", fontsize=9)
    ax.set(yticks=range(9), yticklabels=[f'Segment {i}' for i in range(9)],
           xlim=(-2, 305), xlabel="KITTI 00 original source frame ID",
           title="Frozen corrected M4: nine independent map segments; green = accepted second view, red = PnP loss")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    path = figure_dir / "segment_survival.png"
    fig.savefig(path, dpi=160)
    plt.close(fig); created.append(path)

    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharey=True)
    for item, ax in zip(segments, axes.flat):
        support = item["first_five_support"]
        x = [r["frame_id"] for r in support]
        ax.plot(x, [r["map_points"] for r in support], "o-", color="#2468a0", label="map points")
        ax.plot(x, [r["unique_pairs"] for r in support], "s-", color="#db8b23", label="retained pairs")
        ax.plot(x, [np.nan if r["pnp_inliers"] is None else r["pnp_inliers"]
                    for r in support], "^-", color="#1d9a60", label="PnP inliers")
        ax.axhline(12, ls="--", lw=1, color="#b13b3b")
        ax.set_title(f'Seg {item["segment_id"]}: {item["anchor_frame"]}→{item["second_frame"]}; '
                     f'{item["initial_points"]} initial points', fontsize=10)
        ax.set_xticks(x)
        ax.set_xlabel("source frame")
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Frozen online first-frame support; dashed line = 12-point/inlier floor", fontsize=14)
    fig.tight_layout()
    path = figure_dir / "fresh_map_correspondence_support.png"
    fig.savefig(path, dpi=150)
    plt.close(fig); created.append(path)

    colors = {None: "#198f60", "low_parallax": "#df9a21",
              "low_cheirality_fraction": "#ba4b54",
              "insufficient_essential_inliers": "#796bb5",
              "insufficient_accepted_pose_stability_trials": "#75515a"}
    fig, ax = plt.subplots(figsize=(17, 5))
    for reason, color in colors.items():
        subset = [a for a in attempts if a["reason"] == reason]
        ax.scatter([a["candidate_frame"] for a in subset], [a["gap"] for a in subset],
                   color=color, label="accepted initialization" if reason is None else reason,
                   s=58 if reason is None else 35, marker="o" if reason is None else "x")
    for interval in intervals:
        if interval["processed_frames"] >= 20:
            ax.axvspan(interval["first_frame"], interval["last_frame"], color="#b5bcc6", alpha=.16)
    for item in segments:
        if item["loss_frame"] is not None:
            ax.axvline(item["loss_frame"], color="#cf5353", alpha=.25, lw=1)
    ax.set(xlim=(-2, 302), ylim=(2, 12), xlabel="KITTI 00 original source frame ID",
           ylabel="anchor-to-candidate frame gap",
           title="Frozen initialization attempts; gray = long no-pose interval, red lines = tracking losses")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    path = figure_dir / "initialization_recovery_timeline.png"
    fig.savefig(path, dpi=150)
    plt.close(fig); created.append(path)

    fig, ax = plt.subplots(figsize=(15, 5))
    x = np.arange(len(probes))
    pairs = [p["unique_correspondences"] for p in probes]
    inliers = [0 if p["raw_inliers"] is None else p["raw_inliers"] for p in probes]
    bars = ax.bar(x-.2, pairs, .4, label="retained pairs", color="#db8b23")
    ax.bar(x+.2, inliers, .4, label="raw PnP inliers; zero-height if unavailable", color="#2677b8")
    for bar, p in zip(bars, probes):
        if p["passes_unchanged_pnp_gates_offline"]:
            bar.set_edgecolor("#159260"); bar.set_linewidth(3)
    ax.axhline(12, color="#b13b3b", ls="--", lw=1)
    ax.set_xticks(x, [f'{p["lost_frame"]}→{p["frame_id"]}' for p in probes], rotation=55, ha="right")
    ax.set(ylabel="count", xlabel="loss frame → offline probe frame",
           title="Offline retained-map probes only; green border = internal PnP gates pass, not accepted tracking")
    ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    path = figure_dir / "existing_map_counterfactual.png"
    fig.savefig(path, dpi=150)
    plt.close(fig); created.append(path)

    fig, ax = plt.subplots(figsize=(9, 4))
    for index, trial in enumerate(gap_four):
        d = trial["diagnostics"]
        ax.bar(index-.18, d.get("essential_inliers") or 0, .36, color="#2677b8",
               label="essential inliers" if index == 0 else None)
        ax.bar(index+.18, d.get("pose_inliers") or 0, .36, color="#d5902c",
               label="cheirality pose inliers" if index == 0 else None)
        ax.text(index, max(d.get("essential_inliers") or 0, d.get("pose_inliers") or 0)+10,
                trial["reason"], ha="center", fontsize=9)
    ax.set_xticks(range(len(gap_four)), [f'{t["anchor_frame"]}→{t["candidate_frame"]}' for t in gap_four])
    ax.set(ylabel="count", title="Predeclared gap-four offline probes; both rejected under unchanged geometry")
    ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    path = figure_dir / "gap_four_counterfactual.png"
    fig.savefig(path, dpi=150)
    plt.close(fig); created.append(path)

    image_dir = root / "data/sequences/00/image_0"
    initial_pixels = {item["second_frame"]: item["initial_landmark_pixels"] for item in segments}
    groups = (("frames_041_043.png", (41, 42, 43)),
              ("frames_203_207.png", (203, 206, 207)),
              ("frames_088_121.png", (88, 89, 121, 122)))
    for filename, frame_ids in groups:
        panels = []
        for fid in frame_ids:
            raw = cv2.imread(str(image_dir / f"{fid:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                raise FileNotFoundError(fid)
            image = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            for px, py in initial_pixels.get(fid, []):
                cv2.circle(image, (round(px), round(py)), 3, (75, 195, 75), 1, cv2.LINE_AA)
            for row in events.get(str(fid), {}).get("correspondences", []):
                px, py = row["pixel_xy"]
                cv2.circle(image, (round(px), round(py)), 3, (25, 190, 240), 1, cv2.LINE_AA)
            header = np.full((54, image.shape[1], 3), 24, dtype=np.uint8)
            count = len(events.get(str(fid), {}).get("correspondences", []))
            text = f"source {fid}: green=initial points; yellow=retained pairs ({count})"
            cv2.putText(header, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .68,
                        (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(np.vstack((header, image)))
        output = np.hstack(panels)
        path = figure_dir / filename
        if not cv2.imwrite(str(path), output):
            raise OSError(path)
        created.append(path)

    # Two genuinely gate-passing proposals receive a separate image view.
    panels = []
    for probe_frame in (9, 83):
        raw = cv2.imread(str(image_dir / f"{probe_frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        p = next(p for p in probes if p["frame_id"] == probe_frame)
        for pair in p["pairs"]:
            x, y = pair["pixel_xy"]
            cv2.circle(image, (round(x), round(y)), 4, (25, 190, 240), 1, cv2.LINE_AA)
        header = np.full((54, image.shape[1], 3), 24, dtype=np.uint8)
        text = f'OFFLINE ONLY frame {probe_frame}: {p["unique_correspondences"]} pairs, {p["raw_inliers"]} inliers'
        cv2.putText(header, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .73,
                    (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(np.vstack((header, image)))
    path = figure_dir / "recovery_probe_frames_009_083.png"
    if not cv2.imwrite(str(path), np.vstack(panels)):
        raise OSError(path)
    created.append(path)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    for path in make_figures(args.root, args.analysis, args.figures):
        print(path.resolve())


if __name__ == "__main__":
    main()
