"""Separate offline KITTI GT figure for the eligible 88–121 segment only."""

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
from .m5c_data import load_rows, pose_geometry, sha256, validate_sources


def comparison_data(root: Path) -> dict[str, Any]:
    validate_sources(root)
    rows = load_rows(root / "artifacts/m5b-final-evaluation/run-a1/per_frame.jsonl")
    segment_results = json.loads((root / "artifacts/m5b-final-evaluation/evaluation/run-a1/segments.json").read_text(encoding="utf-8"))
    segment = next(s for s in segment_results if s["segment_id"] == 4)
    if segment["status"] != "eligible" or (segment["first_valid_frame"], segment["last_valid_frame"]) != (88, 121):
        raise ValueError("Accepted GT-eligible segment 004 changed")
    alignment = segment["alignment"]
    scale = float(alignment["scale_m_per_estimated_unit"])
    Q = np.asarray(alignment["Q_est_world_to_gt_world"], dtype=float)
    b = np.asarray(alignment["b_gt_m"], dtype=float)
    if scale <= 0 or np.linalg.det(Q) < 1-1e-9:
        raise ValueError("Improper offline Sim(3) alignment")
    ids = np.arange(88, 122)
    estimates = np.asarray([pose_geometry(rows[i])[0] for i in ids])
    gt_all = load_kitti_poses(root / "data/ground_truth/kitti_odometry/00.txt")
    gt = gt_all[ids, :3, 3]
    aligned = (scale * (Q @ estimates.T)).T + b
    errors = np.linalg.norm(aligned - gt, axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))
    accepted = float(segment["in_sample_ate_m"]["rmse"])
    if len(errors) != 34 or not np.isclose(rmse, accepted, atol=1e-10) or not np.isclose(rmse, .22503195165117487, atol=1e-10):
        raise ValueError("Offline GT figure differs from accepted M5-A/M5-B score")
    suffix = segment["prefix_fit_suffix_evaluation"]
    if suffix["status"] != "eligible" or suffix["fit_pose_count"] != 17 or suffix["suffix_pose_count"] != 17:
        raise ValueError("Prefix/suffix population changed")
    return {"ids": ids, "gt": gt, "aligned": aligned, "errors": errors,
            "in_sample_rmse": rmse, "prefix_fit_suffix_rmse": suffix["suffix_ate_m"]["rmse"],
            "scale_m_per_unit": scale, "det_Q": float(np.linalg.det(Q))}


def render(root: Path, output: Path) -> dict[str, Any]:
    value = comparison_data(root)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), dpi=170,
                             gridspec_kw={"width_ratios": [1.25, 1]}, facecolor="#101820")
    for ax in axes:
        ax.set_facecolor("#17232e")
        ax.tick_params(colors="white", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#91a4b1")
    gt, est, ids, errors = value["gt"], value["aligned"], value["ids"], value["errors"]
    axes[0].plot(gt[:, 0], gt[:, 2], c="#f0ae67", lw=3, label="KITTI GT (offline only)")
    axes[0].plot(est[:, 0], est[:, 2], c="#5fe1cf", lw=3, label="Estimated + segment Sim(3) (offline)")
    for source in (88, 97, 109, 121):
        offset = source-88
        axes[0].scatter(gt[offset, 0], gt[offset, 2], c="#f0ae67", s=30)
        axes[0].annotate(str(source), (gt[offset, 0], gt[offset, 2]), color="white",
                         xytext=(5, 7), textcoords="offset points")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].set(xlabel="KITTI world X (m)", ylabel="KITTI world Z (m)")
    axes[0].set_title("Segment 004, source frames 88–121", color="white", fontsize=15)
    axes[0].xaxis.label.set_color("white")
    axes[0].yaxis.label.set_color("white")
    axes[0].legend(facecolor="#17232e", edgecolor="#91a4b1", labelcolor="white")
    axes[1].plot(ids, errors, c="#5fe1cf", marker="o", ms=4, lw=2,
                 label="In-sample aligned position error")
    axes[1].axhline(value["in_sample_rmse"], color="#f0ae67", ls="--", lw=1.5,
                    label=f"34-pose in-sample RMSE {value['in_sample_rmse']:.3f} m")
    axes[1].axvline(104.5, color="#bdc8d0", ls=":", lw=1)
    axes[1].text(105, max(errors)*.95, "prefix fit 88–104 | suffix 105–121",
                 color="white", fontsize=9)
    axes[1].set(xlim=(87.5, 122.5), xlabel="Original KITTI source frame ID",
                ylabel="Aligned center error (m)")
    axes[1].set_title("No pose at frame 122; no value plotted", color="white", fontsize=15)
    axes[1].xaxis.label.set_color("white")
    axes[1].yaxis.label.set_color("white")
    axes[1].legend(facecolor="#17232e", edgecolor="#91a4b1", labelcolor="white")
    fig.suptitle("Offline KITTI GT alignment - NOT used for online tracking", color="white", fontsize=19)
    fig.text(.5, .055,
             f"Segment-only in-sample ATE {value['in_sample_rmse']:.6f} m (34 poses); "
             f"prefix-fit / suffix-score {value['prefix_fit_suffix_rmse']:.6f} m (17 poses). "
             "Full-run 0.184586 m aggregates two independent segments.",
             ha="center", color="white", fontsize=11)
    fig.text(.5, .022, "KITTI Vision Benchmark Suite / Geiger, Lenz, Urtasun (CVPR 2012) / CC BY-NC-SA 3.0",
             ha="center", color="#c7d0d8", fontsize=9)
    fig.subplots_adjust(left=.07, right=.98, top=.83, bottom=.18, wspace=.24)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"path": output.relative_to(root).as_posix(), "sha256": sha256(output),
            "source_range": [88, 121], "segment_id": 4, "online_pose_count": 34,
            "in_sample_rmse_m": value["in_sample_rmse"],
            "prefix_fit_suffix_rmse_m": value["prefix_fit_suffix_rmse"],
            "prefix_fit_pose_count": 17, "suffix_score_pose_count": 17,
            "scale_m_per_unit": value["scale_m_per_unit"], "det_Q": value["det_Q"],
            "ground_truth_stage": "offline only"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.root.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
