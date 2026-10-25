"""Offline M5-B analysis of independently saved full-M4 runs.

Only this module loads KITTI truth. Online success comes from per_frame.jsonl,
and each map segment keeps its own monocular scale and Sim(3) fit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .kitti_pose_evaluation import load_kitti_poses
from .tracking_diagnostics import _normalized
from .trajectory_evaluation import (_pose_error, coverage_report, evaluate_segments,
                                    fit_similarity, load_online_rows,
                                    validate_anchor_keyframes, validate_source_index)


RUNS = {"run-a1": (0, 299), "run-b1": (0, 149), "run-c1": (150, 299)}
GAPS = (1, 5, 10)
EVENT_FILES = ("initialization_attempts", "loss_events", "recovery_events",
               "keyframe_events", "map_growth_events", "ba_events")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def describe(values: Any, unit: str) -> dict[str, Any]:
    numbers = np.asarray(list(values), dtype=float)
    if not len(numbers):
        return {"count": 0, "rmse": None, "median": None, "p90": None,
                "maximum": None, "unit": unit}
    return {"count": int(len(numbers)), "rmse": float(np.sqrt(np.mean(numbers ** 2))),
            "median": float(np.median(numbers)), "p90": float(np.percentile(numbers, 90)),
            "maximum": float(np.max(numbers)), "unit": unit}


def relative_summary(rows: list[dict[str, Any]], eligible_segments: set[int]) -> dict[str, Any]:
    result = {}
    for gap in GAPS:
        group = [row for row in rows if row["source_gap"] == gap]
        eligible = [row for row in group if row["segment_id"] in eligible_segments]
        result[str(gap)] = {
            "all_same_segment_exact_gap_pairs": len(group),
            "alignment_eligible_pairs": len(eligible),
            "rotation_deg_all": describe((r["rotation_error_deg"] for r in group), "deg"),
            "direction_deg_all": describe((r["translation_direction_error_deg"] for r in group
                                           if r["translation_direction_error_deg"] is not None), "deg"),
            "rotation_deg_eligible": describe((r["rotation_error_deg"] for r in eligible), "deg"),
            "direction_deg_eligible": describe((r["translation_direction_error_deg"] for r in eligible
                                                if r["translation_direction_error_deg"] is not None), "deg"),
            "fixed_scale_displacement_error_m": describe((r["displacement_error_m"] for r in eligible
                                                            if r["displacement_error_m"] is not None), "m"),
            "fixed_scale_translation_vector_error_m": describe(
                (r["relative_translation_vector_error_m"] for r in eligible
                 if r["relative_translation_vector_error_m"] is not None), "m"),
            "fixed_scale_ratio": describe((r["scale_ratio"] for r in eligible
                                           if r["scale_ratio"] is not None), "ratio")}
    return result


def evaluate_saved_run(run_dir: Path, sequence: Path, gt: np.ndarray,
                       start: int, end: int) -> dict[str, Any]:
    index = validate_source_index(sequence, len(gt), start, end)
    rows, poses = load_online_rows(run_dir / "per_frame.jsonl", start, end)
    summary = _load_json(run_dir / "summary.json")
    events = {name: _load_json(run_dir / (name + ".json")) for name in EVENT_FILES}
    recovery = events["recovery_events"]
    coverage = coverage_report(rows, poses, events["initialization_attempts"],
                               events["loss_events"], summary, events["keyframe_events"],
                               events["map_growth_events"], recovery)
    anchors = {int(key): frame for key, frame in coverage["segment_anchor_frames"].items()}
    anchor_validation = validate_anchor_keyframes(run_dir / "final_keyframes.json", anchors)
    segments, errors, relative = evaluate_segments(poses, gt, anchors)
    eligible_ids = {s["segment_id"] for s in segments if s["status"] == "eligible"}
    profile = _load_json(run_dir / "resource_profile.json")
    if profile["processed_frames"] != len(rows) or summary["total_frames"] != len(rows):
        raise ValueError("Profile, summary and online output disagree on processed count")
    metrics = {"run": run_dir.name, "source_index": index,
               "coverage": coverage,
               "online_pose_fraction_percent": 100 * len(poses) / len(rows),
               "missing_pose_fraction_percent": 100 * (len(rows) - len(poses)) / len(rows),
               "accepted_pnp_poses": sum(r["diagnostics"].get("pnp_status") == "accepted" for r in rows),
               "accepted_initialization_pairs": summary["accepted_initialization_pairs"],
               "initialization_rejection_reasons": summary["initialization_rejections"],
               "loss_frames": [e["frame_id"] for e in events["loss_events"]],
               "map": ({key: summary.get(key) for key in
                        ("keyframes", "new_map_points", "active_map_points_at_end",
                         "valid_map_points_all_segments", "culled_map_points",
                         "observation_count", "map_points_per_segment")}
                       | {"initial_map_points": sum(
                           a["diagnostics"]["usable_map_points"]
                           for a in events["initialization_attempts"]
                           if a["initialization_status"] == "accepted")}),
               "ba": {"attempts": summary["ba_attempts"], "accepted": summary["ba_accepted"],
                      "rejected": summary["ba_rejected"],
                      "rejection_reasons": dict(Counter(e.get("reason") for e in events["ba_events"]
                                                        if not e["applied"]))},
               "eligible_segment_count": len(eligible_ids), "eligible_online_poses": len(errors),
               "in_sample_ate_m": describe((e["ate_m"] for e in errors), "m"),
               "in_sample_aligned_orientation_deg": describe((e["orientation_deg"] for e in errors), "deg"),
               "relative_by_gap": relative_summary(relative, eligible_ids),
               "segment_scales_m_per_unit": {str(s["segment_id"]):
                                              s["alignment"]["scale_m_per_estimated_unit"]
                                              for s in segments if s["alignment"] is not None},
               "resource": profile}
    # The accepted evaluator reports anchor orientation per segment, not per pose;
    # preserve those separate populations instead of fabricating an aggregate.
    return {"metrics": metrics, "rows": rows, "poses": poses, "segments": segments,
            "errors": errors, "relative": relative, "anchors": anchor_validation,
            "events": events}


def common_pose_accuracy(first: dict[int, Any], second: dict[int, Any],
                         gt: np.ndarray) -> dict[str, Any]:
    """M5-A matched-ID, matched-fit-ID iterative eligibility for two saved runs."""
    common = set(first) & set(second)
    original = set(common)
    iterations = []
    fits: dict[str, dict[int, Any]] = {}
    reasons: dict[str, dict[str, str]] = {}
    for _ in range(len(original) + 1):
        eligible_sets = []
        fits, reasons = {}, {}
        for name, poses in (("full", first), ("restart", second)):
            groups: dict[int, list[int]] = defaultdict(list)
            for frame in sorted(common):
                groups[poses[frame].segment_id].append(frame)
            eligible, fit_by_segment, errors = set(), {}, {}
            for segment_id, ids in groups.items():
                fit, reason = fit_similarity(
                    np.asarray([poses[i].center for i in ids]), gt[ids, :3, 3])
                if fit is None:
                    errors[str(segment_id)] = reason
                else:
                    fit_by_segment[segment_id] = fit
                    eligible.update(ids)
            fits[name], reasons[name] = fit_by_segment, errors
            eligible_sets.append(eligible)
        retained = set.intersection(*eligible_sets)
        iterations.append({"input_count": len(common), "retained_count": len(retained),
                           "ineligible_segments": reasons})
        if retained == common:
            break
        common = retained
    else:
        raise RuntimeError("Common-frame eligibility did not converge")
    scores = {}
    for name, poses in (("full", first), ("restart", second)):
        errors = [_pose_error(poses[i], gt[i], fits[name][poses[i].segment_id])
                  for i in sorted(common)]
        scores[name] = {"ate_m": describe((e["ate_m"] for e in errors), "m"),
                        "orientation_deg": describe((e["orientation_deg"] for e in errors), "deg"),
                        "segment_scales_m_per_unit": {str(k): fit.scale for k, fit in fits[name].items()}}
    return {"initial_common_online_frame_count": len(original),
            "eligible_common_frame_count": len(common),
            "eligible_common_frame_ids": sorted(common),
            "excluded_initial_common_frame_ids": sorted(original - common),
            "eligibility_iterations": iterations, "scores": scores,
            "method": "M5-A common original-ID intersection, iterative segment eligibility and independent per-run segment Sim(3) fits on the same retained IDs"}


def compare_restart(full: dict[str, Any], restart: dict[str, Any],
                    gt: np.ndarray) -> dict[str, Any]:
    full_rows = {r["frame_id"]: r for r in full["rows"]}
    restart_rows = {r["frame_id"]: r for r in restart["rows"]}
    ids = sorted(restart_rows)
    if any(frame not in full_rows for frame in ids):
        raise ValueError("Restart range is outside full-window range")
    categories = {"both_valid": [], "full_only": [], "restart_only": [], "both_invalid": []}
    state_agreement = []
    for frame in ids:
        a, b = full_rows[frame], restart_rows[frame]
        category = ("both_valid" if a["pose_valid"] and b["pose_valid"] else
                    "full_only" if a["pose_valid"] else
                    "restart_only" if b["pose_valid"] else "both_invalid")
        categories[category].append(frame)
        if a["state"] == b["state"]:
            state_agreement.append(frame)
    full_valid = {i for i in ids if full_rows[i]["pose_valid"]}
    restart_valid = {i for i in ids if restart_rows[i]["pose_valid"]}
    return {"source_range_inclusive": [ids[0], ids[-1]],
            "categories": categories,
            "category_counts": {key: len(value) for key, value in categories.items()},
            "valid_intersection_ids": sorted(full_valid & restart_valid),
            "valid_union_ids": sorted(full_valid | restart_valid),
            "valid_intersection_count": len(full_valid & restart_valid),
            "valid_union_count": len(full_valid | restart_valid),
            "state_agreement_count": len(state_agreement),
            "state_disagreement_ids": sorted(set(ids) - set(state_agreement)),
            "full_segment_boundaries": [{key: s[key] for key in
                                         ("segment_id", "first_valid_frame", "last_valid_frame", "online_valid_poses")}
                                        for s in full["segments"] if s["first_valid_frame"] <= ids[-1] and
                                        s["last_valid_frame"] >= ids[0]],
            "restart_segment_boundaries": [{key: s[key] for key in
                                            ("segment_id", "first_valid_frame", "last_valid_frame", "online_valid_poses")}
                                           for s in restart["segments"]],
            "full_loss_frames_in_window": [e["frame_id"] for e in full["events"]["loss_events"]
                                           if ids[0] <= e["frame_id"] <= ids[-1]],
            "restart_loss_frames": [e["frame_id"] for e in restart["events"]["loss_events"]],
            "full_accepted_initializations_in_window": [
                [a["anchor_frame"], a["candidate_frame"]]
                for a in full["events"]["initialization_attempts"]
                if a["initialization_status"] == "accepted" and
                ids[0] <= a["candidate_frame"] <= ids[-1]],
            "restart_accepted_initializations": [
                [a["anchor_frame"], a["candidate_frame"]]
                for a in restart["events"]["initialization_attempts"]
                if a["initialization_status"] == "accepted"],
            "common_pose_accuracy": common_pose_accuracy(full["poses"], restart["poses"], gt)}


def compare_saved_outputs(left: Path, right: Path,
                          *, allow_ba_difference: bool = False) -> dict[str, Any]:
    """Exact online/geometry repeatability, excluding only measured timing."""
    left_rows = [_load_json_line for _load_json_line in
                 (json.loads(line) for line in (left / "per_frame.jsonl").read_text(encoding="utf-8").splitlines())]
    right_rows = [json.loads(line) for line in (right / "per_frame.jsonl").read_text(encoding="utf-8").splitlines()]
    core = ("frame_id", "state", "segment_id", "pose_valid", "failure_reason",
            "T_cw_R", "T_cw_t", "camera_center_world")
    core_equal = (len(left_rows) == len(right_rows) and
                  all(all(a.get(key) == b.get(key) for key in core)
                      for a, b in zip(left_rows, right_rows)))
    diag_equal = (len(left_rows) == len(right_rows) and
                  all(_normalized({key: val for key, val in a["diagnostics"].items()
                                   if not (allow_ba_difference and key == "bundle_adjustment")}) ==
                      _normalized({key: val for key, val in b["diagnostics"].items()
                                   if not (allow_ba_difference and key == "bundle_adjustment")})
                      for a, b in zip(left_rows, right_rows)))
    events = {}
    for name in EVENT_FILES:
        if allow_ba_difference and name == "ba_events":
            continue
        events[name] = _normalized(_load_json(left / (name + ".json"))) == \
                       _normalized(_load_json(right / (name + ".json")))
    exports = {name: (left / name).read_bytes() == (right / name).read_bytes()
               for name in ("estimated_poses.json", "final_keyframes.json")}
    left_ply = {p.name: p.read_bytes() for p in (left / "maps").glob("*.ply")}
    right_ply = {p.name: p.read_bytes() for p in (right / "maps").glob("*.ply")}
    ply_equal = left_ply == right_ply
    return {"core_online_equal": core_equal, "non_timing_diagnostics_equal": diag_equal,
            "event_lists_equal": events, "pose_exports_byte_equal": exports,
            "all_ply_byte_equal": ply_equal, "ply_count": len(left_ply),
            "passed": core_equal and diag_equal and all(events.values()) and
                      all(exports.values()) and ply_equal,
            "timing_policy": "Ignore per-frame timing_ms and nested *_ms event values only; optionally exclude BA candidate diagnostics when comparing pre-correction M4"}


def repeatability_report(root: Path) -> dict[str, Any]:
    repeats = {}
    for stem in ("a", "b", "c"):
        original = root / "artifacts/m5b-final-evaluation" / f"run-{stem}1"
        repeat = root / "artifacts/m5b-final-evaluation" / f"run-{stem}2"
        agreement = compare_saved_outputs(original, repeat)
        if not agreement["passed"]:
            raise ValueError(f"Non-timing fresh-process repeat changed: {stem}")
        profiles = [_load_json(folder / "resource_profile.json") for folder in (original, repeat)]
        repeats[stem] = {"non_timing_comparison": agreement,
                         "fresh_process_pids": [p["fresh_process_pid"] for p in profiles],
                         "wall_seconds": [p["timing"]["wall_seconds"] for p in profiles],
                         "os_peak_working_set_bytes": [p["memory"]["os_peak_working_set_bytes"]
                                                       for p in profiles],
                         "recorded_frame_seconds": [p["timing"]["recorded_frame_processing_seconds"]
                                                    for p in profiles]}
    return repeats


def analyze(root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    sequence = root / "data/sequences/00"
    gt = load_kitti_poses(root / "data/ground_truth/kitti_odometry/00.txt")
    values = {name: evaluate_saved_run(root / "artifacts/m5b-final-evaluation" / name,
                                       sequence, gt, *bounds)
              for name, bounds in RUNS.items()}
    comparison = {"first_half": compare_restart(values["run-a1"], values["run-b1"], gt),
                  "second_half": compare_restart(values["run-a1"], values["run-c1"], gt)}
    frozen = compare_saved_outputs(root / "artifacts/m5b-final-evaluation/run-a1",
                                   root / "artifacts/m4-kitti-00-ba-validation-final/full",
                                   allow_ba_difference=True)
    corrected = compare_saved_outputs(root / "artifacts/m5b-final-evaluation/run-a1",
                                      root / "artifacts/m45c-ba-depth-correction/corrected_full")
    if not frozen["passed"] or not corrected["passed"]:
        raise ValueError("Instrumented full run differs from frozen online baseline")
    m5a_segments = _load_json(root / "artifacts/m5a-frozen-evaluation/full/segments.json")
    m5a_errors = _load_json(root / "artifacts/m5a-frozen-evaluation/full/per_frame_errors.json")
    m5a_relative = _load_json(root / "artifacts/m5a-frozen-evaluation/full/relative_pairs.json")
    # M5-A writes singular-value tuples as JSON arrays; compare serialized values.
    m5a_equal = (json.loads(json.dumps(values["run-a1"]["segments"])) == m5a_segments and
                 values["run-a1"]["errors"] == m5a_errors and
                 values["run-a1"]["relative"] == m5a_relative)
    if not m5a_equal:
        raise ValueError("Accepted M5-A scoring results changed for full run")
    repeats = repeatability_report(root)
    output.mkdir(parents=True)
    for name, value in values.items():
        folder = output / name
        _save(folder / "metrics.json", value["metrics"])
        _save(folder / "segments.json", value["segments"])
        _save(folder / "per_frame_errors.json", value["errors"])
        _save(folder / "relative_pairs.json", value["relative"])
        _save(folder / "anchor_validation.json", value["anchors"])
    _save(output / "restart_comparison.json", comparison)
    _save(output / "frozen_comparison.json", {"original_corrected_M4": frozen,
                                              "m45c_corrected_objective": corrected,
                                              "m5a_scoring_exact_equal": m5a_equal})
    _save(output / "repeatability.json", repeats)
    return {"runs": {name: v["metrics"] for name, v in values.items()},
            "comparison": comparison, "frozen": frozen, "corrected": corrected,
            "repeatability": repeats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root, args.output)
    print(json.dumps({name: {"online_valid": run["coverage"]["online_valid_poses"],
                             "eligible": run["eligible_online_poses"]}
                      for name, run in report["runs"].items()}, indent=2))


if __name__ == "__main__":
    main()
