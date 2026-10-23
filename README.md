# orb-slam-mono

An independent, inspectable Python/OpenCV monocular visual SLAM project. M4 adds keyframes, image-based map growth, bounded ORB descriptor evidence, and sparse local bundle adjustment to M3's sequential PnP tracker. Every map segment has its own arbitrary monocular origin and scale; invalid frame poses remain null.

## Layout

| Path | Purpose |
| --- | --- |
| `src/orb_slam_mono/camera.py` | Validated pinhole calibration and KITTI `P0` parsing |
| `src/orb_slam_mono/datasets.py` | Image sequence contract and KITTI grayscale loader |
| `src/orb_slam_mono/synthetic.py` | Deterministic two-image fallback |
| `src/orb_slam_mono/features.py` | ORB extraction |
| `src/orb_slam_mono/matching.py` | Hamming KNN, ratio filter, optional mutual check |
| `src/orb_slam_mono/diagnostics.py` | Machine-readable stage counts and future diagnostic fields |
| `src/orb_slam_mono/demo.py` | Command-line demonstration and PNG/JSON output |
| `src/orb_slam_mono/geometry.py` | Essential matrix, relative pose, triangulation filters, initialization decision |
| `src/orb_slam_mono/synthetic3d.py` | Reproducible physical 3D benchmark and rendered ORB image pair |
| `src/orb_slam_mono/geometry_demo.py` | M2 CLI, diagnostics, images, PLY/JSON export |
| `src/orb_slam_mono/map_state.py` | Frame, map-point, segment, and observation identity invariants |
| `src/orb_slam_mono/tracking.py` | Adaptive initializer, map associations, calibrated PnP, states and recovery |
| `src/orb_slam_mono/synthetic_sequence.py` | Physical multi-view rendered-image fixture with separate truth |
| `src/orb_slam_mono/sequence_demo.py` | M3 sequential runner, per-frame diagnostics, plots, poses, and PLY exports |
| `src/orb_slam_mono/local_mapping.py` | Keyframe policy, calibrated map growth, landmark quality |
| `src/orb_slam_mono/bundle_adjustment.py` | Gauge-fixed sparse local reprojection optimization |
| `src/orb_slam_mono/m4_ablation.py` | Four fixed M4 configurations and comparison CSV/JSON |
| `src/orb_slam_mono/kitti_pose_evaluation.py` | Offline KITTI relative-pose scoring; never imported by the estimator |
| `configs/default.json` | Tunable ORB, matching, geometry, and visualization settings |
| `scripts/extract_kitti_00_subset.py` | Selective ZIP extraction with byte/CRC verification |
| `scripts/extract_kitti_00_poses.py` | Extract sequence-00 ground truth to ignored offline data |
| `scripts/summarize_kitti_00.py` | Fixed-pair CSV/JSON validation summary |
| `scripts/audit_kitti_correspondences.py` | Original 20→25 correspondence and model-stability audit |
| `scripts/score_m4_synthetic_offline.py` | Offline-only known-baseline scoring of saved M4 synthetic poses |
| `tests/` | Standard-library unit tests, including optional local KITTI smoke test |

## M4 keyframes, map growth, and local BA

Install dependencies from `requirements.txt` (including CPU SciPy). From the project root in PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m unittest discover -s tests -v
python -m orb_slam_mono.m4_ablation --mode synthetic --vary-visibility --output-dir artifacts/m4-synthetic-varied-final
python -m orb_slam_mono.m4_ablation --mode kitti --kitti-sequence data/sequences/00 --start 0 --end 299 --output-dir artifacts/m4-kitti-00-final
```

Each ablation processes the same chronological frames and shared M2/PnP thresholds: `baseline` preserves M3 fixed landmarks and descriptors; `growth` adds keyframes and triangulation; `descriptors` also updates landmark Hamming medoids; `full` also schedules local BA. `comparison.csv` and `comparison.json` summarize the four runs. Each subfolder holds per-frame CSV/JSONL, keyframe/map/BA event JSON, state and count plots, segment-local PLY, correspondence images, and `estimated_poses.json`. M4 pose JSON distinguishes final BA-revised keyframe poses from online per-frame reports. Valid points across historical segments and points in the currently active segment are reported separately. All outputs stay under ignored `artifacts/`.

On KITTI 00 frames 0–299, the four configurations yield **97, 126, 145, and 144 valid poses** and longest valid spans **13, 30, 34, and 60** respectively. Full M4 makes 1,816 new points, accepts 4/10 BA windows, and still has **156/300 frames without poses**. Six BA windows reject for held-out reprojection regression even though their training cost drops. KITTI M4 trajectory accuracy is not yet independently scored. The varying-visibility synthetic sequence uses projected landmarks with separate truth; it yields 12, 13, 15, and 15 valid poses. Full M4's first-segment offline median/max errors are 1.265°/2.059° in rotation and 0.145/0.346 scene units in center position after evaluation-only scale normalization. No ground-truth pose or scale enters tracking or BA.

**These are the preserved original M4 results.** The targeted BA safety correction rechecks every stored observation affected by a proposed point or camera update, including frames outside the local optimization window. It also enforces the configured 1,000-point active capacity before admitting new landmarks and plots final adjusted keyframe poses with final map points. Under the corrected fixed KITTI 00 0–299 comparison, valid poses are **97, 126, 145, 145** and longest spans are **13, 30, 34, 34**. All ten full-M4 BA candidates terminate at the 12-evaluation budget and fail external validation; none are applied. The original 60-pose span is therefore not retained. The corrected varied-visibility synthetic comparison still has **12, 13, 15, 15** valid poses, but its one full-BA candidate is also rejected. Training reprojection improves while outside-window or other held-out measurements worsen. M4 is complete for its defined functional and safety criteria. Real-image BA benefit remains unproven; KITTI trajectory accuracy and drift are still unscored, and process peak RSS was not measured.

Reproduce the corrected outputs into separate ignored directories:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m unittest discover -s tests -v
python -m orb_slam_mono.m4_ablation --mode synthetic --vary-visibility --output-dir artifacts/m4-synthetic-varied-ba-validation-final
python scripts/score_m4_synthetic_offline.py --results-dir artifacts/m4-synthetic-varied-ba-validation-final
python -m orb_slam_mono.m4_ablation --mode kitti --kitti-sequence data/sequences/00 --start 0 --end 299 --output-dir artifacts/m4-kitti-00-ba-validation-final
```

`ba_events.json` reports solver termination, training and external median/p90/maximum pixel errors, validation counts inside and outside the window, and newly over-limit measurements. A budget-terminated intermediate solution is eligible only if all geometric and external gates pass; lack of validation rejects it. `per_frame.jsonl` remains the online pose record. `estimated_poses.json` revises online-valid keyframe poses but keeps the unestimated anchor null in its chronological frame list. `final_keyframes.json` contains every final keyframe pose, including that anchor; segment-map PNGs use these final poses with final points. PLY files contain final points only, in arbitrary segment-local scale. The original `artifacts/m4-kitti-00-final/` and `artifacts/m4-synthetic-varied-final/` are unchanged.

Keyframe insertion waits at least five frames and uses estimated viewpoint displacement, declining tracked coverage, or a ten-frame maximum gap. The active BA window has at most eight keyframes, including the two fixed initialization poses that remove pose and scale gauge freedoms. New points use estimated `T_cw` poses and `K[R|t]` pixel projections in the established segment scale. Descriptor evidence is a bounded set of eight actual ORB descriptors with a deterministic Hamming medoid. BA uses SciPy `least_squares` with sparse Jacobian structure and checks held-out observations before applying any update. See `NOTES.md` for thresholds and ownership rules.

## M3 sequential demonstration

From the project root with CPU NumPy and OpenCV installed:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m orb_slam_mono.sequence_demo --mode synthetic --output-dir artifacts/m3-synthetic-corrected
python -m orb_slam_mono.sequence_demo --mode kitti --kitti-sequence data/sequences/00 --start 0 --end 100 --output-dir artifacts/m3-kitti-00-smoke
python -m orb_slam_mono.sequence_demo --mode kitti --kitti-sequence data/sequences/00 --start 0 --end 299 --output-dir artifacts/m3-kitti-00-corrected
python -m unittest discover -s tests -v
python -m unittest tests.test_m3.StateTests.test_recovery_budget_exhaustion_is_terminal_and_clears_anchor -v
```

The ignored output directory contains `per_frame.csv`, `per_frame.jsonl`, `summary.json`, `initialization_attempts.json`, `loss_events.json`, `recovery_events.json`, `estimated_poses.json`, `tracking_states.png`, `pnp_inliers.png`, `reprojection_px.png`, correspondence views under `visualizations/`, and per-segment `maps/segment-*.ply` plus top-down `maps/segment-*.png`. The CSV has one explicit outcome for every processed frame. `estimated_poses.json` uses world-to-camera `T_cw`, reports camera centers as `-R_cw.T @ t_cw`, and leaves failed or pre-initialization poses null. Separate segment coordinates and scales must not be joined without a separate alignment. Original M3 outputs remain under `artifacts/m3-synthetic/` and `artifacts/m3-kitti-00/`.

Default initialization attempts anchor gaps 3, 5, 7, 9, and 11, then replaces an exhausted anchor. It reuses M2 geometry thresholds and only maps accepted triangulations using original ORB keypoint identities. Tracking matches current ORB features against fixed initial map descriptors and accepts PnP only after RANSAC, inlier, positive-depth, and reprojection checks. A loss records no current pose; recovery starts a new segment. Tune only through `configs/default.json` after controlled validation.

`tracking.max_recovery_frames` (default 60) counts processed frames **after** failed PnP; skipped source indices do not consume the budget. The final permitted frame may search. If still unsuccessful, the next frame enters terminal LOST, clears the anchor, and all later chronological frames remain pose-null without another initialization attempt. Construct a fresh tracker for a separate run. Per-frame output distinguishes association candidates, unique descriptor matches, PnP input and returned inliers, accepted/rejected PnP, and terminal loss. Summary distributions label all tracking attempts, accepted frames, and rejected solver returns separately; missing solver inlier counts remain null.

M3 meets its defined functional and validation criteria. The physical seed-19 synthetic run initializes 0→3 with 229 points and tracks eight later frames. Its offline known-baseline comparison gives median/max rotation error 0.416°/0.603° and position error 0.034/0.072 scene units. A separate synthetic timeout scenario is saved under `artifacts/m3-synthetic-recovery-timeout/`. On official KITTI 00 frames 0–299, the unchanged defaults produce 84 PnP-tracked frames, 13 accepted initialization pairs, 12 losses, 191 initializing frames, and zero recovery timeouts. The corrected run reproduces every original decision and accepted pose. The longest segment has 13 valid poses, showing limited continuity with fixed landmarks and single anchor descriptors. KITTI M3 trajectory accuracy has not been independently evaluated; no ground-truth pose enters this runner. All **62 tests pass**, including the 44 M2 baseline tests.

## Reproduce on Windows PowerShell

From this directory, with Python 3.9 and NumPy/OpenCV available:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m orb_slam_mono.demo --output-dir artifacts/demo
python -m orb_slam_mono.geometry_demo --mode synthetic --output-dir artifacts/geometry-synthetic
python -m orb_slam_mono.geometry_demo --mode synthetic-images --output-dir artifacts/geometry-synthetic-images
python -m orb_slam_mono.robustness_study --label accepted-trial-reproduction --output-dir artifacts/accepted-trial-reproduction/synthetic
python -m unittest discover -s tests -v
```

For a clean Python environment, first create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Use `.\.venv\Scripts\python` in place of `python` after installation. A Conda specification is also provided in `environment.yml`. The original M1 demo still writes `matches.png`, two feature images, and `stats.json`. M2 writes `essential_correspondences.png` (green RANSAC inliers, red others), `pose_correspondences.png` (green pose inliers), `triangulated_correspondences.png` (green accepted points), `sparse_map.png` (oblique view), `landmarks.ply`, `poses.json`, and `stats.json`. `artifacts/` is ignored by Git.

`--mode synthetic` gives the geometry estimator noisy projected 2D correspondences and calibration, then evaluates against held-out truth. `--mode synthetic-images` renders the 3D scene as grayscale images and uses actual ORB extraction and descriptor matching before geometry. The old M1 shifted-image fallback remains a matching demo; it is not used as a physical motion benchmark.

The audit study runs five fixed seeds over eight noise/outlier/baseline conditions and writes `robustness.csv`, `robustness.json`, and `summary.txt`. Both accepted and rejected attempts are included. Synthetic truth is read only after each initialization for evaluation.

## KITTI data

The user-provided KITTI **odometry grayscale** and calibration ZIPs contain sequence 00. The checked subset is frames `000000`–`000299` from rectified left grayscale `image_0`, with matching sequence-00 `calib.txt` and the first 300 original timestamps. Extract only this prefix from the local archives:

```powershell
python scripts/extract_kitti_00_subset.py --gray-archive "$HOME\Downloads\data_odometry_gray.zip" --calib-archive "$HOME\Downloads\data_odometry_calib.zip" --output-dir data/sequences/00 --manifest artifacts/kitti-00-validation/subset_manifest.json --frame-count 300
```

The script checks ZIP CRCs and preserves PNG bytes; existing files must match the archive exactly. The user-supplied official archives remain outside this repository. ZIP CRCs establish archive/extraction integrity, while the user's official-download provenance supplies the source claim. The local ignored layout is:

```text
data/sequences/00/image_0/000000.png
data/sequences/00/image_0/000001.png
...
data/sequences/00/image_0/000299.png
data/sequences/00/calib.txt
data/sequences/00/times.txt  # exactly 300 matching original values
```

The original seven-pair baseline was run before the pose-stability correction and remains unchanged under `artifacts/kitti-00-validation/`; its report/configuration files are also preserved under `artifacts/kitti-00-pose-audit/baseline-original/`. Run all predetermined pairs with the **current** global configuration into a separate output directory:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
$pairs = @(@(0,1), @(0,5), @(10,15), @(20,25), @(40,45), @(60,65), @(80,85))
foreach ($pair in $pairs) {
  $output = 'artifacts/accepted-trial-reproduction/original/pair-{0:D6}-{1:D6}' -f $pair[0], $pair[1]
  python -m orb_slam_mono.geometry_demo --mode kitti --kitti-sequence data/sequences/00 --first-index $pair[0] --second-index $pair[1] --output-dir $output
}
```

The loader uses only `P0`, `image_0`, and optional timestamps, never ground-truth poses. The original baseline accepted 5/7, including a now-confirmed false acceptance at 20→25. The current pose-stability gate rejects that pair, so current acceptance is 4/7. It repeats the one-shot geometry check on eight fixed permutations of the same matches. Acceptance requires four usable trial rotations, at most 10° median angular disagreement, **and four fully accepted trials**; provisional rotations from rejected trials contribute only to the angular diagnostic. The gate never selects an alternative pose. Results from this correction are in `artifacts/kitti-00-pose-audit/accepted-trial-gate/`, separate from earlier runs. The old public-mirror probe under `artifacts/kitti-source-probe/` was not used.

## Offline KITTI pose-accuracy audit

Extract only sequence-00 truth from the user-provided official poses ZIP, then score the preserved original results. Ground truth stays under ignored `data/ground_truth/`, outside `KittiGraySequence` and `initialize_two_view`:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python scripts/extract_kitti_00_poses.py --archive "$HOME\Downloads\data_odometry_poses.zip" --output data/ground_truth/kitti_odometry/00.txt
python -m orb_slam_mono.kitti_pose_evaluation --ground-truth data/ground_truth/kitti_odometry/00.txt --results-dir artifacts/kitti-00-validation --output-dir artifacts/kitti-00-pose-audit/original-offline --pair-set original
python -m orb_slam_mono.kitti_pose_evaluation --ground-truth data/ground_truth/kitti_odometry/00.txt --results-dir artifacts/kitti-00-pose-audit/original-after --output-dir artifacts/kitti-00-pose-audit/original-after-offline --pair-set original
python scripts/audit_kitti_correspondences.py --sequence data/sequences/00 --report artifacts/kitti-00-validation/pair-000020-000025/stats.json --config configs/default.json --output-dir artifacts/kitti-00-pose-audit/anomaly-20-25
```

For fixed held-out checks, run pairs 100→105, 120→125, 140→145, 180→185, and 220→225 through `geometry_demo` with the same configuration into `artifacts/kitti-00-pose-audit/heldout-after/pair-.../`, then use `--pair-set heldout` in the offline evaluator. The original 20→25 estimate has **42.18° rotation error and 110.86° translation-direction error**. The same match list in different orders usually yields a roughly 1° rotation with comparable or better essential support, so the added stability gate rejects the original 41.82° pose. The other four original accepted estimates retain 0.27–0.75° rotation error and 0.67–5.08° direction error. Held-out acceptance remains 1/5; the accepted pair has 0.069°/2.023° errors. Full vectors, baselines, and inlier/point counts are in the ignored offline CSV/JSON reports. See `STATUS.md` and `NOTES.md` for the complete evidence and limitations.

Each pair folder contains `stats.json`, highlighted ORB correspondence PNGs, a sparse-map PNG, `landmarks.ply`, and `poses.json`. Rejected exports have no usable map or second-camera pose; rejected visualizations are explicitly labeled. KITTI odometry images are rectified; another camera or unrectified images require distortion handling.

## Interpreting the result

The original M1 `stats.json` fields and their meanings are unchanged. In M2, `retained_matches` remains a descriptor-filter count; `essential_inliers` counts RANSAC inliers; `pose_inliers` counts those retained by cheirality-based pose recovery; `accepted_points` counts triangulations surviving numerical, positive-depth, parallax, and reprojection gates. A rejected initialization has `initialization_status: rejected` and a `failure_reason`; stages not reached use `null` counts. `synthetic` mode has `null` ORB fields because it tests geometry on point correspondences directly.

`T_cw` maps world to camera: `X_c = R_cw X_w + t_cw`. Camera 1 defines the world origin. `R_21` and `t_21_unit` are camera 2 relative to camera 1; `C2 = -R_21.T @ t_21_unit`. `landmarks.ply` uses camera-1 world coordinates, and `poses.json` records both world-to-camera matrices. One translation baseline is defined as one arbitrary unit; PLY/JSON output is **not metric**. Synthetic ground truth is used only after estimation to score rotation, direction, and an explicitly offline aligned 3D error.

Initialization also requires pose cheirality support from at least 90% of essential inliers, a median rotation-only fit residual above 1.5 px, and pose stability across permuted match order. On rejection, `R_21`, `t_21_unit`, and camera-2 pose are `null`, the public map is empty, and PLY contains zero vertices. `diagnostics.accepted_points` remains the **provisional triangulation-filter survivor count**; `diagnostics.usable_map_points` is zero. Rejected PNGs are labeled diagnostic only. A rerun replaces the demo's own artifact files so an earlier successful map does not linger.

The accepted-trial minimum is configured as `geometry.min_pose_stability_accepted_trials` (default 4). `pose_stability_valid_candidates` counts all trial rotations, `pose_stability_provisional_candidates` counts rotations from rejected trials, and `pose_stability_accepted_trials` counts full successes. `insufficient_accepted_pose_stability_trials` means the angle statistic passed but too few trials produced a usable initialization. Set `pose_stability_trials` to 0 only for a controlled one-shot comparison. This stricter gate leaves the fixed KITTI decisions unchanged. Two earlier synthetic acceptance expectations were corrected after truth-based review: small translation with moderate rotation (1.212°/10.065° one-shot rotation/direction errors; 2/8 accepted trials) and a planar scene (3.798°/24.258°; 1/8) are conservatively rejected. The full suite passes **44/44 tests**. At M2 closure, its geometry was ready for M3; the subsequently implemented sequence-level initializer waits for sufficient motion and retries another image pair without changing M2 geometry.

KITTI pose-file rows are camera-to-first-camera/world `T_wc`. Offline scoring forms `T_cj_ci = inverse(T_wc_j) @ T_wc_i`, then compares rotation and translation **direction** to the estimate. True baseline magnitude is reported for evaluation only; two-view monocular output remains arbitrary-scale. No two-view result measures sequence tracking or drift.

The offline 3D metric is an **origin-fixed, scale-only** fit, not Umeyama Sim(3). The report includes its fitted scale, median/p90 3D errors and depth distributions, plus a separate true-pose triangulation reference using the same noisy synthetic observations. None of those truth-based quantities affects initialization or the exported map. The rendered-image ORB mode has no known match-to-landmark identity, so its 3D truth-alignment fields are `null`.

See `PROJECT_PLAN.md`, `NOTES.md`, `REFERENCES.md`, and `STATUS.md` for scope, decisions, sources, and the next task.
