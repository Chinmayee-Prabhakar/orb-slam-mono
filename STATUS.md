# Status

## Current milestone

M2 — calibrated two-view initialization: **technical M2 acceptance complete for the defined tests and fixed validation sets**. The accepted-trial gate remains conservative; two outdated synthetic acceptance expectations were corrected after independent truth-based review. All 44 tests pass, and pair 20→25 remains rejected. The implementation is ready to support a bounded M3 proposal, but M3 has not started.

## M2 — test-expectation review and closure

- In OpenCV 4.8.1, the fixed small-translation/moderate-rotation scene has a one-shot estimate with 1.212° rotation and 10.065° translation-direction error, yet only 2/8 permutation trials pass full initialization; six provide provisional rotations. One rejected trial's provisional direction has 172.056° error. The fixed planar scene has 3.798°/24.258° one-shot rotation/direction errors, only 1/8 fully accepted trials, and seven provisional rotations; their direction errors are about 85–91°. Both median rotation disagreements are below 10° (1.916° and 8.232°), so the 4/8 accepted-trial requirement rejects them as `insufficient_accepted_pose_stability_trials`. The planar one-shot direction error is substantial; the small-translation pose is reasonably close but unreliable across trials. Truth was used only after estimation in tests.
- Updated the two expectations in `tests/test_pose_stability.py` from acceptance to conservative rejection, retained numerical one-shot truth checks, checked the accepted/provisional trial diagnostics, verified hazardous rejected-trial directions, and asserted no usable pose/map on rejection. Large rotation, forward/lateral motion, near-planar acceptance, and nonuniform-feature false-acceptance controls remain unchanged. The full suite is **44/44 passed, 0 failed, 0 skipped** using `python -m unittest discover -s tests -v`.
- Fresh 40-case synthetic study: clean, nominal, 1 px noise, 2 px noise, and 30% incorrect matches each accept 5/5; weak baseline, very weak baseline, and pure rotation each accept 0/5. No accepted run has >45° direction error. M1 demo remains at 966 retained matches; M2 geometry and rendered-image ORB demos accept 427 and 260 points. New outputs are under `artifacts/kitti-00-pose-audit/final-m2-closure/`.
- Re-ran all 12 fixed KITTI pairs with the unchanged shared defaults in separate output folders. Original decisions remain 0→1 reject, 0→5 accept, 10→15 accept, 20→25 reject, 40→45 reject, 60→65 accept, 80→85 accept (**4/7**). Held-out decisions remain 100→105 accept and 120→125, 140→145, 180→185, 220→225 reject (**1/5**). Prior offline pose-accuracy and before/after outputs remain untouched. Accepted fixed KITTI poses retain their previously measured offline errors; the original 20→25 false acceptance is no longer exposed as a usable pose.
- Interpretation: the well-conditioned synthetic scene and five current accepted fixed KITTI pairs exemplify successful initialization; the small-translation case is a conservative rejection of a moderately accurate one-shot estimate; the planar case, nonuniform-feature counterexample, and KITTI 20→25 show rejection of inaccurate or unstable estimates. The original pre-correction 20→25 output remains a documented historical false acceptance. No current false acceptance was found in these fixed tests, which is not a universal guarantee. Future sequence tracking may wait for sufficient motion and try another pair; that behavior belongs to M3, not this correction.

## M2 — interim accepted-trial stability correction and validation

- The prior gate counted provisional rotations from rejected permutation trials when computing angular disagreement but did not require any trial to pass full initialization. It could accept an original pose when all eight trials rejected yet their provisional rotations agreed. The gate now requires at least 4/8 **fully accepted** trials in addition to 4 usable rotations and median disagreement ≤10°. `pose_stability_valid_candidates` includes accepted and provisional rotations; `pose_stability_provisional_candidates` counts rejected trials that supplied a diagnostic rotation; `pose_stability_accepted_trials` counts full one-shot successes. If only the accepted-trial minimum fails, the reason is `insufficient_accepted_pose_stability_trials`. Rejection clears public pose/map; `pose_stability_trials=0` keeps one-shot behavior. No truth enters the gate.
- Mocked tests confirm 0 and 3 accepted trials reject despite eight agreeing rotations; 4 accepted trials preserve original pose, landmarks, and masks; eight accepted trials with 30° disagreement reject. Invalid accepted-trial minima reject. Baseline before this change was 41/41 tests passing; this interim suite ran **44 tests, 42 passed and 2 failed, 0 skipped**. At that stage, `small_translation_moderate_rotation` and `planar` assertions still expected acceptance, but the gate rejected with `insufficient_accepted_pose_stability_trials`: only 2/8 and 1/8 trials respectively passed full initialization. Their rotations remained available provisionally (median disagreements 1.92° and 8.23°). The subsequent truth-based review and passing tests are recorded above.
- Re-ran all seven original and five held-out KITTI pairs into `artifacts/kitti-00-pose-audit/accepted-trial-gate/`, preserving prior outputs. Original decisions remain: 0→1 cheirality reject; 0→5 accept (8/8 trials); 10→15 accept (7/8); 20→25 unstable-pose reject (1/8, 42.36° disagreement); 40→45 parallax reject; 60→65 accept (7/8); 80→85 accept (7/8). Held-out decisions remain: 100→105 accept (8/8); 120→125 cheirality reject; 140→145, 180→185, and 220→225 parallax reject. Offline pose errors for accepted pairs are unchanged. New results, pair visualizations, and offline CSV/JSON are in separate folders under that root.
- The fixed 40-run synthetic grid now gives 5/5 acceptance in each clean, nominal, 1 px noise, 2 px noise, and 30% incorrect-match group; weak baseline falls from 2/5 to **0/5**; very weak baseline and pure rotation stay 0/5. M1 demo remains 1500/1500 features and 966 matches; M2 point-correspondence and rendered-image demos remain accepted with 427 and 260 points. The valid 20° rotation, forward and lateral motion, and near-planar tests pass; the small-translation and planar expectations remain unresolved. On the four accepted original KITTI pairs, median stability-stage runtime was 187.60 ms before and 203.18 ms in this rerun; this 15.58 ms difference is a noisy cross-run comparison, not an algorithmic timing guarantee.
- The new minimum was not lowered to preserve prior acceptances: doing so would weaken the specific reliability condition under review. The subsequent review found the two expectations outdated and closed this issue as described above. No Git index or history changes were made.

## M2 — official pose audit and bounded reliability correction

- The user supplied `data_odometry_poses.zip` from the [official KITTI odometry benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). Extracted only `dataset/poses/00.txt` to ignored `data/ground_truth/kitti_odometry/00.txt`. The ZIP member is 728,871 bytes, CRC-32 `2e4f2dcb`, SHA-256 `90791a4113df979b149fa9e1104e960ea59f525a8318a202dbb6aec1a3d88793`. It has 4,541 finite 12-number rows, matching sequence 00's 4,541 frames; the first pose is identity to rounding precision. Rotation determinants span approximately 0.99999980–1.00000011, with maximum orthogonality residual about 2.32×10⁻⁷. The supplied official archive is the provenance claim; ZIP CRC verifies extraction integrity.
- KITTI rows are camera-to-first-camera/world `T_wc`. For images `i,j`, offline evaluation computes `T_cj_ci = inverse(T_wc_j) @ T_wc_i`, then compares its `R` and **unit translation direction** against the estimator's world-to-camera `R_21,t_21_unit`. The true baseline is reported only as an offline reference; no true magnitude, pose, or direction enters `initialize_two_view`. All seven true baselines exceed 0.86 m, so their direction comparisons are numerically meaningful under the 0.05 m reporting cutoff. Composition and sign/frame convention have dedicated tests.
- The prior seven-pair `stats.json` files, CSV/JSON summary, and original configuration were copied byte-for-byte into ignored `artifacts/kitti-00-pose-audit/baseline-original/` before changing geometry. The original outcomes below are therefore retained. A dash means no accepted estimate exists; rejected pairs have no public pose to score. The full offline CSV/JSON includes estimated and true direction vectors, inlier counts, parallax, reprojection errors, map-point counts, and true baseline.

| Pair | Original result | Estimated / true rotation (°) | Rotation error (°) | Direction error (°) | True baseline (m, offline) | After gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0→1 | rejected: cheirality | — / 0.140 | — | — | 0.860 | rejected: cheirality |
| 0→5 | accepted | 1.378 / 0.695 | 0.752 | 4.231 | 4.300 | accepted |
| 10→15 | accepted | 0.679 / 0.640 | 0.494 | 1.133 | 4.295 | accepted |
| 20→25 | accepted | 41.816 / 0.391 | **42.180** | **110.857** | 4.582 | rejected: unstable pose |
| 40→45 | rejected: parallax | — / 0.901 | — | — | 5.190 | rejected: parallax |
| 60→65 | accepted | 0.273 / 0.304 | 0.268 | 0.666 | 4.754 | accepted |
| 80→85 | accepted | 0.816 / 0.467 | 0.465 | 5.075 | 3.286 | accepted |

- Original 20→25 E RANSAC returned one 3×3 candidate with 79/102 inliers; `recoverPose` kept 79/79 and triangulation kept 79, with 0.303/0.329 px median reprojection error. However, 75/79 essential inliers lie in one 4×3 image-grid cell. Inlier flow magnitude has p10/median/p90 1.73/4.55/46.38 px. A least-squares rotation-only bearing fit is 2.21° (10.65 px median residual); a robust fit from small bearing subsets is 0.17° (4.47 px median residual). Neither fit is treated as ground truth or a direct pose replacement. The original 41.53° inferred parallax is computed using the wrong 41.82° rotation, so it cannot validate the physical baseline.
- Five OpenCV RNG seeds with the original match order reproduce the same wrong 41.82° model. Eight fixed match-order permutations produce provisional rotations of 0.73–1.78°, each 42.01–42.80° from the original, with similarly strong 81–86 essential inliers. One permuted solution passes the old gates with 84 essential/81 pose inliers and has 1.05° rotation and 5.36° direction error offline. The strongest supported explanation is an order-sensitive essential-model choice on spatially concentrated and mixed-motion/ambiguous correspondences; the available images do not prove a particular moving object caused it. `artifacts/kitti-00-pose-audit/anomaly-20-25/` contains the reproducible numeric audit and annotated inlier-flow image.
- The **earlier pose-stability gate** repeated the one-shot initializer on 8 fixed permutations of the same matches, observed accepted or explicitly provisional rotations, and required at least 4 valid candidates plus median angular disagreement ≤10° from the original pose. It used no ground truth, did not choose a better pose from the trials, and ran only after the original geometry gates passed. For 20→25, disagreement was 42.36° and the result rejected as `unstable_pose_under_resampling`; public pose/map were cleared, PLY had zero vertices, and `poses.json` had no second-camera pose. The other six original outcomes were unchanged: **before 5/7 accepted, after 4/7 accepted**. The four remaining original accepted pairs retained their original offline pose errors. Median stability disagreement on them was 0.29–0.85°. Additional CPU cost was approximately 0.10–0.43 s in those runs. The accepted-trial requirement was added afterward, as documented above.
- Five held-out pairs were fixed separately: 100→105 stays accepted (true/estimated rotation 15.242°/15.267°, 0.069° rotation and 2.023° direction error, 139 points); 120→125 rejects for low cheirality fraction; 140→145, 180→185, and 220→225 reject for low parallax. **Held-out acceptance is 1/5 before and after.** Their true baselines are 2.018–3.531 m, so the rejected cases deserve later sensitivity analysis; no threshold was relaxed to force acceptance. Full before/after outputs and offline evaluations are under `artifacts/kitti-00-pose-audit/`.
- The earlier 40-case fixed synthetic grid with the angular-only gate gave 5/5 acceptance in clean, nominal, 1 px noise, 2 px noise, and 30% incorrect-match groups; weak baseline gave 2/5, very weak baseline 0/5, pure rotation 0/5. Controlled synthetic tests at that stage kept a legitimate 20° rotation, small translation plus moderate rotation, forward and lateral motion, planar, and near-planar scenes accepted; a spatially concentrated scene with an 18.36° original rotation error rejected. The new accepted-trial requirement changes the small-translation, planar, and weak-baseline results as recorded above.
- Baseline before this audit: **35/35 tests passed**. After the offline convention and stability/counterexample tests, **41/41 passed**. The M1 demo still produces 1500/1500 features and 966 retained matches; M2 synthetic geometry still accepts 427 points with 1.246°/1.842° rotation/direction errors; rendered-image ORB still accepts 260 points with 0.165°/0.686° errors. No M3 code or dependency was added.

Offline reproduction from the project root:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m orb_slam_mono.kitti_pose_evaluation --ground-truth data/ground_truth/kitti_odometry/00.txt --results-dir artifacts/kitti-00-validation --output-dir artifacts/kitti-00-pose-audit/original-offline --pair-set original
python -m orb_slam_mono.kitti_pose_evaluation --ground-truth data/ground_truth/kitti_odometry/00.txt --results-dir artifacts/kitti-00-pose-audit/heldout-after --output-dir artifacts/kitti-00-pose-audit/heldout-after-offline --pair-set heldout
python scripts/audit_kitti_correspondences.py --sequence data/sequences/00 --report artifacts/kitti-00-validation/pair-000020-000025/stats.json --config configs/default.json --output-dir artifacts/kitti-00-pose-audit/anomaly-20-25
python -m orb_slam_mono.robustness_study --label pose-stability-after --output-dir artifacts/kitti-00-pose-audit/synthetic-after
python -m unittest discover -s tests -v
```

## M2 — original KITTI sequence-00 validation baseline

- The user-provided `data_odometry_gray.zip` (23,166,524,501 bytes) and separate `data_odometry_calib.zip` (600,238 bytes) opened successfully. The grayscale archive contains 4,541 sequence-00 `image_0` PNGs, `calib.txt`, and `times.txt`; the separate ZIP has the matching sequence-00 calibration and a byte-identical full timestamp file. The `P0` entries match numerically. This is provenance from the user's official download; ZIP CRCs verify internal and extracted-byte integrity, not an independent external signature.
- Extracted only original PNG bytes `000000.png`–`000299.png` (300 continuous frames, 76,288,708 image bytes) into ignored `data/sequences/00/image_0/`, plus matching `calib.txt` and the first 300 original timestamp lines (0.0–31.00138 seconds). Every selected PNG passed ZIP CRC and extracted-byte checks. All decode as 1241×376 `uint8` grayscale. `KittiGraySequence` loads all 300 with monotonic timestamps and matching positional indices. `P0` gives `fx=fy=718.856`, `cx=607.1928`, `cy=185.2157`; no pose or depth file entered the estimator. `artifacts/kitti-00-validation/subset_manifest.json` records per-image CRC/SHA-256. The old unverified mirror probe was not used.
- Original `configs/default.json` thresholds were unchanged. CSV and JSON summaries are under `artifacts/kitti-00-validation/`; each pair folder contains the ORB/essential and pose/triangulation correspondence views, map view, PLY, poses JSON, and full statistics. A dash means the stage was not reached and is `null` in JSON. Reprojection entries are view 1/view 2 medians in pixels.

| Pair | ORB features | Retained matches | Essential inliers (%) | Pose / essential | Rotation-only residual (px) | Median parallax (°) | Accepted points | Median reprojection (px) | Result | Runtime (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 0→1 | 1500/1500 | 643 | 575 (89.42%) | 424/575 (0.737) | 3.88 | — | — | — | rejected: `low_cheirality_fraction` | 2394.5 |
| 0→5 | 1500/1500 | 131 | 109 (83.21%) | 109/109 (1.000) | 10.56 | 1.34 | 84 | 0.30/0.33 | accepted | 1395.7 |
| 10→15 | 1500/1500 | 156 | 135 (86.54%) | 126/135 (0.933) | 19.11 | 1.52 | 91 | 0.39/0.46 | accepted | 1749.2 |
| 20→25 | 1500/1500 | 102 | 79 (77.45%) | 79/79 (1.000) | 10.65 | 41.53 | 79 | 0.30/0.33 | accepted; suspect pose | 1210.4 |
| 40→45 | 1500/1500 | 179 | 148 (82.68%) | 146/148 (0.986) | 13.96 | 0.98 | — | — | rejected: `low_parallax` | 1177.0 |
| 60→65 | 1500/1500 | 199 | 177 (88.94%) | 160/177 (0.904) | 21.17 | 1.05 | 130 | 0.26/0.30 | accepted | 1206.3 |
| 80→85 | 1500/1500 | 375 | 324 (86.40%) | 323/324 (0.997) | 26.30 | 2.38 | 313 | 0.39/0.45 | accepted | 1188.5 |

- Gate outcomes: **5/7 accepted (71.43%)**, **2/7 rejected (28.57%)**; rejection reasons are one low cheirality fraction and one low parallax. These are two-view outcomes, not trajectory or map accuracy. Median retained matches are only 102–375 for five-frame gaps despite 1500 ORB features per image; real texture, repeated structures, and limited overlap matter. The 0→1 pair retains many matches and essential inliers but only 73.7% pose cheirality support, demonstrating why inlier count alone is insufficient. The 40→45 pair has 0.976° parallax, just below the unchanged 1.0° median gate; do not relax it based on this pair alone.
- **Original 20→25 false acceptance:** estimated rotation is 41.82° and inferred median parallax 41.53°, while median raw displacement among its essential inliers is only 4.55 px. Its median exported depth is 1.45 arbitrary baselines, versus roughly 8–9 for the other accepted five-frame pairs. Reprojection of 0.30/0.33 px does not resolve this disagreement because pose and depths can jointly fit the same correspondences. Ground truth was not available during this initial experiment; its later offline evaluation above confirms the error. The original accepted classification remains in the preserved baseline files.
- Re-ran the original M1 demo (1500/1500 features, 966 retained matches), geometry-only synthetic demo (427 accepted points; 1.246°/1.842° rotation/direction errors), and rendered-image ORB demo (260 accepted points; 0.165°/0.686° errors). The original **34/34 tests** passed before changes; after adding an optional real-KITTI smoke test, **35/35 passed** with the local subset. The optional test skips when KITTI is absent. CPU OpenCV/NumPy only; no GPU/Blender dependencies were installed.

Reproduce the original configuration from the project root without overwriting its saved results:

```powershell
python scripts/extract_kitti_00_subset.py --gray-archive "$HOME\Downloads\data_odometry_gray.zip" --calib-archive "$HOME\Downloads\data_odometry_calib.zip" --output-dir data/sequences/00 --manifest artifacts/kitti-00-validation/subset_manifest.json --frame-count 300
$env:PYTHONPATH = (Resolve-Path .\src).Path
$pairs = @(@(0,1), @(0,5), @(10,15), @(20,25), @(40,45), @(60,65), @(80,85))
foreach ($pair in $pairs) {
  $output = 'artifacts/kitti-00-pose-audit/original-baseline-reproduction/pair-{0:D6}-{1:D6}' -f $pair[0], $pair[1]
  python -m orb_slam_mono.geometry_demo --mode kitti --kitti-sequence data/sequences/00 --first-index $pair[0] --second-index $pair[1] --config artifacts/kitti-00-pose-audit/baseline-original/default.json --output-dir $output
}
python scripts/summarize_kitti_00.py --results-dir artifacts/kitti-00-pose-audit/original-baseline-reproduction
python -m unittest discover -s tests -v
```

## M2 — earlier genuine KITTI acquisition check

- Baseline on the committed M2 correction checkpoint `1b31c510ee18bc381f332d385d5946614ed1ac20`: **34/34 tests passed** before any validation changes. Documents, Downloads, and D: contained no local KITTI odometry images or calibration.
- The [official KITTI odometry page](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) lists a roughly 22 GB grayscale archive and a separate calibration download. Its [download policy](https://www.cvlibs.net/datasets/kitti/user_login.php) requires registration and a stated usage purpose. No full archive was downloaded.
- A public [sequence-00-labeled mirror](https://huggingface.co/datasets/yujie2696/kitti_odometry_00) was investigated with one PNG, `calib.txt`, and `times.txt`, saved only under ignored `artifacts/kitti-source-probe/`. The PNG decodes as 1241×376, 8-bit grayscale, and the mirrored `P0` is `fx=fy=718.856`, `cx=607.1928`, `cy=185.2157`. Its 1,101-line timestamps file does not cover the documented 4,541-frame sequence 00, and the mirror has no dataset card establishing its source or unchanged image bytes. The probe was **not** used as validated KITTI input.
- At that check, no real image pair had been attempted; all seven fixed pairs remained pending, and no KITTI result was claimed. The synthetic KITTI-layout test was not real-data validation. The user's subsequent official ZIP download resolved the data blocker, and the results are now recorded above.
- The earlier minimum acquisition target was original frames `000000.png` through `000100.png` with matching sequence-00 `calib.txt` and timestamp prefix. The official subset now extends through `000299.png`.
- No algorithm, loader, configuration, or test code changed in this acquisition check. No Git files were staged or committed.

## M2 correctness audit and checkpoint

- Baseline before edits: clean `main` at `7154988320bf3dd23355908aea5eb367f5e818ec`; 27/27 tests passed. Reproduced the independent-review seed-7, 0.15-baseline false acceptance on OpenCV 4.8.1: 405 essential inliers, 284 pose inliers, 284 provisional points, 173.466° translation-direction error, 1.373° estimated parallax, and about 0.330 px reprojection error. A separate pure-rotation seed 23 was also falsely accepted.
- Rejected initializations now have no public `R_21`, `t_21`, points, or accepted-point mask. Provisional counts and an explicitly invalid debug pose remain for diagnosis. PLY, poses JSON, and PNGs identify rejection and contain no usable camera-2/map geometry. A rejected rerun replaces stale demo-owned successful artifacts without touching unrelated files. Added a specific `insufficient_triangulated_points` reason for a high required-count case.
- Two conservative quality gates were added: pose/essential cheirality fraction ≥0.9 and best-fit rotation-only median residual ≥1.5 px. These use 2D observations and calibration only. The reviewed reversed-translation case now rejects as `low_cheirality_fraction`; pure rotation seed 23 rejects as `rotation_only_explains_matches`.
- Fixed 40-run synthetic study (five seeds × eight conditions), same config throughout: clean, nominal, 1 px noise, 2 px noise, and 30% wrong-match groups each remain 5/5 accepted. Weak baseline changes 4/5 → 2/5 accepted, very weak baseline 1/5 → 0/5, pure rotation 1/5 → 0/5. Severe accepted direction errors (>45°) change 1 → 0. Complete pre/post CSV, JSON, and summaries are ignored under `artifacts/m2-audit-before/` and `artifacts/m2-audit-after/`.
- Final tests: **34/34 passed**, retaining all prior 27. The added test exercises every triangulation filter stage with counts 8→7→5→4→3→1, source indexing, rejected provisional geometry, stale artifact replacement, reviewed low baseline, pure rotation, offline scale evaluation, and nominal-seed acceptance. The original M1 demo still gives 1500/1500 features and 966 retained matches. Geometry-only M2 still accepts 427 points; rendered-image ORB M2 still accepts 260. The latter uses actual ORB matches, not truth correspondence.
- Offline seed-7 scale-only fit is 0.58469 scene units per estimated unit (not Sim(3)); median/p90 3D errors are 0.56524/0.81074 scene units. True-pose triangulation of the same noisy observations gives 0.03271/0.11305. These truth-based references never enter initialization. See `NOTES.md` for depth/error strata and interpretation.
- A scoped check at the correction checkpoint found no local KITTI data. The later official-ZIP extraction and validation are recorded above. The synthetic KITTI-layout and missing-data CLI tests still pass.
- The approved M2 corrections are included in one local checkpoint. Generated artifacts remain ignored; no remote was created or pushed.

Audit reproduction from the project root:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m unittest discover -s tests -v
python -m orb_slam_mono.demo --output-dir artifacts/demo
python -m orb_slam_mono.geometry_demo --mode synthetic --output-dir artifacts/geometry-synthetic
python -m orb_slam_mono.geometry_demo --mode synthetic-images --output-dir artifacts/geometry-synthetic-images
python -m orb_slam_mono.robustness_study --label original-rerun --config artifacts/kitti-00-pose-audit/baseline-original/default.json --output-dir artifacts/kitti-00-pose-audit/synthetic-original-rerun
```

## Milestone history and Git

- M0/M1 established the independent Python package, KITTI grayscale/P0 loader, deterministic image-pair fallback, ORB, matching, visualization, diagnostics, and 10 passing tests. The M1 shifted-image fallback is not physical 3D motion.
- M1 is preserved in the local `main` history. The user confirmed the Git identity Chinmayee Prabhakar <prabhakar.chinmayee@gmail.com>. No remote was configured or pushed.
- M2 adds physical synthetic 3D observations, essential RANSAC, relative pose, triangulation, conservative rejection, an ORB-to-geometry CLI, diagnostics, and PLY/JSON/PNG output. The earlier M2 implementation is preserved in local commit `7154988320bf3dd23355908aea5eb367f5e818ec`; the correction checkpoint follows it. No remote/push was requested.

## Verification and example results

- Before M2 edits: all original **10/10 M1 tests passed**.
- After M2 edits: **27/27 tests passed**, including the original ten, geometry correctness/failure cases, M1 regressions, the rendered-image ORB path, and a synthetic KITTI-layout fixture. The original M1 demo was run without changing its code and again produced 1500/1500 features, 1005 ratio-passed, and 966 retained matches.
- Geometry-only seed-7 benchmark: 450 visible points, 0.35 px noise, 0.8 scene-unit baseline, 3° nominal rotation; 427 essential inliers, 427 pose inliers, 427 accepted points; 1.246° rotation error, 1.842° direction error, 4.039° median parallax, 0.261/0.261 px median reprojection error. Synthetic truth was used only after estimation. Offline single-scale aligned median 3D error: 0.565 scene units.
- Rendered 3D image pair via ORB: 1500/1500 features, 348 retained descriptor matches, 261 essential inliers, 260 pose inliers and accepted points; 0.165° rotation error, 0.686° direction error. These rendered sprites test wiring, not natural-scene performance.
- Pure rotation, 0.005 scene-unit baseline, and 85% incorrect correspondences reject in the synthetic tests. Real KITTI fixed-pair outcomes are reported in the current validation section above.

## Reproduce from project root in PowerShell

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m unittest discover -s tests -v
python -m orb_slam_mono.demo --output-dir artifacts/demo
python -m orb_slam_mono.geometry_demo --mode synthetic --output-dir artifacts/geometry-synthetic
python -m orb_slam_mono.geometry_demo --mode synthetic-images --output-dir artifacts/geometry-synthetic-images
```

M2 folders contain `stats.json`, essential and accepted correspondence PNGs, an oblique sparse-map PNG, `landmarks.ply`, and `poses.json`. Generated artifacts are ignored by Git.

The official sequence-00 subset is installed. The fixed-pair commands and summary procedure appear above; a single-pair invocation is:

```powershell
python -m orb_slam_mono.geometry_demo --mode kitti --kitti-sequence data/sequences/00 --first-index 0 --second-index 5 --output-dir artifacts/geometry-kitti-00-000000-000005
```

## Environment and limitations

- Local Python 3.9.0 has OpenCV 4.8.1, NumPy 1.26.1, SciPy 1.11.3. `pytest` is absent, so tests use standard-library `unittest`. M2 uses CPU OpenCV/NumPy; no GPU package was installed.
- The user specifies Ryzen 7000-series, Radeon integrated graphics, and RTX 3050 Ti Laptop GPU. This execution environment reports Ryzen 7 4800H instead; confirm the intended host before hardware-specific choices. `nvidia-smi` reports RTX 3050 Ti, 4096 MiB VRAM, driver 566.07, CUDA compatibility 12.7; that does not establish an installed CUDA Toolkit.
- Blender 3.5.1 is installed at `C:\Program Files\Blender Foundation\Blender 3.5\blender.exe`. A later optional Blender importer may use OptiX after verification, with CPU fallback. The current PLY/JSON export is independent of Blender.
- The official KITTI sequence-00 grayscale subset is installed and tested. TUM data remains unavailable. The original seven-pair M2 run measured initialization behavior on real rectified images; the later offline truth comparison above quantifies its pose accuracy and confirms the 20→25 false acceptance.
- Monocular scale is arbitrary and may drift in later sequence tracking. Unit-baseline M2 points are not metric; no ground-truth pose/depth enters estimation. Pose errors cause 3D depth error even when reprojection errors are small. Homography dominance is deliberately conservative but may reject a legitimate planar, very low-parallax pair.
- No persistent map, PnP tracking, keyframes, bundle adjustment, trajectory estimation, or loop closure exists.

## Next recommended task

Propose a bounded M3 map-based tracking milestone with explicit observation identities, calibrated PnP, tracking states, and initialization retry after sufficient motion. The conservative held-out rejection rate remains a useful M3 failure-analysis target; do not use truth to tune per-pair thresholds. M3 has not started. Any future commit requires separate explicit approval and newly supplied author and committer dates with timezone; do not reuse a previous authorization.
