# Project plan — orb-slam-mono

## Goal and scope

Build an original, inspectable monocular ORB feature-based visual localization and sparse mapping pipeline for GPS-denied navigation research. The eventual data flow is image → ORB features → descriptor/geometric filtering → monocular initialization → map points → map-based PnP tracking → keyframes/map maintenance → local bundle adjustment → trajectory/map output → offline evaluation and failure analysis. Loop closure is optional.

**Scientific invariants:** KITTI ground-truth poses never enter online estimation, map initialization, scale selection, or parameter tuning per frame. Monocular translation and reconstruction have unknown global scale, with scale drift expected over time. Ground truth may enter only an offline evaluation module; use Umeyama Sim(3) alignment there for ATE and explicitly report the alignment, raw scale behavior, and drift. A successful descriptor match is not a geometric inlier.

Start development on a few hundred frames of a KITTI odometry training subsequence (suggested: sequence 00, frames 0–299), then try other short subsequences. Keep the first test data slice and parameter set fixed for comparisons; do not use the full sequences as the first target.

## Milestones and acceptance criteria

### M0 — repository and reproducibility (complete)

- Independent Python package, requirements, Conda environment file, configuration, logging, tests, and project documentation exist.
- The external ROS repository is recorded as context, not imported or copied.
- Environment and local data availability are recorded in `STATUS.md`.

### M1 — calibrated image pair, ORB, descriptor matching (complete)

- Load two consecutive rectified grayscale images from a KITTI odometry `image_0` directory and parse `P0` intrinsics from `calib.txt`; validate malformed calibration, image size changes, and missing images.
- Run the same feature/matcher path on a deterministic synthetic two-image fallback without KITTI.
- Save separate feature views, a match visualization, and `stats.json` with feature counts, KNN pairs, ratio-passed and retained matches. Future geometry fields remain explicitly unknown.
- Unit tests cover calibration, image loading, empty descriptors, clear matches, and mutual filtering; demo runs without a GUI. Synthetic smoke run produces at least 100 features per image and 30 retained matches. These counts verify software wiring only.

### M2 — two-view geometry and monocular initialization (next; not implemented)

- Add inspectable normalized point handling, essential-matrix RANSAC, `recoverPose`, cheirality/parallax checks, and triangulation with reprojection/depth filtering.
- On a controlled synthetic 3D scene with known calibration, recover rotation and translation **direction** within stated tolerances; do not claim translation magnitude. Reject pure rotation/low-parallax initialization and report the reason.
- On a fixed short KITTI slice, log candidate matches, essential-matrix inliers, parallax, accepted 3D points, and rejected initialization attempts; produce an initialization visualization without ground-truth input.

### M3 — map-based tracking and explicit state (planned)

- Introduce `Frame`, `Keyframe`, `MapPoint`, and `Map` when their invariants and ownership become necessary. Record 2D observations and map-point identity explicitly.
- Track map points in subsequent images, solve calibrated PnP with RANSAC, and distinguish `initializing`, `tracking`, and `lost` states. Do not substitute ground-truth motion for failed tracking.
- On the fixed few-hundred-frame slice, export a pose attempt or explicit failure for every frame; log PnP correspondences/inliers and failure reason. Tests cover observation consistency and lost/recovery state transitions.

### M4 — keyframes, map management, local optimization (planned)

- Add documented keyframe insertion/culling rules and map-point creation, observation updates, outlier removal, and culling.
- Optimize a local sliding window of approximately **5–10 keyframes** using `scipy.optimize.least_squares` with a sparse Jacobian structure; fix gauge freedoms and keep nonlocal states outside the active variable block.
- A synthetic optimization test must reduce median reprojection error without increasing it for accepted observations. Assert active keyframe count stays within the configured window, and report runtime and memory on the fixed KITTI slice.

### M5 — trajectory/map output and failure analysis (planned)

- Export an unscaled estimated trajectory and sparse map, plus per-frame diagnostics: detected features, descriptor matches, essential-matrix inliers, PnP inliers, reprojection error, tracking state, and failure reason.
- In a separate offline evaluator, use Umeyama Sim(3) to align estimated and training-set ground-truth trajectories for ATE. Report aligned ATE, unaligned trajectory, fitted scale, per-segment scale drift, lost-frame rate, and diagnostic timelines. Never feed the fitted transform or ground truth back into online tracking.
- Reproduce plots/metrics for at least two fixed short KITTI subsequences, including one failure case. Record exact data ranges and configuration in the report.

### M6 — loop closure (stretch goal; not required for core completion)

- Only after M5, consider place recognition, geometric verification, and graph correction. Measure drift before/after on a looped sequence and inspect false-positive handling. Core completion does not depend on this milestone.

### Optional final visualization/export (outside core SLAM)

- Export sparse 3D landmarks to PLY and estimated camera poses/trajectory to documented JSON with coordinate frame, transform direction, units, and unknown monocular scale stated explicitly.
- An optional Blender Python script may import these exports to show camera frustums, sparse points, trajectory lines, and rendered views. Keep Blender outside runtime dependencies and SLAM mathematics. Default to CPU rendering; consider AMD HIP only after identifying the precise GPU and confirming support.

## Working rule

Only implement the currently authorized milestone. Update `STATUS.md` after each session with completed work, commands, test results, issues, and the next prompt. Do not advance past M1 without an explicit request.
