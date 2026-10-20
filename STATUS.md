# Status

## Current milestone

M2 — calibrated two-view initialization: **corrected and synthetic-validated; real KITTI validation pending**. The correction work is preserved in a local checkpoint. Do not begin M3.

## M2 correctness audit and checkpoint

- Baseline before edits: clean `main` at `7154988320bf3dd23355908aea5eb367f5e818ec`; 27/27 tests passed. Reproduced the independent-review seed-7, 0.15-baseline false acceptance on OpenCV 4.8.1: 405 essential inliers, 284 pose inliers, 284 provisional points, 173.466° translation-direction error, 1.373° estimated parallax, and about 0.330 px reprojection error. A separate pure-rotation seed 23 was also falsely accepted.
- Rejected initializations now have no public `R_21`, `t_21`, points, or accepted-point mask. Provisional counts and an explicitly invalid debug pose remain for diagnosis. PLY, poses JSON, and PNGs identify rejection and contain no usable camera-2/map geometry. A rejected rerun replaces stale demo-owned successful artifacts without touching unrelated files. Added a specific `insufficient_triangulated_points` reason for a high required-count case.
- Two conservative quality gates were added: pose/essential cheirality fraction ≥0.9 and best-fit rotation-only median residual ≥1.5 px. These use 2D observations and calibration only. The reviewed reversed-translation case now rejects as `low_cheirality_fraction`; pure rotation seed 23 rejects as `rotation_only_explains_matches`.
- Fixed 40-run synthetic study (five seeds × eight conditions), same config throughout: clean, nominal, 1 px noise, 2 px noise, and 30% wrong-match groups each remain 5/5 accepted. Weak baseline changes 4/5 → 2/5 accepted, very weak baseline 1/5 → 0/5, pure rotation 1/5 → 0/5. Severe accepted direction errors (>45°) change 1 → 0. Complete pre/post CSV, JSON, and summaries are ignored under `artifacts/m2-audit-before/` and `artifacts/m2-audit-after/`.
- Final tests: **34/34 passed**, retaining all prior 27. The added test exercises every triangulation filter stage with counts 8→7→5→4→3→1, source indexing, rejected provisional geometry, stale artifact replacement, reviewed low baseline, pure rotation, offline scale evaluation, and nominal-seed acceptance. The original M1 demo still gives 1500/1500 features and 966 retained matches. Geometry-only M2 still accepts 427 points; rendered-image ORB M2 still accepts 260. The latter uses actual ORB matches, not truth correspondence.
- Offline seed-7 scale-only fit is 0.58469 scene units per estimated unit (not Sim(3)); median/p90 3D errors are 0.56524/0.81074 scene units. True-pose triangulation of the same noisy observations gives 0.03271/0.11305. These truth-based references never enter initialization. See `NOTES.md` for depth/error strata and interpretation.
- A scoped check of project `data/sequences/00` and common Downloads/Documents/D: KITTI locations found no real odometry images/calibration. Actual KITTI validation is pending; no dataset was downloaded. The existing synthetic KITTI-layout and missing-data CLI tests pass.
- The approved M2 corrections are included in one local checkpoint. Generated artifacts remain ignored; no remote was created or pushed.

Audit reproduction from the project root:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m unittest discover -s tests -v
python -m orb_slam_mono.demo --output-dir artifacts/demo
python -m orb_slam_mono.geometry_demo --mode synthetic --output-dir artifacts/geometry-synthetic
python -m orb_slam_mono.geometry_demo --mode synthetic-images --output-dir artifacts/geometry-synthetic-images
python -m orb_slam_mono.robustness_study --label after --output-dir artifacts/m2-audit-after
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
- Pure rotation, 0.005 scene-unit baseline, and 85% incorrect correspondences reject. Real KITTI sequence data was not available; no KITTI result is claimed.

## Reproduce from project root in PowerShell

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m unittest discover -s tests -v
python -m orb_slam_mono.demo --output-dir artifacts/demo
python -m orb_slam_mono.geometry_demo --mode synthetic --output-dir artifacts/geometry-synthetic
python -m orb_slam_mono.geometry_demo --mode synthetic-images --output-dir artifacts/geometry-synthetic-images
```

M2 folders contain `stats.json`, essential and accepted correspondence PNGs, an oblique sparse-map PNG, `landmarks.ply`, and `poses.json`. Generated artifacts are ignored by Git.

For real KITTI odometry, provide `data/sequences/00/image_0/*.png` and `data/sequences/00/calib.txt` (optional `times.txt`) from the official dataset, then run a **fixed** set of short-slice pairs, recording successes and rejections:

```powershell
python -m orb_slam_mono.geometry_demo --mode kitti --kitti-sequence data/sequences/00 --first-index 0 --second-index 5 --output-dir artifacts/geometry-kitti-00-000000-000005
```

## Environment and limitations

- Local Python 3.9.0 has OpenCV 4.8.1, NumPy 1.26.1, SciPy 1.11.3. `pytest` is absent, so tests use standard-library `unittest`. M2 uses CPU OpenCV/NumPy; no GPU package was installed.
- The user specifies Ryzen 7000-series, Radeon integrated graphics, and RTX 3050 Ti Laptop GPU. This execution environment reports Ryzen 7 4800H instead; confirm the intended host before hardware-specific choices. `nvidia-smi` reports RTX 3050 Ti, 4096 MiB VRAM, driver 566.07, CUDA compatibility 12.7; that does not establish an installed CUDA Toolkit.
- Blender 3.5.1 is installed at `C:\Program Files\Blender Foundation\Blender 3.5\blender.exe`. A later optional Blender importer may use OptiX after verification, with CPU fallback. The current PLY/JSON export is independent of Blender.
- KITTI/TUM data was not found in the scoped local search; actual KITTI initialization quality, fixed-pair acceptance rates, and rectified-camera assumptions remain unverified on real data. The KITTI CLI missing-data path is tested.
- Monocular scale is arbitrary and may drift in later sequence tracking. Unit-baseline M2 points are not metric; no ground-truth pose/depth enters estimation. Pose errors cause 3D depth error even when reprojection errors are small. Homography dominance is deliberately conservative but may reject a legitimate planar, very low-parallax pair.
- No persistent map, PnP tracking, keyframes, bundle adjustment, trajectory estimation, or loop closure exists.

## Next recommended task

Add a small real KITTI sequence 00 development slice for fixed-pair validation before treating M2 as natural-scene validated. M3 map-based tracking remains the next implementation milestone, but has not started. Any future commit requires separate explicit approval and newly supplied author and committer dates with timezone; do not reuse a previous authorization.
