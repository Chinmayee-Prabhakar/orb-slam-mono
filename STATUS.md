# Status


## Current milestone

M2 — calibrated two-view initialization: **implemented and synthetic-validated; real KITTI validation pending**. Do not begin M3 without a new request.

## Milestone history and Git

- M0/M1 established the independent Python package, KITTI grayscale/P0 loader, deterministic image-pair fallback, ORB, matching, visualization, diagnostics, and 10 passing tests. The M1 shifted-image fallback is not physical 3D motion.
- M1 is preserved in the earlier local `main` checkpoint. No remote was configured or pushed.
- M2 adds physical synthetic 3D observations, essential RANSAC, relative pose, triangulation, conservative rejection, an ORB-to-geometry CLI, diagnostics, and PLY/JSON/PNG output. The incremental M2 commit can be identified with `git log -1 --format=fuller`; the source file cannot contain its own commit hash. No remote/push was requested.

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

"Implement M3 only: introduce minimal Frame/Keyframe/MapPoint/Map observation identities, track initialized map points through a fixed short KITTI slice with calibrated PnP RANSAC, record per-frame tracking/failure states and inlier/reprojection diagnostics, and test observation consistency and lost-state transitions. Keep M2 initialization, ground-truth isolation, and arbitrary scale intact; do not add bundle adjustment or loop closure."
