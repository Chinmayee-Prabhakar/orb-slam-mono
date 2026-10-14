# orb-slam-mono

An independent, inspectable Python/OpenCV monocular visual SLAM project. M2 estimates a calibrated **two-view relative pose and sparse 3D initialization**. It does not track a sequence or maintain a persistent map. Recovered translation and 3D coordinates have arbitrary monocular scale.

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
| `configs/default.json` | Tunable ORB, matching, geometry, and visualization settings |
| `tests/` | Standard-library unit tests |

## Reproduce on Windows PowerShell

From this directory, with Python 3.9 and NumPy/OpenCV available:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m orb_slam_mono.demo --output-dir artifacts/demo
python -m orb_slam_mono.geometry_demo --mode synthetic --output-dir artifacts/geometry-synthetic
python -m orb_slam_mono.geometry_demo --mode synthetic-images --output-dir artifacts/geometry-synthetic-images
python -m unittest discover -s tests -v
```

For a clean Python environment, first create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Use `.\.venv\Scripts\python` in place of `python` after installation. A Conda specification is also provided in `environment.yml`. The original M1 demo still writes `matches.png`, two feature images, and `stats.json`. M2 writes `essential_correspondences.png` (green RANSAC inliers, red others), `pose_correspondences.png` (green pose inliers), `triangulated_correspondences.png` (green accepted points), `sparse_map.png` (oblique view), `landmarks.ply`, `poses.json`, and `stats.json`. `artifacts/` is ignored by Git.

`--mode synthetic` gives the geometry estimator noisy projected 2D correspondences and calibration, then evaluates against held-out truth. `--mode synthetic-images` renders the 3D scene as grayscale images and uses actual ORB extraction and descriptor matching before geometry. The old M1 shifted-image fallback remains a matching demo; it is not used as a physical motion benchmark.

## KITTI data

Download KITTI **odometry grayscale images** and **calibration files** from the [official KITTI odometry page](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). Place them in the expected odometry layout, for example:

```text
data/sequences/00/image_0/000000.png
data/sequences/00/image_0/000001.png
data/sequences/00/calib.txt
data/sequences/00/times.txt  # optional
```

Then run a consecutive pair:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m orb_slam_mono.demo --kitti-sequence data/sequences/00 --first-index 0 --output-dir artifacts/kitti-00-pair
python -m orb_slam_mono.geometry_demo --mode kitti --kitti-sequence data/sequences/00 --first-index 0 --second-index 5 --output-dir artifacts/geometry-kitti-00-000000-000005
```

The loader reads `P0` and `image_0` only, plus optional timestamps. It never reads ground-truth poses. No real KITTI sequence was available for M2 validation; the command and missing-data error are tested, but real-data results remain pending. Start with a fixed short sequence 00 slice and record all attempted pairs, including rejections. KITTI odometry images are rectified; for another camera or unrectified images, add distortion handling before geometry.

## Interpreting the result

The original M1 `stats.json` fields and their meanings are unchanged. In M2, `retained_matches` remains a descriptor-filter count; `essential_inliers` counts RANSAC inliers; `pose_inliers` counts those retained by cheirality-based pose recovery; `accepted_points` counts triangulations surviving numerical, positive-depth, parallax, and reprojection gates. A rejected initialization has `initialization_status: rejected` and a `failure_reason`; stages not reached use `null` counts. `synthetic` mode has `null` ORB fields because it tests geometry on point correspondences directly.

`T_cw` maps world to camera: `X_c = R_cw X_w + t_cw`. Camera 1 defines the world origin. `R_21` and `t_21_unit` are camera 2 relative to camera 1; `C2 = -R_21.T @ t_21_unit`. `landmarks.ply` uses camera-1 world coordinates, and `poses.json` records both world-to-camera matrices. One translation baseline is defined as one arbitrary unit; PLY/JSON output is **not metric**. Synthetic ground truth is used only after estimation to score rotation, direction, and an explicitly offline aligned 3D error.

See `PROJECT_PLAN.md`, `NOTES.md`, `REFERENCES.md`, and `STATUS.md` for scope, decisions, sources, and the next task.
