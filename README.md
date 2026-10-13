# orb-slam-mono

An independent, inspectable Python/OpenCV monocular visual SLAM project. The current implementation is **milestone 1 only**: calibrated grayscale image loading, ORB feature extraction, descriptor matching, visualizations, and diagnostics. It does not estimate motion or construct a map.

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
| `configs/default.json` | Tunable feature and match settings |
| `tests/` | Standard-library unit tests |

## Reproduce on Windows PowerShell

From this directory, with Python 3.9 and NumPy/OpenCV available:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m orb_slam_mono.demo --output-dir artifacts/demo
python -m unittest discover -s tests -v
```

For a clean Python environment, first create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Use `.\.venv\Scripts\python` in place of `python` after installation. A Conda specification is also provided in `environment.yml`. The demo writes `matches.png`, two feature images, and `stats.json` (including camera intrinsics). `artifacts/` is ignored by Git.

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
```

The loader reads `P0` and `image_0` only, plus optional timestamps. It never reads ground-truth poses. KITTI odometry images are already rectified; distortion is not modeled in this milestone. For a new camera or unrectified images, calibration and distortion handling must be added before geometry.

## Interpreting the result

`retained_matches` counts descriptor correspondences after a 0.75 nearest-neighbor ratio test and a reverse nearest-neighbor consistency check. It is **not** a geometric inlier count. The `geometric_inliers`, `pnp_inliers`, and `mean_reprojection_error_px` fields are `null`, and `tracking_state` is `not_estimated`, until their later pipeline stages exist. Synthetic images are a repeatable software fallback, not a realistic SLAM benchmark.

See `PROJECT_PLAN.md`, `NOTES.md`, `REFERENCES.md`, and `STATUS.md` for scope, decisions, sources, and the next task.
