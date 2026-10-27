# Monocular ORB-Based Visual SLAM

This project estimates a camera's movement from a sequence of grayscale images and builds a sparse 3D map. It uses ORB features, two-view geometry, map-based PnP tracking, keyframes, and local bundle adjustment.

It works on short parts of a real KITTI driving sequence. It does not track the whole sequence continuously. When tracking fails, the pose stays missing and the system starts a new, independently scaled map.

## Course inspiration

This project was inspired by the material and public assignments from WPI’s RBE/CS 549: Computer Vision course, taught by Prof. Nitin J. Sanket. I used those ideas to build an ORB-based system that tracks camera motion and builds a sparse 3D map, then tested it on KITTI.

[Public course material](https://rbe549.github.io/).

## Results at a glance

| KITTI 00 window | Accepted poses | Separate maps | Longest continuous span | In-sample position RMSE |
| --- | ---: | ---: | ---: | --- |
| Frames 0–299 | 145/300 | 9 | 34 frames | 0.184586 m on 58 eligible poses |
| Frames 0–149, fresh start | 76/150 | 5 | 34 frames | 0.225032 m on 34 eligible poses |
| Frames 150–299, fresh start | 68/150 | 3 | 33 frames | Unavailable under the full-segment alignment rule |

Each score uses a separate offline Sim(3) fit for each eligible map. These are not accuracy scores for every frame. For the segment at frames 88–121, fitting the first 17 poses and testing on the next 17 gives 0.614886 m RMSE. None of the ten full-run BA updates passed external validation.

Read the [short project report](docs/REPORT.pdf) or its [Markdown version](docs/REPORT.md) for the method, design choices, results, and limitations.

## Setup

The instructions below use **Windows PowerShell**, which was used for the measured runs. The memory profiler calls a Windows API. The ordinary sequence runner does not require that profiler.

Open a terminal in this repository's root folder. Use Python 3.9 or a compatible environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-visualization.txt
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

The reference numerical environment used Python 3.9, OpenCV 4.8.1, NumPy 1.26.1, and SciPy 1.11.3. Matplotlib and Pillow are needed for the evaluation figures. Library builds and hardware can affect numerical reproducibility and timing; do not change acceptance thresholds to force a match.

Run the tests:

```powershell
python -m unittest discover -s tests -v
```

Tests needing local KITTI images or generated media are separate from synthetic tests and may skip until those files are available. The reviewed implementation reported 156 passing tests in its complete local environment; release-helper tests are additional.

## Get the data

Download the **odometry grayscale data** and **odometry ground-truth poses** from the [official KITTI odometry page](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). Registration may be needed. This repository does not include the raw dataset.

The extraction scripts keep only the first 300 left grayscale frames from sequence 00. Replace the example archive paths with your own local downloads:

```powershell
python scripts/extract_kitti_00_subset.py --gray-archive "<path-to-data_odometry_gray.zip>" --output-dir data/sequences/00 --frame-count 300
python scripts/extract_kitti_00_poses.py --archive "<path-to-data_odometry_poses.zip>" --output data/ground_truth/kitti_odometry/00.txt
```

The resulting layout should be:

```text
data/
  sequences/00/
    image_0/000000.png ... 000299.png
    calib.txt
    times.txt
  ground_truth/kitti_odometry/00.txt
```

Keep the original filenames and calibration. `times.txt` must contain the corresponding first 300 source timestamps. These are sensor measurements, not development timestamps. Do not renumber the second-half run. Ground truth is used only after tracking, for evaluation.

## Run the tracker

For a simple full-sequence run without Windows memory profiling:

```powershell
python -m orb_slam_mono.sequence_demo --mode kitti --kitti-sequence data/sequences/00 --start 0 --end 299 --seed 19 --ablation full --config configs/default.json --output-dir artifacts/demo
```

Look in `artifacts/demo/` for `summary.json`, `per_frame.jsonl`, event files, and the `maps/` folder. `per_frame.jsonl` is the authoritative record of accepted and missing online poses.

## Reproduce the final evaluation

Use these six commands for the three windows and their independent repeats. Each command starts a fresh process. This profiling route requires Windows.

```powershell
python -m orb_slam_mono.m5b_runner --sequence data/sequences/00 --config configs/default.json --output artifacts/m5b-final-evaluation/run-a1 --start 0 --end 299 --seed 19
python -m orb_slam_mono.m5b_runner --sequence data/sequences/00 --config configs/default.json --output artifacts/m5b-final-evaluation/run-a2 --start 0 --end 299 --seed 19
python -m orb_slam_mono.m5b_runner --sequence data/sequences/00 --config configs/default.json --output artifacts/m5b-final-evaluation/run-b1 --start 0 --end 149 --seed 19
python -m orb_slam_mono.m5b_runner --sequence data/sequences/00 --config configs/default.json --output artifacts/m5b-final-evaluation/run-b2 --start 0 --end 149 --seed 19
python -m orb_slam_mono.m5b_runner --sequence data/sequences/00 --config configs/default.json --output artifacts/m5b-final-evaluation/run-c1 --start 150 --end 299 --seed 19
python -m orb_slam_mono.m5b_runner --sequence data/sequences/00 --config configs/default.json --output artifacts/m5b-final-evaluation/run-c2 --start 150 --end 299 --seed 19
```

Then calculate the scores and draw the figures:

```powershell
python scripts/evaluate_release.py --root . --output artifacts/m5b-final-evaluation/evaluation
python -m orb_slam_mono.m5b_figures --root . --output artifacts/m5b-final-evaluation/figures
```

`evaluate_release.py` calls the existing scoring functions and checks the three repeats. It does not need private milestone archives. It does not claim to reverify historical baseline comparisons; the original `m5b_evaluation` command retains that stricter historical-audit role.

Open `evaluation/run-a1/metrics.json` for the full-run numbers and `figures/coverage_comparison.png` to see where tracking worked. `resource_profile.json` in each run contains timing and whole-process memory measurements.

**Keep these folder names for this workflow.** The figure and video tools currently expect them. Run in a fresh checkout or preserve the existing outputs separately before rerunning: the evaluation tools refuse to overwrite existing evidence. Do not delete a previous experiment just to make a command succeed.

## Make the two demonstration videos

After the evaluation above has completed:

```powershell
python -m orb_slam_mono.m5c_data --root . --output artifacts/m5c-visual-demonstration/capture
python -m orb_slam_mono.m5c_render --root . --output artifacts/m5c-visual-demonstration
python -m orb_slam_mono.m5c_offline_gt --root . --output artifacts/m5c-visual-demonstration/figures/segment_004_gt_comparison.png
```

Open these files in a desktop video player:

```text
artifacts/m5c-visual-demonstration/videos/segment_004_tracking_088_122.mp4
artifacts/m5c-visual-demonstration/videos/segment_007_tracking_252_282.mp4
```

The first clip has 34 accepted poses and then loss at frame 122. The second has 30 accepted poses in another map and then loss at 282. They play at 5 fps for inspection, not to claim real-time processing. The point clouds shown throughout are the **final offline maps**, not historical snapshots of map growth.

The video renderer is deliberately tied to the verified sequence and checks its expected source IDs, segments, and map counts. If those checks fail in a different numerical environment, investigate rather than changing or bypassing them. These commands reproduce the project media, not a generic visualizer for arbitrary runs.

## Files worth reading

| File | What it does |
| --- | --- |
| `features.py`, `matching.py` | Extract ORB features and match binary descriptors |
| `geometry.py` | Initialize from two views and triangulate landmarks |
| `tracking.py`, `map_state.py` | Estimate poses and preserve landmark/observation identities |
| `local_mapping.py`, `bundle_adjustment.py` | Add keyframes/points and validate local optimization |
| `sequence_demo.py` | Run the image sequence and save poses/maps |
| `m5b_runner.py` | Run one independently profiled KITTI window |
| `scripts/evaluate_release.py` | Score the saved runs without private audit archives |
| `m5c_data.py`, `m5c_render.py` | Capture verified feature locations and render the videos |

Python modules are under `src/orb_slam_mono/`. The older diagnostic modules are useful for studying failures, but are not needed for the main reproduction route.

## Limits

The full run misses 155 poses. The nine maps have unrelated origins and scales; they must not be joined into one trajectory. There is no loop closure or validated relocalization, and the tested BA candidates did not improve the published map. Short aligned trajectory errors do not demonstrate long-term scale stability. Timing tails do not support a sustained real-time claim.

## Dataset credit and references

Images, calibration, timestamps, and reference poses come from the **KITTI Vision Benchmark Suite** by Andreas Geiger, Philip Lenz, and Raquel Urtasun. KITTI publishes its data under **CC BY-NC-SA 3.0**. Keep that attribution and the applicable terms with image-derived media. See [ATTRIBUTION.md](ATTRIBUTION.md) and [REFERENCES.md](REFERENCES.md). The presentation PDF and PowerPoint are not part of this repository's public release.

## License

The independently written project source code is licensed under the [MIT License](LICENSE). Third-party materials retain their own licenses. KITTI data and image-derived material are not relicensed as MIT; see [ATTRIBUTION.md](ATTRIBUTION.md) for the dataset terms.
