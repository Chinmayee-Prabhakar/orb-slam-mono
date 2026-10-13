# Design notes

## Session 1 — M1 choices

1. **Independent project.** The supplied GitHub repository is ROS/catkin material, while this project targets inspectable Python/OpenCV. No source was transplanted. The local task directory was empty and not a Git checkout when inspected.
2. **Narrow image abstraction.** `ImageSample`, `ImageSequence`, `KittiGraySequence`, and `SyntheticSequence` are enough for two-image processing. `Frame`, `Keyframe`, `MapPoint`, and `Map` wait until observation identities, poses, and ownership rules are clear.
3. **Camera convention.** `CalibratedCamera` stores a zero-skew pinhole matrix in pixels. KITTI odometry `P0` supplies the left rectified grayscale camera intrinsic block; `image_0` determines image dimensions. This milestone assumes rectified images and does not handle radial distortion or other camera models.
4. **No pose leakage.** The loader reads images, `P0`, and optional timestamps only. It has no ground-truth pose path or scale input.
5. **ORB and matching.** OpenCV ORB provides keypoints and 256-bit descriptors. Brute-force Hamming KNN provides two candidates; Lowe-style 0.75 ratio filtering removes ambiguous neighbors. A reverse nearest-neighbor consistency check removes duplicate/nonmutual assignments. These are descriptor filters, not geometric verification. Alternatives considered: pure cross-check (simpler but no ambiguity ratio) and optical flow (useful later, but not this ORB descriptor milestone).
6. **Diagnostics as a stable boundary.** Stage counts are written to JSON. Unknown future fields are `null`; tracking is `not_estimated`. This prevents the current match count from being mistaken for essential-matrix/PnP inliers.
7. **Synthetic fallback.** Fixed-seed grayscale texture plus a fixed image translation gives a repeatable demonstration and tests. It is not a 3D scene, an initialization test, or a scale benchmark.
8. **Configuration and logging.** `configs/default.json` contains ORB, match, and drawing settings. The CLI uses Python logging for progress and JSON for persistent measurements. No GUI is required.

## Decisions to make in later milestones

- Define camera-to-world versus world-to-camera transforms and frame-index/observation conventions before geometry code.
- Decide initialization parallax, cheirality, and reprojection gates using synthetic geometry tests, then a fixed KITTI development slice.
- Fix gauge freedoms explicitly in sliding-window BA. Keep the active window near 5–10 keyframes, use `least_squares` with sparse Jacobian structure, and measure the tradeoff between accuracy and runtime.
- Separate online estimation from offline evaluation modules. An Umeyama Sim(3) alignment may be used for ATE only; scale drift remains a reported limitation.
- Loop closure is optional after the core system and failure analysis work.

## Session 2 — Git checkpoint and optional visualization preparation

- The project had no Git repository, parent Git checkout, branch, commit, or remote before checkpoint preparation. The existing global author identity is Chinmayee Prabhakar <prabhakar.chinmayee@gmail.com>; the user confirmed this email is associated with the connected GitHub account. Global Git settings were not changed.
- M1 was verified again: 10/10 tests passed; the synthetic demo produced 1500/1500 features, 1005 ratio-passed matches, and 966 retained descriptor matches. Generated artifacts remain ignored.
- Hardware query in this environment reports an AMD Ryzen 7 4800H CPU, AMD Radeon(TM) Graphics, and NVIDIA GeForce RTX 3050 Ti Laptop GPU. This conflicts with the supplied Ryzen 7000-series description, so the exact intended deployment hardware should be checked before GPU-specific decisions. Core dependencies stay CPU-compatible; no CUDA, ROCm, DirectML, PyTorch, or neural-network framework is required.
- Blender 3.5.1 is present at `C:\Program Files\Blender Foundation\Blender 3.5\blender.exe`. It is optional for final visualization, not part of the estimator or tests. Future export: landmarks as PLY and poses/trajectory as JSON with explicit coordinate conventions and scale status. A separate Blender Python script could build frustums, a sparse point cloud, trajectory lines, and rendered views. Use CPU rendering by default; assess AMD HIP only after the precise GPU and driver support are established.
