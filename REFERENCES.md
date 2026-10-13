# References and implementation boundary

## Conceptual references

- [Mur-Artal, Montiel, and Tardós, ORB-SLAM (2015)](https://arxiv.org/abs/1502.00956): overall monocular tracking, mapping, initialization, and failure considerations.
- [Mur-Artal and Tardós, ORB-SLAM2 (2016)](https://arxiv.org/abs/1610.06475): modular tracking/local mapping concepts and evaluation context.
- [Campos et al., ORB-SLAM3 (2020)](https://arxiv.org/abs/2007.11898): broader architectural context, not a feature list to reproduce.
- [Rublee et al., ORB (ICCV 2011)](https://ieeexplore.ieee.org/document/6126544/): ORB detector/descriptor design; this project calls OpenCV's implementation.
- [KITTI odometry benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php): official image, calibration, and evaluation data format. Development begins with short training-set subsequences.
- [OpenCV BFMatcher documentation](https://docs.opencv.org/4.x/d3/da1/classcv_1_1BFMatcher.html): Hamming distance for binary descriptors and KNN matching API.

## What is original here

The package layout, calibration/data interface, diagnostic schema, descriptor filtering flow, synthetic fallback, CLI, tests, and later geometry/state-management decisions are written for this project. OpenCV supplies ORB and brute-force Hamming primitives. The cited SLAM papers guide concepts only; no published SLAM implementation is copied. When later modules are added, each coordinate convention, scale assumption, rejection gate, and optimizer variable must be documented in `NOTES.md` and tested in isolation.
